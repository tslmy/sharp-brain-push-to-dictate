/*
 * moonstt: Moonshine-tiny speech recognition for ARMv5TE without FPU/NEON.
 *
 * Weights are int8 (per output row), activations are int16 (12-bit, dynamic
 * scale), and the matrix kernels use SMLABB/SMLATT on the offset-binary
 * packed weights written by tools/convert.py.  Attention scores and P*V are
 * integer too.  Remaining elementwise work (layer norm, GELU, rotary, ...) is
 * float and is the next thing to move to fixed point.
 */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "moonstt.h"

#ifndef TILE_ROWS
#define TILE_ROWS 16
#endif
#define D 288
#define FF 1152
#define NH 8
#define HD 36
#define ROT 32
#define NL 6
#define VOCAB 32000
#define TOK_BOS 1
#define TOK_EOS 2
#define MAXDEC 194

/* ------------------------------------------------------------------ */
/* time / profiling                                                    */

static double now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

enum { P_CONV, P_C1, P_C1T, P_GN, P_C2Q, P_C2, P_C2G, P_C3Q, P_C3, P_ENC_LIN, P_ENC_ATT, P_ENC_ELT, P_XKV, P_DEC_LIN, P_DEC_ATT, P_DEC_ELT, P_HEAD, P_N };
static const char *pname[P_N] = { "conv misc", "conv1 matmul", "conv1 tanh", "groupnorm", "conv2 quant", "conv2 matmul", "conv2 gelu", "conv3 quant", "conv3 matmul", "enc linear",   "enc attention", "enc elementwise", "cross K/V",
				  "dec linear",	    "dec attention", "dec elementwise", "lm head" };
static double prof[P_N];
#define TIC(v) double v = now()
#define TOC(v, id) (prof[id] += now() - (v))

/* ------------------------------------------------------------------ */
/* integer kernels                                                     */

#if defined(__arm__) && defined(__ARM_FEATURE_DSP)
#define SMLA2(acc, x, y)                                                   \
	__asm__("smlabb %0,%1,%2,%0\n\tsmlatt %0,%1,%2,%0" : "+r"(acc) : "r"(x), "r"(y))
#define HAVE_SMLA 1
#ifdef USE_PLD /* measured slower on ARM926 */
#define PLD(p) __asm__ volatile("pld [%0]" : : "r"(p))
#else
#define PLD(p) ((void)0)
#endif
#else
#define PLD(p) ((void)0)
#endif

/* sum_k (w[k]+128) * a[k] over n4*4 taps; w packed as in convert.py */
static inline int32_t dot_w(const uint32_t *w, const int16_t *a, int n4)
{
	int32_t s = 0;
	int i;
#ifdef HAVE_SMLA
	const int32_t *a32 = (const int32_t *)a;
	for (i = 0; i < n4; i++) {
		uint32_t x = w[i], lo = x & 0x00ff00ffu, hi = (x >> 8) & 0x00ff00ffu;
		int32_t a0 = a32[2 * i], a1 = a32[2 * i + 1];
		SMLA2(s, lo, a0);
		SMLA2(s, hi, a1);
	}
#else
	for (i = 0; i < n4; i++) {
		uint32_t x = w[i];
		s += (int32_t)(x & 0xff) * a[4 * i] + (int32_t)((x >> 16) & 0xff) * a[4 * i + 1] +
		     (int32_t)((x >> 8) & 0xff) * a[4 * i + 2] + (int32_t)(x >> 24) * a[4 * i + 3];
	}
#endif
	return s;
}

/* same weights against two activation rows (weight loads shared) */
static inline void dot_w2(const uint32_t *w, const int16_t *a, const int16_t *b, int n4, int32_t *ra, int32_t *rb)
{
	int32_t s = 0, t = 0;
	int i;
#ifdef HAVE_SMLA
	const int32_t *a32 = (const int32_t *)a, *b32 = (const int32_t *)b;
	for (i = 0; i < n4; i++) {
		uint32_t x = w[i], lo = x & 0x00ff00ffu, hi = (x >> 8) & 0x00ff00ffu;
		int32_t a0 = a32[2 * i], a1 = a32[2 * i + 1], b0 = b32[2 * i], b1 = b32[2 * i + 1];
		SMLA2(s, lo, a0);
		SMLA2(s, hi, a1);
		SMLA2(t, lo, b0);
		SMLA2(t, hi, b1);
	}
#else
	s = dot_w(w, a, n4);
	t = dot_w(w, b, n4);
	(void)i;
#endif
	*ra = s;
	*rb = t;
}

/* two weight rows x two activation rows: r = {w0.a, w1.a, w0.b, w1.b} */
static inline void dot_w22(const uint32_t *w0, const uint32_t *w1, const int16_t *a, const int16_t *b, int n4,
			   int32_t *r)
{
#ifdef HAVE_SMLA
	int32_t s0 = 0, s1 = 0, t0 = 0, t1 = 0;
	const int32_t *a32 = (const int32_t *)a, *b32 = (const int32_t *)b;
	int i;
	for (i = 0; i < n4; i++) {
		uint32_t x = w0[i], y = w1[i];
		uint32_t lo = x & 0x00ff00ffu, hi = (x >> 8) & 0x00ff00ffu;
		uint32_t lo2 = y & 0x00ff00ffu, hi2 = (y >> 8) & 0x00ff00ffu;
		int32_t a0 = a32[2 * i], a1 = a32[2 * i + 1], b0 = b32[2 * i], b1 = b32[2 * i + 1];
		if (!(i & 7)) {
			PLD(w0 + i + 16);
			PLD(w1 + i + 16);
		}
		SMLA2(s0, lo, a0);
		SMLA2(s0, hi, a1);
		SMLA2(s1, lo2, a0);
		SMLA2(s1, hi2, a1);
		SMLA2(t0, lo, b0);
		SMLA2(t0, hi, b1);
		SMLA2(t1, lo2, b0);
		SMLA2(t1, hi2, b1);
	}
	r[0] = s0;
	r[1] = s1;
	r[2] = t0;
	r[3] = t1;
#else
	r[0] = dot_w(w0, a, n4);
	r[1] = dot_w(w1, a, n4);
	r[2] = dot_w(w0, b, n4);
	r[3] = dot_w(w1, b, n4);
#endif
}

/* two weight rows x one activation row */
static inline void dot_w12(const uint32_t *w0, const uint32_t *w1, const int16_t *a, int n4, int32_t *r)
{
#ifdef HAVE_SMLA
	int32_t s0 = 0, s1 = 0;
	const int32_t *a32 = (const int32_t *)a;
	int i;
	for (i = 0; i < n4; i++) {
		uint32_t x = w0[i], y = w1[i];
		uint32_t lo = x & 0x00ff00ffu, hi = (x >> 8) & 0x00ff00ffu;
		uint32_t lo2 = y & 0x00ff00ffu, hi2 = (y >> 8) & 0x00ff00ffu;
		int32_t a0 = a32[2 * i], a1 = a32[2 * i + 1];
		if (!(i & 7)) {
			PLD(w0 + i + 16);
			PLD(w1 + i + 16);
		}
		SMLA2(s0, lo, a0);
		SMLA2(s0, hi, a1);
		SMLA2(s1, lo2, a0);
		SMLA2(s1, hi2, a1);
	}
	r[0] = s0;
	r[1] = s1;
#else
	r[0] = dot_w(w0, a, n4);
	r[1] = dot_w(w1, a, n4);
#endif
}

/* signed int16 . int16, n multiple of 2 */
static inline int32_t dot16(const int16_t *a, const int16_t *b, int n)
{
	int32_t s = 0;
	int i;
#ifdef HAVE_SMLA
	int32_t u = 0, nw = n / 2;
	const int32_t *a32 = (const int32_t *)a, *b32 = (const int32_t *)b;
	for (i = 0; i + 1 < nw; i += 2) {
		int32_t x0 = a32[i], y0 = b32[i], x1 = a32[i + 1], y1 = b32[i + 1];
		SMLA2(s, x0, y0);
		SMLA2(u, x1, y1);
	}
	if (i < nw) {
		int32_t x0 = a32[i], y0 = b32[i];
		SMLA2(s, x0, y0);
	}
	s += u;
#else
	for (i = 0; i < n; i++)
		s += (int32_t)a[i] * b[i];
#endif
	return s;
}

/* ------------------------------------------------------------------ */
/* model                                                               */

typedef struct {
	int o, i;
	const float *sw, *bias;
	const uint32_t *w;
} lin_t;

typedef struct {
	const float *ln1;
	lin_t q, k, v, o;
	const float *ln2;
	lin_t cq, ck, cv, co; /* decoder only */
	const float *ln3;     /* decoder only */
	lin_t fc1, fc2;
} layer_t;

struct moonstt {
	void *map;
	size_t maplen;
	lin_t c1, c2, c3;
	const float *gn_w, *gn_b;
	layer_t enc[NL], dec[NL];
	const float *enc_ln, *dec_ln;
	lin_t emb;
	char *tokblob;
	const uint32_t *tokoff;
	void *tokmap;
	size_t toklen;
	int *lm_ids; /* output vocabulary: ASCII-only tokens + EOS */
	int n_lm;
	/* tables */
	int16_t exp_lut[4096];
	float *rope_c, *rope_s;
	int rope_n;
	float inv_freq[ROT / 2];
};

static const unsigned char *rd_p;

static const void *take(size_t n)
{
	const void *p;
	rd_p = (const unsigned char *)(((uintptr_t)rd_p + 31) & ~(uintptr_t)31);
	p = rd_p;
	rd_p += n;
	return p;
}

static lin_t take_lin(int o, int i)
{
	lin_t l;
	l.o = o;
	l.i = i;
	l.sw = take(o * 4);
	l.bias = take(o * 4);
	l.w = take((size_t)o * i);
	return l;
}

static int map_file(const char *path, void **p, size_t *len)
{
	struct stat st;
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;
	if (fstat(fd, &st) < 0) {
		close(fd);
		return -1;
	}
	*p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (*p == MAP_FAILED)
		return -1;
	*len = st.st_size;
	return 0;
}

static void rope_prepare(moonstt *m, int n)
{
	int p, i;
	if (n <= m->rope_n)
		return;
	free(m->rope_c);
	free(m->rope_s);
	m->rope_c = malloc((size_t)n * (ROT / 2) * 4);
	m->rope_s = malloc((size_t)n * (ROT / 2) * 4);
	for (p = 0; p < n; p++)
		for (i = 0; i < ROT / 2; i++) {
			double a = p * (double)m->inv_freq[i];
			m->rope_c[p * (ROT / 2) + i] = (float)cos(a);
			m->rope_s[p * (ROT / 2) + i] = (float)sin(a);
		}
	m->rope_n = n;
}

static void luts_init(void);

moonstt *moonstt_load(const char *dir)
{
	char path[512];
	moonstt *m = calloc(1, sizeof(*m));
	int l, i;
	const int *hdr;

	snprintf(path, sizeof path, "%s/model.bin", dir);
	if (map_file(path, &m->map, &m->maplen) < 0) {
		fprintf(stderr, "moonstt: cannot map %s: %s\n", path, strerror(errno));
		free(m);
		return NULL;
	}
	if (memcmp(m->map, "MSTT0001", 8)) {
		fprintf(stderr, "moonstt: bad model magic\n");
		return NULL;
	}
	hdr = (const int *)((const char *)m->map + 8);
	if (hdr[0] != D || hdr[1] != FF || hdr[2] != NH || hdr[3] != NL || hdr[5] != VOCAB) {
		fprintf(stderr, "moonstt: unexpected model shape\n");
		return NULL;
	}
	rd_p = (const unsigned char *)m->map + 8 + 32;
	m->c1 = take_lin(D, 128);
	m->c2 = take_lin(2 * D, 7 * D);
	m->c3 = take_lin(D, 3 * 2 * D);
	m->gn_w = take(D * 4);
	m->gn_b = take(D * 4);
	for (l = 0; l < NL; l++) {
		layer_t *y = &m->enc[l];
		y->ln1 = take(D * 4);
		y->q = take_lin(D, D);
		y->k = take_lin(D, D);
		y->v = take_lin(D, D);
		y->o = take_lin(D, D);
		y->ln2 = take(D * 4);
		y->fc1 = take_lin(FF, D);
		y->fc2 = take_lin(D, FF);
	}
	m->enc_ln = take(D * 4);
	for (l = 0; l < NL; l++) {
		layer_t *y = &m->dec[l];
		y->ln1 = take(D * 4);
		y->q = take_lin(D, D);
		y->k = take_lin(D, D);
		y->v = take_lin(D, D);
		y->o = take_lin(D, D);
		y->ln2 = take(D * 4);
		y->cq = take_lin(D, D);
		y->ck = take_lin(D, D);
		y->cv = take_lin(D, D);
		y->co = take_lin(D, D);
		y->ln3 = take(D * 4);
		y->fc1 = take_lin(2 * FF, D);
		y->fc2 = take_lin(D, FF);
	}
	m->dec_ln = take(D * 4);
	m->emb = take_lin(VOCAB, D);
	if ((size_t)(rd_p - (const unsigned char *)m->map) > m->maplen) {
		fprintf(stderr, "moonstt: model file truncated\n");
		return NULL;
	}

	snprintf(path, sizeof path, "%s/tok.bin", dir);
	if (map_file(path, &m->tokmap, &m->toklen) < 0) {
		fprintf(stderr, "moonstt: cannot map %s\n", path);
		return NULL;
	}
	m->tokoff = (const uint32_t *)((const char *)m->tokmap + 12);
	m->tokblob = (char *)m->tokmap + 12 + (VOCAB + 1) * 4;

	m->lm_ids = malloc(VOCAB * sizeof(int));
	for (i = 2; i < VOCAB; i++) {
		uint32_t a = m->tokoff[i], b = m->tokoff[i + 1], k;
		int ascii = 1;
		for (k = a; k < b; k++)
			if ((unsigned char)m->tokblob[k] >= 0x80)
				ascii = 0;
		if (ascii)
			m->lm_ids[m->n_lm++] = i;
	}
	if (m->n_lm & 1) /* rows are scored in pairs */
		m->lm_ids[m->n_lm++] = m->lm_ids[0];
	for (i = 0; i < 4096; i++)
		m->exp_lut[i] = (int16_t)lrint(2048.0 * exp(-i / 256.0));
	for (i = 0; i < ROT / 2; i++)
		m->inv_freq[i] = (float)(1.0 / pow(10000.0, (2.0 * i) / ROT));
	rope_prepare(m, MAXDEC);
	luts_init();
	return m;
}

void moonstt_free(moonstt *m)
{
	if (!m)
		return;
	munmap(m->map, m->maplen);
	munmap(m->tokmap, m->toklen);
	free(m->rope_c);
	free(m->rope_s);
	free(m->lm_ids);
	free(m);
}

/* ------------------------------------------------------------------ */
/* float helpers                                                       */

static void layernorm(const float *x, const float *w, float *y, int n)
{
	float mean = 0, var = 0, r;
	int i;
	for (i = 0; i < n; i++)
		mean += x[i];
	mean /= n;
	for (i = 0; i < n; i++) {
		float d = x[i] - mean;
		var += d * d;
	}
	r = 1.0f / sqrtf(var / n + 1e-5f);
	for (i = 0; i < n; i++)
		y[i] = (x[i] - mean) * r * w[i];
}

/* Activation quantisation without soft-float: the scale is a power of two
 * (so the max lands in [1024, 2047]) and conversion is pure integer work on
 * the IEEE bit pattern. */
static inline uint32_t fbits(float x)
{
	uint32_t u;
	memcpy(&u, &x, 4);
	return u;
}

/* scale exponent k so that |x|*2^k < 2^bits for |x| <= value with bits mxb */
static inline int pow2_k(uint32_t mxb, int bits)
{
	int E = (int)(mxb >> 23) - 127;
	if (E < -100)
		E = -100;
	return bits - 1 - E;
}

static inline float pow2f(int e) /* 2^e as float, e in [-100, 100] */
{
	uint32_t u = (uint32_t)(127 + e) << 23;
	float f;
	memcpy(&f, &u, 4);
	return f;
}

/* round(x * 2^k), clamped to +-lim */
static inline int q_one(float x, int k, int lim)
{
	uint32_t u = fbits(x), mag = u & 0x7fffffffu;
	int e = (int)(mag >> 23), sh = 150 - e - k, v;
	if (sh >= 24)
		return 0;
	if (sh < 1)
		sh = 1;
	v = (int)(((mag & 0x7fffffu) | 0x800000u) + (1u << (sh - 1))) >> sh;
	if (v > lim)
		v = lim;
	return (u >> 31) ? -v : v;
}

/* round(x * 2^k) with no clamp; caller guarantees small magnitude */
static inline int q_one_raw(float x, int k)
{
	uint32_t u = fbits(x), mag = u & 0x7fffffffu;
	int e = (int)(mag >> 23), sh = 150 - e - k, v;
	if (sh >= 24)
		return 0;
	v = (int)(((mag & 0x7fffffu) | 0x800000u) + (1u << (sh - 1))) >> sh;
	return (u >> 31) ? -v : v;
}

static void quant(const float *x, int n, int16_t *q, float *scale, int32_t *sum)
{
	uint32_t mxb = 0;
	int i, k;
	int32_t s = 0;
	for (i = 0; i < n; i++) {
		uint32_t u = fbits(x[i]) & 0x7fffffffu;
		if (u > mxb)
			mxb = u;
	}
	if (mxb < 0x00800000u) { /* all zero / denormal */
		for (i = 0; i < n; i++)
			q[i] = 0;
		*scale = 1e-20f;
		*sum = 0;
		return;
	}
	k = pow2_k(mxb, 12);
	for (i = 0; i < n; i++) {
		int v = q_one(x[i], k, 4095);
		q[i] = (int16_t)v;
		s += v;
	}
	*scale = pow2f(-k);
	*sum = s;
}

static float gelu_lut[2049]; /* gelu(k/128), k = -1024..1024 */
static float tanh_lut[2562]; /* x in [-5, 5], step 1/256 */

static void luts_init(void)
{
	int i;
	for (i = 0; i < 2049; i++) {
		double x = (i - 1024) / 128.0;
		gelu_lut[i] = (float)(0.5 * x * (1.0 + erf(x * 0.70710678118654752)));
	}
	for (i = 0; i < 2562; i++)
		tanh_lut[i] = (float)tanh(-5.0 + i / 256.0);
}

static inline float gelu(float x)
{
	uint32_t u = fbits(x), mag = u & 0x7fffffffu;
	if (mag >= 0x41000000u) /* |x| >= 8 */
		return (u >> 31) ? 0.0f : x;
	return gelu_lut[q_one_raw(x, 7) + 1024];
}

static inline float tanh_fast(float x)
{
	float f;
	int i;
	if (x >= 4.99f)
		return 1.0f;
	if (x <= -4.99f)
		return -1.0f;
	f = (x + 5.0f) * 256.0f;
	i = (int)f;
	return tanh_lut[i] + (f - (float)i) * (tanh_lut[i + 1] - tanh_lut[i]);
}

static inline float silu(float x)
{
	return x / (1.0f + expf(-x));
}

/* (float)x * 2^e without soft-float: normalise with CLZ and build the bits */
static inline float i2f_scaled(int32_t x, int e)
{
	uint32_t sign = (uint32_t)x & 0x80000000u, ax = sign ? (uint32_t)(-x) : (uint32_t)x, bits;
	int lz;
	float f;
	if (!ax)
		return 0.0f;
	lz = __builtin_clz(ax);
	bits = sign | ((uint32_t)(127 + 31 - lz + e) << 23) | (((ax << lz) >> 8) & 0x7fffffu);
	memcpy(&f, &bits, 4);
	return f;
}

static inline int exp_of(float pow2) /* exact exponent of a power-of-two float */
{
	return (int)((fbits(pow2) >> 23) & 0xff) - 127;
}

/* out[t*o+j] = W[j] . qa[t*stride ...]  for t<T */
/* Large-K matrices: a 16-frame activation tile would not fit the 16 KB data
 * cache, so walk K in 288-element chunks over an 8-frame tile and keep the
 * partial sums in an int32 tile accumulator. */
#define KCW 72 /* chunk size in 32-bit weight words (288 taps) */
#ifndef CTB
#define CTB 16
#endif

static void lin_run_chunked(const lin_t *L, const int16_t *qa, int stride, const float *sc, const int32_t *sm, int T,
			    float *out)
{
	const int n4 = L->i / 4, o = L->o, nch = n4 / KCW;
	int32_t acc[CTB * 576];
	int tb, j, t, c;
	for (tb = 0; tb < T; tb += CTB) {
		int te = tb + CTB < T ? tb + CTB : T, nt = te - tb;
		memset(acc, 0, (size_t)nt * o * 4);
		for (c = 0; c < nch; c++) {
			for (j = 0; j < o; j += 2) {
				const uint32_t *w0 = L->w + (size_t)j * n4 + (size_t)c * KCW, *w1 = w0 + n4;
				for (t = tb; t + 1 < te; t += 2) {
					int32_t r[4];
					int32_t *a0 = acc + (size_t)(t - tb) * o + j, *a1 = a0 + o;
					dot_w22(w0, w1, qa + (size_t)t * stride + c * (KCW * 4),
						qa + (size_t)(t + 1) * stride + c * (KCW * 4), KCW, r);
					a0[0] += r[0];
					a0[1] += r[1];
					a1[0] += r[2];
					a1[1] += r[3];
				}
				if (t < te) {
					int32_t r[2], *a0 = acc + (size_t)(t - tb) * o + j;
					dot_w12(w0, w1, qa + (size_t)t * stride + c * (KCW * 4), KCW, r);
					a0[0] += r[0];
					a0[1] += r[1];
				}
			}
		}
		for (t = tb; t < te; t++) {
			int e = exp_of(sc[t]);
			const int32_t *ar = acc + (size_t)(t - tb) * o;
			float *orow = out + (size_t)t * o;
			for (j = 0; j < o; j++)
				orow[j] = i2f_scaled(ar[j] - 128 * sm[t], e) * L->sw[j] + L->bias[j];
		}
	}
}

static void lin_run(const lin_t *L, const int16_t *qa, int stride, const float *sc, const int32_t *sm, int T, float *out)
{
	const int n4 = L->i / 4, o = L->o;
	int tb, j, t, TB = TILE_ROWS;
	if (L->i >= 1152 && L->i % (KCW * 4) == 0 && T >= 8 && o <= 576) {
		lin_run_chunked(L, qa, stride, sc, sm, T, out);
		return;
	}
	for (tb = 0; tb < T; tb += TB) {
		int te = tb + TB < T ? tb + TB : T;
		for (j = 0; j < o; j += 2) {
			const uint32_t *w0 = L->w + (size_t)j * n4, *w1 = w0 + n4;
			float sw0 = L->sw[j], sw1 = L->sw[j + 1], b0 = L->bias[j], b1 = L->bias[j + 1];
			for (t = tb; t + 1 < te; t += 2) {
				int32_t r[4];
				int ea = exp_of(sc[t]), eb = exp_of(sc[t + 1]);
				float *o0 = out + (size_t)t * o + j, *o1 = o0 + o;
				dot_w22(w0, w1, qa + (size_t)t * stride, qa + (size_t)(t + 1) * stride, n4, r);
				o0[0] = i2f_scaled(r[0] - 128 * sm[t], ea) * sw0 + b0;
				o0[1] = i2f_scaled(r[1] - 128 * sm[t], ea) * sw1 + b1;
				o1[0] = i2f_scaled(r[2] - 128 * sm[t + 1], eb) * sw0 + b0;
				o1[1] = i2f_scaled(r[3] - 128 * sm[t + 1], eb) * sw1 + b1;
			}
			if (t < te) {
				int32_t r[2];
				int ea = exp_of(sc[t]);
				float *o0 = out + (size_t)t * o + j;
				dot_w12(w0, w1, qa + (size_t)t * stride, n4, r);
				o0[0] = i2f_scaled(r[0] - 128 * sm[t], ea) * sw0 + b0;
				o0[1] = i2f_scaled(r[1] - 128 * sm[t], ea) * sw1 + b1;
			}
		}
	}
}

static void rotary(const moonstt *m, float *v, int pos)
{
	const float *c = m->rope_c + (size_t)pos * (ROT / 2), *s = m->rope_s + (size_t)pos * (ROT / 2);
	int h, i;
	for (h = 0; h < NH; h++) {
		float *p = v + h * HD;
		for (i = 0; i < ROT / 2; i++) {
			float x0 = p[2 * i], x1 = p[2 * i + 1];
			p[2 * i] = x0 * c[i] - x1 * s[i];
			p[2 * i + 1] = x1 * c[i] + x0 * s[i];
		}
	}
}

/* ------------------------------------------------------------------ */
/* attention on quantised K / V^T                                      */

typedef struct {
	int T, Tp;
	int16_t *K;  /* [NH][T][HD] */
	int16_t *VT; /* [NH][HD][Tp] */
	float ks[NH], vs[NH];
} kv_t;

static void kv_alloc(kv_t *kv, int T)
{
	kv->T = T;
	kv->Tp = (T + 1) & ~1;
	kv->K = malloc((size_t)NH * T * HD * 2);
	kv->VT = calloc((size_t)NH * HD * kv->Tp, 2);
}

static void kv_free(kv_t *kv)
{
	free(kv->K);
	free(kv->VT);
	kv->K = kv->VT = NULL;
}

/* k, v: float [T][D] (head h = columns h*HD..).  One scale per head. */
static void kv_fill(kv_t *kv, const float *k, const float *v)
{
	int T = kv->T, h, t, d;
	for (h = 0; h < NH; h++) {
		uint32_t mk = 0, mv = 0;
		int kk, kvv;
		for (t = 0; t < T; t++)
			for (d = 0; d < HD; d++) {
				uint32_t a = fbits(k[t * D + h * HD + d]) & 0x7fffffffu;
				uint32_t b = fbits(v[t * D + h * HD + d]) & 0x7fffffffu;
				if (a > mk)
					mk = a;
				if (b > mv)
					mv = b;
			}
		kk = pow2_k(mk < 0x00800000u ? 0x00800000u : mk, 11);
		kvv = pow2_k(mv < 0x00800000u ? 0x00800000u : mv, 10);
		kv->ks[h] = pow2f(-kk);
		kv->vs[h] = pow2f(-kvv);
		for (t = 0; t < T; t++)
			for (d = 0; d < HD; d++) {
				kv->K[((size_t)h * T + t) * HD + d] = (int16_t)q_one(k[t * D + h * HD + d], kk, 2047);
				kv->VT[((size_t)h * HD + d) * kv->Tp + t] = (int16_t)q_one(v[t * D + h * HD + d], kvv, 1023);
			}
	}
}

/* Queries are processed QB at a time so each K row and V^T row is streamed
 * from DRAM once per block instead of once per query. */
#ifndef QB
#define QB 8
#endif
#define ATT_SCRATCH(T) ((size_t)QB * (((T) + 2) * 2 + (T) * 4) + 64)

/* B queries (B*D floats) against kv; writes B*D floats.  scr: ATT_SCRATCH(T) bytes */
static void attend_blk(const moonstt *m, const float *q, int B, const kv_t *kv, float *out, void *scr)
{
	const int T = kv->T, Tp = kv->Tp;
	int16_t *e = (int16_t *)scr;
	int32_t *acc = (int32_t *)(e + (size_t)QB * Tp);
	int h, j, d, b;
	for (h = 0; h < NH; h++) {
		int16_t qq[QB][HD];
		int32_t mx[QB], dummy;
		float qs[QB], inv[QB];
		for (b = 0; b < B; b++) {
			quant(q + (size_t)b * D + h * HD, HD, qq[b], &qs[b], &dummy);
			mx[b] = INT32_MIN;
		}
		for (j = 0; j < T; j++) {
			const int16_t *kr = kv->K + ((size_t)h * T + j) * HD;
			for (b = 0; b < B; b++) {
				int32_t a = dot16(qq[b], kr, HD);
				acc[(size_t)b * T + j] = a;
				if (a > mx[b])
					mx[b] = a;
			}
		}
		for (b = 0; b < B; b++) {
			int32_t sume = 0, *ab = acc + (size_t)b * T;
			int16_t *eb = e + (size_t)b * Tp;
			float c = qs[b] * kv->ks[h] * (1.0f / 6.0f) * 256.0f; /* table index per unit of (mx-acc) */
			int ex, sh;
			int64_t cf;
			float mant = frexpf(c, &ex); /* c = mant * 2^ex, mant in [0.5,1) */
			cf = (int64_t)(mant * 2097152.0f);
			sh = 21 - ex;
			if (sh > 62)
				sh = 62;
			if (sh < 0)
				sh = 0;
			for (j = 0; j < T; j++) {
				int64_t idx = ((int64_t)(mx[b] - ab[j]) * cf) >> sh;
				int ev = idx >= 4096 ? 0 : m->exp_lut[idx];
				eb[j] = (int16_t)ev;
				sume += ev;
			}
			if (Tp != T)
				eb[T] = 0;
			inv[b] = kv->vs[h] / (float)sume;
		}
		for (d = 0; d < HD; d++) {
			const int16_t *vt = kv->VT + ((size_t)h * HD + d) * Tp;
			for (b = 0; b < B; b++)
				out[(size_t)b * D + h * HD + d] = (float)dot16(e + (size_t)b * Tp, vt, Tp) * inv[b];
		}
	}
}

/* ------------------------------------------------------------------ */
/* encoder                                                             */

typedef struct {
	int T;
	float *x; /* [T][D] */
} enc_t;

static int conv_len(int n)
{
	int n1 = (n - 127) / 64 + 1, n2 = (n1 - 7) / 3 + 1;
	return (n2 - 3) / 2 + 1;
}

static void encode(moonstt *m, const int16_t *pcm, int n, enc_t *E)
{
	int n1 = (n - 127) / 64 + 1, n2 = (n1 - 7) / 3 + 1, n3 = (n2 - 3) / 2 + 1;
	int t, j, l, i;
	TIC(t0);

	/* conv1 on raw audio: window u starts at 64u, 128 taps (last weight 0) */
	size_t padn = (size_t)n1 * 64 + 128;
	int16_t *a = calloc(padn, 2);
	memcpy(a, pcm, (size_t)n * 2);
	float *x1 = malloc((size_t)n1 * D * 4);
	{
		int32_t *pre = malloc((padn + 1) * 4), *wsm = malloc(n1 * 4);
		float *wsc = malloc(n1 * 4);
		size_t k, N = (size_t)n1 * D;
		pre[0] = 0;
		for (k = 0; k < padn; k++)
			pre[k + 1] = pre[k] + a[k];
		for (t = 0; t < n1; t++) {
			wsm[t] = pre[t * 64 + 128] - pre[t * 64];
			wsc[t] = 1.0f / 32768.0f;
		}
		{ TIC(c1); lin_run(&m->c1, a, 64, wsc, wsm, n1, x1); TOC(c1, P_C1); }
		{
			TIC(c1t);
			for (k = 0; k < N; k++)
				x1[k] = tanh_fast(x1[k]);
			TOC(c1t, P_C1T);
		}
		free(pre);
		free(wsm);
		free(wsc);
	}
	free(a);
	/* group norm over everything (1 group): float partial sums, double totals */
	{
		TIC(gn);
		double s = 0, s2 = 0;
		float mean, r;
		size_t N = (size_t)n1 * D;
		for (t = 0; t < n1; t++) {
			float p = 0;
			for (j = 0; j < D; j++)
				p += x1[(size_t)t * D + j];
			s += p;
		}
		mean = (float)(s / N);
		for (t = 0; t < n1; t++) {
			float p = 0;
			for (j = 0; j < D; j++) {
				float d = x1[(size_t)t * D + j] - mean;
				p += d * d;
			}
			s2 += p;
		}
		r = 1.0f / sqrtf((float)(s2 / N) + 1e-5f);
		for (t = 0; t < n1; t++)
			for (j = 0; j < D; j++)
				x1[(size_t)t * D + j] = (x1[(size_t)t * D + j] - mean) * r * m->gn_w[j] + m->gn_b[j];
		TOC(gn, P_GN);
	}
	/* conv2: one global scale, rows are 7 consecutive frames */
	{
		size_t N = (size_t)n1 * D, k;
		int16_t *q = malloc(N * 2);
		int32_t *fs = calloc(n1, 4), *sm = malloc(n2 * 4);
		float sc, *scs = malloc(n2 * 4), *x2 = malloc((size_t)n2 * 2 * D * 4);
		int32_t dummy;
		uint32_t mxb = 0x00800000u;
		int kq;
		TIC(c2q);
		for (k = 0; k < N; k++) {
			uint32_t u = fbits(x1[k]) & 0x7fffffffu;
			if (u > mxb)
				mxb = u;
		}
		kq = pow2_k(mxb, 11);
		sc = pow2f(-kq);
		for (k = 0; k < N; k++)
			q[k] = (int16_t)q_one(x1[k], kq, 2047);
		for (t = 0; t < n1; t++)
			for (j = 0; j < D; j++)
				fs[t] += q[(size_t)t * D + j];
		for (t = 0; t < n2; t++) {
			sm[t] = 0;
			for (i = 0; i < 7; i++)
				sm[t] += fs[3 * t + i];
			scs[t] = sc;
		}
		TOC(c2q, P_C2Q);
		{ TIC(c2); lin_run(&m->c2, q, 3 * D, scs, sm, n2, x2); TOC(c2, P_C2); }
		{
			TIC(c2g);
			for (k = 0; k < (size_t)n2 * 2 * D; k++)
				x2[k] = gelu(x2[k]);
			TOC(c2g, P_C2G);
		}
		free(q);
		free(fs);
		free(sm);
		free(scs);
		free(x1);
		(void)dummy;
		/* conv3 */
		N = (size_t)n2 * 2 * D;
		q = malloc(N * 2);
		fs = calloc(n2, 4);
		sm = malloc(n3 * 4);
		scs = malloc(n3 * 4);
		E->x = malloc((size_t)n3 * D * 4);
		TIC(c3q);
		mxb = 0x00800000u;
		for (k = 0; k < N; k++) {
			uint32_t u = fbits(x2[k]) & 0x7fffffffu;
			if (u > mxb)
				mxb = u;
		}
		kq = pow2_k(mxb, 11);
		sc = pow2f(-kq);
		for (k = 0; k < N; k++)
			q[k] = (int16_t)q_one(x2[k], kq, 2047);
		for (t = 0; t < n2; t++)
			for (j = 0; j < 2 * D; j++)
				fs[t] += q[(size_t)t * 2 * D + j];
		for (t = 0; t < n3; t++) {
			sm[t] = fs[2 * t] + fs[2 * t + 1] + fs[2 * t + 2];
			scs[t] = sc;
		}
		TOC(c3q, P_C3Q);
		{ TIC(c3); lin_run(&m->c3, q, 2 * 2 * D, scs, sm, n3, E->x); TOC(c3, P_C3); }
		for (k = 0; k < (size_t)n3 * D; k++)
			E->x[k] = gelu(E->x[k]);
		free(q);
		free(fs);
		free(sm);
		free(scs);
		free(x2);
	}
	E->T = n3;
	TOC(t0, P_CONV);

	/* transformer */
	{
		const int T = n3;
		float *h = malloc((size_t)T * D * 4), *qf = malloc((size_t)T * D * 4), *kf = malloc((size_t)T * D * 4),
		      *vf = malloc((size_t)T * D * 4), *ao = malloc((size_t)T * D * 4), *ff = malloc((size_t)T * FF * 4),
		      *sc = malloc((size_t)T * 4);
		int16_t *qa = malloc((size_t)T * FF * 2), *ebuf = malloc(ATT_SCRATCH(T));
		int32_t *sm = malloc((size_t)T * 4);
		kv_t kv;
		rope_prepare(m, T);
		for (l = 0; l < NL; l++) {
			const layer_t *Y = &m->enc[l];
			TIC(a0);
			for (t = 0; t < T; t++) {
				layernorm(E->x + (size_t)t * D, Y->ln1, h + (size_t)t * D, D);
				quant(h + (size_t)t * D, D, qa + (size_t)t * D, &sc[t], &sm[t]);
			}
			TOC(a0, P_ENC_ELT);
			TIC(a1);
			lin_run(&Y->q, qa, D, sc, sm, T, qf);
			lin_run(&Y->k, qa, D, sc, sm, T, kf);
			lin_run(&Y->v, qa, D, sc, sm, T, vf);
			TOC(a1, P_ENC_LIN);
			TIC(a2);
			for (t = 0; t < T; t++) {
				rotary(m, qf + (size_t)t * D, t);
				rotary(m, kf + (size_t)t * D, t);
			}
			TOC(a2, P_ENC_ELT);
			TIC(a3);
			kv_alloc(&kv, T);
			kv_fill(&kv, kf, vf);
			for (t = 0; t < T; t += QB)
				attend_blk(m, qf + (size_t)t * D, T - t < QB ? T - t : QB, &kv, ao + (size_t)t * D, ebuf);
			kv_free(&kv);
			TOC(a3, P_ENC_ATT);
			TIC(a4);
			for (t = 0; t < T; t++)
				quant(ao + (size_t)t * D, D, qa + (size_t)t * D, &sc[t], &sm[t]);
			TOC(a4, P_ENC_ELT);
			TIC(a5);
			lin_run(&Y->o, qa, D, sc, sm, T, h);
			TOC(a5, P_ENC_LIN);
			TIC(a6);
			for (t = 0; t < T; t++) {
				int k;
				float *xr = E->x + (size_t)t * D, *hr = h + (size_t)t * D;
				for (k = 0; k < D; k++)
					xr[k] += hr[k];
				layernorm(xr, Y->ln2, hr, D);
				quant(hr, D, qa + (size_t)t * D, &sc[t], &sm[t]);
			}
			TOC(a6, P_ENC_ELT);
			TIC(a7);
			lin_run(&Y->fc1, qa, D, sc, sm, T, ff);
			TOC(a7, P_ENC_LIN);
			TIC(a8);
			for (t = 0; t < T; t++) {
				int k;
				for (k = 0; k < FF; k++)
					ff[(size_t)t * FF + k] = gelu(ff[(size_t)t * FF + k]);
				quant(ff + (size_t)t * FF, FF, qa + (size_t)t * FF, &sc[t], &sm[t]);
			}
			TOC(a8, P_ENC_ELT);
			TIC(a9);
			lin_run(&Y->fc2, qa, FF, sc, sm, T, h);
			TOC(a9, P_ENC_LIN);
			TIC(a10);
			for (t = 0; t < T * D; t++)
				E->x[t] += h[t];
			TOC(a10, P_ENC_ELT);
		}
		TIC(a11);
		for (t = 0; t < T; t++)
			layernorm(E->x + (size_t)t * D, m->enc_ln, h + (size_t)t * D, D);
		memcpy(E->x, h, (size_t)T * D * 4);
		TOC(a11, P_ENC_ELT);
		free(h);
		free(qf);
		free(kf);
		free(vf);
		free(ao);
		free(ff);
		free(sc);
		free(qa);
		free(ebuf);
		free(sm);
	}
}

/* ------------------------------------------------------------------ */
/* decoder                                                             */

typedef struct {
	kv_t cross[NL];
	float *sk[NL], *sv[NL]; /* self-attention cache [MAXDEC][D] */
} dec_t;

static void embed_row(const lin_t *E, int id, float *out)
{
	const uint32_t *w = E->w + (size_t)id * (D / 4);
	float s = E->sw[id];
	int i;
	for (i = 0; i < D / 4; i++) {
		uint32_t x = w[i];
		out[4 * i] = (float)((int)(x & 0xff) - 128) * s;
		out[4 * i + 1] = (float)((int)((x >> 16) & 0xff) - 128) * s;
		out[4 * i + 2] = (float)((int)((x >> 8) & 0xff) - 128) * s;
		out[4 * i + 3] = (float)((int)(x >> 24) - 128) * s;
	}
}

static int dec_step(moonstt *m, dec_t *S, int tok, int pos, int16_t *ebuf)
{
	float x[D], h[D], q[D], k[D], v[D], ao[D], ff[2 * FF], sc;
	int16_t qa[2 * FF];
	int32_t sm;
	int l, i, hh, j, best;
	float bestv;

	embed_row(&m->emb, tok, x);
	for (l = 0; l < NL; l++) {
		const layer_t *Y = &m->dec[l];
		TIC(a0);
		layernorm(x, Y->ln1, h, D);
		quant(h, D, qa, &sc, &sm);
		TOC(a0, P_DEC_ELT);
		TIC(a1);
		lin_run(&Y->q, qa, D, &sc, &sm, 1, q);
		lin_run(&Y->k, qa, D, &sc, &sm, 1, k);
		lin_run(&Y->v, qa, D, &sc, &sm, 1, v);
		TOC(a1, P_DEC_LIN);
		TIC(a2);
		rotary(m, q, pos);
		rotary(m, k, pos);
		memcpy(S->sk[l] + (size_t)pos * D, k, D * 4);
		memcpy(S->sv[l] + (size_t)pos * D, v, D * 4);
		for (hh = 0; hh < NH; hh++) {
			float p[MAXDEC], mx = -1e30f, sum = 0;
			for (j = 0; j <= pos; j++) {
				const float *kk = S->sk[l] + (size_t)j * D + hh * HD;
				float s = 0;
				for (i = 0; i < HD; i++)
					s += q[hh * HD + i] * kk[i];
				p[j] = s * (1.0f / 6.0f);
				if (p[j] > mx)
					mx = p[j];
			}
			for (j = 0; j <= pos; j++) {
				p[j] = expf(p[j] - mx);
				sum += p[j];
			}
			for (i = 0; i < HD; i++)
				ao[hh * HD + i] = 0;
			for (j = 0; j <= pos; j++) {
				const float *vv = S->sv[l] + (size_t)j * D + hh * HD;
				float w = p[j] / sum;
				for (i = 0; i < HD; i++)
					ao[hh * HD + i] += w * vv[i];
			}
		}
		TOC(a2, P_DEC_ATT);
		TIC(a3);
		quant(ao, D, qa, &sc, &sm);
		TOC(a3, P_DEC_ELT);
		TIC(a4);
		lin_run(&Y->o, qa, D, &sc, &sm, 1, h);
		TOC(a4, P_DEC_LIN);
		for (i = 0; i < D; i++)
			x[i] += h[i];
		/* cross attention */
		TIC(a5);
		layernorm(x, Y->ln2, h, D);
		quant(h, D, qa, &sc, &sm);
		TOC(a5, P_DEC_ELT);
		TIC(a6);
		lin_run(&Y->cq, qa, D, &sc, &sm, 1, q);
		TOC(a6, P_DEC_LIN);
		TIC(a7);
		attend_blk(m, q, 1, &S->cross[l], ao, ebuf);
		TOC(a7, P_DEC_ATT);
		TIC(a8);
		quant(ao, D, qa, &sc, &sm);
		TOC(a8, P_DEC_ELT);
		TIC(a9);
		lin_run(&Y->co, qa, D, &sc, &sm, 1, h);
		TOC(a9, P_DEC_LIN);
		for (i = 0; i < D; i++)
			x[i] += h[i];
		/* MLP (SwiGLU) */
		TIC(a10);
		layernorm(x, Y->ln3, h, D);
		quant(h, D, qa, &sc, &sm);
		TOC(a10, P_DEC_ELT);
		TIC(a11);
		lin_run(&Y->fc1, qa, D, &sc, &sm, 1, ff);
		TOC(a11, P_DEC_LIN);
		TIC(a12);
		for (i = 0; i < FF; i++)
			ff[i] = ff[i] * silu(ff[FF + i]);
		quant(ff, FF, qa, &sc, &sm);
		TOC(a12, P_DEC_ELT);
		TIC(a13);
		lin_run(&Y->fc2, qa, FF, &sc, &sm, 1, h);
		TOC(a13, P_DEC_LIN);
		for (i = 0; i < D; i++)
			x[i] += h[i];
	}
	/* head */
	{
		TIC(a14);
		layernorm(x, m->dec_ln, h, D);
		quant(h, D, qa, &sc, &sm);
		best = 0;
		bestv = -1e30f;
		for (j = 0; j < m->n_lm; j += 2) {
			int32_t r[2];
			int j0 = m->lm_ids[j], j1 = m->lm_ids[j + 1];
			float v0, v1;
			dot_w12(m->emb.w + (size_t)j0 * (D / 4), m->emb.w + (size_t)j1 * (D / 4), qa, D / 4, r);
			v0 = (float)(r[0] - 128 * sm) * m->emb.sw[j0];
			v1 = (float)(r[1] - 128 * sm) * m->emb.sw[j1];
			if (v0 > bestv) {
				bestv = v0;
				best = j0;
			}
			if (v1 > bestv) {
				bestv = v1;
				best = j1;
			}
		}
		TOC(a14, P_HEAD);
	}
	return best;
}

/* ------------------------------------------------------------------ */

static void detok(const moonstt *m, const int *ids, int n, char *out, size_t cap)
{
	size_t o = 0;
	int i;
	for (i = 0; i < n; i++) {
		uint32_t a = m->tokoff[ids[i]], b = m->tokoff[ids[i] + 1];
		if (o + (b - a) + 1 >= cap)
			break;
		memcpy(out + o, m->tokblob + a, b - a);
		o += b - a;
	}
	out[o] = 0;
	if (out[0] == ' ')
		memmove(out, out + 1, o);
}

int moonstt_transcribe(moonstt *m, const int16_t *pcm, int n, char *text, size_t cap, int verbose)
{
	enc_t E = { 0, NULL };
	dec_t S;
	int ids[MAXDEC], nid = 0, maxtok, tok = TOK_BOS, pos, l;
	double t0 = now(), t1, t2;
	int16_t *ebuf;

	memset(prof, 0, sizeof prof);
	text[0] = 0;
	if (n < 127 + 64 * 10 + 400)
		return 0;
	if (conv_len(n) < 4)
		return 0;
	encode(m, pcm, n, &E);
	t1 = now();
	/* cross-attention K/V */
	{
		const int T = E.T;
		float *k = malloc((size_t)T * D * 4), *v = malloc((size_t)T * D * 4), *sc = malloc(T * 4);
		int16_t *qa = malloc((size_t)T * D * 2);
		int32_t *sm = malloc(T * 4);
		int t;
		TIC(a0);
		for (t = 0; t < T; t++)
			quant(E.x + (size_t)t * D, D, qa + (size_t)t * D, &sc[t], &sm[t]);
		for (l = 0; l < NL; l++) {
			lin_run(&m->dec[l].ck, qa, D, sc, sm, T, k);
			lin_run(&m->dec[l].cv, qa, D, sc, sm, T, v);
			kv_alloc(&S.cross[l], T);
			kv_fill(&S.cross[l], k, v);
		}
		TOC(a0, P_XKV);
		free(k);
		free(v);
		free(sc);
		free(qa);
		free(sm);
	}
	for (l = 0; l < NL; l++) {
		S.sk[l] = malloc((size_t)MAXDEC * D * 4);
		S.sv[l] = malloc((size_t)MAXDEC * D * 4);
	}
	ebuf = malloc(ATT_SCRATCH(E.T));
	maxtok = (int)(6.5 * n / 16000.0);
	if (maxtok < 4)
		maxtok = 4;
	if (maxtok > MAXDEC - 2)
		maxtok = MAXDEC - 2;
	for (pos = 0; pos < maxtok + 1; pos++) {
		tok = dec_step(m, &S, tok, pos, ebuf);
		if (tok == TOK_EOS)
			break;
		ids[nid++] = tok;
		if (nid >= maxtok)
			break;
	}
	t2 = now();
	detok(m, ids, nid, text, cap);
	if (verbose) {
		int i;
		fprintf(stderr, "audio %.2fs  encoder %.2fs  decoder %.2fs (%d tokens)  RTF %.2f\n", n / 16000.0, t1 - t0,
			t2 - t1, nid, (t2 - t0) / (n / 16000.0));
		for (i = 0; i < P_N; i++)
			fprintf(stderr, "  %-18s %7.2fs\n", pname[i], prof[i]);
		if (getenv("MOONSTT_IDS")) {
			for (i = 0; i < nid; i++)
				fprintf(stderr, "%d ", ids[i]);
			fprintf(stderr, "\n");
		}
	}
	if (getenv("MOONSTT_DUMP_ENC")) {
		FILE *f = fopen(getenv("MOONSTT_DUMP_ENC"), "wb");
		if (f) {
			fwrite(E.x, 4, (size_t)E.T * D, f);
			fclose(f);
		}
	}
	for (l = 0; l < NL; l++) {
		kv_free(&S.cross[l]);
		free(S.sk[l]);
		free(S.sv[l]);
	}
	free(ebuf);
	free(E.x);
	return nid;
}
