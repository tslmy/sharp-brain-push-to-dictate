/* moonstt: transcribe a WAV file (or raw S16LE mono 16 kHz on stdin with "-"). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "moonstt.h"

#ifndef MOONSTT_DATADIR
#define MOONSTT_DATADIR "/usr/share/moonstt"
#endif

static int rd32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24; }
static int rd16(const unsigned char *p) { return p[0] | p[1] << 8; }

/* Returns mono 16 kHz samples (malloc'd) or NULL. */
static int16_t *load_wav(FILE *f, int *n_out)
{
	unsigned char h[12], ck[8];
	int ch = 1, rate = 16000, bits = 16;
	unsigned char *data = NULL;
	long dlen = 0;
	int16_t *pcm, *mono;
	long n, i, nout;
	int chsel;

	if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4))
		return NULL;
	while (fread(ck, 1, 8, f) == 8) {
		long len = (unsigned)rd32(ck + 4);
		if (!memcmp(ck, "fmt ", 4)) {
			unsigned char fm[40];
			long r = len < 40 ? len : 40;
			if (fread(fm, 1, r, f) != (size_t)r)
				return NULL;
			if (len > r)
				fseek(f, len - r, SEEK_CUR);
			ch = rd16(fm + 2);
			rate = rd32(fm + 4);
			bits = rd16(fm + 14);
		} else if (!memcmp(ck, "data", 4)) {
			if (len <= 0 || len > (1L << 28)) { /* streamed WAV: read to EOF */
				size_t cap = 1 << 20, got = 0, r;
				data = malloc(cap);
				while ((r = fread(data + got, 1, cap - got, f)) > 0) {
					got += r;
					if (got == cap)
						data = realloc(data, cap *= 2);
				}
				dlen = got;
			} else {
				data = malloc(len);
				dlen = fread(data, 1, len, f);
			}
			break;
		} else
			fseek(f, len + (len & 1), SEEK_CUR);
	}
	if (!data || bits != 16 || ch < 1)
		return NULL;
	pcm = (int16_t *)data;
	n = dlen / (2 * ch);
	/* The codec carries the mono mic in one I2S slot: use the louder channel
	 * rather than averaging, which would halve the level. */
	chsel = 0;
	if (ch > 1) {
		double best = -1;
		int c;
		for (c = 0; c < ch; c++) {
			double e = 0;
			for (i = 0; i < n; i++)
				e += (double)pcm[i * ch + c] * pcm[i * ch + c];
			if (e > best)
				best = e, chsel = c;
		}
	}
	/* box-filter resample to 16 kHz */
	nout = (long)((double)n * 16000 / rate);
	mono = malloc((nout + 1) * 2);
	for (i = 0; i < nout; i++) {
		double a = (double)i * rate / 16000, b = (double)(i + 1) * rate / 16000;
		long s = (long)a, e = (long)b, k;
		double acc = 0, wsum = 0;
		if (e >= n)
			e = n - 1;
		for (k = s; k <= e; k++) {
			double wk = 1.0, lo = k > a ? k : a, hi = k + 1 < b ? k + 1 : b;
			double v;
			wk = hi - lo;
			if (wk <= 0)
				continue;
			v = pcm[k * ch + chsel];
			acc += wk * v;
			wsum += wk;
		}
		mono[i] = wsum > 0 ? (int16_t)(acc / wsum) : 0;
	}
	free(data);
	*n_out = (int)nout;
	return mono;
}

/* Drop leading/trailing silence (keeps 0.3 s of context each side). */
static void trim_silence(int16_t *pcm, int *np)
{
	const int F = 320; /* 20 ms */
	int n = *np, nf = n / F, f, first = -1, last = -1, i, margin = 15;
	double *rms, *srt, thr;
	if (nf < 20)
		return;
	rms = malloc(nf * sizeof *rms);
	srt = malloc(nf * sizeof *srt);
	for (f = 0; f < nf; f++) {
		double e = 0;
		for (i = 0; i < F; i++)
			e += (double)pcm[f * F + i] * pcm[f * F + i];
		rms[f] = srt[f] = sqrt(e / F);
	}
	for (i = 1; i < nf; i++) { /* insertion sort: nf is small */
		double v = srt[i];
		int k = i - 1;
		while (k >= 0 && srt[k] > v)
			srt[k + 1] = srt[k], k--;
		srt[k + 1] = v;
	}
	thr = srt[nf / 10] * 3.0;
	if (thr < 150.0)
		thr = 150.0;
	if (thr > srt[nf - 1] * 0.5)
		thr = srt[nf - 1] * 0.5;
	for (f = 0; f < nf; f++)
		if (rms[f] > thr) {
			if (first < 0)
				first = f;
			last = f;
		}
	if (first >= 0) {
		/* Keep [first-margin, last+margin]; inside it, squeeze silent runs
		 * longer than 0.5 s (25 frames) down to 0.24 s (6 frames per side). */
		int s = first - margin, e = last + 1 + margin, w = 0;
		if (s < 0)
			s = 0;
		if (e > nf)
			e = nf;
		for (f = s; f < e;) {
			int g = f;
			if (rms[f] <= thr) {
				while (g < e && rms[g] <= thr)
					g++;
				if (g - f > 25) { /* silent run [f, g): keep 6 frames at each end */
					memmove(pcm + (size_t)w * F, pcm + (size_t)f * F, (size_t)6 * F * 2);
					w += 6;
					memmove(pcm + (size_t)w * F, pcm + (size_t)(g - 6) * F, (size_t)6 * F * 2);
					w += 6;
					f = g;
					continue;
				}
			} else
				g = f + 1;
			memmove(pcm + (size_t)w * F, pcm + (size_t)f * F, (size_t)(g - f) * F * 2);
			w += g - f;
			f = g;
		}
		*np = w * F;
	}
	free(rms);
	free(srt);
}

int main(int argc, char **argv)
{
	const char *dir = MOONSTT_DATADIR, *file = NULL;
	int verbose = 0, notrim = 0, i, n = 0;
	int16_t *pcm;
	char text[2048];
	moonstt *m;
	FILE *f;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-v"))
			verbose = 1;
		else if (!strcmp(argv[i], "-T"))
			notrim = 1;
		else if (!strcmp(argv[i], "-m") && i + 1 < argc)
			dir = argv[++i];
		else
			file = argv[i];
	}
	if (!file) {
		fprintf(stderr, "usage: moonstt [-v] [-m modeldir] file.wav\n");
		return 2;
	}
	f = !strcmp(file, "-") ? stdin : fopen(file, "rb");
	if (!f) {
		perror(file);
		return 1;
	}
	pcm = load_wav(f, &n);
	if (!pcm) {
		fprintf(stderr, "moonstt: need a 16-bit PCM WAV\n");
		return 1;
	}
	if (!notrim) {
		int n0 = n;
		trim_silence(pcm, &n);
		if (verbose && n != n0)
			fprintf(stderr, "trimmed %.2fs -> %.2fs\n", n0 / 16000.0, n / 16000.0);
	}
	m = moonstt_load(dir);
	if (!m)
		return 1;
	moonstt_transcribe(m, pcm, n, text, sizeof text, verbose);
	puts(text);
	moonstt_free(m);
	return 0;
}
