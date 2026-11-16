// SPDX-License-Identifier: MIT
/*
 * moonstt-dictate: hold-to-talk dictation for SHARP Brain.
 *
 * While the talk key is held it records the microphone and shows a
 * "Listening" HUD on the framebuffer; on release it shows "Transcribing" while
 * moonstt runs, then types the text into the focused Linux console as if it
 * had been typed on the keyboard (TIOCSTI).
 *
 * The keyboard is only observed through evdev (never grabbed), the same way
 * keyoverlay does it, so the talk key must be one the console ignores.  The
 * 8x16 font comes from keyoverlay (MIT).
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <linux/fb.h>
#include <linux/input.h>
#include <linux/uinput.h>

#include "font8x16.h"

#ifndef KEY_HOMEPAGE
#define KEY_HOMEPAGE 172
#endif

static double now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* HUD on the framebuffer                                              */
/* ------------------------------------------------------------------ */

#define SCALE 2
#define HUD_W 400
#define HUD_H 56

struct hud {
	int fd;
	uint8_t *mem, *save;
	size_t len;
	uint32_t xres, yres, line, bpp;
	struct fb_var_screeninfo var;
	int x, y, shown;
};

static uint32_t pack(const struct hud *h, int r, int g, int b)
{
	return ((uint32_t)(r >> (8 - h->var.red.length)) << h->var.red.offset) |
	       ((uint32_t)(g >> (8 - h->var.green.length)) << h->var.green.offset) |
	       ((uint32_t)(b >> (8 - h->var.blue.length)) << h->var.blue.offset);
}

static void put(const struct hud *h, int x, int y, uint32_t v)
{
	uint8_t *p;
	if (x < 0 || y < 0 || (uint32_t)x >= h->xres || (uint32_t)y >= h->yres)
		return;
	p = h->mem + (size_t)y * h->line + (size_t)x * h->bpp;
	if (h->bpp == 2)
		*(uint16_t *)p = (uint16_t)v;
	else
		*(uint32_t *)p = v;
}

static void rect(const struct hud *h, int x, int y, int w, int ht, uint32_t v)
{
	int i, j;
	for (j = y; j < y + ht; j++)
		for (i = x; i < x + w; i++)
			put(h, i, j, v);
}

static int hud_open(struct hud *h, const char *path)
{
	struct fb_fix_screeninfo fix;
	memset(h, 0, sizeof *h);
	h->fd = open(path, O_RDWR | O_CLOEXEC);
	if (h->fd < 0)
		return -1;
	if (ioctl(h->fd, FBIOGET_VSCREENINFO, &h->var) < 0 || ioctl(h->fd, FBIOGET_FSCREENINFO, &fix) < 0)
		return -1;
	h->xres = h->var.xres;
	h->yres = h->var.yres;
	h->line = fix.line_length;
	h->bpp = h->var.bits_per_pixel / 8;
	if (h->bpp != 2 && h->bpp != 4)
		return -1;
	h->len = fix.smem_len ? fix.smem_len : (size_t)h->line * h->yres;
	h->mem = mmap(NULL, h->len, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, 0);
	if (h->mem == MAP_FAILED)
		return -1;
	h->x = ((int)h->xres - HUD_W) / 2;
	h->y = 10;
	h->save = malloc((size_t)HUD_W * HUD_H * h->bpp);
	return h->save ? 0 : -1;
}

static void hud_hide(struct hud *h)
{
	int j;
	if (!h->shown)
		return;
	for (j = 0; j < HUD_H; j++)
		memcpy(h->mem + (size_t)(h->y + j) * h->line + (size_t)h->x * h->bpp,
		       h->save + (size_t)j * HUD_W * h->bpp, (size_t)HUD_W * h->bpp);
	h->shown = 0;
}

/* state colour: 0 listening (red), 1 transcribing (amber), 2 neutral (grey) */
static void hud_show(struct hud *h, int state, const char *text)
{
	static const int col[3][3] = { { 230, 50, 50 }, { 240, 170, 30 }, { 150, 150, 150 } };
	int j, i, tx;
	uint32_t panel = pack(h, 24, 24, 28), accent = pack(h, col[state][0], col[state][1], col[state][2]),
		 white = pack(h, 245, 245, 245);
	if (!h->shown) {
		for (j = 0; j < HUD_H; j++)
			memcpy(h->save + (size_t)j * HUD_W * h->bpp,
			       h->mem + (size_t)(h->y + j) * h->line + (size_t)h->x * h->bpp, (size_t)HUD_W * h->bpp);
		h->shown = 1;
	}
	rect(h, h->x, h->y, HUD_W, HUD_H, accent);
	rect(h, h->x + 3, h->y + 3, HUD_W - 6, HUD_H - 6, panel);
	/* status dot */
	rect(h, h->x + 18, h->y + HUD_H / 2 - 9, 18, 18, accent);
	tx = h->x + 52;
	for (i = 0; text[i]; i++) {
		unsigned char ch = (unsigned char)text[i];
		const unsigned char *g;
		int row, c;
		if (ch < FONT_FIRST || ch > FONT_LAST)
			ch = '?';
		g = &font8x16[(ch - FONT_FIRST) * FONT_H];
		for (row = 0; row < FONT_H; row++)
			for (c = 0; c < FONT_W; c++)
				if (g[row] & (0x80 >> c))
					rect(h, tx + i * FONT_W * SCALE + c * SCALE, h->y + (HUD_H - FONT_H * SCALE) / 2 + row * SCALE,
					     SCALE, SCALE, white);
	}
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static int find_keyboard(const char *name, char *path, size_t cap)
{
	int i;
	for (i = 0; i < 32; i++) {
		char p[64], n[128] = "";
		int fd;
		snprintf(p, sizeof p, "/dev/input/event%d", i);
		fd = open(p, O_RDONLY | O_CLOEXEC);
		if (fd < 0)
			continue;
		ioctl(fd, EVIOCGNAME(sizeof n), n);
		close(fd);
		if (strstr(n, name)) {
			snprintf(path, cap, "%s", p);
			return 0;
		}
	}
	return -1;
}

static pid_t spawn_record(const char *dev, const char *wav)
{
	pid_t pid = fork();
	if (pid == 0) {
		execlp("arecord", "arecord", "-q", "-D", dev, "-f", "S16_LE", "-r", "44100", "-c", "2", wav, (char *)NULL);
		_exit(127);
	}
	return pid;
}

static pid_t spawn_transcribe(const char *bin, const char *dir, const char *wav, int *rfd)
{
	int p[2];
	pid_t pid;
	if (pipe2(p, O_CLOEXEC) < 0)
		return -1;
	pid = fork();
	if (pid == 0) {
		dup2(p[1], 1);
		execlp(bin, bin, "-m", dir, wav, (char *)NULL);
		_exit(127);
	}
	close(p[1]);
	*rfd = p[0];
	return pid;
}

/* ---- typing via uinput (works for any focused app, console or X) ---- */

struct ukey {
	char ch;
	unsigned short code;
	unsigned char shift;
};

static const struct ukey ukeys[] = {
	{ 'a', KEY_A, 0 }, { 'b', KEY_B, 0 }, { 'c', KEY_C, 0 }, { 'd', KEY_D, 0 }, { 'e', KEY_E, 0 },
	{ 'f', KEY_F, 0 }, { 'g', KEY_G, 0 }, { 'h', KEY_H, 0 }, { 'i', KEY_I, 0 }, { 'j', KEY_J, 0 },
	{ 'k', KEY_K, 0 }, { 'l', KEY_L, 0 }, { 'm', KEY_M, 0 }, { 'n', KEY_N, 0 }, { 'o', KEY_O, 0 },
	{ 'p', KEY_P, 0 }, { 'q', KEY_Q, 0 }, { 'r', KEY_R, 0 }, { 's', KEY_S, 0 }, { 't', KEY_T, 0 },
	{ 'u', KEY_U, 0 }, { 'v', KEY_V, 0 }, { 'w', KEY_W, 0 }, { 'x', KEY_X, 0 }, { 'y', KEY_Y, 0 },
	{ 'z', KEY_Z, 0 }, { 'A', KEY_A, 1 }, { 'B', KEY_B, 1 }, { 'C', KEY_C, 1 }, { 'D', KEY_D, 1 },
	{ 'E', KEY_E, 1 }, { 'F', KEY_F, 1 }, { 'G', KEY_G, 1 }, { 'H', KEY_H, 1 }, { 'I', KEY_I, 1 },
	{ 'J', KEY_J, 1 }, { 'K', KEY_K, 1 }, { 'L', KEY_L, 1 }, { 'M', KEY_M, 1 }, { 'N', KEY_N, 1 },
	{ 'O', KEY_O, 1 }, { 'P', KEY_P, 1 }, { 'Q', KEY_Q, 1 }, { 'R', KEY_R, 1 }, { 'S', KEY_S, 1 },
	{ 'T', KEY_T, 1 }, { 'U', KEY_U, 1 }, { 'V', KEY_V, 1 }, { 'W', KEY_W, 1 }, { 'X', KEY_X, 1 },
	{ 'Y', KEY_Y, 1 }, { 'Z', KEY_Z, 1 }, { '1', KEY_1, 0 }, { '2', KEY_2, 0 }, { '3', KEY_3, 0 },
	{ '4', KEY_4, 0 }, { '5', KEY_5, 0 }, { '6', KEY_6, 0 }, { '7', KEY_7, 0 }, { '8', KEY_8, 0 },
	{ '9', KEY_9, 0 }, { '0', KEY_0, 0 }, { '!', KEY_1, 1 }, { '@', KEY_2, 1 }, { '#', KEY_3, 1 },
	{ '$', KEY_4, 1 }, { '%', KEY_5, 1 }, { '^', KEY_6, 1 }, { '&', KEY_7, 1 }, { '*', KEY_8, 1 },
	{ '(', KEY_9, 1 }, { ')', KEY_0, 1 }, { ' ', KEY_SPACE, 0 }, { '\n', KEY_ENTER, 0 },
	{ '-', KEY_MINUS, 0 }, { '_', KEY_MINUS, 1 }, { '=', KEY_EQUAL, 0 }, { '+', KEY_EQUAL, 1 },
	{ '[', KEY_LEFTBRACE, 0 }, { '{', KEY_LEFTBRACE, 1 }, { ']', KEY_RIGHTBRACE, 0 },
	{ '}', KEY_RIGHTBRACE, 1 }, { ';', KEY_SEMICOLON, 0 }, { ':', KEY_SEMICOLON, 1 },
	{ '\'', KEY_APOSTROPHE, 0 }, { '"', KEY_APOSTROPHE, 1 }, { '`', KEY_GRAVE, 0 }, { '~', KEY_GRAVE, 1 },
	{ '\\', KEY_BACKSLASH, 0 }, { '|', KEY_BACKSLASH, 1 }, { ',', KEY_COMMA, 0 }, { '<', KEY_COMMA, 1 },
	{ '.', KEY_DOT, 0 }, { '>', KEY_DOT, 1 }, { '/', KEY_SLASH, 0 }, { '?', KEY_SLASH, 1 },
};
#define NUKEYS ((int)(sizeof ukeys / sizeof ukeys[0]))

static int ui_fd = -1;

static void ui_emit(int type, int code, int value)
{
	struct input_event e;
	memset(&e, 0, sizeof e);
	e.type = type;
	e.code = code;
	e.value = value;
	if (write(ui_fd, &e, sizeof e) < 0) {
	}
}

static int uinput_open(void)
{
	static const char *paths[] = { "/dev/uinput", "/dev/input/uinput" };
	struct uinput_setup us;
	int i;
	for (i = 0; i < 2 && ui_fd < 0; i++)
		ui_fd = open(paths[i], O_WRONLY | O_NONBLOCK | O_CLOEXEC);
	if (ui_fd < 0)
		return -1;
	ioctl(ui_fd, UI_SET_EVBIT, EV_KEY);
	ioctl(ui_fd, UI_SET_EVBIT, EV_SYN);
	ioctl(ui_fd, UI_SET_KEYBIT, KEY_LEFTSHIFT);
	for (i = 0; i < NUKEYS; i++)
		ioctl(ui_fd, UI_SET_KEYBIT, ukeys[i].code);
	memset(&us, 0, sizeof us);
	us.id.bustype = BUS_VIRTUAL;
	us.id.vendor = 0x1209;
	us.id.product = 0x0001;
	snprintf(us.name, sizeof us.name, "moonstt-dictate");
	if (ioctl(ui_fd, UI_DEV_SETUP, &us) < 0 || ioctl(ui_fd, UI_DEV_CREATE) < 0) {
		close(ui_fd);
		ui_fd = -1;
		return -1;
	}
	usleep(200000); /* let udev / X pick the new keyboard up */
	return 0;
}

static int inject_uinput(const char *text)
{
	const char *p;
	for (p = text; *p; p++) {
		int i;
		for (i = 0; i < NUKEYS && ukeys[i].ch != *p; i++)
			;
		if (i == NUKEYS)
			continue; /* not typeable on a US layout */
		if (ukeys[i].shift) {
			ui_emit(EV_KEY, KEY_LEFTSHIFT, 1);
			ui_emit(EV_SYN, SYN_REPORT, 0);
		}
		ui_emit(EV_KEY, ukeys[i].code, 1);
		ui_emit(EV_SYN, SYN_REPORT, 0);
		ui_emit(EV_KEY, ukeys[i].code, 0);
		ui_emit(EV_SYN, SYN_REPORT, 0);
		if (ukeys[i].shift) {
			ui_emit(EV_KEY, KEY_LEFTSHIFT, 0);
			ui_emit(EV_SYN, SYN_REPORT, 0);
		}
		usleep(4000);
	}
	return 0;
}

/* Type text into a console as if it came from the keyboard. */
static int inject(const char *tty, const char *text)
{
	char path[64];
	int fd, ok = 0;
	const char *p;
	if (!tty) {
		char act[32] = "";
		FILE *f = fopen("/sys/class/tty/tty0/active", "r");
		if (f) {
			if (fscanf(f, "%31s", act) != 1)
				act[0] = 0;
			fclose(f);
		}
		snprintf(path, sizeof path, "/dev/%s", act[0] ? act : "tty1");
		tty = path;
	}
	fd = open(tty, O_RDWR | O_NOCTTY | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "dictate: open %s: %s\n", tty, strerror(errno));
		return -1;
	}
	for (p = text; *p; p++)
		if (ioctl(fd, TIOCSTI, p) < 0) {
			fprintf(stderr, "dictate: TIOCSTI on %s: %s\n", tty, strerror(errno));
			ok = -1;
			break;
		}
	close(fd);
	return ok;
}

static volatile sig_atomic_t quit;
static void on_sig(int s)
{
	(void)s;
	quit = 1;
}

enum { IDLE, REC, XCR };

int main(int argc, char **argv)
{
	const char *kbname = "brain-kbd", *fbpath = "/dev/fb0", *adev = "hw:0,1", *dir = "/usr/share/moonstt",
		   *bin = "moonstt", *tty = NULL, *wav = "/tmp/dictate.wav", *output = "auto";
	int key = KEY_HOMEPAGE, verbose = 0, trailing = 1, gain = 7, i, evfd = -1, state = IDLE, xfd = -1;
	double sim = 0, t_press = 0, t_state = 0, t_msg = 0, sim_end = 0;
	pid_t rec = -1, xcr = -1;
	struct hud hud;
	int have_hud;
	char evpath[64] = "", out[4096];
	size_t outn = 0;
	char label[64];

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-k") && i + 1 < argc)
			key = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-d") && i + 1 < argc)
			kbname = argv[++i];
		else if (!strcmp(argv[i], "-f") && i + 1 < argc)
			fbpath = argv[++i];
		else if (!strcmp(argv[i], "-a") && i + 1 < argc)
			adev = argv[++i];
		else if (!strcmp(argv[i], "-m") && i + 1 < argc)
			dir = argv[++i];
		else if (!strcmp(argv[i], "-b") && i + 1 < argc)
			bin = argv[++i];
		else if (!strcmp(argv[i], "-t") && i + 1 < argc)
			tty = argv[++i];
		else if (!strcmp(argv[i], "-o") && i + 1 < argc)
			output = argv[++i]; /* auto | uinput | tty */
		else if (!strcmp(argv[i], "-g") && i + 1 < argc)
			gain = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-S") && i + 1 < argc)
			sim = atof(argv[++i]); /* test: simulate holding the key for N seconds, then exit */
		else if (!strcmp(argv[i], "-n"))
			trailing = 0;
		else if (!strcmp(argv[i], "-v"))
			verbose = 1;
		else {
			fprintf(stderr,
				"usage: moonstt-dictate [-k keycode] [-d kbd-name] [-f fb] [-a alsa-dev] [-m modeldir]\n"
				"       [-b moonstt] [-o auto|uinput|tty] [-t tty] [-g miccap] [-n no trailing space] [-S secs] [-v]\n"
				"Hold the talk key (default KEY_HOMEPAGE=172), speak, release.\n");
			return 2;
		}
	}

	if (!sim && find_keyboard(kbname, evpath, sizeof evpath) < 0) {
		fprintf(stderr, "dictate: no input device matching '%s'\n", kbname);
		return 1;
	}
	if (!sim) {
		evfd = open(evpath, O_RDONLY | O_CLOEXEC);
		if (evfd < 0) {
			perror(evpath);
			return 1;
		}
	}
	have_hud = hud_open(&hud, fbpath) == 0;
	if (!have_hud)
		fprintf(stderr, "dictate: no framebuffer HUD (%s)\n", fbpath);
	{
		char cmd[64];
		snprintf(cmd, sizeof cmd, "amixer -q sset Mic %d >/dev/null 2>&1", gain);
		if (system(cmd)) {
		}
	}
	if (strcmp(output, "tty") && uinput_open() < 0 && !strcmp(output, "uinput")) {
		fprintf(stderr, "dictate: cannot open /dev/uinput\n");
		return 1;
	}
	if (verbose)
		fprintf(stderr, "dictate: typing via %s\n", ui_fd >= 0 ? "uinput" : "console (TIOCSTI)");
	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);
	signal(SIGPIPE, SIG_IGN);
	if (verbose)
		fprintf(stderr, "dictate: key %d on %s, model %s\n", key, sim ? "(simulated)" : evpath, dir);

	if (sim) { /* act as if the key went down now */
		unlink(wav);
		rec = spawn_record(adev, wav);
		state = REC;
		t_press = t_state = now();
		sim_end = t_press + sim;
	}

	while (!quit) {
		struct pollfd pf[2];
		int np = 0, evi = -1, xi = -1, r;
		double t = now();
		if (evfd >= 0)
			evi = np, pf[np++] = (struct pollfd){ evfd, POLLIN, 0 };
		if (xfd >= 0)
			xi = np, pf[np++] = (struct pollfd){ xfd, POLLIN, 0 };
		r = poll(pf, np, 150);
		if (r < 0 && errno != EINTR)
			break;
		t = now();

		if (evi >= 0 && r > 0 && (pf[evi].revents & POLLIN)) {
			struct input_event ev[16];
			ssize_t n = read(evfd, ev, sizeof ev);
			int k;
			if (n <= 0 && errno != EINTR && errno != EAGAIN)
				break;
			for (k = 0; k < (int)(n / (ssize_t)sizeof ev[0]); k++) {
				if (ev[k].type != EV_KEY || ev[k].code != key)
					continue;
				if (ev[k].value == 1 && state == IDLE) {
					unlink(wav);
					rec = spawn_record(adev, wav);
					state = REC;
					t_press = t_state = t;
				} else if (ev[k].value == 0 && state == REC)
					sim_end = t; /* handled below as release */
			}
		}

		if (state == IDLE && t_msg && t > t_msg) {
			if (have_hud)
				hud_hide(&hud);
			t_msg = 0;
		}

		if (state == REC) {
			double held = t - t_press;
			int released = (sim_end && t >= sim_end) || (!sim && sim_end && sim_end >= t_press);
			if (held > 25.0)
				released = 1;
			if (!released) {
				snprintf(label, sizeof label, "Listening... %.1fs", held);
				if (have_hud)
					hud_show(&hud, 0, label);
			} else {
				struct stat st;
				if (rec > 0) {
					kill(rec, SIGINT);
					waitpid(rec, NULL, 0);
					rec = -1;
				}
				sim_end = 0;
				if (held < 0.4 || stat(wav, &st) < 0 || st.st_size < 44100 * 4 / 4) {
					if (have_hud)
						hud_hide(&hud);
					state = IDLE;
					if (sim)
						break;
				} else {
					xcr = spawn_transcribe(bin, dir, wav, &xfd);
					outn = 0;
					state = XCR;
					t_state = t;
				}
			}
		}

		if (state == XCR) {
			if (xi >= 0 && r > 0 && (pf[xi].revents & (POLLIN | POLLHUP))) {
				ssize_t n = read(xfd, out + outn, sizeof out - 1 - outn);
				if (n > 0)
					outn += (size_t)n;
				else if (n == 0) { /* EOF: done */
					close(xfd);
					xfd = -1;
					waitpid(xcr, NULL, 0);
					xcr = -1;
					out[outn] = 0;
					while (outn && (out[outn - 1] == '\n' || out[outn - 1] == ' '))
						out[--outn] = 0;
					if (verbose)
						fprintf(stderr, "dictate: text '%s' (%.1fs)\n", out, t - t_state);
					if (outn) {
						if (trailing && outn + 1 < sizeof out)
							out[outn++] = ' ', out[outn] = 0;
						if (have_hud)
							hud_hide(&hud);
						if (ui_fd >= 0 ? inject_uinput(out) < 0 : inject(tty, out) < 0)
							puts(out);
					} else if (have_hud)
						hud_show(&hud, 2, "Nothing heard"), t_msg = t + 1.2;
					state = IDLE;
					if (sim)
						break;
				}
			}
			if (state == XCR) {
				snprintf(label, sizeof label, "Transcribing... %ds", (int)(t - t_state));
				if (have_hud)
					hud_show(&hud, 1, label);
			}
		}
	}
	if (rec > 0) {
		kill(rec, SIGINT);
		waitpid(rec, NULL, 0);
	}
	if (xcr > 0) {
		kill(xcr, SIGTERM);
		waitpid(xcr, NULL, 0);
	}
	if (have_hud)
		hud_hide(&hud);
	return 0;
}
