# sharp-brain-push-to-dictate

Offline hold-to-talk dictation for SHARP Brain (engine: `moonstt`).

Hold a key, speak, let go: the text is typed at your cursor. Everything runs
on the device (i.MX28, ARM926EJ-S, no FPU/NEON, 128 MB RAM) with
[Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)
(27M parameters, English, MIT).

- **`moonstt`**: speech-to-text for a WAV file.
- **`moonstt-dictate`**: the daemon. Watches a key, shows a "Listening..." /
  "Transcribing..." HUD on the framebuffer, runs `moonstt`, types the result.
- **`moonstt-ptt`**: a tiny Enter-to-record shell script for testing over SSH.

Speed is the catch: about **5x real time** (a 5 s sentence takes ~25 s). It is
for note-taking, not live captions.

## Requirements

| Need | Why |
|---|---|
| Microphone capture in ALSA (`arecord -D hw:0,1 -f S16_LE -r 44100 -c 2`) | [linux-brain PR #41](https://github.com/brain-hackers/linux-brain/pull/41). The codec puts the mono mic in the left slot; `moonstt` picks the louder channel and resamples. |
| A kernel that reports the HOME key | The PW-SH3 device tree leaves HOME unmapped. See [The HOME key](#the-home-key), or pick another key with `-k`. |
| `arecord` (alsa-utils) | Recording. |
| ~64 MB free RAM, ~28 MB of storage | Model is memory-mapped. |

## Install

### A. Debian / Brainux (`.deb`)

```sh
make deb                      # builds ./dist/moonstt_*_armel.deb and moonstt-model_*_all.deb (needs Docker)
dpkg -i dist/moonstt-model_*.deb dist/moonstt_*.deb
```

The service starts on install (`systemctl status moonstt-dictate`). Options go in
`/etc/default/moonstt-dictate`.

### B. Buildroot

This repository is a Buildroot external tree. Add it to `BR2_EXTERNAL`
(`make BR2_EXTERNAL=/path/to/sharp-brain-push-to-dictate ...`), enable
`BR2_PACKAGE_MOONSTT`, and build the model once:

```sh
tools/build-model.sh model     # needs curl + python3 + numpy
# then in menuconfig: moonstt -> model directory = /abs/path/to/model
```

The package installs the binaries, `/etc/init.d/S96moonstt-dictate` and
`/etc/default/moonstt-dictate`.

### C. By hand

```sh
make            # static armel binaries (arm-linux-gnueabi-gcc); `make host` for a PC build of the engine
tools/build-model.sh model
# copy to the device (Dropbear has no sftp, so pipe over ssh):
ssh root@brain 'cat > /usr/bin/moonstt-dictate && chmod +x /usr/bin/moonstt-dictate' < src/moonstt-dictate
ssh root@brain 'mkdir -p /usr/share/moonstt && cat > /usr/share/moonstt/model.bin' < model/model.bin
# ...same for moonstt, moonstt-ptt, tok.bin and init/S96moonstt-dictate -> /etc/init.d/
```

## Use

Hold **HOME**, speak, release. A red *Listening* pill shows while you hold; it
turns amber (*Transcribing... Ns*) after release; the text is then typed with a
trailing space. Taps under 0.4 s are ignored, recordings stop at 25 s, leading
and trailing silence is trimmed, and pauses over 0.5 s are shortened (so a
full stop can turn into a comma).

```sh
moonstt-dictate -v                 # run in the foreground with logging
moonstt-dictate -k 110             # use another key (110 = KEY_INSERT)
moonstt -v -m /usr/share/moonstt clip.wav   # transcribe a file, with per-stage timings
```

| Option | Meaning |
|---|---|
| `-k CODE` | talk key (default 172, `KEY_HOMEPAGE`) |
| `-o auto\|uinput\|tty` | typing backend (below) |
| `-g N` | mic capture gain 0-7 (default 7) |
| `-n` | no trailing space |
| `-t TTY` | console to type into (default: the active VT) |
| `-a DEV` | ALSA capture device (default `hw:0,1`) |
| `-S SECS` | test mode: simulate holding the key |

### Typing backends

- **`uinput`** (preferred, used automatically when `/dev/uinput` exists): a
  virtual US-layout keyboard, so text goes to whatever app has focus, console or
  X. Needs `CONFIG_INPUT_UINPUT` in the kernel
  (`brain_defconfig` has it off: set `CONFIG_INPUT_UINPUT=y`).
- **`tty`** (fallback): `TIOCSTI` into the active virtual console. Works on the
  stock kernel, but only reaches the Linux console. Kernels 6.2+ may disable
  it (`dev.tty.legacy_tiocsti`).

### The HOME key

The matrix position exists but is commented out in `imx28-pwsh3.dts`. Replace the
two `/* <... 6 Home>, */` comments (in `keymap` and `keymap-symbol`) with:

```
<0 6 KEY_HOMEPAGE>,   /* Home */
```

`KEY_HOMEPAGE` does nothing on the console, so holding it types nothing. This is
the whole kernel change (not merged upstream yet). On a board that boots
from a U-Boot `uEnv.txt` you can try it without reflashing the kernel: put the
rebuilt `.dtb` next to the kernel and set `fdt_file=` (keep a copy of the old one).

## Performance (PW-SH3)

| Clip | Real-time factor |
|---|---|
| 2 s | ~4.8 |
| 4-6 s | ~5.0-5.3 |
| 8 s, with pauses squeezed | ~5.5 |

The engine matches the Hugging Face float reference token-for-token on most test
clips (one near-tie word can differ). Per-stage timings: `moonstt -v`.

## How it works

- Weights: int8 per output row, offset-binary, packed so `x & 0x00ff00ff` feeds
  ARMv5TE `SMLABB/SMLATT` directly (`tools/convert.py`).
- Activations: int16, dynamic power-of-two scale; attention scores and `P*V` are
  integer; GELU / tanh / exp are table lookups. Layer norm, rotary and residuals
  are still soft-float.
- Matrix kernels do 2 weight rows x 2 frames per pass (~230 MMAC/s in cache,
  ~140 on real data). K-chunked tiling keeps the 16 KB data cache hot.
- Output vocabulary is limited to ASCII tokens.
- `pld` prefetch and `ldm` bursts were measured and are *slower* on the ARM926.

## Troubleshooting

- **Nothing happens on HOME**: check the kernel reports it, e.g. dump
  `/dev/input/event0` (`key 172`). If not, see [The HOME key](#the-home-key).
- **"no input device matching 'brain-kbd'"**: use `-d <name substring>`.
- **Garbled or empty text**: raise the mic gain (`-g 7`), speak close to the
  mic, check `arecord` works; background music hurts a lot.
- **Typed text goes nowhere**: with `tty` backend it types into the *active* VT
  (`cat /sys/class/tty/tty0/active`); a shell must be reading it.
- **HUD overlaps something**: it draws a 400x56 box at the top centre of `/dev/fb0`
  and restores the pixels when it hides.

## Development

```sh
make host test          # needs reference transcripts: see tests/README.md
tools/build-model.sh    # re-download and re-convert the weights
```

Layout: `src/` (engine + daemon), `tools/` (model conversion), `init/` (SysV,
systemd, defaults), `package/moonstt/` (Buildroot package; the repo root is the
external tree), `packaging/` (Dockerfile, `.deb` assembly), `tests/`,
`.github/workflows/` (CI and tagged releases). Licences: see `LICENSE` and `THIRD-PARTY.md`.
