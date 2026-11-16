#!/bin/sh
# Assemble moonstt_<ver>_armel.deb and moonstt-model_<ver>_all.deb.
# Usage: build.sh <version> <src-dir> <model-dir> <out-dir>
# <src-dir> holds the built moonstt, moonstt-dictate; <model-dir> model.bin, tok.bin.
set -e
VER=$1 SRC=$(cd "$2" && pwd) MODEL=$(cd "$3" && pwd) OUT=$4
PKG=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# --- moonstt: binaries, service, defaults --------------------------------
R=$T/moonstt
mkdir -p "$R/DEBIAN" "$R/usr/bin" "$R/lib/systemd/system" "$R/etc/default" "$R/etc/init.d" "$R/usr/share/doc/moonstt"
install -m 0755 "$SRC/moonstt" "$SRC/moonstt-dictate" "$SRC/moonstt-ptt" "$R/usr/bin/"
install -m 0644 "$PKG/init/moonstt-dictate.service" "$R/lib/systemd/system/"
install -m 0644 "$PKG/init/moonstt-dictate.default" "$R/etc/default/moonstt-dictate"
install -m 0755 "$PKG/init/S96moonstt-dictate" "$R/etc/init.d/moonstt-dictate"
install -m 0644 "$PKG/README.md" "$PKG/LICENSE" "$PKG/THIRD-PARTY.md" "$R/usr/share/doc/moonstt/"
cat > "$R/DEBIAN/control" <<EOC
Package: moonstt
Version: $VER
Architecture: armel
Maintainer: brain-hackers <noreply@example.invalid>
Depends: alsa-utils, moonstt-model (>= $VER)
Section: sound
Priority: optional
Description: Offline hold-to-talk dictation for SHARP Brain
 Moonshine Tiny speech recognition for the i.MX28 (ARM926, no FPU), a
 framebuffer HUD and a daemon that types the transcript while a key is held.
EOC
echo /etc/default/moonstt-dictate > "$R/DEBIAN/conffiles"
cat > "$R/DEBIAN/postinst" <<'EOC'
#!/bin/sh
set -e
if [ -d /run/systemd/system ]; then
	systemctl daemon-reload || true
	systemctl enable moonstt-dictate.service || true
	systemctl restart moonstt-dictate.service || true
fi
EOC
cat > "$R/DEBIAN/prerm" <<'EOC'
#!/bin/sh
set -e
if [ -d /run/systemd/system ]; then
	systemctl disable --now moonstt-dictate.service || true
fi
EOC
chmod 0755 "$R/DEBIAN/postinst" "$R/DEBIAN/prerm"
fakeroot dpkg-deb --build "$R" "$OUT/moonstt_${VER}_armel.deb"

# --- moonstt-model: converted weights ------------------------------------
M=$T/model
mkdir -p "$M/DEBIAN" "$M/usr/share/moonstt"
install -m 0644 "$MODEL/model.bin" "$MODEL/tok.bin" "$M/usr/share/moonstt/"
cat > "$M/DEBIAN/control" <<EOC
Package: moonstt-model
Version: $VER
Architecture: all
Maintainer: brain-hackers <noreply@example.invalid>
Section: sound
Priority: optional
Description: Moonshine Tiny weights in the moonstt int8 format
 Converted from UsefulSensors/moonshine-tiny (MIT licence). Little-endian only.
EOC
fakeroot dpkg-deb --build "$M" "$OUT/moonstt-model_${VER}_all.deb"
ls -la "$OUT"/*.deb
