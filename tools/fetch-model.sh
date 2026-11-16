#!/bin/sh
# Fetch prebuilt model files onto the device and verify them.
# Usage: MOONSTT_MODEL_URL=https://.../download/model-v1 tools/fetch-model.sh [dir]
# (BusyBox wget has no TLS: use curl or fetch the files on a PC and copy them.)
set -e
URL=${MOONSTT_MODEL_URL:?set MOONSTT_MODEL_URL to the directory holding model.bin, tok.bin and SHA256SUMS}
DIR=${1:-/usr/share/moonstt}
mkdir -p "$DIR"
for f in model.bin tok.bin SHA256SUMS; do
	if command -v curl >/dev/null; then curl -fL -o "$DIR/$f" "$URL/$f"; else wget -O "$DIR/$f" "$URL/$f"; fi
done
(cd "$DIR" && sha256sum -c SHA256SUMS)
