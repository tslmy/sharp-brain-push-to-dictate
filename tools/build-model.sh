#!/bin/sh
# Download the pinned Moonshine Tiny weights from Hugging Face and convert them
# to the moonstt int8 format.  Needs curl and python3 with numpy.
# Usage: tools/build-model.sh [out-dir]      (default: ./model)
set -e
REV=390624ed33d594443aa4aa221f5b9f283b545b5a
BASE=https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/$REV
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-model}
mkdir -p "$OUT" "$OUT/.dl"
curl -fL --retry 3 -o "$OUT/.dl/model.safetensors" "$BASE/model.safetensors"
curl -fL --retry 3 -o "$OUT/.dl/tokenizer.json" "$BASE/tokenizer.json"
python3 "$HERE/convert.py" "$OUT/.dl/model.safetensors" "$OUT/.dl/tokenizer.json" "$OUT"
rm -rf "$OUT/.dl"
cd "$OUT"
if command -v sha256sum >/dev/null; then sha256sum model.bin tok.bin; else shasum -a 256 model.bin tok.bin; fi > SHA256SUMS
cat SHA256SUMS
