"""Write ref.json with Hugging Face (float32) transcripts for every wav in a directory.

usage: mkref.py <hf-model-dir-or-id> <clips-dir>      (needs torch, transformers, soundfile)
"""
import glob, json, os, sys
import soundfile as sf
from transformers import AutoProcessor, MoonshineForConditionalGeneration

mid, cdir = sys.argv[1:3]
m = MoonshineForConditionalGeneration.from_pretrained(mid).eval()
pr = AutoProcessor.from_pretrained(mid)
ref = {}
for f in sorted(glob.glob(os.path.join(cdir, "*.wav"))):
    a, sr = sf.read(f, dtype="float32")
    out = m.generate(**pr(a, sampling_rate=sr, return_tensors="pt"), max_new_tokens=int(6.5 * len(a) / sr))
    ref[os.path.basename(f)[:-4]] = pr.decode(out[0], skip_special_tokens=True)
json.dump(ref, open(os.path.join(cdir, "ref.json"), "w"), indent=1)
print(json.dumps(ref, indent=1))
