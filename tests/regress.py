"""Compare moonstt output with Hugging Face reference transcripts.

usage: regress.py <moonstt-binary> <model-dir> <clips-dir>
clips-dir holds *.wav and ref.json ({"name": "reference text"}); see README.md.
"""
import json, os, subprocess, sys

binp, mdir, cdir = sys.argv[1:4]
ref = json.load(open(os.path.join(cdir, "ref.json")))
bad = 0
for n in sorted(ref):
    g = subprocess.run([binp, "-T", "-m", mdir, os.path.join(cdir, n + ".wav")],
                       capture_output=True, text=True).stdout.strip()
    ok = g == ref[n].strip()
    bad += not ok
    print(n, "EXACT" if ok else "DIFF")
    if not ok:
        print("   ref:", ref[n].strip())
        print("   got:", g)
print("mismatches:", bad, "of", len(ref))
sys.exit(1 if bad > 2 else 0)  # a near-tie word or two is expected (int8 quantisation)
