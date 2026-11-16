#!/usr/bin/env python3
"""Convert Moonshine-tiny (HF safetensors) to the moonstt int8 format.

Usage: convert.py <model.safetensors> <tokenizer.json> <out-dir>
Writes <out-dir>/model.bin and <out-dir>/tok.bin.

Linear weights are quantised per output row to int8 (symmetric), stored
offset-binary (+128) and packed so that one 32-bit word holds four weights
k..k+3 as bytes [k, k+2, k+1, k+3].  `x & 0x00ff00ff` and `(x >> 8) &
0x00ff00ff` then yield two pairs of 16-bit lanes that feed SMLABB/SMLATT
directly on ARMv5TE (no unpack instructions needed).

Only the Python standard library plus numpy/safetensors-free parsing is used
for the safetensors container, so this runs anywhere with numpy.
"""
import json, struct, sys, os
import numpy as np

MAGIC = b"MSTT0001"
D, FF, NH, NL, VOCAB = 288, 1152, 8, 6, 32000


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            assert v["dtype"] == "F32", (k, v["dtype"])
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype="<f4").reshape(v["shape"]).copy()
        return out


class Out:
    def __init__(self):
        self.buf = bytearray()

    def align(self, n=32):
        while len(self.buf) % n:
            self.buf.append(0)

    def f32(self, a):
        self.align()
        self.buf += np.ascontiguousarray(a, dtype="<f4").tobytes()

    def lin(self, w, bias=None):
        """w: [out][in] float.  Layout: sw[out] f32, bias[out] f32, packed."""
        o, i = w.shape
        assert i % 4 == 0
        s = np.abs(w).max(axis=1) / 127.0
        s[s == 0] = 1.0
        q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int32) + 128
        g = q.reshape(o, i // 4, 4)
        p = np.stack([g[..., 0], g[..., 2], g[..., 1], g[..., 3]], axis=-1).astype(np.uint8)
        self.f32(s)
        self.f32(bias if bias is not None else np.zeros(o, dtype=np.float32))
        self.align()
        self.buf += p.tobytes()


def main():
    st, tokjson, outdir = sys.argv[1:4]
    W = load_safetensors(st)
    o = Out()
    o.buf += MAGIC
    o.buf += struct.pack("<8i", D, FF, NH, NL, NL, VOCAB, 0, 0)
    e = "model.encoder."
    # conv1: [288,1,127] -> [288][128]
    c1 = np.zeros((D, 128), dtype=np.float32)
    c1[:, :127] = W[e + "conv1.weight"][:, 0, :]
    o.lin(c1)
    # conv2: [576,288,7] -> [576][k*288+c]
    o.lin(W[e + "conv2.weight"].transpose(0, 2, 1).reshape(2 * D, 7 * D), W[e + "conv2.bias"])
    o.lin(W[e + "conv3.weight"].transpose(0, 2, 1).reshape(D, 3 * 2 * D), W[e + "conv3.bias"])
    o.f32(W[e + "groupnorm.weight"])
    o.f32(W[e + "groupnorm.bias"])
    for l in range(NL):
        p = f"{e}layers.{l}."
        o.f32(W[p + "input_layernorm.weight"])
        for n in ("q", "k", "v", "o"):
            o.lin(W[p + f"self_attn.{n}_proj.weight"])
        o.f32(W[p + "post_attention_layernorm.weight"])
        o.lin(W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"])
        o.lin(W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"])
    o.f32(W[e + "layer_norm.weight"])
    d = "model.decoder."
    for l in range(NL):
        p = f"{d}layers.{l}."
        o.f32(W[p + "input_layernorm.weight"])
        for n in ("q", "k", "v", "o"):
            o.lin(W[p + f"self_attn.{n}_proj.weight"])
        o.f32(W[p + "post_attention_layernorm.weight"])
        for n in ("q", "k", "v", "o"):
            o.lin(W[p + f"encoder_attn.{n}_proj.weight"])
        o.f32(W[p + "final_layernorm.weight"])
        o.lin(W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"])
        o.lin(W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"])
    o.f32(W[d + "norm.weight"])
    # tied embedding / LM head, first VOCAB rows (ids >= 32000 are specials never emitted)
    o.lin(W[d + "embed_tokens.weight"][:VOCAB])
    o.align()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "model.bin"), "wb") as f:
        f.write(o.buf)
    print("model.bin", len(o.buf) / 1e6, "MB")

    t = json.load(open(tokjson))
    vocab = sorted(t["model"]["vocab"].items(), key=lambda kv: kv[1])
    strs = []
    for s, i in vocab[:VOCAB]:
        if s.startswith("<0x") and s.endswith(">") and len(s) == 6:
            b = bytes([int(s[3:5], 16)])
        elif i < 3:
            b = b""
        else:
            b = s.replace("▁", " ").encode()
        strs.append(b)
    assert len(strs) == VOCAB
    offs, blob = [], bytearray()
    for b in strs:
        offs.append(len(blob))
        blob += b
    offs.append(len(blob))
    with open(os.path.join(outdir, "tok.bin"), "wb") as f:
        f.write(b"MTOK0001" + struct.pack("<i", VOCAB) + struct.pack(f"<{VOCAB + 1}I", *offs) + blob)
    print("tok.bin", len(blob))


if __name__ == "__main__":
    main()
