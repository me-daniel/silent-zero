"""Read the imatrix metadata that llama-quantize stores in a GGUF header, for the builds given.

    PYTHONPATH=<llama.cpp>/gguf-py .venv/bin/python audit/imatrix_facts.py --out data/audit/imatrix.json \
        public=<public IQ3_XXS.gguf> W=<own IQ3_XXS_W.gguf> ...

Needs the `gguf` package (llama.cpp's gguf-py, or `pip install gguf`). Writes {label: {key: value}} for the
quantize.imatrix.* keys, plus "tensor_types_sha256", a hash of every tensor's name and quant type. The header stores the chunk count but not the chunk size.

With --calib FILE --tokenizer tokenizer.json, also writes "calibration_file": the file's token count and
the chunk count llama-imatrix would get from it at several chunk sizes (tokens // size), to compare with
the chunk counts in the headers.
"""
import argparse
import hashlib
import os
import json

import gguf


def imatrix_meta(path):
    r = gguf.GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        if k.startswith("quantize.imatrix."):
            v = f.contents()
            out[k.removeprefix("quantize.imatrix.")] = v
    # the type of every tensor, so builds can be compared on what the quantizer chose for each weight
    types = sorted((t.name, t.tensor_type.name) for t in r.tensors)
    out["tensor_types_sha256"] = hashlib.sha256(json.dumps(types).encode()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", help="calibration text file to count")
    ap.add_argument("--tokenizer", help="tokenizer.json of the model (needs the tokenizers package)")
    ap.add_argument("builds", nargs="+", help="label=path.gguf")
    a = ap.parse_args()
    res = {}
    if a.calib:
        from tokenizers import Tokenizer
        raw = open(a.calib, "rb").read()
        n = len(Tokenizer.from_file(a.tokenizer).encode(raw.decode("utf-8"), add_special_tokens=False).ids)
        res["calibration_file"] = {"name": os.path.basename(a.calib), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "tokens": n, "chunks_at": {str(c): n // c for c in (128, 256, 512, 1024)}}
    for b in a.builds:
        label, path = b.split("=", 1)
        meta = imatrix_meta(path)
        meta.pop("file", None)  # local path of the imatrix on the quantizer's machine; not needed
        if isinstance(meta.get("dataset"), str):  # keep the file name only, never a local path
            meta["dataset"] = os.path.basename(meta["dataset"])
        res[label] = meta
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
