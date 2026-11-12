"""Read the imatrix metadata that llama-quantize stores in a GGUF header, for the builds given.

    PYTHONPATH=<llama.cpp>/gguf-py .venv/bin/python audit/imatrix_facts.py --out data/audit/imatrix.json \
        public=<public IQ3_XXS.gguf> W=<own IQ3_XXS_W.gguf> ...

Needs the `gguf` package (llama.cpp's gguf-py, or `pip install gguf`). Writes {label: {key: value}} for the
quantize.imatrix.* keys; chunks_count x chunk size is the number of calibration tokens the imatrix saw.
"""
import argparse
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("builds", nargs="+", help="label=path.gguf")
    a = ap.parse_args()
    res = {}
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
