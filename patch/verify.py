"""Re-score saved BFCL results with the patched evaluator and compare with the original score files.

    .venv/bin/python patch/verify.py --raw data/<outputs> --out data/verify [--mode default|tolerant|tolerant+repair] [BUILD ...]

default mode: every score header and every failed entry (id, valid, error_type) must equal the original; the only
allowed difference is the added "tool_call_parse" field. Exit code 1 on any mismatch.
Other modes: single-turn categories only (multi-turn needs re-generation); prints the score next to the original.
No model is run; `bfcl evaluate` only parses saved outputs and executes the simulated tool calls.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC"
MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC"
CATS = {"simple_python": "non_live", "live_simple": "live", "multi_turn_base": "multi_turn"}
MODES = {"default": {}, "tolerant": {"SILENT_ZERO_TOLERANT": "1"},
         "tolerant+repair": {"SILENT_ZERO_TOLERANT": "1", "SILENT_ZERO_REPAIR_KEYS": "1"}}


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def key(e):
    et = e.get("error_type") or (e.get("error") or {}).get("error_type")
    return e["id"], e["valid"], et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="default", choices=MODES)
    ap.add_argument("builds", nargs="*")
    a = ap.parse_args()
    builds = a.builds or sorted(d for d in os.listdir(a.raw) if os.path.isdir(os.path.join(a.raw, d, "result")))
    missing = [b for b in builds if not os.path.isdir(os.path.join(a.raw, b, "result"))]
    if missing:
        sys.exit(f"no result folder for build(s): {missing}")
    bfcl = os.path.join(os.path.dirname(sys.executable), "bfcl")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SILENT_ZERO_")}
    env.update(MODES[a.mode])
    bad = 0
    for b in builds:
        for cat, sub in CATS.items():
            if a.mode != "default" and cat.startswith("multi_turn"):
                continue
            src = os.path.join(a.raw, b, "result", MODEL_DIR, sub, f"BFCL_v4_{cat}_result.json")
            if not os.path.exists(src):
                continue
            root = os.path.join(a.out, a.mode, b, cat)
            shutil.rmtree(root, ignore_errors=True)
            os.makedirs(os.path.join(root, "result", MODEL_DIR, sub))
            shutil.copy(src, os.path.join(root, "result", MODEL_DIR, sub))
            r = subprocess.run([bfcl, "evaluate", "--model", MODEL, "--test-category", cat],
                               env={**env, "BFCL_PROJECT_ROOT": os.path.abspath(root)},
                               capture_output=True, text=True)
            with open(os.path.join(root, "evaluate.log"), "w") as f:
                f.write(r.stdout + r.stderr)
            new_p = os.path.join(root, "score", MODEL_DIR, sub, f"BFCL_v4_{cat}_score.json")
            old_p = os.path.join(a.raw, b, "score", MODEL_DIR, sub, f"BFCL_v4_{cat}_score.json")
            if r.returncode or not os.path.exists(new_p):
                print(f"FAIL  {b:10s} {cat:16s} evaluate exited {r.returncode}; see {root}/evaluate.log")
                bad += 1
                continue
            if not os.path.exists(old_p):
                print(f"FAIL  {b:10s} {cat:16s} no original score file to compare with ({old_p})")
                bad += 1
                continue
            new, old = jl(new_p), jl(old_p)
            if a.mode != "default":
                print(f"{a.mode:16s} {b:10s} {cat:16s} {old[0]['correct_count']:>4}/{old[0]['total_count']} -> "
                      f"{new[0]['correct_count']:>4}/{new[0]['total_count']}  ({100 * new[0]['accuracy']:.2f}%)")
                continue
            same_head = {k: new[0][k] for k in ("accuracy", "correct_count", "total_count")} == \
                        {k: old[0][k] for k in ("accuracy", "correct_count", "total_count")}
            same_rows = [key(e) for e in new[1:]] == [key(e) for e in old[1:]]
            diag = all("tool_call_parse" in e for e in new[1:])
            extra = {k for e in new[1:] for k in e} - {k for e in old[1:] for k in e}
            ok = same_head and same_rows and diag and extra <= {"tool_call_parse"}
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'}  {b:10s} {cat:16s} {new[0]['correct_count']:>4}/{new[0]['total_count']}"
                  f"  header={'same' if same_head else 'DIFF'} rows={'same' if same_rows else 'DIFF'}"
                  f" diagnosis={'all' if diag else 'MISSING'} extra_fields={sorted(extra)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
