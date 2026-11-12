"""Render writeup/draft.md from writeup/draft.template.md. Every number in the text comes from here, never typed.

    python3 -I writeup/build.py [--audit data/audit] [--verify data/verify]

Inputs (git-ignored, produced by the repo's own scripts):
    <audit>/audit.json, <audit>/paired.json        python -I audit/classify.py --raw <outputs> --out <audit> <builds>
    <verify>/<mode>/<build>/<cat>/score/...         .venv/bin/python patch/verify.py --raw <outputs> --out <verify> ...
    <verify>/default.txt                            the printed output of the default verify.py run
    <audit>/imatrix.json                            audit/imatrix_facts.py on the public, W and other builds
Placeholders are {{name}}; an unknown name is an error, so the text cannot silently drift from the data.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC"
SUB = {"simple_python": "non_live", "live_simple": "live", "multi_turn_base": "multi_turn"}
IQ3 = ["IQ3_XXS", "IQ3_XXS_R", "IQ3_XXS_G", "IQ3_XXS_T", "IQ3_XXS_X", "IQ3_XXS_W", "IQ3_XXS_V"]


def score_head(verify, mode, build, cat):
    p = os.path.join(verify, mode, build, cat, "score", MODEL_DIR, SUB[cat], f"BFCL_v4_{cat}_score.json")
    with open(p) as f:
        return json.loads(f.readline())


def pct(x):
    return f"{100 * x:.2f}"


def numbers(audit_dir, verify):
    A = json.load(open(os.path.join(audit_dir, "audit.json")))
    PR = json.load(open(os.path.join(audit_dir, "paired.json")))
    n = {}

    def off(b, c):
        return A[b][c]["official"]

    # official and tolerant single-turn scores
    for b in ["Q8_0"] + IQ3:
        k = b.lower().replace("iq3_xxs_", "").replace("iq3_xxs", "iq3")
        for c, short in (("simple_python", "sp"), ("live_simple", "ls")):
            n[f"{k}_{short}_off"] = pct(off(b, c)["accuracy"])
            n[f"{k}_{short}_tol"] = pct(score_head(verify, "tolerant", b, c)["accuracy"])
    tol_sp = [score_head(verify, "tolerant", b, "simple_python")["accuracy"] for b in IQ3]
    n["iq3_sp_tol_min"], n["iq3_sp_tol_max"] = pct(min(tol_sp)), pct(max(tol_sp))
    n["iq3_sp_parser_points"] = f'{100 * (score_head(verify, "tolerant", "IQ3_XXS", "simple_python")["accuracy"] - off("IQ3_XXS", "simple_python")["accuracy"]):.1f}'
    n["iq3_sp_total_points"] = f'{100 * (off("Q8_0", "simple_python")["accuracy"] - off("IQ3_XXS", "simple_python")["accuracy"]):.1f}'

    # single-turn classes
    sp = A["IQ3_XXS"]["simple_python"]
    n["sp_items"] = sp["items"]
    n["iq3_sp_unclosed"] = sp["classes"].get("unclosed", 0)
    n["iq3_sp_recoverable"] = sp["flags"].get("recoverable", 0)
    n["iq3_sp_wrong_count"] = sum(v for k, v in sp["classes_by_official_outcome"].get(
        "simple_function_checker:wrong_count", {}).items())
    for b in IQ3 + ["Q8_0"]:
        k = b.lower().replace("iq3_xxs_", "").replace("iq3_xxs", "iq3")
        n[f"{k}_sp_unclosed"] = A[b]["simple_python"]["classes"].get("unclosed", 0)
    n["ls_items"] = A["IQ3_XXS"]["live_simple"]["items"]
    n["iq3_ls_unclosed"] = A["IQ3_XXS"]["live_simple"]["classes"].get("unclosed", 0)

    # multi-turn
    malformed = ["untagged_call", "unclosed", "bad_newlines", "keys_dropped", "schema_echo", "bad_json", "no_name",
                 "bad_name", "bad_args", "not_object"]
    for b in ["Q8_0", "Q4_K_M", "IQ3_XXS", "IQ2_M", "IQ3_XXS_G", "IQ3_XXS_T", "IQ3_XXS_R"]:
        k = b.lower().replace("iq3_xxs_", "").replace("iq3_xxs", "iq3")
        m = A[b]["multi_turn_base"]
        n[f"{k}_mt_correct"] = m["official"]["correct_count"]
        n[f"{k}_mt_steps"] = m["steps"]
        n[f"{k}_mt_malformed"] = sum(m["classes"].get(c, 0) for c in malformed)
        n[f"{k}_mt_keys_dropped"] = m["classes"].get("keys_dropped", 0)
        n[f"{k}_mt_unclosed"] = m["classes"].get("unclosed", 0)
        n[f"{k}_mt_untagged"] = m["classes"].get("untagged_call", 0)
        n[f"{k}_mt_schema_echo"] = m["classes"].get("schema_echo", 0)
        n[f"{k}_mt_tasks_malformed"] = m["tasks_with_malformed_step"]
        n[f"{k}_mt_crashes"] = len(m["crashed_tasks"])
        n[f"{k}_mt_crash_args"] = sum("'arguments'" in c["error"] for c in m["crashed_tasks"])
        n[f"{k}_mt_crash_name"] = sum("'name'" in c["error"] for c in m["crashed_tasks"])
        if b in PR:
            n[f"{k}_mt_new_fail"] = PR[b]["new_failures"]
            n[f"{k}_mt_new_fail_broken"] = PR[b]["new_failures_with_broken_call"]
    n["mt_tasks"] = A["Q8_0"]["multi_turn_base"]["tasks"]

    # the IQ3_XXS multi-turn failures that 8-bit passes and that hold no broken call: how do they fail?
    m = A["IQ3_XXS"]["multi_turn_base"]
    clean = (set(A["Q8_0"]["multi_turn_base"]["passed_ids"]) & set(m["failed_ids"])) - set(m["malformed_or_crash_ids"])
    et = [m["failed_error_types"][i] for i in clean]
    n["iq3_mt_clean_fail"] = len(clean)
    n["iq3_mt_clean_empty"] = sum(e.endswith("empty_turn_model_response") for e in et)
    n["iq3_mt_clean_state"] = sum(e.endswith("instance_state_mismatch") for e in et)
    n["iq3_mt_clean_exec"] = sum(e.endswith("execution_response_mismatch") for e in et)
    assert n["iq3_mt_clean_empty"] + n["iq3_mt_clean_state"] + n["iq3_mt_clean_exec"] == len(clean), set(et)

    # imatrix sizes from the GGUF headers (audit/imatrix_facts.py), both at chunk size 512
    im = json.load(open(os.path.join(audit_dir, "imatrix.json")))
    n["imatrix_public_chunks"] = im["public"]["chunks_count"]
    n["imatrix_w_chunks"] = im["W"]["chunks_count"]
    n["imatrix_ratio"] = f'{im["public"]["chunks_count"] / im["W"]["chunks_count"]:.1f}'

    # default re-scoring check: lines of patch/verify.py's default run, saved to <verify>/default.txt
    lines = [l for l in open(os.path.join(verify, "default.txt")) if l[:4] in ("ok  ", "FAIL")]
    n["verify_combos"] = len(lines)
    n["verify_identical"] = sum(l.startswith("ok") for l in lines)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/audit")
    ap.add_argument("--verify", default="data/verify")
    a = ap.parse_args()
    n = numbers(a.audit, a.verify)
    t = open(os.path.join(HERE, "draft.template.md")).read()
    used = set(re.findall(r"\{\{(\w+)\}\}", t))
    unknown = used - set(n)
    if unknown:
        sys.exit(f"unknown placeholders: {sorted(unknown)}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(n[m.group(1)]), t)
    header = "<!-- Generated by writeup/build.py from draft.template.md. Edit the template, not this file. -->\n\n"
    with open(os.path.join(HERE, "draft.md"), "w") as f:
        f.write(header + out)
    print(f"wrote writeup/draft.md ({len(out.split())} words, {len(used)} numbers from data)")


if __name__ == "__main__":
    main()
