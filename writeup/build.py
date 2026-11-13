"""Render writeup/draft.md from writeup/draft.template.md. Every number in the text comes from here, never typed.

    python3 -I writeup/build.py [--audit data/audit] [--verify data/verify]

Inputs (git-ignored, produced by the repo's own scripts):
    <audit>/audit.json, <audit>/paired.json        python -I audit/classify.py --raw <outputs> --out <audit> <builds>
    <verify>/<mode>/<build>/<cat>/score/...         .venv/bin/python patch/verify.py --raw <outputs> --out <verify> ...
    <verify>/default.txt                            the printed output of the default verify.py run
    <raw>/replay_<build>.jsonl                      next-token probabilities at chosen prefixes (local replay)
    <audit>/imatrix.json                            audit/imatrix_facts.py on the public, W and other builds
Placeholders are {{name}}; an unknown name is an error, so the text cannot silently drift from the data.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC"
SUB = {"simple_python": "non_live", "live_simple": "live", "multi_turn_base": "multi_turn"}
IQ3 = ["IQ3_XXS", "IQ3_XXS_R", "IQ3_XXS_G", "IQ3_XXS_T", "IQ3_XXS_X", "IQ3_XXS_W", "IQ3_XXS_V"]

# Names in the text, tables and figures. The public builds keep their file names. My builds are all IQ3_XXS and
# are named after their imatrix, in the order of section 2: each differs from an earlier one in one thing.
NAMES = {
    "Q8_0": "Q8_0", "Q4_K_M": "Q4_K_M", "IQ3_XXS": "IQ3_XXS", "IQ2_M": "IQ2_M",
    "IQ3_XXS_R": "public-imatrix",     # the public imatrix file, quantized with b10964
    "IQ3_XXS_W": "public-text",        # imatrix recomputed from the public calibration text, 512-token chunks
    "IQ3_XXS_V": "public-text-b6096",  # as public-text, imatrix computed with b6096
    "IQ3_XXS_X": "my-text",            # as public-text, my own general text
    "IQ3_XXS_G": "my-text-2048",       # as my-text, 2048-token chunks
    "IQ3_XXS_T": "my-text-tools",      # as my-text-2048, plus tool-call dialogues
}
PUBLIC = ["Q8_0", "Q4_K_M", "IQ3_XXS", "IQ2_M"]
MINE = ["IQ3_XXS_R", "IQ3_XXS_W", "IQ3_XXS_V", "IQ3_XXS_X", "IQ3_XXS_G", "IQ3_XXS_T"]
IM_KEY = {"IQ3_XXS": "public", "IQ3_XXS_R": "R", "IQ3_XXS_W": "W", "IQ3_XXS_V": "V", "IQ3_XXS_X": "X",
          "IQ3_XXS_G": "G", "IQ3_XXS_T": "T"}  # labels in data/audit/imatrix.json


def score_head(verify, mode, build, cat):
    p = os.path.join(verify, mode, build, cat, "score", MODEL_DIR, SUB[cat], f"BFCL_v4_{cat}_score.json")
    with open(p) as f:
        return json.loads(f.readline())


def failed_ids(verify, mode, build, cat):
    p = os.path.join(verify, mode, build, cat, "score", MODEL_DIR, SUB[cat], f"BFCL_v4_{cat}_score.json")
    with open(p) as f:
        next(f)
        return {json.loads(l)["id"] for l in f if l.strip()}


def one_decimal(x):
    """Round half up to one decimal (25.25 -> 25.3), the way a reader would, not float's half-even."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct(x):
    return f"{100 * x:.2f}"


def numbers(audit_dir, verify, raw):
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
            n[f"{k}_{short}_gain"] = f'{float(n[f"{k}_{short}_tol"]) - float(n[f"{k}_{short}_off"]):.2f}'
    tol_sp = [score_head(verify, "tolerant", b, "simple_python")["accuracy"] for b in IQ3]
    n["iq3_sp_tol_min"], n["iq3_sp_tol_max"] = pct(min(tol_sp)), pct(max(tol_sp))
    n["iq3_sp_parser_points"] = one_decimal(100 * (score_head(verify, "tolerant", "IQ3_XXS", "simple_python")["accuracy"] - off("IQ3_XXS", "simple_python")["accuracy"]))
    n["iq3_sp_total_points"] = one_decimal(100 * (off("Q8_0", "simple_python")["accuracy"] - off("IQ3_XXS", "simple_python")["accuracy"]))

    # single-turn classes
    sp = A["IQ3_XXS"]["simple_python"]
    n["sp_items"] = sp["items"]
    n["iq3_sp_unclosed"] = sp["classes"].get("unclosed", 0)
    n["iq3_sp_recoverable"] = sp["flags"].get("recoverable", 0)
    n["iq3_sp_one_short"] = sp["unclosed_one_token_short"]  # only "</tool_call>" (one token) missing
    assert 2 * n["iq3_sp_one_short"] > n["iq3_sp_unclosed"], "abstract: usually one token short"
    tok = json.load(open(os.path.join(HERE, "..", "vendor", "tokenizer", "Qwen3-4B-Instruct-2507", "tokenizer.json")))
    assert any(t["content"] == "</tool_call>" for t in tok["added_tokens"]), "text: </tool_call> is a single token"
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
        n[f"{k}_mt_steps"] = f'{m["steps"]:,}'
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
    n["calib_tokens"] = f'{im["calibration_file"]["tokens"]:,}'
    n["calib_chunks_512"] = im["calibration_file"]["chunks_at"]["512"]
    n["calib_chunks_128"] = im["calibration_file"]["chunks_at"]["128"]
    assert im["calibration_file"]["name"] == im["public"]["dataset"] == im["W"]["dataset"]
    assert im["W"]["chunks_count"] == im["calibration_file"]["chunks_at"]["512"]  # our run matches the count
    shared = [b for b in ("Q4_K_M", "IQ2_M", "R") if (im[b]["chunks_count"], im[b]["dataset"], im[b]["entries_count"])
              == (im["public"]["chunks_count"], im["public"]["dataset"], im["public"]["entries_count"])]
    assert shared == ["Q4_K_M", "IQ2_M", "R"], shared  # same imatrix file name and chunk count as the public IQ3_XXS
    # "all 3-bit builds give every tensor the same quantization type": they differ only in imatrix and llama.cpp
    assert len({im[IM_KEY[b]]["tensor_types_sha256"] for b in IQ3}) == 1
    assert {im[IM_KEY[b]]["dataset"] for b in ("IQ3_XXS_X", "IQ3_XXS_G")} == {im["X"]["dataset"]}  # same text
    assert im["X"]["chunks_count"] > 2 * im["G"]["chunks_count"]  # ... cut into longer chunks for my-text-2048
    assert im["T"]["chunks_count"] > im["G"]["chunks_count"]  # "which also makes the text longer"
    tok = im["calibration_file"]["tokens"]  # "no chunk size gives exactly <public chunks>" from this file
    assert all(tok // c != im["public"]["chunks_count"] for c in range(1, tok + 1))

    # single-turn: items the tolerant parser turns into correct answers
    tol_ok = score_head(verify, "tolerant", "IQ3_XXS", "simple_python")["correct_count"]
    n["iq3_sp_tol_gain_items"] = tol_ok - off("IQ3_XXS", "simple_python")["correct_count"]
    for b in IQ3:  # "it creates no new failures": tolerant failures are a subset of default failures
        assert failed_ids(verify, "tolerant", b, "simple_python") <= failed_ids(verify, "default", b, "simple_python"), b
    for b in IQ3:
        k = b.lower().replace("iq3_xxs_", "").replace("iq3_xxs", "iq3")
        n[f"{k}_ls_unclosed"] = A[b]["live_simple"]["classes"].get("unclosed", 0)

    gains = [float(pct(score_head(verify, "tolerant", b, "live_simple")["accuracy"]))
             - float(pct(off(b, "live_simple")["accuracy"])) for b in IQ3 if b not in ("IQ3_XXS", "IQ3_XXS_R")]
    n["other_ls_gain_max"] = f"{max(gains):.2f}"  # from the displayed scores, so it can be recomputed
    n["other_sp_unclosed_max"] = max(A[b]["simple_python"]["classes"].get("unclosed", 0)
                                     for b in IQ3 if b not in ("IQ3_XXS", "IQ3_XXS_R"))

    # counts that the prose spells out as words
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    n["n_builds"] = words[len(A)]
    n["n_iq3"] = words[len(IQ3)]
    n["n_own"] = words[sum(b.startswith("IQ3_XXS_") for b in A)]
    n["n_cats"] = words[len({c for b in A for c in A[b]})]

    # multi-turn: tasks that go the other way (8-bit fails, public 3-bit passes)
    q8 = A["Q8_0"]["multi_turn_base"]
    n["iq3_mt_reverse"] = len(set(q8["failed_ids"]) & set(m["passed_ids"]))

    # default re-scoring check: lines of patch/verify.py's default run, saved to <verify>/default.txt
    lines = [l for l in open(os.path.join(verify, "default.txt")) if l[:4] in ("ok  ", "FAIL")]
    n["verify_combos"] = len(lines)
    n["verify_identical"] = sum(l.startswith("ok") for l in lines)
    assert n["verify_identical"] == n["verify_combos"], "the text says the default mode reproduces scores exactly"

    # multi-turn: crashes, a noise floor, and what passes
    q8 = A["Q8_0"]["multi_turn_base"]
    q8_pass, q8_fail = set(q8["passed_ids"]), set(q8["failed_ids"])
    i2m = A["IQ2_M"]["multi_turn_base"]
    n["iq2_m_mt_crash_q8pass"] = len({c["id"] for c in i2m["crashed_tasks"]} & q8_pass)
    mt3 = [b for b in IQ3 if "multi_turn_base" in A[b]]
    crashes3 = [len(A[b]["multi_turn_base"]["crashed_tasks"]) for b in mt3]
    n["iq3_mt_crash_min"], n["iq3_mt_crash_max"] = min(crashes3), max(crashes3)
    q4 = A["Q4_K_M"]["multi_turn_base"]
    assert sum(q4["classes"].get(c, 0) for c in malformed) == 0 and not q4["crashed_tasks"]  # "no malformed call"
    n["q4_k_m_mt_new_fail"] = len(q8_pass & set(q4["failed_ids"]))
    n["q4_k_m_mt_reverse"] = len(q8_fail & set(q4["passed_ids"]))
    assert n["q4_k_m_mt_new_fail"] == PR["Q4_K_M"]["new_failures"]
    for b, d in A.items():  # "no task with a malformed step or a crash passes, for any build"
        if "multi_turn_base" in d:
            m_ = d["multi_turn_base"]
            assert not set(m_["malformed_or_crash_ids"]) & set(m_["passed_ids"]), b
    n["iq2_m_mt_unclosed"] = i2m["classes"].get("unclosed", 0)
    n["iq2_m_mt_unclosed_schema"] = i2m["unclosed_schema_like"]
    assert 2 * n["iq2_m_mt_unclosed_schema"] > n["iq2_m_mt_unclosed"], "text: most of them start like a schema copy"
    # coverage, as the text states it
    cover = {b: set(d) for b, d in A.items()}
    assert {b for b, c in cover.items() if c == {"multi_turn_base"}} == {"Q4_K_M", "IQ2_M"}
    assert {b for b, c in cover.items() if "multi_turn_base" not in c} == {"IQ3_XXS_W", "IQ3_XXS_V", "IQ3_XXS_X"}
    n["n_combos"] = sum(len(c) for c in cover.values())

    # replay: next-token probabilities after the JSON of unclosed simple_python outputs (prefix ends in "}}\n")
    EOT, CLOSE_ID = 151645, 151658  # <|im_end|> and </tool_call> in Qwen3's vocabulary
    rep_ = {b: [json.loads(l) for l in open(os.path.join(raw, f"replay_{b}.jsonl"))] for b in ("IQ3_XXS", "Q8_0")}
    # the replay took the first ten unclosed items; those cut after "}}\n" miss exactly the one token
    sp3 = [r for r in rep_["IQ3_XXS"] if r["kind"] == "sp_unclosed" and r["continuation_tail"].endswith("}}\n")]
    sp8 = [r for r in rep_["Q8_0"] if r["kind"] == "sp_unclosed" and r["continuation_tail"].endswith("}}\n")]
    assert [r["id"] for r in sp3] == [r["id"] for r in sp8]
    n["rep_all"] = sum(r["kind"] == "sp_unclosed" for r in rep_["IQ3_XXS"])
    n["rep_n"] = len(sp3)
    n["rep_eot_first"] = sum(r["top10"][0]["id"] == EOT for r in sp3)
    pc = sorted(next((math.exp(t["logprob"]) for t in r["top10"] if t["id"] == CLOSE_ID), 0.0) for r in sp3)
    n["rep_close_p_median"] = f"{(pc[len(pc) // 2 - 1] + pc[len(pc) // 2]) / 2:.2f}"
    n["rep_q8_close_first"] = sum(r["top10"][0]["id"] == CLOSE_ID for r in sp8)
    assert n["rep_eot_first"] == n["rep_q8_close_first"] == n["rep_n"], "text: 'at every one of them', 'every time'"
    an3 = [r for r in rep_["IQ3_XXS"] if r["kind"] == "mt_announce_stop"]
    an8 = [r for r in rep_["Q8_0"] if r["kind"] == "mt_announce_stop"]
    n["rep_announce_n"] = len(an3)
    n["rep_announce_eot_iq3"] = sum(r["top10"][0]["id"] == EOT for r in an3)
    n["rep_announce_eot_q8"] = sum(r["top10"][0]["id"] == EOT for r in an8)

    # further statements the text makes
    i2 = A["IQ2_M"]["multi_turn_base"]["crashed_tasks"]
    assert n["iq2_m_mt_crash_args"] + n["iq2_m_mt_crash_name"] == len(i2), "crash split must cover all crashes"
    assert n["iq3_sp_tol_gain_items"] <= n["iq3_sp_recoverable"], "gains are at most the recoverable calls"
    tol_ls = {b: score_head(verify, "tolerant", b, "live_simple")["accuracy"] for b in IQ3}
    top2 = sorted(tol_ls, key=tol_ls.get, reverse=True)[:2]
    assert set(top2) == {"IQ3_XXS", "IQ3_XXS_R"}, "text: the two public-imatrix builds score highest of the 3-bit builds on live_simple"
    assert "simple_python" not in A["Q4_K_M"], "text: Q4_K_M was run on multi-turn only"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/audit")
    ap.add_argument("--verify", default="data/verify")
    ap.add_argument("--raw", default="data/raw", help="raw outputs, for the replay_<build>.jsonl files")
    a = ap.parse_args()
    n = numbers(a.audit, a.verify, a.raw)
    t = open(os.path.join(HERE, "draft.template.md")).read()
    used = set(re.findall(r"\{\{(\w+)\}\}", t))
    unknown = used - set(n)
    if unknown:
        sys.exit(f"unknown placeholders: {sorted(unknown)}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(n[m.group(1)]), t)
    for b in MINE:  # every build of mine is named and described in the text
        assert f"`{NAMES[b]}`" in out, NAMES[b]
    assert not re.search(r"\b[PRWVXGT]\b(?![-_.])", re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", out, flags=re.S))), \
        "a single-letter build name is left in the text"
    for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", out):  # figures: run writeup/figures/fig_*.py first
        if not os.path.exists(os.path.join(HERE, img)):
            sys.exit(f"missing figure {img}: run .venv/bin/python -I writeup/figures/<script>.py")
    header = "<!-- Generated by writeup/build.py from draft.template.md. Edit the template, not this file. -->\n\n"
    with open(os.path.join(HERE, "draft.md"), "w") as f:
        f.write(header + out)
    print(f"wrote writeup/draft.md ({len(out.split())} words, {len(used)} numbers from data)")


if __name__ == "__main__":
    main()
