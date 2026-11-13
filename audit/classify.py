"""Classify every model step in saved BFCL outputs by what the Qwen FC handler does with it.

BFCL's QwenFCHandler (bfcl_eval/model_handler/local_inference/qwen_fc.py, commit 6ea5797) extracts calls with
    re.findall(r"<tool_call>\\n(.*?)\\n</tool_call>", text, re.DOTALL)
and silently skips any match that is not valid JSON. Everything it cannot read is scored as if the model had
answered in plain text. This script tells those cases apart.

Step classes (exactly one per step):
    ok            the official parser yields >= 1 call, and every yielded call is well formed: string "name",
                  object "arguments", no "function" key
    text          no "<tool_call>" and no call object in the output: a genuine non-call
    untagged_call no "<tool_call>", but a well-formed call object in plain text (the tags are missing)
    unclosed      a "<tool_call>" block is opened but never closed (generation ended first)
    bad_newlines  tags are closed, but not newline-delimited as the official regex requires
    keys_dropped  invalid JSON of the shape {"ls", "arguments": {...}} or {"ls", {...}}: the function name and the
                  arguments are there, the "name" key (and maybe the "arguments" key) is not. A near miss: the
                  intended call is unambiguous
    schema_echo   the block is a copy of the tool's schema ({"name": ..., "description": ...}), not a call
    bad_json      other invalid JSON (including a block that runs into a second "<tool_call>")
    no_name       valid JSON object without "name". Upstream raises on it (decoder failure in single-turn,
                  a crash of the whole task in multi-turn)
    bad_name      "name" is not a non-empty string, or a "function" key is present (also crashes multi-turn)
    bad_args      valid JSON object with "name" but "arguments" missing or not an object
    not_object    valid JSON that is not an object
Flags (any number per step):
    partial_drop  the official parser yields some calls but drops at least one other "<tool_call>" block
    recoverable   a tolerant parser (accepts a final unclosed block and loose whitespace) yields a well-formed
                  call where the official parser yields none. Recoverable says nothing about correctness.
Task-level (multi-turn): crash = the task has no inference log because generation raised (scored inference_error).

The classification rules are kept identical to the patched BFCL module tool_call_parsing.py (a test checks it).

Usage:
    python -I audit/classify.py --raw data/<outputs> --out data/audit [--model-dir DIR] [BUILD ...]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

OPEN, CLOSE = "<tool_call>", "</tool_call>"
OFFICIAL = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)  # upstream's pattern, unchanged
KEYS_DROPPED = re.compile(r'^\{\s*"([^"]+)"\s*,\s*(\{|"arguments"\s*:)')
SCHEMA_ECHO = re.compile(r'^\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"(description|parameters)"\s*:')
NAME_START = re.compile(r'\{\s*"name"\s*:')

CLASSES = ["ok", "text", "untagged_call", "unclosed", "bad_newlines", "keys_dropped", "schema_echo", "bad_json",
           "no_name", "bad_name", "bad_args", "not_object"]
MALFORMED = CLASSES[2:]  # everything except ok and text
CATEGORIES = {  # category -> (subfolder, file stem)
    "simple_python": ("non_live", "BFCL_v4_simple_python"),
    "live_simple": ("live", "BFCL_v4_live_simple"),
    "multi_turn_base": ("multi_turn", "BFCL_v4_multi_turn_base"),
}


def official_blocks(text):
    """Same result as OFFICIAL.findall(text), in linear time. The pattern is a literal opener, a lazy DOTALL
    group and a literal closer, so the first closer after each opener ends the match; if no closer follows an
    opener, no later opener can match either."""
    out, pos = [], 0
    while True:
        i = text.find("<tool_call>\n", pos)
        if i < 0:
            return out
        j = text.find("\n</tool_call>", i + 12)
        if j < 0:
            return out
        out.append(text[i + 12:j])
        pos = j + 13


def well_formed(o):
    """A call the chat template and the executor can use: string name, object arguments, no "function" key."""
    return (isinstance(o, dict) and isinstance(o.get("name"), str) and o["name"] != ""
            and isinstance(o.get("arguments"), dict) and "function" not in o)


def _loads(blob):
    try:
        return True, json.loads(blob)
    except Exception:
        return False, None


def loose_blocks(text):
    """Split at every opener, so no block crosses another opener. Returns (closed blocks, final unclosed block
    or None). Linear time; whitespace around each block is stripped."""
    parts = text.split(OPEN)[1:]
    closed, final = [], None
    for i, seg in enumerate(parts):
        j = seg.find(CLOSE)
        if j >= 0:
            closed.append(seg[:j].strip())
        elif i == len(parts) - 1:
            final = seg.strip()
    return closed, final


def json_issue(blob):
    """None if blob is a well-formed call, else the class name of its defect."""
    blob = blob.strip()
    ok, o = _loads(blob)
    if not ok:
        if KEYS_DROPPED.match(blob):
            return "keys_dropped"
        if SCHEMA_ECHO.match(blob):
            return "schema_echo"
        return "bad_json"
    if not isinstance(o, dict):
        return "not_object"
    if "name" not in o:
        return "no_name"
    if not isinstance(o.get("arguments"), dict):
        return "schema_echo" if ("description" in o or "parameters" in o) else "bad_args"
    if not isinstance(o["name"], str) or o["name"] == "" or "function" in o:
        return "bad_name"
    return None


def official_calls(text):
    """Exactly what QwenFCHandler._extract_tool_calls returns: every match that parses as JSON."""
    out = []
    for m in official_blocks(text):
        ok, o = _loads(m)
        if ok:
            out.append(o)
    return out


def tolerant_calls(text):
    """Well-formed calls from closed blocks with loose whitespace, plus one final unclosed block."""
    closed, final = loose_blocks(text)
    out = []
    for b in closed + ([final] if final is not None else []):
        ok, o = _loads(b)
        if ok and well_formed(o):
            out.append(o)
    return out


def untagged_call(text):
    """True if the text holds a well-formed call object outside any <tool_call> tags. Tries at most the first
    20 '{"name":' positions, so degenerate input stays cheap."""
    dec = json.JSONDecoder()
    for k, m in enumerate(NAME_START.finditer(text)):
        if k >= 20:
            return False
        try:
            o, _ = dec.raw_decode(text, m.start())
        except (ValueError, RecursionError):
            continue
        if well_formed(o):
            return True
    return False


def classify(text):
    """Return (class, flags) for one assistant output."""
    text = text if isinstance(text, str) else json.dumps(text)
    flags = []
    n_open = text.count(OPEN)
    if n_open == 0:
        return ("untagged_call" if untagged_call(text) else "text"), flags
    calls = official_calls(text)
    if calls and all(well_formed(c) for c in calls):
        if len(calls) < n_open:
            flags.append("partial_drop")
        return "ok", flags
    if not calls and tolerant_calls(text):
        flags.append("recoverable")
    # find the defect of the first block the official parser could not use
    for c in calls:  # parsed but not well formed
        issue = json_issue(json.dumps(c))
        if issue:
            return issue, flags
    for m in official_blocks(text):  # matched but invalid JSON
        issue = json_issue(m)
        if issue in ("bad_json", "keys_dropped", "schema_echo"):
            return issue, flags
    if n_open > text.count(CLOSE):
        return "unclosed", flags
    closed, _ = loose_blocks(text)
    for b in closed:  # closed, but the official regex did not match it
        issue = json_issue(b)
        return ("bad_newlines" if issue is None else issue), flags
    return "bad_json", flags  # tags present in an order no rule matches


def one_token_short(text):
    """True if appending only "</tool_call>" (one token in Qwen3's vocabulary) makes the official parser yield a
    well-formed call for every opened block: the model wrote the call and its final newline, then stopped."""
    if not isinstance(text, str):
        return False
    calls = official_calls(text + CLOSE)
    return len(calls) == text.count(OPEN) and all(well_formed(c) for c in calls)


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def score_info(raw, build, model_dir, cat):
    sub, stem = CATEGORIES[cat]
    p = os.path.join(raw, build, "score", model_dir, sub, f"{stem}_score.json")
    if not os.path.exists(p):
        return None, {}
    L = jl(p)
    return L[0], {r["id"]: r for r in L[1:]}


def audit_single(raw, build, model_dir, cat):
    sub, stem = CATEGORIES[cat]
    rows = jl(os.path.join(raw, build, "result", model_dir, sub, f"{stem}_result.json"))
    head, failed = score_info(raw, build, model_dir, cat)
    cls, flg, by_err = Counter(), Counter(), defaultdict(Counter)
    one_short = 0
    for r in rows:
        c, fl = classify(r["result"])
        cls[c] += 1
        flg.update(fl)
        one_short += c == "unclosed" and one_token_short(r["result"])
        f = failed.get(r["id"])
        by_err[f["error_type"] if f else "pass"][c] += 1
    return {"items": len(rows), "classes": dict(cls), "flags": dict(flg), "unclosed_one_token_short": one_short,
            "official": head, "classes_by_official_outcome": {k: dict(v) for k, v in by_err.items()}}


def turns_of(row):
    return [t for t in row.get("inference_log") or [] if isinstance(t, dict)]


def handler_status(events):
    """What BFCL's own log says about a step. Note: the handler logs "Successfully decoded" even when the decoded
    list is empty, i.e. when every <tool_call> block was silently dropped; that case is "decoded_empty" here."""
    for e in events:
        if e.get("role") != "handler_log":
            continue
        c = e.get("content", "")
        if c.startswith("Successfully"):
            dec = e.get("model_response_decoded")
            return "decoded" if dec and any(dec) else "decoded_empty"
        if c.startswith("Error decoding") or c.startswith("Failed to decode"):
            return "decode_error"
    return "no_log"


def audit_multi(raw, build, model_dir):
    sub, stem = CATEGORIES["multi_turn_base"]
    rows = jl(os.path.join(raw, build, "result", model_dir, sub, f"{stem}_result.json"))
    head, failed = score_info(raw, build, model_dir, "multi_turn_base")
    cls, flg, status_x_class = Counter(), Counter(), Counter()
    crashes, tasks_malformed, failed_with_malformed = [], set(), set()
    unclosed_schema = 0
    for r in rows:
        tid = r["id"]
        turns = turns_of(r)
        if not turns:
            crashes.append({"id": tid, "error": str(r.get("result"))[:200]})
            continue
        for turn in turns:
            for k in sorted((k for k in turn if k.startswith("step_")), key=lambda k: int(k.split("_")[1])):
                ev = turn[k]
                a = next((x.get("content", "") for x in ev if x.get("role") == "assistant"), "")
                c, fl = classify(a)
                if c == "unclosed" and SCHEMA_ECHO.match(loose_blocks(a)[1] or ""):
                    unclosed_schema += 1  # an unclosed block that starts like a copy of the tool schema
                cls[c] += 1
                flg.update(fl)
                status_x_class[f"{handler_status(ev)}|{c}"] += 1
                if c in MALFORMED or "partial_drop" in fl:
                    tasks_malformed.add(tid)
                    if tid in failed:
                        failed_with_malformed.add(tid)
    err = Counter(f["error"]["error_type"] for f in failed.values()) if failed else Counter()
    return {"tasks": len(rows), "steps": sum(cls.values()), "classes": dict(cls), "flags": dict(flg),
            "unclosed_schema_like": unclosed_schema,
            "failed_ids": sorted(failed), "passed_ids": sorted({r["id"] for r in rows} - set(failed)),
            "failed_error_types": {i: f["error"]["error_type"] for i, f in sorted(failed.items())},
            "malformed_or_crash_ids": sorted(tasks_malformed | {c["id"] for c in crashes}),
            "official": head, "official_error_types": dict(err),
            "crashed_tasks": crashes, "tasks_with_malformed_step": len(tasks_malformed),
            "failed_tasks_with_malformed_step": len(failed_with_malformed),
            "handler_status_x_class": dict(status_x_class)}


def paired(report, ref):
    """Tasks the reference build passes and another build fails, and how many of them hold a broken call."""
    if ref not in report or "multi_turn_base" not in report[ref]:
        return {}
    ref_pass = set(report[ref]["multi_turn_base"]["passed_ids"])
    out = {}
    for b, d in report.items():
        m = d.get("multi_turn_base")
        if b == ref or not m:
            continue
        new = ref_pass & set(m["failed_ids"])
        bad = set(m["malformed_or_crash_ids"])
        out[b] = {"new_failures": len(new), "new_failures_with_broken_call": len(new & bad),
                  "all_failures": len(m["failed_ids"]), "all_failures_with_broken_call": len(set(m["failed_ids"]) & bad)}
    return out


def md_paired(pr, ref):
    if not pr:
        return ""
    lines = [f"### multi_turn_base: failures relative to {ref} (tasks)", "",
             f"| build | {ref} pass, build fail | of these with a broken call or crash | all failures | of these with a broken call or crash |",
             "|---|---|---|---|---|"]
    for b, v in pr.items():
        lines.append(f'| {b} | {v["new_failures"]} | {v["new_failures_with_broken_call"]} | {v["all_failures"]} | {v["all_failures_with_broken_call"]} |')
    return "\n".join(lines) + "\n"


def md_tables(report):
    lines = []
    for cat in CATEGORIES:
        rows = [(b, d[cat]) for b, d in report.items() if cat in d]
        if not rows:
            continue
        unit = "steps" if cat == "multi_turn_base" else "items"
        lines += [f"### {cat} ({unit})", "",
                  "| build | score | " + " | ".join(CLASSES) + " | recoverable | partial_drop |"
                  + (" crashed tasks | tasks w/ malformed | failed w/ malformed |" if cat == "multi_turn_base" else ""),
                  "|" + "---|" * (len(CLASSES) + 4 + (3 if cat == "multi_turn_base" else 0))]
        for b, d in rows:
            o = d["official"] or {}
            sc = f'{o.get("correct_count")}/{o.get("total_count")}' if o else "?"
            cells = [str(d["classes"].get(c, 0)) for c in CLASSES]
            extra = ""
            if cat == "multi_turn_base":
                extra = f' {len(d["crashed_tasks"])} | {d["tasks_with_malformed_step"]} | {d["failed_tasks_with_malformed_step"]} |'
            lines.append(f"| {b} | {sc} | " + " | ".join(cells)
                         + f' | {d["flags"].get("recoverable", 0)} | {d["flags"].get("partial_drop", 0)} |' + extra)
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True, help="folder with one subfolder per build (result/, score/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default="Qwen_Qwen3-4B-Instruct-2507-FC")
    ap.add_argument("--reference", default="Q8_0", help="build to compare multi-turn failures against")
    ap.add_argument("builds", nargs="*")
    a = ap.parse_args()
    builds = a.builds or sorted(d for d in os.listdir(a.raw) if os.path.isdir(os.path.join(a.raw, d, "result")))
    report = {}
    for b in builds:
        report[b] = {}
        for cat, (sub, stem) in CATEGORIES.items():
            if not os.path.exists(os.path.join(a.raw, b, "result", a.model_dir, sub, f"{stem}_result.json")):
                continue
            report[b][cat] = audit_multi(a.raw, b, a.model_dir) if cat == "multi_turn_base" \
                else audit_single(a.raw, b, a.model_dir, cat)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "audit.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    pr = paired(report, a.reference)
    with open(os.path.join(a.out, "paired.json"), "w") as f:
        json.dump(pr, f, indent=1)
    md = md_tables(report) + "\n" + md_paired(pr, a.reference)
    with open(os.path.join(a.out, "audit.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
