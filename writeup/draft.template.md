# Silent Zero: a 3-bit model drops one token, and the benchmark scores it as silence

*Draft. All numbers are filled in by `writeup/build.py` from the audit and re-scoring outputs.*

**In short**

- On BFCL's `simple_python`, a public IQ3_XXS build (about 3 bits per weight) of Qwen3-4B-Instruct-2507 scores {{iq3_sp_off}}%, against {{q8_0_sp_off}}% for the 8-bit build. {{iq3_sp_parser_points}} of those {{iq3_sp_total_points}} points come from one thing: the model writes a complete call and stops before the closing `</tool_call>`. BFCL's parser then drops the call without a trace and scores the item as if the model had answered in prose.
- Whether a 3-bit build leaves out that tag depends on its importance matrix (imatrix). Builds I made from the same weights with other imatrices rarely do it on the two single-turn categories. Once the parser tolerates the missing tag, all seven 3-bit builds score {{iq3_sp_tol_min}}–{{iq3_sp_tol_max}}% on `simple_python`.
- Multi-turn tasks are different. Of the {{iq3_mt_new_fail}} tasks that the 8-bit build passes and the 3-bit build fails, {{iq3_mt_new_fail_broken}} contain a malformed call. The other {{iq3_mt_clean_fail}} contain none, so a more tolerant parser couldn't rescue them.

**Terms.** A *build* is one GGUF file of the model. Low-bit builds are usually made with an *imatrix*: per-weight importance statistics that llama.cpp collects by running the full-precision model over a calibration text, cut into *chunks* of a fixed number of tokens. In multi-turn tasks, a *task* has several user *turns*, and each turn has one or more model *steps* (one generation each).

## 1. The parser

BFCL's handler for Qwen-style models extracts tool calls like this (`qwen_fc.py` at commit `6ea5797`, the latest `main` as of November 2026):

```python
pattern = r"<tool_call>\n(.*?)\n</tool_call>"
matches = re.findall(pattern, input_string, re.DOTALL)

# Process matches into a list of dictionaries
result = []
for match in matches:
    try:
        match = json.loads(match)
        result.append(match)
    except Exception as e:
        pass
return result
```

Anything that doesn't match is gone: an unclosed tag, missing newlines around the JSON, invalid JSON. Single-turn items then fail as `wrong_count` ("Wrong number of functions"), exactly like a model that called nothing. In multi-turn tasks the turn simply ends, and the handler's own log says "Successfully decoded model response" with an empty list.

There's a second path. A call that *is* valid JSON but lacks a usable `"name"` or `"arguments"` key is kept and written into the chat history. When the next prompt is built, the template indexes `tool_call["name"]` or `tool_call["arguments"]` and raises. The whole multi-turn task is then recorded as `inference_error`, including turns the model might have passed.

## 2. What the 3-bit model actually writes

I classified every saved model output by what the parser does with it. The builds are listed in section 3; all were served with llama.cpp and greedy decoding.

**Single-turn** (`simple_python`, {{sp_items}} items). The public IQ3_XXS build leaves {{iq3_sp_unclosed}} calls unclosed. {{iq3_sp_recoverable}} of them are complete, parseable calls apart from the missing tag, so the model stopped after the JSON, not in the middle of it. The 8-bit build, on the same server and settings, leaves {{q8_0_sp_unclosed}} unclosed. On `live_simple` ({{ls_items}} items), the 3-bit build leaves {{iq3_ls_unclosed}} unclosed.

A typical output, scored as "no function call":

```
<tool_call>
{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}
```

**Multi-turn** (`multi_turn_base`, {{mt_tasks}} tasks). The 8-bit build produces {{q8_0_mt_malformed}} malformed calls in {{q8_0_mt_steps}} steps. The public 3-bit build produces {{iq3_mt_malformed}}, spread over {{iq3_mt_tasks_malformed}} tasks. The largest group is a different near miss: the key names go missing ({{iq3_mt_keys_dropped}} steps).

```
<tool_call>
{"cd", "arguments": {"folder": "document"}}
</tool_call>
```

The intended call is clear, but it isn't JSON. Another {{iq3_mt_untagged}} steps are a complete call with no tags at all: the whole output is `{"name": "ls", "arguments": {...}}`, which BFCL reads as plain text. A build calibrated on tool-call dialogues (T) does this more often ({{t_mt_untagged}} steps). Calls with the function name as the key, `{"ls": {...}}`, also appear.

At 2 bits (IQ2_M), the model also starts copying the tool's schema instead of calling it (at least {{iq2_m_mt_schema_echo}} steps; more of the unclosed steps look like schema copies too). It loses {{iq2_m_mt_crashes}} whole tasks to the crash path: {{iq2_m_mt_crash_args}} on a call with a `"name"` but no `"arguments"` (which is what a copied schema looks like), {{iq2_m_mt_crash_name}} on a missing `"name"`. A crashed task keeps no log, so the output that crashed it isn't recorded; the shapes are inferred from the error messages.

## 3. Official vs tolerant scoring

I re-scored the same saved outputs with a patched evaluator that accepts a final unclosed `<tool_call>` and loose whitespace around the JSON. Nothing else changes: names, argument types and values are still checked by BFCL's own checker.

All builds come from the same 16-bit weights of Qwen3-4B-Instruct-2507. Q8_0 and P are public files (`bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, revision `ae44f08`); the others I quantized myself. "Unclosed" counts `simple_python` items.

| Build | What it is | Unclosed | simple_python official | tolerant | live_simple official | tolerant |
|---|---|---|---|---|---|---|
| Q8_0 | public 8-bit | {{q8_0_sp_unclosed}} | {{q8_0_sp_off}} | {{q8_0_sp_tol}} | {{q8_0_ls_off}} | {{q8_0_ls_tol}} |
| P | public IQ3_XXS | {{iq3_sp_unclosed}} | {{iq3_sp_off}} | {{iq3_sp_tol}} | {{iq3_ls_off}} | {{iq3_ls_tol}} |
| R | P's imatrix, re-quantized with a newer llama.cpp | {{r_sp_unclosed}} | {{r_sp_off}} | {{r_sp_tol}} | {{r_ls_off}} | {{r_ls_tol}} |
| W | imatrix from the calibration file P's header names, chunks of 512 | {{w_sp_unclosed}} | {{w_sp_off}} | {{w_sp_tol}} | {{w_ls_off}} | {{w_ls_tol}} |
| V | as W, imatrix computed with llama.cpp b6096 | {{v_sp_unclosed}} | {{v_sp_off}} | {{v_sp_tol}} | {{v_ls_off}} | {{v_ls_tol}} |
| X | imatrix from my own general text, chunks of 512 | {{x_sp_unclosed}} | {{x_sp_off}} | {{x_sp_tol}} | {{x_ls_off}} | {{x_ls_tol}} |
| G | as X, chunks of 2048 | {{g_sp_unclosed}} | {{g_sp_off}} | {{g_sp_tol}} | {{g_ls_off}} | {{g_ls_tol}} |
| T | as G, plus tool-call dialogues | {{t_sp_unclosed}} | {{t_sp_off}} | {{t_sp_tol}} | {{t_ls_off}} | {{t_ls_tol}} |

For P on `simple_python`, the tolerant parser turns {{iq3_sp_tol_gain_items}} items into correct answers; it creates no new failures. With all patch options off, the patched evaluator reproduces the official scores and failed entries exactly ({{verify_identical}} of {{verify_combos}} build × category combinations).

## 4. The tag depends on the imatrix

- **P's imatrix:** P leaves {{iq3_sp_unclosed}} tags unclosed. R uses the same imatrix with a newer quantizer and leaves {{r_sp_unclosed}}. So the quantizer version isn't the cause; the imatrix is.
- **The calibration file P's header names** (`calibration_datav3.txt`, the revision I downloaded), run through llama-imatrix myself: {{w_sp_unclosed}} unclosed tag with the newer llama.cpp (W) and {{v_sp_unclosed}} with b6096 (V), the release the public model card names.
- **My own general text:** {{x_sp_unclosed}} at chunks of 512 (X), {{g_sp_unclosed}} at 2048 (G).

So I couldn't reproduce the failure from the published calibration file, and I couldn't reproduce P's imatrix either. P's header records {{imatrix_public_chunks}} chunks. The file I used is {{calib_tokens}} tokens long: {{calib_chunks_512}} chunks at 512 tokens (what W's header records), {{calib_chunks_128}} at 128. No chunk size gives exactly {{imatrix_public_chunks}} from this file, and the header doesn't store the chunk size. P's imatrix was made from different text, with different settings, or both. I haven't tested other chunk sizes or options on the published file, or other revisions of it. The whole effect also rests on one imatrix file.

Low bits are part of it too. Q4_K_M, a public 4-bit build whose header records the same imatrix as P's (same file name, same chunk count), produces {{q4_k_m_mt_malformed}} malformed steps in multi-turn.

With the tag forgiven, P and R are as good as the other 3-bit builds on `simple_python`. On `live_simple` they score highest of the 3-bit builds, but the gaps are a handful of items with one greedy run each, so I don't read much into the order. On these two categories, most of what separated P from my builds was the closing tag, not the choice of call.

The effect also depends on the task. G leaves {{g_sp_unclosed}} tags unclosed on `simple_python`, but {{g_mt_unclosed}} in multi-turn steps. P does the reverse: {{iq3_sp_unclosed}} on `simple_python`, {{iq3_mt_unclosed}} in multi-turn. A build that looks clean on one category isn't necessarily clean on another. W, V and X were only run on the single-turn categories.

## 5. What the parser does not explain

In `multi_turn_base`, the 8-bit build passes {{q8_0_mt_correct}} of {{mt_tasks}} tasks and P passes {{iq3_mt_correct}}. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and P fails, {{iq3_mt_new_fail_broken}} contain a malformed call or a crash. That doesn't mean the broken call caused the failure, only that it could have. The other {{iq3_mt_clean_fail}} contain no malformed call:

- In {{iq3_mt_clean_empty}}, a turn ends without any call, typically right after the model announces what it is about to do ("Let me do that for you."). That looks like generation ending too early, a relative of the missing tag, but there is no call for a parser to recover.
- In {{iq3_mt_clean_state}}, the calls run but leave the environment in the wrong state.
- In {{iq3_mt_clean_exec}}, the calls return the wrong results.

At 2 bits the share with a broken call is larger ({{iq2_m_mt_new_fail_broken}} of {{iq2_m_mt_new_fail}}), but still a minority. The base is small: {{iq3_mt_reverse}} tasks go the other way (8-bit fails, P passes).

Multi-turn outputs also can't be re-scored after the fact. A call the parser dropped was never executed, so the model never saw its result, and every later turn depends on that. Measuring how much a tolerant parser would recover in multi-turn needs a new generation run with the tolerant parser active. I haven't done that here.

## 6. Objections

- **A server or stop-token problem?** The 8-bit build and W ran on the same server with the same settings and leave {{q8_0_sp_unclosed}} and {{w_sp_unclosed}} tags unclosed.
- **Truncated outputs?** {{iq3_sp_recoverable}} of the {{iq3_sp_unclosed}} unclosed outputs are complete calls; the model stopped right after the JSON.
- **A wrong leaderboard number?** No. None of these builds is on the BFCL leaderboard. The point is about anyone who runs BFCL on a quantized build.
- **A bad public quant?** This is about one imatrix of one model, tested on three categories. It says nothing about the publisher's other models or about their process in general.

## 7. What to do about it

- **Benchmark users:** report malformed calls separately from "no call". The patch in this repo adds a `tool_call_parse` field to every failed entry ("unclosed", "keys_dropped", "schema_echo", a crash message, …) without changing any score.
- **Benchmark maintainers:** a call without `"name"` or `"arguments"` should not be able to crash a task. Filtering malformed calls out of the chat history prevents the crash shapes inferred above (opt-in `SILENT_ZERO_SAFE_HISTORY` in the patch). This is tested in unit tests, not yet by re-generation.
- **People using quantized models in agents:** run a short tool-call smoke test before trusting a low-bit build, and check the tags specifically. Grammar-constrained decoding (llama.cpp and vLLM both offer it) prevents syntactically malformed calls, as long as the grammar covers the tags. It doesn't make a wrong call right.
- **People making quants:** the imatrix can decide format tokens. A few hundred tool-call prompts are a cheap check.

## Setup and limits

- **Model:** Qwen3-4B-Instruct-2507, served with llama.cpp `llama-server` (build tag b10964) on Apple Silicon. One slot, greedy decoding, server prompt cache off.
- **Determinism:** greedy runs reproduced byte-identically across server restarts. I checked this for the 8-bit build and P on `multi_turn_base`.
- **Benchmark:** BFCL at `6ea5797`, model entry `Qwen/Qwen3-4B-Instruct-2507-FC`.
- **Limits:**
  - One model family, one run per build. Differences between near-identical imatrices need repeated imatrix runs before they mean much.
  - "Tolerant" accepts only a missing final tag and loose whitespace.
  - Repairing dropped keys (`SILENT_ZERO_REPAIR_KEYS`) only matters in multi-turn and needs re-generation.
- **Reproduce:** `audit/classify.py` (classification), `audit/imatrix_facts.py` (imatrix headers and chunk counts), `patch/` (patched BFCL, tests, `verify.py`), `writeup/build.py` (this text). Tables: `audit/results.md`.
