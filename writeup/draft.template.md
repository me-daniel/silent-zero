# Silent Zero: a 3-bit model drops one token, and the benchmark scores it as silence

*Draft. All numbers are filled in by `writeup/build.py` from the audit and re-scoring outputs.*

**In short**

- On BFCL's `simple_python`, a public 3-bit (IQ3_XXS) build of Qwen3-4B-Instruct-2507 scores {{iq3_sp_off}}%, against {{q8_0_sp_off}}% for 8-bit. About {{iq3_sp_parser_points}} of those {{iq3_sp_total_points}} points come from one thing: the model writes a complete, correct call and stops before the closing `</tool_call>`. BFCL's parser then drops the call without a trace and scores the item as if the model had answered in prose.
- Whether a 3-bit build forgets that tag depends on its calibration data. Builds made from the same weights with the public calibration file almost never do it. Once the parser tolerates the missing tag, every 3-bit build scores {{iq3_sp_tol_min}}–{{iq3_sp_tol_max}}%.
- In multi-turn tasks the story is different, and the parser explains little. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and 3-bit fails, only {{iq3_mt_new_fail_broken}} contain a broken call. The rest are real decision failures.

## 1. Five lines of parsing

BFCL's handler for Qwen-style models extracts tool calls like this (`qwen_fc.py`, commit `6ea5797`, the current `main`):

```python
pattern = r"<tool_call>\n(.*?)\n</tool_call>"
matches = re.findall(pattern, input_string, re.DOTALL)
for match in matches:
    try:
        result.append(json.loads(match))
    except Exception as e:
        pass
```

Anything that doesn't match is gone: an unclosed tag, missing newlines around the JSON, invalid JSON. Single-turn items then fail as `wrong_count` ("Wrong number of functions"), exactly like a model that called nothing. In multi-turn tasks the turn simply ends, and the handler's own log says "Successfully decoded model response" with an empty list.

There's a second path. A call that *is* valid JSON but lacks a usable `"name"` or `"arguments"` key is kept and written into the chat history. When the next prompt is built, the template indexes `tool_call["name"]` or `tool_call["arguments"]` and raises. The whole multi-turn task is then recorded as `inference_error`, including turns the model might have passed.

## 2. What the 3-bit model actually writes

I classified every saved model output by what the parser does with it. Model: Qwen3-4B-Instruct-2507; one 8-bit and several 3-bit GGUF builds; llama.cpp, greedy decoding.

**Single-turn** (`simple_python`, {{sp_items}} items). The public IQ3_XXS build leaves {{iq3_sp_unclosed}} calls unclosed; {{iq3_sp_recoverable}} of them are complete, parseable calls apart from the missing tag. The 8-bit build leaves {{q8_0_sp_unclosed}} unclosed. On `live_simple` ({{ls_items}} items), the 3-bit build leaves {{iq3_ls_unclosed}} unclosed.

A typical output, scored as "no function call":

```
<tool_call>
{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}
```

**Multi-turn** (`multi_turn_base`, {{mt_tasks}} tasks). The 8-bit build produces {{q8_0_mt_malformed}} malformed calls in {{q8_0_mt_steps}} steps. The 3-bit build produces {{iq3_mt_malformed}} in {{iq3_mt_tasks_malformed}} tasks. The largest group is a different near miss: the key names go missing ({{iq3_mt_keys_dropped}} steps).

```
<tool_call>
{"cd", "arguments": {"folder": "document"}}
</tool_call>
```

The intended call is unambiguous, but it isn't JSON. Another {{iq3_mt_untagged}} steps are a complete call with no tags at all: the whole output is `{"name": "ls", "arguments": {...}}`, which BFCL reads as plain text. A build calibrated on tool-call dialogues does this even more often ({{t_mt_untagged}} steps). At 2 bits (IQ2_M), the model also starts copying the tool's schema instead of calling it ({{iq2_m_mt_schema_echo}} steps). It also loses {{iq2_m_mt_crashes}} whole tasks to the crash: {{iq2_m_mt_crash_args}} on a call with a `"name"` but no `"arguments"` (which is what a copied schema looks like), {{iq2_m_mt_crash_name}} on a missing `"name"`. A crashed task keeps no log, so the exact output that crashed it isn't recorded.

## 3. Official vs tolerant scoring

I re-scored the same saved outputs with a patched evaluator that accepts a final unclosed `<tool_call>` and loose whitespace. Nothing else changes: names, argument types and values are still checked by BFCL's own checker.

| Build | simple_python official | tolerant | live_simple official | tolerant |
|---|---|---|---|---|
| 8-bit (Q8_0) | {{q8_0_sp_off}} | {{q8_0_sp_tol}} | {{q8_0_ls_off}} | {{q8_0_ls_tol}} |
| 3-bit, public build | {{iq3_sp_off}} | {{iq3_sp_tol}} | {{iq3_ls_off}} | {{iq3_ls_tol}} |
| 3-bit, public imatrix, newer quantizer | {{r_sp_off}} | {{r_sp_tol}} | {{r_ls_off}} | {{r_ls_tol}} |
| 3-bit, public calibration file | {{w_sp_off}} | {{w_sp_tol}} | {{w_ls_off}} | {{w_ls_tol}} |
| 3-bit, own general-text calibration | {{g_sp_off}} | {{g_sp_tol}} | {{g_ls_off}} | {{g_ls_tol}} |

With all patch options off, the patched evaluator reproduces the official scores and failed entries exactly ({{verify_identical}} of {{verify_combos}} build × category combinations).

## 4. Calibration decides the tag, not the call

Builds 2–5 in the table come from the same 16-bit weights. They differ in their importance matrix (imatrix), and build 3 also in the llama.cpp version used to quantize.

- **With the public imatrix:** {{iq3_sp_unclosed}} unclosed tags (the public file), and {{r_sp_unclosed}} when that imatrix is re-quantized with a newer llama.cpp.
- **With the public calibration text run through llama-imatrix ourselves:** {{w_sp_unclosed}} unclosed tag at chunk length 512 with the newer llama.cpp, and {{v_sp_unclosed}} with b6096, the release the public model card names.
- **With our own general-text corpus:** {{x_sp_unclosed}} at chunk length 512.

So neither chunk length, nor text preprocessing, nor the imatrix program version, nor the quantizer version reproduces the failure. What remains is the corpus the public imatrix was computed on. Its GGUF header records {{imatrix_public_chunks}} chunks of 512 tokens. The published file gives {{imatrix_w_chunks}} chunks at the same size, so the public imatrix saw about {{imatrix_ratio}} times as much text. That corpus isn't published, so the exact cause stays open.

The tolerant scores add a twist. Once the tag is forgiven, the builds with the public imatrix are as good as any other 3-bit build on `simple_python`, and on `live_simple` they score highest of all 3-bit builds (one greedy run each). On these two categories, the calibration changed one token, not the model's ability to choose and fill a call.

The effect also depends on the task. Our general-text build leaves only {{g_sp_unclosed}} tags unclosed on `simple_python`, but {{g_mt_unclosed}} in multi-turn steps. The public build does the reverse: {{iq3_sp_unclosed}} on `simple_python`, {{iq3_mt_unclosed}} in multi-turn. A calibration that looks clean on one benchmark category isn't clean everywhere.

## 5. What the parser does not explain

In `multi_turn_base`, 8-bit passes {{q8_0_mt_correct}} of {{mt_tasks}} tasks and the public 3-bit build {{iq3_mt_correct}}. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and 3-bit fails, {{iq3_mt_new_fail_broken}} contain a malformed call or a crash. The other {{iq3_mt_clean_fail}} fail with well-formed behaviour. In {{iq3_mt_clean_empty}}, a turn ends without any call, typically right after the model announces what it is about to do. In {{iq3_mt_clean_state}}, the calls run but leave the environment in the wrong state, and in {{iq3_mt_clean_exec}} they return the wrong results. At 2 bits the share is larger ({{iq2_m_mt_new_fail_broken}} of {{iq2_m_mt_new_fail}}), but still a minority.

Multi-turn outputs also can't be re-scored after the fact. A call the parser dropped was never executed, so the model never saw its result, and every later turn depends on that. Measuring how much a tolerant parser would recover in multi-turn needs a new generation run with the tolerant parser active. That hasn't been done here.

## 6. What to do about it

- **Benchmark users:** report malformed calls separately from "no call". The patch in this repo adds a `tool_call_parse` field to every failed entry ("unclosed", "keys_dropped", "schema_echo", a crash message, …) without changing any score.
- **Benchmark maintainers:** a call without `"name"` or `"arguments"` should not be able to crash a task. Filtering malformed calls out of the chat history prevents every crash shape found here (opt-in `SILENT_ZERO_SAFE_HISTORY` in the patch).
- **People using quantized models in agents:** run a short tool-call smoke test before trusting a low-bit build, and check the tags specifically. Grammar-constrained decoding (llama.cpp and vLLM both offer it) rules out malformed calls by construction.
- **People making quants:** calibration can decide format tokens. A few hundred tool-call prompts are a cheap check.

## Setup and limits

- **Model:** Qwen3-4B-Instruct-2507, served with llama.cpp `llama-server` b10964 on Apple Silicon. One slot, greedy decoding, server prompt cache off. Greedy runs reproduced byte-identically across server restarts (checked for the 8-bit and the public 3-bit build on `multi_turn_base`).
- **Benchmark:** BFCL at `6ea5797`, model entry `Qwen/Qwen3-4B-Instruct-2507-FC`.
- **Limits:**
  - One model family, one run per build. Paired differences between near-identical calibrations need repeated calibrations before they mean much.
  - "Tolerant" accepts only a missing final tag and loose whitespace.
  - Repairing dropped keys (`SILENT_ZERO_REPAIR_KEYS`) only matters in multi-turn and needs re-generation.
- **Reproduce:** `audit/classify.py` (classification), `patch/` (patched BFCL, tests, `verify.py`), `writeup/build.py` (this text). Tables: `audit/results.md`.
