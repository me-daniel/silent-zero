# Silent Zero: a 3-bit model drops one token, and the benchmark scores it as silence

*Draft. All numbers are filled in by `writeup/build.py` from the audit and re-scoring outputs.*

## Abstract

Quantized GGUF files are a common way to run open-weight language models on local hardware, and tool calling is what turns those models into agents. Ask a 3-bit model for a tool call and it may write the call correctly, then stop one token short. The benchmark cannot tell that apart from a model that called nothing.

This note audits how the Berkeley Function Calling Leaderboard (BFCL) scores {{n_builds}} GGUF builds of one model, Qwen3-4B-Instruct-2507, from 8 down to 2 bits. All builds come from the same 16-bit weights and run on the same server with greedy decoding. Every saved output is classified by what BFCL's parser does with it, and the single-turn outputs are re-scored with a patched parser that accepts a missing closing tag. With its options off, the patched evaluator reproduces the official scores exactly.

The main findings are: (1) On `simple_python`, most of the 3-bit score loss is a parsing artefact. The public IQ3_XXS build scores {{iq3_sp_off}}% against {{q8_0_sp_off}}% for 8-bit, and {{iq3_sp_parser_points}} of those {{iq3_sp_total_points}} points come from complete calls that lack the closing `</tool_call>`. (2) Whether the tag goes missing depends on the importance matrix used for quantization, not on the quantizer version. With the tag tolerated, all {{n_iq3}} 3-bit builds score between {{iq3_sp_tol_min}}% and {{iq3_sp_tol_max}}% on `simple_python`. (3) In multi-turn tasks the parser explains little. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and 3-bit fails, {{iq3_mt_new_fail_broken}} contain a malformed call and the other {{iq3_mt_clean_fail}} contain none. (4) Some malformed calls crash a whole multi-turn task. At 2 bits, {{iq2_m_mt_crashes}} of {{mt_tasks}} tasks are lost this way. The patch in this repository reports each of these failures separately without changing any score.

## 1. How BFCL reads a tool call

BFCL's handler for Qwen-style models extracts tool calls with a single regular expression (`qwen_fc.py` at commit `6ea5797`, the latest `main` as of November 2026):

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

Anything that does not match is dropped without a trace. That covers an unclosed tag, missing newlines around the JSON and invalid JSON. A single-turn item then fails as `wrong_count` ("Wrong number of functions"), exactly like a model that called nothing. In a multi-turn task the turn simply ends, and the handler's own log reports "Successfully decoded model response" with an empty list.

A second path is worse. A call that is valid JSON but lacks a usable `"name"` or `"arguments"` key is kept and written into the chat history. When the next prompt is built, the chat template indexes `tool_call["name"]` or `tool_call["arguments"]` and raises an exception. BFCL then records the whole multi-turn task as `inference_error`, including turns the model might have passed.

## 2. Builds and setup

A build is one GGUF file of the model. Low-bit builds are usually made with an **importance matrix** (imatrix): per-weight statistics that llama.cpp collects by running the 16-bit model over a calibration text, cut into **chunks** of a fixed number of tokens. Q8_0, Q4_K_M, IQ2_M and the public IQ3_XXS build (called P below) come from `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` at revision `ae44f08`. The other {{n_own}} are IQ3_XXS builds I quantized myself from the same weights. They differ from P in the imatrix and, for one of them, in the llama.cpp version used to quantize. Section 4 lists them all.

All builds were served with llama.cpp `llama-server` (build tag b10964) on Apple Silicon, with one slot, greedy decoding and the server prompt cache off. Greedy runs reproduced byte-identically across server restarts. I checked this for the 8-bit build and P on `multi_turn_base`. The benchmark is BFCL at `6ea5797` with the model entry `Qwen/Qwen3-4B-Instruct-2507-FC`, on {{n_cats}} categories: `simple_python` ({{sp_items}} items), `live_simple` ({{ls_items}} items) and `multi_turn_base` ({{mt_tasks}} tasks). In a multi-turn task, the user speaks in several turns, and the model answers each turn in one or more steps, one generation each.

## 3. What the models write

### Single-turn

P leaves {{iq3_sp_unclosed}} of its {{sp_items}} `simple_python` calls unclosed. {{iq3_sp_recoverable}} of them are complete, parseable calls apart from the missing tag, so the model stopped after the JSON and not in the middle of it. A typical output, scored as "no function call", looks like this:

```
<tool_call>
{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}
```

The 8-bit build, on the same server with the same settings, leaves {{q8_0_sp_unclosed}} calls unclosed. On `live_simple`, P leaves {{iq3_ls_unclosed}} of {{ls_items}} unclosed.

### Multi-turn

The 8-bit build produces {{q8_0_mt_malformed}} malformed calls in {{q8_0_mt_steps}} steps. P produces {{iq3_mt_malformed}}, spread over {{iq3_mt_tasks_malformed}} tasks. The largest group is a different near miss, in which the key names go missing ({{iq3_mt_keys_dropped}} steps):

```
<tool_call>
{"cd", "arguments": {"folder": "document"}}
</tool_call>
```

The intended call is clear, but the text is not JSON. Another {{iq3_mt_untagged}} steps are a complete call with no tags at all. The whole output is `{"name": "ls", "arguments": {...}}`, which BFCL reads as plain text. A build calibrated on tool-call dialogues (T in section 4) does this more often, in {{t_mt_untagged}} steps. Calls that use the function name as the key, such as `{"ls": {...}}`, also appear.

At 2 bits (IQ2_M), the model also starts copying the tool's schema instead of calling it, in at least {{iq2_m_mt_schema_echo}} steps. More of its unclosed steps look like schema copies too. IQ2_M loses {{iq2_m_mt_crashes}} whole tasks to the crash path described in section 1. In {{iq2_m_mt_crash_args}} of them the call had a `"name"` but no `"arguments"`, which is what a copied schema looks like, and in {{iq2_m_mt_crash_name}} the `"name"` was missing. A crashed task keeps no log, so the output that crashed it is not recorded and the shapes are inferred from the error messages.

## 4. Official and tolerant scores

I re-scored the saved single-turn outputs with a patched evaluator that accepts a final unclosed `<tool_call>` and loose whitespace around the JSON. Nothing else changes. Names, argument types and values are still checked by BFCL's own checker. "Unclosed" counts `simple_python` items.

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

For P on `simple_python`, the tolerant parser turns {{iq3_sp_tol_gain_items}} items into correct answers and creates no new failures. With all patch options off, the patched evaluator reproduces the official scores and failed entries exactly, in {{verify_identical}} of {{verify_combos}} build and category combinations.

## 5. The tag depends on the imatrix

P and R share an imatrix and differ in the quantizer version. P leaves {{iq3_sp_unclosed}} tags unclosed and R leaves {{r_sp_unclosed}}, so the newer quantizer does not remove the failure. Every build with a different imatrix stays low. W and V use the calibration file that P's header names (`calibration_datav3.txt`, the revision I downloaded), run through llama-imatrix with the newer llama.cpp and with b6096, the release the public model card names. Each leaves {{w_sp_unclosed}} tag unclosed. X and G, made from my own general text, leave {{x_sp_unclosed}} and {{g_sp_unclosed}}.

I could not reproduce P's imatrix from the published file. P's header records {{imatrix_public_chunks}} chunks. The file I used is {{calib_tokens}} tokens long, which gives {{calib_chunks_512}} chunks at 512 tokens (the count W's header records) and {{calib_chunks_128}} at 128. No chunk size gives exactly {{imatrix_public_chunks}} from this file, and the header does not store the chunk size. P's imatrix was therefore made from different text, with different settings, or both. I have not tested other chunk sizes or options on the published file, or other revisions of it, and the whole effect rests on a single imatrix.

Low bits are part of the explanation. Q4_K_M, whose header records the same imatrix as P's (same file name, same chunk count), produces {{q4_k_m_mt_malformed}} malformed steps in multi-turn.

With the tag tolerated, P and R are as good as the other 3-bit builds on `simple_python`. On `live_simple` they score highest of the 3-bit builds, but the gaps are a handful of items with one greedy run each, so the order means little. On these two categories, most of what separated P from my builds was the closing tag and not the choice of call.

The effect also depends on the task. G leaves {{g_sp_unclosed}} tags unclosed on `simple_python` but {{g_mt_unclosed}} in multi-turn steps. P does the reverse, with {{iq3_sp_unclosed}} on `simple_python` and {{iq3_mt_unclosed}} in multi-turn. A build that looks clean on one category is not necessarily clean on another, and W, V and X were only run on the single-turn categories.

## 6. What the parser does not explain

In `multi_turn_base`, the 8-bit build passes {{q8_0_mt_correct}} of {{mt_tasks}} tasks and P passes {{iq3_mt_correct}}. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and P fails, {{iq3_mt_new_fail_broken}} contain a malformed call or a crash. A broken call in a failed task could have caused the failure, but the audit does not show that it did.

The other {{iq3_mt_clean_fail}} tasks contain no malformed call. In {{iq3_mt_clean_empty}} of them, a turn ends without any call, typically right after the model announces what it is about to do ("Let me do that for you."). That looks like generation ending too early, a relative of the missing tag, but there is no call for a parser to recover. In {{iq3_mt_clean_state}}, the calls run but leave the environment in the wrong state, and in {{iq3_mt_clean_exec}} they return the wrong results. At 2 bits the share with a broken call is larger ({{iq2_m_mt_new_fail_broken}} of {{iq2_m_mt_new_fail}}) but still a minority. The base is small, and {{iq3_mt_reverse}} tasks go the other way, failing at 8 bits and passing with P.

Multi-turn outputs also cannot be re-scored after the fact. A call the parser dropped was never executed, so the model never saw its result, and every later turn depends on that. Measuring how much a tolerant parser would recover in multi-turn needs a new generation run with the tolerant parser active, which this note does not include.

## 7. Limitations

The study covers one model family, with one greedy run per build. Differences between near-identical imatrices would need repeated imatrix runs before they mean much. The tolerant mode accepts only a missing final tag and loose whitespace, and the key repair in the patch (`SILENT_ZERO_REPAIR_KEYS`) matters only in multi-turn, where it needs re-generation.

Two objections are worth answering directly. The missing tag is not a server or stop-token problem, because the 8-bit build and W ran on the same server with the same settings and leave {{q8_0_sp_unclosed}} and {{w_sp_unclosed}} tags unclosed. The outputs are not truncated either, because {{iq3_sp_recoverable}} of the {{iq3_sp_unclosed}} unclosed outputs are complete calls.

None of these builds is on the BFCL leaderboard, so no leaderboard number is claimed wrong. The finding concerns anyone who runs BFCL on a quantized build. The failure also comes from one imatrix of one model, tested on {{n_cats}} categories, and says nothing about the publisher's other models.

## 8. Conclusion

This note asked how much of a quantized model's tool-calling score on BFCL reflects the calls it makes, and how much reflects the parser that reads them. The audit covered {{n_builds}} builds of Qwen3-4B-Instruct-2507 step by step, and the single-turn outputs were re-scored with a tolerant parser that leaves BFCL's checks otherwise unchanged.

The main findings are:

1. Most of the 3-bit loss on `simple_python` is a missing closing tag. {{iq3_sp_parser_points}} of {{iq3_sp_total_points}} points come from items whose call is complete apart from `</tool_call>`. A tolerant parser turns {{iq3_sp_tol_gain_items}} such items into correct answers.
2. Whether the tag goes missing follows the imatrix. P and R share an imatrix and fail, while every build with another imatrix leaves at most {{other_sp_unclosed_max}} tags unclosed. The published calibration file does not reproduce P's imatrix, so its exact origin stays open.
3. Multi-turn failures are mostly not parsing failures. {{iq3_mt_clean_fail}} of {{iq3_mt_new_fail}} new failures contain no malformed call. Whether a tolerant parser would rescue the other {{iq3_mt_new_fail_broken}} can only be measured by re-generation.
4. Malformed calls can cost whole tasks. At 2 bits, {{iq2_m_mt_crashes}} of {{mt_tasks}} multi-turn tasks crash on a call without `"name"` or `"arguments"`, and filtering such calls out of the chat history prevents every crash shape inferred here (`SILENT_ZERO_SAFE_HISTORY`, tested in unit tests).

The patch in this repository makes the distinction visible without changing any score. Every failed entry gets a `tool_call_parse` field that names the failure ("unclosed", "keys_dropped", "schema_echo" or a crash message), so a benchmark user can report malformed calls separately from missing ones. Anyone running a low-bit build in an agent can get the same information from a short tool-call smoke test that checks the tags specifically. Grammar-constrained decoding, which llama.cpp and vLLM both offer, prevents syntactically malformed calls as long as the grammar covers the tags, but it does not make a wrong call right. For people making quants, a few hundred tool-call prompts are a cheap check on what the imatrix did to the format tokens.

Several directions follow. A multi-turn run with the tolerant parser active would measure what the parser costs there, where re-scoring cannot. Repeated imatrix runs would show how much of the spread between near-identical builds is noise. Other models and other BFCL handlers parse tool calls differently, so the same audit on them would show how general the missing tag is. Finally, finding the text and settings behind P's imatrix would explain why that one imatrix removes the tag. A missing tag and a missing answer are different failures, and a benchmark should be able to tell them apart.
