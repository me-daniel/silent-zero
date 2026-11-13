# Silent Zero: a 3-bit model drops one token, and the benchmark scores it as silence

*Draft. All numbers are filled in by `writeup/build.py` from the audit and re-scoring outputs, and the figures are drawn from the same files.*

## Abstract

Quantized GGUF files are a common way to run open-weight language models on local hardware, and tool calling is what turns those models into agents. Ask a 3-bit model for a tool call and it may write the call correctly, then stop, usually one token short. The benchmark cannot tell that apart from a model that called nothing.

This note audits how the Berkeley Function Calling Leaderboard (BFCL) scores {{n_builds}} GGUF builds of one model, Qwen3-4B-Instruct-2507: four public builds from 8 down to 2 bits, and {{n_own}} 3-bit builds of my own, each of which changes one step in how the importance matrix was made. All builds come from the same 16-bit weights and run on the same server with greedy decoding. I classify every saved output by what BFCL's parser does with it, and re-score the single-turn outputs with a patched parser that accepts a missing closing tag. With its options off, the patched evaluator reproduces the official scores exactly.

The main findings are: (1) For the public 3-bit build, most of the score loss on `simple_python` is a parsing artefact. The public IQ3_XXS build scores {{iq3_sp_off}}% against {{q8_0_sp_off}}% for 8-bit, and {{iq3_sp_parser_points}} of those {{iq3_sp_total_points}} points come from complete calls that lack the closing `</tool_call>`. (2) On the single-turn categories, whether the tag goes missing depends on the importance matrix used for quantization, not on the llama.cpp version. With the tag tolerated, all {{n_iq3}} 3-bit builds score between {{iq3_sp_tol_min}}% and {{iq3_sp_tol_max}}% on `simple_python`. (3) In multi-turn tasks the parser explains less. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and the public 3-bit build fails, {{iq3_mt_new_fail_broken}} contain a malformed call and the other {{iq3_mt_clean_fail}} contain none. (4) Some malformed calls crash a whole multi-turn task. At 2 bits, {{iq2_m_mt_crashes}} of {{mt_tasks}} tasks are lost this way. The patch in this repository reports each of these failures separately without changing any score.

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

Anything that does not match is dropped without a trace. That covers an unclosed tag, missing newlines around the JSON and invalid JSON. A single-turn item then fails as `wrong_count` ("Wrong number of functions"), exactly like a model that called nothing. In a multi-turn task the turn ends, and the handler's own log reports "Successfully decoded model response" with an empty list.

A second path is worse. A call that is valid JSON but lacks a usable `"name"` or `"arguments"` key is kept and written into the chat history. When the next prompt is built, the chat template indexes `tool_call["name"]` or `tool_call["arguments"]` and raises an exception. BFCL then records the whole multi-turn task as `inference_error`, including turns the model might have passed. Figure 1 shows the four kinds of output this note deals with and what BFCL does with each.

![Figure 1: what BFCL's parser does with four kinds of model output](figures/generated/fig_1_parser.svg)

*Figure 1. What BFCL's parser does with four kinds of model output.* Kinds 2 to 4 occur in the saved outputs of the low-bit builds (arguments shortened). Blue: what changes with the patch's opt-in mode for that kind of output (`SILENT_ZERO_TOLERANT`, `SILENT_ZERO_REPAIR_KEYS` and `SILENT_ZERO_SAFE_HISTORY` in `patch/README.md`). All three are off by default, and then the patch only adds a label to each failed entry that records what the parser saw.

## 2. Builds and setup

A build is one GGUF file of the model. Low-bit builds are usually made with an **importance matrix** (imatrix), the per-weight statistics that llama.cpp collects by running the 16-bit model over a calibration text. The text is cut into chunks of a fixed number of tokens. All builds come from the same 16-bit weights.

Four builds are public: Q8_0, Q4_K_M, IQ3_XXS and IQ2_M, at 8, 4, 3 and 2 bits, from `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` at revision `ae44f08`. Their headers record the same imatrix, computed from `calibration_datav3.txt`. I call these the public imatrix and the public calibration text below. The public model card names llama.cpp b6096.

To find out where the missing tag comes from, I made {{n_own}} more 3-bit builds with llama.cpp b10964. Each one is the same as an earlier build except for one thing, and each is named after its imatrix. All {{n_iq3}} 3-bit builds, public and mine, give every tensor the same quantization type, so they differ only in the imatrix and in the llama.cpp version.

| My build | Same as | Except |
|---|---|---|
| `public-imatrix` | IQ3_XXS | quantized by me with b10964, from the same imatrix file |
| `public-text` | `public-imatrix` | imatrix recomputed from the public calibration text, in 512-token chunks |
| `public-text-b6096` | `public-text` | imatrix computed with b6096, the release the public model card names |
| `my-text` | `public-text` | imatrix computed from my own general text |
| `my-text-2048` | `my-text` | 2048-token chunks instead of 512 |
| `my-text-tools` | `my-text-2048` | tool-call dialogues added to the text, which also makes it longer |

The first two steps separate the quantizer from the imatrix, the third tests the llama.cpp version behind the imatrix, and the last three change the calibration text and how it is cut.

All builds were served with llama.cpp `llama-server` (b10964) on Apple Silicon, with one slot, greedy decoding and the server prompt cache off. Greedy runs reproduced byte-identically across server restarts. I checked this for Q8_0 and IQ3_XXS on `multi_turn_base`.

The benchmark is BFCL at `6ea5797` with the model entry `Qwen/Qwen3-4B-Instruct-2507-FC`, on {{n_cats}} categories: `simple_python` ({{sp_items}} items), `live_simple` ({{ls_items}} items) and `multi_turn_base` ({{mt_tasks}} tasks). In a multi-turn task, the user speaks in several turns, and the model answers each turn in one or more steps, one generation each.

## 3. What the models write

### Single-turn

The public IQ3_XXS build leaves {{iq3_sp_unclosed}} of its {{sp_items}} `simple_python` calls unclosed. Of these, {{iq3_sp_recoverable}} are complete, parseable calls apart from the missing tag, so the model stopped after the JSON and not in the middle of it. In {{iq3_sp_one_short}} of them only `</tool_call>` itself is missing, which is a single token in the model's vocabulary. A typical output, scored as "no function call", looks like this:

```
<tool_call>
{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}
```

The 8-bit build, on the same server with the same settings, leaves {{q8_0_sp_unclosed}} calls unclosed. On `live_simple`, IQ3_XXS leaves {{iq3_ls_unclosed}} of {{ls_items}} unclosed.

### Multi-turn

The 8-bit build produces {{q8_0_mt_malformed}} malformed calls in {{q8_0_mt_steps}} steps. IQ3_XXS produces {{iq3_mt_malformed}}, spread over {{iq3_mt_tasks_malformed}} tasks. The largest group is a different near miss, in which the key names go missing ({{iq3_mt_keys_dropped}} steps):

```
<tool_call>
{"cd", "arguments": {"folder": "document"}}
</tool_call>
```

The intended call is clear, but the text is not JSON. Another {{iq3_mt_untagged}} steps are a complete call with no tags at all. The whole output is `{"name": "ls", "arguments": {...}}`, which BFCL reads as plain text. `my-text-tools`, whose calibration text includes tool-call dialogues, does this more often, in {{t_mt_untagged}} steps. Calls that use the function name as the key, such as `{"ls": {...}}`, also appear.

At 2 bits (IQ2_M), the model also starts copying the tool's schema instead of calling it, in at least {{iq2_m_mt_schema_echo}} steps. More of its unclosed steps look like schema copies too. IQ2_M loses {{iq2_m_mt_crashes}} whole tasks to the crash path described in section 1. In {{iq2_m_mt_crash_args}} of them the call had a `"name"` but no `"arguments"`, which is what a copied schema looks like, and in {{iq2_m_mt_crash_name}} the `"name"` was missing. A crashed task keeps no log, so the output that crashed it is not recorded and the shapes are inferred from the error messages.

Figure 2 puts the builds side by side. Each 3-bit and 2-bit build breaks its calls in its own way, even among builds with the same format.

![Figure 2: malformed steps in multi-turn by kind, per build](figures/generated/fig_2_malformed.svg)

*Figure 2. The 3-bit and 2-bit builds break their calls in different ways.* Malformed model steps in `multi_turn_base` per build, by kind. Q8_0 and Q4_K_M produce none. Three of my builds were run on multi-turn. Tasks lost to a crash are not counted here ({{iq2_m_mt_crashes}} for IQ2_M).

## 4. Scores with the missing tag tolerated

I re-scored the saved single-turn outputs with a patched evaluator, the tolerant parser, which also accepts a final unclosed `<tool_call>` and loose whitespace around the JSON. Nothing else changes. Names, argument types and values are still checked by BFCL's own checker. Figure 3 shows how much of each build's `simple_python` score the missing tag costs, and the table below gives the exact numbers.

![Figure 3: simple_python score per build, with and without the missing tag tolerated](figures/generated/fig_3_scores.svg)

*Figure 3. Tolerating the missing closing tag lifts the two builds with the public imatrix to the level of my other 3-bit builds.* Grey: BFCL's score on `simple_python`. Amber: what the same saved outputs gain when the missing closing tag is tolerated. Dotted line: the score of Q8_0. One greedy run per build.

| Build | Calls without closing tag (of {{sp_items}}) | BFCL score (%) | Score with the tag tolerated (%) | Difference (points) |
|---|---|---|---|---|
| *Public* | | | | |
| Q8_0 (8-bit) | {{q8_0_sp_unclosed}} | {{q8_0_sp_off}} | {{q8_0_sp_tol}} | {{q8_0_sp_gain}} |
| IQ3_XXS (3-bit) | {{iq3_sp_unclosed}} | {{iq3_sp_off}} | {{iq3_sp_tol}} | {{iq3_sp_gain}} |
| *Mine, all 3-bit* | | | | |
| `public-imatrix` | {{r_sp_unclosed}} | {{r_sp_off}} | {{r_sp_tol}} | {{r_sp_gain}} |
| `public-text` | {{w_sp_unclosed}} | {{w_sp_off}} | {{w_sp_tol}} | {{w_sp_gain}} |
| `public-text-b6096` | {{v_sp_unclosed}} | {{v_sp_off}} | {{v_sp_tol}} | {{v_sp_gain}} |
| `my-text` | {{x_sp_unclosed}} | {{x_sp_off}} | {{x_sp_tol}} | {{x_sp_gain}} |
| `my-text-2048` | {{g_sp_unclosed}} | {{g_sp_off}} | {{g_sp_tol}} | {{g_sp_gain}} |
| `my-text-tools` | {{t_sp_unclosed}} | {{t_sp_off}} | {{t_sp_tol}} | {{t_sp_gain}} |

On `live_simple` ({{ls_items}} items) the pattern is the same. IQ3_XXS rises from {{iq3_ls_off}}% to {{iq3_ls_tol}}% and `public-imatrix` from {{r_ls_off}}% to {{r_ls_tol}}%, while no other 3-bit build gains more than {{other_ls_gain_max}} points. Every category and build is in `audit/results.md`.

For IQ3_XXS on `simple_python`, the tolerant parser turns {{iq3_sp_tol_gain_items}} items into correct answers and creates no new failures. With all patch options off, the patched evaluator reproduces the official scores and failed entries exactly, in {{verify_identical}} of {{verify_combos}} build and category combinations.

## 5. The tag depends on the imatrix

The chain of builds in section 2 points to the imatrix. IQ3_XXS and `public-imatrix` share the public imatrix but were quantized with different llama.cpp versions, and both drop the tag often, in {{iq3_sp_unclosed}} and {{r_sp_unclosed}} of {{sp_items}} items. `public-imatrix` and `public-text` differ only in the imatrix, and `public-text` drops the tag in {{w_sp_unclosed}} item. `public-text-b6096`, whose imatrix was computed with b6096, also drops it in {{v_sp_unclosed}}. Between these two releases, the llama.cpp version therefore does not decide whether the tag goes missing, neither for quantizing nor for computing the imatrix. Every build with an imatrix other than the public one drops the tag in at most {{other_sp_unclosed_max}} items, whether that imatrix came from the public calibration text or from my own text.

I could not reproduce the public imatrix from the public calibration text (the revision I downloaded). The public header records {{imatrix_public_chunks}} chunks. The file I used is {{calib_tokens}} tokens long, which gives {{calib_chunks_512}} chunks at 512 tokens (the count the header of `public-text` records) and {{calib_chunks_128}} at 128. No chunk size gives exactly {{imatrix_public_chunks}} from this file, and the header does not store the chunk size. The public imatrix was therefore made from different text, with different settings, or both. I have not tested other chunk sizes or options on the public calibration text, or other revisions of it, and the whole effect rests on a single imatrix.

Low bits are part of the explanation. Q4_K_M, whose header records the same imatrix as IQ3_XXS (same file name, same chunk count), produces {{q4_k_m_mt_malformed}} malformed steps in multi-turn, where IQ3_XXS produces {{iq3_mt_malformed}}. Q4_K_M was run on multi-turn only.

With the tag tolerated, the two builds with the public imatrix are as good as my other 3-bit builds on `simple_python`. On `live_simple` they score highest of the 3-bit builds, but the gaps are a handful of items with one greedy run each, so the order means little. On these two categories, most of what separated them from my other builds was the closing tag and not the choice of call.

The effect also depends on the task. `my-text-2048` leaves {{g_sp_unclosed}} tags unclosed on `simple_python` but {{g_mt_unclosed}} in multi-turn steps. IQ3_XXS does the reverse, with {{iq3_sp_unclosed}} on `simple_python` and {{iq3_mt_unclosed}} in multi-turn. A build that looks clean on one category is not necessarily clean on another, and `public-text`, `public-text-b6096` and `my-text` were only run on the single-turn categories.

## 6. What the parser does not explain

In `multi_turn_base`, the 8-bit build passes {{q8_0_mt_correct}} of {{mt_tasks}} tasks and IQ3_XXS passes {{iq3_mt_correct}}. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and IQ3_XXS fails, {{iq3_mt_new_fail_broken}} contain a malformed call or a crash. A broken call in a failed task could have caused the failure, but the audit does not show that it did.

The other {{iq3_mt_clean_fail}} tasks contain no malformed call. In {{iq3_mt_clean_empty}} of them, a turn ends without any call, typically right after the model announces what it is about to do ("Let me do that for you."). That looks like generation ending too early, a relative of the missing tag, but there is no call for a parser to recover. In {{iq3_mt_clean_state}}, the calls run but leave the environment in the wrong state, and in {{iq3_mt_clean_exec}} they return the wrong results. At 2 bits the share with a broken call is larger ({{iq2_m_mt_new_fail_broken}} of {{iq2_m_mt_new_fail}}) but still a minority, and the same holds for every low-bit build (Figure 4). The sample is small, and {{iq3_mt_reverse}} tasks go the other way, failing at 8 bits and passing with IQ3_XXS.

![Figure 4: why the multi-turn tasks fail that the 8-bit build passes](figures/generated/fig_4_failures.svg)

*Figure 4. For every low-bit build, most of the tasks it newly fails contain no malformed call.* Each bar counts the `multi_turn_base` tasks that Q8_0 passes and the build fails. Orange: tasks with a malformed call or a crash. Dark grey: a turn where the model stops without calling anything. Light grey: the calls run but leave the wrong state or return the wrong results. A parser cannot fix the grey parts.

Multi-turn outputs also cannot be re-scored after the fact. A call the parser dropped was never executed, so the model never saw its result, and every later turn depends on that. Measuring how much a tolerant parser would recover in multi-turn needs a new generation run with the tolerant parser active, which this note does not include.

## 7. Limitations

The study covers one model, with one greedy run per build. Differences between near-identical imatrices would need repeated imatrix runs before they mean much. The chain of builds also has gaps. Only three of my builds were run on multi-turn, the public calibration text was tried at one chunk size only, and `my-text-tools` changes the length of the text along with its content. The tolerant mode accepts only a missing final tag and loose whitespace, and the key repair in the patch (`SILENT_ZERO_REPAIR_KEYS`) matters only in multi-turn, where it needs re-generation.

Two objections come up. The missing tag is not caused by the server or a wrong stop setting, because the 8-bit build and `public-text` ran on the same server with the same settings and leave {{q8_0_sp_unclosed}} and {{w_sp_unclosed}} tags unclosed. The outputs are not truncated either, because {{iq3_sp_recoverable}} of the {{iq3_sp_unclosed}} unclosed outputs are complete calls.

None of these builds is on the BFCL leaderboard, so no leaderboard number is claimed wrong. The finding concerns anyone who runs BFCL on a quantized build. The failure also comes from one imatrix of one model, tested on {{n_cats}} categories, and says nothing about the publisher's other models.

## 8. Conclusion

This note asked how much of a quantized model's tool-calling score on BFCL reflects the calls it makes, and how much reflects the parser that reads them. The audit covered {{n_builds}} builds of Qwen3-4B-Instruct-2507 step by step, and the single-turn outputs were re-scored with a tolerant parser that leaves BFCL's checks otherwise unchanged.

The main findings are:

1. Most of the public 3-bit build's loss on `simple_python` is a missing closing tag. Of the {{iq3_sp_total_points}} points it loses, {{iq3_sp_parser_points}} come from items whose call is complete apart from `</tool_call>`. A tolerant parser turns {{iq3_sp_tol_gain_items}} such items into correct answers.
2. Whether the tag goes missing follows the imatrix. On `simple_python`, the two builds with the public imatrix leave {{iq3_sp_unclosed}} and {{r_sp_unclosed}} tags unclosed, while every build with another imatrix leaves at most {{other_sp_unclosed_max}}. Changing the llama.cpp version, for quantizing or for the imatrix, does not change this. In multi-turn the pattern differs, as section 5 shows. The published calibration file does not reproduce the public imatrix, so its exact origin stays open.
3. For the public build, multi-turn failures are mostly not parsing failures. Of the {{iq3_mt_new_fail}} tasks that 8-bit passes and IQ3_XXS fails, {{iq3_mt_clean_fail}} contain no malformed call. Whether a tolerant parser would rescue the other {{iq3_mt_new_fail_broken}} can only be measured by re-generation.
4. Malformed calls can cost whole tasks. At 2 bits, {{iq2_m_mt_crashes}} of {{mt_tasks}} multi-turn tasks crash on a call without `"name"` or `"arguments"`, and filtering such calls out of the chat history prevents every kind of crash inferred here (`SILENT_ZERO_SAFE_HISTORY`, tested in unit tests).

The patch in this repository makes the distinction visible without changing any score. Every failed entry gets a `tool_call_parse` field that records what the parser saw ("unclosed", "keys_dropped", "schema_echo", a crash message, or "ok" when the call was well formed but wrong), so a benchmark user can report malformed calls separately from missing ones. Anyone running a low-bit build in an agent can get the same information from a short tool-call smoke test that checks the tags specifically. Grammar-constrained decoding, which llama.cpp and vLLM both offer, should rule out syntactically malformed calls if the grammar covers the tags. I did not test it, and it cannot make a wrong call right. For people making quantized builds, a few hundred tool-call prompts are a cheap check on what the imatrix did to the format tokens.

Several directions follow. A multi-turn run with the tolerant parser active would measure what the parser costs there, where re-scoring cannot. Repeated imatrix runs would show how much of the spread between near-identical builds is noise. Other models and other BFCL handlers parse tool calls differently, so the same audit on them would show how general the missing tag is. Finally, finding the text and settings behind the public imatrix would explain why that one imatrix makes the model drop the tag. A missing tag and a missing answer are different failures, and a benchmark should be able to tell them apart.

## Code and data

The classifier is `audit/classify.py`, and `audit/results.md` has the tables for every build and class. The BFCL patch, its tests and `verify.py` are in `patch/`. `writeup/build.py` turns the outputs into this text, and the scripts in `writeup/figures/` draw the figures.
