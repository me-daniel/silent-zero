# BFCL patch

`bfcl-silent-zero.patch` applies to [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (the latest `main` as of November 2026). It touches five files in `berkeley-function-call-leaderboard/bfcl_eval/`:

| File | Change |
|---|---|
| `model_handler/local_inference/tool_call_parsing.py` | **New.** Upstream-identical parser (a linear-time equivalent of upstream's regex, fuzz-tested against it), a classifier for failed calls, and the opt-in modes |
| `model_handler/local_inference/qwen_fc.py` | `QwenFCHandler` uses the module; adds `diagnose_tool_calls` |
| `model_handler/base_handler.py` | Keeps the per-step diagnosis in the inference log |
| `_llm_response_generation.py` | Records active opt-in modes in each result (only when one is set) |
| `eval_checker/eval_runner.py` | Adds a `tool_call_parse` field to failed score entries; refuses to score multi-turn and agentic results with different modes than they were generated with |

## Default behaviour: unchanged scores

With no environment variable set, the parser returns exactly what upstream returns. `patch/verify.py` re-scores saved results and compares them with the original score files. For 23 build × category combinations (simple_python, live_simple, multi_turn_base), every header and every failed entry (id, valid, error_type) is identical. The only difference is the added `tool_call_parse` field, for example:

```json
"tool_call_parse": {"class": "unclosed", "flags": ["recoverable"]}
"tool_call_parse": {"step_classes": {"ok": 4, "text": 3, "keys_dropped": 1}}
"tool_call_parse": {"crash": "Error during inference: 'arguments'"}
```

The classes are defined in `tool_call_parsing.py` and match `audit/classify.py`; a test checks that the two agree.

## Opt-in modes

Set any of these to `1` (or `true`, `yes`, `on`). Each one changes results, so report it separately from official scores.

| Variable | Effect | Use at |
|---|---|---|
| `SILENT_ZERO_SAFE_HISTORY` | Only well-formed calls (a non-empty string `name`, object `arguments`, no `function` key) enter the chat history and get executed; malformed ones are skipped. This prevents the crash shapes we know of in `_format_prompt`, which turn a whole multi-turn task into `inference_error`. | generation and evaluation |
| `SILENT_ZERO_TOLERANT` | Also accepts a final unclosed `<tool_call>` block and loose whitespace around the JSON. | generation and single-turn evaluation |
| `SILENT_ZERO_REPAIR_KEYS` | Also repairs `{"fn", "arguments": {...}}` and `{"fn", {...}}` into `{"name": "fn", "arguments": {...}}`. The result must have exactly these two keys and no duplicate keys. This guesses structure. | generation and single-turn evaluation |

**Multi-turn: generate and evaluate with the same settings of all three variables.** They decide during generation which calls get executed, so re-scoring old multi-turn outputs with different settings would credit trajectories that never happened. The evaluator raises an error in that case.

## Apply, test, verify

```sh
git clone https://github.com/ShishirPatil/gorilla.git vendor/gorilla
git -C vendor/gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
git -C vendor/gorilla apply ../../patch/bfcl-silent-zero.patch
uv venv -p 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -e vendor/gorilla/berkeley-function-call-leaderboard

.venv/bin/python -m unittest discover -s patch/tests -v
.venv/bin/python patch/verify.py --raw <saved outputs> --out <dir> [--mode default|tolerant|tolerant+repair]
```

The tests cover:
- upstream-identical extraction, including a 20,000-case fuzz test of the linear scan against upstream's regex;
- linear running time on degenerate input;
- each opt-in mode, and repair rejecting ambiguous input;
- the six crash shapes (missing `name`, missing `arguments`, a `"function"` key that the prompt template unwraps to a string, and a `name` that is a number or `null`);
- that safe history prevents them and keeps the good calls of a mixed step;
- the evaluator guard and the recorded modes.
