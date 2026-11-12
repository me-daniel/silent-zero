"""Tests for the silent-zero BFCL patch. Run from the repo root with the project venv:
    .venv/bin/python -m unittest discover -s patch/tests -v
Requires vendor/gorilla on branch silent-zero, installed editable (see the README)."""
import json
import os
import re
import sys
import time
import types
import unittest
from unittest import mock

from bfcl_eval import _llm_response_generation as gen
from bfcl_eval.eval_checker import eval_runner
from bfcl_eval.model_handler.local_inference import tool_call_parsing as P
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "audit"))
import classify as audit_classify  # noqa: E402

OK = '<tool_call>\n{"name": "ls", "arguments": {"a": false}}\n</tool_call>'
FIXTURES = [
    OK,
    "Let me look.\n\n" + OK,
    "No call here.",
    '<tool_call>\n{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}\n',  # unclosed
    '<tool_call>{"name": "ls", "arguments": {"a": false}}</tool_call>',  # bad newlines
    '<tool_call>\n{"cd", "arguments": {"folder": "document"}}\n</tool_call>',  # keys dropped
    '<tool_call>\n{"close_ticket", {"ticket_id": 1}}\n</tool_call>',  # keys dropped, both keys
    '<tool_call>\n{"name": "startEngine", "description": "This tool belongs to\n</tool_call>',  # schema echo
    '<tool_call>\n{"name": "startEngine", "description": "x", "parameters": {}}\n</tool_call>',  # schema echo, valid
    '<tool_call>\n{"name": "startEngine",", "arguments": {}}\n</tool_call>',  # bad json
    '<tool_call>\n{"function": "ls", "arguments": {}}\n</tool_call>',  # no name
    '<tool_call>\n["ls"]\n</tool_call>',  # not an object
    OK + '\n<tool_call>\n{"cd", "arguments": {"folder": "x"}}\n</tool_call>',  # partial drop
    'I will call {"name": "ls", "arguments": {"a": false}} now.',  # untagged call
    '<tool_call>\n{"name": 5, "arguments": {}}\n</tool_call>',  # bad name
    '<tool_call>\n{"name": "f", "arguments": {}, "function": "f"}\n</tool_call>',  # bad name ("function" key)
    '<tool_call>\n{"name": "lockDoors", "arguments": {}}\n<tool_call>\n<tool_call>\n{"name": "startEngine", '
    '"arguments": {}}\n</tool_call>',  # stray openers (real IQ3_XXS shape)
    OK + '\n<tool_call>\n{"x": 1}\n</tool_call>',  # good call plus a call without name
]


def upstream_extract(input_string):
    """Verbatim copy of QwenFCHandler._extract_tool_calls at upstream commit 6ea5797."""
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, input_string, re.DOTALL)
    result = []
    for match in matches:
        try:
            match = json.loads(match)
            result.append(match)
        except Exception as e:
            pass
    return result


def env(**modes):
    return mock.patch.dict(os.environ, {P.ENV[m]: "1" for m in modes}, clear=False)


def clean_env():
    return mock.patch.dict(os.environ, {v: "" for v in P.ENV.values()})


def fake_response(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(text=text)],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def handler():
    return QwenFCHandler.__new__(QwenFCHandler)  # no server or tokenizer needed for these methods


class DefaultIsUpstream(unittest.TestCase):
    def test_extract_identical_to_upstream(self):
        with clean_env():
            for t in FIXTURES:
                self.assertEqual(QwenFCHandler._extract_tool_calls(t), upstream_extract(t), t)

    def test_classifier_matches_audit(self):
        for t in FIXTURES:
            self.assertEqual(P.classify(t), audit_classify.classify(t), t)

    def test_official_blocks_equals_regex(self):
        import random
        rng = random.Random(0)
        tokens = ["<tool_call>", "</tool_call>", "\n", "{", "}", '"name"', ":", " ", "x", "<tool_call>\n",
                  "\n</tool_call>", "\n\n"]
        pattern = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
        for _ in range(20000):
            t = "".join(rng.choice(tokens) for _ in range(rng.randint(0, 25)))
            self.assertEqual(P.official_blocks(t), pattern.findall(t), repr(t))
            self.assertEqual(audit_classify.official_blocks(t), pattern.findall(t), repr(t))

    def test_new_classes(self):
        self.assertEqual(P.classify(FIXTURES[13])[0], "untagged_call")
        self.assertEqual(P.classify(FIXTURES[14])[0], "bad_name")
        self.assertEqual(P.classify(FIXTURES[15])[0], "bad_name")

    def test_linear_time(self):
        for text in ("<tool_call>" + " " * 20000 + "x", "<tool_call>\n{" * 20000, '{"name":' * 20000):
            t0 = time.perf_counter()
            P.classify(text)
            P.extract_tool_calls(text, frozenset({"tolerant", "repair_keys"}))
            self.assertLess(time.perf_counter() - t0, 1.0, text[:30])


class OptInModes(unittest.TestCase):
    def test_tolerant(self):
        with clean_env(), env(tolerant=True):
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[3]),
                             [{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}])
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[4]), [{"name": "ls", "arguments": {"a": False}}])
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[5]), [])  # not repaired without repair_keys

    def test_repair_keys(self):
        with clean_env(), env(repair_keys=True):
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[5]),
                             [{"name": "cd", "arguments": {"folder": "document"}}])
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[6]),
                             [{"name": "close_ticket", "arguments": {"ticket_id": 1}}])
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[3]), [])  # unclosed needs tolerant
            self.assertEqual(QwenFCHandler._extract_tool_calls(FIXTURES[9]), [])  # other bad JSON stays dropped

    def test_repair_rejects_ambiguous(self):
        for blob in ('{"name", "arguments": {"a": 1}}',  # function literally called "name"
                     '{"arguments", {"a": 1}}',
                     '{"cd", "arguments": {"folder": "x"}, "name": "rm"}',  # duplicate key would win
                     '{"cd", {"folder": "x"}, "x": 2}',  # extra key
                     '{"cd", "arguments": {"a": 1, "a": 2}}'):  # duplicate inside arguments
            self.assertIsNone(P.repair_keys(blob), blob)

    def test_tolerant_stray_opener_and_superset(self):
        modes = frozenset({"tolerant"})
        self.assertEqual(P.extract_tool_calls(FIXTURES[16], modes), [{"name": "startEngine", "arguments": {}}])
        self.assertEqual(P.official_calls(FIXTURES[16]), [])
        for t in FIXTURES:  # every well-formed call the official parser finds, tolerant finds too
            wf = [c for c in P.official_calls(t) if P.well_formed(c)]
            tol = P.extract_tool_calls(t, modes)
            self.assertTrue(all(c in tol for c in wf), t)

    def test_decode_ast_tolerant(self):
        h = handler()
        with clean_env():
            self.assertEqual(h.decode_ast(FIXTURES[3], "python", False), [])
        with clean_env(), env(tolerant=True):
            self.assertEqual(h.decode_ast(FIXTURES[3], "python", False),
                             [{"algebra.quadratic_roots": {"a": 1, "b": -3, "c": 2}}])

    def test_modes_recorded(self):
        with clean_env(), env(tolerant=True, safe_history=True):
            self.assertEqual(P.active_modes(), {"tolerant", "safe_history"})
        with clean_env():
            self.assertEqual(P.active_modes(), set())
        for v in ("true", "ON", " yes "):
            with mock.patch.dict(os.environ, {P.ENV["tolerant"]: v}):
                self.assertEqual(P.active_modes(), {"tolerant"}, v)

    def test_modes_written_to_result(self):
        fake = types.SimpleNamespace(inference=lambda *a: ("out", {}))
        case = {"id": "x", "function": []}
        with clean_env():
            self.assertNotIn("silent_zero_modes", gen.multi_threaded_inference(fake, case, False, True))
        with clean_env(), env(tolerant=True):
            self.assertEqual(gen.multi_threaded_inference(fake, case, False, True)["silent_zero_modes"], ["tolerant"])


class CrashPath(unittest.TestCase):
    """A parsed call without "name"/"arguments" enters the history and crashes the next prompt build."""

    def history_after(self, text):
        h = handler()
        data = h._parse_query_response_prompting(fake_response(text))
        msgs = [{"role": "user", "content": "do it"}, data["model_responses_message_for_chat_history"],
                {"role": "tool", "content": "{}"}]
        return h, msgs, data

    NO_NAME = '<tool_call>\n{"tool": "ls", "arguments": {}}\n</tool_call>'
    CRASHES = [  # (text, exception, key) as seen in saved outputs or reachable from them
        (NO_NAME, KeyError, "name"),
        (FIXTURES[8], KeyError, "arguments"),  # schema echo: has "name", no "arguments"
        (FIXTURES[10], TypeError, None),  # {"function": "ls", ...}: the template unwraps "function" to a string
        (FIXTURES[14], TypeError, None),  # "name": 5, string concatenation fails
        ('<tool_call>\n{"name": null, "arguments": {}}\n</tool_call>', TypeError, None),
        (FIXTURES[15], TypeError, None),  # has "name", but the "function" key is unwrapped first
    ]

    def test_upstream_crashes(self):
        with clean_env():
            for text, exc, key in self.CRASHES:
                h, msgs, _ = self.history_after(text)
                with self.assertRaises(exc) as cm:
                    h._format_prompt(msgs, [])
                if key:
                    self.assertEqual(cm.exception.args[0], key)

    def test_safe_history_does_not_crash(self):
        with clean_env(), env(safe_history=True):
            for text, _, _ in self.CRASHES:
                h, msgs, data = self.history_after(text)
                h._format_prompt(msgs, [])  # must not raise
                self.assertNotIn("tool_calls", data["model_responses_message_for_chat_history"])
                self.assertEqual(h.decode_execute(text, has_tool_call_tag=False), [])

    def test_safe_history_keeps_good_calls(self):
        text = FIXTURES[17]  # one good call plus {"x": 1}
        with clean_env():
            with self.assertRaises(KeyError):
                handler().decode_execute(text, has_tool_call_tag=False)
        with clean_env(), env(safe_history=True):
            h, msgs, data = self.history_after(text)
            self.assertEqual(data["model_responses_message_for_chat_history"]["tool_calls"],
                             [{"name": "ls", "arguments": {"a": False}}])
            self.assertEqual(h.decode_execute(text, has_tool_call_tag=False), ["ls(a=False)"])
            h._format_prompt(msgs, [])

    def test_diagnosis_in_parse_result(self):
        with clean_env():
            _, _, data = self.history_after(FIXTURES[5])
            self.assertEqual(data["tool_call_parse"], {"class": "keys_dropped", "flags": []})


class Evaluator(unittest.TestCase):
    def test_mode_guard(self):
        with clean_env():
            eval_runner._check_multi_turn_parser_modes({"id": "a"})
            with self.assertRaises(ValueError):
                eval_runner._check_multi_turn_parser_modes({"id": "a", "silent_zero_modes": ["tolerant"]})
            with self.assertRaises(ValueError):  # safe history changes decode_execute, so it is guarded too
                eval_runner._check_multi_turn_parser_modes({"id": "a", "silent_zero_modes": ["safe_history"]})
        with clean_env(), env(safe_history=True):
            eval_runner._check_multi_turn_parser_modes({"id": "a", "silent_zero_modes": ["safe_history"]})
        with clean_env(), env(tolerant=True):
            with self.assertRaises(ValueError):
                eval_runner._check_multi_turn_parser_modes({"id": "a"})
            eval_runner._check_multi_turn_parser_modes({"id": "a", "silent_zero_modes": ["tolerant"]})

    def test_diagnosis_attached(self):
        e = {"valid": False}
        eval_runner._attach_tool_call_diagnosis(handler(), e, FIXTURES[3])
        self.assertEqual(e["tool_call_parse"], {"class": "unclosed", "flags": ["recoverable"]})
        e = {"valid": False}
        eval_runner._attach_tool_call_diagnosis(handler(), e, "Error during inference: 'name'")
        self.assertEqual(e["tool_call_parse"], {"crash": "Error during inference: 'name'"})
        e = {"valid": False}
        eval_runner._attach_tool_call_diagnosis(handler(), e, [[OK, "done"], [FIXTURES[5]]])
        self.assertEqual(e["tool_call_parse"], {"step_classes": {"ok": 1, "text": 1, "keys_dropped": 1}})
        self.assertFalse(e["valid"])


if __name__ == "__main__":
    unittest.main()
