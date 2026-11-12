"""Tests for audit/classify.py. Run: python -I -m unittest discover -s audit -v
Fixtures are the real shapes seen in saved outputs (arguments shortened)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as C  # noqa: E402

OK = '<tool_call>\n{"name": "ls", "arguments": {"a": false}}\n</tool_call>'


class TestClassify(unittest.TestCase):
    def check(self, text, cls, flags=()):
        c, fl = C.classify(text)
        self.assertEqual(c, cls, text)
        self.assertEqual(sorted(fl), sorted(flags), text)

    def test_ok(self):
        self.check(OK, "ok")
        self.check("Let me check first.\n\n" + OK, "ok")
        self.check(OK + '\n<tool_call>\n{"name": "pwd", "arguments": {}}\n</tool_call>', "ok")

    def test_text(self):
        self.check("I will now move the file.", "text")

    def test_unclosed_is_recoverable(self):
        # complete call, generation ended before </tool_call>
        self.check('<tool_call>\n{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}\n',
                   "unclosed", ["recoverable"])

    def test_unclosed_truncated_is_not_recoverable(self):
        self.check('<tool_call>\n{"name": "ls", "arguments": {"a": fa', "unclosed")

    def test_bad_newlines(self):
        self.check('<tool_call>{"name": "ls", "arguments": {"a": false}}</tool_call>', "bad_newlines", ["recoverable"])

    def test_keys_dropped(self):
        self.check('<tool_call>\n{"cd", "arguments": {"folder": "document"}}\n</tool_call>', "keys_dropped")
        self.check('<tool_call>\n{"close_ticket", {"ticket_id": 1}}\n</tool_call>', "keys_dropped")

    def test_schema_echo(self):
        # truncated copy of the tool description: invalid JSON
        self.check('<tool_call>\n{"name": "startEngine", "description": "This tool belongs to the vehicle\n</tool_call>',
                   "schema_echo")
        # complete copy: valid JSON, no "arguments"
        self.check('<tool_call>\n{"name": "startEngine", "description": "Start it.", "parameters": {}}\n</tool_call>',
                   "schema_echo")

    def test_bad_json(self):
        self.check('<tool_call>\n{"name": "startEngine",", "arguments": {"ignitionMode": "START"}}\n</tool_call>',
                   "bad_json")

    def test_no_name_and_bad_args(self):
        self.check('<tool_call>\n{"function": "ls", "arguments": {}}\n</tool_call>', "no_name")
        self.check('<tool_call>\n{"name": "ls", "arguments": "a=false"}\n</tool_call>', "bad_args")

    def test_not_object(self):
        self.check('<tool_call>\n["ls"]\n</tool_call>', "not_object")

    def test_untagged_call(self):
        self.check('I will run {"name": "ls", "arguments": {"a": false}} now.', "untagged_call")
        self.check('The format is {"name": <function-name>}.', "text")  # not a parseable call

    def test_bad_name(self):
        self.check('<tool_call>\n{"name": 5, "arguments": {}}\n</tool_call>', "bad_name")
        self.check('<tool_call>\n{"name": "f", "arguments": {}, "function": "f"}\n</tool_call>', "bad_name")

    def test_stray_opener(self):
        # the official regex swallows the stray opener into the first block; a tolerant parser finds the call
        self.check('<tool_call>\n{"name": "a", "arguments": {}}\n<tool_call>\n<tool_call>\n'
                   '{"name": "b", "arguments": {}}\n</tool_call>', "bad_json", ["recoverable"])

    def test_partial_drop(self):
        broken = '<tool_call>\n{"cd", "arguments": {"folder": "x"}}\n</tool_call>'
        self.check(OK + "\n" + broken, "ok", ["partial_drop"])

    def test_official_matches_bfcl(self):
        # mirror of QwenFCHandler._extract_tool_calls at commit 6ea5797
        self.assertEqual(C.official_calls(OK), [{"name": "ls", "arguments": {"a": False}}])
        self.assertEqual(C.official_calls(OK[:-len("</tool_call>")]), [])
        self.assertEqual(C.official_calls('<tool_call>\n{"cd", "arguments": {}}\n</tool_call>'), [])


if __name__ == "__main__":
    unittest.main()
