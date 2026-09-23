"""Judge schema and independence checks; live judge calls are never faked as PASS."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from evaluator import evaluate, main
from llm_client import LLMError


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.verdict = {"grounding_ok": True, "citation_ok": True, "relevance_ok": True,
                        "score": "PASS", "reasoning": "Ownership matches the supplied record."}

    def test_judge_receives_original_evidence_in_independent_call(self):
        evidence = {"role_tool": [{"Priya": {"owns": ["GBR-1234"]}}]}
        answer = "Priya owns GBR-1234. [Role-DB]"
        with patch("evaluator.generate_text", return_value=json.dumps(self.verdict)) as judge:
            verdict = evaluate("What role does Priya own?", evidence, answer)
        self.assertEqual(verdict, self.verdict)
        payload = judge.call_args.args[1]
        self.assertEqual(payload["tool_outputs"], evidence)
        self.assertEqual(payload["agent_answer"], answer)
        self.assertEqual(judge.call_args.kwargs["model_env"], "OPENAI_EVALUATOR_MODEL")
        self.assertIn("schema", judge.call_args.kwargs)

    def test_invalid_or_contradictory_verdict_rejected(self):
        variants = ["not JSON", "[]", "{}"]
        for changes in [{"grounding_ok": "true"}, {"score": "FAIL"}, {"reasoning": ""}]:
            variants.append(json.dumps({**self.verdict, **changes}))
        for response in variants:
            with self.subTest(response=response), patch("evaluator.generate_text", return_value=response):
                with self.assertRaises(LLMError):
                    evaluate("What role does Priya own?", {}, "answer")

    def test_failed_judgment_is_preserved(self):
        self.verdict.update(citation_ok=False, score="FAIL")
        with patch("evaluator.generate_text", return_value=json.dumps(self.verdict)):
            self.assertEqual(evaluate("query", {}, "answer")["score"], "FAIL")

    def test_malformed_saved_execution_has_clean_error(self):
        for verification in [None, [], "SUCCESS"]:
            saved = json.dumps({"verification": verification})
            output = io.StringIO()
            with patch("evaluator.Path.read_text", return_value=saved), contextlib.redirect_stdout(output):
                self.assertEqual(main(["--result-file", "unused.json"]), 2)
            self.assertIn("error", json.loads(output.getvalue()))


if __name__ == "__main__":
    unittest.main()
