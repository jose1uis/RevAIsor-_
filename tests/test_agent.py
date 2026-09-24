"""End-to-end behavior and failure paths; no network or paid calls."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import agent
from llm_client import LLMError


class AgentTests(unittest.TestCase):
    def test_required_queries_and_variants(self):
        cases = [
            ("What role does Priya own?", "role_ownership", "GBR-1234"),
            ("What is the difference between GBR and ZGBR?", "role_comparison", "per-session"),
            ("Show me Priya's access review history.", "access_history", "2025-01-20"),
            ("Order a pizza for the office", "out_of_scope", "access-review"),
            ("What is the difference between Priya's GBR role and a ZGBR?", "combined", "GBR-1234"),
            ("what roles does priya own?", "role_ownership", "GBR-1234"),
            ("compare gbr and zgbr", "role_comparison", "per-session"),
            ("What roles does Fatima own?", "role_ownership", "ZGBR-0001"),
            ("Show Carlos's access review history", "access_history", "FLAGGED"),
        ]
        with patch("agent.generate_text", side_effect=AssertionError("Unexpected LLM call")):
            for query, intent, fragment in cases:
                with self.subTest(query=query):
                    result = agent.run_agent_detailed(query, mock=True)
                    self.assertEqual(result["routing"]["intent"], intent)
                    self.assertIn(fragment, result["answer"])
                    self.assertEqual(result["verification"]["status"], "SUCCESS", result)

    def test_refusals_call_neither_tools_nor_llm(self):
        with patch.dict(agent.TOOLS, {
            "role_tool": unittest.mock.Mock(side_effect=AssertionError("Unexpected lookup")),
            "document_tool": unittest.mock.Mock(side_effect=AssertionError("Unexpected lookup")),
        }), patch("agent.generate_text", side_effect=AssertionError("Unexpected LLM call")):
            for query in ["Order a pizza for Priya", "Revoke Priya's role", "What is our VPN policy?"]:
                with self.subTest(query=query):
                    result = agent.run_agent_detailed(query)
                    self.assertEqual(result["tool_outputs"], {})
                    self.assertEqual(result["verification"]["status"], "SUCCESS", result)

    def test_unknown_user_is_not_substituted(self):
        result = agent.run_agent_detailed("What role does Zoe own?", mock=True)
        self.assertEqual(result["answer"], agent.INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(result["verification"]["status"], "SUCCESS")
        self.assertNotIn("GBR-1234", json.dumps(result["tool_outputs"]))

    def test_known_and_unknown_people_are_both_acknowledged(self):
        for query in ["What roles do Priya and Morgan own?", "List roles for Priya and Morgan.",
                      "Show Morgan and Priya's access review history.", "Compare Morgan and Priya roles"]:
            with self.subTest(query=query):
                result = agent.run_agent_detailed(query, mock=True)
                self.assertIn("Morgan", result["routing"]["entities"]["unknown_users"])
                self.assertIn("Morgan", result["answer"])
                self.assertIn("not found", result["answer"])
                self.assertEqual(result["verification"]["status"], "SUCCESS", result)

    def test_mutation_wrappers_remain_read_only(self):
        for query in ["I would like you to grant Priya a GBR role.",
                      "Please give Priya a GBR role.",
                      "Can you tell me Priya's roles and then revoke her access?"]:
            with self.subTest(query=query):
                result = agent.run_agent_detailed(query)
                self.assertEqual(result["routing"]["intent"], "unsupported_access")
                self.assertEqual(result["tool_outputs"], {})

    def test_role_type_ownership_walks_to_actual_owners(self):
        result = agent.run_agent_detailed("Who owns the ZGBR role?", mock=True)
        self.assertEqual(result["routing"]["intent"], "role_ownership")
        self.assertEqual(set(result["routing"]["entities"]["users"]), {"Carlos", "Fatima"})
        self.assertEqual(result["verification"]["status"], "SUCCESS")

    def test_both_tools_are_executed(self):
        result = agent.run_agent_detailed("Compare Priya's role with ZGBR", mock=True)
        self.assertEqual(set(result["tool_outputs"]), {"role_tool", "document_tool"})
        self.assertIn("[Role-DB]", result["answer"])
        self.assertIn("[Policy-Doc-", result["answer"])

    def test_synthesis_uses_raw_evidence_and_system_prompt(self):
        with patch("agent.generate_text", return_value="Priya owns GBR-1234. [Role-DB]") as generate:
            result = agent.run_agent_detailed("What role does Priya own?")
        self.assertEqual(result["verification"]["status"], "SUCCESS")
        prompt, payload = generate.call_args.args
        self.assertIn("Access Review", prompt)
        self.assertEqual(payload["tool_outputs"], result["tool_outputs"])
        self.assertIn("paths", payload["graph_context"])

    def test_ownership_synthesis_cannot_see_uncalled_policy_evidence(self):
        with patch("agent.generate_text", return_value="Priya owns GBR-1234. [Role-DB]") as generate:
            result = agent.run_agent_detailed("What role does Priya own?")
        payload = generate.call_args.args[1]
        self.assertEqual(payload["allowed_citations"], ["Role-DB"])
        self.assertNotIn("Policy-Doc-", json.dumps(payload))
        self.assertNotIn("per-session", json.dumps(payload))
        self.assertIn("owns", json.dumps(payload["graph_context"]["paths"]))
        # Full graph traversal remains visible for reviewers, separate from synthesis.
        self.assertIn("Policy-Doc-", json.dumps(result["routing"]["graph_context"]))

    def test_combined_synthesis_includes_only_retrieved_policy_sources(self):
        query = "Compare Priya's GBR role with ZGBR"
        decision = agent.route(query)
        outputs = agent.collect_tool_outputs(query, decision)
        payload = agent.synthesis_payload(query, decision, outputs)
        expected = {"Role-DB"} | {doc["source_id"] for doc in outputs["document_tool"]}
        self.assertEqual(set(payload["allowed_citations"]), expected)
        self.assertEqual(payload["tool_outputs"], outputs)
        for path in payload["graph_context"]["paths"]:
            if "documented_by" in path:
                self.assertIn(path[-1], expected)

    def test_policy_synthesis_does_not_receive_graph_user_records(self):
        query = "What is the difference between GBR and ZGBR?"
        decision = agent.route(query)
        outputs = agent.collect_tool_outputs(query, decision)
        payload = agent.synthesis_payload(query, decision, outputs)
        self.assertNotIn("Role-DB", payload["allowed_citations"])
        self.assertNotIn("users", payload["graph_context"])
        self.assertNotIn("role_instances", payload["graph_context"])

    def test_unverified_generation_is_withheld(self):
        for candidate in ["Priya owns GBR-8888. [Role-DB]", "Priya owns GBR-1234."]:
            with self.subTest(candidate=candidate), patch("agent.generate_text", return_value=candidate):
                result = agent.run_agent_detailed("What role does Priya own?")
                self.assertEqual(result["verification"]["status"], "FAIL")
                self.assertEqual(result["answer"], agent.VERIFICATION_FAILURE_RESPONSE)
                self.assertNotIn(candidate, json.dumps(result))

    def test_service_error_is_actionable(self):
        with patch("agent.generate_text", side_effect=LLMError("Service unavailable")):
            result = agent.run_agent_detailed("What role does Priya own?")
        self.assertEqual(result["error"], "Service unavailable")
        self.assertEqual(result["verification"]["status"], "FAIL")

    def test_missing_key_does_not_silently_use_mock(self):
        with patch.dict(os.environ, {}, clear=True):
            result = agent.run_agent_detailed("What role does Priya own?")
        self.assertIn("OPENAI_API_KEY", result["error"])
        self.assertEqual(result["mode"], "live_llm")

    def test_json_cli_and_exit_codes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = agent.main(["What role does Priya own?", "--mock", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["verification"]["status"], "SUCCESS")
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.main(["What role does Priya own?"]), 2)

    def test_documented_entry_point_from_subprocess(self):
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "agent.py", "What role does Priya own?", "--mock"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for section in ["ROUTING", "TOOL OUTPUT", "AGENT ANSWER", "VERIFICATION"]:
            self.assertIn(section, result.stdout)

    def test_starter_function_returns_a_string(self):
        self.assertIsInstance(agent.run_agent("What role does Priya own?", mock=True), str)

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            agent.run_agent("  ", mock=True)


if __name__ == "__main__":
    unittest.main()
