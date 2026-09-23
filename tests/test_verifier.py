"""Negative checks cover invented IDs and citations to unrelated evidence."""

import unittest
from unittest.mock import patch

from router import route as actual_route
from tools import document_tool, role_tool
from verifier import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    UNSUPPORTED_ACCESS_RESPONSE,
    verify_response,
)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.query = "What role does Priya own?"
        self.evidence = {"role_tool": [role_tool("Priya")]}
        self.route = patch("verifier.route", return_value={
            "intent": "role_ownership",
            "entities": {"users": ["Priya"], "role_types": []},
            "graph_context": {"role_types": {}},
            "tools_to_call": ["role_tool"],
        })
        self.routing = self.route.start()
        self.addCleanup(self.route.stop)

    def verify(self, answer, evidence=None):
        return verify_response(self.query, answer, self.evidence if evidence is None else evidence)

    def assertFailedCheck(self, result, name):
        self.assertEqual(result["status"], "FAIL")
        check = next(item for item in result["checks"] if item["name"] == name)
        self.assertFalse(check["passed"], check["detail"])

    def test_cited_ownership(self):
        self.assertEqual(self.verify("Priya owns GBR-1234. [Role-DB]")["status"], "SUCCESS")

    def test_two_argument_api_uses_scoped_static_tools(self):
        result = verify_response(self.query, "Priya owns GBR-1234. [Role-DB]")
        self.assertEqual(result["status"], "SUCCESS")

    def test_no_citation_fails(self):
        self.assertFailedCheck(self.verify("Priya owns GBR-1234."), "citation_present")

    def test_invented_role_fails(self):
        self.assertFailedCheck(self.verify("Priya owns GBR-8888. [Role-DB]"), "no_invented_roles")

    def test_invented_role_with_letters_fails(self):
        self.assertFailedCheck(self.verify("Priya owns GBR-FAKE. [Role-DB]"), "no_invented_roles")

    def test_invented_other_numeric_role_prefix_fails(self):
        self.assertFailedCheck(self.verify("Priya owns ADMIN-777. [Role-DB]"), "no_invented_roles")

    def test_role_ids_in_brackets_are_checked(self):
        self.assertFailedCheck(self.verify("Priya owns [GBR-8888]. [Role-DB]"), "no_invented_roles")

    def test_known_but_unrelated_role_fails(self):
        result = self.verify("Priya owns ZGBR-5678. [Role-DB]")
        self.assertFailedCheck(result, "roles_in_retrieved_evidence")

    def test_swapped_owners_with_both_users_retrieved_fails(self):
        evidence = {"role_tool": [role_tool("Priya"), role_tool("Carlos")]}
        result = self.verify("Priya owns ZGBR-5678. Carlos owns GBR-1234. [Role-DB]", evidence)
        self.assertFailedCheck(result, "ownership_matches")

    def test_correct_owners_in_one_sentence(self):
        evidence = {"role_tool": [role_tool("Priya"), role_tool("Carlos")]}
        result = self.verify("Priya owns GBR-1234 and Carlos owns ZGBR-5678. [Role-DB]", evidence)
        self.assertEqual(result["status"], "SUCCESS")

    def test_negated_known_role_does_not_become_an_ownership_claim(self):
        evidence = {"role_tool": [role_tool("Priya"), role_tool("Carlos")]}
        result = self.verify(
            "Priya owns GBR-1234, not ZGBR-5678. Carlos owns ZGBR-5678. [Role-DB]", evidence)
        self.assertEqual(result["status"], "SUCCESS")

    def test_negation_does_not_hide_an_invented_role(self):
        result = self.verify("Priya owns GBR-1234, not GBR-8888. [Role-DB]")
        self.assertFailedCheck(result, "no_invented_roles")

    def test_negation_does_not_hide_an_unretrieved_role(self):
        result = self.verify("Priya owns GBR-1234, not ZGBR-5678. [Role-DB]")
        self.assertFailedCheck(result, "roles_in_retrieved_evidence")

    def test_later_negation_does_not_erase_a_false_positive_claim(self):
        evidence = {"role_tool": [role_tool("Priya"), role_tool("Carlos")]}
        result = self.verify("Priya owns ZGBR-5678 and not ZGBR-5678. [Role-DB]", evidence)
        self.assertFailedCheck(result, "ownership_matches")

    def test_two_argument_false_premise_preserves_the_explicit_user(self):
        self.routing.side_effect = actual_route
        query = "Does Priya own ZGBR-5678?"
        answer = "Priya owns GBR-1234, not ZGBR-5678. Carlos owns ZGBR-5678. [Role-DB]"
        with patch("verifier.role_tool", wraps=role_tool) as lookup:
            result = verify_response(query, answer)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(lookup.call_args_list[0].args, ("Priya",))
        self.assertFailedCheck(verify_response(query, "Priya owns ZGBR-5678. [Role-DB]"),
                               "ownership_matches")

    def test_known_but_unretrieved_document_citation_fails(self):
        result = self.verify("Priya owns GBR-1234. [Policy-Doc-GBR-001]")
        self.assertFailedCheck(result, "citations_retrieved")

    def test_unknown_citation_fails_even_with_valid_citation(self):
        result = self.verify("Priya owns GBR-1234. [Role-DB] [Policy-Doc-FAKE-999]")
        self.assertFailedCheck(result, "citations_retrieved")

    def test_policy_citations_are_not_role_ids(self):
        self.routing.return_value["intent"] = "role_comparison"
        evidence = {"document_tool": document_tool("Compare GBR and ZGBR")}
        result = self.verify(
            "GBR provides persistent access. [Policy-Doc-GBR-001] "
            "ZGBR requires per-session verification. [Policy-Doc-ZGBR-002]", evidence)
        self.assertEqual(result["status"], "SUCCESS")

    def test_combined_answer_requires_both_source_kinds(self):
        self.routing.return_value["intent"] = "combined"
        evidence = {**self.evidence, "document_tool": document_tool("Compare GBR and ZGBR")}
        result = self.verify("Priya owns GBR-1234. GBR is persistent. [Role-DB]", evidence)
        self.assertFailedCheck(result, "required_source_citations")

    def test_history_with_grounded_date(self):
        self.routing.return_value["intent"] = "access_history"
        result = self.verify("Priya's review on 2024-01-15 was APPROVED by Alice. [Role-DB]")
        self.assertEqual(result["status"], "SUCCESS")

    def test_history_with_invented_date_fails(self):
        result = self.verify("Priya's review on 2099-01-01 was approved. [Role-DB]")
        self.assertFailedCheck(result, "dates_in_retrieved_evidence")

    def test_lookup_error_can_be_cited(self):
        evidence = {"role_tool": [role_tool("Unknown Person")]}
        result = self.verify("User Unknown Person was not found in Role DB. [Role-DB]", evidence)
        self.assertEqual(result["status"], "SUCCESS")

    def test_insufficient_evidence_needs_no_fake_citation(self):
        self.assertEqual(self.verify(INSUFFICIENT_EVIDENCE_RESPONSE, {})["status"], "SUCCESS")

    def test_out_of_scope_refusal_needs_no_tools_or_citations(self):
        self.routing.return_value.update(intent="out_of_scope", tools_to_call=[])
        with patch("verifier.role_tool") as roles, patch("verifier.document_tool") as documents:
            result = verify_response("Order a pizza", OUT_OF_SCOPE_RESPONSE)
        self.assertEqual(result["status"], "SUCCESS")
        roles.assert_not_called()
        documents.assert_not_called()

    def test_out_of_scope_answering_or_tool_calls_fail(self):
        self.routing.return_value.update(intent="out_of_scope", tools_to_call=[])
        self.assertFailedCheck(self.verify("Ordered pizza!", {}), "scope_refusal")
        self.assertFailedCheck(self.verify(OUT_OF_SCOPE_RESPONSE), "no_unnecessary_tools")

    def test_unsupported_access_refusal(self):
        self.routing.return_value.update(intent="unsupported_access", tools_to_call=[])
        self.assertEqual(self.verify(UNSUPPORTED_ACCESS_RESPONSE, {})["status"], "SUCCESS")

    def test_explicit_empty_evidence_does_not_fall_back_to_tools(self):
        with patch("verifier.role_tool") as roles, patch("verifier.document_tool") as documents:
            self.assertFailedCheck(self.verify("Priya owns GBR-1234. [Role-DB]", {}),
                                   "roles_in_retrieved_evidence")
        roles.assert_not_called()
        documents.assert_not_called()

    def test_empty_response_fails(self):
        self.assertFailedCheck(self.verify(""), "response_present")


if __name__ == "__main__":
    unittest.main()
