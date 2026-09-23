"""Routing tests exercise query variations and scope/identity boundaries."""

import unittest
from unittest.mock import patch

import router


class RouterTests(unittest.TestCase):
    def test_ownership_variations(self):
        for query in (
            "What role does Priya own?", "which roles does priya have?",
            "List PRIYA's roles", "Show Priya\u2019s role ownership",
        ):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], "role_ownership")
                self.assertEqual(decision["entities"]["users"], ["Priya"])
                self.assertEqual(decision["tools_to_call"], ["role_tool"])

    def test_entity_word_boundaries(self):
        self.assertEqual(router.extract_entities("Priyanka uses ZGBRX and GBRX"),
                         {"users": [], "role_types": []})
        self.assertEqual(router.extract_entities("zgbr")["role_types"], ["ZGBR"])

    def test_full_role_name_is_not_substring_of_zero_trust_name(self):
        result = router.extract_entities("Explain a Zero-Trust Global Business Role")
        self.assertEqual(result["role_types"], ["ZGBR"])

    def test_policy_comparison(self):
        for query in ("What is the difference between GBR and ZGBR?",
                      "compare gbr with zgbr", "How do GBR and ZGBR differ?"):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], "role_comparison")
                self.assertEqual(decision["tools_to_call"], ["document_tool"])

    def test_history_and_multiple_read_intents(self):
        self.assertEqual(router.route("Show me Priya's access review history")["intent"],
                         "access_history")
        decision = router.route("Which roles does Carlos own and show his access review history")
        self.assertEqual(decision["intent"], "ownership_history")
        self.assertEqual(decision["tools_to_call"], ["role_tool"])

    def test_user_comparison_selects_both_sources(self):
        for query in ("What is the difference between Priya's GBR role and a ZGBR?",
                      "Compare Priya and Carlos roles"):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], "combined")
                self.assertEqual(decision["tools_to_call"], ["role_tool", "document_tool"])
                self.assertTrue(decision["graph_context"]["paths"])
                self.assertIn("GBR", decision["graph_context"]["role_types"])

    def test_graph_infers_user_types_for_comparison(self):
        result = router.route("Compare Priya and Carlos roles")
        self.assertEqual(result["entities"]["role_types"], [])
        self.assertEqual(set(result["graph_context"]["role_types"]), {"GBR", "ZGBR"})

    def test_unknown_user_is_retained_and_never_replaced(self):
        for query, intent in (("What role does Morgan own?", "role_ownership"),
                              ("Show Morgan\u2019s access review history", "access_history"),
                              ("List roles for Morgan", "role_ownership")):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], intent)
                self.assertEqual(decision["entities"]["users"], [])
                self.assertEqual(decision["entities"]["unknown_users"], ["Morgan"])

    def test_generic_users_are_not_invented_names(self):
        self.assertNotIn("unknown_users", router.extract_entities("What roles do users own?"))

    def test_unknown_user_comparison_requests_missing_user_evidence(self):
        decision = router.route("Compare Morgan's GBR role and ZGBR")
        self.assertEqual(decision["entities"]["unknown_users"], ["Morgan"])
        self.assertEqual(decision["tools_to_call"], ["role_tool", "document_tool"])

    def test_unsupported_mutations_have_no_tool_plan(self):
        for query in ("Grant Priya a ZGBR role", "Please revoke Carlos's access",
                      "Can you assign a GBR to Priya?", "I need you to change access for Carlos"):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], "unsupported_access")
                self.assertEqual(decision["tools_to_call"], [])

    def test_policy_verbs_are_not_mutation_requests(self):
        decision = router.route("Does GBR grant persistent access?")
        self.assertEqual(decision["intent"], "role_definition")
        self.assertEqual(decision["tools_to_call"], ["document_tool"])

    def test_names_alone_do_not_make_an_unrelated_request_in_scope(self):
        for query in ("Order a pizza for the office", "Order pizza for Priya", "", "Review this movie"):
            with self.subTest(query=query):
                decision = router.route(query)
                self.assertEqual(decision["intent"], "out_of_scope")
                self.assertEqual(decision["tools_to_call"], [])

    def test_unsupported_access_question_is_separate_from_out_of_scope(self):
        decision = router.route("When will the vendor VPN access expire?")
        self.assertEqual(decision["intent"], "unsupported_access")
        self.assertEqual(decision["tools_to_call"], [])

    def test_known_role_id_resolves_owners_but_unknown_id_does_not(self):
        known = router.route("Who owns gbr-1234?")
        self.assertEqual(known["entities"]["users"], ["Priya"])
        self.assertEqual(known["entities"]["role_ids"], ["GBR-1234"])
        unknown = router.route("Who owns GBR-5555?")
        self.assertEqual(unknown["entities"]["users"], [])
        self.assertEqual(unknown["entities"]["role_ids"], ["GBR-5555"])

    def test_new_graph_user_is_detected_without_router_changes(self):
        with patch.dict(router.KG["users"], {"Mei": {"owns": [], "review_history": []}}):
            entities = router.extract_entities("Which roles does mei have?")
        self.assertEqual(entities["users"], ["Mei"])
        self.assertNotIn("unknown_users", entities)

    def test_tool_plan_is_not_mutable_shared_state(self):
        router.select_tools("role_ownership").append("document_tool")
        self.assertEqual(router.select_tools("role_ownership"), ["role_tool"])
        self.assertEqual(router.select_tools("unrecognized"), [])


if __name__ == "__main__":
    unittest.main()
