"""Regression tests for starter-data reconciliation and evidence lookup."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from knowledge_graph import KG, find_role_types, get_role_instances, graph_walk
from tools import document_tool, role_tool


# These expected records pin the assessment's source data, including mixed roles
# and pending reviews. They are independent of the production KG values.
STARTER_USERS = {
    "Priya": {
        "owns": ["GBR-1234"],
        "role_type": "GBR",
        "department": "Engineering",
        "review_history": [
            {"date": "2024-01-15", "status": "APPROVED", "reviewer": "Alice"},
            {"date": "2024-07-10", "status": "APPROVED", "reviewer": "Bob"},
            {"date": "2025-01-20", "status": "PENDING", "reviewer": None},
        ],
    },
    "Carlos": {
        "owns": ["ZGBR-5678"],
        "role_type": "ZGBR",
        "department": "Finance",
        "review_history": [
            {"date": "2024-03-10", "status": "APPROVED", "reviewer": "Carol"},
            {"date": "2024-09-18", "status": "FLAGGED", "reviewer": "Carol"},
            {"date": "2025-02-05", "status": "APPROVED", "reviewer": "Dave"},
        ],
    },
    "Fatima": {
        "owns": ["GBR-9999", "ZGBR-0001"],
        "role_type": "mixed",
        "department": "Security",
        "review_history": [
            {"date": "2024-02-20", "status": "APPROVED", "reviewer": "Eve"},
            {"date": "2024-08-14", "status": "APPROVED", "reviewer": "Alice"},
            {"date": "2025-03-01", "status": "PENDING", "reviewer": None},
        ],
    },
}


class KnowledgeGraphTests(unittest.TestCase):
    def test_all_starter_user_records_are_preserved_exactly(self):
        self.assertEqual(KG["users"], STARTER_USERS)
        for name, expected in STARTER_USERS.items():
            with self.subTest(user=name):
                self.assertEqual(role_tool(name), {name: expected})

    def test_hierarchy_and_contrast_are_supported_by_existing_sources(self):
        documents = {doc["source_id"]: doc for doc in KG["documents"]}
        self.assertEqual(len(documents), len(KG["documents"]))
        self.assertEqual(set(documents), {
            "Policy-Doc-GBR-001", "Policy-Doc-ZGBR-002", "Policy-Doc-CMP-003"
        })
        self.assertEqual(KG["role_hierarchy"]["Role"]["parent"], None)
        self.assertEqual(KG["role_hierarchy"]["Ownership Role"]["parent"], "Role")
        for name, role_type in KG["role_types"].items():
            with self.subTest(role_type=name):
                self.assertEqual(role_type["parent"], "Ownership Role")
                self.assertTrue(role_type["source_ids"])
                for source_id in role_type["source_ids"]:
                    self.assertIn(source_id, documents)
                    self.assertIn(name, documents[source_id]["content"])
        self.assertFalse(KG["role_types"]["GBR"]["session_verification"])
        self.assertTrue(KG["role_types"]["ZGBR"]["session_verification"])

    def test_derived_instances_preserve_all_ownership_relationships(self):
        instances = get_role_instances()
        self.assertEqual(set(instances), {
            "GBR-1234", "ZGBR-5678", "GBR-9999", "ZGBR-0001"
        })
        for user, record in STARTER_USERS.items():
            for role_id in record["owns"]:
                with self.subTest(user=user, role_id=role_id):
                    self.assertEqual(instances[role_id]["owners"], [user])
                    self.assertEqual(instances[role_id]["source_id"], "Role-DB")
                    self.assertIn(instances[role_id]["role_type"], KG["role_types"])

    def test_user_walk_is_scoped_and_has_explicit_paths(self):
        context = graph_walk({"users": ["priya"], "role_types": []})
        self.assertEqual(set(context["users"]), {"Priya"})
        self.assertEqual(set(context["role_instances"]), {"GBR-1234"})
        self.assertEqual(set(context["role_types"]), {"GBR"})
        self.assertIn(["user:Priya", "owns", "role:GBR-1234"], context["paths"])
        self.assertIn(["role:GBR-1234", "instance_of", "role_type:GBR"], context["paths"])
        self.assertIn([
            "role_type:GBR", "subtype_of", "Ownership Role", "subtype_of", "Role"
        ], context["paths"])
        self.assertEqual(set(context["source_ids"]), {
            "Role-DB", "Policy-Doc-GBR-001", "Policy-Doc-CMP-003"
        })
        self.assertEqual(
            {doc["source_id"] for doc in context["documents"]},
            set(context["source_ids"]) - {"Role-DB"},
        )

    def test_comparison_walk_does_not_pull_unrelated_users(self):
        context = graph_walk({"role_types": ["GBR", "ZGBR"]})
        self.assertEqual(context["users"], {})
        self.assertEqual(context["role_instances"], {})
        self.assertEqual(set(context["role_types"]), {"GBR", "ZGBR"})
        self.assertEqual(len(context["documents"]), 3)

    def test_explicit_role_walk_resolves_owner_and_review_history(self):
        context = graph_walk({"role_ids": ["zgbr-5678"]})
        self.assertEqual(context["users"], {"Carlos": STARTER_USERS["Carlos"]})
        self.assertEqual(set(context["role_types"]), {"ZGBR"})

    def test_unknown_entities_do_not_create_graph_data(self):
        context = graph_walk({
            "users": ["Unknown User"], "unknown_users": ["Morgan"],
            "role_ids": ["GBR-0000"], "role_types": ["UNKNOWN"],
        })
        self.assertEqual(context["missing_users"], ["Morgan", "Unknown User"])
        for key in ("users", "role_instances", "role_types"):
            self.assertEqual(context[key], {})
        for key in ("paths", "documents", "source_ids"):
            self.assertEqual(context[key], [])
        self.assertEqual(graph_walk({})["paths"], [])

    def test_graph_context_and_derived_instances_are_isolated(self):
        original = deepcopy(KG)
        context = graph_walk({"users": ["Priya"]})
        context["users"]["Priya"]["review_history"][0]["status"] = "CHANGED"
        context["role_types"]["GBR"]["source_ids"].clear()
        context["documents"][0]["keywords"].clear()
        context["hierarchy"]["Role"]["parent"] = "CHANGED"
        get_role_instances()["GBR-1234"]["owners"].clear()
        self.assertEqual(KG, original)
        self.assertEqual(get_role_instances()["GBR-1234"]["owners"], ["Priya"])

    def test_new_user_and_type_work_without_a_second_role_table(self):
        original = deepcopy(KG)
        new_type = {
            "parent": "Ownership Role", "full_name": "Interview Test Role",
            "source_ids": [],
        }
        new_user = {
            "owns": ["TEST-001"], "role_type": "TEST",
            "department": "Test fixture", "review_history": [],
        }
        # patch.dict restores both dictionaries even if an assertion fails.
        with patch.dict(KG["role_types"], {"TEST": new_type}), \
                patch.dict(KG["users"], {"Interview User": new_user}):
            self.assertEqual(find_role_types("interview test role"), ["TEST"])
            self.assertEqual(role_tool("interview user"), {"Interview User": new_user})
            self.assertEqual(get_role_instances()["TEST-001"], {
                "role_type": "TEST", "owners": ["Interview User"], "source_id": "Role-DB"
            })
            context = graph_walk({"users": ["Interview User"]})
            self.assertEqual(set(context["role_instances"]), {"TEST-001"})
            self.assertEqual(set(context["role_types"]), {"TEST"})
        self.assertEqual(KG, original)


class ToolMatchingTests(unittest.TestCase):
    def test_role_lookup_is_case_insensitive_and_trimmed(self):
        self.assertEqual(role_tool("  pRiYa  "), {"Priya": STARTER_USERS["Priya"]})

    def test_unknown_or_invalid_tool_inputs_are_clean_results(self):
        for user in ("Morgan", "", "   ", None, 42):
            with self.subTest(user=user):
                self.assertIn("error", role_tool(user))
        for query in ("unrelated subject", "", "   ", None, 42):
            with self.subTest(query=query):
                self.assertEqual(document_tool(query), [])

    def test_zgbr_acronym_and_full_names_do_not_match_gbr(self):
        for query in ("zgbr", "ZGBR-5678", "Zero-Trust Global Business Role",
                      "zero trust global business role"):
            with self.subTest(query=query):
                self.assertEqual(find_role_types(query), ["ZGBR"])
                self.assertEqual({doc["source_id"] for doc in document_tool(query)}, {
                    "Policy-Doc-ZGBR-002", "Policy-Doc-CMP-003"
                })

    def test_embedded_terms_do_not_match_role_names_or_policy_keywords(self):
        for query in ("ZGBRish notGBR", "MFAx", "persistentish"):
            with self.subTest(query=query):
                self.assertEqual(find_role_types(query), [])
                self.assertEqual(document_tool(query), [])

    def test_distinct_full_role_names_can_both_be_detected(self):
        query = "Compare Global Business Role with Zero-Trust Global Business Role"
        self.assertEqual(find_role_types(query), ["GBR", "ZGBR"])
        self.assertEqual(len(document_tool(query)), 3)

    def test_tool_results_cannot_mutate_the_knowledge_graph(self):
        original = deepcopy(KG)
        role_tool("Priya")["Priya"]["owns"].append("INVENTED-001")
        document_tool("GBR")[0]["keywords"].append("invented")
        self.assertEqual(KG, original)


if __name__ == "__main__":
    unittest.main()
