import unittest

from evaluation.dataset import build_cases, dataset_hash, find_case
from evaluation.evaluators import evaluate_case
from evaluation.run_eval import make_target


class DatasetTests(unittest.TestCase):
    def test_golden_dataset_maps_exactly_23_cases(self):
        cases = build_cases()
        self.assertEqual(len(cases), 23)
        self.assertEqual(cases[0]["case_id"], "PTP-001")
        self.assertEqual(cases[-1]["case_id"], "PTP-023")
        self.assertEqual(len({case["fixture_id"] for case in cases}), 23)
        self.assertEqual(len(dataset_hash(cases)), 64)

    def test_ptp_001_has_expected_fixture_contract(self):
        case = find_case(build_cases(), "PTP-001")
        self.assertEqual(case["fixture_id"], "01_well_stocked_veg_italian")
        self.assertEqual(case["expected"]["days_returned"], 3)
        self.assertTrue(case["expected"]["all_vegetarian"])
        self.assertEqual(len(case["metadata"]["photo_paths"]), 1)


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = find_case(build_cases(), "PTP-001")

    def test_ptp_001_current_planner_passes(self):
        outputs = make_target("deterministic")(self.case["inputs"])
        result = evaluate_case(
            self.case["inputs"], outputs, {"expected": self.case["expected"]}
        )
        self.assertTrue(result["overall_case_pass"])
        self.assertEqual(result["fixture_expectation_score"], 1.0)
        self.assertFalse(any(result["guardrail_violations"].values()))

    def test_vegetarian_violation_fails_guardrail(self):
        outputs = make_target("deterministic")(self.case["inputs"])
        outputs["predicted"]["day_plans"][0]["recipe"]["vegetarian"] = False
        result = evaluate_case(
            self.case["inputs"], outputs, {"expected": self.case["expected"]}
        )
        self.assertFalse(result["hard_constraint_pass"])
        self.assertFalse(result["overall_case_pass"])
        self.assertEqual(result["guardrail_violations"]["vegetarian_violations"], 1)

    def test_shopping_quantity_mismatch_fails(self):
        outputs = make_target("deterministic")(self.case["inputs"])
        outputs["predicted"]["shopping_list"].append(
            {"ingredient_id": "garlic", "quantity_g": 1.0, "contributing_recipe_ids": []}
        )
        result = evaluate_case(
            self.case["inputs"], outputs, {"expected": self.case["expected"]}
        )
        self.assertFalse(result["shopping_list_correct"])
        self.assertFalse(result["overall_case_pass"])


if __name__ == "__main__":
    unittest.main()
