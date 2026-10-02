"""Focused validation tests for the shared Pydantic contracts."""

import unittest

from pydantic import ValidationError

from schemas import (
    CandidateScore,
    NormalizationStatus,
    PantryItem,
    PantryItemCandidate,
    PlanningRequest,
)


class SchemaValidationTests(unittest.TestCase):
    def test_valid_confirmed_item_accepts_zero_after_depletion(self):
        item = PantryItem(
            ingredient_id="chicken_breast",
            display_name="Chicken breast",
            quantity_g=0,
            confidence=1,
        )
        self.assertEqual(item.quantity_g, 0)

    def test_invalid_ingredient_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            PantryItem(
                ingredient_id="Chicken Breast",
                display_name="Chicken breast",
                quantity_g=100,
                confidence=1,
            )

    def test_confidence_outside_unit_interval_is_rejected(self):
        with self.assertRaises(ValidationError):
            PantryItem(
                ingredient_id="tomato",
                display_name="Tomato",
                quantity_g=100,
                confidence=1.1,
            )

    def test_extra_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            PlanningRequest(
                cuisines=["italian"],
                dinner_calorie_target=600,
                days=3,
                vegetarian_required=True,
                goal="general",
                calorie_target=500,
            )

    def test_candidate_records_normalization_status(self):
        item = PantryItemCandidate(
            ingredient_id="bell_pepper",
            display_name="Bell pepper",
            quantity_g=None,
            confidence=0.8,
            source_text="capsicum",
            normalization_status=NormalizationStatus.ALIAS,
            recipe_supported=True,
        )
        self.assertEqual(item.normalization_status, NormalizationStatus.ALIAS)

    def test_ineligible_score_requires_a_reason(self):
        with self.assertRaises(ValidationError):
            CandidateScore(
                recipe_id="tomato_pasta",
                eligible=False,
                reject_reasons=[],
                cuisine_score=1,
                calorie_delta=0,
                pantry_coverage=0.5,
                repeat_penalty=0,
                total_score=0.8,
            )


if __name__ == "__main__":
    unittest.main()
