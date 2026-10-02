"""Tests for image validation and the provider-neutral vision pipeline."""

import unittest
from pathlib import Path

from schemas import NormalizationStatus, RawDetectedIngredient, RawVisionResult
from vision import MockVisionProvider, VisionPipelineError, parse_pantry_image


ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / "fixtures" / "photos" / "01_well_stocked_veg_italian_messy.png"


class VisionPipelineTests(unittest.TestCase):
    def test_mock_provider_uses_same_result_contract_as_live_provider(self):
        result = parse_pantry_image(PHOTO.read_bytes(), MockVisionProvider())
        self.assertEqual(result.provider, "mock")
        self.assertEqual(
            [item.ingredient_id for item in result.items],
            ["tomato", "bell_pepper"],
        )
        self.assertEqual(
            result.items[1].normalization_status,
            NormalizationStatus.ALIAS,
        )

    def test_missing_quantities_are_not_invented(self):
        result = parse_pantry_image(PHOTO.read_bytes(), MockVisionProvider())
        self.assertTrue(all(item.quantity_g is None for item in result.items))
        self.assertEqual(result.warnings.count("QUANTITY_UNKNOWN"), 2)

    def test_low_confidence_and_unmapped_items_are_flagged(self):
        provider = MockVisionProvider(
            RawVisionResult(
                ingredients=[
                    RawDetectedIngredient(
                        raw_name="rambutan",
                        display_name="Rambutan",
                        confidence=0.4,
                    )
                ]
            )
        )
        result = parse_pantry_image(PHOTO.read_bytes(), provider)
        self.assertIn("LOW_CONFIDENCE", result.warnings)
        self.assertIn("UNMAPPED_INGREDIENT", result.warnings)

    def test_known_catalog_item_can_lack_recipe_support(self):
        provider = MockVisionProvider(
            RawVisionResult(
                ingredients=[
                    RawDetectedIngredient(
                        raw_name="dragon fruit",
                        display_name="Dragon fruit",
                        confidence=0.95,
                    )
                ]
            )
        )
        result = parse_pantry_image(PHOTO.read_bytes(), provider)
        self.assertEqual(result.items[0].normalization_status, NormalizationStatus.EXACT)
        self.assertFalse(result.items[0].recipe_supported)
        self.assertIn("NO_RECIPE_SUPPORT", result.warnings)
        self.assertNotIn("UNMAPPED_INGREDIENT", result.warnings)

    def test_empty_detection_is_recoverable(self):
        provider = MockVisionProvider(RawVisionResult())
        result = parse_pantry_image(PHOTO.read_bytes(), provider)
        self.assertEqual(result.items, [])
        self.assertIn("NO_INGREDIENTS_DETECTED", result.warnings)

    def test_invalid_upload_is_rejected_before_detection(self):
        with self.assertRaises(VisionPipelineError) as caught:
            parse_pantry_image(b"not an image", MockVisionProvider())
        self.assertEqual(caught.exception.issue.code, "INVALID_IMAGE")

    def test_confidence_threshold_is_validated(self):
        with self.assertRaises(ValueError):
            parse_pantry_image(PHOTO.read_bytes(), MockVisionProvider(), low_confidence_threshold=2)


if __name__ == "__main__":
    unittest.main()
