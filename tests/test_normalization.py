"""Tests for deterministic ingredient normalization."""

import unittest

from normalization import (
    IngredientNormalizer,
    catalog_ingredient_ids,
    clean_name,
    corpus_ingredient_ids,
)
from schemas import NormalizationStatus


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = IngredientNormalizer()

    def test_cleanup_is_deterministic(self):
        self.assertEqual(clean_name("  Bell-Pepper! "), "bell pepper")

    def test_corpus_id_is_exact(self):
        result = self.normalizer.normalize("bell pepper")
        self.assertEqual(result.ingredient_id, "bell_pepper")
        self.assertEqual(result.status, NormalizationStatus.EXACT)

    def test_semantic_alias_maps_to_corpus_id(self):
        result = self.normalizer.normalize("capsicum")
        self.assertEqual(result.ingredient_id, "bell_pepper")
        self.assertEqual(result.status, NormalizationStatus.ALIAS)

    def test_unknown_is_preserved_but_flagged(self):
        result = self.normalizer.normalize("Rambutan")
        self.assertEqual(result.ingredient_id, "rambutan")
        self.assertEqual(result.status, NormalizationStatus.UNMAPPED)
        self.assertFalse(result.recipe_supported)

    def test_catalog_item_can_be_known_but_not_recipe_supported(self):
        result = self.normalizer.normalize("Dragon Fruit")
        self.assertEqual(result.ingredient_id, "dragon_fruit")
        self.assertEqual(result.status, NormalizationStatus.EXACT)
        self.assertFalse(result.recipe_supported)

    def test_catalog_alias_can_be_known_but_not_recipe_supported(self):
        result = self.normalizer.normalize("pitaya")
        self.assertEqual(result.ingredient_id, "dragon_fruit")
        self.assertEqual(result.status, NormalizationStatus.ALIAS)
        self.assertFalse(result.recipe_supported)

    def test_default_vocabulary_matches_recipe_corpus(self):
        self.assertIn("chicken_breast", corpus_ingredient_ids())
        self.assertIn("tomato", corpus_ingredient_ids())
        self.assertGreater(len(catalog_ingredient_ids()), len(corpus_ingredient_ids()))


if __name__ == "__main__":
    unittest.main()
