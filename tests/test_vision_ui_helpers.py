"""Tests for the vision UI's pure confirmation boundary."""

import unittest

from pydantic import ValidationError

from vision_ui_helpers import editor_rows_to_pantry


class VisionUIHelperTests(unittest.TestCase):
    def test_selected_rows_become_confirmed_pantry(self):
        pantry = editor_rows_to_pantry(
            [
                {
                    "include": True,
                    "ingredient_id": "dragon_fruit",
                    "display_name": "Dragon Fruit",
                    "quantity_g": None,
                    "confidence": 0.9,
                    "source_text": "pitaya",
                },
                {
                    "include": False,
                    "ingredient_id": "tomato",
                    "display_name": "Tomato",
                    "quantity_g": 100,
                    "confidence": 0.8,
                    "source_text": "tomatoes",
                },
            ],
            as_of="2026-09-12",
        )
        self.assertEqual([item.ingredient_id for item in pantry.items], ["dragon_fruit"])
        self.assertEqual(pantry.as_of, "2026-09-12")

    def test_duplicate_known_quantities_are_summed(self):
        pantry = editor_rows_to_pantry(
            [
                _row("tomato", 100, 0.8),
                _row("tomato", 150, 0.9),
            ]
        )
        self.assertEqual(len(pantry.items), 1)
        self.assertEqual(pantry.items[0].quantity_g, 250)
        self.assertEqual(pantry.items[0].confidence, 0.9)

    def test_duplicate_with_unknown_quantity_stays_unknown(self):
        pantry = editor_rows_to_pantry(
            [_row("tomato", 100, 0.8), _row("tomato", None, 0.9)]
        )
        self.assertIsNone(pantry.items[0].quantity_g)

    def test_invalid_manual_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            editor_rows_to_pantry([_row("Dragon Fruit", None, 1.0)])


def _row(ingredient_id, quantity_g, confidence):
    return {
        "include": True,
        "ingredient_id": ingredient_id,
        "display_name": ingredient_id.replace("_", " ").title(),
        "quantity_g": quantity_g,
        "confidence": confidence,
        "source_text": ingredient_id,
    }


if __name__ == "__main__":
    unittest.main()
