"""Pure helpers shared by the independent Streamlit vision UI.

This module intentionally has no Streamlit dependency, so confirmation logic is
cheap to unit-test and can later be reused by another UI or API endpoint.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Iterable

from schemas import PantryItem, PantryItemCandidate, PantryState


def candidate_to_editor_row(candidate: PantryItemCandidate) -> dict[str, Any]:
    """Convert a typed candidate into one editable table row."""

    return {
        "include": True,
        "display_name": candidate.display_name,
        "ingredient_id": candidate.ingredient_id,
        "quantity_g": candidate.quantity_g,
        "confidence": candidate.confidence,
        "source_text": candidate.source_text,
        "normalization_status": candidate.normalization_status.value,
        "recipe_supported": candidate.recipe_supported,
    }


def candidates_to_editor_rows(
    candidates: Iterable[PantryItemCandidate],
) -> list[dict[str, Any]]:
    return [candidate_to_editor_row(candidate) for candidate in candidates]


def editor_rows_to_pantry(
    rows: Iterable[dict[str, Any]], as_of: str | None = None
) -> PantryState:
    """Validate selected rows and merge duplicate canonical IDs.

    Known duplicate quantities are summed. If any duplicate has an unknown
    quantity, the merged quantity remains unknown rather than inventing a total.
    """

    grouped: dict[str, list[PantryItem]] = {}
    for row in rows:
        if not bool(row.get("include", True)):
            continue
        quantity = _optional_float(row.get("quantity_g"))
        item = PantryItem(
            ingredient_id=str(row.get("ingredient_id", "")).strip(),
            display_name=str(row.get("display_name", "")).strip(),
            quantity_g=quantity,
            confidence=_required_float(row.get("confidence", 1.0), "confidence"),
            source_text=_optional_text(row.get("source_text")),
        )
        grouped.setdefault(item.ingredient_id, []).append(item)

    merged: list[PantryItem] = []
    for ingredient_id, items in grouped.items():
        quantities = [item.quantity_g for item in items]
        total = None if any(value is None for value in quantities) else sum(quantities)
        merged.append(
            PantryItem(
                ingredient_id=ingredient_id,
                display_name=items[0].display_name,
                quantity_g=total,
                confidence=max(item.confidence for item in items),
                source_text=", ".join(
                    dict.fromkeys(item.source_text for item in items if item.source_text)
                )
                or None,
            )
        )

    return PantryState(items=merged, as_of=as_of or date.today().isoformat())


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    converted = float(value)
    return None if math.isnan(converted) else converted


def _required_float(value: Any, field: str) -> float:
    converted = _optional_float(value)
    if converted is None:
        raise ValueError(f"{field} is required")
    return converted


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
