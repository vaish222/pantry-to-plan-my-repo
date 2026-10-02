"""Deterministic raw-label to ingredient-id normalization.

The language model identifies visible food. This module owns the stable join key
used by the recipe corpus, pantry state, planner, and shopping list.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable, Mapping

from recipes import RECIPES
from schemas import NormalizationStatus


DATA_DIR = Path(__file__).resolve().parent / "data"
CATALOG_PATH = DATA_DIR / "ingredient_catalog.json"
ALIASES_PATH = DATA_DIR / "ingredient_aliases.json"


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    ingredient_id: str
    display_name: str
    status: NormalizationStatus
    recipe_supported: bool


def clean_name(value: str) -> str:
    """Create a stable comparison key without making food-specific guesses."""

    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    words_only = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
    return " ".join(words_only.split())


def to_ingredient_id(value: str) -> str:
    """Convert cleaned free text to a schema-safe fallback identifier."""

    return clean_name(value).replace(" ", "_") or "unknown_ingredient"


@lru_cache(maxsize=1)
def corpus_ingredient_ids() -> frozenset[str]:
    """Return the recipe corpus vocabulary—the authoritative join keys."""

    return frozenset(
        ingredient.ingredient_id
        for recipe in RECIPES
        for ingredient in recipe.ingredients
    )


@lru_cache(maxsize=4)
def catalog_ingredient_ids(path: Path = CATALOG_PATH) -> frozenset[str]:
    """Load the local catalog and union it with recipe-supported IDs."""

    with path.open(encoding="utf-8") as file:
        catalog = json.load(file)
    additions = {
        to_ingredient_id(value) for value in catalog["additional_ingredient_ids"]
    }
    return frozenset(corpus_ingredient_ids() | additions)


@lru_cache(maxsize=4)
def ingredient_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """Load semantic aliases separately so non-developers can maintain them."""

    with path.open(encoding="utf-8") as file:
        aliases = json.load(file)
    return {
        clean_name(alias): to_ingredient_id(canonical)
        for alias, canonical in aliases.items()
    }


class IngredientNormalizer:
    """Normalize labels against a fixed vocabulary and explicit aliases."""

    def __init__(
        self,
        canonical_ids: Iterable[str] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        ids = canonical_ids if canonical_ids is not None else catalog_ingredient_ids()
        self._canonical_ids = frozenset(to_ingredient_id(value) for value in ids)
        alias_source = ingredient_aliases() if aliases is None else aliases
        self._aliases = {
            clean_name(alias): to_ingredient_id(canonical)
            for alias, canonical in alias_source.items()
            if clean_name(alias)
        }

    def normalize(self, raw_name: str) -> NormalizationResult:
        cleaned = clean_name(raw_name)
        direct_id = to_ingredient_id(cleaned)

        if direct_id in self._canonical_ids:
            ingredient_id = direct_id
            status = NormalizationStatus.EXACT
        elif cleaned in self._aliases and self._aliases[cleaned] in self._canonical_ids:
            ingredient_id = self._aliases[cleaned]
            status = NormalizationStatus.ALIAS
        else:
            singular_id = self._known_singular_id(cleaned)
            if singular_id is not None:
                ingredient_id = singular_id
                status = NormalizationStatus.ALIAS
            else:
                ingredient_id = direct_id
                status = NormalizationStatus.UNMAPPED

        return NormalizationResult(
            ingredient_id=ingredient_id,
            display_name=ingredient_id.replace("_", " ").title(),
            status=status,
            recipe_supported=ingredient_id in corpus_ingredient_ids(),
        )

    def _known_singular_id(self, cleaned: str) -> str | None:
        """Return a singular candidate only when the corpus confirms it."""

        endings = ("es", "s") if cleaned.endswith("es") else ("s",)
        for ending in endings:
            if not cleaned.endswith(ending):
                continue
            candidate = to_ingredient_id(cleaned[: -len(ending)])
            if candidate in self._canonical_ids:
                return candidate
        return None
