"""Recipe corpus loading and lookup.

Thin accessor over the recipe corpus so downstream modules resolve recipe facts
through one place rather than importing recipes.py directly. The blueprint has
this as a shared module; for the prototype it reads the typed corpus from
recipes.py. A future version can back it with data/recipes.json and add aisle
metadata without changing callers.
"""

from typing import Optional

from recipes import RECIPES
from schemas import Recipe

_BY_ID: dict[str, Recipe] = {r.recipe_id: r for r in RECIPES}


def load_recipes() -> list[Recipe]:
    return list(RECIPES)


def recipes_by_id() -> dict[str, Recipe]:
    return dict(_BY_ID)


def get_recipe(recipe_id: str) -> Optional[Recipe]:
    """Resolve a recipe_id to its authoritative Recipe, or None if unknown.

    An unknown id from retrieval is an error to be reported (UNKNOWN_RECIPE_ID),
    never a recipe to be generated.
    """
    return _BY_ID.get(recipe_id)
