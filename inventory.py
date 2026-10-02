"""Pantry depletion and shopping-list reconciliation (inventory.py, Person 3).

Pure functions. apply_recipe never mutates the caller's pantry; it returns a new
state with quantities floored at zero and the unmet amount reported as a
shortage. build_shopping_list recomputes demand independently from the initial
pantry plus every selected recipe, which is the reconciliation check: for every
ingredient, initial pantry + shopping quantity must cover total planned demand.
"""

from schemas import DayPlan, PantryState, Recipe, ShoppingListItem, Shortage
from telemetry import traced


@traced(name="inventory_update", run_type="tool")
def apply_recipe(pantry: PantryState, recipe: Recipe) -> tuple[PantryState, list[Shortage]]:
    """Subtract one serving of `recipe` from `pantry`; return (new_state, shortages)."""
    # Deep-copy items so the caller's state is untouched.
    items = {item.ingredient_id: item.model_copy() for item in pantry.items}
    shortages: list[Shortage] = []

    for ing in recipe.ingredients:
        item = items.get(ing.ingredient_id)
        have = (item.quantity_g or 0.0) if item else 0.0
        need = ing.quantity_g
        if have >= need:
            if item is not None:
                item.quantity_g = round(have - need, 4)
        else:
            shortages.append(
                Shortage(
                    ingredient_id=ing.ingredient_id,
                    quantity_g=round(need - have, 4),
                    contributing_recipe_ids=[recipe.recipe_id],
                )
            )
            if item is not None:
                item.quantity_g = 0.0

    new_state = PantryState(items=list(items.values()), as_of=pantry.as_of)
    return new_state, shortages


@traced(name="shopping_list_reconciliation", run_type="tool")
def build_shopping_list(
    day_plans: list[DayPlan], initial_pantry: PantryState
) -> list[ShoppingListItem]:
    """Aggregate total demand across the plan, subtract the initial pantry, and
    return only ingredients still short (quantity_g > 0)."""
    demand: dict[str, float] = {}
    contributors: dict[str, list[str]] = {}
    for dp in day_plans:
        for ing in dp.recipe.ingredients:
            demand[ing.ingredient_id] = demand.get(ing.ingredient_id, 0.0) + ing.quantity_g
            rid = dp.recipe.recipe_id
            contributors.setdefault(ing.ingredient_id, [])
            if rid not in contributors[ing.ingredient_id]:
                contributors[ing.ingredient_id].append(rid)

    have = {item.ingredient_id: (item.quantity_g or 0.0) for item in initial_pantry.items}

    shopping: list[ShoppingListItem] = []
    for ingredient_id, total_needed in sorted(demand.items()):
        short = round(total_needed - have.get(ingredient_id, 0.0), 4)
        if short > 0:
            shopping.append(
                ShoppingListItem(
                    ingredient_id=ingredient_id,
                    quantity_g=short,
                    contributing_recipe_ids=contributors[ingredient_id],
                )
            )
    return shopping
