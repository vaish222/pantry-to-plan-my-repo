"""Hard eligibility rules (constraints.py, Person 3).

Eligibility is the only place a recipe can be *removed* from consideration.
It runs before scoring and again before a DayPlan is appended. Cuisine and
calories are NOT hard constraints: a cuisine mismatch or an out-of-band recipe
is handled by scoring and flagging, not rejection. The vegetarian gate is hard
and is never relaxed or downgraded to a warning.
"""

from schemas import AppIssue, EligibilityResult, PlanningRequest, Recipe
from telemetry import traced


@traced(name="constraint_check", run_type="tool")
def validate_eligibility(recipe: Recipe, request: PlanningRequest) -> EligibilityResult:
    reject_reasons: list[AppIssue] = []

    if request.vegetarian_required and not recipe.vegetarian:
        reject_reasons.append(
            AppIssue(
                code="VEGETARIAN_REQUIRED",
                message=f"{recipe.recipe_id} is not vegetarian.",
                field="vegetarian",
                recoverable=False,
            )
        )

    return EligibilityResult(
        recipe_id=recipe.recipe_id,
        eligible=not reject_reasons,
        reject_reasons=reject_reasons,
    )
