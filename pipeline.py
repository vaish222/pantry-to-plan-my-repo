"""N-day planning loop (pipeline.py, Person 3).

Two strategies share one deterministic core:

Default (no advisor): a greedy per-day loop with probabilistic selection. For
each day it retrieves top-k, resolves to corpus recipes, scores all (for the
trace), keeps eligible + in-cuisine + unused, samples one by score, flags the
calorie band, applies to the pantry, and records the day. Cuisine exhaustion
ends the plan early with NO_ELIGIBLE_RECIPE (this is how vegetarian exhaustion
surfaces).

Agentic (advisor injected): #3 the advisor proposes a whole-week ordering
optimizing cross-day goals (minimize shopping list, smart depletion, variety);
the deterministic core validates and materializes each pick. #1 any day the
proposal leaves unfilled, or a day the greedy filter cannot satisfy, goes to a
bounded relax-and-repair loop where the advisor chooses a relaxation from a
fixed menu (allow a repeat, cross cuisines as a flagged last resort, or give
up). The vegetarian gate is NEVER relaxed.

The governing rule throughout: the model only DECIDES (ordering, relaxation
choice); scoring, eligibility, the vegetarian gate, depletion, and
shopping-list reconciliation stay in deterministic Python. Agent decisions are
surfaced only through the already-open PlanResult.trace / PlanResult.warnings /
DayPlan.flags fields, so nothing here touches the P1 schemas or the P2 corpus /
retrieval contracts.
"""

import random
from typing import Optional, Protocol

from advisor import (
    ALLOW_REPEAT,
    CROSS_CUISINE,
    RELAXATION_MENU,
    CandidateSummary,
    PlannerAdvisor,
    RepairContext,
    WeekContext,
)
from config import (
    ALL_CUISINES,
    DEFAULT_TOP_K,
    MAX_RELAXATION_ROUNDS,
    SELECTION_TEMPERATURE,
    WEEK_OBJECTIVES,
    WEEK_POOL_TOP_K,
    calorie_band,
)
from constraints import validate_eligibility
from inventory import apply_recipe, build_shopping_list
from repository import get_recipe
from schemas import (
    CandidateScore,
    DayPlan,
    PantryState,
    PlanningRequest,
    PlanResult,
    RecipeCandidate,
    TraceEvent,
)
from scoring import score_recipe, softmax_select
from telemetry import trace_span


class Retriever(Protocol):
    """Matches the retrieval.py contract (Person 2 owns the real adapters)."""

    def search(
        self, pantry: PantryState, request: PlanningRequest, top_k: int
    ) -> list[RecipeCandidate]:
        ...


# --------------------------------------------------------------- shared core
def _resolve_and_score(
    candidates: list[RecipeCandidate],
    current: PantryState,
    request: PlanningRequest,
    used: set[str],
    day: Optional[int],
    trace: list[TraceEvent],
    warnings: list[str],
) -> list[CandidateScore]:
    """Resolve candidate ids to corpus recipes and score them. Unknown ids are
    reported, not invented. Scores are always computed against `request` so
    cuisine_match/calorie_delta reflect the real request even when retrieval was
    broadened for a cross-cuisine relaxation."""
    recipes = []
    for cand in candidates:
        recipe = get_recipe(cand.recipe_id)
        if recipe is None:
            warnings.append("UNKNOWN_RECIPE_ID")
            trace.append(
                TraceEvent(
                    step="retrieve",
                    day=day,
                    message=f"unknown recipe_id ignored: {cand.recipe_id}",
                )
            )
            continue
        recipes.append(recipe)

    scores = [score_recipe(r, current, request, used) for r in recipes]
    for s in scores:
        trace.append(
            TraceEvent(step="score", day=day, message=f"scored {s.recipe_id}", data=s.model_dump())
        )
    return scores


def _selectable(
    scores: list[CandidateScore],
    used: set[str],
    allow_repeat: bool,
    allow_cross_cuisine: bool,
) -> list[CandidateScore]:
    """Candidates that may be picked. Eligibility (the vegetarian gate) is always
    required. Cuisine scope and the no-repeat rule are relaxable by the repair
    loop; nothing else is."""
    out = []
    for s in scores:
        if not s.eligible:
            continue
        if not allow_cross_cuisine and s.cuisine_score <= 0:
            continue
        if not allow_repeat and s.recipe_id in used:
            continue
        out.append(s)
    return out


def _append_day(
    day: int,
    score: CandidateScore,
    recipe,
    request: PlanningRequest,
    low: int,
    high: int,
    cross_cuisine: bool,
    day_plans: list[DayPlan],
    trace: list[TraceEvent],
    current: PantryState,
    select_data: dict,
) -> PantryState:
    """Materialize one selected recipe into a DayPlan and deplete the pantry.
    Returns the new pantry state (apply_recipe never mutates the caller's)."""
    out_of_band = not (low <= recipe.calories_per_serving <= high)
    flags: list[str] = []
    if out_of_band:
        flags.append("calorie_out_of_band")
    if cross_cuisine:
        flags.append("cross_cuisine")

    message = f"selected {recipe.recipe_id}"
    if out_of_band:
        message += " (out of calorie band; flagged)"
    if cross_cuisine:
        message += " (cross-cuisine; flagged)"
    trace.append(
        TraceEvent(
            step="select",
            day=day,
            message=message,
            data={"total_score": score.total_score, "calorie_delta": score.calorie_delta, **select_data},
        )
    )

    day_plans.append(
        DayPlan(
            day=day,
            recipe=recipe,
            cuisine_match=score.cuisine_score > 0,
            calorie_delta=score.calorie_delta,
            pantry_coverage=score.pantry_coverage,
            vegetarian_required=request.vegetarian_required,
            fallback=out_of_band,
            score=score,
            flags=flags,
        )
    )

    new_current, shortages = apply_recipe(current, recipe)
    trace.append(
        TraceEvent(
            step="deplete",
            day=day,
            message=f"applied {recipe.recipe_id}; {len(shortages)} shortage(s)",
        )
    )
    return new_current


# ----------------------------------------------------------- default strategy
def _greedy_plan(
    request, retriever, top_k, rng, temperature, low, high, current, used, day_plans, warnings, trace
) -> PantryState:
    for day in range(1, request.days + 1):
        with trace_span(
            f"planner_day_{day}",
            inputs={"day": day, "pantry": current.model_dump()},
            metadata={"planner_mode": "deterministic"},
        ):
            candidates = retriever.search(current, request, top_k)
            trace.append(
                TraceEvent(
                    step="retrieve",
                    day=day,
                    message=f"retrieved {len(candidates)} candidate(s)",
                    data={"ids": [c.recipe_id for c in candidates]},
                )
            )

            scores = _resolve_and_score(candidates, current, request, used, day, trace, warnings)
            selectable = _selectable(scores, used, allow_repeat=False, allow_cross_cuisine=False)
            if not selectable:
                warnings.append("NO_ELIGIBLE_RECIPE")
                trace.append(
                    TraceEvent(
                        step="select",
                        day=day,
                        message="no eligible unused recipe for the requested cuisine; "
                        "returning a partial plan (hard constraint not relaxed)",
                    )
                )
                break

            chosen = softmax_select(selectable, rng, temperature)
            recipe = get_recipe(chosen.recipe_id)
            current = _append_day(
                day, chosen, recipe, request, low, high, False, day_plans, trace, current,
                {"temperature": temperature, "sampled_from": len(selectable)},
            )
            used.add(recipe.recipe_id)
    return current


# ----------------------------------------------------------- agentic strategy
def _repair_one_day(
    day, request, retriever, top_k, rng, temperature, advisor, low, high, current, used, day_plans, warnings, trace
) -> tuple[bool, PantryState]:
    """#1: fill a single day the normal filter cannot, by asking the advisor for
    a relaxation from the fixed menu and applying it deterministically. Bounded
    by MAX_RELAXATION_ROUNDS; the vegetarian gate is never relaxed."""
    tried: list[str] = []
    allow_repeat = False
    allow_cross = False
    cuisines = list(request.cuisines)

    for _ in range(MAX_RELAXATION_ROUNDS + 1):
        # Retrieval may be broadened (cross-cuisine); scoring stays vs the real request.
        query_request = request.model_copy(update={"cuisines": cuisines})
        candidates = retriever.search(current, query_request, top_k)
        scores = _resolve_and_score(candidates, current, request, used, day, trace, warnings)
        selectable = _selectable(scores, used, allow_repeat, allow_cross)

        if selectable:
            chosen = softmax_select(selectable, rng, temperature)
            recipe = get_recipe(chosen.recipe_id)
            cross = chosen.cuisine_score <= 0
            current = _append_day(
                day, chosen, recipe, request, low, high, cross, day_plans, trace, current,
                {"source": "repair", "tried": list(tried)},
            )
            used.add(recipe.recipe_id)
            return True, current

        ctx = RepairContext(
            day=day,
            reason="no eligible unused recipe for the requested cuisine",
            cuisines=list(request.cuisines),
            vegetarian_required=request.vegetarian_required,
            used_recipe_ids=sorted(used),
            tried_relaxations=list(tried),
            menu=list(RELAXATION_MENU),
        )
        choice = advisor.choose_relaxation(ctx).choice
        trace.append(
            TraceEvent(step="relax", day=day, message=f"advisor chose {choice}", data={"tried": list(tried)})
        )
        # Guarantee progress: a repeated or off-menu choice ends the loop.
        if choice in tried or choice not in RELAXATION_MENU:
            break
        tried.append(choice)

        if choice == ALLOW_REPEAT:
            allow_repeat = True
            warnings.append("RELAXED_REPEAT")
        elif choice == CROSS_CUISINE:
            allow_cross = True
            cuisines = list(ALL_CUISINES)
            warnings.append("RELAXED_CROSS_CUISINE")
        else:  # GIVE_UP
            break

    warnings.append("NO_ELIGIBLE_RECIPE")
    trace.append(
        TraceEvent(
            step="select",
            day=day,
            message="repair exhausted; partial plan (vegetarian gate never relaxed)",
        )
    )
    return False, current


def _agentic_plan(
    request, retriever, top_k, rng, temperature, advisor, low, high, current, used, day_plans, warnings, trace
) -> PantryState:
    # #3: build a week candidate pool and ask the advisor for an ordering.
    pool_k = max(top_k, WEEK_POOL_TOP_K)
    candidates = retriever.search(current, request, pool_k)
    trace.append(
        TraceEvent(
            step="retrieve",
            day=None,
            message=f"week candidate pool: {len(candidates)} candidate(s)",
            data={"ids": [c.recipe_id for c in candidates]},
        )
    )
    scores = _resolve_and_score(candidates, current, request, used, None, trace, warnings)
    summaries = []
    for s in scores:
        recipe = get_recipe(s.recipe_id)
        summaries.append(
            CandidateSummary(
                recipe_id=s.recipe_id,
                title=recipe.title,
                cuisine_tags=recipe.cuisine_tags,
                calories_per_serving=recipe.calories_per_serving,
                vegetarian=recipe.vegetarian,
                ingredient_ids=[i.ingredient_id for i in recipe.ingredients],
                pantry_coverage=s.pantry_coverage,
                cuisine_match=s.cuisine_score > 0,
                calorie_delta=s.calorie_delta,
                total_score=s.total_score,
                eligible=s.eligible,
            )
        )

    ctx = WeekContext(
        days=request.days,
        cuisines=list(request.cuisines),
        dinner_calorie_target=request.dinner_calorie_target,
        calorie_low=low,
        calorie_high=high,
        vegetarian_required=request.vegetarian_required,
        goal=request.goal,
        pantry_ingredient_ids=[i.ingredient_id for i in current.items],
        objectives=list(WEEK_OBJECTIVES),
        candidates=summaries,
    )
    proposal = advisor.plan_week(ctx)
    trace.append(
        TraceEvent(
            step="plan_week",
            day=None,
            message=f"advisor proposed {len(proposal.ordered_recipe_ids)} recipe(s)",
            data={"ordered": list(proposal.ordered_recipe_ids), "rationale": proposal.rationale},
        )
    )

    # Deterministic validation + materialization of the proposal.
    for rid in proposal.ordered_recipe_ids:
        if len(day_plans) >= request.days:
            break
        recipe = get_recipe(rid)
        day = len(day_plans) + 1
        if recipe is None:
            warnings.append("UNKNOWN_RECIPE_ID")
            trace.append(TraceEvent(step="plan_week", message=f"unknown proposed id ignored: {rid}"))
            continue
        if rid in used:
            trace.append(TraceEvent(step="plan_week", message=f"duplicate proposed id skipped: {rid}"))
            continue
        elig = validate_eligibility(recipe, request)
        if not elig.eligible:
            trace.append(
                TraceEvent(
                    step="plan_week",
                    message=f"ineligible proposed id skipped: {rid}",
                    data={"reasons": [r.code for r in elig.reject_reasons]},
                )
            )
            continue
        score = score_recipe(recipe, current, request, used)
        cross = not any(c in recipe.cuisine_tags for c in request.cuisines)
        if cross:
            warnings.append("RELAXED_CROSS_CUISINE")
        current = _append_day(
            day, score, recipe, request, low, high, cross, day_plans, trace, current, {"source": "week_plan"}
        )
        used.add(rid)

    # #1: fill any remaining days via the bounded relax-and-repair loop.
    while len(day_plans) < request.days:
        day = len(day_plans) + 1
        filled, current = _repair_one_day(
            day, request, retriever, top_k, rng, temperature, advisor, low, high, current, used, day_plans, warnings, trace
        )
        if not filled:
            break
    return current


# ------------------------------------------------------------------ entrypoint
def generate_plan(
    pantry: PantryState,
    request: PlanningRequest,
    retriever: Retriever,
    top_k: int = DEFAULT_TOP_K,
    *,
    rng: Optional[random.Random] = None,
    temperature: Optional[float] = None,
    advisor: Optional[PlannerAdvisor] = None,
) -> PlanResult:
    # Non-zero temperature by default => varied plans across re-runs. A caller
    # (evals, reproducible demos) can pin behavior with a seeded rng and/or
    # temperature=0. rng defaults to an entropy-seeded instance so unseeded
    # re-runs genuinely differ.
    temperature = SELECTION_TEMPERATURE if temperature is None else temperature
    rng = rng if rng is not None else random.Random()

    low, high = calorie_band(request)
    initial = pantry.model_copy(deep=True)
    current = pantry.model_copy(deep=True)

    used: set[str] = set()
    day_plans: list[DayPlan] = []
    warnings: list[str] = []
    trace: list[TraceEvent] = []

    # Agentic strategy is opt-in via an injected advisor; otherwise the default
    # deterministic greedy + probabilistic loop runs (behavior unchanged).
    if advisor is None:
        current = _greedy_plan(
            request, retriever, top_k, rng, temperature, low, high, current, used, day_plans, warnings, trace
        )
    else:
        current = _agentic_plan(
            request, retriever, top_k, rng, temperature, advisor, low, high, current, used, day_plans, warnings, trace
        )

    if len(day_plans) < request.days and "NO_ELIGIBLE_RECIPE" not in warnings:
        warnings.append("NO_ELIGIBLE_RECIPE")

    shopping_list = build_shopping_list(day_plans, initial)

    return PlanResult(
        requested_days=request.days,
        day_plans=day_plans,
        final_pantry=current.items,
        shopping_list=shopping_list,
        warnings=warnings,
        trace=trace,
    )
