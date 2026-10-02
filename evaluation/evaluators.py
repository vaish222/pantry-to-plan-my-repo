"""Deterministic code evaluators for planner results."""

from __future__ import annotations

from math import isclose
from typing import Any

from repository import get_recipe, load_recipes

_RECIPES = {recipe.recipe_id: recipe for recipe in load_recipes()}
_KNOWN_INGREDIENTS = {
    ingredient.ingredient_id
    for recipe in _RECIPES.values()
    for ingredient in recipe.ingredients
}


def _predicted(outputs: dict) -> dict:
    return outputs.get("predicted", outputs)


def _expected(reference_outputs: dict) -> dict:
    return reference_outputs.get("expected", reference_outputs)


def _request(inputs: dict) -> dict:
    return inputs["request"]


def _plan_days(predicted: dict) -> list[dict]:
    return predicted.get("day_plans", [])


def _inventory_and_shopping(inputs: dict, predicted: dict) -> dict[str, Any]:
    initial = {key: float(value or 0) for key, value in inputs["pantry"]["items"].items()}
    remaining = dict(initial)
    demand: dict[str, float] = {}
    negative_events = 0
    invalid_recipe_ids = 0
    for day in _plan_days(predicted):
        recipe_id = day.get("recipe", {}).get("recipe_id")
        recipe = get_recipe(recipe_id) if recipe_id else None
        if recipe is None:
            invalid_recipe_ids += 1
            continue
        for ingredient in recipe.ingredients:
            ing_id = ingredient.ingredient_id
            needed = float(ingredient.quantity_g)
            demand[ing_id] = demand.get(ing_id, 0.0) + needed
            have = remaining.get(ing_id, 0.0)
            raw_balance = have - needed
            if raw_balance < 0:
                # A shortage is valid; a stored negative pantry quantity is not.
                raw_balance = 0.0
            remaining[ing_id] = round(raw_balance, 4)

    actual_remaining = {
        item["ingredient_id"]: float(item.get("quantity_g") or 0)
        for item in predicted.get("final_pantry", [])
    }
    for quantity in actual_remaining.values():
        if quantity < 0:
            negative_events += 1
    inventory_correct = set(actual_remaining) == set(initial) and all(
        isclose(actual_remaining[key], remaining[key], abs_tol=1e-4)
        for key in initial
    )

    expected_shopping = {
        key: round(total - initial.get(key, 0.0), 4)
        for key, total in demand.items()
        if round(total - initial.get(key, 0.0), 4) > 0
    }
    actual_shopping: dict[str, float] = {}
    invalid_shopping_quantities = 0
    for item in predicted.get("shopping_list", []):
        key = item.get("ingredient_id")
        quantity = float(item.get("quantity_g") or 0)
        if not key or quantity <= 0 or key in actual_shopping:
            invalid_shopping_quantities += 1
        if key:
            actual_shopping[key] = quantity
    shopping_correct = set(actual_shopping) == set(expected_shopping) and all(
        isclose(actual_shopping[key], expected_shopping[key], abs_tol=1e-4)
        for key in expected_shopping
    )
    return {
        "inventory_correct": inventory_correct,
        "shopping_list_correct": shopping_correct,
        "negative_inventory_events": negative_events,
        "invalid_recipe_ids_materialized": invalid_recipe_ids,
        "invalid_shopping_list_quantities": invalid_shopping_quantities,
    }


def _fixture_checks(inputs: dict, predicted: dict, expected: dict) -> dict[str, bool]:
    days = _plan_days(predicted)
    warnings = predicted.get("warnings", [])
    request = _request(inputs)
    unknown_inputs = set(inputs["pantry"]["items"]) - _KNOWN_INGREDIENTS
    coverages = [float(day.get("pantry_coverage", 0)) for day in days]
    checks: dict[str, bool] = {}
    for key, value in expected.items():
        if key == "notes":
            continue
        if key == "days_returned":
            actual = len(days) == int(value)
        elif key in {"all_vegetarian", "no_non_vegetarian_selected"}:
            actual = all(day.get("recipe", {}).get("vegetarian") for day in days)
        elif key == "all_cuisine_match":
            actual = all(day.get("cuisine_match") for day in days)
        elif key == "all_within_calorie_band":
            actual = all("calorie_out_of_band" not in day.get("flags", []) for day in days)
        elif key == "fallback_flags":
            actual = any(day.get("fallback") for day in days) is bool(value)
            checks[key] = actual
            continue
        elif key == "at_least_one_fallback":
            actual = any(day.get("fallback") for day in days)
        elif key == "every_day_flagged_out_of_band":
            actual = bool(days) and all("calorie_out_of_band" in day.get("flags", []) for day in days)
        elif key == "some_days_flagged_out_of_band":
            actual = any("calorie_out_of_band" in day.get("flags", []) for day in days)
        elif key == "reports_constraint_prevented_completion":
            actual = len(days) < request["days"] and "NO_ELIGIBLE_RECIPE" in warnings
        elif key == "exact_calorie_match_day_exists":
            actual = any(day.get("calorie_delta") == 0 for day in days)
        elif key == "all_pantry_coverage_zero":
            actual = bool(days) and all(isclose(value, 0.0, abs_tol=1e-4) for value in coverages)
        elif key == "low_pantry_coverage":
            actual = bool(days) and sum(coverages) / len(coverages) <= 0.35
        elif key == "high_pantry_coverage":
            actual = bool(days) and sum(coverages) / len(coverages) >= 0.65
        elif key in {"partial_pantry_coverage", "mixed_pantry_coverage"}:
            actual = bool(days) and 0 < sum(coverages) / len(coverages) < 1
        elif key == "shopping_list_small_or_empty":
            actual = len(predicted.get("shopping_list", [])) <= 5
        elif key == "unrecognized_items_ignored":
            final = {item["ingredient_id"]: item.get("quantity_g") for item in predicted.get("final_pantry", [])}
            actual = bool(unknown_inputs) and all(
                isclose(float(final.get(item) or 0), float(inputs["pantry"]["items"][item] or 0), abs_tol=1e-4)
                for item in unknown_inputs
            )
        else:
            raise ValueError(f"Unsupported golden expectation: {key}")
        checks[key] = actual is bool(value) if isinstance(value, bool) else bool(actual)
    return checks


def evaluate_case(inputs: dict, outputs: dict, reference_outputs: dict) -> dict[str, Any]:
    predicted = _predicted(outputs)
    expected = _expected(reference_outputs)
    request = _request(inputs)
    days = _plan_days(predicted)
    warnings = predicted.get("warnings", [])
    trace = predicted.get("trace", [])

    vegetarian_violations = sum(
        1
        for day in days
        if request["vegetarian_required"] and not day.get("recipe", {}).get("vegetarian")
    )
    unreported_hard_failures = int(
        len(days) < request["days"] and "NO_ELIGIBLE_RECIPE" not in warnings
    )
    inventory = _inventory_and_shopping(inputs, predicted)
    fixture_checks = _fixture_checks(inputs, predicted, expected)
    fixture_score = sum(fixture_checks.values()) / len(fixture_checks) if fixture_checks else 1.0

    by_step: dict[str, list[dict]] = {}
    for event in trace:
        by_step.setdefault(event.get("step", ""), []).append(event)
    deterministic_mode = outputs.get("advisor_mode", "none") == "none"
    trajectory_pass = (
        len(by_step.get("select", [])) >= len(days)
        and len(by_step.get("deplete", [])) == len(days)
        and (not deterministic_mode or not by_step.get("plan_week") and not by_step.get("relax"))
    )
    recovery_tags = {"sparse", "empty", "exhaustion", "mismatch", "unclear-labels"}
    recovery_applicable = bool(set(inputs.get("tags", [])) & recovery_tags)
    safe_completion = (
        vegetarian_violations == 0
        and unreported_hard_failures == 0
        and inventory["negative_inventory_events"] == 0
        and inventory["invalid_recipe_ids_materialized"] == 0
        and inventory["invalid_shopping_list_quantities"] == 0
    )
    failure_recovery_pass = safe_completion if recovery_applicable else True
    guardrails = {
        "vegetarian_violations": vegetarian_violations,
        "negative_inventory_events": inventory["negative_inventory_events"],
        "invalid_recipe_ids_materialized": inventory["invalid_recipe_ids_materialized"],
        "unreported_hard_constraint_failures": unreported_hard_failures,
        "invalid_shopping_list_quantities": inventory["invalid_shopping_list_quantities"],
    }
    hard_constraint_pass = vegetarian_violations == 0 and unreported_hard_failures == 0
    overall = (
        hard_constraint_pass
        and all(fixture_checks.values())
        and inventory["inventory_correct"]
        and inventory["shopping_list_correct"]
        and trajectory_pass
        and failure_recovery_pass
        and not any(guardrails.values())
        and outputs.get("error") in (None, "")
    )
    return {
        "hard_constraint_pass": hard_constraint_pass,
        "fixture_expectation_checks": fixture_checks,
        "fixture_expectation_score": round(fixture_score, 4),
        "inventory_correct": inventory["inventory_correct"],
        "shopping_list_correct": inventory["shopping_list_correct"],
        "trajectory_pass": trajectory_pass,
        "failure_recovery_applicable": recovery_applicable,
        "failure_recovery_pass": failure_recovery_pass,
        "guardrail_violations": guardrails,
        "overall_case_pass": overall,
    }


def langsmith_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
    """Return scalar LangSmith feedback while retaining details in comments."""
    result = evaluate_case(inputs, outputs, reference_outputs)
    return [
        {"key": "hard_constraint_pass", "score": int(result["hard_constraint_pass"])},
        {
            "key": "fixture_expectation_score",
            "score": result["fixture_expectation_score"],
            "comment": json_comment(result["fixture_expectation_checks"]),
        },
        {"key": "inventory_correct", "score": int(result["inventory_correct"])},
        {"key": "shopping_list_correct", "score": int(result["shopping_list_correct"])},
        {"key": "trajectory_pass", "score": int(result["trajectory_pass"])},
        {"key": "failure_recovery_pass", "score": int(result["failure_recovery_pass"])},
        {
            "key": "hard_guardrails_pass",
            "score": int(not any(result["guardrail_violations"].values())),
            "comment": json_comment(result["guardrail_violations"]),
        },
        {"key": "overall_case_pass", "score": int(result["overall_case_pass"])},
    ]


def json_comment(value: dict) -> str:
    import json

    return json.dumps(value, sort_keys=True)

