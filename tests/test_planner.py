"""Invariant and scenario tests for the Person 3 planner.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests

Covers the blueprint's Person 3 release bar: deterministic output, the
vegetarian hard gate, cuisine match, exact calorie_delta, non-negative
depletion, and shopping-list reconciliation. Scenario tests drive the four
required eval fixtures plus the fallback/exact-match cases.
"""

import glob
import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import calorie_band
from pipeline import generate_plan
from repository import get_recipe
from schemas import PantryItem, PantryState, PlanningRequest
from scoring import rank
from schemas import CandidateScore
from tests.fakes import FakeRetriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX_DIR = os.path.join(ROOT, "fixtures")


def load_fixture(stem):
    with open(os.path.join(FIX_DIR, stem + ".json")) as fh:
        return json.load(fh)


def all_fixtures():
    for path in sorted(glob.glob(os.path.join(FIX_DIR, "*.json"))):
        with open(path) as fh:
            yield json.load(fh)


def pantry_of(fx):
    items = [
        PantryItem(
            ingredient_id=k,
            display_name=k.replace("_", " "),
            quantity_g=float(v),
            confidence=1.0,
        )
        for k, v in fx["pantry"]["items"].items()
    ]
    return PantryState(items=items, as_of=fx["pantry"].get("as_of"))


def request_of(fx):
    return PlanningRequest(**fx["request"])


def plan_fixture(fx, *, rng=None, temperature=0.0):
    # Deterministic argmax by default (temperature=0) so scenario/invariant
    # tests pin the reproducible contract. Probabilistic behavior is exercised
    # explicitly in ProbabilisticSelectionTests.
    return generate_plan(
        pantry_of(fx), request_of(fx), FakeRetriever(), rng=rng, temperature=temperature
    )


class InvariantTests(unittest.TestCase):
    """Hold for every fixture, whatever the scenario."""

    def test_all_fixtures_hold_invariants(self):
        for fx in all_fixtures():
            with self.subTest(fixture=fx["id"]):
                req = request_of(fx)
                result = generate_plan(
                    pantry_of(fx), req, FakeRetriever(), temperature=0.0
                )

                # Never produce more days than requested.
                self.assertLessEqual(len(result.day_plans), req.days)
                self.assertEqual(result.requested_days, req.days)

                # Each day is exactly one recipe, days number 1..k in order.
                self.assertEqual(
                    [dp.day for dp in result.day_plans],
                    list(range(1, len(result.day_plans) + 1)),
                )

                # No repeated recipe within a plan.
                ids = [dp.recipe.recipe_id for dp in result.day_plans]
                self.assertEqual(len(ids), len(set(ids)))

                # calorie_delta is exact.
                for dp in result.day_plans:
                    self.assertEqual(
                        dp.calorie_delta,
                        dp.recipe.calories_per_serving - req.dinner_calorie_target,
                    )

                # cuisine_match agrees with the tags.
                for dp in result.day_plans:
                    expected = any(c in dp.recipe.cuisine_tags for c in req.cuisines)
                    self.assertEqual(dp.cuisine_match, expected)

                # Final pantry never negative.
                for item in result.final_pantry:
                    self.assertGreaterEqual(item.quantity_g or 0.0, 0.0)

                # Shopping list only lists real shortages.
                for sli in result.shopping_list:
                    self.assertGreater(sli.quantity_g, 0.0)

                # A short plan must carry a machine-readable reason.
                if len(result.day_plans) < req.days:
                    self.assertIn("NO_ELIGIBLE_RECIPE", result.warnings)

    def test_shopping_list_reconciles(self):
        # initial pantry + shopping quantity must cover total planned demand.
        for fx in all_fixtures():
            with self.subTest(fixture=fx["id"]):
                pantry = pantry_of(fx)
                result = generate_plan(
                    pantry, request_of(fx), FakeRetriever(), temperature=0.0
                )

                have = {i.ingredient_id: (i.quantity_g or 0.0) for i in pantry.items}
                buy = {s.ingredient_id: s.quantity_g for s in result.shopping_list}
                demand = {}
                for dp in result.day_plans:
                    for ing in dp.recipe.ingredients:
                        demand[ing.ingredient_id] = demand.get(ing.ingredient_id, 0.0) + ing.quantity_g

                for ing_id, needed in demand.items():
                    supplied = have.get(ing_id, 0.0) + buy.get(ing_id, 0.0)
                    self.assertGreaterEqual(
                        round(supplied - needed, 3), 0.0, msg=f"{fx['id']} short on {ing_id}"
                    )

    def test_deterministic(self):
        for stem in ("01_well_stocked_veg_italian", "02_sparse_chinese_fallback"):
            fx = load_fixture(stem)
            a = plan_fixture(fx)
            b = plan_fixture(fx)
            self.assertEqual(
                [d.recipe.recipe_id for d in a.day_plans],
                [d.recipe.recipe_id for d in b.day_plans],
            )


class VegetarianGateTests(unittest.TestCase):
    def test_vegetarian_required_never_selects_meat(self):
        for fx in all_fixtures():
            if not fx["request"]["vegetarian_required"]:
                continue
            with self.subTest(fixture=fx["id"]):
                result = plan_fixture(fx)
                for dp in result.day_plans:
                    self.assertTrue(dp.recipe.vegetarian, f"{fx['id']}: {dp.recipe.recipe_id}")


class ScenarioTests(unittest.TestCase):
    def test_01_clean_happy_path(self):
        r = plan_fixture(load_fixture("01_well_stocked_veg_italian"))
        self.assertEqual(len(r.day_plans), 3)
        self.assertTrue(all(dp.recipe.vegetarian for dp in r.day_plans))
        self.assertTrue(all(dp.cuisine_match for dp in r.day_plans))
        self.assertFalse(any(dp.fallback for dp in r.day_plans))

    def test_02_sparse_fallback_fires(self):
        r = plan_fixture(load_fixture("02_sparse_chinese_fallback"))
        self.assertEqual(len(r.day_plans), 5)
        self.assertTrue(any(dp.fallback for dp in r.day_plans))

    def test_03_tight_band_every_day_flagged(self):
        r = plan_fixture(load_fixture("03_tight_band_indian"))
        self.assertEqual(len(r.day_plans), 4)
        self.assertTrue(all("calorie_out_of_band" in dp.flags for dp in r.day_plans))

    def test_04_vegetarian_exhaustion_returns_partial(self):
        fx = load_fixture("04_veg_exhaustion_american")
        r = plan_fixture(fx)
        self.assertEqual(fx["request"]["days"], 7)
        self.assertEqual(len(r.day_plans), 6)  # only 6 veg American recipes exist
        self.assertIn("NO_ELIGIBLE_RECIPE", r.warnings)
        self.assertTrue(all(dp.recipe.vegetarian for dp in r.day_plans))

    def test_11_exact_calorie_match_no_fallback(self):
        r = plan_fixture(load_fixture("11_narrow_band_american_exact"))
        self.assertEqual(len(r.day_plans), 2)
        self.assertTrue(any(dp.calorie_delta == 0 for dp in r.day_plans))
        self.assertFalse(any(dp.fallback for dp in r.day_plans))


class ProbabilisticSelectionTests(unittest.TestCase):
    """The default sampler adds variety without breaking any hard constraint."""

    def _ids(self, result):
        return [dp.recipe.recipe_id for dp in result.day_plans]

    def test_same_seed_is_reproducible(self):
        # Non-zero temperature, but a pinned seed => identical plans. This is
        # how evals stay reproducible while the app default stays varied.
        fx = load_fixture("01_well_stocked_veg_italian")
        a = plan_fixture(fx, rng=random.Random(123), temperature=0.5)
        b = plan_fixture(fx, rng=random.Random(123), temperature=0.5)
        self.assertEqual(self._ids(a), self._ids(b))

    def test_different_seeds_produce_variety(self):
        # Across many seeds a fixture with several eligible recipes must yield
        # more than one distinct plan; otherwise selection is not probabilistic.
        fx = load_fixture("01_well_stocked_veg_italian")
        plans = {
            tuple(self._ids(plan_fixture(fx, rng=random.Random(seed), temperature=0.5)))
            for seed in range(30)
        }
        self.assertGreater(len(plans), 1)

    def test_sampling_never_violates_invariants(self):
        # Every seed must still respect the vegetarian gate, no-repeat rule,
        # non-negative depletion, and never exceed the requested day count.
        for fx in all_fixtures():
            for seed in range(8):
                with self.subTest(fixture=fx["id"], seed=seed):
                    req = request_of(fx)
                    r = plan_fixture(fx, rng=random.Random(seed), temperature=0.5)
                    self.assertLessEqual(len(r.day_plans), req.days)
                    ids = self._ids(r)
                    self.assertEqual(len(ids), len(set(ids)))
                    if req.vegetarian_required:
                        self.assertTrue(all(dp.recipe.vegetarian for dp in r.day_plans))
                    for item in r.final_pantry:
                        self.assertGreaterEqual(item.quantity_g or 0.0, 0.0)

    def test_veg_gate_holds_under_sampling(self):
        # A vegetarian-required fixture must never sample a meat recipe, no
        # matter the seed or how hot the temperature is.
        fx = next(f for f in all_fixtures() if f["request"]["vegetarian_required"])
        for seed in range(20):
            r = plan_fixture(fx, rng=random.Random(seed), temperature=5.0)
            self.assertTrue(all(dp.recipe.vegetarian for dp in r.day_plans))


class ScoringTests(unittest.TestCase):
    def test_tie_break_order(self):
        # Same total via equal inputs: higher coverage, then smaller |delta|,
        # then recipe_id wins.
        a = CandidateScore(recipe_id="b", eligible=True, reject_reasons=[], cuisine_score=1.0,
                           calorie_delta=10, pantry_coverage=0.5, repeat_penalty=0.0, total_score=0.8)
        b = CandidateScore(recipe_id="a", eligible=True, reject_reasons=[], cuisine_score=1.0,
                           calorie_delta=10, pantry_coverage=0.5, repeat_penalty=0.0, total_score=0.8)
        c = CandidateScore(recipe_id="c", eligible=True, reject_reasons=[], cuisine_score=1.0,
                           calorie_delta=10, pantry_coverage=0.9, repeat_penalty=0.0, total_score=0.8)
        ordered = [s.recipe_id for s in rank([a, b, c])]
        self.assertEqual(ordered, ["c", "a", "b"])  # c highest coverage; a<b by id


if __name__ == "__main__":
    unittest.main()
