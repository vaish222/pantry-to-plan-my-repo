"""Deterministic test doubles for the planner.

FakeRetriever mirrors the real retriever's cuisine prefilter without embeddings
or a vector store, so the planner can be built and tested before retrieval.py
lands. It intentionally does NOT truncate to top_k or apply the vegetarian
filter: returning the full cuisine-scoped set keeps evals deterministic, and the
planner is the authoritative place the vegetarian gate is enforced.
"""

from repository import load_recipes
from schemas import PantryState, PlanningRequest, RecipeCandidate


class FakeRetriever:
    def __init__(self, recipes=None):
        self._recipes = recipes if recipes is not None else load_recipes()

    def search(
        self, pantry: PantryState, request: PlanningRequest, top_k: int
    ) -> list[RecipeCandidate]:
        matched = [
            r
            for r in self._recipes
            if any(c in r.cuisine_tags for c in request.cuisines)
        ]
        matched.sort(key=lambda r: r.recipe_id)
        return [RecipeCandidate(recipe_id=r.recipe_id, retrieval_score=1.0) for r in matched]
