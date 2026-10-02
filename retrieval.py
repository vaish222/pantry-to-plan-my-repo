"""Recipe retrieval: Pinecone + configurable embeddings, with local fallback.

Defines a Retriever as a Protocol so implementations can be swapped
(Pinecone + Ollama embeddings, local TF-IDF, etc.) without downstream code changing.

PineconeRetriever uses the configured OpenAI-compatible endpoint,
then searches Pinecone for semantic matches. LocalRetriever provides offline fallback
using cached TF-IDF embeddings.

The output is a ranked list of RecipeCandidate objects with recipe IDs and
retrieval scores. Exact recipe facts are resolved by downstream code using
repository.get_recipe(), never inferred by the retriever.
"""

import os
from typing import Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from index_recipes import _index_dimension, index_recipes
from repository import get_recipe, load_recipes
from schemas import PantryState, PlanningRequest, RecipeCandidate


class Retriever(Protocol):
    """Protocol for recipe retrieval backends.

    Retrieve callable objects that take a PlanningRequest, confirmed pantry
    state, and top_k limit, then return a ranked list of recipe candidates.

    The caller is responsible for resolving recipe facts from candidates
    using repository.get_recipe(); retrieval provides IDs and scores only.
    """

    def search(
        self,
        pantry: PantryState,
        request: PlanningRequest,
        top_k: int = 8,
    ) -> list[RecipeCandidate]:
        """Retrieve top-k recipe candidates matching the request and pantry.

        Args:
            request: PlanningRequest with cuisines, calorie_target, days, etc.
            pantry: Current confirmed PantryState with items on hand.
            top_k: Number of candidates to return.

        Returns:
            List of RecipeCandidate objects sorted by retrieval_score (descending).
            If fewer than top_k eligible recipes exist, return what's available.
        """
        ...


class PineconeRetriever:
    """Cloud retriever using Pinecone + configurable embeddings.

    Defaults to Ollama embeddinggemma at 512 dimensions.

    Requires valid Pinecone settings and a running embedding provider.

    For production use or larger corpora, this provides better scaling and
    does not require keeping vectors in memory.
    """

    def __init__(self):
        """Initialize Pinecone and the configured embedding client."""
        from pinecone import Pinecone
        from llm_provider import embedding_endpoint

        endpoint = embedding_endpoint()
        self.embedding_client = endpoint.client()
        self.embedding_provider = endpoint.name
        self.embedding_model = endpoint.model
        self.embedding_dimensions = endpoint.dimensions

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["PINECONE_INDEX_NAME"]
        index_dimension = _index_dimension(self.pc, index_name)
        if index_dimension is not None and index_dimension != self.embedding_dimensions:
            raise ValueError(
                f"Pinecone index {index_name!r} has dimension {index_dimension}, but "
                f"{self.embedding_provider}/{self.embedding_model} is configured for "
                f"{self.embedding_dimensions}. Re-index with matching dimensions."
            )
        self.index = self.pc.Index(index_name)
        self.namespace = os.environ.get("PINECONE_NAMESPACE", "recipes")

        self.recipes_by_id = {r.recipe_id: r for r in load_recipes()}

    def _embed_text(self, text: str) -> list[float]:
        """Embed text using the configured OpenAI-compatible endpoint."""
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimensions
        )
        return response.data[0].embedding

    def search(
        self,
        pantry: PantryState,
        request: PlanningRequest,
        top_k: int = 8,
    ) -> list[RecipeCandidate]:
        """Retrieve top-k recipe candidates using Pinecone.

        Query construction:
            - Ingredient IDs from the current pantry (what's available)
            - Requested cuisine tags
            - Goal (general/nutritional/kids)

        Vegetarian metadata is filtered in Pinecone and checked locally.
        Cuisine mismatches remain available at a reduced score for fallback.

        Returns:
            Top-k RecipeCandidate objects, sorted by retrieval_score descending.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Build search query from pantry and request
        query_text = self._build_query(request, pantry)

        # Embed with the same model used to populate the index.
        query_vector = self._embed_text(query_text)

        # Search Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k * 2,  # Get more candidates for filtering
            namespace=self.namespace,
            include_metadata=True,
            **({"filter": {"vegetarian": {"$eq": True}}}
               if request.vegetarian_required else {}),
        )

        # Process results and apply metadata filtering
        candidates = []
        for match in results["matches"]:
            recipe_id = match["id"]
            score = float(match["score"])

            if recipe_id not in self.recipes_by_id:
                raise ValueError(f"UNKNOWN_RECIPE_ID: {recipe_id}")

            recipe = self.recipes_by_id[recipe_id]

            # Metadata filter: cuisine match
            cuisine_match = any(c in recipe.cuisine_tags for c in request.cuisines)
            if not cuisine_match:
                score *= 0.5

            # Metadata filter: vegetarian requirement
            if request.vegetarian_required and not recipe.vegetarian:
                continue

            candidates.append((recipe_id, score))

        # Sort by score descending and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        result = [
            RecipeCandidate(recipe_id=recipe_id, retrieval_score=score)
            for recipe_id, score in candidates[:top_k]
        ]

        return result

    def _build_query(self, request: PlanningRequest, pantry: PantryState) -> str:
        """Build a search query from the pantry and request."""
        parts = []

        # Add pantry ingredients
        for item in pantry.items:
            if item.quantity_g is not None and item.quantity_g > 0:
                parts.append(item.ingredient_id)

        # Add cuisine preferences
        parts.extend(request.cuisines)

        # Add goal as a signal
        parts.append(request.goal)

        return " ".join(parts)


class LocalRetriever:
    """Local in-memory retriever using TF-IDF embeddings and cosine similarity.

    Builds a search query from pantry ingredients, cuisine preference, and goal,
    then ranks recipes by semantic similarity. Applies metadata filtering
    (cuisine match, vegetarian flag) as a prefilter before returning results.

    For the 20-40 recipe corpus, this is fast, deterministic, and offline-capable.
    For larger corpora, this can be replaced with a vector database backend
    while keeping the same interface.
    """

    def __init__(self):
        """Load or build the recipe embedding index."""
        self.embeddings, self.recipe_index, self.vectorizer = index_recipes()
        self.recipes = load_recipes()
        self.recipes_by_id = {r.recipe_id: r for r in self.recipes}

    def search(
        self,
        pantry: PantryState,
        request: PlanningRequest,
        top_k: int = 8,
    ) -> list[RecipeCandidate]:
        """Retrieve top-k recipe candidates.

        Query construction:
            - Ingredient IDs from the current pantry (what's available)
            - Requested cuisine tags
            - Goal (general/nutritional/kids)

        Vegetarian recipes are required when requested. Cuisine mismatches
        remain available at a reduced score for the planner fallback.

        Ranking:
            - TF-IDF cosine similarity between query and recipe embeddings
            - Higher score = more relevant

        Returns:
            Top-k RecipeCandidate objects, sorted by retrieval_score descending.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Build search query from pantry and request
        query_text = self._build_query(request, pantry)

        # Vectorize the query using the cached vectorizer
        query_vector = self.vectorizer.transform([query_text]).toarray()[0]

        # Compute cosine similarity against all recipes
        similarities = cosine_similarity([query_vector], self.embeddings)[0]

        # Collect candidates with metadata filtering
        candidates = []
        for recipe in self.recipes:
            recipe_idx = self.recipe_index[recipe.recipe_id]
            score = float(similarities[recipe_idx])

            # Metadata filter: cuisine match
            cuisine_match = any(c in recipe.cuisine_tags for c in request.cuisines)
            if not cuisine_match:
                # Reduce score but don't exclude; scoring.py will filter later
                score *= 0.5

            # Metadata filter: vegetarian requirement
            if request.vegetarian_required and not recipe.vegetarian:
                # Exclude before top-k; the planner checks the gate again.
                continue

            candidates.append((recipe, score))

        # Sort by score descending and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        result = [
            RecipeCandidate(recipe_id=recipe.recipe_id, retrieval_score=score)
            for recipe, score in candidates[:top_k]
        ]

        return result

    def _build_query(self, request: PlanningRequest, pantry: PantryState) -> str:
        """Build a search query from the pantry and request.

        Combines:
            - Ingredient IDs currently available (what you want to use)
            - Cuisine preferences
            - Goal (general, nutritional, kids)

        Returns:
            A space-separated string for TF-IDF vectorization.
        """
        parts = []

        # Add pantry ingredients (prioritize these)
        for item in pantry.items:
            if item.quantity_g is not None and item.quantity_g > 0:
                parts.append(item.ingredient_id)

        # Add cuisine preferences
        parts.extend(request.cuisines)

        # Add goal as a signal
        parts.append(request.goal)

        return " ".join(parts)


class AutoRetriever:
    """Try cloud retrieval; after a provider failure, stay local for this instance."""

    def __init__(self):
        self.backend = "pinecone"
        self.fallback_reason = None
        self._retriever = None

    def _use_local(self, reason: str) -> None:
        self.fallback_reason = reason
        print(f"{reason}; falling back to LocalRetriever (TF-IDF)", flush=True)
        self._retriever = LocalRetriever()
        self.backend = "local"

    def search(self, pantry: PantryState, request: PlanningRequest,
               top_k: int = 8) -> list[RecipeCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.backend == "local":
            return self._retriever.search(pantry, request, top_k)

        required = ("PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            self._use_local("Missing cloud settings: " + ", ".join(missing))
        else:
            from openai import APIError
            from pinecone.errors.exceptions import PineconeError

            try:
                if self._retriever is None:
                    self._retriever = PineconeRetriever()
                # Empty results are valid. Unknown IDs and validation errors
                # propagate instead of being hidden by fallback.
                return self._retriever.search(pantry, request, top_k)
            except (APIError, PineconeError, ConnectionError, TimeoutError) as exc:
                self._use_local(f"Cloud retrieval failed ({type(exc).__name__})")
        return self._retriever.search(pantry, request, top_k)


def create_retriever(backend: Optional[str] = None) -> Retriever:
    """Choose auto (default), strict Pinecone, or local retrieval.

    RETRIEVAL_BACKEND overrides the default when no argument is supplied.
    Auto handles missing credentials and provider errors during initialization
    or search. Explicit pinecone mode reports failures without switching.
    """
    backend = (backend or os.environ.get("RETRIEVAL_BACKEND", "auto")).lower()
    if backend == "auto":
        return AutoRetriever()
    if backend == "local":
        print("Using LocalRetriever (TF-IDF)")
        return LocalRetriever()
    if backend == "pinecone":
        print("Using PineconeRetriever (Ollama embeddings + Pinecone)")
        return PineconeRetriever()
    raise ValueError(f"Unknown retrieval backend: {backend}")


# Backward compatibility
def create_local_retriever() -> Retriever:
    """Deprecated: Use create_retriever() instead."""
    return LocalRetriever()


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from dotenv import load_dotenv
    from schemas import PantryItem

    parser = argparse.ArgumentParser(description="Search recipes using a pantry fixture")
    # Backend priority: --backend > RETRIEVAL_BACKEND from the environment/.env > auto.
    # Auto tries Pinecone first and falls back to TF-IDF on missing credentials
    # or cloud errors. Explicit pinecone mode has no fallback; local uses TF-IDF.
    parser.add_argument("--backend", choices=("auto", "local", "pinecone"),
                        help="Default: RETRIEVAL_BACKEND or auto (Pinecone with local fallback)")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parent /
                        "fixtures/01_well_stocked_veg_italian.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    fixture = json.loads(args.fixture.read_text())
    pantry_data = dict(fixture["pantry"])
    if isinstance(pantry_data["items"], dict):
        pantry_data["items"] = [PantryItem(
            ingredient_id=name, display_name=name.replace("_", " "),
            quantity_g=quantity, confidence=1,
        ) for name, quantity in pantry_data["items"].items()]
    pantry = PantryState.model_validate(pantry_data)
    request = PlanningRequest.model_validate(fixture["request"])
    retriever = create_retriever(args.backend)
    candidates = retriever.search(pantry, request, args.top_k)
    actual_backend = (retriever.backend if isinstance(retriever, AutoRetriever)
                      else "local" if isinstance(retriever, LocalRetriever) else "pinecone")
    print(f"Backend: {actual_backend}; fixture: {args.fixture.name}", flush=True)
    for candidate in candidates:
        recipe = get_recipe(candidate.recipe_id)
        print(f"{candidate.retrieval_score:.4f}  {recipe.recipe_id}: {recipe.title} "
              f"(vegetarian={recipe.vegetarian})")
