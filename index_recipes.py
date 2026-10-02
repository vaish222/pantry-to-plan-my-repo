"""Recipe corpus validation and Pinecone indexing.

Validates the recipe corpus at startup, ensuring all required fields are valid.
Then uploads recipes to Pinecone with configured embeddings (Ollama by default).

For offline/testing, keeps TF-IDF as fallback, but primary indexing now uses
Ollama + Pinecone for production retrieval.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from recipes import RECIPES
from schemas import Recipe

# Configuration
CACHE_DIR = Path(__file__).resolve().parent / ".embeddings_cache"
EMBEDDINGS_FILE = CACHE_DIR / "recipe_embeddings.json"
VECTORIZER_FILE = CACHE_DIR / "vectorizer.json"

VALID_CUISINES = {"italian", "american", "chinese", "indian"}


def validate_corpus() -> list[str]:
    """Validate all recipes in the corpus. Return a list of validation errors.

    Checks:
    - All required fields present (recipe_id, title, cuisine_tags, ingredients, etc.)
    - Cuisine tags are in the valid set
    - Ingredient IDs are lowercase snake_case
    - Ingredient quantities are positive
    - No duplicate recipe IDs
    """
    errors = []
    seen_ids = set()

    for i, recipe in enumerate(RECIPES):
        # Check required fields
        if not recipe.recipe_id:
            errors.append(f"Recipe {i}: missing recipe_id")
        if not recipe.title:
            errors.append(f"Recipe {i}: missing title")
        if not recipe.cuisine_tags:
            errors.append(f"Recipe {i}: missing cuisine_tags")
        if not recipe.ingredients:
            errors.append(f"Recipe {i}: missing ingredients")

        # Check cuisine tags
        for cuisine in recipe.cuisine_tags:
            if cuisine not in VALID_CUISINES:
                errors.append(
                    f"Recipe {recipe.recipe_id}: invalid cuisine '{cuisine}' "
                    f"(valid: {VALID_CUISINES})"
                )

        # Check for duplicates
        if recipe.recipe_id in seen_ids:
            errors.append(f"Recipe {recipe.recipe_id}: duplicate recipe_id")
        seen_ids.add(recipe.recipe_id)

        # Check ingredients
        for ing in recipe.ingredients:
            # Allow lowercase words separated by single underscores.
            if not re.fullmatch(r"[a-z]+(?:_[a-z]+)*", ing.ingredient_id):
                errors.append(
                    f"Recipe {recipe.recipe_id}: ingredient_id '{ing.ingredient_id}' "
                    "should be lowercase snake_case"
                )
            if ing.quantity_g <= 0:
                errors.append(
                    f"Recipe {recipe.recipe_id}: ingredient '{ing.ingredient_id}' "
                    f"has invalid quantity {ing.quantity_g}"
                )

    return errors


def recipe_text(recipe: Recipe) -> str:
    ingredients = " ".join(ing.ingredient_id for ing in recipe.ingredients)
    return f"{recipe.title} {ingredients} {' '.join(recipe.cuisine_tags)}"


def corpus_fingerprint() -> str:
    content = [(r.recipe_id, recipe_text(r)) for r in RECIPES]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def cached_cloud_embedding(client, text: str, model: str, dimensions: int) -> list[float]:
    """Reuse vectors only when text, model and dimensions all match."""
    key = hashlib.sha256(json.dumps([text, model, dimensions]).encode()).hexdigest()
    folder = CACHE_DIR / "cloud"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    try:
        vector = json.loads(path.read_text())
        if (isinstance(vector, list) and len(vector) == dimensions
                and all(isinstance(x, (int, float)) and np.isfinite(x) for x in vector)):
            return vector
    except (OSError, ValueError):
        pass
    vector = client.embeddings.create(
        model=model, input=text, dimensions=dimensions
    ).data[0].embedding
    if len(vector) != dimensions or not all(np.isfinite(x) for x in vector):
        raise ValueError("Invalid embedding returned by provider")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(vector))
    temporary.replace(path)
    return vector


def _index_dimension(pc, name: str) -> Optional[int]:
    """Best-effort read of an existing index's vector dimension.

    Returns None when the dimension can't be determined. Reads the top-level
    dimension first, then falls back to index stats (which also reports the
    server-side width of an integrated/semantic index).
    """
    try:
        dim = getattr(pc.describe_index(name), "dimension", None)
        if dim:
            return int(dim)
    except Exception:
        pass
    try:
        dim = pc.Index(name).describe_index_stats().get("dimension")
        return int(dim) if dim else None
    except Exception:
        return None


def ensure_pinecone_index(pc, name: str, dimensions: int) -> None:
    """Make sure a dense Pinecone index `name` exists at `dimensions`.

    - Missing: create a dense, cosine, serverless index at `dimensions`
      (cloud/region overridable via PINECONE_CLOUD / PINECONE_REGION).
    - Exists at the right dimension: reuse it as-is.
    - Exists at a different dimension (e.g. a stale index or a 1024-dim
      integrated-embedding index): raise with a clear message so the user
      decides, rather than silently proceeding or destroying data.
    """
    from pinecone import ServerlessSpec

    if name not in {i["name"] for i in pc.list_indexes()}:
        cloud = os.environ.get("PINECONE_CLOUD", "aws")
        region = os.environ.get("PINECONE_REGION", "us-east-1")
        print(f"Creating Pinecone index '{name}' (dim={dimensions}, cosine, {cloud}/{region})...")
        pc.create_index(
            name=name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
        return

    current = _index_dimension(pc, name)
    if current is not None and current != dimensions:
        raise ValueError(
            f"Pinecone index '{name}' already exists with dimension {current}, but the "
            f"embedding config expects {dimensions}. This usually means the index was "
            f"created for a different embedding model (for example a 1024-dim integrated "
            f"index). Delete and recreate it at {dimensions} dims, or set "
            f"OLLAMA_EMBEDDING_DIMENSIONS to match the index, then re-run."
        )
    print(f"Reusing existing Pinecone index '{name}' (dim={current}).")


def index_recipes_to_pinecone() -> None:
    """Upload all recipes to Pinecone with configured embeddings.

    Defaults to Ollama embeddinggemma (512 dims) to embed recipes.
    Ensures the target index exists at the expected dimension first (creating
    it when missing, flagging a dimension mismatch), then reuses cached
    embeddings and replaces records with the same IDs on re-runs.

    Requires:
        PINECONE_API_KEY
        PINECONE_INDEX_NAME
        OLLAMA_EMBEDDING_MODEL (default embeddinggemma)
        OLLAMA_EMBEDDING_DIMENSIONS (default 512)
    """

    # Check for required env vars
    if not all([
        os.environ.get("PINECONE_API_KEY"),
        os.environ.get("PINECONE_INDEX_NAME"),
    ]):
        raise ValueError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME")

    errors = validate_corpus()
    if errors:
        raise ValueError("Corpus validation failed:\n" + "\n".join(errors))
    from pinecone import Pinecone
    from llm_provider import embedding_endpoint

    # Initialize clients
    endpoint = embedding_endpoint()
    embedding_client = endpoint.client()
    embedding_model = endpoint.model
    embedding_dims = endpoint.dimensions

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    ensure_pinecone_index(pc, index_name, embedding_dims)
    index = pc.Index(index_name)
    namespace = os.environ.get("PINECONE_NAMESPACE", "recipes")

    print(f"Indexing {len(RECIPES)} recipes to Pinecone...")

    # Prepare vectors to upsert
    vectors_to_upsert = []

    for recipe in RECIPES:
        embedding_vector = cached_cloud_embedding(
            embedding_client, recipe_text(recipe), embedding_model, embedding_dims
        )

        # Prepare metadata. Instructions ride along so the vector record stays
        # consistent with the corpus; retrieval still resolves exact recipe
        # facts (including steps) from recipes.py via repository.get_recipe(),
        # and instructions are intentionally left out of the embedded text.
        metadata = {
            "recipe_id": recipe.recipe_id,
            "title": recipe.title,
            "cuisine_tags": recipe.cuisine_tags,
            "vegetarian": recipe.vegetarian,
            "calories_per_serving": recipe.calories_per_serving,
            "instructions": recipe.instructions,
        }

        vectors_to_upsert.append((
            recipe.recipe_id,
            embedding_vector,
            metadata
        ))

    # Upsert to Pinecone in batches
    batch_size = 10
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  ✓ Upserted {min(i + batch_size, len(vectors_to_upsert))}/{len(vectors_to_upsert)}")

    print(f"✓ Pinecone indexing complete: {len(RECIPES)} recipes indexed")


def build_embeddings() -> tuple[np.ndarray, dict[str, int], TfidfVectorizer]:
    """Build TF-IDF embeddings for all recipes (fallback mode).

    Returns:
        embeddings: (n_recipes, n_features) array of embedding vectors
        recipe_index: dict mapping recipe_id to row index in embeddings array
        vectorizer: fitted TfidfVectorizer for encoding new queries
    """
    recipe_index = {}
    texts = []

    for idx, recipe in enumerate(RECIPES):
        recipe_index[recipe.recipe_id] = idx

        # Build a search text from recipe title, ingredients, and cuisine
        ingredients_text = " ".join(ing.ingredient_id for ing in recipe.ingredients)
        search_text = f"{recipe.title} {ingredients_text} {' '.join(recipe.cuisine_tags)}"
        texts.append(search_text)

    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=200,  # Limit to 200 features for efficiency
        ngram_range=(1, 2),  # Unigrams and bigrams
    )
    embeddings = vectorizer.fit_transform(texts).toarray()

    return embeddings, recipe_index, vectorizer


def save_embeddings(embeddings: np.ndarray, recipe_index: dict[str, int], vectorizer) -> None:
    """Cache embeddings and vectorizer to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Save embeddings as JSON
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump({
            "fingerprint": corpus_fingerprint(),
            "embeddings": embeddings.tolist(),
            "recipe_index": recipe_index,
        }, f)

    # Save vectorizer vocabulary (convert numpy types to native Python types)
    with open(VECTORIZER_FILE, "w") as f:
        json.dump({
            "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
            "idf": vectorizer.idf_.tolist(),
        }, f)

    print(f"Cached embeddings to {EMBEDDINGS_FILE}")


def load_embeddings() -> Optional[tuple[np.ndarray, dict[str, int], dict]]:
    """Load cached embeddings and vectorizer from disk."""
    if not EMBEDDINGS_FILE.exists() or not VECTORIZER_FILE.exists():
        return None

    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            data = json.load(f)
            if data.get("fingerprint") != corpus_fingerprint():
                return None
            embeddings = np.array(data["embeddings"])
            recipe_index = data["recipe_index"]

        with open(VECTORIZER_FILE, "r") as f:
            vectorizer_data = json.load(f)

        return embeddings, recipe_index, vectorizer_data
    except Exception as e:
        print(f"Failed to load cached embeddings: {e}")
        return None


def index_recipes(force_rebuild: bool = False) -> tuple[np.ndarray, dict[str, int], TfidfVectorizer]:
    """Load or build the recipe embedding index.

    This function is entirely local, even when cloud credentials are configured.

    Args:
        force_rebuild: If True, rebuild TF-IDF embeddings even if cache exists.

    Returns:
        embeddings: (n_recipes, n_features) array
        recipe_index: dict mapping recipe_id to embedding row
        vectorizer: fitted TfidfVectorizer for new queries

    Raises:
        ValueError: If corpus validation fails.
    """
    # Validate corpus first
    errors = validate_corpus()
    if errors:
        error_msg = "Corpus validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    # Try to load cached embeddings
    if not force_rebuild:
        cached = load_embeddings()
        if cached:
            embeddings, recipe_index, vectorizer_data = cached
            # Reconstruct vectorizer (we only need it for new queries)
            vectorizer = TfidfVectorizer(
                vocabulary=vectorizer_data["vocabulary"], lowercase=True,
                stop_words="english", max_features=200, ngram_range=(1, 2),
            )
            vectorizer.idf_ = np.array(vectorizer_data["idf"])
            return embeddings, recipe_index, vectorizer

    # Build from scratch
    print("Building recipe embeddings (fallback TF-IDF)...")
    embeddings, recipe_index, vectorizer = build_embeddings()
    save_embeddings(embeddings, recipe_index, vectorizer)

    return embeddings, recipe_index, vectorizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "pinecone"), default="local")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild local TF-IDF cache")
    args = parser.parse_args()
    try:
        if args.backend == "pinecone":
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent / ".env")
            index_recipes_to_pinecone()
        else:
            embeddings, _, _ = index_recipes(force_rebuild=args.force_rebuild)
            print(f"Local index ready: {len(RECIPES)} recipes, shape {embeddings.shape}")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
