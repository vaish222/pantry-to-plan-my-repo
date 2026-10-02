"""UI-supporting business logic (app_services.py, Person 4).

Everything the Streamlit app needs that is NOT a widget lives here so app.py
stays a thin view layer: preference persistence, pantry sourcing, request
building, retrieval, running the planner, and copy helpers. No Streamlit import
in this module, and every function returns plain data or a typed schema object.

The upstream modules vision.py (P1 image reader) and retrieval.py (P2 retriever)
are wired in. This module calls them when present and falls back to a runnable
local path otherwise (no API key, or a checkout missing those modules), so the UI
works end to end in every environment.
"""

import json
import os
import random
from pathlib import Path

from config import ALL_CUISINES, GOAL_CALORIE_TOLERANCE
from inventory import build_shopping_list  # noqa: F401  (re-exported for callers/tests)
from pipeline import generate_plan
from repository import load_recipes
from schemas import (
    NormalizationStatus,
    PantryItem,
    PantryItemCandidate,
    PantryParseResult,
    PantryState,
    PlanningRequest,
    PlanResult,
    RecipeCandidate,
)
from vision_ui_helpers import candidates_to_editor_rows, editor_rows_to_pantry

ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = ROOT / "fixtures"
DEMO_FIXTURE = FIXTURES_DIR / "01_well_stocked_veg_italian.json"
PREFS_FILE = ROOT / "preferences.json"
PREFS_DEFAULT_FILE = ROOT / "preferences.default.json"
ENV_DEMO_FILE = ROOT / ".env.demo"
ENV_FILE = ROOT / ".env"


def _load_env_demo() -> None:
    """Load KEY=VALUE lines from .env and .env.demo. Existing
    environment values win, so a real export is never overridden. The file is
    gitignored and never logged.

    LangSmith/LangChain tracing vars are intentionally NOT loaded: this project
    does not enable tracing, and pulling them in points LangChain at a LangSmith
    endpoint whose key 403s and floods stderr. We force tracing off below.
    """
    for env_file in (ENV_FILE, ENV_DEMO_FILE):
        if not env_file.exists():
            continue
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.upper().startswith(("LANGSMITH", "LANGCHAIN")):
                continue
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    # Belt and suspenders: keep LangChain from calling LangSmith at all.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# Populate environment at import so live providers see supplied credentials.
_load_env_demo()

# Option lists sourced from the shared contract so the UI can never drift from
# what the planner actually accepts.
CUISINES = list(ALL_CUISINES)
GOALS = list(GOAL_CALORIE_TOLERANCE)

# Items at or below this confidence (or missing a quantity) are surfaced for
# review on the confirm step. The clean demo pantry trips none of these.
LOW_CONFIDENCE_THRESHOLD = 0.6


# Re-exported so app.py can catch a hard vision failure via svc.VisionError
# without importing vision.py itself. A stand-in keeps the import working on a
# checkout where vision.py is absent.
try:
    from vision import VisionPipelineError as VisionError  # type: ignore
except ImportError:  # pragma: no cover - vision.py is present in this repo
    class VisionError(RuntimeError):
        """Fallback when vision.py is unavailable; carries an optional AppIssue."""

        def __init__(self, issue=None):
            super().__init__(getattr(issue, "message", "Vision parsing failed."))
            self.issue = issue

# Safety net only; the real defaults live in preferences.default.json.
_FALLBACK_PREFS = {
    "cuisines": ["italian"],
    "dinner_calorie_target": 600,
    "days": 3,
    "vegetarian_required": False,
    "goal": "general",
}


# --------------------------------------------------------------- preferences
def load_preferences() -> dict:
    """Persisted user prefs if present, else the committed defaults file, else a
    hardcoded safety net."""
    for path in (PREFS_FILE, PREFS_DEFAULT_FILE):
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    return dict(_FALLBACK_PREFS)


def save_preferences(prefs: dict) -> None:
    """Persist changed preferences to preferences.json (runtime, gitignored)."""
    with open(PREFS_FILE, "w") as fh:
        json.dump(prefs, fh, indent=2)


# ------------------------------------------------------------ pantry sourcing
def _candidates_from_fixture(path: Path) -> list[PantryItemCandidate]:
    """Load a pantry fixture as machine-proposed candidates for the confirm step.

    The fixture keys are already canonical corpus ingredient IDs, so each maps
    exactly and is recipe-supported when it appears in the corpus. Confidence is
    a full 1.0 (a curated fixture, not a noisy vision read), and the detected
    label is the human-readable form of the id.
    """
    from normalization import corpus_ingredient_ids

    supported = corpus_ingredient_ids()
    data = json.loads(path.read_text())
    return [
        PantryItemCandidate(
            ingredient_id=ing_id,
            display_name=ing_id.replace("_", " ").title(),
            quantity_g=float(grams),
            confidence=1.0,
            source_text=ing_id.replace("_", " "),
            normalization_status=NormalizationStatus.EXACT,
            recipe_supported=ing_id in supported,
        )
        for ing_id, grams in data["pantry"]["items"].items()
    ]


def load_demo_parse_result() -> PantryParseResult:
    """The bundled well-stocked vegetarian Italian pantry, as if freshly parsed.

    Used both when the user skips the photo and as the fallback parse result for
    an uploaded photo when no live provider is configured.
    """
    return PantryParseResult(
        items=_candidates_from_fixture(DEMO_FIXTURE),
        warnings=[],
        provider="demo",
    )


def parse_pantry_image(file_bytes: bytes) -> PantryParseResult:
    """Read a pantry photo into a PantryParseResult.

    Uses Ollama for vision with a configured Nebius fallback. Without a live
    provider (or if vision.py is unavailable) it falls back to the bundled
    demo pantry so the wizard stays runnable end to end. A hard vision failure
    (VisionPipelineError) is not swallowed, so the UI can surface the issue.
    """
    try:
        from vision import CompatibleVisionProvider
        from vision import parse_pantry_image as run_vision
    except ImportError:
        return load_demo_parse_result()
    from llm_provider import provider_chain
    if not provider_chain("vision"):
        return load_demo_parse_result()
    return run_vision(
        file_bytes,
        CompatibleVisionProvider(),
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
    )


def flagged_rows(rows: list[dict]) -> list[str]:
    """Display names of included table rows worth a second look: low confidence
    or no quantity. Recomputed from the live table so fixing or unchecking a
    row clears its flag. Empty for the clean demo pantry."""
    flagged: list[str] = []
    for row in rows:
        if not bool(row.get("include", True)):
            continue
        conf = row.get("confidence")
        low_confidence = conf not in (None, "") and float(conf) < LOW_CONFIDENCE_THRESHOLD
        missing_quantity = row.get("quantity_g") in (None, "")
        if low_confidence or missing_quantity:
            name = str(row.get("display_name") or row.get("ingredient_id") or "").strip()
            if name:
                flagged.append(name)
    return flagged


def rows_from_parse_result(parse_result: PantryParseResult) -> list[dict]:
    """One editable table row per proposed item for the confirm step. Each row
    carries the Include toggle (checked), the editable name/id/quantity, and the
    read-only confidence, detected label, mapping, and recipe-support fields."""
    return candidates_to_editor_rows(parse_result.items)


def manual_ingredient_row(name: str) -> dict:
    """A single confirm-step row for a user-typed ingredient, normalized to the
    corpus vocabulary so its mapping and recipe-support columns are accurate."""
    from normalization import IngredientNormalizer

    normalized = IngredientNormalizer().normalize(name)
    candidate = PantryItemCandidate(
        ingredient_id=normalized.ingredient_id,
        display_name=normalized.display_name,
        quantity_g=None,
        confidence=1.0,
        source_text="user_added",
        normalization_status=normalized.status,
        recipe_supported=normalized.recipe_supported,
    )
    return candidates_to_editor_rows([candidate])[0]


# Demo-only quantity range: 50g to 500g inclusive, in 50g steps.
_DEMO_QUANTITY_CHOICES = list(range(50, 501, 50))


def fill_random_quantities(rows: list[dict], rng: random.Random | None = None) -> list[dict]:
    """Return rows with any blank quantity filled by a random demo amount.

    A convenience for demos: an uploaded photo often yields items with no
    readable quantity, and this fills each blank with a random value between
    50g and 500g (in 50g increments). Rows that already have a quantity are
    left untouched, so a user's own edits are preserved. Pass a seeded rng for
    reproducibility.
    """
    picker = rng or random
    filled: list[dict] = []
    for row in rows:
        new_row = dict(row)
        if new_row.get("quantity_g") in (None, ""):
            new_row["quantity_g"] = float(picker.choice(_DEMO_QUANTITY_CHOICES))
        filled.append(new_row)
    return filled


def pantry_from_rows(rows: list[dict]) -> PantryState:
    """Turn edited confirm-step rows into the typed pantry the planner uses.

    Only rows with Include checked are kept; duplicate canonical IDs are merged
    (quantities summed, unknown if any is unknown) and malformed IDs are rejected
    by the schema. Delegates to the shared, unit-tested vision_ui_helpers logic.
    """
    return editor_rows_to_pantry(rows)


# ------------------------------------------------------------------- request
def build_request(prefs: dict) -> PlanningRequest:
    """Construct the typed PlanningRequest from a preferences dict."""
    return PlanningRequest(
        cuisines=list(prefs["cuisines"]),
        dinner_calorie_target=int(prefs["dinner_calorie_target"]),
        days=int(prefs["days"]),
        vegetarian_required=bool(prefs["vegetarian_required"]),
        goal=str(prefs["goal"]),
    )


# ----------------------------------------------------------------- retrieval
class _CorpusRetriever:
    """Local stand-in matching pipeline.Retriever until P2's retrieval.py lands.

    Mirrors tests/fakes.FakeRetriever: a cuisine prefilter over the corpus,
    sorted by id, no top_k truncation and no vegetarian filter (the planner is
    the authoritative place the vegetarian gate is enforced).
    """

    def __init__(self):
        self._recipes = load_recipes()

    def search(
        self, pantry: PantryState, request: PlanningRequest, top_k: int
    ) -> list[RecipeCandidate]:
        matched = [
            r for r in self._recipes if any(c in r.cuisine_tags for c in request.cuisines)
        ]
        matched.sort(key=lambda r: r.recipe_id)
        return [RecipeCandidate(recipe_id=r.recipe_id, retrieval_score=1.0) for r in matched]


def get_retriever():
    """Real retriever (auto: Pinecone with local fallback) when available, else
    the local corpus stand-in. The factory reads RETRIEVAL_BACKEND (auto by
    default), so the app queries Pinecone when creds are present and degrades to
    TF-IDF local otherwise."""
    try:
        from retrieval import create_retriever  # type: ignore
    except ImportError:
        return _CorpusRetriever()
    return create_retriever()


# ------------------------------------------------------------------- advisor
def advisor_status() -> tuple[bool, str]:
    """Whether the opt-in agentic planner can run. Returns (available, reason);
    reason is a short note to show when it is disabled."""
    from llm_provider import provider_chain
    if not provider_chain():
        return False, "Start Ollama or set NEBIUS_API_KEY to enable smart planning."
    return True, ""


def build_advisor():
    """The provider-neutral PlannerAdvisor. Only call when advisor_status()
    reports available. Sampling temperature comes from settings (ADVISOR_TEMPERATURE)
    so agentic re-runs on the same pantry vary instead of pinning one plan."""
    from advisor import CompatiblePlannerAdvisor
    from config import load_settings

    settings = load_settings()
    return CompatiblePlannerAdvisor(
        temperature=settings.advisor_temperature,
    )


# ---------------------------------------------------------------- run planner
def run_plan(pantry: PantryState, request: PlanningRequest, advisor=None, progress=None) -> PlanResult:
    """Drive the deterministic pipeline, emitting coarse stage strings through an
    optional progress(msg) callback for the UI to display."""

    def emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    emit("Reading your preferences...")
    retriever = get_retriever()
    emit("Retrieving candidate recipes...")
    emit(f"Scoring recipes and building your {request.days}-day plan...")
    result = generate_plan(pantry, request, retriever, advisor=advisor)
    emit("Aggregating your shopping list...")
    return result


# --------------------------------------------------------------- copy helpers
_WARNING_TEXT = {
    "NO_ELIGIBLE_RECIPE": "We ran out of recipes that fit your preferences, so this plan "
    "is shorter than you asked for. Try more cuisines or a wider calorie goal.",
    "RELAXED_CROSS_CUISINE": "To fill every day we included a dish from another cuisine "
    "(flagged on the day it appears).",
    "RELAXED_REPEAT": "To fill every day a recipe had to repeat.",
    "UNKNOWN_RECIPE_ID": "A retrieved recipe was not found in the corpus and was skipped.",
}

FLAG_LABELS = {
    "cross_cuisine": "Different cuisine",
    "calorie_out_of_band": "Outside calorie band",
}


def describe_warning(code: str) -> str:
    """Friendly text for a PlanResult warning code (falls back to the raw code)."""
    return _WARNING_TEXT.get(code, code)
