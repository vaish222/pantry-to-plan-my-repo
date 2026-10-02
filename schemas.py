"""Core data model for Pantry to Plan (Pydantic).

Everything is normalized to grams for this pass; unit conversion is out of scope.
Ingredient ids are lowercase snake_case (e.g. "chicken_breast") and are the
join key across pantry, recipes, and the shopping list.

Models are ordered so that every referenced type is defined before it is used.
"""

from enum import Enum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AppBaseModel(BaseModel):
    """Strict base contract shared by every pipeline model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


IngredientId = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")]
RecipeId = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")]
NonNegativeGrams = Annotated[float, Field(ge=0)]
PositiveGrams = Annotated[float, Field(gt=0)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]
PositiveInt = Annotated[int, Field(gt=0)]


class NormalizationStatus(str, Enum):
    """How a raw vision label became a canonical ingredient ID."""

    EXACT = "exact"
    ALIAS = "alias"
    UNMAPPED = "unmapped"


class RawDetectedIngredient(AppBaseModel):
    """One observation returned by vision, before canonical normalization."""

    raw_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    quantity_g: PositiveGrams | None = Field(
        default=None,
        description="Set only when an exact gram quantity is visually readable.",
    )
    confidence: UnitInterval
    evidence: str | None = Field(
        default=None,
        description="A short visible cue, not hidden reasoning.",
    )


class RawVisionResult(AppBaseModel):
    """Provider-neutral structured output from an image model."""

    ingredients: list[RawDetectedIngredient] = Field(default_factory=list)


# --------------------------------------------------------------------- leaves
class PantryItem(AppBaseModel):
    """A user-confirmed pantry item used by planning and inventory."""

    ingredient_id: IngredientId
    display_name: str = Field(min_length=1)
    quantity_g: NonNegativeGrams | None = None
    confidence: UnitInterval
    source_text: str | None = None


class PantryItemCandidate(PantryItem):
    """Machine-proposed pantry item awaiting user confirmation."""

    normalization_status: NormalizationStatus
    recipe_supported: bool = Field(
        description="True when at least one recipe uses this ingredient ID."
    )


class RecipeIngredient(AppBaseModel):
    ingredient_id: IngredientId
    quantity_g: PositiveGrams
    display_name: str | None = None


class ShoppingListItem(AppBaseModel):
    ingredient_id: IngredientId
    quantity_g: PositiveGrams                # grams still short after depletion
    aisle: str | None = None                 # grocery aisle group for the UI
    display_name: str | None = None
    contributing_recipe_ids: list[RecipeId] = Field(default_factory=list)


class TraceEvent(AppBaseModel):
    step: str = Field(min_length=1)           # retrieve, score, select, deplete
    day: int | None = Field(default=None, ge=1)
    message: str = Field(min_length=1)
    data: dict[str, Any] | None = None


class CandidateScore(AppBaseModel):
    recipe_id: RecipeId
    eligible: bool
    reject_reasons: list[str]    # frozen contract keeps this list[str]
    cuisine_score: UnitInterval
    calorie_delta: int
    pantry_coverage: UnitInterval
    repeat_penalty: UnitInterval
    total_score: float

    @model_validator(mode="after")
    def eligibility_matches_reasons(self) -> Self:
        if self.eligible and self.reject_reasons:
            raise ValueError("eligible scores cannot contain reject reasons")
        if not self.eligible and not self.reject_reasons:
            raise ValueError("ineligible scores require at least one reject reason")
        return self


class AppIssue(AppBaseModel):
    """Structured error/warning shared across modules (see the blueprint's
    error-code table: INVALID_IMAGE, NO_ELIGIBLE_RECIPE, ...). Used for
    eligibility rejections and dependency/pipeline failures. The frozen
    CandidateScore.reject_reasons and PlanResult.warnings stay list[str]."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    field: str | None = None
    recoverable: bool
    suggested_action: str | None = None


class RecipeCandidate(AppBaseModel):
    """Retriever output: a recipe reference plus its retrieval score. Exact
    recipe facts are always resolved from the corpus, never from the
    retriever; an unknown recipe_id is an error, not a generatable recipe."""

    recipe_id: RecipeId
    retrieval_score: float


class EligibilityResult(AppBaseModel):
    """constraints.validate_eligibility output. Eligibility runs before scoring
    and again before a DayPlan is appended; the vegetarian gate is never
    relaxed."""

    recipe_id: RecipeId
    eligible: bool
    reject_reasons: list[AppIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def eligibility_matches_reasons(self) -> Self:
        if self.eligible and self.reject_reasons:
            raise ValueError("eligible results cannot contain reject reasons")
        if not self.eligible and not self.reject_reasons:
            raise ValueError("ineligible results require at least one reject reason")
        return self


class Shortage(AppBaseModel):
    """inventory.apply_recipe output: unmet demand for one ingredient after
    depleting the pantry for a selected recipe."""

    ingredient_id: IngredientId
    quantity_g: PositiveGrams
    aisle: str | None = None
    contributing_recipe_ids: list[RecipeId] = Field(default_factory=list)


# ------------------------------------------------------------------ composite
class Recipe(AppBaseModel):
    recipe_id: RecipeId
    title: str = Field(min_length=1)
    cuisine_tags: list[str] = Field(min_length=1)
    calories_per_serving: PositiveInt
    vegetarian: bool
    ingredients: list[RecipeIngredient] = Field(min_length=1)
    # Ordered, concise cooking steps. Source of truth for the UI drilldown;
    # not part of the retrieval/embedding text (title + ingredients + cuisine).
    instructions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    cook_time_min: PositiveInt | None = None


class DayPlan(AppBaseModel):
    day: int = Field(ge=1)       # 1..N, not a real date for the demo
    recipe: Recipe
    cuisine_match: bool
    calorie_delta: int           # actual - target
    pantry_coverage: UnitInterval  # fraction of ingredient lines already on hand
    vegetarian_required: bool    # request-level hard constraint
    fallback: bool = False       # picked on closest-calorie fallback, flagged
    score: CandidateScore | None = None
    flags: list[str] = Field(default_factory=list)


# ----------------------------------------------------------- request/response
class PantryParseResult(AppBaseModel):
    """Vision/normalization output consumed by the confirmation screen."""

    items: list[PantryItemCandidate]
    warnings: list[str] = Field(default_factory=list)
    issues: list[AppIssue] = Field(default_factory=list)
    model_latency_ms: int | None = Field(default=None, ge=0)
    provider: str | None = None


class PantryState(AppBaseModel):
    """The confirmed pantry the planner operates on.

    Distinct from PantryParseResult: that is raw vision output (with confidence
    and source text) awaiting user confirmation; this is the authoritative
    "what is true now" state the pipeline threads through each day, depleting
    quantities as recipes are applied. Items reuse PantryItem so state flows
    unchanged into PlanResult.final_pantry.
    """

    items: list[PantryItem]
    as_of: str | None = None     # fixed demo date, not a real plan date


class PlanningRequest(AppBaseModel):
    cuisines: list[str] = Field(min_length=1)
    dinner_calorie_target: PositiveInt
    days: int = Field(ge=1, le=7)  # fixtures currently exercise 1..7
    vegetarian_required: bool
    goal: str = Field(pattern=r"^(general|nutritional|kids)$")


class PlanResult(AppBaseModel):
    requested_days: int = Field(ge=1, le=7)
    day_plans: list[DayPlan]
    final_pantry: list[PantryItem]
    shopping_list: list[ShoppingListItem]
    warnings: list[str]          # frozen contract; use AppIssue codes as strings
    trace: list[TraceEvent]


# ------------------------------------------------------------------ evaluation
# Person 4's eval contract, drafted from the blueprint's run_eval example.
# Fields may firm up when the eval harness lands; checks/metrics stay open dicts
# so new assertions don't force a schema change.
class CaseEvaluation(AppBaseModel):
    case_id: str
    passed: bool
    checks: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(AppBaseModel):
    cases: list[CaseEvaluation]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
