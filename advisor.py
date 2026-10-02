"""Agentic planning advisor (advisor.py, Person 3).

Injected decision-maker for the two agentic strategies:
  #3 whole-week planning  -> plan_week(ctx) proposes an ordered week
  #1 relax-and-repair      -> choose_relaxation(ctx) picks a fixed relaxation

The advisor only DECIDES (which recipes, in what order; which relaxation to
try). All scoring, eligibility, the vegetarian gate, depletion, and
shopping-list reconciliation stay in deterministic Python (pipeline.py /
scoring.py / constraints.py / inventory.py), so the model never touches the
arithmetic and evals stay reproducible with the deterministic fake.

No schemas.py / recipes.py / retrieval changes: the advisor's I/O types live
here (not in the frozen shared contract), and its decisions are surfaced through
PlanResult.trace / PlanResult.warnings / DayPlan.flags. Cross-cuisine widening
reuses retriever.search with a broadened request copy, within its existing
contract.

The real adapter calls OpenAI directly (the openai>=2 Responses API with
Pydantic structured output), matching vision.py's client pattern so the whole
project shares one OpenAI client and one dependency.
"""

from typing import Protocol

from pydantic import BaseModel

from config import ADVISOR_MODEL

# ------------------------------------------------------------- relaxation menu
# Closed set the repair loop understands. OUT_OF_BAND is deliberately excluded:
# the calorie band is a soft flag, never a selection filter, so an out-of-band
# pick never blocks a day and "relaxing" it would be a no-op. The vegetarian
# gate is never in this menu; it is never relaxed.
ALLOW_REPEAT = "ALLOW_REPEAT"        # permit reusing an already-used recipe
CROSS_CUISINE = "CROSS_CUISINE"      # pull from other cuisines (flagged), last resort
GIVE_UP = "GIVE_UP"                  # stop; return a partial plan
RELAXATION_MENU = (ALLOW_REPEAT, CROSS_CUISINE, GIVE_UP)


# ---------------------------------------------------------------- advisor I/O
class CandidateSummary(BaseModel):
    """A scored candidate, flattened for the advisor. Numbers are precomputed by
    the deterministic scorer; the advisor only reasons over them, never derives
    them."""

    recipe_id: str
    title: str
    cuisine_tags: list[str]
    calories_per_serving: int
    vegetarian: bool
    ingredient_ids: list[str]
    pantry_coverage: float
    cuisine_match: bool
    calorie_delta: int
    total_score: float
    eligible: bool


class WeekContext(BaseModel):
    days: int
    cuisines: list[str]
    dinner_calorie_target: int
    calorie_low: int
    calorie_high: int
    vegetarian_required: bool
    goal: str
    pantry_ingredient_ids: list[str]
    objectives: list[str]            # e.g. minimize_shopping_list, smart_depletion, variety
    candidates: list[CandidateSummary]


class WeekProposal(BaseModel):
    """The agent's proposed week: an ordered list of recipe_ids (length <= days).
    The deterministic core validates and materializes it; anything invalid
    (unknown id, repeat, veg violation) is dropped, so the proposal cannot break
    an invariant."""

    ordered_recipe_ids: list[str]
    rationale: str = ""


class RepairContext(BaseModel):
    day: int
    reason: str
    cuisines: list[str]
    vegetarian_required: bool
    used_recipe_ids: list[str]
    tried_relaxations: list[str]
    menu: list[str]                  # the closed RELAXATION_MENU


class RelaxationChoice(BaseModel):
    choice: str                      # one of RELAXATION_MENU; anything else -> treated as GIVE_UP
    rationale: str = ""


class PlannerAdvisor(Protocol):
    """Matches the injected-decider contract. Mirrors the Retriever Protocol
    pattern so tests can pass a deterministic fake."""

    def plan_week(self, ctx: WeekContext) -> WeekProposal:
        ...

    def choose_relaxation(self, ctx: RepairContext) -> RelaxationChoice:
        ...


# --------------------------------------------------------------- LLM adapter
_WEEK_SYSTEM = (
    "You are a meal-plan sequencer. Given scored recipe candidates, propose an "
    "ordered week of distinct recipe_ids (at most `days`). You only choose and "
    "order recipes; you never invent recipes, ingredients, or numbers. Optimize "
    "the stated objectives: minimize_shopping_list means prefer a set of recipes "
    "whose ingredients overlap each other and the pantry; smart_depletion means "
    "order days to use limited pantry stock early; variety means avoid repeating "
    "similar recipes or the same dominant ingredients. Only use recipe_ids that "
    "appear in the candidate list. Prefer eligible candidates."
)

_RELAX_SYSTEM = (
    "A day of the meal plan cannot be filled under the current rules. Choose ONE "
    "relaxation from the provided menu to try next, or GIVE_UP to stop with a "
    "partial plan. You must return a choice that is exactly one of the menu "
    "values. The vegetarian requirement is never relaxable and is not on the menu."
)


class CompatiblePlannerAdvisor:
    """Advisor backed by Ollama with automatic Nebius failover.

    openai is imported lazily so the planner and the fake-based tests import fine
    without an API key. `temperature` suits gpt-4o (the default ADVISOR_MODEL);
    drop it if ADVISOR_MODEL is pointed at a model that only allows the default.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.temperature = temperature
        self.provider_name: str | None = None

    def plan_week(self, ctx: WeekContext) -> WeekProposal:
        return self._decide(_WEEK_SYSTEM, ctx, WeekProposal)

    def choose_relaxation(self, ctx: RepairContext) -> RelaxationChoice:
        return self._decide(_RELAX_SYSTEM, ctx, RelaxationChoice)

    def _decide(self, system: str, ctx: BaseModel, output_model: type[BaseModel]):
        """One structured Responses call: system prompt + JSON-encoded context in,
        a validated `output_model` instance out."""
        from llm_provider import json_schema_format, parse_json_model, run_with_fallback

        def invoke(endpoint, client):
            response = client.chat.completions.create(
                model=endpoint.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": ctx.model_dump_json()},
                ],
                response_format=json_schema_format(output_model),
            )
            return parse_json_model(response.choices[0].message.content, output_model)

        result, self.provider_name = run_with_fallback(
            invoke, preferred=self.provider_name
        )
        return result


OpenAIPlannerAdvisor = CompatiblePlannerAdvisor
