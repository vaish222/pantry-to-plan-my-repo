"""Deterministic test doubles used only by evaluation modes."""

from advisor import (
    ALLOW_REPEAT,
    CROSS_CUISINE,
    GIVE_UP,
    RelaxationChoice,
    RepairContext,
    WeekContext,
    WeekProposal,
)
from telemetry import traced


class FakePlannerAdvisor:
    """Reproducible advisor that only chooses from supplied candidates/menu."""

    @traced(name="advisor_call", run_type="tool")
    def plan_week(self, ctx: WeekContext) -> WeekProposal:
        eligible = [c for c in ctx.candidates if c.eligible]
        eligible.sort(
            key=lambda c: (
                not c.cuisine_match,
                -c.total_score,
                -c.pantry_coverage,
                abs(c.calorie_delta),
                c.recipe_id,
            )
        )
        return WeekProposal(
            ordered_recipe_ids=[c.recipe_id for c in eligible[: ctx.days]],
            rationale="Deterministic score ordering for reproducible evaluation.",
        )

    @traced(name="relaxation_decision", run_type="tool")
    def choose_relaxation(self, ctx: RepairContext) -> RelaxationChoice:
        for choice in (ALLOW_REPEAT, CROSS_CUISINE, GIVE_UP):
            if choice in ctx.menu and choice not in ctx.tried_relaxations:
                return RelaxationChoice(
                    choice=choice,
                    rationale="First untried allowed relaxation in fixed evaluation order.",
                )
        return RelaxationChoice(choice=GIVE_UP, rationale="No untried relaxation remains.")

