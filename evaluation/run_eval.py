"""CLI for deterministic Pantry-to-Plan evaluations.

The smoke workflow deliberately accepts one case ID. It cannot accidentally
execute the full 23-case dataset.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import langsmith as ls
from dotenv import load_dotenv

from evaluation.dataset import (
    DATASET_VERSION,
    build_cases,
    find_case,
    write_dataset_files,
)
from evaluation.evaluators import evaluate_case, langsmith_evaluator
from evaluation.fakes import FakePlannerAdvisor
from evaluation.langsmith_utils import (
    TRACE_PROJECT,
    get_client,
    run_one_case_experiment,
    sync_frozen_dataset,
)
from pipeline import generate_plan
from retrieval import LocalRetriever
from schemas import PantryItem, PantryState, PlanningRequest
from telemetry import traced

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN_VERSIONS = {
    "deterministic": "baseline-deterministic-v1",
    "agentic-fake": "baseline-agentic-fake-v1",
}
PASS_BARS = {
    "hard_constraint_compliance": 1.0,
    "fixture_expectation_accuracy": 0.90,
    "inventory_and_shopping_invariants": 1.0,
    "trajectory_correctness": 0.90,
    "failure_recovery": 0.95,
    "local_p95_latency_seconds": 5.0,
    "hard_guardrails": 0,
}


@traced(name="load_fixture", run_type="tool")
def load_case_inputs(inputs: dict) -> tuple[PantryState, PlanningRequest]:
    pantry = PantryState(
        items=[
            PantryItem(
                ingredient_id=ingredient_id,
                display_name=ingredient_id.replace("_", " "),
                quantity_g=float(quantity),
                confidence=1.0,
            )
            for ingredient_id, quantity in inputs["pantry"]["items"].items()
        ],
        as_of=inputs["pantry"].get("as_of"),
    )
    return pantry, build_request(inputs["request"])


@traced(name="build_request", run_type="tool")
def build_request(request: dict) -> PlanningRequest:
    return PlanningRequest.model_validate(request)


class TracingRetriever:
    def __init__(self):
        self.backend = "local-tfidf"
        self.calls = 0
        self._delegate = LocalRetriever()

    @traced(name="retrieval", run_type="retriever")
    def search(self, pantry, request, top_k):
        self.calls += 1
        return self._delegate.search(pantry, request, top_k)


def _metadata(inputs: dict, *, mode: str, latency_ms: int) -> dict[str, Any]:
    return {
        "case_id": inputs["case_id"],
        "fixture_id": inputs["fixture_id"],
        "dataset_version": DATASET_VERSION,
        "scenario_type": inputs["scenario_type"],
        "run_version": RUN_VERSIONS[mode],
        "planner_mode": mode,
        "advisor_mode": "none" if mode == "deterministic" else "fake",
        "retrieval_backend": "local-tfidf",
        "prompt_version": "none" if mode == "deterministic" else "fake-advisor-v1",
        "latency_ms": latency_ms,
        "token_usage": None,
        "error": None,
        "notes": "Structured planner evaluation; vision excluded.",
    }


def make_target(mode: str):
    def target(inputs: dict) -> dict:
        started = time.perf_counter()
        pantry, request = load_case_inputs(inputs)
        retriever = TracingRetriever()
        advisor = None if mode == "deterministic" else FakePlannerAdvisor()
        result = generate_plan(
            pantry,
            request,
            retriever,
            temperature=0.0,
            advisor=advisor,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        metadata = _metadata(inputs, mode=mode, latency_ms=latency_ms)
        current_run = ls.get_current_run_tree()
        if current_run is not None:
            current_run.metadata.update(
                {
                    **metadata,
                    "expected_output": "LangSmith reference_outputs.expected",
                    "predicted_output": {
                        "recipe_ids": [day.recipe.recipe_id for day in result.day_plans],
                        "warnings": result.warnings,
                    },
                }
            )
        return {
            "predicted": result.model_dump(mode="json"),
            "planner_mode": mode,
            "advisor_mode": metadata["advisor_mode"],
            "retrieval_backend": metadata["retrieval_backend"],
            "latency_ms": latency_ms,
            "model_calls": 0,
            "tool_calls": retriever.calls,
            "retry_count": sum(1 for event in result.trace if event.step == "relax"),
            "token_usage": None,
            "error": None,
        }

    target.__name__ = f"pantry_to_plan_{mode.replace('-', '_')}"
    return target


def _local_case_result(case: dict, target_output: dict, evaluation: dict, run_version: str) -> dict:
    return {
        "case_id": case["case_id"],
        "fixture_id": case["fixture_id"],
        "scenario_type": case["scenario_type"],
        "description": case["description"],
        "inputs": case["inputs"],
        "expected": case["expected"],
        "predicted": target_output["predicted"],
        **evaluation,
        "latency_ms": target_output["latency_ms"],
        "model_calls": target_output["model_calls"],
        "tool_calls": target_output["tool_calls"],
        "retry_count": target_output["retry_count"],
        "token_usage": target_output["token_usage"],
        "error": target_output["error"],
        "langsmith_trace_id": None,
        "langsmith_trace_url": None,
        "run_version": run_version,
    }


def _write_results(case_result: dict, mode: str) -> Path:
    run_version = RUN_VERSIONS[mode]
    output_dir = RESULTS_DIR / run_version / f"smoke-{case_result['case_id'].lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_results.json").write_text(
        json.dumps([case_result], indent=2, sort_keys=True) + "\n"
    )
    flat = {
        key: value
        for key, value in case_result.items()
        if key not in {"inputs", "expected", "predicted", "fixture_expectation_checks", "guardrail_violations"}
    }
    for key in ("fixture_expectation_checks", "guardrail_violations"):
        flat[key] = json.dumps(case_result[key], sort_keys=True)
    with (output_dir / "case_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    summary = {
        "dataset_version": DATASET_VERSION,
        "run_version": run_version,
        "case_filter": case_result["case_id"],
        "case_count": 1,
        "passed": int(case_result["overall_case_pass"]),
        "failed": int(not case_result["overall_case_pass"]),
        "pass_bars": PASS_BARS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smoke_run_only": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(
        "# Pantry to Plan evaluation smoke run\n\n"
        f"- Case: {case_result['case_id']}\n"
        f"- Run version: {run_version}\n"
        f"- Overall pass: {case_result['overall_case_pass']}\n"
        f"- Fixture expectation score: {case_result['fixture_expectation_score']:.1%}\n"
        f"- Latency: {case_result['latency_ms']} ms\n"
        "- Scope: one structured planner case; vision excluded\n"
    )
    failures = [] if case_result["overall_case_pass"] else [case_result["case_id"]]
    (output_dir / "failure_cases.md").write_text(
        "# Failure cases\n\n" + ("None.\n" if not failures else "\n".join(f"- {x}" for x in failures) + "\n")
    )
    return output_dir


def _attach_remote_run(case_result: dict, experiment_results, client) -> None:
    rows = list(experiment_results)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one remote result row; got {len(rows)}")
    row = rows[0]
    run = row.get("run") if isinstance(row, dict) else getattr(row, "run", None)
    if run is None:
        return
    case_result["langsmith_trace_id"] = str(run.id)
    try:
        case_result["langsmith_trace_url"] = client.get_run_url(run=run)
    except Exception:
        case_result["langsmith_trace_url"] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactly one Pantry-to-Plan golden case")
    parser.add_argument("--case-id", default="PTP-001")
    parser.add_argument("--mode", choices=sorted(RUN_VERSIONS), default="deterministic")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload/sync the 23-case dataset and evaluate the selected single case in LangSmith.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.eval", override=False)
    cases = build_cases()
    digest = write_dataset_files(cases)
    case = find_case(cases, args.case_id)
    target = make_target(args.mode)

    # Always validate locally first without enabling trace export.
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    target_output = target(case["inputs"])
    evaluation = evaluate_case(case["inputs"], target_output, {"expected": case["expected"]})
    case_result = _local_case_result(case, target_output, evaluation, RUN_VERSIONS[args.mode])

    if args.upload:
        client = get_client()
        dataset = sync_frozen_dataset(client, cases, digest)
        experiment = run_one_case_experiment(
            client=client,
            dataset=dataset,
            case_id=case["case_id"],
            target=target,
            evaluator=langsmith_evaluator,
            experiment_prefix=f"{RUN_VERSIONS[args.mode]}-{case['case_id'].lower()}",
            run_version=RUN_VERSIONS[args.mode],
        )
        _attach_remote_run(case_result, experiment, client)

    output_dir = _write_results(case_result, args.mode)
    print(json.dumps({
        "case_id": case_result["case_id"],
        "overall_case_pass": case_result["overall_case_pass"],
        "dataset_sha256": digest,
        "langsmith_trace_id": case_result["langsmith_trace_id"],
        "langsmith_trace_url": case_result["langsmith_trace_url"],
        "results_dir": str(output_dir.relative_to(ROOT)),
    }, indent=2))
    return 0 if case_result["overall_case_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
