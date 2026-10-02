"""Run the 22 deterministic Golden v1 cases after the PTP-001 smoke review."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import time

from dotenv import load_dotenv

from evaluation.dataset import build_cases, write_dataset_files
from evaluation.evaluators import evaluate_case, langsmith_evaluator
from evaluation.langsmith_utils import TRACE_PROJECT, get_client, sync_frozen_dataset
from evaluation.run_eval import PASS_BARS, RUN_VERSIONS, _local_case_result, make_target

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / RUN_VERSIONS["deterministic"]
    / "remaining-22"
)
SMOKE_CASE_ID = "PTP-001"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def _write_batch_results(case_results: list[dict], elapsed_ms: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "case_results.json").write_text(
        json.dumps(case_results, indent=2, sort_keys=True) + "\n"
    )

    excluded = {
        "inputs",
        "expected",
        "predicted",
        "fixture_expectation_checks",
        "guardrail_violations",
    }
    flat_rows = []
    for result in case_results:
        row = {key: value for key, value in result.items() if key not in excluded}
        row["fixture_expectation_checks"] = json.dumps(
            result["fixture_expectation_checks"], sort_keys=True
        )
        row["guardrail_violations"] = json.dumps(
            result["guardrail_violations"], sort_keys=True
        )
        flat_rows.append(row)
    with (OUTPUT_DIR / "case_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    latencies = [float(result["latency_ms"]) for result in case_results]
    guardrail_totals = {
        key: sum(result["guardrail_violations"][key] for result in case_results)
        for key in case_results[0]["guardrail_violations"]
    }
    summary = {
        "dataset_version": "pantry-to-plan-golden-v1",
        "run_version": RUN_VERSIONS["deterministic"],
        "case_filter": "PTP-002..PTP-023",
        "case_count": len(case_results),
        "passed": sum(result["overall_case_pass"] for result in case_results),
        "failed": sum(not result["overall_case_pass"] for result in case_results),
        "hard_constraint_compliance": sum(
            result["hard_constraint_pass"] for result in case_results
        )
        / len(case_results),
        "fixture_expectation_accuracy": sum(
            result["fixture_expectation_score"] for result in case_results
        )
        / len(case_results),
        "inventory_correct_rate": sum(
            result["inventory_correct"] for result in case_results
        )
        / len(case_results),
        "shopping_list_correct_rate": sum(
            result["shopping_list_correct"] for result in case_results
        )
        / len(case_results),
        "trajectory_correct_rate": sum(
            result["trajectory_pass"] for result in case_results
        )
        / len(case_results),
        "failure_recovery_rate": sum(
            result["failure_recovery_pass"] for result in case_results
        )
        / len(case_results),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "total_runtime_ms": round(elapsed_ms, 3),
        "model_calls": sum(result["model_calls"] for result in case_results),
        "tool_calls": sum(result["tool_calls"] for result in case_results),
        "retry_count": sum(result["retry_count"] for result in case_results),
        "guardrail_violations": guardrail_totals,
        "pass_bars": PASS_BARS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ptp_001_excluded_after_smoke_review": True,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# Pantry to Plan remaining deterministic cases\n\n"
        f"- Cases: PTP-002 through PTP-023 ({len(case_results)})\n"
        f"- Passed: {summary['passed']}\n"
        f"- Failed: {summary['failed']}\n"
        f"- Hard-constraint compliance: {summary['hard_constraint_compliance']:.1%}\n"
        f"- Fixture expectation accuracy: {summary['fixture_expectation_accuracy']:.1%}\n"
        f"- Inventory correctness: {summary['inventory_correct_rate']:.1%}\n"
        f"- Shopping-list correctness: {summary['shopping_list_correct_rate']:.1%}\n"
        f"- Trajectory correctness: {summary['trajectory_correct_rate']:.1%}\n"
        f"- Failure recovery: {summary['failure_recovery_rate']:.1%}\n"
        f"- p95 local latency: {summary['p95_latency_ms']:.3f} ms\n"
        "- Scope: structured deterministic planner; vision excluded\n"
    )
    failures = [result for result in case_results if not result["overall_case_pass"]]
    lines = ["# Failure cases", ""]
    if not failures:
        lines.append("None.")
    else:
        for result in failures:
            failed_checks = [
                key
                for key, passed in result["fixture_expectation_checks"].items()
                if not passed
            ]
            lines.extend(
                [
                    f"## {result['case_id']} {result['fixture_id']}",
                    "",
                    f"- Failed fixture checks: {', '.join(failed_checks) or 'none'}",
                    f"- Hard constraint pass: {result['hard_constraint_pass']}",
                    f"- Inventory correct: {result['inventory_correct']}",
                    f"- Shopping list correct: {result['shopping_list_correct']}",
                    f"- Trajectory pass: {result['trajectory_pass']}",
                    f"- Trace: {result['langsmith_trace_url'] or 'unavailable'}",
                    "",
                ]
            )
    (OUTPUT_DIR / "failure_cases.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.eval", override=False)
    cases = [case for case in build_cases() if case["case_id"] != SMOKE_CASE_ID]
    if len(cases) != 22:
        raise RuntimeError(f"Refusing to run: expected 22 remaining cases, got {len(cases)}")
    digest = write_dataset_files(build_cases())
    target = make_target("deterministic")

    started = time.perf_counter()
    case_results = []
    for case in cases:
        output = target(case["inputs"])
        evaluation = evaluate_case(
            case["inputs"], output, {"expected": case["expected"]}
        )
        case_results.append(
            _local_case_result(
                case, output, evaluation, RUN_VERSIONS["deterministic"]
            )
        )

    client = get_client()
    dataset = sync_frozen_dataset(client, build_cases(), digest)
    examples = [
        example
        for example in client.list_examples(dataset_id=dataset.id)
        if example.metadata.get("case_id") != SMOKE_CASE_ID
    ]
    if len(examples) != 22:
        raise RuntimeError(
            f"Refusing to run: expected 22 remaining LangSmith examples, got {len(examples)}"
        )

    import os

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = TRACE_PROJECT
    experiment = client.evaluate(
        target,
        data=examples,
        evaluators=[langsmith_evaluator],
        experiment_prefix="baseline-deterministic-v1-remaining-22",
        max_concurrency=1,
        metadata={
            "dataset_version": "pantry-to-plan-golden-v1",
            "run_version": RUN_VERSIONS["deterministic"],
            "case_filter": "PTP-002..PTP-023",
            "trace_project": TRACE_PROJECT,
        },
    )
    rows = list(experiment)
    if len(rows) != 22:
        raise RuntimeError(f"Expected 22 remote result rows, got {len(rows)}")
    by_case = {result["case_id"]: result for result in case_results}
    for row in rows:
        run = row.get("run") if isinstance(row, dict) else getattr(row, "run", None)
        if run is None:
            continue
        case_id = run.inputs.get("case_id")
        if case_id not in by_case:
            raise RuntimeError(f"Unexpected remote case ID: {case_id}")
        by_case[case_id]["langsmith_trace_id"] = str(run.id)
        try:
            by_case[case_id]["langsmith_trace_url"] = client.get_run_url(run=run)
        except Exception:
            by_case[case_id]["langsmith_trace_url"] = None

    elapsed_ms = (time.perf_counter() - started) * 1000
    _write_batch_results(case_results, elapsed_ms)
    failed = [result["case_id"] for result in case_results if not result["overall_case_pass"]]
    print(
        json.dumps(
            {
                "cases_run": len(case_results),
                "passed": len(case_results) - len(failed),
                "failed": len(failed),
                "failed_case_ids": failed,
                "dataset_sha256": digest,
                "experiment_name": experiment.experiment_name,
                "results_dir": str(OUTPUT_DIR.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
