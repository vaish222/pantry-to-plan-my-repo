"""LangSmith dataset synchronization and one-case experiment helpers."""

from __future__ import annotations

import os
from typing import Callable

from evaluation.dataset import DATASET_NAME

TRACE_PROJECT = "pantry-to-plan-week4-eval"


class DatasetDriftError(RuntimeError):
    """Raised when a frozen LangSmith dataset differs from local Golden v1."""


def require_credentials() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. Export it in the shell; do not commit it."
        )


def get_client():
    from langsmith import Client

    require_credentials()
    return Client()


def sync_frozen_dataset(client, cases: list[dict], digest: str):
    """Create Golden v1 once, or reuse it only when every fixture hash matches."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id))
        expected_hashes = {
            case["case_id"]: case["metadata"]["fixture_sha256"] for case in cases
        }
        actual_hashes = {
            example.metadata.get("case_id"): example.metadata.get("fixture_sha256")
            for example in existing
        }
        dataset_hashes = {
            example.metadata.get("dataset_sha256") for example in existing
        }
        if (
            len(existing) != len(cases)
            or actual_hashes != expected_hashes
            or dataset_hashes != {digest}
        ):
            raise DatasetDriftError(
                "Existing LangSmith dataset differs from frozen local Golden v1. "
                "Create a new dataset version instead of editing this one."
            )
        return dataset

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Pantry-to-Plan Golden Dataset v1: 23 structured planner fixtures. "
            "Vision photos are metadata only and are not inputs to this planner suite."
        ),
    )
    examples = [
        {
            "inputs": case["inputs"],
            "outputs": {"expected": case["expected"]},
            "metadata": {**case["metadata"], "dataset_sha256": digest},
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset


def run_one_case_experiment(
    *,
    client,
    dataset,
    case_id: str,
    target: Callable[[dict], dict],
    evaluator: Callable,
    experiment_prefix: str,
    run_version: str,
):
    """Evaluate exactly one metadata-filtered example, never the whole dataset."""
    examples = list(
        client.list_examples(dataset_id=dataset.id, metadata={"case_id": case_id})
    )
    if len(examples) != 1:
        raise RuntimeError(
            f"Refusing to run: expected one LangSmith example for {case_id}, "
            f"found {len(examples)}"
        )
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = TRACE_PROJECT
    return client.evaluate(
        target,
        data=examples,
        evaluators=[evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
        metadata={
            "dataset_version": DATASET_NAME,
            "run_version": run_version,
            "case_filter": case_id,
            "trace_project": TRACE_PROJECT,
        },
    )

