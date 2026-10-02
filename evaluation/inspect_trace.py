"""Read-only verification for a completed LangSmith evaluation trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.eval", override=False)
    client = Client()
    runs = list(client.list_runs(trace_id=args.trace_id))
    roots = [run for run in runs if run.parent_run_id is None]
    feedback = []
    for root in roots:
        feedback.extend(client.list_feedback(run_ids=[root.id]))
    print(
        json.dumps(
            {
                "trace_id": args.trace_id,
                "run_count": len(runs),
                "runs": [
                    {
                        "name": run.name,
                        "run_type": run.run_type,
                        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
                        "error": run.error,
                    }
                    for run in sorted(runs, key=lambda item: item.start_time)
                ],
                "feedback": [
                    {"key": item.key, "score": item.score, "value": item.value}
                    for item in feedback
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
