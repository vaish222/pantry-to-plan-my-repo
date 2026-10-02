# Pantry to Plan evaluation harness

This harness evaluates the structured planner against Golden Dataset v1. It
does not call the vision provider and does not change planner selection,
retrieval, scoring, advisor, or inventory behavior.

## Credentials

Set LangSmith credentials in your shell, or place the same keys in the
gitignored `.env` or `.env.eval` file. Never commit them:

```bash
export LANGSMITH_API_KEY="..."
# Required only when the API key has access to multiple workspaces:
export LANGSMITH_WORKSPACE_ID="..."
```

The trace project is `pantry-to-plan-week4-eval`. The LangSmith dataset is
`pantry-to-plan-golden-v1`. Dataset synchronization refuses to edit an existing
dataset if its case count or fixture hashes differ from the frozen local data.

## One-case smoke run

Run the deterministic `PTP-001` smoke case and upload only that experiment:

```bash
uv run python -m evaluation.run_eval --case-id PTP-001 --mode deterministic --upload
```

Run the same validation locally without uploading traces:

```bash
uv run python -m evaluation.run_eval --case-id PTP-001 --mode deterministic
```

The command accepts only one case ID. Expanding to all 23 cases requires a
separate runner change after the smoke trace is reviewed.

After reviewing the `PTP-001` smoke trace, run the remaining 22 deterministic
cases as one experiment:

```bash
uv run python -m evaluation.run_remaining
```

## Metrics

The code evaluators report hard-constraint compliance, applicable fixture
expectations, independent inventory and shopping-list reconciliation, agentic
trajectory correctness, safe failure recovery, hard guardrails, and an overall
conservative case pass. Latency and call counts are stored in local results.

The deterministic mode uses `LocalRetriever`, `temperature=0`, and no advisor.
The future agentic mode uses `FakePlannerAdvisor`, `temperature=0`, and the same
local retrieval backend.
