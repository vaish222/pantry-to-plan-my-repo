"""Optional LangSmith instrumentation shared by the planner and eval harness.

Tracing is inert unless ``LANGSMITH_TRACING=true``. The application UI keeps
that flag disabled; the evaluation CLI enables it only for scoped eval runs.
"""

from contextlib import contextmanager
import os
from typing import Iterator

try:
    import langsmith as ls
except ImportError:  # Keeps the planner importable before optional deps are synced.
    ls = None


def traced(*, name: str, run_type: str = "chain"):
    """Decorate a real operation when LangSmith is installed, else do nothing."""
    if ls is None:
        return lambda func: func
    return ls.traceable(name=name, run_type=run_type)


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict | None = None,
    metadata: dict | None = None,
) -> Iterator[object | None]:
    """Create a child span only during an explicitly enabled eval process."""
    enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    if ls is None or not enabled:
        yield None
        return
    with ls.trace(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        metadata=metadata or {},
    ) as run:
        yield run
