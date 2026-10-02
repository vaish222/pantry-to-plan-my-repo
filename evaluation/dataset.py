"""Build and validate Golden Dataset v1 without modifying source fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "fixtures"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
REVIEW_WORKBOOK = DATASETS_DIR / "pantry_to_plan_golden_v1_review.xlsx"
CANONICAL_JSON = DATASETS_DIR / "pantry_to_plan_golden_v1.json"
LANGSMITH_JSONL = DATASETS_DIR / "pantry_to_plan_golden_v1_langsmith.jsonl"
DATASET_VERSION = "pantry-to-plan-golden-v1"
DATASET_NAME = DATASET_VERSION

_XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cell_value(cell: ET.Element) -> str:
    value = cell.find("x:v", _XML_NS)
    inline = cell.find("x:is/x:t", _XML_NS)
    if value is not None and value.text is not None:
        return value.text
    if inline is not None and inline.text is not None:
        return inline.text
    return ""


def load_review_rows(path: Path = REVIEW_WORKBOOK) -> list[dict[str, str]]:
    """Read the review sheet with stdlib OOXML parsing."""
    if not path.exists():
        raise FileNotFoundError(f"Golden review workbook is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = root.findall(".//x:sheetData/x:row", _XML_NS)
    if not rows:
        raise ValueError("Golden review workbook has no rows")

    def row_values(row: ET.Element) -> list[str]:
        return [_cell_value(cell) for cell in row.findall("x:c", _XML_NS)]

    headers = row_values(rows[0])
    return [dict(zip(headers, row_values(row), strict=False)) for row in rows[1:]]


def _scenario_type(tags: list[str]) -> str:
    return "happy_path" if "happy-path" in tags else "edge_case"


def _photo_paths(fixture_id: str) -> list[str]:
    matches = sorted((FIXTURES_DIR / "photos").glob(f"{fixture_id}*.png"))
    return [str(path.relative_to(ROOT)) for path in matches]


def _fixture_hash(fixture: dict) -> str:
    raw = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_cases(path: Path = REVIEW_WORKBOOK) -> list[dict]:
    """Join workbook case IDs to the exact committed fixture payloads."""
    cases = []
    seen_fixture_ids: set[str] = set()
    for row in load_review_rows(path):
        case_id = row.get("Case ID", "").strip()
        fixture_id = row.get("Fixture ID", "").strip()
        if not case_id or not fixture_id:
            continue
        fixture_path = FIXTURES_DIR / f"{fixture_id}.json"
        if not fixture_path.exists():
            raise ValueError(f"{case_id} references missing fixture {fixture_id}")
        fixture = json.loads(fixture_path.read_text())
        if fixture["id"] != fixture_id:
            raise ValueError(f"Fixture ID mismatch for {case_id}: {fixture['id']}")
        if fixture_id in seen_fixture_ids:
            raise ValueError(f"Duplicate fixture mapping: {fixture_id}")
        seen_fixture_ids.add(fixture_id)
        cases.append(
            {
                "case_id": case_id,
                "fixture_id": fixture_id,
                "dataset_version": DATASET_VERSION,
                "scenario_type": row.get("Scenario Type") or _scenario_type(fixture["tags"]),
                "description": fixture["description"],
                "tags": fixture["tags"],
                "inputs": {
                    "case_id": case_id,
                    "fixture_id": fixture_id,
                    "dataset_version": DATASET_VERSION,
                    "scenario_type": row.get("Scenario Type") or _scenario_type(fixture["tags"]),
                    "description": fixture["description"],
                    "tags": fixture["tags"],
                    "pantry": fixture["pantry"],
                    "request": fixture["request"],
                },
                "expected": fixture["expect"],
                "metadata": {
                    "case_id": case_id,
                    "fixture_id": fixture_id,
                    "dataset_version": DATASET_VERSION,
                    "scenario_type": row.get("Scenario Type") or _scenario_type(fixture["tags"]),
                    "photo_paths": _photo_paths(fixture_id),
                    "human_verified": row.get("Human Verified", "").upper() == "TRUE",
                    "fixture_sha256": _fixture_hash(fixture),
                },
            }
        )
    fixture_count = len(list(FIXTURES_DIR.glob("*.json")))
    if len(cases) != 23 or fixture_count != 23 or len(seen_fixture_ids) != fixture_count:
        raise ValueError(
            f"Golden v1 must map exactly 23 review rows to 23 fixtures; "
            f"got rows={len(cases)}, fixtures={fixture_count}"
        )
    return cases


def dataset_hash(cases: list[dict]) -> str:
    raw = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_dataset_files(cases: list[dict]) -> str:
    """Write deterministic local JSON and LangSmith JSONL representations."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    digest = dataset_hash(cases)
    payload = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": digest,
        "case_count": len(cases),
        "cases": cases,
    }
    CANONICAL_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with LANGSMITH_JSONL.open("w") as handle:
        for case in cases:
            record = {
                "inputs": case["inputs"],
                "outputs": {"expected": case["expected"]},
                "metadata": {**case["metadata"], "dataset_sha256": digest},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return digest


def find_case(cases: list[dict], case_id: str) -> dict:
    matches = [case for case in cases if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one case for {case_id}; found {len(matches)}")
    return matches[0]

