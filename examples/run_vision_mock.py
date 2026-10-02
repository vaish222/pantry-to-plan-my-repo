"""Run the full image pipeline locally without an API key.

From the repository root:
    python examples/run_vision_mock.py
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision import MockVisionProvider, parse_pantry_image  # noqa: E402


photo = ROOT / "fixtures" / "photos" / "01_well_stocked_veg_italian_messy.png"
result = parse_pantry_image(photo.read_bytes(), provider=MockVisionProvider())
print(result.model_dump_json(indent=2))
