"""Run one Ollama/Nebius vision request against a fixture photo.

Prerequisites:
    ollama serve
    # Optional cloud fallback: set NEBIUS_API_KEY in .env
    python examples/run_vision_live.py [optional-image-path]
"""

from pathlib import Path
import sys
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from vision import CompatibleVisionProvider, VisionPipelineError, parse_pantry_image  # noqa: E402


default_photo = ROOT / "fixtures" / "photos" / "01_well_stocked_veg_italian_messy.png"
photo = Path(sys.argv[1]) if len(sys.argv) > 1 else default_photo

try:
    result = parse_pantry_image(photo.read_bytes(), provider=CompatibleVisionProvider())
except VisionPipelineError as exc:
    print(exc.issue.model_dump_json(indent=2))
    raise SystemExit(1) from exc

print(result.model_dump_json(indent=2))
