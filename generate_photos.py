"""Generate fake pantry photos for the eval fixtures via OpenAI images.

One photo per fixture, tailored to the scenario: well-stocked pantries show the
listed items, sparse/empty show mostly-bare shelves, and the vision-noise
fixtures are deliberately messy/blurry with obscured labels (which is what makes
their ambiguous parse plausible).

Usage:
    python3 generate_photos.py [fixture_id ...]   # default: all fixtures
Reads OPENAI_API_KEY from .env.demo. Writes PNGs to fixtures/photos/.
"""

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
FIX_DIR = ROOT / "fixtures"
OUT_DIR = FIX_DIR / "photos"
MODEL = "gpt-image-1"
QUALITY = os.environ.get("PHOTO_QUALITY", "low")
SIZE = "1024x1024"
# STYLE=messy renders cluttered, disorganized, lived-in pantries (for the
# well-stocked scenarios, which otherwise come out looking styled/curated).
STYLE = os.environ.get("PHOTO_STYLE", "tidy")
# SUFFIX is appended to the output filename, e.g. "_messy" -> <id>_messy.png.
SUFFIX = os.environ.get("PHOTO_SUFFIX", "")


def load_key():
    text = (ROOT / ".env.demo").read_text()
    m = re.search(r"^OPENAI_API_KEY=(\S+)", text, re.MULTILINE)
    if not m:
        raise SystemExit("OPENAI_API_KEY not found in .env.demo")
    return m.group(1)


def humanize(ingredient_id):
    return ingredient_id.replace("_", " ")


def build_prompt(fx):
    tags = set(fx.get("tags", []))
    items = list(fx["pantry"]["items"].keys())
    names = [humanize(i) for i in items]

    if not items or "empty" in tags:
        scene = "a nearly empty home pantry shelf with almost no food on it"
        contents = "essentially bare shelves"
    elif "sparse" in tags:
        scene = "a sparse home pantry with only a few items on otherwise empty shelves"
        contents = "just these few items: " + ", ".join(names)
    elif tags & {"vision-noise", "unclear-labels"}:
        scene = (
            "a cluttered, messy home pantry photographed quickly with a phone, "
            "some items with peeling, handwritten, faded, or partially obscured "
            "labels, slightly blurry and poorly lit"
        )
        contents = "items including: " + ", ".join(names)
    elif STYLE == "messy":
        scene = (
            "a cluttered, disorganized, overcrowded home pantry, items crammed "
            "onto shelves at odd angles, mismatched jars, boxes, and bags "
            "stacked haphazardly, a few tipped over, lived-in and not styled, "
            "with minor spills and crumbs"
        )
        contents = "packed with, among other things: " + ", ".join(names)
    else:
        scene = "a well-stocked home pantry and kitchen counter"
        contents = "clearly showing: " + ", ".join(names)

    return (
        f"A realistic eye-level smartphone photo of {scene}, {contents}. "
        "Natural kitchen lighting, photorealistic, no text overlays, no people, "
        "no added packaging text or logos."
    )


def generate(fx_path, key):
    fx = json.loads(fx_path.read_text())
    prompt = build_prompt(fx)
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "size": SIZE, "quality": QUALITY, "n": 1}
    )
    # Homebrew Python lacks a CA bundle here, so use curl (which has one) for TLS.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        tf.write(payload)
        payload_path = tf.name
    try:
        proc = subprocess.run(
            [
                "curl", "-sS", "--max-time", "180",
                "-X", "POST",
                "https://api.openai.com/v1/images/generations",
                "-H", f"Authorization: Bearer {key}",
                "-H", "Content-Type: application/json",
                "-d", f"@{payload_path}",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(payload_path)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed: {proc.stderr[:300]}")
    data = json.loads(proc.stdout)
    if "error" in data:
        raise RuntimeError(f"API error: {json.dumps(data['error'])[:400]}")
    b64 = data["data"][0]["b64_json"]
    out = OUT_DIR / (fx["id"] + SUFFIX + ".png")
    out.write_bytes(base64.b64decode(b64))
    return out


def main():
    key = load_key()
    OUT_DIR.mkdir(exist_ok=True)
    wanted = set(sys.argv[1:])
    paths = sorted(FIX_DIR.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    print(f"quality={QUALITY} size={SIZE} -> generating {len(paths)} photo(s)")
    for p in paths:
        try:
            out = generate(p, key)
            print(f"  OK  {out.name}  ({out.stat().st_size // 1024} KB)")
        except (RuntimeError, KeyError, json.JSONDecodeError) as e:
            print(f"  ERR {p.stem}: {e}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
