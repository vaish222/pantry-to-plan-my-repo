"""Pantry image parsing with interchangeable mock and live providers.

Public entry point: parse_pantry_image. It validates an upload, asks a provider
for raw observations, normalizes them, and returns candidates for human review.
It does not retrieve recipes or make meal-planning decisions.
"""

from __future__ import annotations

import base64
from io import BytesIO
from time import perf_counter
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from normalization import IngredientNormalizer
from schemas import (
    AppIssue,
    NormalizationStatus,
    PantryItemCandidate,
    PantryParseResult,
    RawDetectedIngredient,
    RawVisionResult,
)


SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_LOW_CONFIDENCE = 0.75

VISION_PROMPT = """You extract pantry ingredients from an image.

Identify distinct food ingredients that are visibly supported by the image.
Do not suggest recipes. Do not infer hidden ingredients. Do not include kitchen
equipment or non-food objects. Use a short, common ingredient name. Populate
quantity_g only if an exact quantity in grams is visibly readable on packaging;
otherwise return null. Confidence must be between 0 and 1. Evidence must be one
short visible cue and must not contain hidden reasoning.
"""


class VisionProvider(Protocol):
    """Boundary that keeps mock and live execution interchangeable."""

    name: str

    def extract(self, image_bytes: bytes, media_type: str) -> RawVisionResult:
        ...


class VisionPipelineError(RuntimeError):
    """Exception carrying an issue the UI can display without parsing text."""

    def __init__(self, issue: AppIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


class MockVisionProvider:
    """Predictable provider for UI development, tests, and demo fallback."""

    name = "mock"

    def __init__(self, result: RawVisionResult | None = None) -> None:
        self._result = result or RawVisionResult(
            ingredients=[
                RawDetectedIngredient(
                    raw_name="tomatoes",
                    display_name="Tomatoes",
                    confidence=0.94,
                    evidence="Several red tomatoes are visible",
                ),
                RawDetectedIngredient(
                    raw_name="capsicum",
                    display_name="Green capsicum",
                    confidence=0.82,
                    evidence="One green pepper is visible",
                ),
            ]
        )

    def extract(self, image_bytes: bytes, media_type: str) -> RawVisionResult:
        return self._result.model_copy(deep=True)


class CompatibleVisionProvider:
    """Ollama vision adapter with automatic Nebius failover."""

    name = "configured"

    def __init__(self) -> None:
        self.name = "configured"

    def extract(self, image_bytes: bytes, media_type: str) -> RawVisionResult:
        from llm_provider import json_schema_format, parse_json_model, run_with_fallback

        encoded = base64.b64encode(image_bytes).decode("ascii")
        def invoke(endpoint, client):
            response = client.chat.completions.create(
                model=endpoint.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": VISION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "List the pantry ingredients visible in this image."},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
                    ]},
                ],
                response_format=json_schema_format(RawVisionResult),
            )
            return parse_json_model(response.choices[0].message.content, RawVisionResult)

        result, self.name = run_with_fallback(invoke, capability="vision")
        return result  # type: ignore[return-value]


OpenAIVisionProvider = CompatibleVisionProvider


def validate_image(image_bytes: bytes) -> str:
    """Decode an upload and return a trusted MIME type."""

    if not image_bytes:
        raise _error("INVALID_IMAGE", "The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise _error(
            "INVALID_IMAGE",
            "The image exceeds the 10 MB prototype limit.",
            "Upload a smaller JPEG, PNG, or WebP image.",
        )
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise _error(
            "INVALID_IMAGE",
            "The upload is not a readable image.",
            "Upload a JPEG, PNG, or WebP image.",
        ) from exc

    if image_format not in SUPPORTED_FORMATS:
        raise _error(
            "INVALID_IMAGE",
            f"Unsupported image format: {image_format or 'unknown'}.",
            "Convert the image to JPEG, PNG, or WebP.",
        )
    return SUPPORTED_FORMATS[image_format]


def parse_pantry_image(
    image_bytes: bytes,
    provider: VisionProvider,
    normalizer: IngredientNormalizer | None = None,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE,
) -> PantryParseResult:
    """Turn an image into normalized, user-editable pantry candidates."""

    if not 0 <= low_confidence_threshold <= 1:
        raise ValueError("low_confidence_threshold must be between 0 and 1")

    media_type = validate_image(image_bytes)
    normalizer = normalizer or IngredientNormalizer()
    started = perf_counter()
    try:
        raw_result = provider.extract(image_bytes, media_type)
    except VisionPipelineError:
        raise
    except Exception as exc:
        raise _error(
            "VISION_PROVIDER_ERROR",
            "Ingredient detection failed.",
            "Retry the request or switch to mock mode for the demo.",
        ) from exc

    items: list[PantryItemCandidate] = []
    issues: list[AppIssue] = []
    for index, raw_item in enumerate(raw_result.ingredients):
        normalized = normalizer.normalize(raw_item.raw_name)
        items.append(
            PantryItemCandidate(
                ingredient_id=normalized.ingredient_id,
                display_name=normalized.display_name,
                quantity_g=raw_item.quantity_g,
                confidence=raw_item.confidence,
                source_text=raw_item.raw_name,
                normalization_status=normalized.status,
                recipe_supported=normalized.recipe_supported,
            )
        )

        field_prefix = f"items.{index}"
        if raw_item.confidence < low_confidence_threshold:
            issues.append(
                AppIssue(
                    code="LOW_CONFIDENCE",
                    message=f"Confirm whether '{raw_item.display_name}' is correct.",
                    field=field_prefix,
                    recoverable=True,
                    suggested_action="Edit or remove this item before planning.",
                )
            )
        if raw_item.quantity_g is None:
            issues.append(
                AppIssue(
                    code="QUANTITY_UNKNOWN",
                    message=f"Quantity is unknown for '{raw_item.display_name}'.",
                    field=f"{field_prefix}.quantity_g",
                    recoverable=True,
                    suggested_action="Optionally enter a gram quantity.",
                )
            )
        if normalized.status is NormalizationStatus.UNMAPPED:
            issues.append(
                AppIssue(
                    code="UNMAPPED_INGREDIENT",
                    message=f"'{raw_item.raw_name}' is not in the recipe vocabulary.",
                    field=f"{field_prefix}.ingredient_id",
                    recoverable=True,
                    suggested_action="Confirm the name or add an alias.",
                )
            )
        elif not normalized.recipe_supported:
            issues.append(
                AppIssue(
                    code="NO_RECIPE_SUPPORT",
                    message=f"No current recipe uses '{normalized.display_name}'.",
                    field=f"{field_prefix}.ingredient_id",
                    recoverable=True,
                    suggested_action="Keep the item, but do not expect it to improve recipe coverage.",
                )
            )

    if not items:
        issues.append(
            AppIssue(
                code="NO_INGREDIENTS_DETECTED",
                message="No pantry ingredients were detected.",
                recoverable=True,
                suggested_action="Try a brighter, closer image or add items manually.",
            )
        )

    latency_ms = max(0, round((perf_counter() - started) * 1000))
    return PantryParseResult(
        items=items,
        issues=issues,
        warnings=[issue.code for issue in issues],
        model_latency_ms=latency_ms,
        provider=provider.name,
    )


def _error(code: str, message: str, suggested_action: str | None = None) -> VisionPipelineError:
    return VisionPipelineError(
        AppIssue(
            code=code,
            message=message,
            recoverable=True,
            suggested_action=suggested_action,
        )
    )
