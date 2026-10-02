"""OpenAI-compatible provider routing for Ollama, Nebius, and embeddings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel


Capability = Literal["text", "vision"]
T = TypeVar("T")


class ProviderChainError(RuntimeError):
    """Raised after all configured providers fail."""


@dataclass(frozen=True)
class LLMEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str

    def client(self):
        from openai import OpenAI

        timeout = float(os.getenv(
            "OLLAMA_TIMEOUT_SECONDS" if self.name == "ollama" else "LLM_TIMEOUT_SECONDS",
            "10" if self.name == "ollama" else "60",
        ))
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )


@dataclass(frozen=True)
class EmbeddingEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    dimensions: int

    def client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30")),
            max_retries=0,
        )


def provider_chain(capability: Capability = "text") -> list[LLMEndpoint]:
    """Return Ollama first and configured Nebius second."""
    selected = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    endpoints: list[LLMEndpoint] = []

    if selected == "ollama":
        endpoints.append(LLMEndpoint(
            name="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=os.getenv(
                "OLLAMA_VISION_MODEL" if capability == "vision" else "OLLAMA_MODEL",
                "gemma3:4b" if capability == "vision" else "qwen3:8b",
            ),
        ))

    nebius_key = os.getenv("NEBIUS_API_KEY")
    if selected in {"ollama", "nebius"} and nebius_key:
        endpoints.append(LLMEndpoint(
            name="nebius",
            base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
            api_key=nebius_key,
            model=os.getenv(
                "NEBIUS_VISION_MODEL" if capability == "vision" else "NEBIUS_MODEL",
                os.getenv(
                    "NEBIUS_MODEL_2" if capability == "vision" else "NEBIUS_MODEL_1",
                    "google/gemma-3-27b-it" if capability == "vision"
                    else "Qwen/Qwen3-30B-A3B-Instruct-2507",
                ),
            ),
        ))

    if selected == "openai" and os.getenv("OPENAI_API_KEY"):
        endpoints.append(LLMEndpoint(
            name="openai",
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv(
                "OPENAI_VISION_MODEL" if capability == "vision" else "OPENAI_MODEL",
                "gpt-4o",
            ),
        ))
    return endpoints


def embedding_endpoint() -> EmbeddingEndpoint:
    """Return the configured embedding endpoint (local Ollama by default)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return EmbeddingEndpoint(
            name="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
            dimensions=int(os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "512")),
        )
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return EmbeddingEndpoint(
            name="openai",
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "512")),
        )
    raise ProviderChainError(f"Embedding provider {provider!r} is not configured.")


def run_with_fallback(
    operation: Callable[[LLMEndpoint, Any], T],
    *,
    capability: Capability = "text",
    preferred: str | None = None,
) -> tuple[T, str]:
    endpoints = provider_chain(capability)
    if not endpoints:
        raise ProviderChainError(
            "No LLM provider configured. Start Ollama or set NEBIUS_API_KEY."
        )
    if preferred:
        endpoints.sort(key=lambda endpoint: endpoint.name != preferred)

    failures: list[str] = []
    for endpoint in endpoints:
        try:
            return operation(endpoint, endpoint.client()), endpoint.name
        except Exception as exc:
            failures.append(f"{endpoint.name}: {type(exc).__name__}: {exc}")
    raise ProviderChainError("All LLM providers failed (" + "; ".join(failures) + ")")


def json_schema_format(output_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "strict": True,
            "schema": output_model.model_json_schema(),
        },
    }


def parse_json_model(content: str | None, output_model: type[BaseModel]) -> BaseModel:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return output_model.model_validate(json.loads(text))
