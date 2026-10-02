"""Provider ordering and embedding configuration tests (no network calls)."""

import os
import unittest
from unittest.mock import patch

from llm_provider import embedding_endpoint, provider_chain, run_with_fallback


class LLMProviderTests(unittest.TestCase):
    def test_ollama_primary_nebius_fallback(self):
        env = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "local-model",
            "NEBIUS_API_KEY": "test-key",
            "NEBIUS_MODEL_1": "cloud-model",
        }
        with patch.dict(os.environ, env, clear=True):
            endpoints = provider_chain()
        self.assertEqual([item.name for item in endpoints], ["ollama", "nebius"])
        self.assertEqual([item.model for item in endpoints], ["local-model", "cloud-model"])

    def test_failed_ollama_uses_nebius(self):
        env = {"LLM_PROVIDER": "ollama", "NEBIUS_API_KEY": "test-key"}
        calls = []

        def operation(endpoint, client):
            calls.append(endpoint.name)
            if endpoint.name == "ollama":
                raise ConnectionError("offline")
            return "ok"

        with patch.dict(os.environ, env, clear=True), patch(
            "llm_provider.LLMEndpoint.client", return_value=object()
        ):
            result, provider = run_with_fallback(operation)
        self.assertEqual((result, provider), ("ok", "nebius"))
        self.assertEqual(calls, ["ollama", "nebius"])

    def test_ollama_embeddings(self):
        env = {
            "EMBEDDING_PROVIDER": "ollama",
            "OLLAMA_EMBEDDING_MODEL": "embeddinggemma",
            "OLLAMA_EMBEDDING_DIMENSIONS": "512",
        }
        with patch.dict(os.environ, env, clear=True):
            endpoint = embedding_endpoint()
        self.assertEqual((endpoint.name, endpoint.model, endpoint.dimensions),
                         ("ollama", "embeddinggemma", 512))


if __name__ == "__main__":
    unittest.main()
