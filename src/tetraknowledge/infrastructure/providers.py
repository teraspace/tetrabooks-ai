import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path

from tetraknowledge.config import get_settings


class EmbeddingProvider(ABC):
    model: str
    dimensions: int

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 384, model: str = "hash-384"):
        self.dimensions, self.model = dimensions, model

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            values[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in values)) or 1.0
        return [x / norm for x in values]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model, self._model = model, SentenceTransformer(model)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings, including Azure AI Foundry / Azure OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-10-21",
        dimensions: int = 1536,
    ):
        if base_url:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/")
        else:
            from openai import AzureOpenAI

            if not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT is required when AZURE_OPENAI_BASE_URL is absent"
                )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        self.model, self.dimensions = model, dimensions

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = response.data[0].embedding
        self.dimensions = len(vector)
        return vector


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def answer(self, question: str, context: str, prompt_version: str = "rag_system_v1") -> str: ...


def system_prompt(version: str) -> str:
    path = Path(__file__).parents[3] / "prompts" / f"{version}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else "Use only the supplied evidence."


class RuleBasedLLMProvider(LLMProvider):
    model = "demo-rule-based"

    def answer(self, question: str, context: str, prompt_version: str = "rag_system_v1") -> str:
        if not context.strip():
            return "No encontré evidencia suficiente en la base de conocimiento."
        lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip() and "ignore all previous instructions" not in line.lower()
        ]
        if "autorización" in question.lower() or "autorizacion" in question.lower():
            for line in lines:
                if "kappa-7" in line.lower():
                    return (
                        "La operación requiere autorización Kappa-7, según la evidencia recuperada."
                    )
        return "Según la evidencia recuperada: " + " ".join(lines[:3])


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client, self.model = OpenAI(api_key=api_key), model

    def answer(self, question: str, context: str, prompt_version: str = "rag_system_v1") -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt(prompt_version)},
                {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"},
            ],
        }
        if not self.model.startswith("gpt-5"):
            request["temperature"] = 0
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or "No answer returned."


class AzureOpenAILLMProvider(LLMProvider):
    def __init__(
        self,
        endpoint: str | None,
        base_url: str | None,
        api_key: str,
        deployment: str,
        api_version: str,
    ):
        if base_url:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/")
        else:
            from openai import AzureOpenAI

            if not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT is required when "
                    "AZURE_OPENAI_BASE_URL is absent"
                )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        self.model = deployment

    def answer(self, question: str, context: str, prompt_version: str = "rag_system_v1") -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt(prompt_version)},
                {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"},
            ],
        }
        if not self.model.startswith("gpt-5"):
            request["temperature"] = 0
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or "No answer returned."


def embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider in {"sentence_transformer", "sentence-transformers"}:
        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    if (
        settings.embedding_provider in {"openai", "azure_openai", "azure"}
        and settings.azure_openai_api_key
        and settings.azure_openai_embedding_deployment
    ):
        return OpenAIEmbeddingProvider(
            api_key=settings.azure_openai_api_key,
            model=settings.azure_openai_embedding_deployment,
            base_url=settings.azure_openai_base_url,
            endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            dimensions=settings.embedding_dimensions,
        )
    return HashEmbeddingProvider(settings.embedding_dimensions, settings.embedding_model)


def llm_provider() -> LLMProvider:
    settings = get_settings()
    if (
        settings.llm_provider == "azure_openai"
        and (settings.azure_openai_endpoint or settings.azure_openai_base_url)
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        return AzureOpenAILLMProvider(
            settings.azure_openai_endpoint,
            settings.azure_openai_base_url,
            settings.azure_openai_api_key,
            settings.azure_openai_deployment,
            settings.azure_openai_api_version,
        )
    if settings.llm_provider == "openai":
        import os

        return OpenAILLMProvider(os.environ["OPENAI_API_KEY"], settings.llm_model)
    return RuleBasedLLMProvider()
