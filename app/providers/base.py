from abc import ABC, abstractmethod
from collections.abc import Sequence


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    @abstractmethod
    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    def __init__(self, api_key: str | None, model: str, base_url: str | None = None):
        if not api_key or not model:
            raise ProviderError("LLM_API_KEY and LLM_MODEL are required for an OpenAI-compatible provider")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        result = self.client.chat.completions.create(model=self.model, messages=list(messages))
        return result.choices[0].message.content or ""


class GeminiProvider(Provider):
    def __init__(self, api_key: str | None, model: str):
        if not api_key or not model:
            raise ProviderError("GEMINI_API_KEY and LLM_MODEL are required for Gemini")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        result = self.client.models.generate_content(model=self.model, contents=prompt)
        return result.text or ""


def create_provider(name: str, settings) -> Provider:
    normalized = name.lower().replace("-", "_")
    if normalized == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.llm_model)
    if normalized in {"openai", "openai_compatible", "openai_compat"}:
        return OpenAICompatibleProvider(settings.llm_api_key, settings.llm_model, settings.llm_base_url)
    raise ProviderError(f"Unsupported LLM provider: {name}")
