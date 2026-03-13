from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .env import load_env_file, require_env


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


class OpenAIResponsesClient:
    def __init__(self, model_name: str | None = None, env_path: str = ".env") -> None:
        load_env_file(env_path)
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for live execution. Install project dependencies first."
            ) from exc

        kwargs: dict[str, Any] = {"api_key": require_env("OPENAI_API_KEY")}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1_200,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        request: dict[str, Any] = {
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature

        response = client.responses.create(**request)
        text = getattr(response, "output_text", "") or self._extract_text(response)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return LLMResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=response,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        collected: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if text_value:
                    collected.append(text_value)
        return "\n".join(collected).strip()
