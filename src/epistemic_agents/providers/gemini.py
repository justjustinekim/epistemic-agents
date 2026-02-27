"""Google Gemini provider — REST API via urllib, supports thinking models."""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from epistemic_agents.providers.base import BaseProvider

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiProvider(BaseProvider):
    """Google Gemini via REST API. Defaults to gemini-2.5-flash with thinking."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash",
    ) -> None:
        self.name = "gemini"
        self.model_id = model_id
        self._api_key = api_key

    def analyze(self, task: str, system_prompt: str) -> str:
        url = f"{_GEMINI_URL.format(model=self.model_id)}?key={self._api_key}"

        generation_config: dict = {}
        # Enable thinking for 2.5 models
        if "2.5" in self.model_id:
            generation_config["thinkingConfig"] = {"thinkingBudget": 8192}
        else:
            generation_config["temperature"] = 0.7

        payload = json.dumps({
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "parts": [{"text": task}],
                }
            ],
            "generationConfig": generation_config,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode())
                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError(
                        f"Gemini returned no candidates: {json.dumps(body)[:500]}"
                    )
                candidate = candidates[0]
                # Some responses have finish_reason without content
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    finish = candidate.get("finishReason", "unknown")
                    raise RuntimeError(
                        f"Gemini returned no content parts (finishReason={finish})"
                    )
                # Extract non-thought parts (thinking models return thought=true parts)
                text_parts = [
                    p["text"] for p in parts
                    if "text" in p and not p.get("thought", False)
                ]
                if not text_parts:
                    # Fallback: return all text parts including thoughts
                    text_parts = [p["text"] for p in parts if "text" in p]
                return "\n".join(text_parts)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"Gemini API error (HTTP {e.code}): {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini connection error: {e.reason}") from e

    @property
    def available(self) -> bool:
        return bool(self._api_key)
