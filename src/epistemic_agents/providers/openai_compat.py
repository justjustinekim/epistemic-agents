"""OpenAI-compatible provider — covers GPT, Grok, DeepSeek, QwQ via different base URLs."""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from epistemic_agents.client import CallUsage, compute_cost
from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.schema import PanelResponse


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        name: str,
        reasoning_model: bool = False,
        stream: bool = False,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._reasoning_model = reasoning_model
        self._stream = stream
        self._last_usage: CallUsage | None = None

    @classmethod
    def grok(cls, api_key: str) -> OpenAICompatProvider:
        """Create a Grok provider via xAI API (full grok-3)."""
        return cls(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            model_id="grok-3",
            name="grok",
        )

    @classmethod
    def deepseek(cls, api_key: str) -> OpenAICompatProvider:
        """Create a DeepSeek R1 provider via DeepSeek API."""
        return cls(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model_id="deepseek-reasoner",
            name="deepseek",
        )

    @classmethod
    def qwq(cls, api_key: str) -> OpenAICompatProvider:
        """Create a QwQ provider via Alibaba DashScope API."""
        return cls(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model_id="qwq-plus",
            name="qwq",
            stream=True,
        )

    @classmethod
    def gpt(cls, api_key: str) -> OpenAICompatProvider:
        """Create a GPT provider via OpenAI API (gpt-4o-mini)."""
        return cls(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model_id="gpt-4o-mini",
            name="gpt",
        )

    @classmethod
    def perplexity(cls, api_key: str) -> OpenAICompatProvider:
        """Create a Perplexity provider via Sonar API (grounded in real-time web search)."""
        return cls(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            model_id="sonar-pro",
            name="perplexity",
        )

    def analyze(self, task: str, system_prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"

        if self._reasoning_model:
            # OpenAI reasoning models: use 'developer' role, no temperature
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": task},
                ],
            }
        else:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ],
                "temperature": 0.7,
            }

        if self._stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "epistemic-agents/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if self._stream:
                    return self._read_stream(resp)
                body = json.loads(resp.read().decode())

                # Extract usage from response
                usage = body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = compute_cost(self.model_id, input_tokens, output_tokens)
                self._last_usage = CallUsage(
                    model=self.model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )

                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"{self.name} API error (HTTP {e.code}): {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"{self.name} connection error: {e.reason}") from e

    def _read_stream(self, resp) -> str:
        """Read SSE stream and return concatenated content."""
        content_parts: list[str] = []
        stream_usage: dict = {}
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            # Capture usage from final chunk (OpenAI sends it with stream_options)
            if chunk.get("usage"):
                stream_usage = chunk["usage"]

        input_tokens = stream_usage.get("prompt_tokens", 0)
        output_tokens = stream_usage.get("completion_tokens", 0)
        cost = compute_cost(self.model_id, input_tokens, output_tokens)
        self._last_usage = CallUsage(
            model=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        return "".join(content_parts)

    @property
    def supports_structured_output(self) -> bool:
        # GPT supports structured output; reasoning models and others do not
        return self.name == "gpt"

    def structured_analyze(self, task: str, system_prompt: str) -> PanelResponse:
        """Structured output via OpenAI JSON mode for GPT models."""
        if not self.supports_structured_output:
            raise NotImplementedError(f"{self.name} does not support structured output")

        url = f"{self._base_url}/chat/completions"
        schema = PanelResponse.model_json_schema()

        augmented_task = (
            f"{task}\n\n---\n"
            f"IMPORTANT: Respond with ONLY valid JSON matching this schema — "
            f"no markdown, no code fences:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_task},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "epistemic-agents/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode())
                usage = body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = compute_cost(self.model_id, input_tokens, output_tokens)
                self._last_usage = CallUsage(
                    model=self.model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
                content = body["choices"][0]["message"]["content"]
                return PanelResponse.model_validate_json(content)
        except (urllib.error.HTTPError, urllib.error.URLError):
            raise
        except Exception as e:
            raise RuntimeError(f"Structured output failed for {self.name}: {e}") from e

    @property
    def available(self) -> bool:
        return bool(self._api_key)
