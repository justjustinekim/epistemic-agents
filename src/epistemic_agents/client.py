"""Client that uses the Claude CLI — leverages your Max plan, no API key needed."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Per-call cost tracking via thread-local side channel
# ---------------------------------------------------------------------------

# Approximate per-token costs (USD) — mirrors UsageTracker._COST_PER_1K
_COST_PER_1K: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "opus": (0.015, 0.075),
    "sonnet": (0.003, 0.015),
    "haiku": (0.00025, 0.00125),
    "gemini-2.0-flash": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "grok-3": (0.003, 0.015),
    "grok-3-mini": (0.0003, 0.0005),
    "deepseek-reasoner": (0.00055, 0.00219),
    "qwq-plus": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3": (0.01, 0.04),
    "sonar-pro": (0.003, 0.015),
    "sonar": (0.001, 0.001),
    "code-exec-sonnet": (0.003, 0.015),
}


@dataclass
class CallUsage:
    """Token usage and cost for a single API call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CallCostTracker:
    """Accumulates CallUsage across multiple API calls."""

    calls: list[CallUsage] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def __str__(self) -> str:
        return (
            f"${self.total_cost_usd:.4f} "
            f"({len(self.calls)} calls, "
            f"{self.total_input_tokens} in / {self.total_output_tokens} out)"
        )


_thread_local = threading.local()


def get_last_usage() -> CallUsage | None:
    """Return the CallUsage stored by the most recent request in this thread."""
    return getattr(_thread_local, "last_usage", None)


def _store_usage(usage: CallUsage) -> None:
    """Store usage in thread-local for callers to read after the call."""
    _thread_local.last_usage = usage


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts using the pricing table."""
    costs = _COST_PER_1K.get(model, (0.001, 0.005))
    return (input_tokens / 1000 * costs[0]) + (output_tokens / 1000 * costs[1])


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """Return a copy of the environment with CLAUDECODE unset so we can nest CLI calls."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def structured_request(
    model: str,
    system: str,
    user_message: str,
    response_model: Type[T],
    timeout: int = 1200,
) -> T:
    """Send a request via Claude CLI and parse into a Pydantic model using --json-schema."""
    schema = response_model.model_json_schema()
    schema_str = json.dumps(schema)

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model", model,
            "--system-prompt", system,
            "--output-format", "json",
            "--json-schema", schema_str,
        ],
        input=user_message,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_clean_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr}")

    # --output-format json wraps the response in a JSON envelope
    envelope = json.loads(result.stdout)

    # Extract token usage from envelope
    input_tokens = envelope.get("input_tokens", 0)
    output_tokens = envelope.get("output_tokens", 0)
    if not input_tokens and not output_tokens:
        # Fallback: estimate from text lengths
        input_tokens = len(user_message) // 4
        output_tokens = len(envelope.get("result", "")) // 4
    cost = compute_cost(model, input_tokens, output_tokens)
    _store_usage(CallUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    ))

    # With --json-schema, structured output lands in "structured_output"
    parsed = envelope.get("structured_output")
    if parsed is None:
        # Fallback: try parsing the "result" text field as JSON
        response_text = envelope.get("result", "")
        if response_text:
            parsed = json.loads(response_text)
        else:
            raise RuntimeError(
                f"No structured_output or result in CLI response. Keys: {list(envelope.keys())}"
            )

    return response_model.model_validate(parsed)


def plain_request(
    model: str,
    system: str,
    user_message: str,
    timeout: int = 1200,
) -> str:
    """Send a plain text request via Claude CLI and return the text response."""
    result = subprocess.run(
        [
            "claude",
            "-p",
            "--model", model,
            "--system-prompt", system,
            "--output-format", "json",
        ],
        input=user_message,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_clean_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr}")

    envelope = json.loads(result.stdout)

    # Extract token usage from envelope
    input_tokens = envelope.get("input_tokens", 0)
    output_tokens = envelope.get("output_tokens", 0)
    if not input_tokens and not output_tokens:
        input_tokens = len(user_message) // 4
        response_text = envelope.get("result", "")
        output_tokens = len(response_text) // 4
    else:
        response_text = envelope.get("result", "")
    cost = compute_cost(model, input_tokens, output_tokens)
    _store_usage(CallUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    ))

    return envelope.get("result", "")
