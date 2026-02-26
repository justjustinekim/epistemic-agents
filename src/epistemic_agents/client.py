"""Client that uses the Claude CLI — leverages your Max plan, no API key needed."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
        timeout=600,
        env=_clean_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr}")

    # --output-format json wraps the response in a JSON envelope
    envelope = json.loads(result.stdout)

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
        timeout=600,
        env=_clean_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr}")

    envelope = json.loads(result.stdout)
    return envelope.get("result", "")
