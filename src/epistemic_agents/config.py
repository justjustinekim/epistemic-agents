"""Load available providers from environment variables."""

from __future__ import annotations

import os

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.providers.claude import ClaudeProvider
from epistemic_agents.providers.openai_compat import OpenAICompatProvider
from epistemic_agents.providers.gemini import GeminiProvider


def get_available_providers(
    include_code_executor: bool = False,
) -> list[BaseProvider]:
    """Return all providers that have valid configuration.

    Claude is always available (uses CLI). Other providers require API keys
    set via environment variables.

    Args:
        include_code_executor: If True, include the CodeExecutorProvider.
    """
    from epistemic_agents.providers.code_executor import CodeExecutorProvider

    providers: list[BaseProvider] = [ClaudeProvider()]

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        providers.append(GeminiProvider(api_key=google_key))

    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        providers.append(OpenAICompatProvider.grok(api_key=xai_key))

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        providers.append(OpenAICompatProvider.deepseek(api_key=deepseek_key))

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if dashscope_key:
        providers.append(OpenAICompatProvider.qwq(api_key=dashscope_key))

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        providers.append(OpenAICompatProvider.gpt(api_key=openai_key))

    perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if perplexity_key:
        providers.append(OpenAICompatProvider.perplexity(api_key=perplexity_key))

    if include_code_executor:
        providers.append(CodeExecutorProvider())

    return providers
