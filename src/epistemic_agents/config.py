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


def provider_credit_status() -> str:
    """Return a summary of configured providers and their billing status."""
    lines: list[str] = ["Provider Credit Status:", ""]

    # Claude — always available via Max plan
    lines.append("  claude (opus)        : Max plan (subscription)")

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        lines.append("  gemini (2.5-flash)   : Free tier")
    else:
        lines.append("  gemini               : Not configured")

    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        lines.append("  grok (grok-3)        : Pay-per-use")
    else:
        lines.append("  grok                 : Not configured")

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        # Try DeepSeek balance API
        balance_info = _check_deepseek_balance(deepseek_key)
        lines.append(f"  deepseek (reasoner)  : {balance_info}")
    else:
        lines.append("  deepseek             : Not configured")

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if dashscope_key:
        lines.append("  qwq (qwq-plus)       : Free tier")
    else:
        lines.append("  qwq                  : Not configured")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        lines.append("  gpt (gpt-4o-mini)    : Pay-per-use")
    else:
        lines.append("  gpt                  : Not configured")

    perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if perplexity_key:
        lines.append("  perplexity (sonar)   : Pay-per-use")
    else:
        lines.append("  perplexity           : Not configured")

    return "\n".join(lines)


def _check_deepseek_balance(api_key: str) -> str:
    """Try the DeepSeek balance API. Return status string."""
    import json
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "https://api.deepseek.com/user/balance",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            # DeepSeek returns balance_infos with currency and total_balance
            infos = data.get("balance_infos", [])
            if infos:
                bal = infos[0]
                return f"Pay-per-use (balance: {bal.get('currency', 'USD')} {bal.get('total_balance', '?')})"
            return "Pay-per-use"
    except Exception:
        return "Pay-per-use (balance check failed)"
