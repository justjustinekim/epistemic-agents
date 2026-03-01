"""Code Executor provider — validates claims empirically by running Python code."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents import client


class ClaimTest(BaseModel):
    """A single testable claim with validation code."""

    claim: str = Field(description="The factual claim being tested")
    python_code: str = Field(description="Python code that tests the claim and prints JSON result")
    rationale: str = Field(description="Why this code tests the claim")


class ClaimTestPlan(BaseModel):
    """Plan for empirically validating claims from an analysis."""

    tests: list[ClaimTest] = Field(default_factory=list)
    untestable_claims: list[str] = Field(
        default_factory=list,
        description="Claims that cannot be tested with code (opinions, predictions, etc.)",
    )


CODE_GEN_SYSTEM = """\
You are a claim validator. Given an analysis of a task, extract testable factual claims \
and write Python code to verify each one.

For each testable claim, write a short Python script that:
1. Tests the specific factual claim (math, logic, data lookups, API checks, etc.)
2. Prints a JSON object to stdout with exactly these keys:
   - "passed": bool — whether the claim holds
   - "detail": str — what the test found

Use only the Python standard library. Do NOT use external packages.
Do NOT make network requests or access the filesystem outside /tmp.

Claims that are opinions, predictions, strategic recommendations, or otherwise \
untestable with code should go in untestable_claims.

Focus on claims involving: math/calculations, logical consistency, dates/timelines, \
data relationships, algorithm properties, or verifiable facts."""

_SAFETY_PREAMBLE = """\
import os as _os, sys as _sys
# Safety: prevent shell access and restrict filesystem
_orig_system = _os.system
_os.system = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("os.system disabled"))
_os.chdir(_os.environ.get("TMPDIR", "/tmp"))
"""


class CodeExecutorProvider(BaseProvider):
    """Validates claims by generating and executing Python test code.

    Uses Claude to extract testable claims from an analysis, generates
    validation code, then runs each test in a sandboxed subprocess.
    """

    def __init__(self, model_id: str = "sonnet") -> None:
        self.name = "code-executor"
        self.model_id = f"code-exec-{model_id}"
        self._claude_model = model_id

    def analyze(self, task: str, system_prompt: str) -> str:
        """Generate claim tests from the task, execute them, format results."""
        # Step 1: Ask Claude to generate test plan
        plan = client.structured_request(
            model=self._claude_model,
            system=CODE_GEN_SYSTEM,
            user_message=task,
            response_model=ClaimTestPlan,
        )

        if not plan.tests and not plan.untestable_claims:
            return "No testable claims identified in the analysis."

        # Step 2: Execute each test
        results: list[str] = []
        passed = 0
        failed = 0
        errors = 0

        for i, test in enumerate(plan.tests, 1):
            outcome = self._execute_test(test)
            status = outcome.get("status", "error")
            if status == "passed":
                passed += 1
                icon = "PASS"
            elif status == "failed":
                failed += 1
                icon = "FAIL"
            else:
                errors += 1
                icon = "ERROR"

            results.append(
                f"{i}. [{icon}] {test.claim}\n"
                f"   Rationale: {test.rationale}\n"
                f"   Result: {outcome.get('detail', 'No detail')}"
            )

        # Step 3: Format output
        sections = [
            f"# Empirical Claim Validation\n",
            f"**Summary**: {passed} passed, {failed} failed, {errors} errors "
            f"out of {len(plan.tests)} tests\n",
        ]

        if results:
            sections.append("## Test Results\n" + "\n\n".join(results))

        if plan.untestable_claims:
            sections.append(
                "\n## Untestable Claims\n"
                + "\n".join(f"- {c}" for c in plan.untestable_claims)
            )

        return "\n".join(sections)

    def _execute_test(self, test: ClaimTest) -> dict[str, Any]:
        """Run a claim test in a sandboxed subprocess.

        Returns dict with 'status' ('passed'/'failed'/'error') and 'detail'.
        """
        code = _SAFETY_PREAMBLE + "\n" + test.python_code

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._safe_env(),
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "detail": f"Exit code {result.returncode}: {result.stderr.strip()[:500]}",
                }

            stdout = result.stdout.strip()
            if not stdout:
                return {"status": "error", "detail": "No output produced"}

            try:
                output = json.loads(stdout)
                status = "passed" if output.get("passed") else "failed"
                return {"status": status, "detail": output.get("detail", str(output))}
            except json.JSONDecodeError:
                return {"status": "error", "detail": f"Non-JSON output: {stdout[:500]}"}

        except subprocess.TimeoutExpired:
            return {"status": "error", "detail": "Test timed out (30s limit)"}
        except Exception as exc:
            return {"status": "error", "detail": f"Execution error: {exc}"}
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass

    @staticmethod
    def _safe_env() -> dict[str, str]:
        """Return environment with API keys and secrets stripped."""
        env = os.environ.copy()
        sensitive_patterns = (
            "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
            "AWS_ACCESS", "AWS_SECRET", "ANTHROPIC", "OPENAI",
            "GOOGLE_API", "XAI_", "DEEPSEEK_", "DASHSCOPE_", "PERPLEXITY_",
        )
        to_remove = [
            k for k in env
            if any(pat in k.upper() for pat in sensitive_patterns)
        ]
        for k in to_remove:
            del env[k]
        # Also strip CLAUDECODE to avoid nested CLI issues
        env.pop("CLAUDECODE", None)
        return env

    @property
    def available(self) -> bool:
        """Always available — uses Claude CLI."""
        return True
