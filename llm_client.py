"""Small shared OpenAI Responses API adapter; no client is created at import."""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "gpt-4.1-mini"


class LLMError(RuntimeError):
    """Actionable configuration or service failure, safe to display in the CLI."""


def generate_text(
    instructions: str, payload: dict, *, model_env: str = "OPENAI_MODEL",
    schema: dict | None = None,
) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "OPENAI_API_KEY is missing. Set it in your environment for live LLM "
            "runs, or use agent.py --mock for offline testing."
        )
    try:
        from openai import APIError, OpenAI
    except ImportError as exc:
        raise LLMError("Install dependencies with: python -m pip install -r requirements.txt") from exc
    model = (os.environ.get(model_env) or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    if not model:
        raise LLMError(f"{model_env} must name an available OpenAI model.")
    request = {
        "model": model, "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=True),
        "max_output_tokens": 1600, "store": False,
    }
    if schema is not None:
        request["text"] = {"format": {
            "type": "json_schema", "name": "access_review_evaluation",
            "strict": True, "schema": schema,
        }}
    try:
        with OpenAI(api_key=key, timeout=30.0, max_retries=1) as client:
            response = client.responses.create(**request)
    except APIError as exc:
        # SDK bodies may include credentials or echoed inputs; do not print them.
        code = getattr(exc, "code", None)
        quota_errors = {
            "credit_balance_exhausted":
                "OpenAI API credit balance is exhausted. Add API credits in billing before retrying.",
            "insufficient_quota":
                "OpenAI API quota is unavailable. Check API billing, credits, and usage limits.",
            "organization_spend_limit_exceeded":
                "OpenAI organization spend limit reached. Review that limit before retrying.",
            "project_spend_limit_exceeded":
                "OpenAI project spend limit reached. Review that limit before retrying.",
            "organization_usage_limit_exceeded":
                "OpenAI organization usage limit reached. Review API usage limits before retrying.",
        }
        if code in quota_errors:
            raise LLMError(quota_errors[code]) from None
        status = getattr(exc, "status_code", None)
        suffix = f" (HTTP {status})" if status is not None else ""
        raise LLMError(
            f"OpenAI request failed: {type(exc).__name__}{suffix}. "
            "Check your API key, model access, quota, and network connection."
        ) from None
    if response.status != "completed" or not response.output_text.strip():
        raise LLMError("OpenAI returned an incomplete or empty answer; no answer was accepted.")
    return response.output_text.strip()
