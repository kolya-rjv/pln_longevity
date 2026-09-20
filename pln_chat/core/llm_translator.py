"""Call the OpenAI API and parse the structured JSON response into a
TranslationResult dataclass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import openai

from config import OPENAI_API_KEY, OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS


#: Machine-readable failure classes, so a caller does not have to grep an error
#: string. The HTTP layer maps each to its own status code (see api.py).
ERROR_CODES = (
    "missing_api_key",           # the server has no OPENAI_API_KEY
    "auth",                      # the server's key was rejected
    "rate_limit",                # upstream throttling
    "context_length_exceeded",   # the prompt did not fit the model's window
    "timeout",                   # upstream did not answer in time
    "connection",                # could not reach upstream
    "upstream_error",            # any other OpenAI-side failure
    "bad_json",                  # the model answered with something unparseable
)


@dataclass
class TranslationResult:
    metta_query: str
    explanation: str
    intent: str                    # retrieval | inference | assertion | clarification | error
    requires_pln_inference: bool
    confidence_filter: float
    warnings: list[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None   # one of ERROR_CODES when error is set

    @property
    def ok(self) -> bool:
        return self.error is None


def _error_result(
    message: str,
    code: str,
    raw: Optional[str] = None,
    usage: Optional[dict] = None,
) -> TranslationResult:
    """A failed translation.

    `intent` is "error", NOT "clarification". A clarification is a real answer —
    the model deciding the question needs narrowing — and reusing it for a
    failure made an upstream outage indistinguishable from the engine politely
    asking what you meant (the 2026-09-18 evaluation hit exactly this: two
    context-length failures were returned looking like ordinary empty results).
    """
    return TranslationResult(
        metta_query="",
        explanation="",
        intent="error",
        requires_pln_inference=False,
        confidence_filter=0.0,
        warnings=[],
        raw_response=raw,
        usage=usage,
        error=message,
        error_code=code,
    )


def _classify_api_error(exc: Exception) -> str:
    """Map an OpenAI SDK exception to one of ERROR_CODES.

    `openai.BadRequestError` carries the upstream `code` in its body, which is
    how a prompt that overflows the model's context window is told apart from
    any other 400 — the difference between "your request is too big" (the
    caller can fix it) and "upstream is unwell" (it cannot).
    """
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    code = getattr(exc, "code", None)
    if not code:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            code = (body.get("error") or {}).get("code")
    if code == "context_length_exceeded":
        return "context_length_exceeded"
    return "upstream_error"


def translate(
    user_message: str,
    system_prompt: str,
    history: list[dict],
    model: str,
    temperature: float,
) -> TranslationResult:
    """Translate a natural language message to MeTTa via the OpenAI API.

    Args:
        user_message: The user's current message.
        system_prompt: Fully-built system prompt (ontology + examples injected).
        history: Prior conversation as a list of {"role": ..., "content": ...} dicts.
        model: OpenAI model name.
        temperature: Sampling temperature (low values give more deterministic output).

    Returns:
        A TranslationResult; check `.ok` and `.error` before using `.metta_query`.
    """
    if not OPENAI_API_KEY:
        return _error_result(
            "OPENAI_API_KEY is not set — add it to your .env file.",
            "missing_api_key",
        )

    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except openai.AuthenticationError:
        return _error_result("Invalid OpenAI API key.", "auth")
    except openai.RateLimitError:
        return _error_result(
            "OpenAI rate limit exceeded — please wait and retry.", "rate_limit"
        )
    except openai.APIError as exc:
        return _error_result(f"OpenAI API error: {exc}", _classify_api_error(exc))

    raw: str = response.choices[0].message.content or ""
    usage: dict = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }

    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error_result(
            f"Failed to parse LLM response as JSON: {exc}", "bad_json", raw, usage
        )

    return TranslationResult(
        metta_query=str(data.get("metta_query", "")),
        explanation=str(data.get("explanation", "")),
        intent=str(data.get("intent", "clarification")),
        requires_pln_inference=bool(data.get("requires_pln_inference", False)),
        confidence_filter=float(data.get("confidence_filter", 0.0)),
        warnings=list(data.get("warnings", [])),
        raw_response=raw,
        usage=usage,
    )
