"""Call the OpenAI API and parse the structured JSON response into a
TranslationResult dataclass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import openai

from config import OPENAI_API_KEY


@dataclass
class TranslationResult:
    metta_query: str
    explanation: str
    intent: str                    # retrieval | inference | assertion | clarification
    requires_pln_inference: bool
    confidence_filter: float
    warnings: list[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _error_result(message: str, raw: Optional[str] = None, usage: Optional[dict] = None) -> TranslationResult:
    return TranslationResult(
        metta_query="",
        explanation="",
        intent="clarification",
        requires_pln_inference=False,
        confidence_filter=0.0,
        warnings=[],
        raw_response=raw,
        usage=usage,
        error=message,
    )


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
        return _error_result("OPENAI_API_KEY is not set — add it to your .env file.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
        return _error_result("Invalid OpenAI API key.")
    except openai.RateLimitError:
        return _error_result("OpenAI rate limit exceeded — please wait and retry.")
    except openai.APIError as exc:
        return _error_result(f"OpenAI API error: {exc}")

    raw: str = response.choices[0].message.content or ""
    usage: dict = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }

    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error_result(f"Failed to parse LLM response as JSON: {exc}", raw, usage)

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
