"""Format a complete bot response from translation, validation, and PLN results."""
from __future__ import annotations

from core.llm_translator import TranslationResult
from core.metta_validator import ValidationResult
from core.pln_runner import PLNRunResult


def format_bot_response(
    translation: TranslationResult,
    validation: ValidationResult,
    pln_result: PLNRunResult,
    show_metta: bool,
    show_explanation: bool,
    show_debug: bool,
) -> str:
    parts: list[str] = []

    # Hard error from LLM layer
    if translation.error:
        parts.append(f"**Error:** {translation.error}")
        return "\n\n".join(parts)

    # Natural-language explanation
    if show_explanation and translation.explanation:
        parts.append(translation.explanation)

    # Generated MeTTa query
    if show_metta and translation.metta_query:
        parts.append(
            f"**Generated MeTTa Query:**\n```metta\n{translation.metta_query}\n```"
        )

    # Ontology warnings from the LLM
    if translation.warnings:
        warn_lines = "\n".join(f"- {w}" for w in translation.warnings)
        parts.append(f"**Ontology Warnings:**\n{warn_lines}")

    # Validation issues
    if not validation.valid and validation.issues:
        issue_lines = "\n".join(f"- {i}" for i in validation.issues)
        parts.append(f"**Validation Issues:**\n{issue_lines}")

    # PLN execution results
    if translation.metta_query:
        if pln_result.status == "error":
            parts.append(f"**PLN Error:** {pln_result.error}")
        elif pln_result.status == "empty":
            parts.append("**PLN Result:** No matching atoms found.")
        elif pln_result.results:
            mode_label = " *(stub)*" if pln_result.mode == "stub" else ""
            header = f"**PLN Results**{mode_label} — {pln_result.query_time_ms} ms:"
            result_lines = [header]
            for r in pln_result.results:
                if r.stv:
                    result_lines.append(
                        f"- `{r.atom}` → stv({r.stv['strength']:.2f}, {r.stv['confidence']:.2f})"
                    )
                else:
                    result_lines.append(f"- `{r.atom}`")
            parts.append("\n".join(result_lines))

    # Debug / token usage
    if show_debug:
        debug_lines: list[str] = []
        if translation.usage:
            u = translation.usage
            debug_lines.append(
                f"Tokens — prompt: {u['prompt_tokens']} / "
                f"completion: {u['completion_tokens']} / "
                f"total: {u['total_tokens']}"
            )
        debug_lines.append(f"Intent: {translation.intent}")
        debug_lines.append(f"Requires PLN inference: {translation.requires_pln_inference}")
        if translation.raw_response:
            debug_lines.append(f"Raw LLM response:\n```json\n{translation.raw_response}\n```")
        parts.append("**Debug Info:**\n" + "\n".join(debug_lines))

    return "\n\n".join(parts) if parts else "_No response generated._"
