"""Build the system prompt by injecting raw .metta content and few-shot examples
into the system_prompt.txt template.
"""
from __future__ import annotations

import json
from pathlib import Path

from ontology.registry import OntologyRegistry
from ontology.snapshots import build_ontology_context

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_template() -> str:
    path = _PROMPTS_DIR / "system_prompt.txt"
    return path.read_text(encoding="utf-8")


def _load_few_shot_text() -> str:
    path = _PROMPTS_DIR / "few_shot_examples.json"
    if not path.exists():
        return "_No examples loaded._"
    examples: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for i, ex in enumerate(examples, 1):
        response = {
            k: ex[k]
            for k in ("metta_query", "explanation", "intent",
                      "requires_pln_inference", "confidence_filter", "warnings")
            if k in ex
        }
        lines.append(f"Example {i}:")
        lines.append(f'  NL: "{ex["nl"]}"')
        lines.append(f"  Response: {json.dumps(response)}")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(registry: OntologyRegistry, raw_contents: dict[str, str] | None = None) -> str:
    """Return the fully-composed system prompt with ontology and examples injected.

    Parameters
    ----------
    registry:
        Parsed symbol registry (for the symbol index section).
    raw_contents:
        Filename -> raw .metta text mapping. When provided, the verbatim .metta
        content is injected as the primary ontology context.
    """
    template = _load_template()
    ontology_snapshot = build_ontology_context(raw_contents or {}, registry)
    few_shot_text = _load_few_shot_text()
    return template.format(
        ontology_snapshot=ontology_snapshot,
        few_shot_examples=few_shot_text,
    )
