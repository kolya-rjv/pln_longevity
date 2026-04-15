"""Paper-to-PLN ontology expansion pipeline.

Workflow
--------
1. Extract plain text from an uploaded PDF, .txt, or .metta file.
2. Call the OpenAI API to identify new PLN ontology entries relevant to the
   health / longevity knowledge base.
3. Check every proposed entry against the current merged ontology registry AND
   the raw .metta content to filter out duplicates.
4. Generate a commented MeTTa block for net-new entries.
5. Optionally append that block to a chosen .metta file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai

from config import OPENAI_API_KEY, ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR
from ontology.loader import load_specific_files
from ontology.registry import OntologyRegistry


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntry:
    kind: str         # type | function | predicate | constant | rule | fact
    name: str         # canonical symbol name / identifier
    metta: str        # one or more MeTTa expressions (newline-separated)
    description: str  # human-readable description
    duplicate: bool = False


@dataclass
class PipelineResult:
    new_entries: list[ExtractedEntry] = field(default_factory=list)
    duplicate_entries: list[ExtractedEntry] = field(default_factory=list)
    paper_title: str = ""
    paper_summary: str = ""
    target_file: str = ""
    metta_block: str = ""
    applied: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_upload(data: bytes, filename: str) -> str:
    """Return plain text from an uploaded file (PDF, .txt, or .metta)."""
    fname = (filename or "").lower()
    if fname.endswith(".pdf"):
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as exc:
            raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    # Plain text / metta
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode the uploaded file as text.")


# ── OpenAI extraction ──────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert in PLN (Probabilistic Logic Networks) and MeTTa (the language
of OpenCog Hyperon). Your task is to read a research paper (or abstract) about
health, aging, or longevity and extract NEW ontology entries to add to a PLN
knowledge base.

MeTTa syntax you MUST follow exactly:
  Type declaration:          (: TypeName Type)
  Function/predicate sig:    (: fn-name (-> ArgType1 ArgType2 RetType))
  Constant definition:       (= (fn-name ConstantArg) NumericOrAtomValue)
  Rewrite rule:              (= (rule-name $var ...) body-expression)
  Knowledge fact:            (Inheritance SubAtom SuperAtom (stv s c))
                             (Evaluation (predicate args) (stv s c))
  STV truth value:           (stv strength confidence)  — both floats in [0, 1]
  Variables start with $
  Type/constant names:       CamelCase
  Function/predicate names:  kebab-case

Respond ONLY with a single valid JSON object — no markdown fences, no extra text:
{
  "paper_title": "<string: title of the paper, or 'Unknown'>",
  "paper_summary": "<string: 2-3 sentence summary relevant to PLN/longevity>",
  "entries": [
    {
      "kind": "<one of: type | function | predicate | constant | rule | fact>",
      "name": "<canonical PLN symbol name>",
      "metta": "<one or more MeTTa expressions, newline-separated if multiple>",
      "description": "<plain-English description>"
    }
  ]
}

Extraction guidelines:
- CRITICAL — name reuse: Before proposing any new type, predicate, or function
  name, scan the existing ontology supplied below. If a concept is already
  represented under a different name (e.g. 'can-be-targeted-by' and
  'can-be-modulated-by' express the same relationship), USE THE EXISTING NAME
  unconditionally. Never introduce a synonym for an existing symbol.
- CRITICAL — argument order for symmetric/binary predicates: always use the
  SAME argument order as the existing ontology. If (interconnected-with A B) is
  already present, do not emit (interconnected-with B A) as a new fact.
- CRITICAL — STV variation: do NOT emit a fact whose predicate + arguments
  already exist in the ontology just because the stv values differ slightly.
  A fact is a duplicate if its predicate and all atom arguments match, regardless
  of stv values.
- Extract types for novel biological/medical categories (new InterventionClass,
  Biomarker, EvidenceCategory, Pathway, DrugClass values, etc.).
- Extract function/predicate declarations for novel relationships.
- Extract constant definitions for confidence levels of novel study types.
- Extract rules useful for PLN reasoning about longevity.
- Extract knowledge facts (Inheritance / Evaluation) for specific relationships
  described in the paper with supporting evidence; set (stv s c) accordingly.
- Do NOT include MeTTa builtins (stv, match, Inheritance, etc.) as new entries.
- Do NOT extract vague or purely generic concepts.
- Aim for 5-30 high-quality, domain-specific entries.
"""


def call_extraction_llm(
    paper_text: str,
    existing_symbols: list[str],
    existing_raw_content: str,
    model: str,
    temperature: float,
) -> dict:
    """Call OpenAI to extract ontology entries from paper text.

    Returns the parsed JSON dict from the LLM response.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Allocate token budget: leave room for system prompt + response
    paper_max = 10_000
    ontology_max = 6_000

    text_excerpt = paper_text[:paper_max]
    if len(paper_text) > paper_max:
        text_excerpt += "\n...[paper truncated for length]"

    # Send full raw ontology content so the LLM sees the exact existing MeTTa
    # expressions, not just a symbol list — critical for deduplication.
    ontology_excerpt = existing_raw_content[:ontology_max]
    if len(existing_raw_content) > ontology_max:
        ontology_excerpt += "\n;; ...[ontology truncated for length]"

    existing_block = ""
    if ontology_excerpt.strip():
        existing_block = (
            f"\n\nEXISTING ONTOLOGY (verbatim MeTTa — do NOT re-extract anything "
            f"already present here):\n```metta\n{ontology_excerpt}\n```"
        )
    elif existing_symbols:
        # Fallback: symbol list if no raw content available
        sym_list = ", ".join(existing_symbols)
        existing_block = (
            f"\n\nEXISTING ONTOLOGY SYMBOLS (do not re-extract these): {sym_list}"
        )

    user_message = (
        f"Extract PLN ontology entries from the research paper below.{existing_block}\n\n"
        f"---PAPER START---\n{text_excerpt}\n---PAPER END---"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# ── Duplicate detection ────────────────────────────────────────────────────────

def _load_all_raw_content() -> str:
    """Return the concatenated raw text of every .metta file in the project."""
    parts: list[str] = []
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if base.exists():
            for p in sorted(base.glob("*.metta")):
                try:
                    parts.append(p.read_text(encoding="utf-8"))
                except OSError:
                    pass
    return "\n".join(parts)


# Matches the trailing (stv X Y) on a fact line
_STV_RE = re.compile(r'\(stv\s+[\d.]+\s+[\d.]+\)\s*$')


def _normalise_metta(expr: str) -> str:
    """Strip ;; comments and collapse whitespace."""
    no_comments = re.sub(r';;[^\n]*', '', expr)
    return re.sub(r'\s+', ' ', no_comments).strip()


def _strip_stv(expr: str) -> str:
    """Remove a trailing (stv X Y) so comparisons are STV-value-agnostic."""
    return _STV_RE.sub('', expr).strip()


def _canonical_forms(norm: str) -> list[str]:
    """Return a list of canonical string variants for duplicate detection.

    Generates up to four forms per expression:
    - verbatim normalised
    - STV stripped
    - arg-swapped (for symmetric binary preds: Evaluation (pred A B) / Inheritance A B)
    - arg-swapped + STV stripped
    """
    no_stv = _strip_stv(norm)
    forms = {norm, no_stv}

    # Inheritance A B (stv? already stripped in no_stv)
    m_inh = re.match(r'^(Inheritance)\s+(\S+)\s+(\S+)(.*)', no_stv)
    if m_inh:
        swapped = f"{m_inh.group(1)} {m_inh.group(3)} {m_inh.group(2)}{m_inh.group(4)}"
        forms.add(swapped)

    # Evaluation (pred A B) …
    m_eval = re.match(r'^(Evaluation\s+\()(\S+)\s+(\S+)\s+(\S+)(\))(.*)', no_stv)
    if m_eval:
        pred, a, b = m_eval.group(2), m_eval.group(3), m_eval.group(4)
        swapped = (
            f"{m_eval.group(1)}{pred} {b} {a}{m_eval.group(5)}{m_eval.group(6)}"
        )
        forms.add(swapped)

    return list(forms)


def _build_normalised_set(all_raw: str) -> set[str]:
    """Build the full set of canonical forms from existing .metta content."""
    result: set[str] = set()
    for raw_line in all_raw.splitlines():
        norm = _normalise_metta(raw_line)
        if norm and not norm.startswith(';'):
            for form in _canonical_forms(norm):
                result.add(form)
    return result


def _is_duplicate(
    entry: ExtractedEntry,
    registry: OntologyRegistry,
    all_raw: str,
    normalised_raw_lines: set[str],
) -> bool:
    """Return True if this entry already exists in the ontology.

    Checks:
    1. Parsed-registry symbol lookup.
    2. Line-by-line canonical form comparison (STV-agnostic, arg-order-agnostic).
    3. Word-boundary name search as final fallback.
    """
    # 1. Registry symbol check
    if registry.get(entry.name):
        return True

    # 2. Canonical form comparison for each MeTTa line in this entry
    for raw_line in entry.metta.splitlines():
        norm = _normalise_metta(raw_line)
        if not norm or norm.startswith(';'):
            continue
        for form in _canonical_forms(norm):
            if form in normalised_raw_lines:
                return True

    # 3. Fallback: word-boundary name search (simple CamelCase symbol names)
    if entry.name and not any(c in entry.name for c in ('-', ' ', '(', ')')):
        pattern = rf"\b{re.escape(entry.name)}\b"
        if re.search(pattern, all_raw):
            return True

    return False


# ── MeTTa block generation ────────────────────────────────────────────────────

def generate_metta_block(
    entries: list[ExtractedEntry],
    paper_title: str,
    source_note: str,
) -> str:
    """Produce a commented MeTTa block ready for appending to a .metta file."""
    if not entries:
        return ""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = [
        f";; ── Auto-expanded from: {paper_title} ──────────────────────────────",
        f";; Source file : {source_note}",
        f";; Generated   : {timestamp}",
        "",
    ]
    for entry in entries:
        if entry.description:
            lines.append(f";; [{entry.kind}] {entry.description}")
        lines.extend(entry.metta.strip().splitlines())
        lines.append("")
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _collect_all_metta_paths() -> list[Path]:
    paths: list[Path] = []
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if base.exists():
            paths.extend(sorted(base.glob("*.metta")))
    return paths


def run_expansion_pipeline(
    paper_data: bytes,
    filename: str,
    target_file_path: Path,
    model: str,
    temperature: float,
    apply: bool = False,
) -> PipelineResult:
    """End-to-end pipeline: parse paper → LLM extract → dedup → optionally write.

    Parameters
    ----------
    paper_data:
        Raw bytes of the uploaded file (PDF or plain text).
    filename:
        Original filename — used to determine the file type.
    target_file_path:
        Absolute path to the .metta file to expand (will be created if absent).
    model:
        OpenAI model identifier.
    temperature:
        Sampling temperature (lower = more deterministic).
    apply:
        When True, the generated MeTTa block is appended to *target_file_path*.
    """
    result = PipelineResult(target_file=target_file_path.name)

    # 1. Extract plain text ───────────────────────────────────────────────────
    try:
        paper_text = extract_text_from_upload(paper_data, filename)
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    if not paper_text.strip():
        result.error = "Uploaded file appears to be empty or unreadable."
        return result

    # 2. Build merged registry from all available .metta files ───────────────
    all_paths = _collect_all_metta_paths()
    registry, _ = load_specific_files(all_paths)
    all_raw = _load_all_raw_content()

    # Pre-compute canonical form set for dedup: STV-agnostic + arg-order-agnostic.
    normalised_raw_lines = _build_normalised_set(all_raw)

    # 3. Call the LLM ─────────────────────────────────────────────────────────
    try:
        extracted = call_extraction_llm(
            paper_text=paper_text,
            existing_symbols=registry.all_symbols(),
            existing_raw_content=all_raw,
            model=model,
            temperature=temperature,
        )
    except Exception as exc:
        result.error = f"LLM extraction failed: {exc}"
        return result

    result.paper_title = extracted.get("paper_title", "Unknown")
    result.paper_summary = extracted.get("paper_summary", "")

    raw_entries = extracted.get("entries", [])
    if not isinstance(raw_entries, list):
        result.error = "LLM returned an unexpected response format."
        return result

    # 4. Classify entries: new vs. duplicate ──────────────────────────────────
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        entry = ExtractedEntry(
            kind=item.get("kind", "fact"),
            name=item.get("name", ""),
            metta=item.get("metta", ""),
            description=item.get("description", ""),
        )
        # Discard entries with no name or no MeTTa expression
        if not entry.name.strip() or not entry.metta.strip():
            continue
        # Basic sanity check: MeTTa should contain parentheses
        if "(" not in entry.metta:
            continue
        entry.duplicate = _is_duplicate(entry, registry, all_raw, normalised_raw_lines)
        if entry.duplicate:
            result.duplicate_entries.append(entry)
        else:
            result.new_entries.append(entry)

    # 5. Generate MeTTa block ─────────────────────────────────────────────────
    if result.new_entries:
        result.metta_block = generate_metta_block(
            result.new_entries,
            result.paper_title,
            filename,
        )

    # 6. Write to file (if requested) ─────────────────────────────────────────
    if apply and result.metta_block:
        try:
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file_path, "a", encoding="utf-8") as fh:
                # Add a blank separator if the file already has content
                if target_file_path.exists() and target_file_path.stat().st_size > 0:
                    fh.write("\n\n")
                fh.write(result.metta_block)
            result.applied = True
        except OSError as exc:
            result.error = f"Failed to write to {target_file_path.name}: {exc}"

    return result
