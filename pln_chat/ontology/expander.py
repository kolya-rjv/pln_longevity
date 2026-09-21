"""Paper-to-PLN ontology expansion pipeline.

Workflow
--------
1. Extract plain text from an uploaded PDF, .txt, or .metta file.
2. Call the OpenAI API to identify new PLN ontology entries relevant to the
   health / longevity knowledge base.
3. Gate every proposed entry against the schema the rules actually consume
   (ontology/expansion_schema.py) — rejecting, with a reason, anything that
   invents a predicate, invents a truth value, names a tier the calibration
   table does not score, or arrives with no PMID and no DOI.
4. Check every surviving entry against the current merged ontology registry AND
   the raw .metta content to filter out duplicates.
5. Generate a commented MeTTa block for net-new entries.
6. Optionally append that block to a chosen .metta file.

Why steps 3 and 5 changed
-------------------------
The 2026-09-18 API evaluation (§10) fed this pipeline the taurine abstract
(Singh 2023, Science) and got back a block that was valid MeTTa and inert:

    "From the taurine abstract it minted new predicates (increases-life-span,
     declines-with-aging, reduces) instead of the KB's EvidenceIntervention /
     EvidenceHallmark, assigned LLM-made truth values (stv 0.93 0.9, a 'study
     confidence' constant of 0.92) that bypass the calibration tables, and
     recorded no PMID or DOI in the facts. Applied as-is, the new knowledge
     would not feed the existing PLN rules."

Three causes, all fixed here:

* **The prompt asked for it.** Two lines of `_EXTRACTION_SYSTEM_PROMPT` told
  the model to "extract constant definitions for confidence levels of novel
  study types" and to "set (stv s c) accordingly". They are gone; the prompt
  now enumerates the canonical forms with one verbatim example each, names the
  closed predicate list, and states that confidence is never proposed.
* **The model could not see the schema.** `call_extraction_llm` showed it
  `existing_raw_content[:6000]` — an ALPHABETICAL 6 KB slice of ~290 KB of
  ontology (2.1% of the KB, starting at `cellage_calibration.metta`) that
  contains no example of the target schema at all. It now sees
  `expansion_schema.SCHEMA_EXEMPLAR`, ~4 KB of verbatim canonical forms, plus
  the runtime schema card for deduplication.
* **Nothing checked the output.** Every entry now passes
  `expansion_schema.check_entry` before it can reach the block, and a refusal
  is REPORTED (`PipelineResult.rejected_entries`, `ExpandResponse`), never
  dropped on the floor.

Two duplicate-detection hazards were fixed at the same time, because the
constrained output walks straight into both:

* the dedup was LINE-based, and the canonical `Effect` form is written over TWO
  lines — `_strip_stv` only ever stripped a trailing `(stv x y)` on the same
  line, so the two-line form never matched anything and every re-extraction of
  an existing bridge looked new. Expressions are now split with
  `ontology.inventory.iter_top_level`, which heals line breaks and strips
  comments, and the STV is removed by balanced-paren scan rather than regex.
* the name fallback did `re.search(rf"\b{name}\b", all_raw)` over the whole
  ~290 KB of raw text, INCLUDING `;;` comments and the 107 KB
  `drugage_etl_short.metta` dump that the runtime deliberately excludes. A
  symbol mentioned once in a prose comment — `MechanisticConsensus`, say — was
  therefore reported as an existing duplicate and silently discarded. The
  fallback now asks `ontology.inventory`, which sees only real atoms in the
  runtime-sized file set.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai

from config import (
    OPENAI_API_KEY,
    ONTOLOGY_DIR,
    CUSTOM_ONTOLOGY_DIR,
    PLN_MAX_KB_FILE_BYTES,
)
from ontology.expansion_schema import (
    CANONICAL_PREDICATES,
    SCHEMA_EXEMPLAR,
    GateFinding,
    check_entry,
    evidence_categories,
    find_identifier,
    normalise_truth_values,
    unconsumed_predicates,
)
from ontology.inventory import (
    RuntimeInventory,
    inventory_for,
    iter_top_level,
    schema_card,
    split_args,
)
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
    #: PMID / DOI the entry is traceable to. An entry with neither is refused.
    pmid: Optional[str] = None
    doi: Optional[str] = None
    #: The study type the extractor read off the paper. Its NUMBER is never
    #: taken from the extractor — it is `(evidence-confidence <tier>)`.
    evidence_tier: Optional[str] = None
    #: The reported percent lifespan change, when the paper gives one. Present
    #: means the strength is DERIVED; absent means it is a curated prior.
    effect_size_pct: Optional[float] = None
    #: Values the extractor PROPOSED rather than the pipeline deriving them.
    provisional_fields: list[str] = field(default_factory=list)
    #: Derivation / provenance lines written into the generated comment.
    notes: list[str] = field(default_factory=list)

    @property
    def provisional(self) -> bool:
        return bool(self.provisional_fields)

    @property
    def identifier(self) -> Optional[str]:
        if self.pmid:
            return f"PMID_{self.pmid}"
        return self.doi


@dataclass
class RejectedEntry:
    """An entry the schema gate refused, with the reasons why.

    These are REPORTED, never dropped. The evaluation's complaint was not only
    that the pipeline produced unusable atoms — it was that a caller had no way
    to tell that it had. A rejection carries the offending MeTTa back to the
    caller so a human can fix the paper-reading, not guess at it.
    """
    kind: str
    name: str
    metta: str
    description: str
    findings: list[GateFinding] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [str(f) for f in self.findings]

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.findings]


@dataclass
class PipelineResult:
    new_entries: list[ExtractedEntry] = field(default_factory=list)
    duplicate_entries: list[ExtractedEntry] = field(default_factory=list)
    #: Entries the schema gate refused — see RejectedEntry.
    rejected_entries: list[RejectedEntry] = field(default_factory=list)
    paper_title: str = ""
    paper_summary: str = ""
    target_file: str = ""
    metta_block: str = ""
    applied: bool = False
    #: Predicates in the generated block that the runtime KB grounds nowhere
    #: else — the block would land them with zero existing facts to join.
    unconsumed_predicates: list[str] = field(default_factory=list)
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

# The prompt is BUILT, not a constant: the eleven legal evidence tiers and the
# closed predicate list are read out of the .metta files at call time
# (expansion_schema), so neither can drift from the calibration layer. The two
# lines the evaluation caught — "extract constant definitions for confidence
# levels of novel study types" and "set (stv s c) accordingly" — are gone, and
# their opposite is now stated three times.
_EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert in PLN (Probabilistic Logic Networks) and MeTTa (the language
of OpenCog Hyperon). You read one research paper (or abstract) about health,
aging, or longevity and express its findings in the EXISTING schema of a PLN
knowledge base.

You are NOT designing an ontology. The schema already exists and the reasoning
rules already exist. An entry that does not match one of the canonical forms
below is worthless: `infer`, `explain`, `rank-interventions`,
`recommend-supplements` and `drugage-effect` will never read it.

THE CANONICAL FORMS — copy these shapes exactly, substituting this paper's
atoms. Each one is verbatim from the knowledge base:

```metta
{exemplar}
```

THE CLOSED PREDICATE LIST. You may use these heads and NO others. Do not invent
a predicate, not even a descriptive one. If the paper says taurine increases
lifespan, that is an (Effect Taurine Lifespan Pos ...) link plus a measurement
row — NOT a new `increases-life-span` predicate.

{predicates}

TRUTH VALUES — read this twice.
- You MUST NOT propose a confidence. Confidence is a table lookup, and the table
  is the knowledge base's single authority on it.
- Write the confidence slot as (evidence-confidence <Tier>), UNEVALUATED, where
  <Tier> is one of exactly these {n_categories} values:
  {categories}
- A two-float truth value such as (stv 0.93 0.9) on an Effect link is REJECTED.
- You MUST NOT define a new confidence constant. `(= (study-confidence X) 0.92)`
  and anything like it is REJECTED.
- Report the paper's effect size in the `effect_size_pct` field instead; the
  pipeline derives the strength from it with the knowledge base's own transform.
  When you put a number in the strength slot it is treated as a provisional
  curated prior and labelled as such for a human reviewer.

PROVENANCE — every entry must carry an identifier.
- Put the paper's PMID (or its DOI) in each entry's `identifier` field.
- An entry with neither is REJECTED; it cannot be traced back to the paper.
- Carry the identifier into the atoms too, as (ReportedIn <row> PMID_<digits>)
  on a measurement row and (SupportedByPublication <record> <Publication>) on a
  review-level record — not only into a comment.

Respond ONLY with a single valid JSON object — no markdown fences, no extra text:
{{
  "paper_title": "<string: title of the paper, or 'Unknown'>",
  "paper_summary": "<string: 2-3 sentence summary relevant to PLN/longevity>",
  "identifier": "<string: the paper's PMID or DOI>",
  "entries": [
    {{
      "kind": "<one of: type | fact | evidence | effect | publication>",
      "name": "<canonical PLN symbol name — CamelCase for atoms>",
      "metta": "<one or more MeTTa expressions, newline-separated if multiple>",
      "description": "<plain-English description>",
      "identifier": "<PMID or DOI this entry rests on>",
      "evidence_tier": "<one of the {n_categories} tiers above, or null>",
      "effect_size_pct": <reported percent change in lifespan, or null>
    }}
  ]
}}

Extraction guidelines:
- CRITICAL — name reuse: before proposing any new atom name, scan the existing
  ontology summary supplied below. If the concept is already represented under
  another name, USE THE EXISTING NAME unconditionally.
- CRITICAL — argument order for binary predicates: always use the SAME argument
  order as the canonical forms above.
- CRITICAL — a fact is a duplicate if its predicate and all atom arguments
  match, regardless of truth values. Do not re-emit one with a different stv.
- Prefer the RAW measurement row (form 3) whenever the paper reports a percent
  change: it is source-faithful, and the knowledge base lifts it into a
  calibrated Effect link on its own.
- Effect sign convention: Pos = <from> RAISES <to>; Neg = <from> LOWERS <to>.
  A life-EXTENDING compound is (Effect <compound> Lifespan Pos ...).
- Do NOT emit rules, function signatures or MeTTa builtins.
- Aim for 5-25 high-quality entries; fewer good ones beat many vague ones.
"""


def build_extraction_prompt(
    categories: Optional[list[str]] = None,
    predicates: Optional[list[str]] = None,
) -> str:
    """Render the extraction prompt against the live calibration authorities."""
    cats = list(categories) if categories is not None else list(evidence_categories())
    preds = sorted(predicates) if predicates is not None else sorted(CANONICAL_PREDICATES - {":"})
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        exemplar=SCHEMA_EXEMPLAR.rstrip(),
        predicates="  " + ", ".join(preds),
        categories="  " + ", ".join(cats),
        n_categories=len(cats),
    )


def call_extraction_llm(
    paper_text: str,
    existing_symbols: list[str],
    existing_raw_content: str,
    model: str,
    temperature: float,
) -> dict:
    """Call OpenAI to extract ontology entries from paper text.

    Returns the parsed JSON dict from the LLM response.

    `existing_raw_content` is no longer pasted in verbatim. It used to be sliced
    to its first 6,000 characters — an ALPHABETICAL slice of ~290 KB, i.e. 2.1%
    of the knowledge base and not one example of the schema the model was being
    asked to produce. The schema now lives in the system prompt as verbatim
    canonical forms; what the model gets HERE is the compact schema card (what
    is really grounded, and how much of it), which is what deduplication
    actually needs.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    paper_max = 10_000
    text_excerpt = paper_text[:paper_max]
    if len(paper_text) > paper_max:
        text_excerpt += "\n...[paper truncated for length]"

    existing_block = ""
    if existing_symbols:
        sym_list = ", ".join(existing_symbols[:1200])
        existing_block = (
            "\n\nEXISTING ONTOLOGY SYMBOLS (do not re-extract these, and reuse "
            f"them by name where the paper is about the same atom): {sym_list}"
        )
    if existing_raw_content.strip():
        existing_block += (
            "\n\nWHAT THE RUNTIME KNOWLEDGE BASE ACTUALLY HOLDS:\n"
            f"{existing_raw_content.strip()}"
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
            {"role": "system", "content": build_extraction_prompt()},
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


def _runtime_sized_paths() -> list[Path]:
    """Every .metta file small enough to be a RUNTIME file.

    The name-search fallback used to scan `_load_all_raw_content()`, which
    includes the 107 KB `drugage_etl_short.metta` dump the runtime deliberately
    excludes (config.PLN_MAX_KB_FILE_BYTES — hyperon 0.2.10 aborts on a space
    that size). A compound that occurs ONLY in that dump was therefore reported
    as an existing duplicate of an entry the runtime could never answer about.
    """
    paths: list[Path] = []
    for base in (ONTOLOGY_DIR, CUSTOM_ONTOLOGY_DIR):
        if not base.exists():
            continue
        for path in sorted(base.glob("*.metta")):
            try:
                if path.stat().st_size <= PLN_MAX_KB_FILE_BYTES:
                    paths.append(path)
            except OSError:
                continue
    return paths


def _normalise_metta(expr: str) -> str:
    """Strip ;; comments and collapse whitespace."""
    no_comments = re.sub(r';;[^\n]*', '', expr)
    return re.sub(r'\s+', ' ', no_comments).strip()


def _strip_stv(expr: str) -> str:
    r"""Remove a trailing `(stv …)` — however it is written.

    HAZARD THIS FIXES. The old implementation was a regex anchored to the end of
    a LINE and matching two bare floats:

        re.compile(r'\(stv\s+[\d.]+\s+[\d.]+\)\s*$')

    The canonical Effect form in `mechanistic_bridges.metta` is written over two
    lines AND carries a nested lookup in the confidence slot:

        (Effect CellularSenescence SASP Pos
           (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))

    — so the regex matched neither half, the STV was never stripped, and a
    re-extraction of a bridge the knowledge base already holds came back as
    net-new. This walks back from the end with a paren counter instead, which is
    indifferent to line breaks and to what is nested inside.
    """
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        parts = split_args(text[1:-1].strip())
        if parts and parts[-1].startswith("(stv "):
            return ("(" + " ".join(parts[:-1])).rstrip()
        return text
    # An unwrapped fragment: fall back to the trailing-group scan.
    if not text.endswith(")"):
        return text
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                if text[i:].startswith("(stv "):
                    return text[:i].rstrip()
                return text
    return text


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

    # Inheritance A B (stv? already stripped in no_stv). Left exactly as it
    # was: these two branches are about ARGUMENT ORDER, not line breaks, and
    # widening them to also match the parenthesised form would start treating
    # (Inheritance Supplement Omega3) as a duplicate of (Inheritance Omega3
    # Supplement) — Inheritance is not symmetric. Out of scope here.
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


def _expression_forms(text: str) -> list[str]:
    """Canonical forms for every top-level expression in `text`.

    `iter_top_level` (ontology/inventory.py) is what heals the two-line hazard:
    it strips comments and rejoins a wrapped expression before anything compares
    it, so the canonical two-line `Effect` form is normalised to the same single
    string whether it was written on one line or three.
    """
    out: list[str] = []
    for expr in iter_top_level(text):
        norm = _normalise_metta(expr)
        if norm:
            out.extend(_canonical_forms(norm))
    return out


def _build_normalised_set(all_raw: str) -> set[str]:
    """Build the full set of canonical forms from existing .metta content."""
    return set(_expression_forms(all_raw))


def _is_duplicate(
    entry: ExtractedEntry,
    registry: OntologyRegistry,
    all_raw: str,
    normalised_raw_lines: set[str],
    inventory: Optional[RuntimeInventory] = None,
) -> bool:
    """Return True if this entry already exists in the ontology.

    Checks:
    1. Parsed-registry symbol lookup.
    2. Whole-expression canonical form comparison (STV-agnostic,
       arg-order-agnostic, line-break-agnostic).
    3. Name lookup in the RUNTIME INVENTORY as a final fallback.

    Step 3 used to be `re.search(rf"\b{name}\b", all_raw)` over ~290 KB of raw
    text. That text includes every `;;` comment and the excluded ETL dump, so a
    name that appears once in prose — `MechanisticConsensus` is mentioned in
    `mechanistic_bridges.metta`'s header as a tier that does NOT exist yet — was
    reported as an existing duplicate and the entry was thrown away. The
    inventory sees parsed atoms in runtime-sized files only, so it answers the
    question that was actually being asked.
    """
    # 1. Registry symbol check
    if registry.get(entry.name):
        return True

    # 2. Whole-expression canonical form comparison
    for form in _expression_forms(entry.metta):
        if form in normalised_raw_lines:
            return True

    # 3. Fallback: does the runtime knowledge base really hold this symbol?
    if entry.name and not any(c in entry.name for c in ('-', ' ', '(', ')')):
        if inventory is not None:
            return inventory.knows_symbol(entry.name)
        return bool(re.search(rf"\b{re.escape(entry.name)}\b", all_raw))

    return False


# ── MeTTa block generation ────────────────────────────────────────────────────

def _carry_identifier(metta: str, pmid: Optional[str]) -> str:
    """Put the PMID INTO the atoms, not only into the block's header comment.

    The evaluation's block recorded its source in a `;;` line and nowhere else,
    so a fact lifted out of it had no way back to the paper. Every measurement
    row `(InstanceOf <row> Experiment)` that does not already carry a
    `(ReportedIn <row> …)` edge gets one here — the same provenance shape the
    DrugAge ETL emits and `drugage_calibration.metta` documents.
    """
    if not pmid:
        return metta
    rows: list[str] = []
    reported: set[str] = set()
    for expr in iter_top_level(metta):
        if not (expr.startswith("(") and expr.endswith(")")):
            continue
        parts = split_args(expr[1:-1].strip())
        if len(parts) == 3 and parts[0] == "InstanceOf" and parts[2] == "Experiment":
            rows.append(parts[1])
        elif len(parts) >= 2 and parts[0] == "ReportedIn":
            reported.add(parts[1])
    missing = [r for r in rows if r not in reported]
    if not missing:
        return metta
    added = "\n".join(f"(ReportedIn {row} PMID_{pmid})" for row in missing)
    return metta.rstrip() + "\n" + added



def generate_metta_block(
    entries: list[ExtractedEntry],
    paper_title: str,
    source_note: str,
) -> str:
    """Produce a commented MeTTa block ready for appending to a .metta file.

    The comment above each entry is the audit trail a human reviewer reads. It
    names the identifier the entry rests on, says where the confidence came from
    (always: the calibration table), and says whether the strength was DERIVED
    from a reported effect size or is a CURATED PRIOR the extractor proposed.
    Anything provisional is called provisional, in capitals, in the file itself
    — not only in an API response the reviewer may never see.
    """
    if not entries:
        return ""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = [
        f";; ── Auto-expanded from: {paper_title} ──────────────────────────────",
        f";; Source file : {source_note}",
        f";; Generated   : {timestamp}",
        ";; HONESTY CONTRACT: confidence is (evidence-confidence <Tier>), left",
        ";;   unevaluated — epistemic_calibration.metta remains the authority.",
        ";;   Strength is either a reproducible transform of a reported effect",
        ";;   size or a CURATED PRIOR, and each entry below says which.",
        ";; REVIEW BEFORE TRUSTING: every PROVISIONAL line is a value the",
        ";;   extractor proposed from the paper, not one this layer derived.",
        "",
    ]
    for entry in entries:
        if entry.description:
            lines.append(f";; [{entry.kind}] {entry.description}")
        if entry.identifier:
            provenance = f";;   provenance : {entry.identifier}"
            if entry.doi and entry.pmid:
                provenance += f"  (doi {entry.doi})"
            lines.append(provenance)
        for note in entry.notes:
            lines.append(f";;   {note}")
        if entry.provisional:
            lines.append(
                ";;   PROVISIONAL: " + ", ".join(entry.provisional_fields)
                + " — extractor-proposed, check against the paper."
            )
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

    # What the RUNTIME really holds — the authority for step 4's name fallback
    # and for "would this predicate land with zero facts?" in step 6.
    inventory = inventory_for(_runtime_sized_paths())
    categories = list(evidence_categories())

    # Pre-compute canonical form set for dedup: STV-agnostic + arg-order-agnostic.
    normalised_raw_lines = _build_normalised_set(all_raw)

    # 3. Call the LLM ─────────────────────────────────────────────────────────
    try:
        extracted = call_extraction_llm(
            paper_text=paper_text,
            existing_symbols=registry.all_symbols(),
            existing_raw_content=schema_card(inventory),
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

    paper_identifier = str(extracted.get("identifier") or "")

    # 4. Gate, then classify: rejected vs. duplicate vs. new ──────────────────
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "fact"))
        name = str(item.get("name", ""))
        metta = str(item.get("metta", ""))
        description = str(item.get("description", ""))

        # Discard entries with no name or no MeTTa expression
        if not name.strip() or not metta.strip():
            continue
        # Basic sanity check: MeTTa should contain parentheses
        if "(" not in metta:
            continue

        identifier_text = " ".join(
            str(item.get(key) or "") for key in ("identifier", "pmid", "doi")
        ) + " " + paper_identifier

        findings = check_entry(
            metta,
            identifier_text=identifier_text,
            categories=categories,
            inventory=inventory,
        )
        if findings:
            result.rejected_entries.append(RejectedEntry(
                kind=kind, name=name, metta=metta,
                description=description, findings=findings,
            ))
            continue

        pmid, doi = find_identifier(identifier_text, metta)
        tier = item.get("evidence_tier") or None
        if tier is not None and str(tier) not in categories:
            tier = None
        pct = item.get("effect_size_pct")
        try:
            pct = float(pct) if pct is not None else None
        except (TypeError, ValueError):
            pct = None

        normalised = normalise_truth_values(
            metta,
            evidence_tier=str(tier) if tier else None,
            effect_size_pct=pct,
        )

        entry = ExtractedEntry(
            kind=kind,
            name=name,
            metta=_carry_identifier(normalised.metta, pmid),
            description=description,
            pmid=pmid,
            doi=doi,
            evidence_tier=str(tier) if tier else None,
            effect_size_pct=pct,
            provisional_fields=list(normalised.provisional_fields),
            notes=list(normalised.notes),
        )
        entry.duplicate = _is_duplicate(
            entry, registry, all_raw, normalised_raw_lines, inventory
        )
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
        result.unconsumed_predicates = unconsumed_predicates(
            result.metta_block, inventory
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
