"""The schema a paper has to land in, and the gate that enforces it.

The problem this solves
-----------------------
The 2026-09-18 API evaluation fed the taurine abstract (Singh 2023, Science) to
`POST /ontology/expand` and got back a block that was syntactically valid MeTTa
and epistemically worthless (evaluation §10):

  * it minted three predicates that exist nowhere in the KB —
    `increases-life-span`, `declines-with-aging`, `reduces`;
  * it invented truth values — `(stv 0.93 0.9)` — and even minted a *new
    confidence constant*, `(= (study-confidence ...) 0.92)`, which bypasses
    `epistemic_calibration.metta`, the file the whole calibration layer treats
    as the single authority on confidence;
  * it recorded no PMID and no DOI in any generated fact, only in a header
    comment, so nothing downstream could trace a claim to its paper.

Applied as-is, that block is inert: `infer`, `explain`, `rank-interventions`,
`patient-relevance`, `recommend-supplements` and `drugage-effect` all return
`[[]]` for every atom in it. Measured, not assumed — the schema in this module
was verified end to end against the live runtime KB (hyperon 0.2.10) and every
one of those six entry points consumes it; see
`tests/test_ontology_expansion.py::test_the_canonical_block_is_consumed_by_the_rules`.

What this module is
-------------------
Three things the extractor used to leave to the language model:

1. **The vocabulary.** `CANONICAL_PREDICATES` is the closed list of heads a
   generated block may use, each one chosen because a rule in the repository
   reads it. Anything else is rejected, not silently appended.
2. **The confidence.** Confidence is never proposed. The model may name a STUDY
   TYPE; the emitted atom carries `(evidence-confidence <Tier>)` *unevaluated*,
   exactly as `mechanistic_bridges.metta` writes it, so
   `epistemic_calibration.metta` remains the one place a number lives. The
   eleven legal tiers are READ from that file (`evidence_categories`), never
   restated here.
3. **The strength.** Where the KB already has a reproducible transform for a
   reported effect size — `drugage_calibration.metta`'s saturating
   `s = |pct| / (|pct| + k)`, with `k` read from that file's §1 knob — the
   strength is DERIVED from the paper's number. Where the paper reports no
   usable effect size, the strength is a CURATED PRIOR, and it is labelled as
   one in the generated comment and flagged `provisional` in the API response.

HONESTY CONTRACT (the same one `mechanistic_bridges.metta` states):
  * confidence  is DERIVED — a lookup into the single authority, emitted
                unevaluated so it cannot drift from that authority;
  * strength    is either a reproducible function of a reported effect size, or
                an explicitly LABELLED curated prior;
  * provenance  is mandatory — an entry with no PMID and no DOI is rejected,
                and the identifier is carried into the atoms, not the comment.

Nothing here evaluates MeTTa or calls an LLM: it is a text scan over files the
repository already ships, so it costs milliseconds and cannot panic hyperon.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from config import ONTOLOGY_DIR
from ontology.inventory import RuntimeInventory, iter_top_level, split_args

# ================================================================
# 0. The calibration authorities, read rather than restated
# ================================================================
# Both of these are parsed out of the .metta files at call time. Hard-coding
# either would create a second source of truth for a number the calibration
# layer owns — the precise failure this module exists to stop.

_EPISTEMIC_FILE = "epistemic_calibration.metta"
_DRUGAGE_FILE = "drugage_calibration.metta"

#: Used only when `epistemic_calibration.metta` cannot be read at all (a caller
#: pointed the pipeline at a directory that does not hold it). Kept minimal and
#: deliberately conservative: an unreadable authority must not silently widen
#: the accepted vocabulary.
_FALLBACK_CATEGORIES: tuple[str, ...] = ()

#: `drugage_calibration.metta` §1: strength half-saturation, in percent.
_FALLBACK_HALFSAT = 20.0


def _ontology_file(name: str, ontology_dir: Optional[Path] = None) -> Optional[Path]:
    base = ontology_dir or ONTOLOGY_DIR
    path = base / name
    return path if path.exists() else None


def evidence_categories(ontology_dir: Optional[Path] = None) -> tuple[str, ...]:
    """The closed EvidenceCategory enum, read from the calibration authority.

    `epistemic_calibration.metta` declares the enumeration with eleven
    `(: <Tier> EvidenceCategory)` lines and calls itself "the single authority".
    We take it at its word: the legal tiers are whatever that file declares
    today, in declaration order (highest confidence first, as the file groups
    them), so adding a tier there is all it takes to make it proposable.
    """
    path = _ontology_file(_EPISTEMIC_FILE, ontology_dir)
    if path is None:
        return _FALLBACK_CATEGORIES
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_CATEGORIES

    found: list[str] = []
    for expr in iter_top_level(text):
        if not (expr.startswith("(") and expr.endswith(")")):
            continue
        parts = split_args(expr[1:-1].strip())
        if len(parts) == 3 and parts[0] == ":" and parts[2] == "EvidenceCategory":
            if parts[1] not in found:
                found.append(parts[1])
    return tuple(found)


_HALFSAT_RE = re.compile(r"\(=\s*\(lifespan-halfsat\)\s*([-+\d.eE]+)\s*\)")


def lifespan_halfsat(ontology_dir: Optional[Path] = None) -> float:
    """`k` in `s = |pct| / (|pct| + k)`, read from drugage_calibration.metta §1.

    That file calls its §1 "KNOBS — every tunable constant lives here"; reading
    the knob means a tuning pass there retunes generated blocks too, instead of
    leaving this module quietly stuck on last month's value.
    """
    path = _ontology_file(_DRUGAGE_FILE, ontology_dir)
    if path is None:
        return _FALLBACK_HALFSAT
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_HALFSAT
    match = _HALFSAT_RE.search(text)
    if not match:
        return _FALLBACK_HALFSAT
    try:
        value = float(match.group(1))
    except ValueError:
        return _FALLBACK_HALFSAT
    return value if value > 0 else _FALLBACK_HALFSAT


def lifespan_strength(pct: float, halfsat: Optional[float] = None) -> float:
    """The DrugAge strength transform, reproduced exactly (drugage_calibration §4).

    Magnitude only — the sign is a separate axis. `|pct| / (|pct| + k)` squashes
    an unbounded percent change into [0, 1); a replicated ~+11% mouse extension
    lands near 0.35, a huge +80% near 0.8.
    """
    k = lifespan_halfsat() if halfsat is None else halfsat
    magnitude = abs(float(pct))
    return magnitude / (magnitude + k)


# ================================================================
# 1. The closed vocabulary
# ================================================================
# Every head here is read by something: the comment names what. A generated
# block that stays inside this list is consumed by the existing rules; one that
# leaves it is the evaluation's inert block.
CANONICAL_PREDICATES: frozenset[str] = frozenset({
    # Structural, consumed by everything.
    ":",                        # type / record declaration
    "Inheritance",              # (Inheritance Taurine Supplement) — typing a new node
    "InstanceOf",               # (InstanceOf <row> Experiment)

    # Publication provenance — hallmarks_core.metta / grim_age_core.metta.
    "PublicationTitle",
    "PublicationYear",
    "JournalName",
    "DOI",
    "SupportedByPublication",

    # A raw measurement row — lifted by drugage_calibration.metta's
    # `drugage-effect`, and stepped through by its `infer` equation.
    "UsesIntervention",
    "UsesSpecies",
    "AvgLifespanChangePercent",
    "AvgLifespanSignificance",
    "IsITPStudy",
    "ReportedIn",               # the provenance edge: <row> -> PMID_…

    # A review-level audit record — hallmarks_lopezotin2023_intervention_evidence.
    "EvidenceHallmark",
    "EvidenceIntervention",
    "EvidenceSpeciesModel",
    "EvidenceOutcomeText",
    "EvidenceReferenceNumber",

    # The inference substrate — pln_deduction.metta's `infer` / `explain`,
    # pln_intervention_ranking.metta's `rank-interventions`.
    "Effect",

    # Lookup + Layer-4 modifiers — hallmark_targeting.metta,
    # supplement_evidence.metta, pln_supplement_recommendation.metta.
    "TargetsHallmark",
    "EvidenceLevel",
    "SafetyProfile",
    "Interaction",
})

#: Functions whose definition belongs to the calibration layer. A generated
#: block that re-defines one of these silently replaces the authority — the
#: `(= (study-confidence …) 0.92)` move the evaluation caught.
CALIBRATION_FUNCTIONS: frozenset[str] = frozenset({
    "evidence-confidence",
    "selection-basis-confidence",
    "apply-tv",
    "calibrate-tv",
    "sig-gate",
    "lifespan-halfsat",
    "clade-category",
    "chain-discount",
    "rank-score",
})

#: A `(= (<name> …) <float>)` whose name looks like a confidence knob. The
#: evaluation's block minted exactly this shape under a brand-new name, so
#: matching on the known names alone would not have caught it.
_CONFIDENCE_NAME_RE = re.compile(r"(confidence|conf|-tv|stv|certainty|weight)", re.I)

#: A bare MeTTa symbol name — used to tell a nested predicate head from a
#: number, a string literal or a variable.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")

#: A literal two-float truth value: `(stv 0.93 0.9)`. The second slot MUST be
#: an `(evidence-confidence <Tier>)` lookup instead.
_LITERAL_STV_RE = re.compile(r"\(stv\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)")

#: The calibrated form, with the lookup left unevaluated.
_CALIBRATED_STV_RE = re.compile(
    r"\(stv\s+([-+\d.eE]+)\s+\(evidence-confidence\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)"
)

#: PMID:37289866 / PMID 37289866 / PMID_37289866, and a bare DOI.
_PMID_RE = re.compile(r"\bPMID[\s:_-]*(\d{4,9})\b", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'),;]+)")


def allowed_predicates(inventory: Optional[RuntimeInventory] = None) -> frozenset[str]:
    """The heads a generated block may use.

    The closed canonical list, PLUS whatever the runtime KB already grounds.
    The second half matters because the canonical list is a *curated* claim
    about which predicates rules consume, while `inventory.grounded_predicates()`
    is a measurement of what the loaded files really hold — a predicate with
    four hundred facts in it is not an invention, whoever wrote it.
    """
    if inventory is None:
        return CANONICAL_PREDICATES
    grounded = {p.name for p in inventory.grounded_predicates()}
    return frozenset(CANONICAL_PREDICATES | grounded)


# ================================================================
# 2. The schema exemplar shown to the model
# ================================================================
# What the model used to see was `existing_raw_content[:6000]` — an ALPHABETICAL
# 6 KB slice of ~290 KB of ontology. That is 2.1% of the KB, it begins at
# `cellage_calibration.metta`, and it contains not one example of the target
# schema. Replacing it with one verbatim example of each canonical form is the
# single highest-leverage change in this module.
#
# Every fact line below is copied verbatim out of a committed .metta file;
# `tests/test_ontology_expansion.py::test_every_exemplar_fact_is_verbatim_from_the_kb`
# re-derives that claim from the repository rather than trusting this comment.
SCHEMA_EXEMPLAR = """\
;; ---- 1. Publication provenance  (hallmarks_core.metta §1) ----
(: LopezOtinEtAl2023_Hallmarks Publication)
(PublicationTitle LopezOtinEtAl2023_Hallmarks "Hallmarks of aging: An expanding universe")
(PublicationYear LopezOtinEtAl2023_Hallmarks 2023)
(JournalName LopezOtinEtAl2023_Hallmarks "Cell")
(DOI LopezOtinEtAl2023_Hallmarks "10.1016/j.cell.2022.11.001")

;; ---- 2. Typing a new intervention node  (supplement_evidence.metta §1) ----
(Inheritance Omega3 Supplement)
(Inheritance Resveratrol Supplement)

;; ---- 3. A raw, source-faithful measurement row  (the DrugAge ETL shape that
;;         drugage_calibration.metta's `drugage-effect` lifts) ----
(InstanceOf DrugAgeRow_0 Experiment)
(UsesIntervention DrugAgeRow_0 Ethanol)
(UsesSpecies DrugAgeRow_0 Drosophila_mojavensis)
(AvgLifespanChangePercent DrugAgeRow_0 81.29)
(AvgLifespanSignificance DrugAgeRow_0 Unreported)
(ReportedIn DrugAgeRow_0 PMID_13369)

;; ---- 4. A review-level audit record
;;         (hallmarks_lopezotin2023_intervention_evidence.metta) ----
(: LopezOtin2023_Spermidine_Mouse HallmarkInterventionEvidence)
(EvidenceHallmark LopezOtin2023_Spermidine_Mouse DisabledMacroautophagy)
(EvidenceIntervention LopezOtin2023_Spermidine_Mouse Spermidine)
(EvidenceSpeciesModel LopezOtin2023_Spermidine_Mouse Mouse)
(EvidenceOutcomeText LopezOtin2023_Spermidine_Mouse "extended longevity, reduced cardiac aging and oxidative stress")
(SupportedByPublication LopezOtin2023_Spermidine_Mouse LopezOtinEtAl2023_Hallmarks)

;; ---- 5. The signed Effect link every inference rule consumes
;;         (mechanistic_bridges.metta). NOTE THE TRUTH VALUE: the confidence
;;         slot is an (evidence-confidence <Tier>) LOOKUP, left unevaluated, so
;;         epistemic_calibration.metta stays the one place the number lives.
;;         A two-float (stv 0.93 0.9) here is REJECTED. ----
(Effect CellularSenescence SASP Pos
   (stv 0.85 (evidence-confidence AnimalStudies_Replicated)))

;; ---- 6. Which hallmark an intervention targets  (hallmark_targeting.metta) ----
(TargetsHallmark Fisetin CellularSenescence)

;; ---- 7. Layer-4 modifier atoms the recommender reads
;;         (supplement_evidence.metta §3) ----
(EvidenceLevel Omega3 MultipleHumanTrials)
(SafetyProfile Omega3 GenerallyWellTolerated)
"""


# ================================================================
# 3. The validation gate
# ================================================================

@dataclass
class GateFinding:
    """One reason an entry was refused, plus the text that triggered it."""
    code: str
    message: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}

    def __str__(self) -> str:  # what a human reads in the API response
        return f"{self.code}: {self.message}" + (f" [{self.detail}]" if self.detail else "")


@dataclass
class EntryFacts:
    """Everything the gate and the normaliser learned from one entry's MeTTa."""
    heads: list[str] = field(default_factory=list)
    #: Heads of NESTED expressions, e.g. the `increases-life-span` inside
    #: `(Evaluation (increases-life-span Taurine Lifespan) (stv 0.93 0.9))`.
    #: Reporting only the outer head would name `Evaluation` as the problem and
    #: leave the three predicates the evaluation actually complained about
    #: unmentioned.
    nested_heads: list[str] = field(default_factory=list)
    #: Names given a function signature, `(: increases-life-span (-> …))`.
    declared_signatures: list[str] = field(default_factory=list)
    effect_expressions: list[str] = field(default_factory=list)
    literal_stvs: list[str] = field(default_factory=list)
    referenced_categories: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    has_reported_in: bool = False


def _top_level_expressions(metta: str) -> list[str]:
    """Whole S-expressions, comments stripped and line breaks healed.

    Deliberately `iter_top_level` (ontology/inventory.py) rather than a
    line-based split: the canonical `Effect` form is written over TWO lines, and
    every line-based check in the old pipeline silently missed it.
    """
    return [e for e in iter_top_level(metta) if e.startswith("(") and e.endswith(")")]


def read_entry(metta: str) -> EntryFacts:
    """Scan one entry's MeTTa without evaluating it."""
    facts = EntryFacts()
    for expr in _top_level_expressions(metta):
        parts = split_args(expr[1:-1].strip())
        if not parts:
            continue
        head, args = parts[0], parts[1:]
        facts.heads.append(head)

        if head == "=" and args and args[0].startswith("("):
            inner = split_args(args[0][1:-1].strip())
            if inner:
                facts.defines.append(inner[0])
        elif head == ":" and args:
            facts.subjects.append(args[0])
            if len(args) > 1 and " ".join(args[1:]).startswith("(->"):
                facts.declared_signatures.append(args[0])
        elif args:
            facts.subjects.append(args[0])

        for arg in args:
            if arg.startswith("(") and arg.endswith(")"):
                nested = split_args(arg[1:-1].strip())
                if nested and _NAME_RE.match(nested[0]):
                    facts.nested_heads.append(nested[0])

        if head == "Effect":
            facts.effect_expressions.append(expr)
        if head == "ReportedIn":
            facts.has_reported_in = True

        for match in _LITERAL_STV_RE.finditer(expr):
            facts.literal_stvs.append(match.group(0))
        for match in _CALIBRATED_STV_RE.finditer(expr):
            facts.referenced_categories.append(match.group(2))
    return facts


def _defines_a_number(metta: str, name: str) -> bool:
    """True when `metta` holds `(= (<name> …) <float>)` — a bare numeric knob."""
    pattern = rf"\(=\s*\(\s*{re.escape(name)}\b[^()]*\)\s*[-+]?[\d.]+(?:[eE][-+]?\d+)?\s*\)"
    return bool(re.search(pattern, metta))


def find_identifier(*texts: str) -> tuple[Optional[str], Optional[str]]:
    """Return `(pmid, doi)` — the first of each found in any of `texts`."""
    pmid: Optional[str] = None
    doi: Optional[str] = None
    for text in texts:
        if not text:
            continue
        if pmid is None:
            m = _PMID_RE.search(text)
            if m:
                pmid = m.group(1)
        if doi is None:
            m = _DOI_RE.search(text)
            if m:
                doi = m.group(1).rstrip(".")
    return pmid, doi


def check_entry(
    metta: str,
    *,
    identifier_text: str = "",
    categories: Optional[Iterable[str]] = None,
    inventory: Optional[RuntimeInventory] = None,
) -> list[GateFinding]:
    """Every reason this entry must not be appended to the knowledge base.

    An empty list means the entry passed. The four refusals the evaluation's
    block would have collected are, in order: `unknown_predicate` (three
    times — `increases-life-span`, `declines-with-aging`, `reduces`),
    `invented_truth_value` (`(stv 0.93 0.9)`), `invented_confidence_constant`
    (the 0.92 study-confidence definition) and `missing_identifier`.
    """
    allowed = allowed_predicates(inventory)
    legal_categories = tuple(categories) if categories is not None else evidence_categories()
    facts = read_entry(metta)
    findings: list[GateFinding] = []

    # -- 1. Vocabulary ---------------------------------------------------------
    # Outer heads, heads nested one level down (the `(Evaluation (pred …) …)`
    # shape the old prompt taught), and names given a function signature. All
    # three are ways to mint a predicate, and the evaluation's block used two.
    candidates = list(facts.heads) + list(facts.nested_heads) + list(facts.declared_signatures)
    for head in dict.fromkeys(candidates):
        if head in ("=", ":", "stv", "evidence-confidence", "->"):
            continue
        if head not in allowed:
            findings.append(GateFinding(
                "unknown_predicate",
                f"`{head}` is not a predicate the knowledge base consumes; "
                f"no rule reads it, so facts built on it would be inert.",
                head,
            ))

    # -- 2. Nobody redefines the calibration layer -----------------------------
    for name in dict.fromkeys(facts.defines):
        if name in CALIBRATION_FUNCTIONS:
            findings.append(GateFinding(
                "redefines_calibration",
                f"`{name}` belongs to the calibration layer "
                f"({_EPISTEMIC_FILE} / {_DRUGAGE_FILE}); an extracted block may "
                f"not redefine it.",
                name,
            ))
        elif _CONFIDENCE_NAME_RE.search(name) and _defines_a_number(metta, name):
            # A brand-new confidence knob under a brand-new name, e.g.
            # `(= (study-confidence NovelStudyType) 0.92)` — which is exactly
            # what the evaluation's block minted, so matching only the KNOWN
            # calibration names above would have waved it straight through.
            findings.append(GateFinding(
                "invented_confidence_constant",
                f"`{name}` mints a new confidence constant. Confidence comes "
                f"from (evidence-confidence <Tier>) in {_EPISTEMIC_FILE}, "
                f"never from a number in an extracted block.",
                name,
            ))

    # -- 3. No invented truth values ------------------------------------------
    # The Effect link is the shape that matters — it is what `infer`, `explain`
    # and `rank-interventions` read — but a literal two-float truth value
    # anywhere in an extracted block is the same mistake for the same reason:
    # the second slot is a number the calibration table owns.
    for stv in dict.fromkeys(facts.literal_stvs):
        on_effect = any(stv in expr for expr in facts.effect_expressions)
        where = "An Effect link" if on_effect else "An extracted fact"
        findings.append(GateFinding(
            "invented_truth_value",
            f"{where} may not carry a two-float truth value. The confidence "
            f"slot must be (evidence-confidence <Tier>), left unevaluated, so "
            f"the calibration table stays the authority.",
            stv,
        ))

    # -- 4. The tier has to be one the calibration table scores ----------------
    for category in dict.fromkeys(facts.referenced_categories):
        if legal_categories and category not in legal_categories:
            findings.append(GateFinding(
                "unknown_evidence_category",
                f"`{category}` is not one of the {len(legal_categories)} tiers "
                f"{_EPISTEMIC_FILE} declares; (evidence-confidence {category}) "
                f"has no value and the link's confidence would stay symbolic.",
                category,
            ))

    # -- 5. Provenance is mandatory -------------------------------------------
    pmid, doi = find_identifier(identifier_text, metta)
    if not pmid and not doi:
        findings.append(GateFinding(
            "missing_identifier",
            "No PMID and no DOI. A fact with no identifier cannot be traced "
            "back to the paper that justifies it, so it is refused rather than "
            "appended.",
        ))

    return findings


# ================================================================
# 4. Truth-value normalisation (the derive-or-label rule)
# ================================================================

@dataclass
class Normalisation:
    """The result of forcing one entry's truth value through the calibration layer."""
    metta: str
    #: Values the extractor PROPOSED rather than the pipeline deriving them.
    provisional_fields: list[str] = field(default_factory=list)
    #: Human-readable provenance/derivation lines for the generated comment.
    notes: list[str] = field(default_factory=list)

    @property
    def provisional(self) -> bool:
        return bool(self.provisional_fields)


def normalise_truth_values(
    metta: str,
    *,
    evidence_tier: Optional[str] = None,
    effect_size_pct: Optional[float] = None,
    halfsat: Optional[float] = None,
) -> Normalisation:
    """Rewrite every `Effect` truth value so it obeys the honesty contract.

    * the confidence slot becomes `(evidence-confidence <Tier>)`, unevaluated;
    * the strength becomes `|pct| / (|pct| + k)` when the paper reported a
      percent change, and stays the extractor's number — flagged provisional,
      labelled a curated prior in the comment — when it did not.

    The entry is returned unchanged when it holds no `Effect` link; a
    Publication record or a raw measurement row carries no truth value at all,
    which is exactly why those shapes are the ones worth emitting.
    """
    k = lifespan_halfsat() if halfsat is None else halfsat
    result = Normalisation(metta=metta)

    derived: Optional[float] = None
    if effect_size_pct is not None:
        try:
            derived = lifespan_strength(float(effect_size_pct), k)
        except (TypeError, ValueError):
            derived = None

    expressions = _top_level_expressions(metta)
    has_effect = any(e.startswith("(Effect") for e in expressions)
    if not has_effect:
        return result

    def _rebuild(match: re.Match, proposed_tier: Optional[str]) -> str:
        """One truth value, rebuilt from whatever this layer actually knows.

        With no tier from either side there is nothing honest to put in the
        confidence slot, so the expression is left exactly as it came in — for
        the gate to refuse — rather than quietly emitting
        `(evidence-confidence None)`.
        """
        tier = evidence_tier or proposed_tier
        if not tier:
            return match.group(0)
        strength = repr(derived) if derived is not None else match.group(1)
        return f"(stv {strength} (evidence-confidence {tier}))"

    # Both shapes are rewritten: a two-float `(stv s c)` (which the gate refuses
    # outright, but a caller may normalise before gating) and an already-
    # calibrated `(stv s (evidence-confidence T))` whose strength is still the
    # extractor's guess.
    rewritten = _LITERAL_STV_RE.sub(lambda m: _rebuild(m, None), metta)
    rewritten = _CALIBRATED_STV_RE.sub(lambda m: _rebuild(m, m.group(2)), rewritten)
    result.metta = rewritten

    tier_used = evidence_tier
    if tier_used is None:
        seen = read_entry(rewritten).referenced_categories
        tier_used = seen[0] if seen else None

    if tier_used:
        # The NUMBER is derived; the TIER the number is looked up by is still
        # the extractor's reading of the paper's study design, so it is exactly
        # the kind of proposed value a human reviewer has to check.
        result.provisional_fields.append("evidence_tier")
        result.notes.append(
            f"confidence : (evidence-confidence {tier_used}) — calibration-table "
            f"lookup, left unevaluated ({_EPISTEMIC_FILE} is the authority). "
            f"PROVISIONAL: the TIER is the extractor's reading of the study design."
        )
    if derived is not None:
        pct = abs(float(effect_size_pct))
        result.notes.append(
            f"strength   : {derived!r} = |{pct}| / (|{pct}| + {k}) — "
            f"{_DRUGAGE_FILE} §4's reproducible transform, DERIVED from the "
            f"reported effect size"
        )
    else:
        result.provisional_fields.append("strength")
        result.notes.append(
            "strength   : CURATED PRIOR proposed by the extractor, NOT derived — "
            "the paper reported no percent change this layer knows how to "
            "transform. PROVISIONAL: check the magnitude against the paper."
        )
    return result


# ================================================================
# 5. Would the generated block actually land?
# ================================================================

def unconsumed_predicates(
    metta_block: str, inventory: Optional[RuntimeInventory] = None
) -> list[str]:
    """Heads in `metta_block` that the runtime KB grounds nowhere else.

    Not an error — a genuinely new record type starts here — but it is the
    difference between "this block joins existing data" and "this block starts
    a table of one", and a reviewer deserves to be told which.
    """
    if inventory is None:
        return []
    out: list[str] = []
    for expr in _top_level_expressions(metta_block):
        parts = split_args(expr[1:-1].strip())
        if not parts:
            continue
        head = parts[0]
        if head in ("=", ":"):
            continue
        if not inventory.is_grounded(head) and head not in out:
            out.append(head)
    return out
