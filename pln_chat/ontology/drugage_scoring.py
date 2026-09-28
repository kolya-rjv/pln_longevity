"""Score every DrugAge row in Python, using the MeTTa layer's own constants.

Why a second implementation exists
----------------------------------
"Which drugs extend lifespan in mice with the strongest evidence?" is the first
question anyone asks a longevity engine, and the API could not answer it. The
2026-09-18 evaluation watched the translator invent a four-compound pool and
rank only those, and watched a generic `match` over the DrugAge predicates come
back empty because those rows are deliberately excluded from the runtime space.
There is no top-N.

It cannot be done in MeTTa. Ranking N compounds costs N MeTTa calls at ~70 ms
each (2 minutes for the full 1,043-compound build), and — worse — hyperon 0.2.10
ABORTS the process on a variable-slot `match` once a space passes ~250-300 rows,
which is why `MAX_ROWS` is 150. Scoring the whole build inside the engine is not
slow, it is impossible.

So the scoring arithmetic is reimplemented here, over the parsed rows. That is a
duplication, and duplication of calibration constants is exactly what
`drugage_calibration.metta` §1 warns about ("every knob is in §1 so tuning stays
local"). Two things keep the copy honest:

1. **Nothing is hard-coded.** Every constant — the half-saturation, the
   significance gates, the evidence-tier confidences, the clade map, the chain
   discount — is PARSED out of the .metta files at load time. Tuning
   `(= (lifespan-halfsat) 20.0)` changes this scorer too.
2. **A test asserts equality with the engine.** `tests/test_drugage_discovery.py`
   scores a sample of rows both ways and requires a bit-for-bit match, so drift
   fails the suite rather than the ranking.

The arithmetic, for the record (drugage_calibration.metta §4-§8):

    strength   = |pct| / (|pct| + halfsat)
    tier       = ITP ? conf(ITP_Positive|ITP_Negative) : conf(clade_category(species))
    gate       = ITP ? 1.0 : sig_gate(significance)
    confidence = tier * gate * chain_discount            # the Lifespan->Mortality hop
    sign       = pct < 0 ? "Pos" (harmful) : "Neg" (protective)
    score      = sign == "Neg" ? +strength*confidence : -strength*confidence
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from config import ONTOLOGY_DIR
from ontology.drugage_selector import DrugAgeRow

# ── Reading the knobs out of the .metta sources ──────────────────────────────

_RE_HALFSAT = re.compile(r"\(=\s*\(lifespan-halfsat\)\s*([\d.]+)\s*\)")
_RE_SIG_GATE = re.compile(r"\(=\s*\(sig-gate\s+(\w+)\)\s*([\d.]+)\s*\)")
_RE_CLADE = re.compile(r"\(=\s*\(clade-category\s+(\w+)\)\s*(\w+)\s*\)")
_RE_EVIDENCE_CONF = re.compile(r"\(=\s*\(evidence-confidence\s+(\w+)\)\s*([\d.]+)\s*\)")
_RE_CHAIN = re.compile(r"\(=\s*\(chain-discount\)\s*([\d.]+)\s*\)")
_RE_TAXON = re.compile(r"^\(Inheritance\s+(\S+)\s+(\S+)\s*\)", re.MULTILINE)


@dataclass(frozen=True)
class ScoringKnobs:
    """Every constant the scorer uses, read from the MeTTa layer."""
    halfsat: float
    sig_gates: dict[str, float]
    clade_categories: dict[str, str]
    evidence_confidence: dict[str, float]
    chain_discount: float
    species_clade: dict[str, str]
    #: the default tier for a species with no taxonomy entry
    untaxonomised_category: str = "InVitro"

    def as_dict(self) -> dict:
        return {
            "lifespan_halfsat": self.halfsat,
            "significance_gates": dict(self.sig_gates),
            "clade_categories": dict(self.clade_categories),
            "evidence_confidence": dict(self.evidence_confidence),
            "chain_discount": self.chain_discount,
            "untaxonomised_species_category": self.untaxonomised_category,
            "species_with_taxonomy": len(self.species_clade),
        }


def _read(name: str) -> str:
    try:
        return (ONTOLOGY_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return ""


_KNOBS: Optional[ScoringKnobs] = None


def load_knobs(force: bool = False) -> ScoringKnobs:
    """Parse the calibration constants; cached for the process."""
    global _KNOBS
    if _KNOBS is not None and not force:
        return _KNOBS

    calib = _read("drugage_calibration.metta")
    epistemic = _read("epistemic_calibration.metta")
    deduction = _read("pln_deduction.metta")
    taxonomy = _read("species_taxonomy.metta")

    halfsat_match = _RE_HALFSAT.search(calib)
    chain_match = _RE_CHAIN.search(deduction)
    _KNOBS = ScoringKnobs(
        halfsat=float(halfsat_match.group(1)) if halfsat_match else 20.0,
        sig_gates={k: float(v) for k, v in _RE_SIG_GATE.findall(calib)},
        clade_categories=dict(_RE_CLADE.findall(calib)),
        evidence_confidence={
            k: float(v) for k, v in _RE_EVIDENCE_CONF.findall(epistemic)
        },
        chain_discount=float(chain_match.group(1)) if chain_match else 0.9,
        species_clade=dict(_RE_TAXON.findall(taxonomy)),
    )
    return _KNOBS


# ── Scoring ──────────────────────────────────────────────────────────────────

@dataclass
class RowScore:
    """One row's calibrated, signed effect on mortality."""
    row: DrugAgeRow
    strength: float
    confidence: float
    sign: str          # "Neg" protective | "Pos" harmful
    score: float
    tier_category: str

    @property
    def protective(self) -> bool:
        return self.sign == "Neg"

    @property
    def direction(self) -> str:
        """'protective' | 'harmful' | 'no_effect' — the sign in words.

        `sign` is the MeTTa Effect convention and has no zero: a row reporting
        0.0% lifespan change is `Neg` because it is not negative. Reading that
        straight out as "protective" put a contradiction inside one response —
        `direction: "protective"` next to a `score` of exactly 0.0 and a
        `semantics.zero_score` note saying that a 0.0 is a REPORTED NULL. The
        most consequential rows in the build are exactly these: the ITP nulls
        for metformin and resveratrol, at the highest confidence the KB gives.

        A row the study itself calls NOT significant reports no effect either,
        whatever the sign of its point estimate. Reading the sign alone made
        fisetin `harmful` at confidence 0.81 off a non-significant -1% ITP row —
        a direction the experiment explicitly declined to claim. For an ITP row
        the label is the whole story: the calibration layer scores a well-run
        NULL at the same confidence as a well-run positive, deliberately
        (drugage_calibration.metta §5), so `direction` was the only field in
        which the two differed — and it differed by reporting the sign of the
        noise. Outside the ITP the `sig-gate` still discounts the confidence;
        either way this changes the LABEL, never the arithmetic.

        `Unreported` is left alone on purpose: a study that never stated
        significance is an unknown, not a null, and the 0.6 gate already prices
        that in.
        """
        if self.strength == 0.0:
            return "no_effect"
        if (self.row.significance or "") == "NotSignificant":
            return "no_effect"
        return "protective" if self.protective else "harmful"


def score_row(row: DrugAgeRow, knobs: Optional[ScoringKnobs] = None) -> Optional[RowScore]:
    """Score ONE row, or None when the row reports no lifespan change.

    A row with no `AvgLifespanChangePercent` cannot be lifted into an Effect
    link by `drugage-effect`, so it has no score — the same silence the MeTTa
    path produces, made explicit.
    """
    if row.avg_change is None:
        return None
    k = knobs or load_knobs()

    magnitude = abs(row.avg_change)
    strength = magnitude / (magnitude + k.halfsat) if (magnitude + k.halfsat) else 0.0

    if row.is_itp:
        category = "ITP_Positive" if row.significance == "Significant" else "ITP_Negative"
        tier = k.evidence_confidence.get(category, 0.9)
        gate = 1.0
    else:
        clade = k.species_clade.get(row.species or "")
        category = (
            k.clade_categories.get(clade, k.untaxonomised_category)
            if clade else k.untaxonomised_category
        )
        tier = k.evidence_confidence.get(category, 0.35)
        gate = k.sig_gates.get(row.significance or "Unreported", 0.6)

    confidence = tier * gate * k.chain_discount
    # pct > 0 raises Lifespan (Pos), chained through the curated
    # (Effect Lifespan Mortality Neg) adapter => net Neg = protective.
    sign = "Pos" if row.avg_change < 0 else "Neg"
    score = strength * confidence * (1.0 if sign == "Neg" else -1.0)
    return RowScore(
        row=row,
        strength=strength,
        confidence=confidence,
        sign=sign,
        score=score,
        tier_category=category,
    )


def score_rows(
    rows: Iterable[DrugAgeRow],
    *,
    knobs: Optional[ScoringKnobs] = None,
) -> list[RowScore]:
    k = knobs or load_knobs()
    out: list[RowScore] = []
    for row in rows:
        scored = score_row(row, k)
        if scored is not None:
            out.append(scored)
    return out
