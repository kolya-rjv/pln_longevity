"""Refuse to quote a lever's benefit to a patient who cannot pull it.

`PatientSmoking` was the evaluation's §9 finding: the status sat in the profile
and nothing read it. `lifestyle_evidence.metta` wired the exposure into the
clock, but the lever it added keys on the DNAmPACKYRS z-score alone — so a
NEVER SMOKER with an elevated pack-years surrogate is handed a quantified
benefit from quitting, byte-identical to a smoker's. That is an ordinary data
state, not a pathological input: DNAmPACKYRS is an elastic-net PREDICTOR of
pack-years with real error in both directions, and it responds to second-hand
exposure.

The precondition is declared in the knowledge base as a fact —
`(LeverRequiresSmoking SmokingCessation CurrentSmoker)` — and read here rather
than by a MeTTa rule. That split is forced, and the number behind it is
measured: the curated runtime KB tolerates exactly FOUR more trivial rule
definitions before hyperon 0.2.10 aborts the interpreter on an unrelated
`predict-risk-patient` query, and the lever-edge composition this same change
adds to `pln_counterfactual.metta` §4 spends all four. Ground facts are cheap
by comparison (the same probe takes 8,191 extra atoms and aborts at 8,192), so
the declaration stays in the KB where it belongs and only the rule moves out.

What this buys, and what it does not: `/query` and `/metta/run` warn, in the
response, that the delta they are returning does not apply to the patient named.
A caller talking to hyperon directly still gets the ungated number. The gap is
recorded in `pln_counterfactual.metta` §3b and closes when the curated layers
stop sharing one space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from config import ONTOLOGY_DIR

#: `(LeverRequiresSmoking <Lever> <SmokingStatus>)`
_REQUIRES_RE = re.compile(r"\(LeverRequiresSmoking\s+(\S+)\s+(\S+)\s*\)")
#: `(PatientSmoking <Patient> <SmokingStatus>)`
_PATIENT_SMOKING_RE = re.compile(r"\(PatientSmoking\s+(\S+)\s+(\S+)\s*\)")


@dataclass(frozen=True)
class LeverPrecondition:
    lever: str
    predicate: str
    required: str


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";;")[0] for line in text.splitlines())


@lru_cache(maxsize=1)
def lever_preconditions() -> tuple[LeverPrecondition, ...]:
    """Every declared lever precondition, read from the curated .metta files."""
    found: list[LeverPrecondition] = []
    for path in sorted(Path(ONTOLOGY_DIR).glob("*.metta")):
        try:
            text = _strip_comments(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for lever, required in _REQUIRES_RE.findall(text):
            found.append(LeverPrecondition(lever, "PatientSmoking", required))
    return tuple(dict.fromkeys(found))


@lru_cache(maxsize=1)
def curated_smoking_status() -> dict[str, str]:
    """`PatientSmoking` for every patient the curated KB declares."""
    statuses: dict[str, str] = {}
    for path in sorted(Path(ONTOLOGY_DIR).glob("*.metta")):
        try:
            text = _strip_comments(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for patient, status in _PATIENT_SMOKING_RE.findall(text):
            statuses[patient] = status
    return statuses


def smoking_status_of(patient: str, extra_atoms: Optional[str] = None) -> Optional[str]:
    """The patient's status, preferring atoms supplied with THIS request."""
    if extra_atoms:
        for name, status in _PATIENT_SMOKING_RE.findall(_strip_comments(extra_atoms)):
            if name == patient:
                return status
    return curated_smoking_status().get(patient)


def _named(pattern: str, text: str, names: Iterable[str]) -> list[str]:
    return [n for n in names if re.search(pattern % re.escape(n), text)]


def lever_warnings(
    metta_query: str,
    *,
    extra_atoms: Optional[str] = None,
    known_patients: Iterable[str] = (),
) -> list[str]:
    """Warn for each (lever, patient) pair in the query whose precondition fails.

    Deliberately conservative: a warning is emitted only when the query names
    BOTH a lever that declares a precondition AND a patient whose status is
    known and does not meet it. An unknown status produces no warning — the KB
    does not know the patient smokes, and guessing in either direction would be
    the invention this layer exists to prevent.
    """
    preconditions = lever_preconditions()
    if not preconditions:
        return []

    candidates = set(known_patients) | set(curated_smoking_status())
    if extra_atoms:
        candidates.update(
            name for name, _ in _PATIENT_SMOKING_RE.findall(_strip_comments(extra_atoms))
        )
    patients = _named(r"\b%s\b", metta_query, sorted(candidates))
    if not patients:
        return []

    warnings: list[str] = []
    for rule in preconditions:
        if not re.search(rf"\b{re.escape(rule.lever)}\b", metta_query):
            continue
        for patient in patients:
            status = smoking_status_of(patient, extra_atoms)
            if status is None or status == rule.required:
                continue
            warnings.append(
                f"{rule.lever} presupposes ({rule.predicate} {patient} "
                f"{rule.required}); this patient is {status}. Any non-zero "
                f"expected-delta below is the arithmetic of the exposure "
                f"marker, NOT a benefit this patient can obtain — the "
                f"knowledge base declares "
                f"(LeverRequiresSmoking {rule.lever} {rule.required}) and the "
                f"engine does not yet read it (pln_counterfactual.metta §3b)."
            )
    return warnings
