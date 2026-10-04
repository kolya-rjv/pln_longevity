"""What a caller-supplied patient adds to a request — shared by the API and the UI.

The HTTP API (`/query`, `/metta/run`) and the Gradio chat (with the "My Patient"
tab) must tell the translator about the patient the same way and validate against
the space the query will actually run in the same way, or the two surfaces answer
the same question differently (tests/test_ui_api_parity.py).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional

from config import ONTOLOGY_DIR
from core.patient_builder import BuiltPatient, build_patient
from ontology.inventory import merged_inventory
from ontology.loader import parse_metta_text
from ontology.registry import OntologyRegistry


_SD_TO_YEARS_RE = re.compile(r"\(=\s*\(grimaccel-sd-to-years\)\s*([\d.]+)\s*\)")
_ELEVATED_Z_RE = re.compile(r"\(=\s*\(elevated-z-threshold\)\s*([\d.]+)\s*\)")


@lru_cache(maxsize=1)
def patient_knobs() -> tuple[float, float]:
    """`grimaccel-sd-to-years` and `elevated-z-threshold`, read off the KB.

    Both are documented tunables of the MeTTa layer. Reading them rather than
    copying them keeps a caller's years->z conversion and the Elevated/Low
    labels in lockstep with the engine that will consume them.
    """
    sd_to_years, elevated = 4.2, 1.0
    try:
        m = _SD_TO_YEARS_RE.search(
            (ONTOLOGY_DIR / "pln_risk_prediction.metta").read_text(encoding="utf-8"))
        if m:
            sd_to_years = float(m.group(1))
    except OSError:
        pass
    try:
        m = _ELEVATED_Z_RE.search(
            (ONTOLOGY_DIR / "patient_profile.metta").read_text(encoding="utf-8"))
        if m:
            elevated = float(m.group(1))
    except OSError:
        pass
    return sd_to_years, elevated


def build_caller_patient(payload: dict, existing_ids: Iterable[str]) -> BuiltPatient:
    """`build_patient` with the KB's own knobs — what /query, /patients/preview, the
    UI's My Patient tab and its chat all use, so a patient is the same patient on
    every surface. Raises PatientSpecError."""
    sd_to_years, elevated = patient_knobs()
    return build_patient(payload, existing_ids=set(existing_ids), sd_to_years=sd_to_years,
                         elevated_threshold=elevated)


#: A built-in patient (Patient001 …) or a caller-supplied one (always `Caller_…`).
_PATIENT_ID_RE = re.compile(r"(?<![A-Za-z0-9_])(?:Patient\d+|Caller_[A-Za-z][A-Za-z0-9_]*)(?![A-Za-z0-9_])")


def names_a_patient(metta_query: str) -> bool:
    """Does this program ask about a patient? Then it runs in the patient stack
    (core.pln_runner.patient_stack), where the patient forms do not abort."""
    return bool(_PATIENT_ID_RE.search(metta_query or ""))


def with_injected(registry: OntologyRegistry, inventory, injected: Optional[str],
                  source_name: str = "<api-extra-atoms>"):
    """The (registry, inventory) of a space that holds `injected` on top of its KB.

    Without this a caller's own patient id (Caller_W58) — or any symbol its atoms
    define — is "not found in loaded ontology". `parse_metta_text` harvests only a
    few shapes, so an inventory over the injected text is unioned in as well. The
    caller's registry is copied, never mutated (it may be cached).
    """
    if not injected:
        return registry, inventory
    merged = OntologyRegistry()
    merged.merge(registry)
    merged.merge(parse_metta_text(injected, source_name=source_name))
    return merged, merged_inventory(inventory, injected)


def validation_text(injected: Optional[str], metta_query: str) -> str:
    """What is validated: the injected atoms and the query, as the space will see them."""
    return "\n".join(part for part in (injected, metta_query) if part)


def linage2_prompt_hint(patient: BuiltPatient) -> str:
    return (
        "This patient carries a LinAge2 clinical-clock result (LinAgeDelta and one "
        "LinAgeContribution per lab). Questions about it — biological age, which labs "
        "add years, mortality hazard or risk, what would remove years — map to the "
        "dedicated LinAge2 forms, which take this patient id:\n"
        f"  (linage-decomposition-patient &self {patient.patient_id})\n"
        f"  (linage-drivers-patient &self {patient.patient_id})\n"
        f"  (linage-hazard-patient &self {patient.patient_id})\n"
        f"  (linage-risk-patient &self {patient.patient_id})\n"
        f"  (linage-counterfactual-patient &self {patient.patient_id} <Lever>)\n"
        f"  (linage-project-risk-patient &self {patient.patient_id} <Lever>)\n"
        f"  (linage-scenarios-patient &self {patient.patient_id})\n"
        "They run in their own space; a query may combine them with forms of other "
        "layers, each as its own top-level expression on its own line (never nested "
        "inside another form) — each part runs where it can and the answers come back "
        "in order. The GrimAge forms (predict-risk-patient, "
        "decompose-grimage) read AgeAccelGrim and do NOT see the LinAge2 result.\n"
    )


def patient_prompt_section(patient: BuiltPatient) -> str:
    """Appended AFTER the static system prompt (so the static prefix stays cacheable).

    The translator has to know the id exists, or it answers "I cannot compute a
    personalized risk from the current KB" — which is what the evaluation saw for a
    45-year-old woman with LDL 130 and CRP 4.
    """
    return (
        "\n\n--- THIS REQUEST'S PATIENT ---\n"
        f"The caller submitted a patient, loaded for this request only:\n"
        f"{patient.atoms}\n"
        f"Treat `{patient.patient_id}` as a valid <Patient> for every "
        f"dedicated patient form. When the question says 'me', 'my', 'this "
        f"patient' or gives no id, it means {patient.patient_id}.\n"
        + (linage2_prompt_hint(patient) if patient.has_linage2 else "")
    )
