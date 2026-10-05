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
from core.patient_builder import (CHD_REACHING_CAUSES, BuiltPatient, build_patient, chd_observation_note,
                                  kb_effect_markers, prevalent_chd_note)
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


#: What a program reads when it asks about patients without naming one ("who has an
#: elevated CRP?" -> (match &self (MeasuredZ $p CRP $z) ...)).
_PATIENT_FACT_RE = re.compile(
    r"(?<![\w-])(?:MeasuredZ|MeasuredRaw|MeasuredUnit|PatientAge|PatientSex|PatientSmoking|"
    r"CurrentMedication|PatientCondition|PatientProfile)(?![\w-])|[\w-]+-patient(?![\w-])")


def reads_patients(metta_query: str) -> bool:
    """Names a patient, or reads patient facts: such a program runs in the patient stack."""
    return names_a_patient(metta_query) or bool(_PATIENT_FACT_RE.search(metta_query or ""))


def patient_atoms_for(metta_query: str, atoms: Optional[str]) -> Optional[str]:
    """A session patient's atoms enter only a program that reads patients — which runs
    in the patient stack. The full shared space sits at hyperon's head-symbol edge:
    holding a caller patient, it aborts as soon as a program enumerates patient facts
    ((match &self (MeasuredZ $p CRP $z) ...)). A program that reads no patient fact has
    no use for them, so it never gets them, and the full space never holds a caller."""
    return atoms if reads_patients(metta_query) else None


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


#: The shared-layer forms that personalise from a patient's elevated markers, and what each
#: returns for a patient with none the knowledge base can use (probed: tests/test_patient_stack.py).
_NO_WITNESS_RESULT = {
    "diagnose-patient": "the diagnosis returns ()",
    "recommend-supplements-patient": "every supplement tier is empty",
    "recommend-supplements": "every supplement tier is empty",
    "supplement-for-patient": "the single-supplement form returns nothing",
    "rank-interventions-for-patient": "the ranking is the population ranking, the same as for an "
                                      "unknown patient",
}
_PERSONALISED_FORM_RE = re.compile(
    r"\(\s*(" + "|".join(sorted(_NO_WITNESS_RESULT, key=len, reverse=True)) +
    r")\s+&self\s+([A-Za-z][A-Za-z0-9_]*)")


#: The forms that read AgeAccelGrim and nothing else of the patient's clocks.
_GRIM_RISK_RE = re.compile(
    r"\(\s*(predict-risk-patient|risk-decomposition-patient|project-risk-patient|risk-scenarios)\s+&self\s+"
    r"([A-Za-z][A-Za-z0-9_]*)"
    r"|\(\s*(predict-risk|risk-decomposition|project-risk|absolute-risk-at|absolute-risk|risk-ci|"
    r"risk-confidence)\s+&self\s+([A-Za-z][A-Za-z0-9_]*)\s+CoronaryHeartDisease\b")
_LINAGE_HAZARD_RE = re.compile(r"\(\s*linage-hazard-patient\s+&self\s+([A-Za-z][A-Za-z0-9_]*)")


def _no_grimage_risk_warnings(metta_query: str, patient: BuiltPatient) -> list[str]:
    """A heart-risk form for a patient with no GrimAge value. The CHD model reads AgeAccelGrim
    alone, so it returns nothing; if the same program also asks for the LinAge2 hazard (the
    pair the translator is told to emit for "my heart risk"), say that number is ALL-CAUSE
    mortality and is never a heart risk, and never combined with a GrimAge result."""
    if patient.has_grimage:
        return []
    forms = sorted({f or g for f, who, g, who2 in _GRIM_RISK_RE.findall(metta_query or "")
                    if patient.patient_id in (who, who2)})
    if not forms:
        return []
    note = (f"{patient.patient_id} has no AgeAccelGrim value, and the 10-year heart-disease (CHD) risk "
            f"model reads that clock alone, so there is no heart-specific risk for them: "
            f"{', '.join(forms)} returns nothing.")
    if patient.linage2 is not None and patient.patient_id in _LINAGE_HAZARD_RE.findall(metta_query):
        note += (" The LinAge2 hazard beside it is the ALL-CAUSE mortality multiplier for their "
                 "biological-age delta (outcome AllCauseMortality), not a heart risk, and it is never "
                 "multiplied or added to a GrimAge result. A GrimAge acceleration value is the only "
                 "way to get a heart risk.")
    return [note]


def _no_witness_warnings(metta_query: str, patient: BuiltPatient) -> list[str]:
    """A diagnosis / supplement / ranking form for a patient none of whose values is elevated
    and has a curated edge: () / empty tiers / the population ranking, which reads as "no cause"."""
    if patient.witnesses:
        return []
    results = []
    for form, who in _PERSONALISED_FORM_RE.findall(metta_query or ""):
        text = _NO_WITNESS_RESULT[form]
        if form == "diagnose-patient" and patient.prevalent_chd and any(
                c is None or (c & CHD_REACHING_CAUSES) for c in _diagnoses_of(metta_query, patient)):
            text = "the diagnosis answers from the reported heart disease alone (a prevalence item, not a measurement)"
        if who == patient.patient_id and text not in results:
            results.append(text)
    if not results:
        return []
    return [
        f"{patient.patient_id} has no elevated value the knowledge base can use (it has curated "
        f"edges for {', '.join(sorted(kb_effect_markers()))}; a glucose or a triglyceride counts only when typed as "
        f"fasting; a typed diagnosis, a low value and any other lab count for none of them): "
        f"{'; '.join(results)}. Read that as 'nothing to work from', not 'no cause' or 'no benefit'."
    ]


def _prevalent_chd_warnings(metta_query: str, patient: BuiltPatient) -> list[str]:
    """A 10-year CHD risk for a person who reports CHD, a heart attack or angina: the model
    estimates a first event, and returns the same number with or without that history."""
    if not (patient.prevalent_chd and patient.can_predict_risk):
        return []
    if not any(patient.patient_id in (who, who2) for _, who, _, who2 in _GRIM_RISK_RE.findall(metta_query or "")):
        return []
    return [prevalent_chd_note(patient.prevalent_chd)]


_DIAGNOSE_RE = re.compile(r"\(\s*diagnose-patient\s+&self\s+([A-Za-z][A-Za-z0-9_]*)(?:\s+\(([^()]*)\))?")


def _diagnoses_of(metta_query: str, patient: BuiltPatient) -> list[Optional[set]]:
    """One entry per diagnose-patient form naming this patient: None for the default cause list, else the set of
    causes the form lists."""
    return [None if causes is None or not causes.strip() else set(causes.split())
            for who, causes in ((m[0], m[1] or None) for m in _DIAGNOSE_RE.findall(metta_query or ""))
            if who == patient.patient_id]


def _chd_observation_warnings(metta_query: str, patient: BuiltPatient) -> list[str]:
    """A diagnosis for a person who reports CHD: the report is one of the observations it explains."""
    if not patient.prevalent_chd:
        return []
    calls = _diagnoses_of(metta_query, patient)
    if not calls:
        return []
    if all(c is not None and not (c & CHD_REACHING_CAUSES) for c in calls):
        return [f"Reported {', '.join(patient.prevalent_chd)} adds nothing to this diagnosis: none of the causes listed "
                f"reaches heart disease in the knowledge base ({', '.join(sorted(CHD_REACHING_CAUSES))} do)."]
    return [chd_observation_note(patient.prevalent_chd)]


def patient_form_warnings(metta_query: str, patient: Optional[BuiltPatient]) -> list[str]:
    """Say why a personalised form is about to come back empty or unpersonalised, before it
    does — the same note on /query, /metta/run and the chat (the chat never shows the
    builder's own notes, and an empty `()` reads as "no cause", which it is not).

    Four cases, each for a form NAMING this patient: a heart-risk form for a patient with no
    GrimAge value; a heart-risk form for a patient who reports CHD (a first-event model); a
    diagnosis for a patient who reports CHD (it explains that report as an observation); and
    a diagnosis / supplement / ranking form for a patient none of whose values is elevated and
    has a curated edge (BuiltPatient.witnesses)."""
    if patient is None:
        return []
    return (_no_grimage_risk_warnings(metta_query, patient)
            + _prevalent_chd_warnings(metta_query, patient)
            + _chd_observation_warnings(metta_query, patient)
            + _no_witness_warnings(metta_query, patient))


def linage2_prompt_hint(patient: BuiltPatient) -> str:
    return (
        "This patient carries a LinAge2 clinical-clock result (LinAgeDelta and one "
        "LinAgeContribution per lab). Questions about it — biological age, which labs "
        "add years, mortality hazard or risk, what would remove years — map to the "
        "dedicated LinAge2 forms, which take this patient id:\n"
        f"  (linage-decomposition-patient &self {patient.patient_id})\n"
        f"  (linage-drivers-patient &self {patient.patient_id})   ; drivers of BIOLOGICAL AGE, not of abnormal labs\n"
        f"  (linage-hazard-patient &self {patient.patient_id})\n"
        f"  (linage-risk-patient &self {patient.patient_id})\n"
        f"  (linage-counterfactual-patient &self {patient.patient_id} <Lever>)\n"
        f"  (linage-project-risk-patient &self {patient.patient_id} <Lever>)\n"
        f"  (linage-scenarios-patient &self {patient.patient_id})\n"
        "A lab, organ or condition with no lever and no curated cause (kidney function, blood "
        "pressure, cholesterol, creatinine) is the decomposition plus an "
        "`explanation` saying so — never a lever token made from its name (rule 16). RDW and "
        "the albumin deficit (LowSerumAlbumin) ARE levers, by those names.\n"
        "They run in their own space; a query may combine them with forms of other "
        "layers, each as its own top-level expression on its own line (never nested "
        "inside another form) — each part runs where it can and the answers come back "
        "in order. The GrimAge forms never see the LinAge2 result: the CHD-risk forms and "
        "decompose-grimage need AgeAccelGrim, counterfactual-patient the DNAm components.\n"
    )


def no_grimage_prompt_hint(patient: BuiltPatient) -> str:
    """For a patient with no GrimAge value, what "my heart risk" maps to. The static prompt
    cannot say it: whether the patient has AgeAccelGrim is a fact about this request."""
    if patient.has_grimage:
        return ""
    pid = patient.patient_id
    if patient.linage2 is None:          # a bare LinAgeAccel marker has no LinAgeDelta: no hazard to pair
        return (f"This patient has NO AgeAccelGrim value, so there is no heart-specific (10-year "
                f"CHD) risk model for them: a heart-risk question is (predict-risk-patient &self "
                f"{pid}), which returns nothing, and the answer says why.\n")
    return (f"This patient has NO AgeAccelGrim value, so there is no heart-specific (10-year CHD) "
            f"risk model for them. For a question about their heart / cardiovascular / CHD risk "
            f"emit these two forms, each on its own line, and say in `explanation` that the second "
            f"is the ALL-CAUSE LinAge2 mortality hazard, not a heart risk; never multiply or add "
            f"the two clocks:\n  (predict-risk-patient &self {pid})\n"
            f"  (linage-hazard-patient &self {pid})\n"
            f"A what-if about a lever (\"if my inflammation were normal\", \"if I quit smoking\") is the LinAge2 form "
            f"(linage-counterfactual-patient &self {pid} <Lever>): counterfactual-patient reads the DNAm "
            f"components, not a GrimAge value, and returns a zero result for a patient without them.\n")


def chd_observation_prompt_hint(patient: BuiltPatient) -> str:
    """For a patient who reports CHD: what (diagnose-patient &self <P>) then explains besides their labs."""
    if not patient.prevalent_chd:
        return ""
    return (f"This patient reports {', '.join(patient.prevalent_chd)}. (diagnose-patient &self {patient.patient_id}) reads "
            f"that as one more observation to explain, labelled in the answer as a prevalence item (\"ever told\"), "
            f"not a measured value; say so in `explanation`. The supplement plan and the intervention ranking do "
            f"not read it.\n")


def prevalent_chd_prompt_hint(patient: BuiltPatient) -> str:
    """For a patient who reports CHD and has a GrimAge value (so the CHD model answers)."""
    if not (patient.prevalent_chd and patient.can_predict_risk):
        return ""
    return (f"This patient reports {', '.join(patient.prevalent_chd)}. The 10-year CHD risk model "
            f"(predict-risk-patient &self {patient.patient_id}) estimates a FIRST coronary event "
            f"in someone without CHD, so its number does not apply to them: say so when you "
            f"present it (the response carries the same note).\n")


def medication_prompt_hint(patient: BuiltPatient) -> str:
    """The medication is NOT in the atoms listed above (they also feed the LinAge2 space, where it
    has no room); the supplement forms read it from the shared space, so the translator is told."""
    if not patient.medications:
        return ""
    return (f"This patient currently takes {', '.join(patient.medications)} (recorded as "
            f"(CurrentMedication {patient.patient_id} <drug>) in the space the supplement forms read; it "
            f"is not in the atoms above). A supplement plan, or (supplement-for-patient &self "
            f"{patient.patient_id} <Supplement>), flags an interaction with it.\n")


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
        f"What causes or drives this patient's abnormal LABS (not their biological age) "
        f"is (diagnose-patient &self {patient.patient_id}) over the knowledge base's "
        f"default candidate causes — never a hand-typed hallmark list (rule 17).\n"
        + no_grimage_prompt_hint(patient)
        + prevalent_chd_prompt_hint(patient)
        + chd_observation_prompt_hint(patient)
        + medication_prompt_hint(patient)
        + (linage2_prompt_hint(patient) if patient.linage2 is not None else "")
    )
