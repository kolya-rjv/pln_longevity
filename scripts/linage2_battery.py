"""The LinAge2 query battery: specification, runner, and PDF.

    python scripts/linage2_battery.py --date 2026-10-04

Every entry carries the natural-language question as a person would type it in the
"My Patient" -> "PLN Query" flow, the query form the translator should emit, and the
behaviour that counts as a pass — the shape of the PLN Longevity Query Battery. What
this battery adds is a fourth line, OBSERVED: each entry that has a query form is
RUN, through `POST /query` with only the LLM translator stubbed, in worker processes
exactly as the server runs it, against
patients built from the tab's own example texts, and checked. Entries about the tab
itself are checked against the tab's handlers. Entries whose correct answer is a
direct reply with nothing executed (∅) are graded live, because grading them needs
the LLM translator.

Writes docs/linage2_battery/results.json, battery.html and LinAge2QueryBattery.pdf.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pln_chat"))
OUT = REPO / "docs" / "linage2_battery"

from core.linage2_router import (  # noqa: E402
    counterfactual_to_dict,
    decomposition_to_dict,
    hazard_to_dict,
    parse_sexpr,
)
from core.patient_text import EXAMPLES, read_patient_text  # noqa: E402

# ═══════════════════════════ the patients ══════════════════════════════════════
# Built exactly as the tab builds them: read_patient_text -> to_patient("Me").

SMOKER = EXAMPLES["58-year-old smoker"]
PATIENT_TEXT = {
    "P1": SMOKER,
    "P2": EXAMPLES["healthy 45-year-old woman"],
    "P3": EXAMPLES["six labs only"],
    "P4": SMOKER + "\nGrimAge acceleration +4.5 years",
    # the smoker's labs with cotinine MEASURED but no smoking status stated
    "P5": SMOKER.replace("58 year old male, current smoker", "58 year old male\ncotinine 250 ng/mL"),
}
PATIENT_LABEL = {
    "P1": "the 58-year-old smoker example",
    "P2": "the healthy 45-year-old woman example",
    "P3": "the six-labs example (66, male, former smoker)",
    "P4": "P1 plus a line 'GrimAge acceleration +4.5 years'",
    "P5": "P1 with 'current smoker' replaced by 'cotinine 250 ng/mL'",
}


def payload(pid: str) -> dict:
    return read_patient_text(PATIENT_TEXT[pid]).to_patient("Me")[0]


# ═══════════════════════════ reading answers ═══════════════════════════════════

def atoms(resp: dict) -> list[str]:
    return [r["atom"] for r in resp.get("pln_results") or []]


def first(resp: dict, head: str) -> Optional[list]:
    for a in atoms(resp):
        sx = parse_sexpr(a)
        while isinstance(sx, list) and sx and isinstance(sx[0], list) and len(sx) == 1:
            sx = sx[0]
        if isinstance(sx, list) and sx and sx[0] == head:
            return sx
        if isinstance(sx, list) and sx and isinstance(sx[0], list) and sx[0] and sx[0][0] == head:
            return sx[0]
    return None


def decomposition(resp: dict) -> Optional[dict]:
    sx = first(resp, "LinAgeDecomposition")
    return decomposition_to_dict(sx) if sx else None


def contribution(d: dict, symbol: str) -> Optional[dict]:
    for c in d["measured"] + d["imputed"]:
        if c["symbol"] == symbol:
            return c
    return None


def counterfactual(resp: dict) -> Optional[dict]:
    sx = first(resp, "LinAgeCounterfactual")
    return counterfactual_to_dict(sx) if sx else None


def hazard(resp: dict) -> Optional[dict]:
    sx = first(resp, "LinAgeHazard")
    return hazard_to_dict(sx) if sx else None


Check = Callable[[dict], tuple[bool, str]]


# ═══════════════════════════ the entries ═══════════════════════════════════════

@dataclass
class Entry:
    section: str
    query: str
    emits: Optional[str]                  # None = ∅ (a direct answer, nothing executed)
    passes: str
    where: str = "In PLN Query"
    patient: Optional[str] = "P1"
    run: Optional[str] = None             # defaults to `emits`
    check: Optional[Check] = None         # for a run: response -> (ok, observed)
    tab_check: Optional[Callable[[], tuple[bool, str]]] = None
    n: int = 0
    result: dict = field(default_factory=dict)


def _read(text: str):
    return read_patient_text(text)


def _reading(text: str):
    p = _read(text)
    return p, (p.readings[0] if p.readings else None)


def t_albumin_ok():
    import patient_tab
    p, r = _reading("albumin 4.1 g/dL")
    row = "| Albumin | `albumin 4.1 g/dL` | 41 g/L | ✓ | 45 g/L |"
    shown = row in patient_tab.on_read("58 year old male\nalbumin 4.1 g/dL")
    return (r.status == "ok" and abs(r.value - 41) < 1e-9 and shown,
            f"{r.typed} → {r.value:g} g/L, {r.status}; typical for a 58-year-old man: 45 g/L"
            + ("" if shown else " — NOT shown"))


def t_albumin_refused():
    p, r = _reading("albumin 4.2 g/L")
    return (r.status == "out of range" and "g/dL" in r.note, f"{r.status}: {r.note}")


def t_albumin_assumed():
    p, r = _reading("albumin 4.2")
    return (r.status == "unit assumed" and abs(r.value - 42) < 1e-9, f"{r.status}: {r.note}")


def t_crp_needs_unit():
    p, r = _reading("CRP 3.1")
    return (r.status == "needs a unit" and not p.ok, f"{r.status}: {r.note}")


def t_crp_two_units():
    p = _read("50 year old man\nCRP 3.1 mg/L")
    r = p.readings[0]
    kb = p.kb_markers()["CRP"]
    ok = abs(r.value - 0.31) < 1e-9 and kb == {"value": 3.1, "unit": "mg/L"}
    return ok, f"LinAge2 LBXCRP = {r.value:g} mg/dL; knowledge-base CRP marker = {kb['value']:g} {kb['unit']}"


def t_hba1c_ifcc():
    p, r = _reading("HbA1c 48 mmol/mol")
    return (r.status == "ok" and abs(r.value - (48 * 0.09148 + 2.152)) < 1e-9, f"{r.typed} → {r.value:.2f} %")


def t_differential():
    a = _read("lymphocytes 30").readings[0]
    b = _read("lymphocytes 1.9").readings[0]
    return (a.code == "LBXLYPCT" and b.code == "LBDLYMNO",
            f"'lymphocytes 30' → {a.label}; 'lymphocytes 1.9' → {b.label}")


def t_duplicate():
    p = _read("albumin 4.1 g/dL\nalbumin 38 g/L")
    return (p.readings[1].status == "duplicate" and not p.ok, f"second reading: {p.readings[1].status}")


def t_not_understood():
    p = _read("my mood is great\nfrobnicate 3")
    return (not p.readings and len(p.not_understood) == 2, f"not understood: {p.not_understood}")


def t_smoker():
    p = _read("58 year old male, current smoker")
    return ((p.age, p.sex, p.smoking, p.cotinine_level) == (58, "Male", "CurrentSmoker", 3),
            f"age {p.age:g}, {p.sex}, {p.smoking}, cotinine level {p.cotinine_level}")


def t_cotinine():
    p = _read("cotinine 150 ng/mL")
    return (p.cotinine_level == 2 and p.smoking is None, f"level {p.cotinine_level}; smoking status {p.smoking}")


def t_no_age():
    import patient_tab
    first_state = patient_tab.on_build(SMOKER, None)[5]
    out = patient_tab.on_build("albumin 4.1 g/dL", first_state)
    return (out[5] is first_state and "Not built" in out[1], "build refused; previous patient kept")


def t_later_years():
    p = _read("58 year old male\nquit smoking 20 years ago")
    return ((p.age, p.smoking) == (58, "FormerSmoker") and not p.not_understood,
            f"age {p.age:g}, {p.smoking}, nothing left over")


def t_someone_else():
    p = _read("45 year old female, never smoked\nmy husband smokes")
    return ((p.sex, p.smoking) == ("Female", "NeverSmoker") and p.set_aside == ["my husband smokes"],
            f"{p.sex}, {p.smoking}; set aside: {p.set_aside}")


def t_anaemic():
    p, r = _reading("hemoglobin 9.5")
    return (r.status == "needs a unit", f"{r.status}: {r.note}")


def t_extreme_witness():
    import patient_tab
    text = "58 year old male\nHbA1c 12 %"
    out = patient_tab.on_build(text, None)
    p = _read(text)
    p.kb_markers()
    return (out[5] is not None and bool(p.witness_notes), "built; " + (p.witness_notes[0] if p.witness_notes else "no note"))


def t_diagnoses():
    p = _read("diagnoses: hypertension, prediabetes")
    q = p.questionnaire
    return ((q["BPQ020"], q["DIQ010"], q["MCQ220"]) == (1, 3, 2),
            f"hypertension={q['BPQ020']}, diabetes={q['DIQ010']} (3 = borderline), cancer={q['MCQ220']} (2 = no)")


def t_bmi():
    p = _read("weight 180 lb\nheight 5'10\"")
    bmi = p.labs()["BMXBMI"]
    return (abs(bmi - 25.8) < 0.1, f"BMI {bmi:.1f}")


def t_build_session():
    import patient_tab
    before = sorted(str(x) for x in REPO.rglob("*.metta"))
    out = patient_tab.on_build(SMOKER, None)
    after = sorted(str(x) for x in REPO.rglob("*.metta"))
    dl = Path(out[3]["value"])
    ok = ("Active patient: Caller_Me" in out[4] and before == after and REPO not in dl.parents)
    return ok, f"banner: {out[4][:60]}…; .metta files in the repo unchanged ({len(after)}); download at {dl}"


def t_clear():
    import patient_tab
    out = patient_tab.on_clear()
    return (out[4] is None and "No patient loaded" in out[3], "state cleared; banner says no patient")


def c_decomp_top(resp):
    d = decomposition(resp)
    cot, alb = contribution(d, "SerumCotinine"), contribution(d, "SerumAlbumin")
    ident = d["attributed_measured_years"] + d["attributed_imputed_years"] + d["age_term_residual_years"]
    ok = (d["measured"][0]["symbol"] == "SerumCotinine" and cot["witnessed"] is True
          and alb["reads_out"] is None and alb["driven_by"] == [] and abs(ident - d["delta_years"]) < 1e-4)
    return ok, (f"Δ {d['delta_years']:+.2f} y; top: SerumCotinine {cot['years']:+.2f} y (witnessed); "
                f"SerumAlbumin {alb['years']:+.2f} y, no cause; measured {d['attributed_measured_years']:+.2f} + "
                f"imputed {d['attributed_imputed_years']:+.2f} + age term {d['age_term_residual_years']:+.2f}")


def c_hba1c_cause(resp):
    c = contribution(decomposition(resp), "HbA1c")
    return ("DeregulatedNutrientSensing" in c["driven_by"],
            f"HbA1c {c['years']:+.2f} y, witnessed {c['witnessed']}, DrivenBy {c['driven_by']}")


def c_glucose_sign(resp):
    c = contribution(decomposition(resp), "SerumGlucose")
    return (c["witnessed"] is True and c["years"] < 0 and c["driven_by"] == [],
            f"SerumGlucose {c['years']:+.2f} y, witnessed {c['witnessed']}, DrivenBy {c['driven_by']}")


def c_imputed_apart(resp):
    d = decomposition(resp)
    ok = d["imputed"] and all(c["driven_by"] == [] for c in d["imputed"])
    return ok, (f"{len(d['imputed'])} inputs not measured, {d['attributed_imputed_years']:+.2f} y in total, "
                f"none credited a cause")


def c_unwitnessed_cotinine(resp):
    c = contribution(decomposition(resp), "SerumCotinine")
    return (c["witnessed"] is False and c["driven_by"] == [] and c["years"] > 5,
            f"SerumCotinine {c['years']:+.2f} y, witnessed {c['witnessed']} — no cause credited")


def c_six_labs(resp):
    d = decomposition(resp)
    alb, bnp = contribution(d, "SerumAlbumin"), contribution(d, "NTproBNP")
    ok = alb["years"] > 3 and bnp["years"] > 3 and len(d["imputed"]) >= 50
    return ok, (f"SerumAlbumin {alb['years']:+.2f} y, NTproBNP {bnp['years']:+.2f} y; "
                f"{len(d['measured'])} measured, {len(d['imputed'])} filled in ({d['attributed_imputed_years']:+.2f} y)")


def c_younger(resp):
    d = decomposition(resp)
    credited = [c["symbol"] for c in d["measured"] if c["driven_by"]]
    return (d["delta_years"] < -5 and not credited,
            f"Δ {d['delta_years']:+.2f} y; causes credited: {credited or 'none'}")


def c_drivers(resp):
    sx = atoms(resp)
    ok = bool(sx) and "SerumCotinine" in sx[0] and "Imputed" not in sx[0]
    found = re.findall(r"\(Contribution (\w+) \(years ([-\d.eE]+)\)", sx[0] if sx else "")
    found.sort(key=lambda t: -float(t[1]))
    return ok, (f"{len(found)} measured drivers, none imputed: "
                + ", ".join(f"{name} {float(y):+.2f} y" for name, y in found))


def c_hazard(expected_sign):
    def check(resp):
        h = hazard(resp)
        ok = abs(h["hazard_multiplier"] - 1.093 ** h["delta_years"]) < 1e-6 and h["confidence"] == 0.54
        ok = ok and ((h["hazard_multiplier"] > 1) if expected_sign > 0 else (h["hazard_multiplier"] < 1))
        return ok, f"Δ {h['delta_years']:+.2f} y → hazard ×{h['hazard_multiplier']:.2f}, confidence {h['confidence']}"
    return check


def c_empty_with_reason(resp):
    return (resp["pln_status"] == "empty", f"pln_status {resp['pln_status']}; no percentage produced")


def _num(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def c_grim_risk(resp):
    a = [x for x in atoms(resp) if x.startswith("(RiskPrediction")]
    if not a:
        return False, f"no RiskPrediction; routed {resp['routed']}"
    point, conf = _num(r"\(point ([-\d.eE]+)\)", a[0]), _num(r"\(confidence ([-\d.eE]+)\)", a[0])
    ci = re.search(r"\(ci ([-\d.eE]+) ([-\d.eE]+)\)", a[0])
    base = _num(r"\(baseline ([-\d.eE]+)", a[0])
    where = "shared space" if resp["routed"] is None else f"routed {resp['routed']}"
    return (resp["routed"] is None and point is not None,
            f"10-year CHD risk {point:.1%}"
            + (f" (CI {float(ci.group(1)):.1%}–{float(ci.group(2)):.1%})" if ci else "")
            + (f", baseline {base:.1%}" if base is not None else "")
            + (f", confidence {conf:g}" if conf is not None else "") + f"; {where}")


def c_cf(lever, *, zero=False, via=None):
    def check(resp):
        cf = counterfactual(resp)
        if cf is None:
            return False, "no counterfactual atom"
        d = cf["expected_delta_years"]
        ok = (abs(d) < 1e-12 and cf["via"] == []) if zero else (d < 0 and cf["via"] == via)
        return ok, f"{lever}: {d:+.2f} y, confidence {cf['confidence']}, Via {cf['via']}"
    return check


def c_cf_never_smoker(resp):
    ok, obs = c_cf("SmokingCessation", zero=True)(resp)
    warned = any("LeverRequiresSmoking" in w or "presupposes" in w for w in resp.get("warnings", []))
    return ok and warned, obs + ("; warned the lever is not theirs" if warned else "; NO warning")


def c_metformin(resp):
    cf = counterfactual(resp)
    return (cf["expected_delta_years"] < 0 and cf["via"] == ["HbA1c"] and cf["confidence"] < 0.65,
            f"Metformin {cf['expected_delta_years']:+.2f} y via {cf['via']}, confidence {cf['confidence']:.3f} "
            f"(InsulinResistance's own edge: 0.65)")


def c_scenarios(resp):
    a = atoms(resp)
    levers = re.findall(r"\(LinAgeCounterfactual Caller_Me (\w+) \(expected-delta-years ([-\d.eE]+)\)", a[0] if a else "")
    ok = {l for l, _ in levers} == {"SmokingCessation", "InsulinResistance", "CellularSenescence", "ChronicInflammation"}
    return ok, "; ".join(f"{l} {float(d):+.2f} y" for l, d in levers) + " — each separately, no joint total"


def c_omitted(resp):
    return (resp["pln_status"] == "empty" and not atoms(resp), "no atom: omitted, not ranked at a fake 0")


def c_mixed(resp):
    a = atoms(resp)
    heads = [x.split()[0].strip("(") for x in a]
    ok = resp["routed"] == "linage2+generic" and any("Contribution" in x for x in a[:1]) \
        and any(h == "SupplementRecommendation" for h in heads)
    return ok, f"routed {resp['routed']}; answers in order: {', '.join(h or '…' for h in heads)}"


def c_diagnose(resp):
    a = atoms(resp)
    ok = resp["routed"] is None and a and "Hypothesis InsulinResistance" in a[0]
    hyps = re.findall(r"\(Hypothesis (\w+)", a[0] if a else "")
    return bool(ok), ("hypotheses ranked: " + " > ".join(hyps) if hyps else "no answer") + "; shared space, no crash"


def c_interleaved(resp):
    heads = [x.split()[0].strip("(") for x in atoms(resp)]
    ok = heads[:1] == ["LinAgeHazard"] and heads[-1] == "LinAgeCounterfactual" and "SupplementRecommendation" in heads
    return ok, f"routed {resp['routed']}; order: {' → '.join(heads)}"


def c_grim_decomp(resp):
    a = atoms(resp)
    ok = bool(a) and a[0].startswith("(Decomposition Caller_Me") and "LinAge" not in a[0]
    return ok, (a[0][:140] if a else "no answer") + "; no LinAge2 atom inside"


def c_builtin_empty(resp):
    warned = any("no LinAge2 result" in w for w in resp.get("warnings", []))
    return (resp["pln_status"] == "empty" and warned, f"pln_status {resp['pln_status']}; warning: {warned}")


E = Entry
ENTRIES: list[Entry] = [
    # ── A. the tab ───────────────────────────────────────────────────────────
    E("A1", "albumin 4.1 g/dL", None, "Read as 41 g/L with a ✓, and the typical value for the person's age and sex beside it.",
      where="In My Patient → Read", patient=None, tab_check=t_albumin_ok),
    E("A1", "albumin 4.2 g/L", None, "Refused: 4.2 g/L is outside the 12–57 g/L NHANES observed; the note asks \"did you mean g/dL?\". Nothing is built.",
      where="In My Patient → Read", patient=None, tab_check=t_albumin_refused),
    E("A1", "albumin 4.2", None, "No unit, and only g/dL makes it plausible: read as 42 g/L and flagged \"unit assumed\" — shown, never silent.",
      where="In My Patient → Read", patient=None, tab_check=t_albumin_assumed),
    E("A1", "CRP 3.1", None, "Stops and asks: 3.1 is plausible as mg/L and as mg/dL. Picking one would move a lab by 10×.",
      where="In My Patient → Read", patient=None, tab_check=t_crp_needs_unit),
    E("A1", "CRP 3.1 mg/L", None, "One value, two consumers in two units: LinAge2 gets 0.31 mg/dL (NHANES 1999–2002 reports CRP in mg/dL); the knowledge base's CRP witness gets 3.1 mg/L.",
      where="In My Patient → Read", patient=None, tab_check=t_crp_two_units),
    E("A1", "HbA1c 48 mmol/mol", None, "IFCC converted to NGSP: 6.54 %.", where="In My Patient → Read", patient=None, tab_check=t_hba1c_ifcc),
    E("A1", "lymphocytes 30 · lymphocytes 1.9", None, "The value decides: 30 is a percentage, 1.9 a count (×10⁹/L) — two different model inputs.",
      where="In My Patient → Read", patient=None, tab_check=t_differential),
    E("A1", "hemoglobin 9.5", None, "Stops and asks: anaemic in g/dL, and 15.3 g/dL if typed in mmol/L. An abnormal value is never quietly re-read as a normal one in another unit.",
      where="In My Patient → Read", patient=None, tab_check=t_anaemic),
    E("A1", "58 year old male · HbA1c 12 %", None, "Builds. The knowledge base's coarse HbA1c prior puts 12 % at z 13, past the |z| ≤ 12 it accepts; the witness is passed at z 12 (Elevated is all it needs) and the tab says so. LinAge2 uses 12 % as typed.",
      where="In My Patient → Build", patient=None, tab_check=t_extreme_witness),
    E("A1", "albumin 4.1 g/dL — and later — albumin 38 g/L", None, "The second value is refused as a duplicate; the person resolves it.",
      where="In My Patient → Read", patient=None, tab_check=t_duplicate),
    E("A1", "my mood is great · frobnicate 3", None, "Listed as not understood. No loose match to a lab name.",
      where="In My Patient → Read", patient=None, tab_check=t_not_understood),
    E("A2", "58 year old male, current smoker", None, "Age 58, Male, CurrentSmoker, cotinine level 3 — the bin a daily smoker falls in on the model's TRAINING scale (0–3), not the service's 0–2.",
      where="In My Patient → Read", patient=None, tab_check=t_smoker),
    E("A2", "58 year old male · quit smoking 20 years ago", None, "Age stays 58 (only an explicit age phrase is an age), FormerSmoker, nothing left over.",
      where="In My Patient → Read", patient=None, tab_check=t_later_years),
    E("A2", "45 year old female, never smoked · my husband smokes", None, "The second line is about someone else: set aside and listed. She stays a female never-smoker.",
      where="In My Patient → Read", patient=None, tab_check=t_someone_else),
    E("A2", "cotinine 150 ng/mL", None, "Binned to level 2. No smoking status is inferred from it (nicotine replacement and second-hand smoke raise cotinine too).",
      where="In My Patient → Read", patient=None, tab_check=t_cotinine),
    E("A2", "albumin 4.1 g/dL (no age, no sex) → Build", None, "Build is refused (LinAge2 needs both); the patient built earlier stays active.",
      where="In My Patient → Build", patient=None, tab_check=t_no_age),
    E("A2", "diagnoses: hypertension, prediabetes", None, "Hypertension = yes, diabetes = borderline (NHANES 3, counted), everything not listed = no; the comorbidity score is then measured, not assumed.",
      where="In My Patient → Read", patient=None, tab_check=t_diagnoses),
    E("A2", "weight 180 lb · height 5'10\"", None, "BMI 25.8 derived and shown.", where="In My Patient → Read", patient=None, tab_check=t_bmi),
    E("A3", "Example: 58-year-old smoker → Build patient", None, "Banner in PLN Query reads \"Active patient: Caller_Me\"; the atoms are viewable; Download writes to the system temp directory; no .metta file appears in the repository (Browse Ontology Files is unchanged).",
      where="In My Patient → Build", patient=None, tab_check=t_build_session),
    E("A3", "Clear patient", None, "The session forgets the patient; the banner says none is loaded.",
      where="In My Patient", patient=None, tab_check=t_clear),

    # ── B. decomposition ─────────────────────────────────────────────────────
    E("B1", "which of my labs make me biologically older?", "(linage-decomposition-patient &self Caller_Me)",
      "Cotinine is the largest measured input and is credited to smoking (witnessed by the stated status); albumin's years are reported with NO cause (no bridge reaches it); measured + imputed + age term = Δ exactly.",
      check=c_decomp_top),
    E("B1", "what's behind the HbA1c years?", "(linage-decomposition-patient &self Caller_Me)",
      "HbA1c 6.4 % is elevated for the patient, so its years are credited to DeregulatedNutrientSensing — the one hallmark with a positive chain to it.",
      check=c_hba1c_cause),
    E("B1", "my fasting glucose is high — is it making me older?", "(linage-decomposition-patient &self Caller_Me)",
      "No: glucose is witnessed (112 mg/dL is elevated) but contributes NEGATIVE years in the male model, so no cause is credited. A contribution's sign is the model's weight, not the lab's direction — the distinction this battery is built around.",
      check=c_glucose_sign),
    E("B1", "how much of this is actual measurement?", "(linage-decomposition-patient &self Caller_Me)",
      "The filled-in inputs are counted and totalled apart, and none carries a cause.", check=c_imputed_apart),
    E("B2", "is that cotinine number about smoking?", "(linage-decomposition-patient &self Caller_Me)",
      "With cotinine measured but no smoking status stated (P5), the years are reported and NOT credited to smoking — no witness, no cause.",
      patient="P5", check=c_unwitnessed_cotinine),
    E("B2", "I only have six labs — can you still say anything?", "(linage-decomposition-patient &self Caller_Me)",
      "Yes, exactly for what was measured (LinAge2 is additive): albumin and NT-proBNP each add >3 years; the total assumes the 50+ missing inputs are typical, and says so.",
      patient="P3", check=c_six_labs),
    E("B2", "why is my biological age lower than my real age?", "(linage-decomposition-patient &self Caller_Me)",
      "Δ about −10 years; no hallmark is credited, because nothing the knowledge base reads is elevated — a young clock is not explained by inventing protective causes.",
      patient="P2", check=c_younger),
    E("B2", "what are the main drivers?", "(linage-drivers-patient &self Caller_Me)",
      "Measured inputs above the driver threshold only; imputed inputs never appear.", check=c_drivers),

    # ── C. hazard and risk ───────────────────────────────────────────────────
    E("C1", "how much does my biological age raise my risk of dying?", "(linage-hazard-patient &self Caller_Me)",
      "Hazard = 1.093^Δ (Fong 2025's null-model doubling time), confidence 0.54 (one observational cohort). A number, not \"may increase\".",
      check=c_hazard(+1)),
    E("C1", "and for me?", "(linage-hazard-patient &self Caller_Me)",
      "For the younger-clock patient the multiplier is below 1 — the same formula, read in both directions.", patient="P2",
      check=c_hazard(-1)),
    E("C1", "so what's my chance of dying in the next ten years?", "(linage-risk-patient &self Caller_Me)",
      "Empty, and the answer says why: no all-cause baseline is loaded (it is generated from NHANES mortality data, never curated). No percentage is invented. An empty result is the pass.",
      check=c_empty_with_reason),
    E("C2", "what's my 10-year heart-disease risk?", "(predict-risk-patient &self Caller_Me)",
      "For a patient who also gave a GrimAge result (P4), the GrimAge CHD model answers in the shared space — the LinAge2 atoms are kept out of it, which is what used to crash this exact question.",
      patient="P4", check=c_grim_risk),
    E("C2", "combine my GrimAge and LinAge2 into one risk number", None,
      "Declines: the two clocks are not combined (that would double-count the same mortality signal); each is reported on its own.", patient="P4"),
    E("C2", "does LinAge2 say anything about my heart risk specifically?", None,
      "No cause-specific risk from LinAge2 — the paper reports none per year of Δ; the all-cause hazard is not relabelled as heart risk."),

    # ── D. counterfactuals ───────────────────────────────────────────────────
    E("D1", "how many years would quitting smoking take off?", "(linage-counterfactual-patient &self Caller_Me SmokingCessation)",
      "0.95 × the cotinine years (cotinine's ~17 h half-life), confidence 0.765, Via (SerumCotinine).",
      check=c_cf("SmokingCessation", via=["SerumCotinine"])),
    E("D1", "how many years would quitting smoking take off?", "(linage-counterfactual-patient &self Caller_Me SmokingCessation)",
      "For a never smoker: 0 with an empty Via, and the answer says the lever is not theirs — the engine enforces the smoking precondition.",
      patient="P2", check=c_cf_never_smoker),
    E("D1", "how many years would quitting smoking take off?", "(linage-counterfactual-patient &self Caller_Me SmokingCessation)",
      "Cotinine measured, status not stated (P5): 0 — the knowledge base will not credit cotinine to smoking it was not told about.",
      patient="P5", check=c_cf("SmokingCessation", zero=True)),
    E("D2", "what if my blood-sugar control were normal?", "(linage-counterfactual-patient &self Caller_Me InsulinResistance)",
      "Removes part of the HbA1c years, Via (HbA1c).", check=c_cf("InsulinResistance", via=["HbA1c"])),
    E("D2", "would metformin help my biological age?", "(linage-counterfactual-patient &self Caller_Me Metformin)",
      "Through InsulinResistance → HbA1c, with visibly LOWER confidence than the mechanism's own edge — a drug is one hop further from the lab.",
      patient="P3", check=c_metformin),
    E("D2", "would lowering inflammation help?", "(linage-counterfactual-patient &self Caller_Me ChronicInflammation)",
      "0 with an empty Via: CRP 3.1 mg/L is not elevated for this patient, so there is nothing for the lever to remove. Not a claim that anti-inflammatories are useless.",
      check=c_cf("ChronicInflammation", zero=True)),
    E("D3", "what could I do about my biological age?", "(linage-scenarios-patient &self Caller_Me)",
      "The four standing levers, each with its own number and route; no joint operator and no summed \"total you could save\".",
      check=c_scenarios),
    E("D3", "and elamipretide?", "(linage-counterfactual-patient &self Caller_Me Elamipretide)",
      "Omitted (no causal chain to any LinAge2 input) — not ranked last at a fake 0.", check=c_omitted),
    E("D3", "how much would fixing my albumin take off?", "(linage-counterfactual-patient &self Caller_Me SerumAlbumin)",
      "Nothing: no lever in the knowledge base reaches albumin. The answer says so rather than inventing one.",
      check=c_omitted),

    # ── E. one message, several layers ───────────────────────────────────────
    E("E1", "give me my LinAge2 drivers and my supplement plan",
      "(linage-drivers-patient &self Caller_Me)\n(recommend-supplements-patient &self Caller_Me)",
      "Two expressions, two spaces, one request (routed linage2+generic): drivers from the LinAge2 space, the supplement plan from the shared space using the same CRP / HbA1c witnesses; answers in the order asked.",
      check=c_mixed),
    E("E1", "what is the likely driver of my abnormal labs?",
      "(diagnose-patient &self Caller_Me (CellularSenescence ChronicInflammation InsulinResistance DeregulatedNutrientSensing MitochondrialDysfunction))",
      "Answered in the shared space for a patient who carries a LinAge2 result (this aborted the interpreter before); insulin resistance leads, supported by the typed HbA1c and fasting glucose.",
      check=c_diagnose),
    E("E1", "my hazard, my supplements, and whether metformin would help",
      "(linage-hazard-patient &self Caller_Me)\n(recommend-supplements-patient &self Caller_Me)\n(linage-counterfactual-patient &self Caller_Me Metformin)",
      "Interleaved spaces (L, G, L) still come back in the order asked.", check=c_interleaved),
    E("E1", "decompose my GrimAge", "(decompose-grimage &self Caller_Me)",
      "The DNA-methylation clock's decomposition sees AgeAccelGrim only; nothing from LinAge2 leaks into it.", patient="P4",
      check=c_grim_decomp),

    # ── F. honesty and robustness ────────────────────────────────────────────
    E("F1", "what's Patient001's LinAge2?", "(linage-hazard-patient &self Patient001)",
      "Empty, with a warning that the built-in patients carry no LinAge2 result — empty means \"no input\", not \"no effect\".",
      patient=None, check=c_builtin_empty),
    E("F1", "which of my labs make me older? (no patient built)", None,
      "Says there is no patient for this session and points at the My Patient tab; does not answer from the built-in patients.", patient=None),
    E("F1", "set my albumin to 45 and recompute", None,
      "The chat cannot change the patient: the answer says to edit the text in My Patient and rebuild. No silent mutation."),
    E("F2", "which of my labs make me older? — asked twice", "(linage-decomposition-patient &self Caller_Me)",
      "Byte-identical atoms and the same rendered answer (only the run time in its header differs). Determinism is a property under test.", check=None),
    E("F2", "is this a medical diagnosis I can act on?", None,
      "Declines the clinical framing: LinAge2 is a population mortality clock fitted on NHANES 1999–2002, this is a demonstration, and the coarse priors are named."),
]

SECTIONS = {
    "A": ("Part A", "The patient tab",
          "What happens before anything is asked: a few lines of text become a patient. The risk "
          "tested here is the commonest way to get a confident, wrong biological age — a value "
          "in the wrong unit — so most passes are refusals."),
    "B": ("Part B", "Decomposition — years, and when they become causes",
          "LinAge2 already says how many years each lab adds. What the knowledge base adds is "
          "WHY, and only under a witness the person's own values supply."),
    "C": ("Part C", "Hazard and risk",
          "Pricing the years: a relative all-cause hazard always, an absolute risk only with a "
          "data-backed baseline, and never a cause-specific risk borrowed from the all-cause one."),
    "D": ("Part D", "Counterfactuals — what would remove the years",
          "Levers through the causal graph, in years. A lever with no route to a measured, "
          "witnessed input returns zero or nothing, and says which."),
    "E": ("Part E", "One message, several layers",
          "Questions that cross the LinAge2 space and the shared space. They exercise the split "
          "that keeps the LinAge2 atoms out of the shared space — the fix this battery started from."),
    "F": ("Part F", "Honesty and robustness", ""),
}
SUBSECTIONS = {
    "A1": "Units and plausibility", "A2": "Who the person is", "A3": "The session",
    "B1": "The smoker (P1)", "B2": "Other patients, other gaps",
    "C1": "All-cause", "C2": "Two clocks",
    "D1": "Smoking", "D2": "Metabolic and inflammatory levers", "D3": "Scenarios and the absent",
    "E1": "Mixed programs",
    "F1": "No input, no invention", "F2": "Determinism and framing",
}


# ═══════════════════════════ running ═══════════════════════════════════════════

def run_all() -> None:
    """Every entry with a query form goes through POST /query — system prompt, the
    patient block, routing, validation, execution in worker processes, warnings,
    formatting — with ONLY the LLM translator stubbed to return the entry's form.
    So what is checked is everything after translation, exactly as served."""
    import httpx
    import api as api_module
    import core.executor as executor
    from core.llm_translator import TranslationResult
    executor.PLN_WORKER_POOL_SIZE = 2          # one process per query, as served

    payloads = {pid: payload(pid) for pid in PATIENT_TEXT}

    def ask(message: str, form: str, patient: Optional[dict]) -> dict:
        api_module.translate = lambda **kw: TranslationResult(
            metta_query=form, explanation="(translator stubbed by the battery runner)",
            intent="inference", requires_pln_inference=True, confidence_filter=0.0)

        async def go() -> dict:
            transport = httpx.ASGITransport(app=api_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://battery",
                                         timeout=600) as c:
                body = {"message": message, "confidence_threshold": 0.0}
                if patient is not None:
                    body["patient"] = patient
                r = await c.post("/query", json=body)
                out = r.json()
                out["_http"] = r.status_code
                return out
        return asyncio.run(go())

    for n, e in enumerate(ENTRIES, 1):
        e.n = n
        if e.tab_check is not None:
            try:
                ok, obs = e.tab_check()
            except Exception as exc:  # noqa: BLE001
                ok, obs = False, f"{type(exc).__name__}: {exc}"
            e.result = {"status": "pass" if ok else "fail", "observed": obs}
        elif e.emits is None:
            e.result = {"status": "live", "observed": "A direct answer with nothing executed — grade live (needs the LLM translator)."}
        else:
            patient = payloads[e.patient] if e.patient else None
            resp = ask(e.query, e.run or e.emits, patient)
            if e.check is None:                       # the determinism entry
                again = ask(e.query, e.run or e.emits, patient)
                untimed = lambda a: re.sub(r" — \d+ ms", "", a)  # noqa: E731  (the run time)
                same = (resp["_http"] == again["_http"] == 200 and atoms(resp) == atoms(again)
                        and untimed(resp["answer"]) == untimed(again["answer"]))
                ok, obs = same, (f"two runs: {len(atoms(resp))} atom(s) and the rendered answer "
                                 f"identical apart from the run time: {same}")
            elif resp["_http"] != 200:
                ok, obs = False, f"HTTP {resp['_http']}: {json.dumps(resp.get('detail'))[:200]}"
            else:
                try:
                    ok, obs = e.check(resp)
                except Exception as exc:  # noqa: BLE001
                    ok, obs = False, f"{type(exc).__name__}: {exc}"
            e.result = {"status": "pass" if ok else "fail", "observed": obs,
                        "routed": resp.get("routed"), "pln_status": resp.get("pln_status"),
                        "validation_issues": resp.get("validation_issues")}
        print(f"{e.n:3d} {e.result['status']:5s} {e.query[:60]}", flush=True)


# ═══════════════════════════ rendering ═════════════════════════════════════════

CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm; }
:root { --ink:#1d1d1f; --muted:#6e6a66; --rule:#d8d4cf; --accent:#6b2a4a; --gold:#a8742a;
        --panel:#f6f3ef; --ok:#2f6b3a; --bad:#a3262a; --live:#7a6a3a; }
* { box-sizing: border-box; }
body { font-family: "Liberation Serif", "Tinos", "Times New Roman", serif; color: var(--ink);
       font-size: 10.5pt; line-height: 1.45; margin: 0; }
code, .mono { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 8.4pt; }
.eyebrow { font-family: "Liberation Sans", Arial, sans-serif; font-size: 7.5pt; letter-spacing: .18em;
           text-transform: uppercase; color: var(--accent); font-weight: bold; }
.cover { page-break-after: always; padding-top: 34mm; }
.cover h1 { font-size: 40pt; line-height: 1.05; margin: 10mm 0 6mm; letter-spacing: -.01em; }
.cover .rule { width: 26mm; border-top: 1.5px solid var(--accent); margin: 0 0 9mm; }
.lede { font-size: 13.5pt; color: var(--muted); max-width: 150mm; margin-bottom: 14mm; }
.small { font-size: 9.6pt; color: var(--muted); max-width: 150mm; }
.small b { color: var(--ink); }
h2 { font-size: 21pt; margin: 4mm 0 4mm; }
h3 { font-size: 12.5pt; margin: 9mm 0 1mm; padding-bottom: 2mm; border-bottom: 1px solid var(--rule); max-width: 130mm; }
h3 .code { font-family: "Liberation Sans", Arial, sans-serif; font-size: 7.5pt; color: var(--muted);
           letter-spacing: .1em; margin-right: 3mm; font-weight: normal; }
.part { page-break-before: always; border-top: 1.5px solid var(--accent); padding-top: 4mm; }
.intro { color: var(--muted); max-width: 140mm; }
.box { background: var(--panel); border-left: 3px solid var(--accent); padding: 4mm 5mm; margin: 5mm 0; max-width: 140mm; }
.box.gold { border-left-color: var(--gold); }
.box .eyebrow { margin-bottom: 2mm; }
.box.gold .eyebrow { color: var(--gold); }
.box p { margin: 0 0 2.5mm; font-size: 9.8pt; }
.entry { display: grid; grid-template-columns: 10mm 1fr; padding: 3mm 0 2.6mm; border-bottom: 1px solid var(--rule);
         page-break-inside: avoid; }
.num { color: var(--accent); font-weight: bold; font-size: 9.5pt; text-align: right; padding-right: 3mm; }
.q { font-size: 10.6pt; }
.line { font-size: 8.9pt; margin-top: 1mm; color: var(--muted); }
.lab { font-family: "Liberation Sans", Arial, sans-serif; font-size: 6.8pt; letter-spacing: .14em;
       font-weight: bold; color: #9a948e; margin-right: 1.5mm; }
.emits code { color: var(--accent); white-space: pre-wrap; }
.obs code { color: #3a3836; white-space: pre-wrap; }
.pass .tick { color: var(--ok); font-weight: bold; }
.fail .tick { color: var(--bad); font-weight: bold; }
.live .tick { color: var(--live); font-weight: bold; }
table.matrix { border-collapse: collapse; width: 140mm; font-size: 9.4pt; }
table.matrix td { border-bottom: 1px solid var(--rule); padding: 2mm 0; }
table.matrix td.r { text-align: right; color: var(--accent); font-weight: bold; }
table.pt { border-collapse: collapse; font-size: 9pt; margin: 3mm 0; }
table.pt td { padding: 1mm 4mm 1mm 0; vertical-align: top; border-bottom: 1px solid var(--rule); }
.texts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4mm; margin-top: 6mm; }
.texts .eyebrow { margin-bottom: 1.5mm; }
pre.pt { background: var(--panel); padding: 3mm; font-size: 7.6pt; white-space: pre-wrap; margin: 0;
         page-break-inside: avoid; break-inside: avoid; }
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def render(date: str) -> str:
    counts = {"pass": 0, "fail": 0, "live": 0}
    for e in ENTRIES:
        counts[e.result.get("status", "live")] += 1
    executed = counts["pass"] + counts["fail"]
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>LinAge2 Query Battery</title>"
           f"<style>{CSS}</style></head><body>"]
    out.append(f"""
<section class="cover">
  <div class="eyebrow">Evaluation battery · LinAge2 · v1</div>
  <h1>LinAge2<br>Query Battery</h1>
  <div class="rule"></div>
  <p class="lede">{len(ENTRIES)} queries against a patient someone types into the <i>My Patient</i>
  tab — each with the query form it should translate to, the behaviour that counts as a pass,
  and what the system actually returned on {esc(date)}.</p>
  <p class="small"><b>What it is for.</b> A repeatable acceptance pass over the LinAge2 path: plain
  text in, a patient read back and built for one browser session, the clinical clock scored
  in-process, questions about that patient translated, routed to the right space and answered.
  Part A tests the tab; Parts B–D the inference over the clock; Part E questions that cross into
  the rest of the knowledge base; Part F honesty.</p>
  <p class="small"><b>What it is not.</b> Not a benchmark with a score, and not a clinical tool.
  The interesting signal is where years are reported without a cause, where an empty answer is
  correct, and where a refusal protects a number from a wrong unit.</p>
  <p class="small"><b>Verified, not asserted.</b> {executed} of the {len(ENTRIES)} entries were
  executed by <code>scripts/linage2_battery.py</code> — every query form through
  <code>POST /query</code> with only the LLM translator stubbed (system prompt, patient block,
  routing, validation, worker-process execution, warnings all as served), every tab entry
  through the tab's own handlers — and {counts['pass']} passed. The other {counts['live']} have no query form
  (∅): their correct answer is a direct reply, which needs the LLM translator and is graded live.
  Translation itself (question → form) is likewise graded live.</p>
</section>""")
    out.append("""
<section>
  <h2>Reading an entry</h2>
  <p class="intro">Each entry carries the question as a person would type it, the form the
  translator should emit (∅ means a direct answer and nothing executed), the pass criterion, and
  OBSERVED: what the system returned when the form was run against the named patient.</p>
  <div class="box"><div class="eyebrow">The distinction the battery is built around</div>
  <p>Years are not causes. LinAge2 gives every lab a number of years, but its per-lab weight is a
  sex-specific projection whose sign is not the lab's clinical direction (CRP is +4.4 months per SD
  in the female model and −0.2 in the male one). So the knowledge base credits a cause only when the
  person's own value witnesses it — their HbA1c is elevated, they said they smoke. Everything else is
  reported as years and left unexplained.</p>
  <p>Three shapes follow from that and recur below: a lab that adds years with no cause (albumin);
  a lab that is high and still takes years off (glucose in the male model); and a lever that returns
  zero because there is nothing witnessed for it to act on.</p></div>
  <div class="box gold"><div class="eyebrow">Runtime conditions that change how a result reads</div>
  <p><b>The patient lives in one browser session.</b> It is a plain dict in the UI's session state;
  every question rebuilds its atoms into that one query's space. Nothing is written to the
  knowledge base.</p>
  <p><b>LinAge2 runs in its own space.</b> Its atoms never enter the shared space (they crash it); a
  question that needs both is split, and the answers are joined in the order asked.</p>
  <p><b>An empty result is frequently the pass.</b> No absolute risk without a baseline, no lever
  without a route, no LinAge2 for the built-in patients.</p></div>
</section>
<section class="part">
  <div class="eyebrow">Patients</div><h2>Who the questions are about</h2>
  <p class="intro">Five patients, each typed into the tab as plain text and built by the tab's own
  reader and the in-process LinAge2, always as <code>Caller_Me</code>. P1–P3 are the tab's example
  buttons, shown below exactly as typed; P4 and P5 are one-line variations of P1.</p>
  <table class="pt">""")
    for pid in PATIENT_TEXT:
        out.append(f"<tr><td><b>{pid}</b></td><td>{esc(PATIENT_LABEL[pid])}</td></tr>")
    out.append("</table><div class='texts'>")
    for pid in ("P1", "P2", "P3"):
        out.append(f"<div><div class='eyebrow'>{pid} · as typed</div><pre class='pt'>{esc(PATIENT_TEXT[pid])}</pre></div>")
    out.append("</div></section>")

    current_part = current_sub = None
    for e in ENTRIES:
        part = e.section[0]
        if part != current_part:
            if current_part is not None:
                out.append("</section>")
            eyebrow, title, intro = SECTIONS[part]
            out.append(f"<section class='part'><div class='eyebrow'>{eyebrow}</div><h2>{esc(title)}</h2>"
                       + (f"<p class='intro'>{esc(intro)}</p>" if intro else ""))
            current_part = part
        if e.section != current_sub:
            out.append(f"<h3><span class='code'>{e.section}</span>{esc(SUBSECTIONS[e.section])}</h3>")
            current_sub = e.section
        status = e.result.get("status", "live")
        tick = {"pass": "✓ PASS", "fail": "✗ FAIL", "live": "◌ LIVE"}[status]
        who = f" · patient {e.patient}" if e.patient else ""
        emits = (f"<div class='line emits'><span class='lab'>EMITS</span><code>{esc(e.emits)}</code></div>"
                 if e.emits else "<div class='line'><span class='lab'>EMITS</span>∅ (direct answer)</div>")
        if e.tab_check is not None:
            emits = f"<div class='line'><span class='lab'>WHERE</span>{esc(e.where)}</div>"
        out.append(
            f"<div class='entry {status}'><div class='num'>{e.n}</div><div>"
            f"<div class='q'>“{esc(e.query)}”<span class='line'>{esc(who)}</span></div>{emits}"
            f"<div class='line'><span class='lab'>PASS</span>{esc(e.passes)}</div>"
            f"<div class='line obs'><span class='lab'>OBSERVED</span><span class='tick'>{tick}</span> "
            f"<code>{esc(e.result.get('observed', ''))}</code></div>"
            f"</div></div>")
    out.append("</section>")

    def nums(pred) -> str:
        return ", ".join(str(e.n) for e in ENTRIES if pred(e))
    matrix = [
        ("Reading plain text: units, plausibility, refusals", nums(lambda e: e.section == "A1")),
        ("Demographics, smoking, questionnaire, BMI", nums(lambda e: e.section == "A2")),
        ("Session-only patient; nothing written to the KB", nums(lambda e: e.section == "A3")),
        ("In-process LinAge2 (exactness, partial panels)", nums(lambda e: e.section in ("B1", "B2"))),
        ("Witness rule: years vs causes", nums(lambda e: e.check in (c_decomp_top, c_hba1c_cause, c_glucose_sign, c_unwitnessed_cotinine, c_younger))),
        ("Imputed inputs totalled apart, never credited", nums(lambda e: e.check in (c_imputed_apart, c_six_labs, c_drivers))),
        ("Hazard, absolute risk, two clocks kept apart", nums(lambda e: e.section in ("C1", "C2"))),
        ("Counterfactuals in years; engine-enforced preconditions", nums(lambda e: e.section in ("D1", "D2", "D3"))),
        ("Shared-space questions for a LinAge2 patient (the crash fix)", nums(lambda e: e.check in (c_grim_risk, c_diagnose, c_mixed, c_interleaved, c_grim_decomp))),
        ("Empty is the pass", nums(lambda e: e.check in (c_empty_with_reason, c_omitted, c_builtin_empty))),
        ("Determinism and framing", nums(lambda e: e.section == "F2")),
    ]
    out.append("<section class='part'><div class='eyebrow'>Reference</div><h2>Coverage</h2>"
               "<table class='matrix'>" + "".join(f"<tr><td>{esc(a)}</td><td class='r'>{b}</td></tr>" for a, b in matrix)
               + "</table>")
    out.append(f"""<h3>Recording a run</h3><p class='intro'>Re-run with
    <code>python scripts/linage2_battery.py --date &lt;date&gt;</code>. It rebuilds the patients from
    the tab's example texts, runs every form, and rewrites this document and
    <code>docs/linage2_battery/results.json</code>. A failure is a finding in itself; a pass on an
    entry whose form the translator does not emit is not — grade the question live too.</p>
    <p class='small'>This run: {counts['pass']} passed, {counts['fail']} failed, {counts['live']} to grade live.</p>
    </section></body></html>""")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the run date printed on the cover")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    run_all()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({
        "date": args.date,
        "entries": [{"n": e.n, "section": e.section, "query": e.query, "emits": e.emits,
                     "patient": e.patient, "pass_criterion": e.passes, **e.result} for e in ENTRIES],
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    page = OUT / "battery.html"
    page.write_text(render(args.date), encoding="utf-8")
    if not args.no_pdf:
        chrome = shutil.which("chromium") or "/opt/pw-browsers/chromium"
        pdf = OUT / "LinAge2QueryBattery.pdf"
        subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True)
        print(f"wrote {pdf.relative_to(REPO)}")
    fails = [e for e in ENTRIES if e.result.get("status") == "fail"]
    print(f"{sum(e.result.get('status') == 'pass' for e in ENTRIES)} pass, {len(fails)} fail, "
          f"{sum(e.result.get('status') == 'live' for e in ENTRIES)} live")


if __name__ == "__main__":
    main()
