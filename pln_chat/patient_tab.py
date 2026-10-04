"""The "My Patient" tab: type a few lines about yourself, check what was read,
build a patient that lives only in this browser session, then ask about it.

Nothing here is written into the knowledge base. The patient is a plain dict in a
`gr.State` (one per browser session, gone on reload); every chat turn rebuilds its
atoms with `build_patient` and injects them into that one query's space, exactly as
`/query` does with a `patient` object. The "Download .metta" file is written to a
temp directory (pruned after an hour) — never to an ontology folder, which every KB
loader scans.

The flow, and why it has two buttons:

    Read   -> core.patient_text.read_patient_text: every value with the unit it was
              typed in, the unit LinAge2 takes, and a status. A value whose unit
              cannot be pinned down stops here, with the reason.
    Build  -> core.linage2_model (LinAge2, in-process) -> build_patient -> state.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr

from core.linage2_model import load_model
from core.patient_builder import BuiltPatient, PatientSpecError
from core.patient_context import build_caller_patient
from core.patient_text import (
    ASSUMED_UNIT,
    EXAMPLES,
    OK,
    SPECS,
    ParsedPatient,
    read_patient_text,
)

PATIENT_ID = "Me"

#: Questions that exercise the patient, for the "Try asking" buttons. Each maps
#: to a form the translator knows (prompt rule 16 and the per-patient hint).
SUGGESTED_QUESTIONS = (
    "Which of my labs make me biologically older, and why?",
    "How much does my biological age raise my risk of dying?",
    "How many years would quitting smoking take off my biological age?",
    "What could I do about my biological age?",
    "Give me my LinAge2 drivers and my supplement plan.",
    "What is the likely driver of my abnormal labs?",
)

_STATUS_ICON = {OK: "✓", ASSUMED_UNIT: "⚠ assumed"}

#: Readable names for the model inputs the reader does not type directly.
_INPUT_LABEL = {
    "LBXCOT": "Cotinine (smoking)", "fs1Score": "Comorbidity score",
    "fs2Score": "Self-rated health", "fs3Score": "Healthcare use",
    "crAlbRat": "Urine albumin/creatinine",
}


def _label(code: str) -> str:
    return _INPUT_LABEL.get(code) or (SPECS[code].label if code in SPECS else code)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def render_reading(parsed: ParsedPatient) -> str:
    """The 'here's what we read' table, as Markdown."""
    who = []
    who.append(f"**Age** {parsed.age:g}" if parsed.age is not None else "**Age** —")
    who.append(f"**Sex** {parsed.sex or '—'}")
    if parsed.smoking:
        who.append(f"**Smoking** {parsed.smoking} (cotinine level {parsed.cotinine_level})")
    elif parsed.cotinine_level is not None:
        who.append(f"**Cotinine level** {parsed.cotinine_level}")
    lines = ["### What was read", " · ".join(who), ""]
    model = load_model()
    if parsed.readings:
        lines += ["| Input | You typed | LinAge2 value | Check | Typical* |",
                  "|---|---|---|---|---|"]
        for r in parsed.readings:
            typical = "—"
            if parsed.age is not None and parsed.sex in ("Male", "Female") and r.code in model.lab_inputs \
                    and 20 <= parsed.age <= 90:
                unit = r.unit or (SPECS[r.code].unit if r.code in SPECS else "")
                typical = f"{_fmt(model.imputed_value(parsed.sex, parsed.age, r.code))} {unit}"
            status = _STATUS_ICON.get(r.status, f"✗ {r.status}")
            note = f"<br><sub>{r.note}</sub>" if r.note else ""
            value = f"{_fmt(r.value)} {r.unit}" if r.value is not None else "—"
            lines.append(f"| {r.label} | `{r.typed}` | {value} | {status}{note} | {typical} |")
        lines.append("\n<sub>*median for NHANES 1999-2000 participants of the same sex and "
                     "about the same age; any lab not given is filled with this value and "
                     "flagged.</sub>")
    else:
        lines.append("_No lab values read yet._")
    if parsed.cotinine_note:
        lines.append(f"\n- Smoking → {parsed.cotinine_note}")
    for note in parsed.questionnaire_notes:
        lines.append(f"- Questionnaire → {note}")
    for note in parsed.notes:
        lines.append(f"- {note}")
    if not parsed.all_problems():
        parsed.kb_markers()
        for note in parsed.witness_notes:
            lines.append(f"- {note}")
    if parsed.not_understood:
        lines.append("\n**Not understood** (ignored — rephrase as `name value unit`):")
        lines += [f"- `{s}`" for s in parsed.not_understood]
    problems = parsed.all_problems()
    if problems:
        lines.append("\n**Fix before building:**")
        lines += [f"- ✗ {p}" for p in problems]
    else:
        lines.append("\n✓ **Ready to build.**")
    return "\n".join(lines)


def render_summary(built: BuiltPatient) -> str:
    """What the built patient is, in a few lines — and what to ask next."""
    lin = built.linage2
    sign = "+" if lin.delta_years >= 0 else "−"
    head = (f"### {built.patient_id} — built for this session\n"
            f"LinAge2 biological age **{lin.biological_age:.1f}** at chronological age "
            f"{lin.chronological_age:g}: **{sign}{abs(lin.delta_years):.1f} years** "
            f"({lin.status.lower()} on the clock's spread, z = {lin.z:.2f}).")
    measured = sorted(lin.measured, key=lambda c: -c.years)
    older = [c for c in measured if c.years > 0.05][:6]
    younger = [c for c in sorted(lin.measured, key=lambda c: c.years) if c.years < -0.05][:3]
    lines = [head, ""]
    if older:
        lines.append("- **Measured labs adding years:** " + " · ".join(
            f"{_label(c.code)} {c.years:+.2f} y" for c in older))
    if younger:
        lines.append("- **Measured labs taking years off:** " + " · ".join(
            f"{_label(c.code)} {c.years:+.2f} y" for c in younger))
    imputed_total = sum(c.years for c in lin.imputed)
    lines.append(
        f"- **Not measured:** {len(lin.imputed)} of {len(lin.contributions)} inputs were filled "
        f"in ({imputed_total:+.2f} y in total); the knowledge base totals them apart and never "
        f"credits a cause to one.")
    lines.append("\nYears are not causes: a lab's years depend on LinAge2's sex-specific "
                 "weights, so the knowledge base credits a cause only when your own value "
                 "is elevated (or, for cotinine, you smoke). Ask it in **PLN Query**:")
    return "\n".join(lines)


def render_banner(state: Optional[dict]) -> str:
    if not state:
        return ("**No patient loaded.** Questions about \"me\" need one — build it in the "
                "**My Patient** tab. The built-in patients (Patient001, Patient002) have no "
                "LinAge2 result.")
    lin = state.get("linage2") or {}
    meta = lin.get("metadata") or {}
    delta = meta.get("delta_ba_ca")
    bits = [f"**Active patient: Caller_{state.get('id', PATIENT_ID)}**",
            f"{state.get('age'):g}-year-old {str(state.get('sex', '')).lower()}"]
    if state.get("smoking"):
        bits.append(state["smoking"])
    if delta is not None:
        bits.append(f"LinAge2 {float(delta):+.1f} y")
    bits.append("ask about \"me\" / \"my labs\"")
    return " · ".join(bits)


#: One directory per process for the downloads; files older than this are pruned on
#: each build (Gradio serves a download from its own cache copy, made when it is shown).
_DOWNLOAD_DIR = Path(tempfile.mkdtemp(prefix="pln_patients_"))
_DOWNLOAD_TTL_S = 3600
atexit.register(shutil.rmtree, _DOWNLOAD_DIR, True)


def _write_download(built: BuiltPatient) -> str:
    """A .metta copy for the person to keep: a NEW temp file per build (never shared
    between sessions), in a temp directory — not a folder the KB loads."""
    now = time.time()
    for old in _DOWNLOAD_DIR.glob("*.metta"):
        try:
            if now - old.stat().st_mtime > _DOWNLOAD_TTL_S:
                old.unlink()
        except OSError:
            pass
    header = (f";; {built.patient_id} — built in the PLN 'My Patient' tab for one browser\n"
              f";; session. Not part of the knowledge base; load it as extra_atoms or\n"
              f";; send the same text again to rebuild it.\n")
    _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"pln_patient_{built.patient_id}_", suffix=".metta",
                                dir=_DOWNLOAD_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(header + built.atoms + "\n")
    return path


#: The builder's notes are written for /patients/preview callers, who send z-scores and
#: whole LinAge2 blocks. The tab's user typed text: the same facts, in their terms.
_TAB_WORDING = (
    ("The LinAge2 block was sent without any of the markers",
     "None of your values can be a knowledge-base witness (CRP, HbA1c, or a glucose marked "
     "fasting) and current smoking was not stated. The per-lab years are reported, but no "
     "cause can be credited and every counterfactual returns 0: a lab's years carry the sign "
     "of LinAge2's sex-specific weights, so the knowledge base needs your own elevated value "
     "to call a lab high."),
    ("Standardised server-side from a raw value:",
     None),     # rewritten below with its marker list
    ("No AgeAccelGrim measurement:",
     "No GrimAge acceleration given: the 10-year heart-disease risk model reads that clock "
     "and returns nothing for you (add a line such as 'GrimAge acceleration +3 years' if you "
     "have one). Diagnosis, supplement ranking and intervention ranking still work."),
)


def _in_tab_words(note: str) -> str:
    for prefix, text in _TAB_WORDING:
        if note.startswith(prefix):
            if text is not None:
                return text
            markers = note[len(prefix):].split(".")[0].strip()
            return (f"{markers}: turned into z-scores against one pooled reference (there is no "
                    f"age- and sex-specific table here), so the knowledge base's Elevated / "
                    f"Normal call on them is not adjusted for your age or sex.")
    return note.replace("measurement was sent", "measurement was given")


# ── handlers ──────────────────────────────────────────────────────────────────

def on_read(text: str) -> str:
    return render_reading(read_patient_text(text or ""))


def on_build(text: str, state: Optional[dict]):
    """-> (reading, summary, atoms, download, banner, state, suggestions visible)"""
    parsed = read_patient_text(text or "")
    reading = render_reading(parsed)
    try:
        payload, result = parsed.to_patient(PATIENT_ID)
        built = build_caller_patient(payload, ())
    except PatientSpecError as exc:
        return (reading, f"### Not built\n✗ {exc.message}", gr.update(),
                gr.update(), render_banner(state), state, gr.update())
    # the model's own caveats (age outside 40-85, questionnaire assumed); its count of
    # filled-in labs repeats the builder's IMPUTED note, so it is left to that one
    model_notes = [w for w in result.warnings if "were not given and were filled" not in w]
    warnings = list(dict.fromkeys(model_notes + parsed.witness_notes
                                  + [_in_tab_words(w) for w in built.warnings]))
    summary = render_summary(built)
    caveat = next((w for w in model_notes if "extrapolation" in w), None)
    if caveat:
        summary = summary.replace("\n\n", f"\n\n⚠ {caveat}\n\n", 1)
    if warnings:
        summary += "\n\n<details><summary>Notes on this patient</summary>\n\n" + "\n".join(
            f"- {w}" for w in warnings) + "\n</details>"
    return (reading, summary, gr.update(value=built.atoms, visible=True),
            gr.update(value=_write_download(built), visible=True), render_banner(payload),
            payload, gr.update(visible=True))


def on_clear():
    """-> (summary, atoms, download, banner, state, suggestions visible)"""
    return ("_No patient built._", gr.update(value="", visible=False), gr.update(visible=False),
            render_banner(None), None, gr.update(visible=False))


def build_tab(patient_state: gr.State, banner: gr.Markdown, question_box: gr.Textbox,
              tabs: gr.Tabs, query_tab_id: str) -> None:
    """Lay out the tab inside the caller's `gr.Tabs()` context and wire it."""
    gr.Markdown(
        "Describe yourself in a few lines — age, sex, smoking, any lab values with their "
        "units, diagnoses. **Read** shows what was understood; **Build** scores LinAge2 "
        "(the blood-panel mortality clock of Fong et al. 2025) and makes you a patient for "
        "this browser session only. Nothing is added to the knowledge base (the optional "
        "download is a temporary file made for you alone)."
    )
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(label="About you", lines=16, max_lines=40,
                              value=EXAMPLES["58-year-old smoker"])
            with gr.Row():
                example_btns = [gr.Button(f"Example: {name}", size="sm") for name in EXAMPLES]
            with gr.Row():
                read_btn = gr.Button("Read", variant="secondary")
                build_btn = gr.Button("Build patient", variant="primary")
                clear_btn = gr.Button("Clear patient", variant="stop", size="sm")
        with gr.Column(scale=3):
            reading = gr.Markdown(on_read(EXAMPLES["58-year-old smoker"]))
    summary = gr.Markdown("_No patient built._")
    with gr.Column(visible=False) as suggestions:
        with gr.Row():
            ask_btns = [gr.Button(q, size="sm") for q in SUGGESTED_QUESTIONS]
    with gr.Accordion("This patient as MeTTa atoms (this session only)", open=False):
        atoms = gr.Code(value="", language=None, interactive=False, visible=False,
                        label="Injected into each query's space; the LinAge2 atoms only into "
                              "the LinAge2 space")
        download = gr.DownloadButton("Download .metta", visible=False)

    for btn, name in zip(example_btns, EXAMPLES):
        btn.click(lambda n=name: (EXAMPLES[n], on_read(EXAMPLES[n])), outputs=[text, reading])
    read_btn.click(on_read, inputs=text, outputs=reading)
    build_btn.click(on_build, inputs=[text, patient_state],
                    outputs=[reading, summary, atoms, download, banner, patient_state, suggestions])
    clear_btn.click(on_clear, outputs=[summary, atoms, download, banner, patient_state, suggestions])
    for btn, question in zip(ask_btns, SUGGESTED_QUESTIONS):
        btn.click(lambda q=question: (q, gr.Tabs(selected=query_tab_id)),
                  outputs=[question_box, tabs])
