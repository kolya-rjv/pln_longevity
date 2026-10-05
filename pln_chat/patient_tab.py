"""The "My Patient" tab: type a few lines about yourself, check what was read,
build a patient that lives only in this browser session, then ask about it.

Nothing here is written into the knowledge base. The patient is a plain dict in a
`gr.State` (one per browser session, gone on reload); every chat turn rebuilds its
atoms with `build_patient` and injects them into that one query's space, exactly as
`/query` does with a `patient` object. The "Download .metta" file is written to a
temp directory (pruned after an hour) — never to an ontology folder, which every KB
loader scans.

The flow, and why it has two buttons:

    Read   -> core.patient_read.read_patient: the fixed rules (core.patient_text),
              and — only from this button, only if OPENAI_API_KEY is set — a model
              that rewrites what the rules did not understand into the rules' own
              wording (never a value: numbers and units are copied from what you
              typed). Every value with the unit it was typed in, the unit LinAge2
              takes, and a status; a row from a rewrite says "· model". A value
              whose unit cannot be pinned down stops here, with the reason; what
              the rules refused is only ever offered as a wording to click.
    Build  -> the reading stored at Read (never the model again) -> core.linage2_model
              (LinAge2, in-process) -> build_patient -> state. If the text changed
              since Read, Build asks for Read first.

The page's first reading and the example buttons use the rules alone.
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
from core.patient_builder import (NO_WITNESS_CHD_PREFIX, NO_WITNESS_PREFIX, STILL_WORK, BuiltPatient,
                                  PatientSpecError)
from core.patient_context import build_caller_patient
from core.patient_extract import OpenAIExtractor, configured_model
from core.patient_read import PatientRead, read_patient
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
#: to a form the translator knows (prompt rules 16-17 and the per-patient hint; the
#: last one is the abductive diagnosis, not a LinAge2 form).
SUGGESTED_QUESTIONS = (
    "Which of my labs make me biologically older, and why?",
    "How much does my biological age raise my risk of dying?",
    "How many years would quitting smoking take off my biological age?",
    "What could I do about my biological age?",
    "Give me my LinAge2 drivers and my supplement plan.",
    "What is the likely driver of my abnormal labs?",
)

#: What in the patient makes each button's form answer rather than come back empty or at
#: zero. Measured per patient (8 patients, every form), not guessed:
#:   always             the LinAge2 forms (decomposition, hazard): every built patient has a result
#:   smoker             quitting smoking is 0.0 for anyone who is not a current smoker
#:   witness_or_smoker  "what could I do": a lever moves only for an elevated witness or a smoker
#:   witness            the supplement plan and the diagnosis read witnesses only
_NEEDS = ("always", "always", "smoker", "witness_or_smoker", "witness", "witness")

#: The fifth button without a witness: the plan half would be empty, the drivers half still answers.
DRIVERS_ONLY_QUESTION = "What are the main drivers of my biological age?"


def _offered(need: str, built: BuiltPatient) -> bool:
    smoker = built.smoking == "CurrentSmoker"
    return {"always": True, "smoker": smoker, "witness": bool(built.witnesses),
            "witness_or_smoker": bool(built.witnesses) or smoker}[need]


def question_buttons(built: Optional[BuiltPatient]) -> list[tuple[str, bool]]:
    """(label, offered) for each suggested-question slot, for this patient. A question whose
    form would come back empty or at zero for them is not offered (`hidden_note` says so);
    with no patient yet, only the always-answering ones are."""
    out = []
    for q, need in zip(SUGGESTED_QUESTIONS, _NEEDS):
        offered = _offered(need, built) if built is not None else need == "always"
        if q == SUGGESTED_QUESTIONS[4] and built is not None and not built.witnesses:
            out.append((DRIVERS_ONLY_QUESTION, True))
        else:
            out.append((q, offered))
    return out


def hidden_note(built: Optional[BuiltPatient]) -> str:
    """One line on what was left out of the buttons and why ("" when nothing was)."""
    if built is None:
        return ""
    smoker = built.smoking == "CurrentSmoker"
    parts = []
    if not smoker:
        parts.append("quitting smoking (you are not a current smoker)")
    if not built.witnesses:
        parts.append("the diagnosis and the supplement plan (none of your labs is elevated with "
                     "a cause the knowledge base has curated)")
        if not smoker:
            parts.append("\"what could I do\" (nothing it can move)")
    if not parts:
        return ""
    return ("_Not offered for you, because they would come back empty: " + "; ".join(parts)
            + ". You can still type them in **PLN Query**._")


_STATUS_ICON = {OK: "✓", ASSUMED_UNIT: "⚠ assumed"}

#: How many "Use this wording" buttons the tab has room for (the rest are listed).
SUGGESTION_SLOTS = 6

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


def _from_model(parsed: ParsedPatient, read: Optional[PatientRead], fact: str) -> str:
    """' (model)' when the statement that gave `fact` came from a model rewrite."""
    if read is None or not read.model_quotes:
        return ""
    return " (model)" if any(fact in st.facts and read.source(st.index) == "model"
                             for st in parsed.statements) else ""


def render_reading(parsed: ParsedPatient, read: Optional[PatientRead] = None) -> str:
    """The 'here's what we read' table, as Markdown. With a PatientRead, rows from a
    model rewrite say '· model' and show what was typed; the rest is unchanged."""
    who = []
    who.append(f"**Age** {parsed.age:g}{_from_model(parsed, read, 'age')}"
               if parsed.age is not None else "**Age** —")
    who.append(f"**Sex** {parsed.sex or '—'}{_from_model(parsed, read, 'sex') if parsed.sex else ''}")
    if parsed.smoking:
        who.append(f"**Smoking** {parsed.smoking} (cotinine level {parsed.cotinine_level})"
                   f"{_from_model(parsed, read, 'smoking')}")
    elif parsed.cotinine_level is not None:
        who.append(f"**Cotinine level** {parsed.cotinine_level}")
    header = read.header() if read is not None else "Read by rules only"
    lines = ["### What was read", f"<sub>{header}</sub>", "", " · ".join(who), ""]
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
            typed = r.typed
            if read is not None and read.source(r.statement) == "model":
                status, typed = f"{status} · model", read.model_quotes[r.statement]
            note = f"<br><sub>{r.note}</sub>" if r.note else ""
            value = f"{_fmt(r.value)} {r.unit}" if r.value is not None else "—"
            lines.append(f"| {r.label} | `{typed}` | {value} | {status}{note} | {typical} |")
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
    if read is not None and read.reader == "rules+model":
        lines += _render_model_part(read)
    return "\n".join(lines)


def _code(text: str) -> str:
    """Text shown verbatim in a Markdown code span: nothing in it can close the span."""
    return "`" + str(text).replace("`", "'").replace("\n", " ").replace("\r", " ") + "`"


def _render_model_part(read: PatientRead) -> list:
    out = []
    if read.substitutions:
        out += ["\n**Read as** — the model's rewrites of what the rules did not understand, which "
                "the rules then read:", "```", read.read_as, "```"]
    if read.suggestions:
        out.append("\n**Wordings to choose** (the buttons below put one in place of what you typed, "
                   "and read again):")
        for s_ in read.suggestions:
            words = " or ".join(_code(w) for w in s_.wordings)
            out.append(f"- {'⛔ ' if s_.blocking else ''}{_code(s_.original)} — {s_.reason}: {words}")
    for note in read.notes:
        out.append(f"- Model → {note}")
    if read.discarded:
        out.append(f"\n<details><summary>Model items not used ({len(read.discarded)})</summary>\n")
        out += [f"- {_code(q)} ({_code(kind)}): {why}" for q, kind, why in read.discarded]
        out.append("</details>")
    return out


def suggestion_choices(read: Optional[PatientRead]) -> list:
    """(suggestion index, wording index, button label) for each button slot, in order."""
    out = []
    if read is None:
        return out
    for i, s_ in enumerate(read.suggestions):
        # the part that tells the wordings apart goes first: "…; former smoker" vs "…; current smoker"
        common = 0
        if len(s_.wordings) > 1:
            parts = [w.split("; ") for w in s_.wordings]
            while all(len(p) > common + 1 and p[common] == parts[0][common] for p in parts):
                common += 1
        for j, wording in enumerate(s_.wordings):
            shown = ("…; " if common else "") + "; ".join(wording.split("; ")[common:])
            label = f"Use “{shown}” for “{s_.original}”"
            label = label if len(label) <= 90 else label[:87] + "…"
            if any(label == o[2] for o in out):
                label = f"{label[:80]} (choice {j + 1})"
            out.append((i, j, label))
    return out[:SUGGESTION_SLOTS]


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


def _atoms_text(built: BuiltPatient) -> str:
    """Every atom of this patient: `atoms` (what the LinAge2 space gets) plus the medication and the reported
    condition, which go only to the shared space — so the box, and the file to download, rebuild the patient fully."""
    extra = [ln for ln in built.shared_atoms.splitlines()
             if ln.startswith(("(CurrentMedication ", "(PatientCondition "))]
    return built.atoms + ("\n" + "\n".join(extra) if extra else "")


def _write_download(built: BuiltPatient, read_as: Optional[str] = None) -> str:
    """A .metta copy for the person to keep: a NEW temp file per build (never shared
    between sessions), in a temp directory — not a folder the KB loads. With the text
    the rules read, so the patient can be rebuilt from it by the rules alone."""
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
    if read_as is not None:
        header += (";; Read as (the fixed rules alone rebuild this patient from these lines):\n"
                   + "".join(f";;   {line}\n" for line in read_as.splitlines()))
    _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"pln_patient_{built.patient_id}_", suffix=".metta",
                                dir=_DOWNLOAD_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(header + _atoms_text(built) + "\n")
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
    ("No AgeAccelGrim measurement:", None),    # two variants, chosen below
    (NO_WITNESS_CHD_PREFIX,
     "Nothing you typed gives the supplement plan or the ranking a witness: none of your values is elevated and has "
     "a curated cause or effect (it has them for CRP, HbA1c, a fasting glucose, a fasting triglyceride, RDW, a low "
     "albumin and three DNA-methylation markers: PAI-1, GDF-15 and pack-years). The plan will have no tiers and the "
     "ranking will be the same as for anyone. The diagnosis answers from your reported heart disease alone, a "
     "prevalence item (\"ever told you had\"), not a measurement. That is \"nothing to work from\" for the plan "
     "and the ranking, not \"no cause\"."),
    (NO_WITNESS_PREFIX,
     "Nothing you typed gives the knowledge base a witness: none of your values is elevated "
     "and has a curated cause or effect (it has them for CRP, HbA1c, a fasting glucose, a fasting triglyceride, RDW, a low albumin "
     "and three DNA-methylation markers: PAI-1, GDF-15 and pack-years). So the diagnosis will come back empty, the "
     "supplement plan will have no tiers and the intervention ranking will be the same as for anyone. That is "
     "\"nothing to work from\", not \"no cause\". Diagnoses you listed, labs such as creatinine, blood "
     "pressure or cholesterol, values below normal and a glucose not marked fasting do not count "
     "there — LinAge2 still uses them."),
)

_NO_GRIM_TAB = ("No GrimAge acceleration given: the 10-year heart-disease risk model reads that "
                "clock and returns nothing for you (add a line such as 'GrimAge acceleration "
                "+3 years' if you have one). Ask about your heart risk anyway and you get the "
                "LinAge2 hazard, labelled all-cause mortality — it is not a heart risk.")


def _in_tab_words(note: str) -> str:
    for prefix, text in _TAB_WORDING:
        if note.startswith(prefix):
            if prefix == "No AgeAccelGrim measurement:":
                return _NO_GRIM_TAB + (" The diagnosis can still work from your elevated labs the "
                                       "knowledge base has edges for; the supplement plan and the "
                                       "ranking personalise only where a supplement or an "
                                       "intervention reaches them." if STILL_WORK in note else "")
            if text is not None:
                return text
            markers = note[len(prefix):].split(".")[0].strip()
            return (f"{markers}: turned into z-scores against one pooled reference (there is no "
                    f"age- and sex-specific table here), so the knowledge base's Elevated / "
                    f"Normal call on them is not adjusted for your age or sex.")
    return note.replace("measurement was sent", "measurement was given")


# ── handlers ──────────────────────────────────────────────────────────────────

def on_read(text: str) -> str:
    """The rules alone: the page's first reading and the example buttons."""
    return render_reading(read_patient_text(text or ""))


def rules_read(text: str) -> PatientRead:
    return read_patient(text or "")


def _extractor():
    """The model reader if configured; else one that says why not (rules only)."""
    return OpenAIExtractor()


def read_note() -> str:
    """What the Read button does with the text, said next to it."""
    model, error = configured_model()
    if error is None:
        return f"<sub>Read sends your text to OpenAI ({model}) to rewrite what the rules do not understand.</sub>"
    if error.code == "no_key":
        return "<sub>Read uses the fixed rules only (no OPENAI_API_KEY).</sub>"
    return f"<sub>Read uses the fixed rules only — model reader misconfigured: {error.message}</sub>"


def _button_updates(read: Optional[PatientRead]) -> list:
    choices = suggestion_choices(read)
    out = []
    for slot in range(SUGGESTION_SLOTS):
        if slot < len(choices):
            out.append(gr.update(value=choices[slot][2], visible=True))
        else:
            out.append(gr.update(visible=False))
    return out


def on_read_model(text: str, extractor=None):
    """The Read button: the rules, then the model (if configured) on what they missed.
    -> (reading, read state, *suggestion buttons)"""
    read = read_patient(text or "", extractor if extractor is not None else _extractor())
    return (render_reading(read.parsed, read), read, *_button_updates(read))


def on_example(name: str):
    """-> (text, reading, read state, *suggestion buttons): an example, by the rules."""
    read = rules_read(EXAMPLES[name])
    return (EXAMPLES[name], render_reading(read.parsed, read), read, *_button_updates(None))


def on_use_suggestion(text: str, read: Optional[PatientRead], slot: int) -> str:
    """Put the chosen wording in place of what was typed (the caller reads again)."""
    choices = suggestion_choices(read)
    if read is None or slot >= len(choices) or (text or "") != read.text:
        return text
    i, j, _ = choices[slot]
    s_ = read.suggestions[i]
    return s_.apply(text, s_.wordings[j])


def on_build(text: str, state: Optional[dict], read: Optional[PatientRead] = None):
    """-> (reading, summary, atoms, download, banner, state, suggestions visible)

    Builds from the reading made at Read (`read`); never calls the model. Without one
    (a caller that is not the tab), the rules read the text."""
    if read is None:
        read = rules_read(text)
    elif (text or "") != read.text:
        return (gr.update(), "### Not built\n✗ The text changed since Read — press **Read**, check "
                "what was read, then build.", gr.update(), gr.update(), render_banner(state), state,
                gr.update())
    parsed = read.parsed
    reading = render_reading(parsed, read)
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
    return (reading, summary, gr.update(value=_atoms_text(built), visible=True),
            gr.update(value=_write_download(built, read.read_as if read.substitutions else None),
                      visible=True), render_banner(payload), payload, gr.update(visible=True))


def on_clear():
    """-> (summary, atoms, download, banner, state, suggestions visible)"""
    return ("_No patient built._", gr.update(value="", visible=False), gr.update(visible=False),
            render_banner(None), None, gr.update(visible=False))


def on_show_questions(state: Optional[dict]):
    """-> (*the suggested-question buttons, the note on what is not offered), for the patient in
    `state`. Runs after Build; the patient is rebuilt from its payload (pure Python, no MeTTa)."""
    built = None
    if state:
        try:
            built = build_caller_patient(state, ())
        except PatientSpecError:
            built = None
    buttons = question_buttons(built)
    note = hidden_note(built)
    return (*[gr.update(value=q, visible=v) for q, v in buttons],
            gr.update(value=note, visible=bool(note)))


def build_tab(patient_state: gr.State, banner: gr.Markdown, question_box: gr.Textbox,
              tabs: gr.Tabs, query_tab_id: str) -> None:
    """Lay out the tab inside the caller's `gr.Tabs()` context and wire it."""
    gr.Markdown(
        "Describe yourself in a few lines — age, sex, smoking, any lab values with their "
        "units, diagnoses. **Read** shows what was understood — by fixed rules, and, if a model "
        "is configured, by a model that rewrites only what the rules missed (each rewrite is "
        "shown; values are always yours); **Build** scores LinAge2 "
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
            gr.Markdown(read_note())
            with gr.Column():
                use_btns = [gr.Button("", size="sm", visible=False) for _ in range(SUGGESTION_SLOTS)]
        with gr.Column(scale=3):
            first = rules_read(EXAMPLES["58-year-old smoker"])
            reading = gr.Markdown(render_reading(first.parsed, first))
    read_state = gr.State(first)
    summary = gr.Markdown("_No patient built._")
    with gr.Column(visible=False) as suggestions:
        with gr.Row():
            ask_btns = [gr.Button(q, size="sm") for q in SUGGESTED_QUESTIONS]
        not_offered = gr.Markdown(visible=False)
    with gr.Accordion("This patient as MeTTa atoms (this session only)", open=False):
        atoms = gr.Code(value="", language=None, interactive=False, visible=False,
                        label="Injected into each query's space; the LinAge2 atoms only into "
                              "the LinAge2 space, a medication only into the shared space")
        download = gr.DownloadButton("Download .metta", visible=False)

    for btn, name in zip(example_btns, EXAMPLES):
        btn.click(lambda n=name: on_example(n), outputs=[text, reading, read_state, *use_btns])
    read_btn.click(on_read_model, inputs=text, outputs=[reading, read_state, *use_btns],
                   concurrency_limit=4)
    for slot, btn in enumerate(use_btns):
        btn.click(lambda t, r, i=slot: on_use_suggestion(t, r, i), inputs=[text, read_state],
                  outputs=text).then(on_read_model, inputs=text, outputs=[reading, read_state, *use_btns],
                                     concurrency_limit=4)
    build_btn.click(on_build, inputs=[text, patient_state, read_state],
                    outputs=[reading, summary, atoms, download, banner, patient_state, suggestions]
                    ).then(on_show_questions, inputs=patient_state, outputs=[*ask_btns, not_offered])
    clear_btn.click(on_clear, outputs=[summary, atoms, download, banner, patient_state, suggestions])
    for btn in ask_btns:              # the label is the question: it changes with the patient
        btn.click(lambda q: (q, gr.Tabs(selected=query_tab_id)), inputs=btn,
                  outputs=[question_box, tabs])
