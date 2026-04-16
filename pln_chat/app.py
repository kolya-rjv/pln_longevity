"""PLN Natural Language Query Interface — Gradio entry point."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the pln_chat package root is on sys.path so submodule imports work
# whether the file is run directly or via `python -m`.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

from config import (
    AVAILABLE_MODELS,
    CUSTOM_ONTOLOGY_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ONTOLOGY_DIR,
    SHOW_DEBUG_DEFAULT,
    SHOW_EXPLANATION_DEFAULT,
    SHOW_METTA_DEFAULT,
)
from ontology.loader import load_specific_files
from ontology.registry import OntologyRegistry, BUILTIN_REGISTRY
from ontology.expander import run_expansion_pipeline
from core.context_builder import build_system_prompt
from core.llm_translator import translate
from core.metta_validator import validate
from core.pln_runner import run_query
from utils.formatting import format_bot_response
from utils.logging import log_turn


# ── Discover available .metta files ───────────────────────────────────────────

def _discover_metta_files() -> dict[str, Path]:
    """Return {filename: absolute_path} for every .metta file in the project."""
    files: dict[str, Path] = {}
    search_roots = [
        ONTOLOGY_DIR,         # pln_longevity/ repo root (primary)
        CUSTOM_ONTOLOGY_DIR,  # drop-in directory for extra files
    ]
    for base in search_roots:
        if base.exists():
            for p in sorted(base.glob("*.metta")):
                files[p.name] = p
    return files


_METTA_FILES = _discover_metta_files()
_ONTOLOGY_CHOICES = list(_METTA_FILES.keys()) or ["(no .metta files found)"]
_DEFAULT_SELECTION = (
    [k for k in _ONTOLOGY_CHOICES if "epistemic" in k.lower()]
    or _ONTOLOGY_CHOICES[:1]
)


# ── Registry helper ────────────────────────────────────────────────────────────

# All KB files available for execution — always the complete set.
_ALL_KB_PATHS: list[Path] = list(_METTA_FILES.values())


def _build_context(selected_files: list[str]) -> tuple[OntologyRegistry, dict[str, str]]:
    """Load selected .metta files for the LLM system prompt context.

    The UI selection controls what the LLM sees (system prompt / symbol index).
    Execution always uses the full KB (_ALL_KB_PATHS) regardless of selection.
    """
    paths = [_METTA_FILES[f] for f in selected_files if f in _METTA_FILES]
    if paths:
        registry, raw_contents = load_specific_files(paths)
        return registry, raw_contents
    return BUILTIN_REGISTRY, {}


# ── Core chat handler ──────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    selected_files: list[str],
    model: str,
    temperature: float,
    confidence_threshold: float,
    show_metta: bool,
    show_explanation: bool,
    show_debug: bool,
) -> tuple[list[dict], str]:
    if not user_message.strip():
        return history, ""

    registry, raw_contents = _build_context(selected_files)
    system_prompt = build_system_prompt(registry, raw_contents)

    # Gradio 6 history is already a list of {role, content} dicts
    history_msgs: list[dict] = list(history)

    translation = translate(
        user_message=user_message,
        system_prompt=system_prompt,
        history=history_msgs,
        model=model,
        temperature=temperature,
    )

    # ─── TEMPORARY DEBUG ───
    print("=" * 60)
    print("GENERATED METTA QUERY:")
    print(repr(translation.metta_query))
    print("=" * 60)
    # ─── END DEBUG ─────────

    validation = validate(translation.metta_query, registry)  # uses parsed registry for symbol checks

    pln_result = run_query(
        metta_query=translation.metta_query,
        confidence_threshold=confidence_threshold,
        kb_files=_ALL_KB_PATHS,
    )

    bot_response = format_bot_response(
        translation=translation,
        validation=validation,
        pln_result=pln_result,
        show_metta=show_metta,
        show_explanation=show_explanation,
        show_debug=show_debug,
    )

    log_turn(user_message, translation, pln_result)

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": bot_response},
    ]
    return history, ""


# ── Ontology Expander helpers ──────────────────────────────────────────────────

def _expander_target_choices() -> list[str]:
    """Collect all known .metta target paths, adding a 'Create new…' sentinel."""
    choices = list(_METTA_FILES.keys())
    choices.append("(create new file…)")
    return choices


_EXPANDER_CHOICES = _expander_target_choices()


def _resolve_target_path(dropdown_val: str, new_name_val: str) -> Path:
    """Return an absolute Path for the chosen target .metta file."""
    if dropdown_val == "(create new file…)":
        stem = (new_name_val or "expanded_ontology").strip().rstrip(".metta")
        stem = stem if stem else "expanded_ontology"
        # Write new files into CUSTOM_ONTOLOGY_DIR
        CUSTOM_ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
        return CUSTOM_ONTOLOGY_DIR / f"{stem}.metta"
    return _METTA_FILES.get(dropdown_val, ONTOLOGY_DIR / dropdown_val)


def run_extraction(
    paper_file,          # gr.File value: dict with 'name' (tmp path) and 'orig_name'
    paper_text: str,
    target_dropdown: str,
    new_filename: str,
    exp_model: str,
    exp_temperature: float,
) -> tuple[str, str, str, dict]:
    """Handler for the 'Extract from Paper' button.

    Returns
    -------
    status_md : str   — Markdown status message shown to the user.
    new_preview : str — Proposed MeTTa block (for the code preview box).
    dup_preview : str — Human-readable list of duplicate entries.
    state : dict      — Serialisable state passed to the apply handler.
    """
    # Determine input source: uploaded file takes priority over pasted text
    if paper_file is not None:
        tmp_path = Path(paper_file["name"] if isinstance(paper_file, dict) else paper_file.name)
        orig_name = (
            paper_file.get("orig_name", tmp_path.name)
            if isinstance(paper_file, dict)
            else getattr(paper_file, "orig_name", tmp_path.name)
        )
        try:
            raw_bytes = tmp_path.read_bytes()
        except OSError as exc:
            return f"**Error reading uploaded file:** {exc}", "", "", {}
    elif paper_text.strip():
        raw_bytes = paper_text.encode("utf-8")
        orig_name = "pasted_text.txt"
    else:
        return "**Please upload a file or paste paper text before extracting.**", "", "", {}

    target_path = _resolve_target_path(target_dropdown, new_filename)

    result = run_expansion_pipeline(
        paper_data=raw_bytes,
        filename=orig_name,
        target_file_path=target_path,
        model=exp_model,
        temperature=exp_temperature,
        apply=False,
    )

    if not result.ok:
        return f"**Extraction error:** {result.error}", "", "", {}

    n_new = len(result.new_entries)
    n_dup = len(result.duplicate_entries)

    status_lines = [
        f"### Extraction complete",
        f"**Paper:** {result.paper_title}",
        f"> {result.paper_summary}" if result.paper_summary else "",
        f"",
        f"- **{n_new}** new entries ready to add",
        f"- **{n_dup}** duplicate(s) skipped (already in ontology)",
        f"- Target file: `{target_path.name}`",
    ]
    if n_new == 0:
        status_lines.append(
            "\n_No new entries found — everything extracted already exists in the ontology._"
        )

    dup_lines: list[str] = []
    for e in result.duplicate_entries:
        dup_lines.append(f"[{e.kind}] {e.name}")
        if e.description:
            dup_lines.append(f"  → {e.description}")
        dup_lines.append("")

    state = {
        "metta_block": result.metta_block,
        "target_path":  str(target_path),
        "n_new":  n_new,
        "n_dup":  n_dup,
        "paper_title": result.paper_title,
    }

    return (
        "\n".join(l for l in status_lines if l is not None),
        result.metta_block,
        "\n".join(dup_lines) if dup_lines else "(none)",
        state,
    )


def apply_to_ontology(state: dict) -> tuple[str, gr.Button]:
    """Handler for the 'Apply to Ontology' button.

    Writes the previously-extracted MeTTa block to disk.
    """
    if not state or not state.get("metta_block"):
        return "**Nothing to apply.** Run extraction first.", gr.update(interactive=False)

    metta_block: str = state["metta_block"]
    target_path = Path(state["target_path"])

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as fh:
            if target_path.exists() and target_path.stat().st_size > 0:
                fh.write("\n\n")
            fh.write(metta_block)
    except OSError as exc:
        return f"**Write error:** {exc}", gr.update(interactive=True)

    n = state.get("n_new", "?")
    return (
        f"**Applied successfully!** {n} new entr{'y' if n == 1 else 'ies'} "
        f"written to `{target_path.name}`.",
        gr.update(interactive=False),
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="PLN Natural Language Query Interface",
) as demo:
    gr.Markdown(
        "# PLN Natural Language Query Interface\n"
        "Ask questions about the longevity knowledge base in plain English, "
        "or expand the PLN ontology from a research paper."
    )

    with gr.Tabs():

        # ════════════════════════════════════════════════════════════════════
        # Tab 1 — PLN Query chat
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("PLN Query"):
            with gr.Row():
                # ── Main chat panel ────────────────────────────────────────
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=540,
                        buttons=["copy"],
                        render_markdown=True,
                    )
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask a question about the PLN knowledge base…",
                            label="",
                            scale=8,
                            lines=1,
                            max_lines=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

                # ── Sidebar ────────────────────────────────────────────────
                with gr.Column(scale=1, min_width=270):
                    gr.Markdown("### Ontology Scope")
                    ontology_checkboxes = gr.CheckboxGroup(
                        choices=_ONTOLOGY_CHOICES,
                        value=_DEFAULT_SELECTION,
                        label="Active .metta files",
                        info="Only selected files are injected into the LLM context.",
                    )

                    gr.Markdown("### LLM Settings")
                    model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,
                        label="Model",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=DEFAULT_TEMPERATURE,
                        label="Temperature",
                        info="Lower = more deterministic MeTTa output.",
                    )

                    gr.Markdown("### Result Filtering")
                    confidence_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=DEFAULT_CONFIDENCE_THRESHOLD,
                        label="Min confidence threshold",
                        info="PLN results with lower confidence are hidden.",
                    )

                    gr.Markdown("### Display Options")
                    show_metta_toggle       = gr.Checkbox(value=SHOW_METTA_DEFAULT,       label="Show MeTTa query")
                    show_explanation_toggle = gr.Checkbox(value=SHOW_EXPLANATION_DEFAULT, label="Show explanation")
                    show_debug_toggle       = gr.Checkbox(value=SHOW_DEBUG_DEFAULT,        label="Show debug info")

            # ── Chat event wiring ──────────────────────────────────────────
            _chat_inputs = [
                user_input, chatbot,
                ontology_checkboxes, model_dropdown, temperature_slider,
                confidence_slider, show_metta_toggle, show_explanation_toggle, show_debug_toggle,
            ]
            _chat_outputs = [chatbot, user_input]

            send_btn.click(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs)
            user_input.submit(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs)
            clear_btn.click(fn=lambda: ([], ""), outputs=_chat_outputs)

        # ════════════════════════════════════════════════════════════════════
        # Tab 2 — Ontology Expander
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Ontology Expander"):
            gr.Markdown(
                "## Expand PLN Ontology from a Research Paper\n"
                "Upload a paper (PDF or plain text) or paste an abstract. "
                "An LLM will extract new types, functions, rules, and facts for the "
                "PLN knowledge base, skipping anything already present in the ontology."
            )

            with gr.Row():
                # ── Input panel ────────────────────────────────────────────
                with gr.Column(scale=2):
                    paper_upload = gr.File(
                        label="Upload paper (PDF / .txt / .metta)",
                        file_types=[".pdf", ".txt", ".metta"],
                        file_count="single",
                    )
                    paper_text_input = gr.Textbox(
                        label="Or paste paper text / abstract here",
                        lines=8,
                        placeholder=(
                            "Paste the abstract or full text of the paper as an alternative "
                            "to uploading a file…"
                        ),
                    )

                # ── Config panel ───────────────────────────────────────────
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### Target File")
                    exp_target_dropdown = gr.Dropdown(
                        choices=_EXPANDER_CHOICES,
                        value=_EXPANDER_CHOICES[0] if _EXPANDER_CHOICES else None,
                        label="Append to .metta file",
                        info="New entries will be appended to this file.",
                    )
                    exp_new_filename = gr.Textbox(
                        label="New filename (without .metta extension)",
                        placeholder="e.g. rapamycin_study",
                        visible=False,
                    )

                    gr.Markdown("### LLM Settings")
                    exp_model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,
                        label="Model",
                    )
                    exp_temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=0.1,
                        label="Temperature",
                        info="Lower = more consistent MeTTa syntax.",
                    )

            # Show/hide the "new filename" box based on dropdown selection
            def _toggle_new_filename(choice: str):
                return gr.update(visible=(choice == "(create new file…)"))

            exp_target_dropdown.change(
                fn=_toggle_new_filename,
                inputs=exp_target_dropdown,
                outputs=exp_new_filename,
            )

            # ── Action buttons ─────────────────────────────────────────────
            with gr.Row():
                exp_extract_btn = gr.Button("Extract from Paper", variant="primary")
                exp_apply_btn   = gr.Button("Apply to Ontology",  variant="secondary",
                                            interactive=False)
                exp_clear_btn   = gr.Button("Clear", variant="stop")

            exp_status = gr.Markdown(
                "_Upload a paper (or paste text) and click **Extract from Paper** to begin._"
            )

            # ── Preview panels ─────────────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Proposed New Entries")
                    exp_new_preview = gr.Textbox(
                        label="Generated MeTTa (review before applying)",
                        lines=20,
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("### Skipped — Already in Ontology")
                    exp_dup_preview = gr.Textbox(
                        label="Duplicate entries (not written)",
                        lines=20,
                        interactive=False,
                    )

            # Holds serialisable result between Extract and Apply steps
            exp_state = gr.State(value={})

            # ── Expander event wiring ──────────────────────────────────────
            _exp_extract_inputs = [
                paper_upload, paper_text_input,
                exp_target_dropdown, exp_new_filename,
                exp_model_dropdown, exp_temperature_slider,
            ]
            _exp_extract_outputs = [exp_status, exp_new_preview, exp_dup_preview, exp_state]

            def _on_extract(*args):
                status, new_prev, dup_prev, state = run_extraction(*args)
                apply_interactive = bool(state.get("metta_block"))
                return (
                    status,
                    new_prev,
                    dup_prev,
                    state,
                    gr.update(interactive=apply_interactive),
                )

            exp_extract_btn.click(
                fn=_on_extract,
                inputs=_exp_extract_inputs,
                outputs=_exp_extract_outputs + [exp_apply_btn],
            )

            exp_apply_btn.click(
                fn=apply_to_ontology,
                inputs=exp_state,
                outputs=[exp_status, exp_apply_btn],
            )

            exp_clear_btn.click(
                fn=lambda: (
                    "_Upload a paper (or paste text) and click **Extract from Paper** to begin._",
                    "", "", {}, gr.update(interactive=False),
                ),
                outputs=[exp_status, exp_new_preview, exp_dup_preview, exp_state, exp_apply_btn],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
