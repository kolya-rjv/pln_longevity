"""The Gradio UI and the HTTP API must share the same hardening.

They are one process: `create_combined_app()` mounts Gradio on the FastAPI app,
and hyperon holds the GIL for the whole of `MeTTa.run()`. So every protection
added to the API — the worker pool, the per-request deadline, the admission
limit, the abort containment — applied to only half the traffic while the UI's
chat handler still called `run_query` and `route_drugage_ranking` inline. A
query typed into the UI froze every REST caller, exactly as the evaluation's
35-compound ranking froze `/health`.

The same is true of the fixes that are about honesty rather than load: the UI
validated against the symbol registry alone (so it inherited both the
false-positive and the false-negative halves of that bug) and pasted every
selected file into the prompt verbatim (so the 417,000-token failure was still
reachable through the file picker).

gradio is not installed in CI, so these assertions read app.py's source with
`ast` rather than importing it. A textual guard is weaker than an executable
one; it is here because the alternative is no guard at all.

Run from the repository root:
    pytest tests/test_ui_api_parity.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "pln_chat" / "app.py"
API = REPO / "pln_chat" / "api.py"


@pytest.fixture(scope="module")
def app_source() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_tree(app_source: str) -> ast.Module:
    return ast.parse(app_source)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in app.py")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


# ── load: the UI must not block the API ──────────────────────────────────────

def test_the_chat_handler_runs_metta_through_the_worker_pool(app_tree):
    chat = _function(app_tree, "chat")
    called = _called_names(chat)
    assert "run_offloaded" in called, (
        "app.chat must route PLN execution through core.executor, or a UI query "
        "starves every REST caller in the same process"
    )


def test_the_chat_handler_makes_no_bare_metta_call(app_tree):
    """`run_query` / `route_drugage_ranking` may appear only inside the thunk."""
    chat = _function(app_tree, "chat")
    bare: list[str] = []
    for node in ast.walk(chat):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name not in {"run_query", "route_drugage_ranking"}:
            continue
        # A call inside a lambda is the inline fallback run_offloaded takes when
        # the pool is disabled — that one is correct.
        inside_lambda = any(
            isinstance(parent, ast.Lambda) and node in ast.walk(parent)
            for parent in ast.walk(chat)
            if isinstance(parent, ast.Lambda)
        )
        if not inside_lambda:
            bare.append(name)
    assert not bare, f"unoffloaded MeTTa call(s) in app.chat: {bare}"


def test_the_chat_handler_reports_an_execution_failure_as_a_failure(app_tree, app_source):
    """A UI cannot return 504, but it must not render a timeout as 'no results'.

    Read off the handler's own `except` clause, not scanned for in the file.
    The three names are imported at the top of app.py, so a substring check was
    satisfied by the import block alone — narrowing the except tuple to one
    exception would have left it green while the UI silently stopped handling
    the other two.
    """
    chat = _function(app_tree, "chat")
    caught: set[str] = set()
    for node in ast.walk(chat):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        kinds = (
            node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        )
        caught.update(k.id for k in kinds if isinstance(k, ast.Name))
    assert {"PLNExecutionTimeout", "PLNOverloaded", "PLNWorkerCrashed"} <= caught, (
        f"app.chat does not catch every execution failure; it catches {sorted(caught)}"
    )
    assert 'status="error"' in app_source


# ── honesty: the UI must see the same knowledge base the API does ────────────

def test_the_ui_validates_against_the_runtime_inventory(app_tree, app_source):
    chat = _function(app_tree, "chat")
    for node in ast.walk(chat):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "validate":
            assert len(node.args) >= 3, (
                "app.chat must pass the runtime inventory to validate(), or the "
                "UI keeps reporting real symbols as unknown and waving empty "
                "predicates through"
            )
            break
    else:
        raise AssertionError("no validate() call found in app.chat")


def test_the_ui_prompt_carries_the_grounded_schema_card(app_tree):
    chat = _function(app_tree, "chat")
    for node in ast.walk(chat):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "build_system_prompt":
            assert len(node.args) >= 3, "the UI prompt must include the inventory"
            return
    raise AssertionError("no build_system_prompt() call found in app.chat")


def test_the_ui_summarises_an_oversized_file_like_the_api_does(app_tree):
    build_context = _function(app_tree, "_build_context")
    assert "summarise_oversized" in _called_names(build_context), (
        "selecting a gene ETL file in the UI's picker must not rebuild the "
        "417,000-token prompt the API was fixed for"
    )


# ── the two modules must not drift ───────────────────────────────────────────

def test_the_inference_stack_is_identical_in_both_modules(app_source):
    """_INFERENCE_STACK is duplicated verbatim; it must stay that way."""

    def stack_of(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == "_INFERENCE_STACK":
                return [ast.literal_eval(e) for e in node.value.elts]
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_INFERENCE_STACK" for t in node.targets
            ):
                return [ast.literal_eval(e) for e in node.value.elts]
        raise AssertionError(f"_INFERENCE_STACK not found as a literal in {path.name}")

    assert stack_of(APP) == stack_of(API)


def test_no_debug_printing_of_user_queries_remains(app_source):
    """The handler used to print every generated query to stdout."""
    assert "TEMPORARY DEBUG" not in app_source
    assert "GENERATED METTA QUERY" not in app_source
