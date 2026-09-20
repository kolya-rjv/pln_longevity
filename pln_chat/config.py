import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# Primary ontology dir: the pln_longevity repo root (parent of pln_chat)
ONTOLOGY_DIR = BASE_DIR.parent
# Drop additional .metta files here to extend the knowledge base
CUSTOM_ONTOLOGY_DIR = BASE_DIR / "ontology" / "metta_files"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL: str = os.getenv("PLN_MODEL", "gpt-5.4-mini")
AVAILABLE_MODELS: list[str] = ["gpt-5.4-mini", "gpt-5.4", "gpt-4o", "gpt-4-turbo"]
DEFAULT_TEMPERATURE: float = 0.2
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_RETRIES: int = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "1")))

# ── PLN runtime ────────────────────────────────────────────────────────────────
# Auto-detected: true when the `hyperon` package is importable.
# Override with PLN_RUNTIME_AVAILABLE=false in .env to force stub mode.
def _detect_hyperon() -> bool:
    _env = os.getenv("PLN_RUNTIME_AVAILABLE", "").lower()
    if _env == "false":
        return False
    if _env == "true":
        return True
    try:
        import importlib.util  # NB: submodule must be imported explicitly
        return importlib.util.find_spec("hyperon") is not None
    except Exception:
        return False

PLN_RUNTIME_AVAILABLE: bool = _detect_hyperon()

# Max size (bytes) of a .metta file loaded into the hyperon runtime space.
# Files larger than this are SKIPPED at execution time: hyperon 0.2.10 panics
# (hyperon-space trie: "Option::unwrap() on None") — or silently mis-matches —
# once a space grows past a few thousand atoms, and the ~107 KB
# drugage_etl_short.metta dump trips it. A size cap excludes that (and any future
# oversized ETL output) generically while keeping every hand-written ontology /
# inference file (all < 10 KB). Such bulk data stays queryable in stub mode.
# Raise this (or set PLN_MAX_KB_FILE_BYTES) once the runtime handles larger spaces.
PLN_MAX_KB_FILE_BYTES: int = int(os.getenv("PLN_MAX_KB_FILE_BYTES", "60000"))

# ── DrugAge ranking ────────────────────────────────────────────────────────────
# Hard cap on the compound pool one ranking request may ask for. The MeTTa
# insertion sort behind `rank-interventions` is O(n^2) with a large constant
# (measured: n=10 -> 6.1 s), which is how one 35-compound request blocked the
# API for 115 s. The default `linear` strategy is ~70 ms per compound, so this
# cap bounds a single request at a few seconds rather than minutes.
PLN_MAX_RANK_COMPOUNDS: int = int(os.getenv("PLN_MAX_RANK_COMPOUNDS", "60"))
# Max DrugAge source rows echoed back in a ranking response (0 disables the
# per-row listing). Rapamycin alone has 37 rows, so an unbounded list makes a
# big response out of a small question.
PLN_MAX_RANK_ROWS: int = int(os.getenv("PLN_MAX_RANK_ROWS", "120"))
# Max entries in a request's `ontology_files` selection. The list is read and
# re-parsed as given, so repeats used to amplify work with no ceiling; the API
# now also de-duplicates it. There are ~30 .metta files in total, so 64 is
# generous.
PLN_MAX_ONTOLOGY_FILES: int = int(os.getenv("PLN_MAX_ONTOLOGY_FILES", "64"))

# ── PLN execution workers ──────────────────────────────────────────────────────
# hyperon 0.2.10 HOLDS THE GIL for the whole of MeTTa.run() (measured: two runs
# in two threads take exactly as long as two runs in sequence, ratio 0.998), so
# a MeTTa query starves the event loop and every other request with it — one
# 35-compound ranking blocked the whole API for 115 s. A thread pool cannot fix
# that; a process pool can, and it also makes a per-request timeout enforceable
# and contains hyperon's non-unwinding Rust abort (which kills the interpreter
# in-process but only a worker out-of-process).
#   0 = run inline, in the request thread (the pre-2026-09 behaviour; used by
#       the in-process contract tests, which monkeypatch module globals a child
#       process could never see).
PLN_WORKER_POOL_SIZE: int = max(0, int(os.getenv("PLN_WORKER_POOL_SIZE", "2")))
# Per-request PLN budget in seconds. 0 disables. 60 s is deliberately generous:
# a legitimate 20-compound ranking is a few seconds, so this only catches the
# pathological shapes.
PLN_QUERY_TIMEOUT_SECONDS: float = float(os.getenv("PLN_QUERY_TIMEOUT_SECONDS", "60"))
# Max PLN tasks queued or running before new ones are refused with 503.
# 0 = derive it (4x the worker count).
PLN_MAX_INFLIGHT_QUERIES: int = max(0, int(os.getenv("PLN_MAX_INFLIGHT_QUERIES", "0")))
# Recycle a worker after this many tasks (0 = never). A fresh hyperon
# interpreter is cheap and does not inherit a previous query's space growth.
PLN_WORKER_MAX_TASKS: int = max(0, int(os.getenv("PLN_WORKER_MAX_TASKS", "50")))

# ── UI defaults ────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0
SHOW_METTA_DEFAULT: bool = True
SHOW_EXPLANATION_DEFAULT: bool = True
SHOW_DEBUG_DEFAULT: bool = False

# ── HTTP API (api.py) ────────────────────────────────────────────────────────
# Standalone API-only listener (`python api.py`). The normal combined listener
# below serves both the Gradio UI and these routes from one origin.
PLN_API_HOST: str = os.getenv("PLN_API_HOST", "0.0.0.0")
PLN_API_PORT: int = int(os.getenv("PLN_API_PORT", "8000"))
# Combined Gradio + API server used by `python app.py`. Defaults preserve the
# original Gradio listener so an existing ngrok tunnel to port 7860 keeps working.
PLN_SERVER_HOST: str = os.getenv("PLN_SERVER_HOST", "127.0.0.1")
PLN_SERVER_PORT: int = int(os.getenv("PLN_SERVER_PORT", "7860"))
PLN_CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("PLN_CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]
