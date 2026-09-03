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
