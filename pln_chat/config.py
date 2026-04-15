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
        import importlib
        return importlib.util.find_spec("hyperon") is not None
    except Exception:
        return False

PLN_RUNTIME_AVAILABLE: bool = _detect_hyperon()

# ── UI defaults ────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
SHOW_METTA_DEFAULT: bool = True
SHOW_EXPLANATION_DEFAULT: bool = True
SHOW_DEBUG_DEFAULT: bool = False
