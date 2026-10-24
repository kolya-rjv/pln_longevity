"""Shared helpers of the gate-G0 scripts (not production code; nothing in pln_chat imports this).

Relocatable: the repository root is found from this file's place (docs/petta_migration/fork_design/g0), the scratch
directory is $PLN_G0_SCRATCH or a fresh temporary directory, the interpreter is the one running the script.
python-dotenv is switched off before any repo module is imported, so pln_chat/.env is never read."""
import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
import sys, time, json, hashlib, re, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
try:
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
except Exception:
    pass
HERE = Path(__file__).resolve().parent
REPO = str(HERE.parents[3])
sys.path.insert(0, REPO + "/pln_chat")
sys.path.insert(0, REPO + "/docs/petta_migration/fork_design/sim")
VENDOR = REPO + "/vendor/pln_fork"
LIB_FULL = VENDOR + "/lib_pln_fork.metta"
LIB_RESTRICTED = VENDOR + "/lib_pln_restricted.metta"
CONTRACT = REPO + "/docs/petta_migration/fork_design/contract"
SCRATCH = os.environ.get("PLN_G0_SCRATCH") or tempfile.mkdtemp(prefix="pln_g0_")
os.makedirs(SCRATCH, exist_ok=True)
PY = sys.executable
PETTA_HOME = REPO + "/.petta/PeTTa"


def lib_text(which="restricted"):
    return open(LIB_RESTRICTED if which == "restricted" else LIB_FULL, encoding="utf-8").read()


def runtime():
    import config
    config.LOGS_DIR = Path(SCRATCH) / "logs"       # the runner would write logs into the work tree otherwise
    from core import metta_runtime as mr
    return mr.PettaRuntime()
