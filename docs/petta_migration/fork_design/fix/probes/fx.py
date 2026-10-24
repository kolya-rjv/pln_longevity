"""Common harness of the fix phase: production entry points only (core.pln_runner._runtime_run, run_drugage_ranking),
real PeTTa. Never reads pln_chat/.env (PYTHON_DOTENV_DISABLED)."""
import os, sys, time, tempfile, pathlib, json, re, itertools, random
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
REPO = pathlib.Path(__file__).resolve().parents[5]
#: where outputs and the regenerated DrugAge build go (set FIX_OUT to keep them; nothing is written under the repo)
FIX = pathlib.Path(os.environ["FIX_OUT"]) if os.environ.get("FIX_OUT") else pathlib.Path(tempfile.mkdtemp(prefix="pln_fix_"))
(FIX / "out").mkdir(parents=True, exist_ok=True)
BUILD = FIX / "build_out" / "drugage_etl.metta"      # written by mk_etl.py
sys.path.insert(0, str(REPO / "pln_chat"))
import config
from core import pln_chainer as pc, pln_runner

MIN = [REPO / f for f in ("system_types.metta", "logical_predicates.metta", "epistemic_calibration.metta", "pln_deduction.metta")]


def use(chainer="pln", runtime="petta", **kw):
    config.PLN_RUNTIME = runtime
    config.PLN_CHAINER = chainer
    config.LOGS_DIR = pathlib.Path(tempfile.mkdtemp())
    for k, v in kw.items():
        setattr(config, k, v)


def run(files, extra, queries):
    t = time.time()
    r = pln_runner._runtime_run("\n".join(queries), 0.0, files, extra)
    return r, time.time() - t


def effects(links):
    """links: (a, b, sign, s, c) -> Effect atoms"""
    return "\n".join(f"(Effect {a} {b} {sg} (stv {s} {c}))" for a, b, sg, s, c in links) + "\n"


def short(r, n=160):
    return {"status": r.status, "code": r.error_code, "error": (r.error or "")[:n].replace("\n", " "),
            "results": [a.atom for a in r.results][:12], "n": len(r.results)}


def layered(width, depth, fan, seed):
    rnd = random.Random(seed)
    names = [[f"L{d}N{i}" for i in range(width)] for d in range(depth)]
    links = []
    for d in range(depth - 1):
        for a in names[d]:
            for b in rnd.sample(names[d + 1], min(fan, width)):
                links.append((a, b, rnd.choice(["Pos", "Neg"]), round(rnd.uniform(0.2, 0.9), 2), round(rnd.uniform(0.4, 0.95), 2)))
    return names, links
