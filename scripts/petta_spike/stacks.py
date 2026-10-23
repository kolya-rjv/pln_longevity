"""Each runtime stack's file order, exactly as the repo defines it.

Primary path: IMPORT `core.pln_runner` and `api` from the work tree (import only, no query is run
from here; importing `api` takes under a second and writes nothing). `pln_chat/` is the sys.path root, as in the
repo's own tests: its modules import `config`, `core.*` and `ontology.*` as top-level names, so `pln_chat.core.pln_runner`
would not import. Because pln_chat/config.py calls
load_dotenv() at import, `repo_import()` first switches python-dotenv off, so pln_chat/.env is never read.

Fallback path (`--source parse`): parse the same lists out of the source with `ast`, for a machine where
the import is heavy or needs environment. `check_import_vs_source()` compares the two.

Stacks (names are the kit's, files are the repo's):
  generic         pln_runner.generic_stack(runtime)         programs that read no patient
  patient         pln_runner.patient_stack(runtime)         programs that name a patient
  human_evidence  pln_runner.human_evidence_stack(runtime)  only (human-evidence …) forms
  linage2         pln_runner.linage2_patient_kb()           the LinAge2 clinical-clock forms
  drugage_slice   pln_runner.DRUGAGE_STACK + a DrugAge row slice (tests/fixtures/drugage_real_rows.metta)
  cellage_slice   pln_runner.CELLAGE_STACK + a CellAge row slice (tests/fixtures/cellage_real_rows.metta)
  full            api._runtime_kb_paths()                   the shared stack (hyperon aborts on it)
  nhanes_patient  pln_runner.nhanes_patient_kb()            the NHANES grounding stack
  smoke10         the ten files named in pln_deduction.metta section 5
  fork_pln        patham9/PLN src/ in build.sh order        comparison only; needs a fork checkout
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from kit_common import petta_home, repo_root, sha256_text

#: the hyperon-era byte cap on a file of the runtime stack (pln_chat/config.py PLN_MAX_KB_FILE_BYTES default)
DEFAULT_MAX_KB_FILE_BYTES = 60000
CALLER_ID = "W58"
DRUGAGE_SLICE_COMPOUNDS = ["Rapamycin", "Acarbose", "Resveratrol", "Metformin"]
CELLAGE_SLICE_GENES = ["TP53", "CDKN1A", "CDKN2A"]


@dataclass
class StackSpec:
    name: str
    description: str
    files: list[str]                              # paths relative to the repo root (absolute when outside it), in load order
    extra_atoms: str = ""                         # appended after the files (kit-added: a row slice or a caller patient)
    extra_label: str = ""
    canary: list[str] = field(default_factory=list)   # '!' expressions that must answer after a good load
    source: str = "import"                        # import | parsed
    notes: list[str] = field(default_factory=list)

    def read_texts(self, repo: Optional[Path] = None) -> list[tuple[str, str]]:
        repo = repo or repo_root()
        out = []
        for f in self.files:
            p = Path(f)
            p = p if p.is_absolute() else repo / p
            out.append((f, p.read_text(encoding="utf-8")))
        return out

    def kb_text(self, repo: Optional[Path] = None, fixes=None) -> str:
        """The text pln_runner._hyperon_run would load: the files' contents, then the extra atoms, joined by newlines."""
        texts = self.read_texts(repo)
        if fixes:
            from kb_fixes import fix_kb_texts
            texts, _ = fix_kb_texts(texts, fixes)
        blocks = [t for _, t in texts]
        if self.extra_atoms:
            blocks.append(self.extra_atoms)
        return "\n".join(blocks)

    def summary(self) -> dict:
        d = asdict(self)
        d["extra_atoms_bytes"] = len(self.extra_atoms)
        d["extra_atoms_sha256"] = sha256_text(self.extra_atoms) if self.extra_atoms else ""
        d.pop("extra_atoms")
        return d


# ── repo import (read-only) ──────────────────────────────────────────────────

_IMPORTED = False


def repo_import() -> None:
    """Make `import api`, `import core.pln_runner` work against the work tree without side effects."""
    global _IMPORTED
    if _IMPORTED:
        return
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"        # python-dotenv >= 1.1: load_dotenv() becomes a no-op
    sys.dont_write_bytecode = True                    # never write __pycache__ into the work tree
    try:                                              # and for older python-dotenv, make it a no-op by hand
        import dotenv
        dotenv.load_dotenv = lambda *a, **k: False
    except Exception:
        pass
    chat = str(repo_root() / "pln_chat")
    if chat not in sys.path:
        sys.path.insert(0, chat)
    import config                                     # noqa: F401  (pln_chat/config.py)
    # the runner writes pln_chat/logs/last_query.metta on every run; keep that out of the work tree
    from kit_common import out_dir
    logs = out_dir() / "hyperon_logs"
    config.LOGS_DIR = logs
    _IMPORTED = True


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(repo_root()))
    except ValueError:
        return str(p)


# ── the smoke queries and their load order, read out of pln_deduction.metta section 5 ──

def parse_smoke(repo: Optional[Path] = None) -> tuple[list[str], list[dict]]:
    """(load order as file names, [{'expr': '!(…)', 'expected': '(…)'}, …]) from the section-5 comment block."""
    repo = repo or repo_root()
    lines = (repo / "pln_deduction.metta").read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if "5. Smoke-test queries" in l)
    except StopIteration:
        raise RuntimeError("pln_deduction.metta: section 5 'Smoke-test queries' not found")
    order: list[str] = []
    i = start
    while i < len(lines) and ";; Load order:" not in lines[i]:
        i += 1
    if i < len(lines):
        order += re.findall(r"[A-Za-z0-9_]+", lines[i].split("Load order:")[1])
        i += 1
        while i < len(lines) and re.match(r"^;;\s{5,}\S", lines[i]):
            order += re.findall(r"[A-Za-z0-9_]+", lines[i][2:])
            i += 1
    queries: list[dict] = []
    while i < len(lines):
        m = re.match(r"^;; (!\()", lines[i])
        if m:
            text = lines[i][3:]
            depth, end = 0, None
            for k, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = k + 1
                        break
            expr = text[:end]
            rest = text[end:]
            exp = rest.split(";", 1)[1].strip() if ";" in rest else ""
            j = i + 1
            while j < len(lines) and re.match(r"^;;\s+;\s*\S", lines[j]):
                exp += " " + re.match(r"^;;\s+;\s*(.*)$", lines[j]).group(1).strip()
                j += 1
            queries.append({"expr": expr, "expected": re.sub(r"\s+", " ", exp).strip()})
            i = j
            continue
        i += 1
    return [f"{n}.metta" for n in order], queries


# ── building the stacks by IMPORT ────────────────────────────────────────────

def _caller_patient():
    """The synthetic caller of tests/test_linage2.py: Caller_W58, 33 measured and 26 imputed LinAge2 inputs."""
    import json
    from core.patient_builder import build_patient
    fx = json.loads((repo_root() / "tests" / "fixtures" / "linage2_response.json").read_text(encoding="utf-8"))
    return build_patient({"id": CALLER_ID, "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
                          "markers": {"HbA1c": 1.6, "CRP": 0.3}, "linage2": fx})


def caller_patient():
    repo_import()
    return _caller_patient()


def drugage_slice(compounds=None, source: Optional[Path] = None):
    """(slice text, rows) from the committed real-rows fixture, one best row per compound."""
    repo_import()
    from ontology.drugage_selector import build_drugage_slice
    src = source or (repo_root() / "tests" / "fixtures" / "drugage_real_rows.metta")
    return build_drugage_slice(compounds or DRUGAGE_SLICE_COMPOUNDS, best_per_compound=True, source=src)


def cellage_slice(genes=None, source: Optional[Path] = None):
    repo_import()
    from ontology.cellage_selector import build_cellage_slice
    src = source or (repo_root() / "tests" / "fixtures" / "cellage_real_rows.metta")
    return build_cellage_slice(genes or CELLAGE_SLICE_GENES, source=src)


def _fork_files() -> list[str]:
    root = petta_home() / "repos" / "PLN" / "src"
    # build.sh: cat Utils Constraints Formulas Rules Config Deriver > ../PLN.metta
    return [str(root / f"{n}.metta") for n in ("Utils", "Constraints", "Formulas", "Rules", "Config", "Deriver")]


def stacks_from_import() -> dict[str, StackSpec]:
    repo_import()
    import api
    from core import pln_runner as R

    rt = api._runtime_kb_paths()
    smoke_order, smoke_q = parse_smoke()
    smoke_exprs = [q["expr"] for q in smoke_q]
    generic_canary = ["!(infer &self CellularSenescence CoronaryHeartDisease)",
                      "!(match &self (TargetsHallmark $i $h) (pair $i $h))"]
    patient_canary = ["!(patient-status &self Patient001 DNAmPAI1)", "!(diagnose-patient &self Patient001)"]

    caller = _caller_patient()
    da_text, da_rows = drugage_slice()
    ca_text, ca_rows = cellage_slice()

    S: dict[str, StackSpec] = {}
    S["smoke10"] = StackSpec("smoke10", "the ten files pln_deduction.metta section 5 names, in its load order",
                             smoke_order, canary=smoke_exprs, source="parsed",
                             notes=["file list parsed from the comment block, not from Python"])
    S["generic"] = StackSpec("generic", "runtime stack minus the four patient layers (programs that read no patient)",
                             [_rel(p) for p in R.generic_stack(rt)], canary=generic_canary,
                             notes=[f"excluded: {', '.join(R.GENERIC_STACK_EXCLUDED)}"])
    S["patient"] = StackSpec("patient", "runtime stack minus seven files no patient form reads",
                             [_rel(p) for p in R.patient_stack(rt)], canary=patient_canary,
                             notes=[f"excluded: {', '.join(R.PATIENT_STACK_EXCLUDED)}"])
    S["human_evidence"] = StackSpec("human_evidence", "four files, for programs made only of (human-evidence …) forms",
                                    [_rel(p) for p in R.human_evidence_stack(rt)],
                                    canary=["!(human-evidence &self Metformin)"])
    S["linage2"] = StackSpec("linage2", "the LinAge2 scoped stack (+ the generated baseline when build/ has it)",
                             [_rel(p) for p in R.linage2_patient_kb()],
                             extra_atoms=caller.atoms, extra_label=f"caller patient Caller_{CALLER_ID} (tests/fixtures/linage2_response.json)",
                             canary=[f"!(linage-hazard-patient &self Caller_{CALLER_ID})"],
                             notes=["extra_atoms is kit-added: the LinAge2 forms read a caller's LinAgeDelta atoms"])
    S["drugage_slice"] = StackSpec("drugage_slice", "DRUGAGE_STACK + a row slice (one best row per compound)",
                                   [_rel(p) for p in R.DRUGAGE_STACK],
                                   extra_atoms=da_text,
                                   extra_label=f"{len(da_rows)} DrugAge rows of tests/fixtures/drugage_real_rows.metta ({', '.join(sorted({r.compound for r in da_rows}))})",
                                   canary=[f"!(score-candidate &self {c} Mortality)" for c in sorted({r.compound for r in da_rows})[:2]],
                                   notes=["build/drugage_etl.metta is absent from the work tree (gitignored), so the committed fixture is the slice source"])
    S["cellage_slice"] = StackSpec("cellage_slice", "CELLAGE_STACK + a row slice",
                                   [_rel(p) for p in R.CELLAGE_STACK],
                                   extra_atoms=ca_text,
                                   extra_label=f"{len(ca_rows)} CellAge rows of tests/fixtures/cellage_real_rows.metta ({', '.join(CELLAGE_SLICE_GENES)})",
                                   canary=[f"!(cellage-effect &self {ca_rows[0].row_id})"] if ca_rows else [])
    S["full"] = StackSpec("full", "api._runtime_kb_paths(): the shared stack, which aborts hyperon 0.2.10 on some programs",
                          [_rel(p) for p in rt], canary=generic_canary,
                          notes=["on hyperon, programs that read back a late-stored atom abort the process (trie.rs:179); PeTTa has no such cap"])
    S["nhanes_patient"] = StackSpec("nhanes_patient", "NHANES grounding stack (+ build/ outputs when present)",
                                    [_rel(p) for p in R.nhanes_patient_kb()],
                                    canary=["!(patient-status &self Patient001 DNAmPAI1)"])
    fork = _fork_files()
    S["fork_pln"] = StackSpec("fork_pln", "patham9/PLN src/ in build.sh order (comparison; the checkout is read, never run)",
                              fork, canary=[], source="parsed",
                              notes=["build.sh: cat Utils Constraints Formulas Rules Config Deriver > ../PLN.metta",
                                     "present only when the PeTTa checkout has repos/PLN"])
    return S


# ── building the same lists by PARSING the source ────────────────────────────

def _str_constants(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _assigned(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            out[n.target.id] = n.value
    return out


def stacks_from_source() -> dict[str, StackSpec]:
    """The same stacks without importing the repo: `ast` over pln_runner.py and api.py.

    Slices (DrugAge, CellAge) and the caller patient need the repo's code, so here they degrade to the raw
    fixture text and no extra atoms, and the notes say so.
    """
    repo = repo_root()
    runner = _assigned(ast.parse((repo / "pln_chat" / "core" / "pln_runner.py").read_text(encoding="utf-8")))
    api_ = _assigned(ast.parse((repo / "pln_chat" / "api.py").read_text(encoding="utf-8")))
    cfg_src = (repo / "pln_chat" / "config.py").read_text(encoding="utf-8")
    m = re.search(r'PLN_MAX_KB_FILE_BYTES[^=]*=\s*int\(os\.getenv\("PLN_MAX_KB_FILE_BYTES",\s*"(\d+)"\)\)', cfg_src)
    cap = int(m.group(1)) if m else DEFAULT_MAX_KB_FILE_BYTES

    def names(table: dict, key: str) -> list[str]:
        return _str_constants(table[key])

    inference = names(api_, "_INFERENCE_STACK")
    discovered = {p.name: p for p in sorted(repo.glob("*.metta"))}
    custom = repo / "pln_chat" / "ontology" / "metta_files"
    if custom.exists():
        discovered.update({p.name: p for p in sorted(custom.glob("*.metta"))})
    rt = [n for n in inference if n in discovered and discovered[n].stat().st_size <= cap]
    g_ex, p_ex = set(names(runner, "GENERIC_STACK_EXCLUDED")), set(names(runner, "PATIENT_STACK_EXCLUDED"))
    he = set(names(runner, "HUMAN_EVIDENCE_FILES"))
    lin = names(runner, "LINAGE2_PATIENT_STACK")
    gen_baseline = repo / "build" / "nhanes_mortality_baseline.metta"
    smoke_order, smoke_q = parse_smoke()

    def rel(n: str) -> str:
        p = discovered.get(n)
        return _rel(p) if p else n

    S: dict[str, StackSpec] = {}
    S["smoke10"] = StackSpec("smoke10", "the ten files pln_deduction.metta section 5 names", smoke_order,
                             canary=[q["expr"] for q in smoke_q], source="parsed")
    S["generic"] = StackSpec("generic", "runtime stack minus the four patient layers", [rel(n) for n in rt if n not in g_ex], source="parsed")
    S["patient"] = StackSpec("patient", "runtime stack minus seven files", [rel(n) for n in rt if n not in p_ex], source="parsed")
    S["human_evidence"] = StackSpec("human_evidence", "four files", [rel(n) for n in rt if n in he], source="parsed")
    S["linage2"] = StackSpec("linage2", "LinAge2 scoped stack", [rel(n) for n in lin] + ([_rel(gen_baseline)] if gen_baseline.exists() else []),
                             source="parsed", notes=["no caller patient in parse mode"])
    drug = names(runner, "DRUGAGE_STACK")
    cell = names(runner, "CELLAGE_STACK")
    fx_d = repo / "tests" / "fixtures" / "drugage_real_rows.metta"
    fx_c = repo / "tests" / "fixtures" / "cellage_real_rows.metta"
    S["drugage_slice"] = StackSpec("drugage_slice", "DRUGAGE_STACK + the whole 8-row fixture", [rel(n) for n in drug],
                                   extra_atoms=fx_d.read_text(encoding="utf-8") if fx_d.exists() else "",
                                   extra_label="whole drugage_real_rows.metta (parse mode: no selector)", source="parsed")
    S["cellage_slice"] = StackSpec("cellage_slice", "CELLAGE_STACK + the whole 25-row fixture", [rel(n) for n in cell],
                                   extra_atoms=fx_c.read_text(encoding="utf-8") if fx_c.exists() else "",
                                   extra_label="whole cellage_real_rows.metta (parse mode: no selector)", source="parsed")
    S["full"] = StackSpec("full", "api._runtime_kb_paths()", [rel(n) for n in rt], source="parsed")
    nh = names(runner, "NHANES_PATIENT_STACK")
    S["nhanes_patient"] = StackSpec("nhanes_patient", "NHANES grounding stack", [rel(n) for n in nh], source="parsed")
    S["fork_pln"] = StackSpec("fork_pln", "patham9/PLN src/", _fork_files(), source="parsed")
    return S


# ── the public entry points ──────────────────────────────────────────────────

CORE_STACKS = ["generic", "patient", "human_evidence", "linage2", "drugage_slice", "cellage_slice"]
EXTRA_STACKS = ["smoke10", "full", "nhanes_patient", "fork_pln"]


def list_stacks(source: str = "import", include_missing: bool = False) -> dict[str, StackSpec]:
    """name -> StackSpec. `source` is 'import' (default) or 'parse'.

    A stack whose files are not on disk (fork_pln without the fork checkout) is dropped unless include_missing.
    """
    S = stacks_from_import() if source == "import" else stacks_from_source()
    if not include_missing:
        for name in list(S):
            spec = S[name]
            if any(not (Path(f) if Path(f).is_absolute() else repo_root() / f).exists() for f in spec.files):
                S.pop(name)
    return S


def stack_files(name: str, source: str = "import") -> list[str]:
    """The file order of one stack, e.g. stack_files('patient')."""
    return list_stacks(source)[name].files


def check_import_vs_source() -> dict:
    """Do the imported and the parsed file lists agree? {stack: (ok, detail)}."""
    a, b = list_stacks("import"), list_stacks("parse")
    out = {}
    for name in sorted(set(a) & set(b)):
        same = a[name].files == b[name].files
        out[name] = {"same_files": same, "n_import": len(a[name].files), "n_parse": len(b[name].files)}
        if not same:
            out[name]["only_import"] = [f for f in a[name].files if f not in b[name].files]
            out[name]["only_parse"] = [f for f in b[name].files if f not in a[name].files]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="List the runtime stacks' file orders (import only; runs no query).")
    ap.add_argument("--source", choices=["import", "parse"], default="import")
    ap.add_argument("--check", action="store_true", help="compare the imported lists with the parsed ones")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--texts", action="store_true", help="also show the size and sha256 of each assembled KB text")
    a = ap.parse_args(argv)
    if a.check:
        res = check_import_vs_source()
        print(json.dumps(res, indent=1))
        return 0 if all(v["same_files"] for v in res.values()) else 1
    S = list_stacks(a.source)
    if a.json:
        print(json.dumps({k: v.summary() for k, v in S.items()}, indent=1))
        return 0
    for name, spec in S.items():
        line = f"{name:15s} {len(spec.files):3d} files  {spec.description}"
        if a.texts:
            t = spec.kb_text()
            line += f"  [{len(t)} bytes, sha256 {sha256_text(t)[:12]}]"
        print(line)
        for f in spec.files:
            print(f"    {f}")
        if spec.extra_label:
            print(f"    + extra atoms: {spec.extra_label} ({len(spec.extra_atoms)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
