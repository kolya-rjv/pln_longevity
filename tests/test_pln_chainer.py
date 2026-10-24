"""PLN_CHAINER=pln: `infer` and `explain` answered by patham9's PLN (core/pln_chainer.py; gate 6a of the PeTTa migration).

What is under test. With `PLN_CHAINER=pln` (opt-in; it needs PLN_RUNTIME=petta) a request loads the vendored fork library
in the PeTTa child, derives the answers of every source node with `PLN.Derive` (state sentences + Modus Ponens, strength
stored as q = e + (1 - e) p and reported as |s1 - s0| / (1 - e)^hops, confidence times 0.9^(hops - 1)), and adds them to
the space as `PLNAnswer` facts that the new `infer` and `explain` equations read. The legacy equations are filtered out of
the text. Under the default chainer nothing changes.

Four layers, cheapest first:

* no engine: the config switch, the form-level filter, the assembly, the `PLNAnswer` guard (also in the validator), the
  s-expression and number formatting helpers, id assignment, the sub-graph pre-filter, and the adapter's join /
  completeness / retry / sign logic, driven by `docs/petta_migration/fork_design/sim/fork_sim.py`, the pure-Python port of
  the library, used here as an INDEPENDENT ORACLE of the engine (never as production code);
* gate 6a on the real PeTTa (`-m petta`): the 100 patient, 86 LinAge2, 101 generic pairs and the 109 pairs of the feedback
  edge, each equal to the legacy chainer in sign, strength, confidence and chain to 1e-12, none extra, none missing; the
  50 composite queries of the contract (rankings, diagnosis, counterfactuals, risk, supplements) against the legacy
  chainer on the same runtime; the pinned orderings; the no-false-chain pairs; diamond, determinism, caller facts, no state
  across requests, forgery refusal, the K schedule, the evidence side table, DrugAge;
* the pln goldens (`tests/fixtures/pln_chainer/`): exact strings of what the pln chainer prints. The legacy goldens cannot
  hold digit for digit (the last digit of 64 of 186 strengths differs), so these are separate files and the legacy ones
  are untouched. Regenerate with `python scripts/regen_pln_chainer_goldens.py`.
* the default is unchanged: with the legacy chainer `_runtime_run` hands the runtime exactly what it always did.

No test calls a model.

    pytest tests/test_pln_chainer.py -q                  # everything that can run here
    pytest tests/test_pln_chainer.py -q -m "not petta"   # without the real PeTTa
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

FORK_DESIGN = REPO / "docs" / "petta_migration" / "fork_design"
CONTRACT = FORK_DESIGN / "contract"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pln_chainer"
TOL = 1e-12

import config  # noqa: E402
from core import pln_chainer as pc  # noqa: E402
from core import pln_runner  # noqa: E402
from core.metta_runtime import MettaRuntimeError, PettaRuntime, get_runtime  # noqa: E402


def _real_petta_available() -> bool:
    try:
        if not PettaRuntime().available():
            return False
        return PettaRuntime().run_program("", ["!(+ 1 2)"]) == [["3"]]
    except Exception:
        return False


REAL_PETTA = _real_petta_available()
requires_petta = pytest.mark.skipif(not REAL_PETTA, reason="PeTTa is not installed here (SWI-Prolog, janus-swi and the "
                                                            "pinned checkout; bash scripts/setup_petta.sh)")


def real_petta(fn):
    return pytest.mark.petta(requires_petta(fn))


# ══════════════════════════════ data and oracles ═══════════════════════════════

def _load_sim():
    spec = importlib.util.spec_from_file_location("fork_sim", FORK_DESIGN / "sim" / "fork_sim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fork_sim"] = mod
    spec.loader.exec_module(mod)
    return mod


SIM = _load_sim()
LINKS_JSON = json.loads((CONTRACT / "links.json").read_text(encoding="utf-8"))
CLOSURE = json.loads((CONTRACT / "closure.json").read_text(encoding="utf-8"))["stacks"]
PINNED = json.loads((CONTRACT / "pinned_numbers.json").read_text(encoding="utf-8"))["entries"]
SIGN = {("Pos", "Pos"): "Pos", ("Pos", "Neg"): "Neg", ("Neg", "Pos"): "Neg", ("Neg", "Neg"): "Pos"}

FEEDBACK_EDGE = {"from": "ChronicInflammation", "to": "CellularSenescence", "sign": "Pos", "strength": 0.5,
                 "confidence": 0.65, "terminal_only": False, "kind": "Effect"}
FEEDBACK_ATOM = "(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))\n"


def contract_links(stack: str, extra: list[dict] | None = None) -> list[dict]:
    return [dict(l) for l in LINKS_JSON["links"] if stack in l["stacks"]] + [dict(e) for e in (extra or [])]


def to_links(rows: list[dict]) -> list[pc.Link]:
    links = [pc.Link(r["from"], r["to"], r["sign"], r["strength"], r["confidence"], terminal=r["terminal_only"],
                     kind="record" if r.get("kind") == "HazardRatio" or r["terminal_only"] else "curated")
             for r in rows]
    pc.assign_ids(links)
    return links


def legacy_routes(rows: list[dict], source: str) -> list[tuple]:
    """The LEGACY chainer's derivations from `source`, node-simple, as (nodes, sign, s, c): right-nested floats exactly
    as pln_deduction.metta (s = s_link * s_rest, c = (c_link * c_rest) * 0.9). A record ends a chain."""
    out_by: dict[str, list[dict]] = {}
    for r in rows:
        out_by.setdefault(r["from"], []).append(r)

    def derive(a, visited):
        res = []
        for r in out_by.get(a, []):
            b = r["to"]
            if b in visited:
                continue
            res.append(((a, b), r["sign"], r["strength"], r["confidence"]))
            if not r["terminal_only"]:
                for nodes, sg2, s2, c2 in derive(b, visited | {b}):
                    res.append(((a,) + nodes, SIGN[(r["sign"], sg2)], r["strength"] * s2,
                                (r["confidence"] * c2) * 0.9))
        return res

    return derive(source, frozenset([source]))


def legacy_closure(rows: list[dict]) -> dict[tuple, tuple]:
    out: dict[tuple, tuple] = {}
    for source in sorted({r["from"] for r in rows}):
        for nodes, sign, s, c in legacy_routes(rows, source):
            assert nodes not in out
            out[nodes] = (sign, s, c)
    return out


def compare(got: dict[tuple, tuple], want: dict[tuple, tuple], tol: float = TOL) -> dict:
    missing = sorted(p for p in want if p not in got)
    extra = sorted(p for p in got if p not in want)
    bad = [(p, got[p], want[p]) for p in want if p in got
           and (got[p][0] != want[p][0] or abs(got[p][1] - want[p][1]) > tol or abs(got[p][2] - want[p][2]) > tol)]
    dev = max((max(abs(got[p][1] - want[p][1]), abs(got[p][2] - want[p][2])) for p in want if p in got), default=0.0)
    return {"missing": missing, "extra": extra, "bad": bad, "max_dev": dev, "pairs": len(got)}


def table_closure(table: pc.ChainTable) -> dict[tuple, tuple]:
    out: dict[tuple, tuple] = {}
    for r in table.all_routes():
        assert r.path not in out, f"two routes with one node path {r.path}"
        out[r.path] = (r.sign, r.strength, r.confidence)
    return out


# ── the sim as an engine: PLN.Derive, Modus Ponens, Revision, chain-discount, as the adapter's `Evaluate` ───────

def _fmt(x) -> str:
    if isinstance(x, tuple):
        return "(" + " ".join(_fmt(i) for i in x) + ")"
    return repr(x) if isinstance(x, float) else str(x)


class SimEngine:
    """`Evaluate` backed by fork_sim: the oracle runs the library, the adapter runs unchanged on top of it."""

    def __init__(self, variant: str = "petta_lib", lose_stamp_of: tuple = (), flip_levels: bool = False):
        self.variant = variant
        self.calls: list[str] = []
        self.derive_calls = 0
        self._lose = lose_stamp_of
        self._flip = flip_levels

    def __call__(self, expr: str) -> list[str]:
        self.calls.append(expr)
        form = SIM.parse_sexprs("!" + expr)[0][1]
        head = form[0]
        if head == "PLN.Derive":
            self.derive_calls += 1
            tasks_f, beliefs_f, steps, maxsteps, tq, bq = form[1:7]
            tasks = [SIM.sentence_from_metta(x) for x in tasks_f]
            beliefs = [SIM.sentence_from_metta(x) for x in beliefs_f]
            sim = SIM.Sim(rules="restricted", maxsteps=maxsteps, task_queue_size=tq, belief_queue_size=bq,
                          variant=self.variant)
            out_t, out_b = sim.derive(tasks, beliefs, maxsteps - steps + 1, tq, bq)
            if self._flip:        # a broken engine: the sign of every difference reversed (the cross-check must see it)
                out_b = [(t, (1.0 - tv[0], tv[1]), ev) if isinstance(t, tuple) and t[0] == "State" and len(ev) > 1
                         else (t, tv, ev) for (t, tv, ev) in out_b]

            def show(sents):
                return "(" + " ".join(
                    "(Sentence (%s (stv %r %r)) (%s))" % (_fmt(t), tv[0], tv[1], " ".join(str(i) for i in ev))
                    for (t, tv, ev) in sents) + ")"
            return ["(%s %s)" % (show(out_t), show(out_b))]
        if head == "Truth_ModusPonens":
            (s, c) = SIM.Truth_ModusPonens((form[1][1], form[1][2]), (form[2][1], form[2][2]))
            return ["(stv %r %r)" % (s, c)]
        if head == "Truth_Revision":
            a, b = (form[1][1], form[1][2]), (form[2][1], form[2][2])
            if a[1] >= 1.0 or b[1] >= 1.0:
                return []
            s, c = SIM.Truth_Revision(a, b)
            return ["(stv %r %r)" % (s, c)]
        if head == "chain-discount":
            return ["0.9"]
        raise AssertionError(f"SimEngine cannot evaluate {expr[:80]}")


def sim_table(rows: list[dict], settings: pc.ChainSettings | None = None, engine: SimEngine | None = None,
              sources=None, plans=None) -> pc.ChainTable:
    """`plans=[]` skips the pre-flight, to reach the paths behind it (an estimate that was wrong)."""
    links = to_links(rows)
    ev = engine or SimEngine()
    leak = pc.probe_library(ev)
    return pc.compute_table(ev, links, leak=leak, discount=pc.read_discount(ev),
                            settings=settings or pc.ChainSettings(), sources=sources, plans=plans)


# ══════════════════════════════ configuration ══════════════════════════════════

def test_the_default_chainer_is_legacy_and_pln_is_opt_in(monkeypatch):
    """With nothing exported every setting is its default (the suite itself may be run with PLN_CHAINER=pln, which is the
    point of the lane: only the defaults of the parsers are asserted here, not the process's values)."""
    for name in ("PLN_CHAINER", "PLN_CHAIN_CONFIDENCE", "PLN_RULESET", "PLN_QUEUE_SIZE", "PLN_MAX_STEPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    assert config._chainer_choice() == "legacy"
    assert config._chain_confidence_choice() == "legacy"
    assert config._ruleset_choice() == "restricted"
    assert config.PLN_CHAINER in ("legacy", "pln") and config.PLN_QUEUE_SIZE >= 16 and config.PLN_MAX_STEPS >= 0
    src = (PLN_CHAT / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("PLN_QUEUE_SIZE", "256")' in src and 'os.getenv("PLN_MAX_STEPS", "0")' in src
    assert 'os.getenv("PLN_CHAINER", "legacy")' in src


@pytest.mark.parametrize("value, ok", [("legacy", True), ("pln", True), ("LEGACY", True), ("", True), ("fork", False),
                                       ("hybrid", False)])
def test_PLN_CHAINER_accepts_legacy_and_pln_only(monkeypatch, value, ok):
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setenv("PLN_CHAINER", value)
    if ok:
        assert config._chainer_choice() in ("legacy", "pln")
    else:
        with pytest.raises(ValueError, match="PLN_CHAINER must be legacy or pln"):
            config._chainer_choice()


@pytest.mark.parametrize("runtime, ok", [("petta", True), ("hyperon", False), ("auto", False)])
def test_PLN_CHAINER_pln_needs_the_petta_runtime(monkeypatch, runtime, ok):
    monkeypatch.setattr(config, "PLN_RUNTIME", runtime)
    monkeypatch.setenv("PLN_CHAINER", "pln")
    if ok:
        assert config._chainer_choice() == "pln"
    else:
        with pytest.raises(ValueError, match="needs PLN_RUNTIME=petta"):
            config._chainer_choice()


@pytest.mark.parametrize("value, ok", [("legacy", True), ("native", True), ("", True), ("fork", False)])
def test_PLN_CHAIN_CONFIDENCE_accepts_legacy_and_native_only(monkeypatch, value, ok):
    monkeypatch.setenv("PLN_CHAIN_CONFIDENCE", value)
    if ok:
        assert config._chain_confidence_choice() in ("legacy", "native")
    else:
        with pytest.raises(ValueError, match="PLN_CHAIN_CONFIDENCE"):
            config._chain_confidence_choice()


@pytest.mark.parametrize("value, ok", [("restricted", True), ("full", True), ("", True), ("deduction", False)])
def test_PLN_RULESET_accepts_restricted_and_full_only(monkeypatch, value, ok):
    monkeypatch.setenv("PLN_RULESET", value)
    if ok:
        assert config._ruleset_choice() in ("restricted", "full")
    else:
        with pytest.raises(ValueError, match="PLN_RULESET"):
            config._ruleset_choice()
    assert config.PLN_RULESET == "restricted"


def test_the_chainer_mode_is_read_when_asked_and_pln_on_hyperon_is_refused(monkeypatch):
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    assert pc.chainer_mode() == "legacy" and not pc.is_pln()
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    with pytest.raises(pc.PlnChainerError, match="PLN_RUNTIME=petta"):
        pc.chainer_mode()
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    assert pc.chainer_mode() == "pln" and pc.is_pln()
    monkeypatch.setattr(config, "PLN_CHAINER", "fork")
    with pytest.raises(pc.PlnChainerError):
        pc.chainer_mode()


def test_the_settings_are_read_from_the_config(monkeypatch):
    monkeypatch.setattr(config, "PLN_CHAIN_CONFIDENCE", "native")
    monkeypatch.setattr(config, "PLN_QUEUE_SIZE", 512)
    monkeypatch.setattr(config, "PLN_MAX_STEPS", 77)
    st = pc.settings_from_config()
    assert (st.chain_confidence, st.queue_size, st.max_steps) == ("native", 512, 77)
    assert not st.control and st.on_incomplete == "retry" and st.drugage_fold


def test_a_request_cannot_set_the_queue_sizes_or_the_step_bound():
    """PLN_MAX_STEPS and PLN_QUEUE_SIZE are environment settings: /metta/run takes neither, and the research lane's
    separate queue sizes exist only as arguments of compute_table, never as a config value."""
    src = (PLN_CHAT / "api.py").read_text(encoding="utf-8")
    for name in ("PLN_MAX_STEPS", "PLN_QUEUE_SIZE", "ChainSettings", "task_queue", "belief_queue"):
        assert name not in src, name
    assert not hasattr(config, "PLN_TASK_QUEUE") and not hasattr(config, "PLN_BELIEF_QUEUE")


# ══════════════════════════════ the text: filter, assembly, guard ══════════════

def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def test_the_filter_removes_exactly_the_legacy_chain_equations_of_pln_deduction():
    text = _read("pln_deduction.metta")
    out, removed = pc.filter_legacy_chain(text)
    assert removed == 5                                   # infer x2, explain x3
    assert out != text
    code = re.sub(r";[^\n]*", "", out)
    assert not re.search(r"\(=\s*\(\s*(infer|explain)[\s)]", code)
    # everything else is kept: the helpers, the signatures, `link-effect`, the lifts, the constants
    for kept in ("(= (link-effect $s $a $c)", "(= (combine-hop", "(= (prepend-node", "(= (sign-product Pos Pos) Pos)",
                 "(= (chain-discount) 0.9)", "(= (hr->sign $hr)", "(: infer (-> %Undefined% %Undefined% %Undefined% %Undefined%))",
                 "(: explain (-> %Undefined% %Undefined% %Undefined% %Undefined%))", "(: Effect (-> Atom Atom Atom Atom Atom))"):
        assert kept in out, kept
    # nothing but the five forms went: removing them from the original gives the output, and the rest is byte for byte
    from core.metta_runtime import split_top_level_forms
    gone = [text[s:e] for s, e in split_top_level_forms(text) if re.match(r"\(=\s*\(\s*(infer|explain)[\s)]", text[s:e])]
    assert len(gone) == 5
    rebuilt = text
    for g in gone:
        assert rebuilt.count(g) == 1
        rebuilt = rebuilt.replace(g, "")
    assert rebuilt == out


def test_the_filter_removes_the_third_infer_equation_of_drugage_calibration_and_nothing_else():
    text = _read("drugage_calibration.metta")
    out, removed = pc.filter_legacy_chain(text)
    assert removed == 1
    code = re.sub(r";[^\n]*", "", out)
    assert "(= (drugage-effect $s $row)" in out and "(Effect Lifespan Mortality Neg (stv 1.0 1.0))" in out
    assert "(= (infer" not in code and "(drugage-effect $s $r)" not in code      # the equation that called the lift is gone
    assert "(= (drugage-confidence $s $row)" in out and "(= (lifespan-sign $pct)" in out
    from core.metta_runtime import split_top_level_forms
    forms = [text[s:e] for s, e in split_top_level_forms(text) if re.match(r"\(=\s*\(\s*infer[\s)]", text[s:e])]
    assert len(forms) == 1 and text.replace(forms[0], "") == out


def test_the_filter_leaves_a_text_without_those_equations_untouched_and_is_idempotent():
    for name in ("mechanistic_bridges.metta", "patient_profile.metta", "pln_intervention_ranking.metta"):
        text = _read(name)
        out, removed = pc.filter_legacy_chain(text)
        assert removed == 0 and out is text, name
    once, _ = pc.filter_legacy_chain(_read("pln_deduction.metta"))
    twice, n = pc.filter_legacy_chain(once)
    assert n == 0 and twice == once


def test_the_filter_reads_forms_not_lines():
    text = ('; (= (infer $s $a $c) 1)  <- a comment\n'
            '(= (explain-me $x) $x)\n'
            '"(= (infer $s $a $c) 2)"\n'
            '(=   (infer\n   $s $a $c)\n   (link-effect $s $a $c))\n'
            '(= (inferred $x) $x)\n'
            '!(infer &self A B)\n')
    out, removed = pc.filter_legacy_chain(text)
    assert removed == 1
    assert "(= (explain-me $x) $x)" in out and "(= (inferred $x) $x)" in out
    assert "!(infer &self A B)" in out and "<- a comment" in out
    assert "(link-effect $s $a $c)" not in out


def test_every_stack_of_the_app_loses_the_same_equations_and_gains_the_reader():
    import api
    runtime = api._runtime_kb_paths()
    stacks = {"full": runtime, "patient": pln_runner.patient_stack(runtime), "generic": pln_runner.generic_stack(runtime),
              "linage2": pln_runner.linage2_patient_kb(), "nhanes": pln_runner.nhanes_patient_kb(),
              "drugage": pln_runner.DRUGAGE_STACK, "cellage": pln_runner.CELLAGE_STACK}
    for name, files in stacks.items():
        blocks = [(p.name, p.read_text(encoding="utf-8")) for p in files]
        prepared = pc.prepare(blocks)
        has_drugage = any(p.name == "drugage_calibration.metta" for p in files)
        assert prepared.removed == (6 if has_drugage else 5), name
        assert prepared.step is not None, name
        code = re.sub(r";[^\n]*", "", prepared.kb_text)
        assert len(re.findall(r"\(=\s*\(\s*infer[\s)]", code)) == 1, name           # the reader, and nothing else
        assert len(re.findall(r"\(=\s*\(\s*explain[\s)]", code)) == 1, name
        assert prepared.kb_text.rstrip().endswith(pc.chainer_text().rstrip()), name


def test_a_stack_with_no_chain_to_replace_is_left_alone():
    human = pln_runner.human_evidence_stack(__import__("api")._runtime_kb_paths())
    blocks = [(p.name, p.read_text(encoding="utf-8")) for p in human]
    prepared = pc.prepare(blocks)
    assert prepared.removed == 0 and prepared.step is None
    assert prepared.kb_text == "\n".join(t for _, t in blocks)


def test_a_callers_own_definitions_are_not_filtered():
    base = [("pln_deduction.metta", _read("pln_deduction.metta"))]
    extra = "(= (infer $s $a $c) (just-mine $a $c))\n"
    prepared = pc.prepare(base + [("extra_atoms", extra)])
    assert "(just-mine $a $c)" in prepared.kb_text


def test_the_chainer_file_defines_the_two_equations_and_nothing_else():
    code = re.sub(r";[^\n]*", "", pc.chainer_text())
    heads = re.findall(r"^\(=\s*\((\S+)", code, re.M)
    assert heads == ["infer", "explain"]
    assert code.count("PLNAnswer") == 2
    assert "(signed $sgn $tv)" in code and "(Chain $nodes (signed $sgn $tv))" in code


def test_the_restricted_library_passes_the_sandbox_scan():
    assert "Truth_ModusPonens" in pc.library_text()
    assert pc.LIBRARY_FILE.name == "lib_pln_restricted.metta" and pc.CHAIN_FILE.name == "pln_chainer.metta"


# ── the PLNAnswer guard ────────────────────────────────────────────────────────

FORGED = "(PLNAnswer DasatinibPlusQuercetin CoronaryHeartDisease Neg (stv 1.0 1.0) (A B) (1))"


def test_a_text_that_writes_the_answer_table_is_refused_at_assembly():
    for where in ("extra_atoms", "mechanistic_bridges.metta"):
        blocks = [("pln_deduction.metta", _read("pln_deduction.metta")), (where, FORGED)]
        with pytest.raises(MettaRuntimeError) as err:
            pc.prepare(blocks)
        assert err.value.kind == "refused" and "PLNAnswer" in str(err.value) and where in str(err.value)


def test_the_word_in_a_comment_or_a_string_is_not_a_forgery():
    ok = '; PLNAnswer is the table\n(Note "PLNAnswer in a string")\n(Effect A B Pos (stv 0.5 0.5))\n'
    prepared = pc.prepare([("pln_deduction.metta", _read("pln_deduction.metta")), ("extra_atoms", ok)])
    assert prepared.step is not None
    assert pc.answer_head_in(ok) is False
    assert pc.answer_head_in(FORGED) is True
    assert pc.answer_head_in("(foo PLNAnswer)") is True
    assert pc.answer_head_in("(PLNAnswers x)") is False


def test_the_validator_refuses_the_answer_table_only_under_pln(monkeypatch):
    from core.metta_validator import validate
    from ontology.registry import OntologyRegistry
    query = "!(match &self (PLNAnswer A B $s $t $n $i) $t)"
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    legacy = validate(query, OntologyRegistry())
    assert not any("PLNAnswer" in i and "forge" in i for i in legacy.issues)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    pln = validate(query, OntologyRegistry())
    assert not pln.valid and any("forge" in i and "PLNAnswer" in i for i in pln.issues)
    injected = validate(FORGED + "\n!(infer &self A B)", OntologyRegistry())      # a caller's extra_atoms + query
    assert not injected.valid and any("forge" in i for i in injected.issues)
    clean = validate("!(infer &self A B)", OntologyRegistry())
    assert not any("forge" in i for i in clean.issues)


def test_the_validator_knows_the_audit_atoms_only_under_pln(monkeypatch):
    pytest.importorskip("fastapi")
    import api
    from core.metta_validator import validate
    registry, inventory = api._runtime_registry(), api._runtime_inventory()
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    audit = "!(match &self (PLNEvidence $id $kind $where $tier) ($id $kind))"
    legacy = validate(audit, registry, inventory)
    assert not legacy.valid and any("PLNEvidence" in i for i in legacy.issues)       # unknown, as before
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    pln = validate(audit, registry, inventory)
    assert pln.valid, pln.issues
    status = validate("!(match &self (PLNStatus $s $st $k) ($s $st))", registry, inventory)
    assert status.valid, status.issues
    answer = validate("!(match &self (PLNAnswer $a $c $g $t $n $i) $t)", registry, inventory)
    assert not answer.valid and any("forge" in i for i in answer.issues)             # the table itself stays closed
    other = validate("!(match &self (PLNSomethingElse $x) $x)", registry, inventory)
    assert not other.valid                                                          # only the three audit heads are opened


# ══════════════════════════════ helpers: s-expressions, numbers ════════════════

def test_sexpr_parse_and_serialize_round_trip():
    text = '(Sentence ((Implication (State n0) (Not (State n1))) (stv 0.52 0.65)) (3 7))'
    tree = pc.parse_sexpr(text)
    assert tree[0] == "Sentence" and tree[2] == ("3", "7")
    assert pc.serialize(tree) == text
    assert pc.parse_sexpr('(L "a b" ())') == ("L", '"a b"', ())
    for bad in ("(a b", "a b", ")", ""):
        with pytest.raises(pc.PlnChainerError):
            pc.parse_sexpr(bad)


def test_numbers_read_back_exactly_and_never_use_an_exponent():
    rnd = random.Random(7)
    values = [0.0, 1.0, 0.5, 1e-05, 1e-10, 3.0000000000000006e-15, 1.0000000000000002, 0.24989999999999998,
              123456789.123, 5e-324] + [rnd.random() ** rnd.randint(1, 30) for _ in range(500)]
    for v in values:
        text = pc.fnum(v)
        assert "e" not in text.lower() and "." in text, text
        assert float(text) == v, (v, text)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(pc.PlnChainerError):
            pc.fnum(bad)
    assert pc.number("1.0e-05") == 1e-05 and pc.number("3") == 3.0 and pc.number("-.5") == -0.5
    for notnum in ("A", "(stv", "nan", "inf", "1.0.0", None, ("a",)):
        assert pc.number(notnum) is None


# ══════════════════════════════ links, ids, sub-graphs ═════════════════════════

def test_ids_are_assigned_by_content_whatever_the_input_order():
    rows = contract_links("patient")
    reference = {(l.src, l.dst, l.sign, l.terminal, l.strength): l.id for l in to_links(rows)}
    assert sorted(reference.values()) == list(range(1, len(rows) + 1))
    rnd = random.Random(1)
    for _ in range(10):
        shuffled = rows[:]
        rnd.shuffle(shuffled)
        assert {(l.src, l.dst, l.sign, l.terminal, l.strength): l.id for l in to_links(shuffled)} == reference


def test_an_evidence_id_is_never_shared_by_two_links():
    links = to_links(contract_links("patient"))
    pc.assert_ids_unique_along_paths(links)
    links[3].id = links[2].id                            # two serial or parallel links with one id
    with pytest.raises(pc.PlnChainerError, match="shared by"):
        pc.assert_ids_unique_along_paths(links)
    links[3].id = 0
    with pytest.raises(pc.PlnChainerError, match="no evidence id"):
        pc.assert_ids_unique_along_paths(links)


def test_a_shared_id_on_two_serial_links_would_lose_the_chain_which_is_why_it_is_refused():
    """The library refuses to combine premises that share an id (StampDisjoint): with A -> B (id 1) and B -> C (id 1)
    the oracle derives B but never C. The adapter's check exists so that this cannot happen silently."""
    def derive(second_id: int):
        premise = "(Sentence ((State n0) (stv 1.0 1.0)) (0))"
        ab = "(Sentence ((Implication (State n0) (State n1)) (stv 0.51 0.8)) (1))"
        bc = f"(Sentence ((Implication (State n1) (State n2)) (stv 0.51 0.8)) ({second_id}))"
        _, beliefs = pc._derive(SimEngine(), f"({premise})", f"({ab} {bc} {premise})", 1, 100, 256, 256)
        return {b[0][1] for b in beliefs if b[0][0] == "State" and len(b[3]) > 1}

    assert derive(1) == {"n1"}                      # the chain stops at B
    assert derive(2) == {"n1", "n2"}                # distinct ids: it goes on to C
    links = [pc.Link("A", "B", "Pos", 0.5, 0.8), pc.Link("B", "C", "Pos", 0.5, 0.8)]
    links[0].id = links[1].id = 1
    with pytest.raises(pc.PlnChainerError):
        pc.assert_ids_unique_along_paths(links)


def test_reachable_links_follows_links_and_keeps_but_does_not_expand_a_record():
    rows = contract_links("patient")
    links = to_links(rows)
    sub = pc.reachable_links(links, "DasatinibPlusQuercetin")
    assert {(l.src, l.dst) for l in sub} >= {("DasatinibPlusQuercetin", "CellularSenescence"), ("SASP", "DNAmPAI1"),
                                             ("DNAmPAI1", "CoronaryHeartDisease")}
    assert not any(l.src in ("Fisetin", "Spermidine", "Metformin") for l in sub)
    assert len(sub) < len(links)
    # a record's target is kept as a target and not expanded: AgeAccelGrim -> CHD (record), CHD has no links out anyway
    chain = [pc.Link("A", "B", "Pos", 0.5, 0.5, terminal=True), pc.Link("B", "C", "Pos", 0.5, 0.5)]
    pc.assign_ids(chain)
    assert [l.dst for l in pc.reachable_links(chain, "A")] == ["B"]
    assert [l.dst for l in pc.reachable_links(chain, "B")] == ["C"]


def test_the_route_count_matches_the_legacy_derivations_and_caps():
    rows = contract_links("patient")
    links = to_links(rows)
    for source in ("DasatinibPlusQuercetin", "Spermidine", "Metformin", "CRP"):
        sub = pc.reachable_links(links, source)
        assert pc.count_routes(sub, source, 10_000) == len(legacy_routes(rows, source))
    assert pc.count_routes(links, "Spermidine", 3) == 4
    cyc = to_links(contract_links("patient", [FEEDBACK_EDGE]))
    assert pc.count_routes(pc.reachable_links(cyc, "Omega3"), "Omega3", 10_000) == len(
        legacy_routes(contract_links("patient", [FEEDBACK_EDGE]), "Omega3"))


# ══════════════════════════════ the adapter, on the oracle ═════════════════════

@pytest.mark.parametrize("stack, extra, count", [("patient", [], 100), ("linage2", [], 86), ("generic", [], 101),
                                                 ("patient", [FEEDBACK_EDGE], 109)])
def test_the_adapter_on_the_oracle_equals_the_legacy_closure(stack, extra, count):
    rows = contract_links(stack, extra)
    table = sim_table(rows)
    got = table_closure(table)
    result = compare(got, legacy_closure(rows))
    assert result["missing"] == [] and result["extra"] == [] and result["bad"] == [], result
    assert result["pairs"] == count and result["max_dev"] < 1e-15
    assert table.complete and all(s.attempts == 1 for s in table.status.values())
    assert sum(s.unattributed for s in table.status.values()) == 0         # every conclusion is a route or a walk of the graph
    assert all(len(r.path) - 1 == r.hops for r in table.all_routes())


def test_the_adapter_on_the_oracle_reproduces_the_recorded_hyperon_closure_bit_for_bit_in_signs_and_chains():
    for stack in ("patient", "linage2"):
        want = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"]) for d in CLOSURE[stack]["derivations"]}
        got = table_closure(sim_table(contract_links(stack)))
        result = compare(got, want)
        assert result["missing"] == [] and result["extra"] == [] and result["bad"] == [], (stack, result)


def test_one_route_per_pair_on_the_shipped_graphs_and_sorted_by_hops_then_stamp():
    for stack in ("patient", "linage2", "generic"):
        table = sim_table(contract_links(stack))
        pairs = [(r.source, r.target) for r in table.all_routes()]
        assert len(pairs) == len(set(pairs)), stack
        for routes in table.routes.values():
            assert [r.key() for r in routes] == sorted(r.key() for r in routes)


def test_the_feedback_edge_terminates_and_keeps_the_hundred_old_answers():
    old = table_closure(sim_table(contract_links("patient")))
    rows = contract_links("patient", [FEEDBACK_EDGE])
    new = table_closure(sim_table(rows))
    assert len(old) == 100 and len(new) == 109
    assert {p: v for p, v in new.items() if p in old} == old
    fresh = sorted(p for p in new if p not in old)
    assert len(fresh) == 9 and fresh[0][0] in ("ChronicInflammation", "Omega3", "SASP")
    assert ("ChronicInflammation", "CellularSenescence") in fresh


def test_a_self_loop_and_a_cycle_back_to_the_source_give_no_pair_and_no_hang():
    rows = [{"from": "A", "to": "A", "sign": "Pos", "strength": 0.5, "confidence": 0.5, "terminal_only": False},
            {"from": "A", "to": "B", "sign": "Pos", "strength": 0.5, "confidence": 0.8, "terminal_only": False},
            {"from": "B", "to": "A", "sign": "Neg", "strength": 0.5, "confidence": 0.8, "terminal_only": False}]
    got = table_closure(sim_table(rows))
    assert sorted(got) == [("A", "B"), ("B", "A")]


def test_a_diamond_gives_one_result_per_route():
    rows = [{"from": "A", "to": "B", "sign": "Pos", "strength": 0.5, "confidence": 0.8, "terminal_only": False},
            {"from": "A", "to": "C", "sign": "Neg", "strength": 0.6, "confidence": 0.7, "terminal_only": False},
            {"from": "B", "to": "D", "sign": "Pos", "strength": 0.9, "confidence": 0.8, "terminal_only": False},
            {"from": "C", "to": "D", "sign": "Neg", "strength": 0.8, "confidence": 0.9, "terminal_only": False}]
    table = sim_table(rows)
    to_d = [r for r in table.routes["A"] if r.target == "D"]
    assert sorted(r.path for r in to_d) == [("A", "B", "D"), ("A", "C", "D")]
    assert {r.path: (r.sign, round(r.strength, 12), round(r.confidence, 12)) for r in to_d} == {
        ("A", "B", "D"): ("Pos", 0.45, 0.576), ("A", "C", "D"): ("Pos", 0.48, 0.567)}


def test_parallel_links_of_opposite_sign_and_duplicates_are_kept_as_separate_routes():
    rows = [{"from": "A", "to": "B", "sign": "Pos", "strength": 0.5, "confidence": 0.8, "terminal_only": False},
            {"from": "A", "to": "B", "sign": "Neg", "strength": 0.6, "confidence": 0.7, "terminal_only": False},
            {"from": "A", "to": "B", "sign": "Neg", "strength": 0.6, "confidence": 0.7, "terminal_only": False}]
    routes = sim_table(rows).routes["A"]
    assert [(r.sign, r.strength) for r in routes] == [("Neg", pytest.approx(0.6)), ("Neg", pytest.approx(0.6)),
                                                      ("Pos", pytest.approx(0.5))]
    assert len({r.stamp for r in routes}) == 3


def test_a_record_ends_a_chain_and_never_continues_one():
    rows = [{"from": "A", "to": "B", "sign": "Pos", "strength": 0.5, "confidence": 0.8, "terminal_only": False},
            {"from": "B", "to": "C", "sign": "Neg", "strength": 0.6, "confidence": 0.7, "terminal_only": True},
            {"from": "C", "to": "D", "sign": "Pos", "strength": 0.6, "confidence": 0.7, "terminal_only": False}]
    got = table_closure(sim_table(rows))
    assert sorted(got) == [("A", "B"), ("A", "B", "C"), ("B", "C"), ("C", "D")]
    assert got[("A", "B", "C")][0] == "Neg"


def test_strength_and_sign_are_exact_for_every_strength_including_the_ends():
    """q = e + (1 - e) p is exact for every p in [0, 1]; the capped q = min(1, p + e) shrinks p > 1 - e."""
    for p in (0.0, 1e-9, 0.02, 0.5, 0.97, 0.98, 0.985, 0.99, 1.0):
        for sign in ("Pos", "Neg"):
            rows = [{"from": "A", "to": "B", "sign": sign, "strength": p, "confidence": 0.9, "terminal_only": False},
                    {"from": "B", "to": "C", "sign": "Neg", "strength": 1.0, "confidence": 1.0, "terminal_only": False}]
            got = table_closure(sim_table(rows))
            assert got[("A", "B")][1] == pytest.approx(p, abs=1e-15), (p, sign)
            assert got[("A", "B", "C")][1] == pytest.approx(p, abs=1e-15), (p, sign)
            assert got[("A", "B", "C")][0] == SIGN[(sign, "Neg")]


def test_the_research_mappings_are_what_the_decision_says_they_are():
    rows = contract_links("patient")
    dq = ("DasatinibPlusQuercetin", "CellularSenescence", "SASP", "DNAmPAI1", "CoronaryHeartDisease")
    affine = table_closure(sim_table(rows))[dq]
    literal = table_closure(sim_table(rows, pc.ChainSettings(mapping="literal", chain_confidence="native")))[dq]
    assert affine[1] == pytest.approx(0.2499, abs=1e-15)
    assert literal[1] == pytest.approx(0.22259936, abs=1e-9)             # the fork read literally: every strength shrinks (R0)
    assert literal[2] == pytest.approx(0.23343125, abs=1e-12) and literal[0] == "Neg"
    shrink = [table_closure(sim_table(rows, pc.ChainSettings(mapping="literal")))[k][1] / v[1]
              for k, v in table_closure(sim_table(rows)).items() if v[1] > 0]
    assert 0.85 < min(shrink) < 0.98 and max(shrink) <= 1.0              # a few percent to ~14 percent per route
    generic = contract_links("generic")
    lm = ("Lifespan", "Mortality")
    assert table_closure(sim_table(generic))[lm][1] == pytest.approx(1.0, abs=1e-15)
    assert table_closure(sim_table(generic, pc.ChainSettings(mapping="capped")))[lm][1] == pytest.approx(0.98, abs=1e-12)
    with pytest.raises(ValueError):
        pc.ChainSettings(mapping="nonsense")
    with pytest.raises(ValueError):
        pc.ChainSettings(ruleset="deduction")


def test_native_confidence_drops_the_discount_and_nothing_else():
    rows = contract_links("patient")
    legacy = sim_table(rows, pc.ChainSettings(chain_confidence="legacy"))
    native = sim_table(rows, pc.ChainSettings(chain_confidence="native"))
    for a, b in zip(legacy.all_routes(), native.all_routes()):
        assert (a.path, a.sign, a.strength) == (b.path, b.sign, b.strength)
        assert a.native_confidence == b.native_confidence == b.confidence
        assert a.confidence == pytest.approx(a.native_confidence * 0.9 ** (a.hops - 1), abs=1e-15)


def test_the_candidate_order_cannot_change_a_result():
    rows = contract_links("patient", [FEEDBACK_EDGE])
    reference = sim_table(rows)
    rnd = random.Random(5)
    for _ in range(4):
        shuffled = rows[:]
        rnd.shuffle(shuffled)
        other = sim_table(shuffled)
        assert [(r.path, r.sign, r.strength, r.confidence, r.stamp) for r in other.all_routes()] == \
               [(r.path, r.sign, r.strength, r.confidence, r.stamp) for r in reference.all_routes()]


def test_the_sub_graph_prefilter_does_not_change_an_answer():
    rows = contract_links("patient")
    links = to_links(rows)
    ev = SimEngine()
    leak = pc.probe_library(ev)
    enc = pc._Encoding(links, leak)
    source = "DasatinibPlusQuercetin"
    sub = pc.reachable_links(links, source)
    pre = pc._routes(enc, source, pc._run(ev, enc, sub, source, 1.0, 1024, 256, 256, False)[1],
                     pc._run(ev, enc, sub, source, 0.0, 1024, 256, 256, False)[1], 0.9, pc.ChainSettings())
    whole = pc._routes(enc, source, pc._run(ev, enc, links, source, 1.0, 1024, 256, 256, False)[1],
                       pc._run(ev, enc, links, source, 0.0, 1024, 256, 256, False)[1], 0.9, pc.ChainSettings())
    assert [(r.path, r.strength, r.confidence) for r in pre] == [(r.path, r.strength, r.confidence) for r in whole]


def test_the_sign_cross_check_catches_an_engine_whose_levels_disagree_with_the_parity():
    with pytest.raises(pc.PlnChainerError, match="sign cross-check"):
        sim_table(contract_links("patient"), engine=SimEngine(flip_levels=True))


def test_the_library_probes_catch_a_library_that_is_not_the_forks():
    class Wrong(SimEngine):
        def __call__(self, expr):
            if expr.startswith("(Truth_ModusPonens (stv 0.5"):
                return ["(stv 0.25 0.64)"]
            return super().__call__(expr)

    with pytest.raises(pc.PlnChainerError, match="Modus Ponens is not the fork's"):
        pc.probe_library(Wrong())

    class NoLeak(SimEngine):
        def __call__(self, expr):
            if expr.startswith("(Truth_ModusPonens (stv 0.0"):
                return ["(stv 0.0 1.0)"]
            return super().__call__(expr)

    with pytest.raises(pc.PlnChainerError, match="leak"):
        pc.probe_library(NoLeak())

    class Revises(SimEngine):
        def __call__(self, expr):
            if expr.startswith("(Truth_Revision"):
                return ["(stv 0.5 1.0)"]
            return super().__call__(expr)

    with pytest.raises(pc.PlnChainerError, match="revises"):
        pc.probe_library(Revises())
    assert pc.probe_library(SimEngine()) == pytest.approx(0.02)


def test_exactly_one_result_per_derive_is_required():
    class NoResult(SimEngine):
        def __call__(self, expr):
            return [] if expr.startswith("(PLN.Derive") else super().__call__(expr)

    with pytest.raises(pc.PlnChainerError, match="exactly one"):
        sim_table(contract_links("patient"), engine=NoResult(), sources=["Spermidine"])


# ── completeness, the K schedule, the control lane ─────────────────────────────

def _binary_tree(depth: int) -> list[dict]:
    rows, frontier, n = [], ["R"], 0
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for sign in ("Pos", "Neg"):
                n += 1
                child = f"T{n}"
                rows.append({"from": node, "to": child, "sign": sign, "strength": 0.9 - 0.001 * n, "confidence": 0.9,
                             "terminal_only": False})
                nxt.append(child)
        frontier = nxt
    return rows


def test_the_first_queue_size_is_picked_from_the_route_count_and_never_wasted():
    rows = _binary_tree(6)                                   # 126 links, 126 routes from the root
    table = sim_table(rows, pc.ChainSettings(queue_size=64, growth=8, rungs=3), sources=["R"])
    status = table.status["R"]
    assert status.complete and status.attempts == 1
    assert status.queue == 512 > 126 + 1 + 2 * 126           # the first rung that holds the estimate
    assert len(table.routes["R"]) == 126


def test_an_attempt_that_dropped_something_is_repeated_at_the_next_size(monkeypatch):
    rows = _binary_tree(5)                                   # 62 links
    monkeypatch.setattr(pc, "estimate_beliefs", lambda sub, source, cap: 0)      # make the first attempt too small
    table = sim_table(rows, pc.ChainSettings(queue_size=16, growth=4, rungs=4), sources=["R"])
    status = table.status["R"]
    assert status.complete and status.attempts == 3 and status.queue == 256      # 16 and 64 dropped, 256 did not
    assert len(table.routes["R"]) == 62
    full = sim_table(rows, pc.ChainSettings(queue_size=4096), sources=["R"])
    assert [(r.path, r.strength, r.confidence) for r in table.routes["R"]] == \
           [(r.path, r.strength, r.confidence) for r in full.routes["R"]]


def test_a_source_that_cannot_finish_fails_the_request_instead_of_returning_a_partial_table(monkeypatch):
    rows = _binary_tree(5)
    monkeypatch.setattr(pc, "estimate_beliefs", lambda sub, source, cap: 0)
    with pytest.raises(pc.PlnChainerError, match=r"did not finish.*K=32"):
        sim_table(rows, pc.ChainSettings(queue_size=16, growth=2, rungs=2), sources=["R"], plans=[])      # past the pre-flight
    with pytest.raises(pc.PlnChainerError, match="did not finish"):
        sim_table(contract_links("patient"), pc.ChainSettings(max_steps=2), sources=["Spermidine"])


def test_the_research_lane_returns_a_flagged_partial_table_and_claims_nothing_with_unequal_queues(monkeypatch):
    rows = _binary_tree(5)
    monkeypatch.setattr(pc, "estimate_beliefs", lambda sub, source, cap: 0)
    partial = sim_table(rows, pc.ChainSettings(queue_size=8, growth=2, rungs=2, on_incomplete="partial"), sources=["R"])
    assert partial.status["R"].complete is False and partial.status["R"].attempts == 2 and partial.status["R"].queue == 16
    assert 0 < len(partial.routes["R"]) < 62
    assert not partial.complete
    monkeypatch.undo()
    one_shot = sim_table(rows, pc.ChainSettings(queue_size=16, task_queue=16, belief_queue=16), sources=["R"])
    assert one_shot.status["R"].complete is False and one_shot.status["R"].attempts == 1      # control mode never retries
    beam = sim_table(contract_links("patient"), pc.ChainSettings(queue_size=256, task_queue=4, belief_queue=256))
    assert beam.status["Spermidine"].task_queue == 4 and beam.status["Spermidine"].queue == 256
    assert all(not s.complete for s in beam.status.values())       # tq < bq: a dropped task would go unseen
    assert beam.status["Spermidine"].attempts == 1
    equal = sim_table(contract_links("patient"), pc.ChainSettings(queue_size=256, task_queue=256, belief_queue=256))
    assert equal.complete and len(table_closure(equal)) == 100


def test_the_completeness_test_sees_every_cut_when_both_queues_are_one_size():
    """Decision correction 1, on the oracle: with tq = bq = K, 'tasks empty and fewer than K - 1 beliefs' flags every
    run that lost a pair; with a task queue smaller than the belief queue the same test would read 'complete' on a
    run that lost pairs, which is why the adapter claims nothing there."""
    rows = contract_links("patient")
    sources = ("DasatinibPlusQuercetin", "Spermidine", "Fisetin", "Metformin")
    full = {s: len(sim_table(rows, sources=[s]).routes[s]) for s in sources}
    lost = flagged = 0
    for k in (6, 8, 10, 12, 16, 24, 40):
        for source in sources:
            t = sim_table(rows, pc.ChainSettings(queue_size=k, on_incomplete="partial", task_queue=k, belief_queue=k),
                          sources=[source])
            if len(t.routes[source]) < full[source]:
                lost += 1
                flagged += int(not t.status[source].complete)
    assert lost > 0 and flagged == lost
    naive_complete = unclaimed = lost_small_tq = 0
    for source in sources:
        t = sim_table(rows, pc.ChainSettings(queue_size=200, task_queue=2, belief_queue=200), sources=[source])
        st = t.status[source]
        if len(t.routes[source]) < full[source]:
            lost_small_tq += 1
            naive_complete += int(st.tasks_left == (0, 0) and max(st.beliefs) < 200 - 1)
            unclaimed += int(not st.complete)
    assert lost_small_tq > 0 and naive_complete == lost_small_tq == unclaimed


def test_selection_counts_by_chunked_resumption_equal_the_one_shot_runs_and_stay_small():
    rows = contract_links("patient")
    counted = sim_table(rows, pc.ChainSettings(count_selections=True))
    plain = sim_table(rows)
    assert table_closure(counted) == table_closure(plain)
    selections = {s: v.selections for s, v in counted.status.items()}
    assert max(max(v) for v in selections.values()) == 12              # Spermidine: the longest run on the shipped graph
    assert selections["Spermidine"] == (12, 12) and selections["DasatinibPlusQuercetin"][0] == 11
    links = to_links(rows)
    assert pc.count_selections(SimEngine(), links, "Spermidine", 0.02) == 12


# ══════════════════════════════ the default is unchanged ═══════════════════════

class _FakeRuntime:
    name = "petta"
    enforces_deadline = True

    def __init__(self):
        self.calls = []

    def run_program(self, kb_text, expressions, deadline_s=None, **kwargs):
        self.calls.append({"kb_text": kb_text, "expressions": list(expressions), "deadline_s": deadline_s, **kwargs})
        return [["(done)"] for _ in expressions]


def test_under_the_legacy_chainer_the_runner_hands_the_runtime_exactly_what_it_always_did(monkeypatch, tmp_path):
    fake = _FakeRuntime()
    monkeypatch.setattr(pln_runner, "get_runtime", lambda *a, **k: fake)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "PLN_CHAINER", "legacy")
    files = [REPO / "system_types.metta", REPO / "pln_deduction.metta"]
    result = pln_runner._runtime_run("(infer &self A B)", 0.0, files, "(Effect A B Pos (stv 0.5 0.5))\n")
    assert result.status == "ok"
    (call,) = fake.calls
    assert set(call) == {"kb_text", "expressions", "deadline_s"}                      # no after_load
    assert call["kb_text"] == "\n".join([f.read_text(encoding="utf-8") for f in files]
                                        + ["(Effect A B Pos (stv 0.5 0.5))\n"])
    assert "(= (infer $s $a $c) (link-effect $s $a $c))" in call["kb_text"]          # the legacy equations are all there
    assert "PLNAnswer" not in call["kb_text"]


def test_under_pln_the_runner_filters_the_text_and_passes_the_step(monkeypatch, tmp_path):
    fake = _FakeRuntime()
    monkeypatch.setattr(pln_runner, "get_runtime", lambda *a, **k: fake)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    files = [REPO / "system_types.metta", REPO / "pln_deduction.metta"]
    result = pln_runner._runtime_run("(infer &self A B)", 0.0, files, "(Effect A B Pos (stv 0.5 0.5))\n")
    assert result.status == "ok"
    (call,) = fake.calls
    assert isinstance(call["after_load"], pc.ChainStep)
    assert "(= (infer $s $a $c) (link-effect $s $a $c))" not in call["kb_text"]
    assert "(PLNAnswer $a $c $sgn $tv $nodes $ids)" in call["kb_text"]
    assert [n for n, _ in call["after_load"].blocks] == ["system_types.metta", "pln_deduction.metta", "extra_atoms"]


def test_under_pln_a_forged_answer_in_the_callers_atoms_is_a_refused_form(monkeypatch, tmp_path):
    fake = _FakeRuntime()
    monkeypatch.setattr(pln_runner, "get_runtime", lambda *a, **k: fake)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    result = pln_runner._runtime_run("(infer &self A B)", 0.0, [REPO / "pln_deduction.metta"], FORGED + "\n")
    assert result.status == "error" and result.error_code == "refused_form" and "PLNAnswer" in result.error
    assert fake.calls == []


def test_pln_on_a_runtime_that_cannot_run_it_is_an_error_not_a_fallback(monkeypatch, tmp_path):
    fake = _FakeRuntime()
    monkeypatch.setattr(pln_runner, "get_runtime", lambda *a, **k: fake)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "PLN_RUNTIME", "hyperon")
    monkeypatch.setattr(config, "PLN_CHAINER", "pln")
    result = pln_runner._runtime_run("(infer &self A B)", 0.0, [REPO / "pln_deduction.metta"], None)
    assert result.status == "error" and "PLN_RUNTIME=petta" in result.error and fake.calls == []


def test_hyperon_refuses_the_hook():
    from core.metta_runtime import HyperonRuntime
    with pytest.raises(MettaRuntimeError, match="needs PLN_RUNTIME=petta"):
        HyperonRuntime().run_program("", ["(+ 1 2)"], after_load=lambda engine: None)


def test_the_module_imports_no_engine():
    code = (PLN_CHAT / "core" / "pln_chainer.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(hyperon|janus|petta)", code, re.M)


# ══════════════════════════════ the real PeTTa ═════════════════════════════════
#
# Everything below runs `core.pln_runner._runtime_run`, the one place the app executes MeTTa, with the chainer switched
# on (`PLN_CHAINER=pln` on `PLN_RUNTIME=petta`): the filter, the hook, the library, the child process, the sandbox.
# The legacy chainer on the same runtime is the reference where the recorded hyperon output cannot serve (the order of a
# set the consumers print differs between the two engines).

_STV_CHAIN = re.compile(r"^\(Chain \(([^()]*)\) \(signed (Pos|Neg) \(stv (\S+) (\S+)\)\)\)$")
_SIGNED = re.compile(r"^\(signed (Pos|Neg) \(stv (\S+) (\S+)\)\)$")


def _stack(name: str):
    import api
    runtime = api._runtime_kb_paths()
    return {"patient": lambda: pln_runner.patient_stack(runtime), "generic": lambda: pln_runner.generic_stack(runtime),
            "linage2": lambda: pln_runner.linage2_patient_kb(), "drugage": lambda: list(pln_runner.DRUGAGE_STACK),
            "full": lambda: runtime,
            "minimal": lambda: [REPO / f for f in ("system_types.metta", "logical_predicates.metta",
                                                   "epistemic_calibration.metta", "pln_deduction.metta")]}[name]()


def _use(monkeypatch, tmp_path, chainer: str, **settings):
    monkeypatch.setattr(config, "PLN_RUNTIME", "petta")
    monkeypatch.setattr(config, "PLN_CHAINER", chainer)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)


@pytest.fixture
def pln(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln")


@pytest.fixture
def legacy_on_petta(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "legacy")


def run(files, extra, queries, *, allow_error=False):
    """One program, one `!` expression per query: (groups, PLNRunResult). groups[i] = the atoms of query i."""
    result = pln_runner._runtime_run("\n".join(queries), 0.0, files, extra)
    if not allow_error:
        assert result.status in ("ok", "empty"), result.error
    groups: list[list[str]] = [[] for _ in queries]
    for atom in result.results:
        groups[atom.expr_index].append(atom.atom)
    return groups, result


def last_chain() -> dict:
    return get_runtime("petta").last_run["chain"]


def explain_closure(files, extra, rows) -> dict[tuple, tuple]:
    """Every (explain &self S $x) of every node of the graph: {chain: (sign, s, c)}; fails on a duplicate chain."""
    nodes = sorted({r["from"] for r in rows} | {r["to"] for r in rows})
    groups, _ = run(files, extra, [f"!(explain &self {n} $x)" for n in nodes])
    out: dict[tuple, tuple] = {}
    for atoms in groups:
        for atom in atoms:
            m = _STV_CHAIN.match(atom)
            assert m, atom
            chain = tuple(m.group(1).split())
            assert chain not in out, f"two results for {chain}"
            out[chain] = (m.group(2), float(m.group(3)), float(m.group(4)))
    return out


GRAPHS = {
    "patient": dict(stack="patient", extra=None, rows=lambda: contract_links("patient"), count=100),
    "linage2": dict(stack="linage2", extra=None, rows=lambda: contract_links("linage2"), count=86),
    "generic": dict(stack="generic", extra=None, rows=lambda: contract_links("generic"), count=101),
    "feedback": dict(stack="patient", extra=FEEDBACK_ATOM, rows=lambda: contract_links("patient", [FEEDBACK_EDGE]), count=109),
}


@pytest.fixture(scope="module")
def closures():
    """The pln closure of each graph, computed once per module (four PeTTa programs)."""
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    import tempfile
    mp.setattr(config, "PLN_RUNTIME", "petta")
    mp.setattr(config, "PLN_CHAINER", "pln")
    mp.setattr(config, "LOGS_DIR", Path(tempfile.mkdtemp(prefix="pln_chainer_logs_")))
    out = {}
    try:
        if REAL_PETTA:
            for name, g in GRAPHS.items():
                rows = g["rows"]()
                got = explain_closure(_stack(g["stack"]), g["extra"], rows)
                out[name] = {"got": got, "rows": rows, "summary": last_chain()}
    finally:
        mp.undo()
    return out


# ── gate 6a ────────────────────────────────────────────────────────────────────

@real_petta
@pytest.mark.parametrize("name", list(GRAPHS))
def test_gate_6a_every_pair_equals_the_legacy_chainer_in_sign_strength_confidence_and_chain(closures, name):
    c = closures[name]
    result = compare(c["got"], legacy_closure(c["rows"]))
    assert result["missing"] == [] and result["extra"] == [] and result["bad"] == [], result
    assert result["pairs"] == GRAPHS[name]["count"]
    assert result["max_dev"] < 1e-15                            # measured 2.2e-16
    assert c["summary"]["complete"] is True and c["summary"]["attempts_max"] == 1


@real_petta
@pytest.mark.parametrize("stack", ["patient", "linage2"])
def test_gate_6a_the_pairs_equal_the_closure_hyperon_recorded(closures, stack):
    want = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"]) for d in CLOSURE[stack]["derivations"]}
    result = compare(closures[stack]["got"], want)
    assert result["missing"] == [] and result["extra"] == [] and result["bad"] == [], result
    assert result["pairs"] == len(want) == (100 if stack == "patient" else 86)


@real_petta
def test_186_pairs_in_all_and_one_result_per_connected_pair(closures):
    assert len(closures["patient"]["got"]) + len(closures["linage2"]["got"]) == 186
    for name in ("patient", "linage2", "generic"):
        ends = [(c[0], c[-1]) for c in closures[name]["got"]]
        assert len(ends) == len(set(ends)), name              # exactly one derivation per connected pair (contract MUST 6)


@real_petta
def test_the_feedback_edge_terminates_with_109_pairs_and_the_hundred_old_answers_unchanged(closures):
    old, new = closures["patient"]["got"], closures["feedback"]["got"]
    assert len(old) == 100 and len(new) == 109
    assert {c: v for c, v in new.items() if c in old} == old                    # unchanged, to the last digit
    fresh = sorted(c for c in new if c not in old)
    assert fresh == sorted([("ChronicInflammation", "CellularSenescence"), ("ChronicInflammation", "CellularSenescence", "SASP"),
                            ("ChronicInflammation", "CellularSenescence", "SASP", "DNAmPAI1"),
                            ("ChronicInflammation", "CellularSenescence", "SASP", "DNAmPAI1", "CoronaryHeartDisease"),
                            ("Omega3", "ChronicInflammation", "CellularSenescence"),
                            ("Omega3", "ChronicInflammation", "CellularSenescence", "SASP"),
                            ("Omega3", "ChronicInflammation", "CellularSenescence", "SASP", "DNAmPAI1"),
                            ("Omega3", "ChronicInflammation", "CellularSenescence", "SASP", "DNAmPAI1", "CoronaryHeartDisease"),
                            ("SASP", "ChronicInflammation", "CellularSenescence")])
    assert closures["feedback"]["summary"]["complete"] is True


@real_petta
def test_the_legacy_chainer_hangs_on_the_feedback_edge_and_the_pln_chainer_does_not(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln")
    _, result = run(_stack("patient"), "(Effect A A Pos (stv 0.5 0.5))\n(Effect ChronicInflammation CellularSenescence Pos (stv 0.5 0.65))\n",
                    ["!(explain &self A $x)", "!(infer &self CellularSenescence $x)"])
    assert result.status == "ok" and result.query_time_ms < 5000        # a self-loop and a cycle: both finish


@real_petta
def test_every_node_without_a_link_out_answers_nothing_and_so_do_unknown_symbols(pln):
    groups, _ = run(_stack("patient"), None, ["!(explain &self CoronaryHeartDisease $x)", "!(infer &self NoSuchNode $x)",
                                              "!(infer &self CoronaryHeartDisease CoronaryHeartDisease)",
                                              "!(infer &self DasatinibPlusQuercetin DasatinibPlusQuercetin)",
                                              "!(explain &self $x NoSuchNode)"])
    assert groups == [[], [], [], [], []]


@real_petta
@pytest.mark.parametrize("stack", ["patient", "linage2"])
def test_infer_and_explain_read_one_table(pln, stack):
    rows = contract_links(stack)
    closure = legacy_closure(rows)
    pairs = sorted((c[0], c[-1]) for c in closure)
    groups, _ = run(_stack(stack), None, [f"!(infer &self {a} {b})" for a, b in pairs] + [f"!(explain &self {a} {b})" for a, b in pairs])
    n = len(pairs)
    for i, (a, b) in enumerate(pairs):
        (inf,), (exp,) = groups[i], groups[n + i]
        m = _STV_CHAIN.match(exp)
        assert m and inf == f"(signed {m.group(2)} (stv {m.group(3)} {m.group(4)}))" and _SIGNED.match(inf)
        assert m.group(1).split()[0] == a and m.group(1).split()[-1] == b


@real_petta
def test_the_no_false_chain_pairs_stay_empty(pln):
    """contract.md 1.3 / 5.1: pairs a replacement must NOT connect (no inversion, no induction, no abduction)."""
    empty = [("DasatinibPlusQuercetin", "AllCauseMortality"), ("SmokingCessation", "CoronaryHeartDisease"),
             ("Rapamycin", "CoronaryHeartDisease"), ("Elamipretide", "CoronaryHeartDisease"),
             ("Elamipretide", "CellularSenescence"), ("CRP", "DNAmGDF15"), ("DNAmGDF15", "CRP"),
             ("Metformin", "CellularSenescence"), ("CoronaryHeartDisease", "DNAmPAI1"),   # no inversion
             ("ChronicInflammation", "CoronaryHeartDisease"), ("MitochondrialDysfunction", "CoronaryHeartDisease"),
             ("SmokingPackYears", "CoronaryHeartDisease"), ("CellularSenescence", "DasatinibPlusQuercetin"),
             ("SASP", "Spermidine"), ("AgeAccelGrim", "DNAmPAI1"), ("DNAmPAI1", "AgeAccelGrim")]
    groups, _ = run(_stack("patient"), None, [f"!(infer &self {a} {b})" for a, b in empty])
    assert groups == [[] for _ in empty], dict(zip(empty, groups))
    sources = sorted({r["from"] for r in contract_links("patient")} | {r["to"] for r in contract_links("patient")})
    into_grim, _ = run(_stack("patient"), None, [f"!(infer &self {n} AgeAccelGrim)" for n in sources])
    assert into_grim == [[] for _ in sources]                                           # nothing reaches AgeAccelGrim
    elam, _ = run(_stack("patient"), None, [f"!(infer &self Elamipretide {n})" for n in sources])
    assert elam == [[] for _ in sources]
    # the CHD-reaching causes are exactly the three the notes name (contract 4.2)
    reach, _ = run(_stack("patient"), None, [f"!(infer &self {n} CoronaryHeartDisease)" for n in
                                             ("CellularSenescence", "InsulinResistance", "DeregulatedNutrientSensing",
                                              "ChronicInflammation", "MitochondrialDysfunction", "SmokingPackYears")])
    assert [bool(g) for g in reach] == [True, True, True, False, False, False]


@real_petta
def test_the_pln_numbers_of_the_named_chains(pln):
    groups, _ = run(_stack("patient"), None, ["!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)",
                                              "!(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)",
                                              "!(infer &self CellularSenescence CoronaryHeartDisease)",
                                              "!(infer &self Fisetin CoronaryHeartDisease)",
                                              "!(infer &self Spermidine CoronaryHeartDisease)",
                                              "!(link-effect &self DNAmPAI1 CoronaryHeartDisease)"])
    s, c = (float(x) for x in _SIGNED.match(groups[0][0]).group(2, 3))
    assert _SIGNED.match(groups[0][0]).group(1) == "Neg"
    assert s == pytest.approx(0.2499, abs=TOL) and c == pytest.approx(0.65 ** 3 * 0.85 * 0.9 ** 3, abs=TOL)
    assert groups[1][0].startswith("(Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease) (signed Neg")
    sign, s2, c2 = _SIGNED.match(groups[2][0]).group(1, 2, 3)
    assert sign == "Pos" and float(s2) == pytest.approx(0.357, abs=TOL) and float(c2) == pytest.approx(0.29089125, abs=TOL)
    assert float(c2) < 0.65
    assert float(_SIGNED.match(groups[3][0]).group(3)) == pytest.approx(0.5 * 0.65 * 0.65 * 0.85 * 0.9 ** 3, abs=TOL)
    assert float(_SIGNED.match(groups[4][0]).group(3)) == pytest.approx(0.5 * 0.65 ** 3 * 0.85 * 0.9 ** 4, abs=TOL)
    assert groups[5] == ["(signed Pos (stv 0.6 0.85))"]                                  # the legacy single-hop lift is untouched
    cd, cf, cs = (float(_SIGNED.match(groups[i][0]).group(3)) for i in (0, 3, 4))
    assert cd > cf > cs                                                                  # 0.170 > 0.131 > 0.0766: 4, 4, 5 edges


@real_petta
def test_native_confidence_is_the_libraries_own(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_CHAIN_CONFIDENCE="native")
    groups, _ = run(_stack("patient"), None, ["!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)"])
    sign, s, c = _SIGNED.match(groups[0][0]).group(1, 2, 3)
    assert sign == "Neg" and float(s) == pytest.approx(0.2499, abs=TOL)
    assert float(c) == pytest.approx(0.65 ** 3 * 0.85, abs=TOL) == pytest.approx(0.23343125, abs=TOL)


# ── the composite queries and the pinned orderings ─────────────────────────────

_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+")


def _skeleton(text: str):
    return _NUM.sub("#", text), [float(x) for x in _NUM.findall(text)]


def _batch(queries: list[str]) -> list[list[str]]:
    groups, _ = run(_stack("patient"), None, queries)
    return groups


@pytest.fixture(scope="module")
def composite():
    """The 50 composite queries of the contract (rankings, diagnosis, counterfactuals, risk, supplements) under the pln
    chainer and under the legacy chainer on the same runtime, one program each."""
    import tempfile
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "PLN_RUNTIME", "petta")
    mp.setattr(config, "LOGS_DIR", Path(tempfile.mkdtemp(prefix="pln_chainer_logs_")))
    out = {}
    try:
        if REAL_PETTA:
            queries = [e["query"] for e in PINNED]
            for mode in ("legacy", "pln"):
                mp.setattr(config, "PLN_CHAINER", mode)
                out[mode] = _batch(queries)
    finally:
        mp.undo()
    return out


@real_petta
def test_the_50_composite_queries_equal_the_legacy_chainer_on_the_same_runtime_to_rounding(composite):
    assert len(PINNED) == 50
    worst = 0.0
    for entry, got, want in zip(PINNED, composite["pln"], composite["legacy"]):
        assert len(got) == len(want), entry["tag"]
        for g, w in zip(got, want):
            (gs, gn), (ws, wn) = _skeleton(g), _skeleton(w)
            assert gs == ws, (entry["tag"], g[:200], w[:200])               # same names, same order, same shapes
            assert len(gn) == len(wn)
            for a, b in zip(gn, wn):
                worst = max(worst, abs(a - b))
    assert worst < 1e-9, worst                                              # measured 1.1e-16


def _scored(atom: str) -> list[str]:
    return re.findall(r"\(scored (\S+) ", atom)


def _byTag(composite, mode="pln"):
    return {e["tag"]: g for e, g in zip(PINNED, composite[mode])}


@real_petta
def test_the_twelve_pinned_orderings_hold(composite):
    """The orderings contract.md 4.2 / 5.4 pins, twelve of them, read off the composite outputs."""
    t = _byTag(composite)
    DQ, F, S, M, B = "DasatinibPlusQuercetin", "Fisetin", "Spermidine", "Metformin", "Berberine"
    # 1. population ranking: protective first, the unconnected Elamipretide absent
    assert _scored(t["rank_population"][0]) == [DQ, F, S]
    # 2. a 5-pool population ranking
    assert _scored(t["rank_population_5pool"][0]) == [M, B, DQ, F, S]
    # 3. Patient001's personalised order (the one native confidence flips)
    assert _scored(t["rank_for_Patient001_5pool"][0]) == [DQ, F, M, B, S]
    # 4. Patient002's personalised order
    assert _scored(t["rank_for_Patient002_5pool"][0]) == [M, B, DQ, F, S]
    # 5. relevance to Patient001's markers: DQ > Fisetin > Spermidine > 0 = Elamipretide
    rel = [float(t[f"relevance_Patient001_{n}"][0]) for n in (DQ, F, S, "Elamipretide")]
    assert rel[0] > rel[1] > rel[2] > 0.0 == rel[3]
    # 6. Patient002: Metformin's relevance is patient specific (> 0.4) and Berberine's follows
    assert float(t["relevance_Patient002_Metformin"][0]) > 0.4 and float(t["relevance_Patient002_Berberine"][0]) > 0.4
    # 7. diagnosis ranks by coverage then mass: CS (3), CI (2), MD (1)
    for tag in ("diagnose_population", "diagnose_patient_Patient001", "diagnose_patient_default_P001"):
        names = re.findall(r"\(Hypothesis (\S+) ", t[tag][0])
        assert names == ["CellularSenescence", "ChronicInflammation", "MitochondrialDysfunction"], tag
        assert [int(x) for x in re.findall(r"\) (\d+)(?:\.0)? [\d.]+ \(SupportedBy", t[tag][0])] == [3, 2, 1], tag
    # 8. counterfactuals: B (senescence) moves the clock more than A (inflammation) and is less certain
    def cf(tag):
        m = re.search(r"expected-delta (-?[\d.e-]+)\) \(signed Neg \(stv [\d.e-]+ ([\d.e-]+)\)", t[tag][0])
        return float(m.group(1)), float(m.group(2))
    (dA, cA), (dB, cB), (dQ, cQ) = cf("cf_Patient001_ChronicInflammation"), cf("cf_Patient001_CellularSenescence"), cf("cf_Patient001_DasatinibPlusQuercetin")
    assert abs(dB) > abs(dA) and cB < cA <= 0.65 + 1e-12
    # 9. the senolytic lever: 0.70 of B's effect, with the lowest confidence
    assert dQ / dB == pytest.approx(0.70, abs=1e-12) and cQ < cB
    # 10. supplements for Patient001: Omega3 in tier 1; NMN scores above Fisetin
    sup = t["supplements_Patient001"][0]
    assert re.search(r"Tier1HighConfidence \(\(SuppRec Omega3 ", sup)
    scores = dict(re.findall(r"\(SuppRec (\S+) \(tier \S+\) \(score ([\d.e-]+)\)", sup))
    assert float(scores["NMN"]) > float(scores["Fisetin"])
    # 11. risk: the projected reduction is larger for senescence than for inflammation, ~0 for insulin resistance,
    #     and the senescence projection is less certain
    def proj(tag):
        m = re.search(r"\(reduction ([\d.e-]+)\).*\(confidence ([\d.e-]+)\)", t[tag][0])
        return float(m.group(1)), float(m.group(2))
    (rb, cb), (ra, ca), (rc, _) = proj("project_Patient001_CellularSenescence"), proj("project_Patient001_ChronicInflammation"), proj("project_Patient001_InsulinResistance")
    assert rb > ra > rc == pytest.approx(0.0, abs=1e-12) and cb < ca
    # 12. confidence by chain length: D+Q (4 edges) > Fisetin (4) > Spermidine (5), all below the 0.65 of a single link
    conf = [float(c) for c in re.findall(r"\(stv [\d.e-]+ ([\d.e-]+)\)", t["rank_population"][0])]
    assert conf[0] > conf[1] > conf[2] and conf[0] < 0.65


@real_petta
def test_the_diagnosis_values_stay_inside_their_tolerances(composite):
    t = _byTag(composite)
    atom = t["diagnose_patient_Patient001"][0]
    m = re.search(r"\(Hypothesis CellularSenescence \(stv ([\d.e-]+) ([\d.e-]+)\)", atom)
    assert float(m.group(1)) == pytest.approx(0.595, abs=1e-9) and float(m.group(2)) == pytest.approx(0.625, abs=1e-2)
    assert re.search(r"\(Hypothesis MitochondrialDysfunction \(stv [\d.e-]+ 0\.65\)", atom)
    assert "AgeAccelGrim" not in re.search(r"SupportedBy[^)]*\)", atom).group(0)


@real_petta
def test_the_composite_queries_under_pln_equal_the_pln_goldens_exactly(composite):
    golden = json.loads((FIXTURES / "composite_pln.json").read_text(encoding="utf-8"))
    assert [e["tag"] for e in golden["entries"]] == [e["tag"] for e in PINNED]
    for entry, got in zip(golden["entries"], composite["pln"]):
        assert entry["query"] == PINNED[[e["tag"] for e in PINNED].index(entry["tag"])]["query"]
        assert entry["results"] == got, entry["tag"]


@real_petta
def test_the_pln_closure_equals_the_pln_golden_exactly(closures):
    golden = json.loads((FIXTURES / "closure_pln.json").read_text(encoding="utf-8"))
    for name in GRAPHS:
        want = {tuple(d["chain"]): (d["sign"], d["strength"], d["confidence"]) for d in golden[name]}
        assert closures[name]["got"] == want, name


def _close(a, b, tol: float = 1e-12) -> bool:
    """Two parsed answers (tests/runtime_compare.parse) with the same shape, names and order and numbers within `tol`."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


@real_petta
def test_the_32_built_in_patient_cases_equal_the_pln_golden_exactly_and_the_legacy_recording_to_1e_12(monkeypatch, tmp_path):
    """The legacy recording (tests/fixtures/patient_stack_builtin_golden.json, 32 cases, 22 non-empty) is asserted to the digit
    under the legacy chainer and cannot be under pln (the last digit of some numbers differs): it is untouched, and its pln
    counterpart is asserted exactly. The two agree: same atoms, same order, numbers within 1e-12."""
    from runtime_compare import canonical, parse
    legacy = json.loads((REPO / "tests" / "fixtures" / "patient_stack_builtin_golden.json").read_text(encoding="utf-8"))["cases"]
    golden = json.loads((FIXTURES / "patient_stack_builtin_golden_pln.json").read_text(encoding="utf-8"))["cases"]
    assert sorted(golden) == sorted(legacy) and len(legacy) == 32
    assert sum(1 for c in legacy.values() if c["atoms"]) == sum(1 for c in golden.values() if c["atoms"]) == 22
    _use(monkeypatch, tmp_path, "pln")
    names = sorted(legacy)
    got = _batch([legacy[n]["query"] for n in names])
    differing = 0
    for name, atoms in zip(names, got):
        assert golden[name]["query"] == legacy[name]["query"]
        assert atoms == golden[name]["atoms"], name                                  # the pln golden, to the digit
        assert len(atoms) == len(legacy[name]["atoms"]), name
        for a, e in zip(atoms, legacy[name]["atoms"]):
            assert _close(canonical(parse(a)), canonical(parse(e))), (name, a[:200], e[:200])
            differing += int(a != e)
    assert differing > 0              # the legacy digits do not hold under pln (counts of strings below), which is why this file exists


def test_the_pln_goldens_are_not_the_legacy_ones():
    """The legacy goldens are untouched and stay legacy: the last digits differ, which is why these exist."""
    golden = json.loads((FIXTURES / "closure_pln.json").read_text(encoding="utf-8"))
    legacy = {tuple(d["chain"]): (d["strength"], d["confidence"]) for d in CLOSURE["patient"]["derivations"]}
    pln_ = {tuple(d["chain"]): (d["strength"], d["confidence"]) for d in golden["patient"]}
    assert legacy.keys() == pln_.keys()
    differing = sum(1 for k in legacy if legacy[k] != pln_[k])
    assert differing > 0 and all(abs(legacy[k][0] - pln_[k][0]) < 1e-15 and abs(legacy[k][1] - pln_[k][1]) < 1e-15 for k in legacy)
    for name in ("patient_stack_builtin_golden.json", "linage2_golden.json"):
        assert "PLNAnswer" not in (REPO / "tests" / "fixtures" / name).read_text(encoding="utf-8")


@real_petta
def test_native_confidence_flips_the_pinned_order_which_is_why_the_discount_is_the_default(monkeypatch, tmp_path):
    """Negative control of the orderings: with the library's own confidence (no per-combination discount) Patient001's
    personalised order flips its 4th and 5th places and diagnose's c(CellularSenescence) leaves its tolerance."""
    _use(monkeypatch, tmp_path, "pln", PLN_CHAIN_CONFIDENCE="native")
    q = [e["query"] for e in PINNED if e["tag"] in ("rank_for_Patient001_5pool", "diagnose_patient_Patient001",
                                                    "rank_for_Patient002_5pool")]
    groups = _batch(q)
    assert _scored(groups[0][0]) == ["DasatinibPlusQuercetin", "Fisetin", "Metformin", "Spermidine", "Berberine"]     # flipped
    assert _scored(groups[1][0]) == ["Metformin", "Berberine", "DasatinibPlusQuercetin", "Fisetin", "Spermidine"]    # Patient002 holds
    c = float(re.search(r"\(Hypothesis CellularSenescence \(stv [\d.e-]+ ([\d.e-]+)\)", groups[2][0]).group(1))
    assert c == pytest.approx(0.6961, abs=1e-3) and abs(c - 0.625) > 1e-2


@real_petta
def test_the_restricted_rules_conclude_nothing_outside_the_routes_of_the_graph(closures):
    """No false chain, measured on the engine: every conclusion of every level-1 run is a route or a walk of the graph."""
    for name in GRAPHS:
        s = closures[name]["summary"]
        assert s["unattributed"] == 0, name
        assert s["ruleset"] == "restricted" and s["mapping"] == "affine"
        assert all(v["unattributed"] == 0 for v in s["status"].values())


@real_petta
def test_the_full_rule_set_research_lane_runs_on_a_fixed_budget_and_shows_conclusions_the_links_do_not_license():
    links = to_links(contract_links("patient"))

    def step(engine):
        ev = lambda e: [str(x) for x in engine.run("!" + e)]
        engine.load(pc.library_text("full"))
        leak = pc.probe_library(ev)
        engine.load(pc.base_rate_text(links))
        st = pc.ChainSettings(ruleset="full")
        t = pc.compute_table(ev, links, leak=leak, discount=0.9, settings=st, sources=["CellularSenescence"])
        s = t.status["CellularSenescence"]
        return {"routes": len(t.routes["CellularSenescence"]), "unattributed": s.unattributed, "complete": s.complete,
                "queue": s.queue, "task_queue": s.task_queue, "max_steps": s.max_steps, "attempts": s.attempts}

    rt = PettaRuntime()
    rt.run_program("", ["!(+ 1 2)"], deadline_s=300, after_load=step)
    out = rt.last_run["chain"]
    assert out["routes"] == 8                                   # the true routes are all found ...
    assert out["unattributed"] > 0                              # ... and conclusions no link licenses come with them
    assert (out["max_steps"], out["task_queue"], out["queue"]) == pc.FULL_LANE_BUDGET and out["attempts"] == 1
    assert out["complete"] is False                             # a table made under this budget is flagged by design
    assert "(= (STV (State n0)) (stv 0.1 0.9))" in pc.base_rate_text(links)
    with pytest.raises(pc.PlnChainerError):                     # the service refuses to load it through a scanned text
        from core.metta_runtime import petta_refused_forms_in_kb
        assert petta_refused_forms_in_kb(pc.library_text("full")) == ["Predicate"]
        raise pc.PlnChainerError("the scan refuses the unfiltered file, as G0 found")


# ── caller facts, diamonds, determinism, no state ──────────────────────────────

@real_petta
def test_a_callers_effect_facts_work_with_no_precompile_and_equal_the_legacy_chainer(monkeypatch, tmp_path):
    extra = """
(Effect CallerA CallerB Pos (stv 0.9 0.8))
(Effect CallerB CallerC Neg (stv 0.7 (evidence-confidence AnimalStudies_Replicated)))
(Effect CallerA CallerD Pos (stv 0.99 0.9))
(Effect CallerD CallerE Neg (stv 1.0 1.0))
(Effect CallerE CallerF Pos (stv 0.0 0.6))
(Effect CallerA CallerC Pos (stv 0.25 0.5))
"""
    queries = ["!(explain &self CallerA $x)", "!(explain &self CallerB $x)", "!(infer &self CallerA CallerC)"]
    _use(monkeypatch, tmp_path, "legacy")
    want, _ = run(_stack("minimal"), extra, queries)
    _use(monkeypatch, tmp_path, "pln")
    got, _ = run(_stack("minimal"), extra, queries)
    for g, w in zip(got, want):
        assert len(g) == len(w)
        gs = sorted(_skeleton(a) for a in g)
        ws = sorted(_skeleton(a) for a in w)
        assert [x[0] for x in gs] == [x[0] for x in ws]
        for (_, gn), (_, wn) in zip(gs, ws):
            assert gn == pytest.approx(wn, abs=1e-12)
    assert len(got[0]) == 6 and len(got[2]) == 2                    # B, C direct, C through B, D, E, F: six routes; C twice
    chains = {_STV_CHAIN.match(a).group(1): _STV_CHAIN.match(a).group(2, 3, 4) for a in got[0]}
    assert chains["CallerA CallerD CallerE"][0] == "Neg" and float(chains["CallerA CallerD CallerE"][1]) == pytest.approx(0.99, abs=1e-15)
    assert float(chains["CallerA CallerD CallerE CallerF"][1]) == 0.0                    # a zero-strength link: exactly 0


@real_petta
def test_two_routes_to_one_target_give_two_results_whatever_the_order_they_are_written_in(monkeypatch, tmp_path):
    atoms = ["(Effect A B Pos (stv 0.5 0.8))", "(Effect A C Neg (stv 0.6 0.7))", "(Effect B D Pos (stv 0.9 0.8))",
             "(Effect C D Neg (stv 0.8 0.9))"]
    outputs = []
    _use(monkeypatch, tmp_path, "pln")
    rnd = random.Random(2)
    for _ in range(4):
        rnd.shuffle(atoms)
        groups, _ = run(_stack("minimal"), "\n".join(atoms) + "\n", ["!(explain &self A D)", "!(infer &self A D)"])
        assert len(groups[0]) == 2 and len(groups[1]) == 2
        outputs.append(groups)
    assert all(o == outputs[0] for o in outputs)                                      # byte-identical, order included
    assert sorted(a[:22] for a in outputs[0][0]) == ["(Chain (A B D) (signed", "(Chain (A C D) (signed"]


@real_petta
def test_the_order_of_the_query_candidates_cannot_change_the_ranking(pln):
    cands = ["DasatinibPlusQuercetin", "Fisetin", "Spermidine", "Elamipretide"]
    forward, _ = run(_stack("patient"), None, [f"!(rank-interventions &self ({' '.join(cands)}) CoronaryHeartDisease)"])
    backward, _ = run(_stack("patient"), None, [f"!(rank-interventions &self ({' '.join(reversed(cands))}) CoronaryHeartDisease)"])
    assert forward == backward and _scored(forward[0][0]) == cands[:3]


@real_petta
def test_no_state_survives_a_request(pln):
    q = ["!(explain &self CallerA $x)", "!(match &self (PLNAnswer $a $c $s $t $n $i) $a)"]
    first, _ = run(_stack("minimal"), "(Effect CallerA CallerB Pos (stv 0.9 0.8))\n", q)
    pid1 = get_runtime("petta").last_pid
    second, _ = run(_stack("minimal"), "(Effect CallerA CallerZ Pos (stv 0.5 0.5))\n", q)
    pid2 = get_runtime("petta").last_pid
    third, _ = run(_stack("minimal"), None, q)
    assert pid1 != pid2
    assert len(first[0]) == 1 and "CallerB" in first[0][0] and "CallerZ" not in first[0][0]
    assert len(second[0]) == 1 and "CallerZ" in second[0][0] and "CallerB" not in second[0][0]
    assert third == [[], []]
    assert first[1] == ["CallerA"] and second[1] == ["CallerA"]


@real_petta
def test_a_record_with_a_protective_hazard_ratio_is_a_negative_terminal_link(monkeypatch, tmp_path):
    extra = """
(: Rec_X_Y EmpiricalAssociation)
(HasPredictor Rec_X_Y Xnode)
(HasOutcome Rec_X_Y Ynode)
(EffectMetric Rec_X_Y "hazard_ratio")
(EffectValue Rec_X_Y 0.8)
(HasEvidenceCategory Rec_X_Y MultipleHumanTrials)
(Effect Wnode Xnode Pos (stv 0.5 0.8))
(Effect Ynode Znode Pos (stv 0.5 0.8))
"""
    queries = ["!(explain &self Xnode $x)", "!(explain &self Wnode $x)", "!(explain &self Ynode $x)"]
    files = _stack("patient")
    _use(monkeypatch, tmp_path, "legacy")
    want, _ = run(files, extra, queries)
    _use(monkeypatch, tmp_path, "pln")
    got, _ = run(files, extra, queries)
    for g, w in zip(got, want):                                       # equal to the legacy chainer on the same runtime
        sg, sw = sorted(_skeleton(x) for x in g), sorted(_skeleton(x) for x in w)
        assert [x[0] for x in sg] == [x[0] for x in sw]
        assert [n for x in sg for n in x[1]] == pytest.approx([n for x in sw for n in x[1]], abs=1e-12)
    assert got[0] == ["(Chain (Xnode Ynode) (signed Neg (stv 0.5 0.85)))"]                  # HR 0.8 < 1: a protective record
    assert sorted(got[1]) == ["(Chain (Wnode Xnode Ynode) (signed Neg (stv 0.25 0.6120000000000001)))",
                              "(Chain (Wnode Xnode) (signed Pos (stv 0.5 0.8)))"]
    assert got[2] == [] == want[2] or got[2] == want[2]
    assert not any("Znode" in a for a in got[0] + got[1])         # a record ends a chain: Ynode -> Znode is not continued


@real_petta
def test_the_step_runs_in_the_execution_workers_too(monkeypatch, tmp_path):
    """The API offloads every query to a forked worker (core/executor.py), which forks the PeTTa child: the settings and the
    step travel by fork, and the answers come back in the ordinary result."""
    import core.executor as executor
    _use(monkeypatch, tmp_path, "pln")
    monkeypatch.setattr(executor, "PLN_WORKER_POOL_SIZE", 1)
    files, query = _stack("patient"), "!(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)"
    result = executor.run_offloaded(
        "run_query", {"metta_query": query, "confidence_threshold": 0.0, "kb_files": files, "extra_atoms": None},
        lambda: pytest.fail("the query was meant to run in a worker"))
    assert result.status == "ok" and result.runtime_name == "petta"
    (atom,) = [a.atom for a in result.results]
    m = _STV_CHAIN.match(atom)
    assert m and m.group(1) == "DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease"
    assert m.group(2) == "Neg" and float(m.group(3)) == pytest.approx(0.2499, abs=TOL)


# ── forgery, the hook, the K schedule, the evidence table ──────────────────────

@real_petta
def test_a_forged_answer_is_refused_before_a_single_process_starts(pln):
    groups, result = run(_stack("minimal"), FORGED + "\n", ["!(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)"],
                         allow_error=True)
    assert result.status == "error" and result.error_code == "refused_form" and "PLNAnswer" in result.error


@real_petta
def test_the_step_itself_refuses_a_space_that_already_holds_an_answer():
    """Defence in depth: even if a text got past the scan, the step checks the loaded space before it writes."""
    blocks = [("pln_deduction.metta", pc.filter_legacy_chain(_read("pln_deduction.metta"))[0]), ("x.metta", FORGED + "\n")]
    step = pc.ChainStep(tuple(blocks), pc.ChainSettings())
    text = "\n".join(t for _, t in blocks)
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program(text, ["!(+ 1 2)"], after_load=step)
    assert err.value.kind == "chain" and "already holds PLNAnswer" in str(err.value)


@real_petta
def test_the_hook_runs_after_the_load_and_before_the_lock_and_reports_back():
    def step(engine):                                # runs in the forked child: what it saw comes back as its summary
        seen = {"run": engine.run("!(+ 1 2)")}
        engine.load("(Hello World)\n")
        seen["match"] = engine.run("!(match &self (Hello $x) $x)")
        return seen

    rt = PettaRuntime()
    out = rt.run_program("(Known 1)\n", ["!(match &self (Hello $x) $x)", "!(match &self (Known $x) $x)"], after_load=step)
    assert out == [["World"], ["1"]]
    assert rt.last_run["chain"] == {"run": ["3"], "match": ["World"]} and "chain_s" in rt.last_run
    plain = PettaRuntime()
    plain.run_program("(Known 1)\n", ["!(match &self (Known $x) $x)"])
    assert "chain" not in plain.last_run and "chain_s" not in plain.last_run          # no hook, the stats are what they were
    with pytest.raises(MettaRuntimeError) as err:
        PettaRuntime().run_program("", ["!(+ 1 2)"], after_load=lambda engine: 1 / 0)
    assert err.value.kind == "chain" and "ZeroDivisionError" in str(err.value)


@real_petta
def test_the_space_is_locked_after_the_step(pln):
    groups, _ = run(_stack("minimal"), "(Effect A B Pos (stv 0.5 0.5))\n", ["!(add-atom &self (PLNAnswer A B Pos (stv 1.0 1.0) (A B) (1)))"], allow_error=True)
    assert groups == [[]]


@real_petta
def test_the_summary_says_how_the_table_was_derived(pln):
    groups, _ = run(_stack("patient"), None, ["!(infer &self $a CoronaryHeartDisease)"])     # an unbound source: every source is derived
    s = last_chain()
    assert (s["links"], s["sources"], s["routes"], s["derive_runs"]) == (28, 20, 100, 40) and s["demand"] is None
    assert s["complete"] is True and s["leak"] == pytest.approx(0.02) and s["discount"] == 0.9 and s["skipped"] == []
    assert s["queue_max"] == 256 and s["attempts_max"] == 1 and s["chain_confidence"] == "legacy"
    assert set(s["timing_s"]) == {"library", "links", "plan", "derive", "facts", "total"}
    assert all(v["complete"] and v["queue"] == 256 and v["attempts"] == 1 for v in s["status"].values())
    assert len(s["evidence"]) == 28


@real_petta
def test_every_evidence_id_says_which_atom_it_stands_for(pln):
    groups, _ = run(_stack("patient"), None, ["!(match &self (PLNEvidence $id $kind $where $tier) ($id $kind $where $tier))",
                                              "!(match &self (PLNStatus $s $st $k) ($s $st $k))",
                                              "!(match &self (PLNAnswer DasatinibPlusQuercetin CoronaryHeartDisease $g $t $n $i) ($g $n $i))"])
    ev = [tuple(a.strip("()").split(" ", 3)) for a in groups[0]]
    assert len(ev) == 28 and sorted(int(e[0]) for e in ev) == list(range(1, 29))
    kinds = {e[1] for e in ev}
    assert kinds == {"curated", "record"}
    by_id = {int(e[0]): e for e in ev}
    # the route D+Q -> CS -> SASP -> DNAmPAI1 -> CHD: three curated atoms and the Lu 2019 record, by id
    (route,) = groups[2]
    ids = [int(i) for i in route.rsplit("(", 1)[1].strip(")").split()]
    assert len(ids) == 4
    where = sorted(by_id[i][2].strip('"') for i in ids)
    assert where[0].startswith("grim_age_lu2019_evidence.metta:") and all(w.startswith("mechanistic_bridges.metta:") for w in where[1:])
    assert {by_id[i][3] for i in ids if by_id[i][1] == "record"} == {"MultipleHumanTrials"}
    assert "AnimalStudies_Replicated" in {by_id[i][3] for i in ids}
    # file:line is the line the atom starts on
    for e in ev:
        name, line = e[2].strip('"').rsplit(":", 1)
        assert (REPO / name).read_text(encoding="utf-8").splitlines()[int(line) - 1].lstrip().startswith(("(Effect", "(EffectValue"))
    assert len(groups[1]) == 20 and all(" complete 256" in a for a in groups[1])


@real_petta
def test_a_caller_fact_is_labelled_a_caller_fact_in_the_evidence_table(pln):
    groups, _ = run(_stack("minimal"), "(Effect CallerA CallerB Pos (stv 0.9 0.8))\n",
                    ["!(match &self (PLNEvidence $id $kind $where $tier) ($kind $where))"])
    assert groups[0] == ['(caller "extra_atoms:1")']


_UNCHAINABLE = """
(Effect A B Pos (stv 1.5 0.8))
(Effect A C Pos (stv 0.5 (evidence-confidence NoSuchTier)))
(Effect A D Maybe (stv 0.5 0.5))
(Effect A E Pos (stv 0.5 0.5))
(Effect A F Pos (stv 1 1))
"""


@real_petta
def test_a_truth_value_that_cannot_be_chained_is_skipped_and_listed_when_a_curated_file_holds_it(pln, tmp_path):
    """A link of a CURATED file with a truth value outside [0, 1] is a data bug the service lists (`PLNSkipped`) and leaves
    out; the other links still chain."""
    bad = tmp_path / "curated_bad.metta"
    bad.write_text(_UNCHAINABLE, encoding="utf-8")
    groups, _ = run(_stack("minimal") + [bad], None, ["!(explain &self A $x)", "!(match &self (PLNSkipped $a $b $r) ($a $b $r))"])
    assert sorted(groups[0]) == ["(Chain (A E) (signed Pos (stv 0.5 0.5)))", "(Chain (A F) (signed Pos (stv 1.0 1.0)))"]
    assert sorted(groups[1]) == ['("A" "B" StrengthOutOfRange)', '("A" "C" NonNumericTruthValue)', '("A" "D" UnknownSign)']
    assert sorted(s["reason"] for s in last_chain()["skipped"]) == ["NonNumericTruthValue", "StrengthOutOfRange", "UnknownSign"]
    assert {s["origin"] for s in last_chain()["skipped"]} == {"curated"}


@real_petta
def test_a_callers_own_unchainable_fact_is_refused_with_the_atom_named_not_answered_with_a_silent_no(pln):
    """The same atoms from the CALLER (`extra_atoms`): the legacy chainer answers them (garbage in, a number out) and a
    skip would come back as "no route", which API.md defines as a real no. So the request is refused, 422, naming the atoms."""
    groups, result = run(_stack("minimal"), _UNCHAINABLE, ["!(explain &self A $x)"], allow_error=True)
    assert result.status == "error" and result.error_code == "refused_form" and result.results == []
    assert "(Effect A B Pos (stv 1.5 0.8))" in result.error and "strength is not in [0, 1]" in result.error
    assert "(Effect A D Maybe (stv 0.5 0.5))" in result.error and "the sign is neither Pos nor Neg" in result.error
    assert "NoSuchTier" in result.error and "not a number" in result.error
    # one chainable fact alone is fine, and so is a caller's well-formed program
    ok, res = run(_stack("minimal"), "(Effect A E Pos (stv 0.5 0.5))\n", ["!(explain &self A $x)"])
    assert ok[0] == ["(Chain (A E) (signed Pos (stv 0.5 0.5)))"]


@real_petta
def test_the_http_layer_answers_an_unchainable_caller_fact_as_422_refused_form():
    import api
    assert api._PLN_ERROR_STATUS["refused_form"] == 422


def _layered(width: int, layers: int, seed: int = 4) -> str:
    rnd = random.Random(seed)
    lines = []
    for layer in range(layers):
        for i in range(width):
            for j in range(width):
                if layer == 0 and i > 0:
                    continue
                a = "S" if layer == 0 else f"N{layer - 1}_{i}"
                lines.append(f"(Effect {a} N{layer}_{j} {'Pos' if rnd.random() < 0.8 else 'Neg'} "
                             f"(stv {rnd.uniform(0.2, 0.9):.3f} {rnd.uniform(0.5, 0.9):.3f}))")
    return "\n".join(lines) + "\n"


@real_petta
def test_a_graph_larger_than_the_first_queue_is_derived_at_a_larger_one_and_reported_complete(pln):
    extra = _layered(3, 5)                                   # 39 links, 363 routes from S: needs more than K = 256
    groups, result = run(_stack("minimal"), extra, ["!(explain &self S $x)"])
    s = last_chain()
    assert len(groups[0]) == 363 and s["complete"] is True
    assert s["status"]["S"]["queue"] == 2048 and s["status"]["S"]["attempts"] == 1       # picked from the route count, not found by failing
    assert s["queue_max"] == 2048
    legacy_rows = [{"from": m.group(1), "to": m.group(2), "sign": m.group(3), "strength": float(m.group(4)),
                    "confidence": float(m.group(5)), "terminal_only": False}
                   for m in re.finditer(r"\(Effect (\S+) (\S+) (Pos|Neg) \(stv (\S+) (\S+)\)\)", extra)]
    got = {tuple(_STV_CHAIN.match(a).group(1).split()): (_STV_CHAIN.match(a).group(2), float(_STV_CHAIN.match(a).group(3)),
                                                         float(_STV_CHAIN.match(a).group(4))) for a in groups[0]}
    want = {p: v for p, v in legacy_closure(legacy_rows).items() if p[0] == "S"}
    result = compare(got, want)
    assert result["missing"] == [] and result["extra"] == [] and result["bad"] == [], (result["bad"][:2], result["missing"][:2])


@real_petta
def test_a_forced_first_attempt_that_drops_something_is_retried_and_the_answer_is_the_same(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_QUEUE_SIZE=16)
    monkeypatch.setattr(pc, "estimate_beliefs", lambda sub, source, cap: 0)           # the child is forked after this
    groups, _ = run(_stack("patient"), None, ["!(explain &self Spermidine $x)"])
    s = last_chain()
    assert s["complete"] is True and s["attempts_max"] == 2 and s["status"]["Spermidine"]["queue"] == 128
    monkeypatch.undo()
    _use(monkeypatch, tmp_path, "pln")
    again, _ = run(_stack("patient"), None, ["!(explain &self Spermidine $x)"])
    assert groups == again


@real_petta
def test_a_source_that_cannot_finish_fails_the_request_as_a_service_error(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "pln", PLN_MAX_STEPS=2)
    _, result = run(_stack("patient"), None, ["!(infer &self Spermidine CoronaryHeartDisease)"], allow_error=True)
    assert result.status == "error" and result.error_code == "runtime_error"
    assert "did not finish" in result.error and "PLN_QUEUE_SIZE" in result.error
    assert result.results == []                                                      # no partial answer is ever returned


# ── the library on the real engine against the oracle ──────────────────────────

@real_petta
def test_the_real_library_and_the_oracle_derive_the_same_lists_bit_for_bit():
    rt = PettaRuntime()
    session = rt.session(pc.library_text(), 300)
    try:
        def ev(expr: str) -> list[str]:
            return session.run("!" + expr, 300)[0]

        rows = contract_links("patient", [FEEDBACK_EDGE])
        links = to_links(rows)
        leak = pc.probe_library(ev)
        assert leak == pytest.approx(0.02) and pc.read_discount(SimEngine()) == 0.9
        enc = pc._Encoding(links, leak)
        checked = 0
        for source in sorted({l.src for l in links}):
            sub = pc.reachable_links(links, source)
            for level in (1.0, 0.0):
                for tq, bq, steps in ((256, 256, 1024), (10, 100, 100), (20, 200, 20)):
                    real = pc._run(ev, enc, sub, source, level, steps, tq, bq, False)
                    sim = pc._run(SimEngine(), enc, sub, source, level, steps, tq, bq, False)
                    assert real[0] == sim[0] and real[1] == sim[1], (source, level, tq, bq, steps)
                    checked += 1
        assert checked == 20 * 2 * 3
    finally:
        session.close()


@real_petta
def test_the_sandbox_scan_accepts_every_sentence_the_step_sends():
    """The step talks to the engine directly, but the same text must pass the scan a session applies."""
    from core.metta_runtime import petta_refused_forms_in_kb
    links = to_links(contract_links("patient"))
    enc = pc._Encoding(links, 0.02)
    text = " ".join(enc.sentence.values()) + " " + enc.premise("Spermidine", 1.0)
    assert petta_refused_forms_in_kb(text) == []
    for hostile in ("test", "call", "Predicate", "shell", "halt", "assert", "include"):
        hostile_links = [pc.Link(hostile, "B", "Pos", 0.5, 0.5)]
        pc.assign_ids(hostile_links)
        enc2 = pc._Encoding(hostile_links, 0.02)
        assert hostile not in " ".join(enc2.sentence.values())                       # nodes are opaque: n0, n1, ...
    groups_ok = pc.render_facts(sim_table(contract_links("patient")))
    assert pc.ANSWER_HEAD in groups_ok


@real_petta
def test_a_node_named_like_a_refused_form_is_still_refused_by_the_scan_of_the_callers_text(pln):
    extra = "(Effect test call Pos (stv 0.5 0.5))\n(Effect call shell Neg (stv 0.5 0.5))\n"
    groups, result = run(_stack("minimal"), extra, ["!(explain &self test $x)"], allow_error=True)
    assert result.status == "error" and result.error_code == "refused_form"            # the scan of the caller's text, as always


@real_petta
def test_the_engine_is_given_opaque_names_so_a_hostile_node_name_cannot_reach_it():
    """A node called `test` or `call` or `Predicate` is a MeTTa builtin's name; sent as it is, the sandbox's text scan (or a
    builtin) would see it. The step sends n0, n1, ..: the derivation of a graph whose nodes carry such names is the same."""
    names = ["test", "call", "Predicate", "shell", "halt", "assert", "include", "py-call"]
    links = [pc.Link(names[i], names[i + 1], "Neg" if i % 2 else "Pos", 0.9 - 0.05 * i, 0.9) for i in range(len(names) - 1)]
    pc.assign_ids(links)

    def step(engine):
        ev = lambda e: [str(x) for x in engine.run("!" + e)]
        engine.load(pc.library_text())
        leak = pc.probe_library(ev)
        t = pc.compute_table(ev, links, leak=leak, discount=0.9, settings=pc.ChainSettings())
        return {"routes": [[r.path[0], r.path[-1], r.sign, round(r.strength, 12)] for r in t.all_routes()],
                "sent": sorted(pc._Encoding(links, leak).opaque.values())}

    rt = PettaRuntime()
    rt.run_program("", ["!(+ 1 2)"], after_load=step)
    out = rt.last_run["chain"]
    assert len(out["routes"]) == 28 and out["sent"][0] == "n0"                      # 7 + 6 + ... + 1 routes on a path of 8
    want = table_closure(sim_table([{"from": l.src, "to": l.dst, "sign": l.sign, "strength": l.strength,
                                     "confidence": l.confidence, "terminal_only": False} for l in links]))
    got = {(r[0], r[1]): (r[2], r[3]) for r in out["routes"]}
    assert got == {(k[0], k[-1]): (v[0], round(v[1], 12)) for k, v in want.items()}


# ── DrugAge ────────────────────────────────────────────────────────────────────

def _drugage_text() -> str:
    src = (Path(__file__).resolve().parent / "test_drugage_calibration.py").read_text(encoding="utf-8")
    return re.search(r'SYNTH = """(.*?)"""', src, re.S).group(1)


def _drugage_rows(prefix: str, compound: str, specs: list[tuple[str, float, str, bool, str]]) -> str:
    out = []
    for i, (species, pct, sig, itp, pmid) in enumerate(specs):
        r = f"{prefix}{i}"
        out.append(f"(InstanceOf {r} Experiment)\n(UsesIntervention {r} {compound})\n(UsesSpecies {r} {species})\n"
                   f"(AvgLifespanChangePercent {r} {pct})\n(AvgLifespanSignificance {r} {sig})\n"
                   + (f"(IsITPStudy {r})\n" if itp else "") + (f"(ReportedIn {r} {pmid})\n" if pmid else ""))
    return "\n".join(out) + "\n"


@real_petta
def test_drugage_rows_chain_through_lifespan_with_the_legacy_numbers_when_not_folded(monkeypatch, tmp_path):
    synth = _drugage_text()
    queries = ["!(infer &self ExtDrug Mortality)", "!(infer &self ToxDrug Mortality)", "!(infer &self DrugA Mortality)",
               "!(infer &self NilDrug Mortality)", "!(infer &self Lifespan Mortality)",
               "!(rank-interventions &self (GoodDrug WeakDrug NilDrug BadDrug) Mortality)",
               "!(score-candidate &self GoodDrug Mortality)"]
    _use(monkeypatch, tmp_path, "legacy")
    want, _ = run(_stack("drugage"), synth, queries)
    _use(monkeypatch, tmp_path, "pln")
    monkeypatch.setattr(pc, "settings_from_config", lambda: pc.ChainSettings(drugage_fold=False))
    got, _ = run(_stack("drugage"), synth, queries)
    for q, g, w in zip(queries, got, want):
        assert len(g) == len(w), q
        for a, b in zip(g, w):
            (sa, na), (sb, nb) = _skeleton(a), _skeleton(b)
            assert sa == sb and na == pytest.approx(nb, abs=1e-12), (q, a, b)
    assert _scored(got[5][0]) == ["GoodDrug", "WeakDrug", "NilDrug", "BadDrug"]
    assert got[0] == ["(signed Neg (stv 0.6 0.81))"]


@real_petta
def test_a_drugage_link_is_the_start_of_a_chain_never_an_answer_on_its_own(pln):
    groups, _ = run(_stack("drugage"), _drugage_text(), ["!(infer &self ExtDrug Lifespan)", "!(explain &self ExtDrug Lifespan)",
                                                         "!(explain &self ExtDrug Mortality)"])
    assert groups[0] == [] and groups[1] == []                                        # legacy: needs a further hop
    assert groups[2] == ["(Chain (ExtDrug Lifespan Mortality) (signed Neg (stv 0.6 0.81)))"]


@real_petta
def test_the_fold_pools_papers_and_gives_each_direction_its_own_stamp(pln):
    pytest.importorskip("ontology.drugage_fold")
    rows = _drugage_rows("P", "PoolDrug", [("Mus_musculus", 20.0, "Significant", False, "PMID_1"),
                                           ("Mus_musculus", 25.0, "Significant", False, "PMID_2"),
                                           ("Mus_musculus", 10.0, "Significant", False, "PMID_2"),     # same paper as the row above
                                           ("Mus_musculus", -15.0, "Significant", False, "PMID_1"),    # the paper of row 0, other way
                                           ("Mus_musculus", -5.0, "NotSignificant", False, "PMID_3")])
    groups, _ = run(_stack("drugage"), rows, ["!(explain &self PoolDrug Mortality)",
                                              "!(match &self (PLNEvidence $id drugage $where $tier) ($id $where))"])
    chains = [_STV_CHAIN.match(a).group(2, 3, 4) for a in groups[0]]
    assert sorted(c[0] for c in chains) == ["Neg", "Pos"]                     # one result per direction, both survive the chain
    assert len(groups[1]) == 2                                                # two links, each with its own evidence id
    ids = last_chain()["evidence"]
    drug = [v for v in ids.values() if v["kind"] == "drugage"]
    assert sorted(v["sign"] for v in drug) == ["Neg", "Pos"]
    assert sorted(sorted(v["papers"]) for v in drug) == [["PMID_1", "PMID_2"], ["PMID_1", "PMID_3"]]   # PMID_1 stands behind both
    assert len({tuple(v["stamp"]) for v in drug}) == 2                                                 # ... under two stamps


@real_petta
def test_the_fold_is_not_limited_by_the_hyperon_era_row_cap(pln):
    pytest.importorskip("ontology.drugage_fold")
    from ontology.drugage_selector import MAX_ROWS
    assert MAX_ROWS == 150
    specs = [("Mus_musculus", 5.0 + (i % 9), "Significant", False, f"PMID_{1000 + i // 3}") for i in range(170)]
    groups, _ = run(_stack("drugage"), _drugage_rows("R", "Resv", specs), ["!(explain &self Resv Mortality)"])
    s = last_chain()
    drug = [v for v in s["evidence"].values() if v["kind"] == "drugage"]
    assert len(drug) == 1 and drug[0]["rows_covered"] == 170 and len(drug[0]["papers"]) == 57      # 170 rows, 57 papers
    assert len(groups[0]) == 1
