"""PLN_CHAINER=pln: `infer` and `explain` answered by patham9's PLN.

What this is
------------
The legacy chainer (pln_deduction.metta) multiplies strengths and confidences along a path. Under
`PLN_CHAINER=pln` (opt-in; it needs `PLN_RUNTIME=petta`; never a default) the same questions are answered by the
fork library of patham9's PLN, vendored under `vendor/pln_fork/` (restricted rule set: Revision, Modus Ponens,
Negation-elimination). Every `infer`/`explain` answer is a conclusion of his `PLN.Derive` loop: his Modus Ponens and
Negation-elimination formulas, his stamp rule (a link is used once on a route), his confidence-ranked bounded queues.

What that does and does not mean, plainly: the mapping below is built so that the answers EQUAL the legacy ones. The reported
strength is the product of the link strengths (recovered exactly from the difference of two Modus Ponens runs, whose 0.02 leak
cancels) and the confidence is the library's `c1*c2...` times the legacy 0.9 per extra hop, so a strength or a confidence is
the legacy number, not a new estimate. What the library contributes is the route enumeration (each LINK used once, so a feedback
edge terminates), the stamp rule, the bounded confidence-ranked queues and the proof of completeness. The design, the reasons
for it and the two reviews of it are in docs/petta_migration/fork_design/decision.md and refute_*.md; this module is its build.

The encoding ("state sentences with Modus Ponens")
--------------------------------------------------
* node X is the term `(State X)`; no per-node sentence, no base rate. Names sent to the engine are opaque (`n0`,
  `n1`, ..) so that no compound name can collide with a form the sandbox refuses.
* a link `(Effect A B Pos (stv p c))` with evidence id k is the sentence
  `((Implication (State A) (State B)) (stv q c))` with stamp `(k)`; a Neg link has consequent `(Not (State B))`
  (Modus Ponens derives it and Not-elimination turns it back into `(State B)` with 1 - s and the same stamp); a
  hazard-ratio record has consequent `(End B)` (`(Not (End B))` for HR < 1): a record can end a chain and never
  continues one.
* q = e + (1 - e) p, where e is the fork's Modus Ponens leak, read from `Truth_ModusPonens (stv 0 1) (stv 0 1)` when
  the library loads (0.02). The fork's Modus Ponens is `s = f1 f2 + e (1 - f1)`: affine in the premise, so the
  DIFFERENCE between two runs that differ only in the premise's strength is the product of (q - e) = (1 - e) p along
  the path. The earlier `q = min(1, p + e)` shrinks every strength above 1 - e, among them the materialised
  `Lifespan -> Mortality (stv 1.0 1.0)` link; this form is exact for every p in [0, 1].
* the intervention A is the single task `(State A)` with strength 1.0, confidence 1.0, evidence id 0, run once at
  level 1 and once at level 0. A route's strength is `|s1 - s0| / (1 - e)^hops` with the two runs joined on the
  route's stamp (its set of evidence ids, which names the route), its sign is the parity of its Neg links
  (cross-checked against the sign of s1 - s0), and its confidence is the library's `c` times
  `chain-discount ^ (hops - 1)` (PLN_CHAIN_CONFIDENCE=legacy, the default) or the library's `c` as it stands
  (`native`). The state levels themselves are artefacts of the leak and are never exposed.
* one result per node-simple route, sorted by (hops, stamp); A = C is dropped; a feedback edge terminates because a
  derivation uses each link once.

How a request runs it
---------------------
`core.pln_runner._runtime_run` hands the assembled knowledge base and the program's expressions to `prepare()`, which (1) refuses
a text that already holds a `PLNAnswer` atom, (2) filters the legacy `infer`/`explain` equations out of the text, (3) appends
pln_chainer.metta (equations that read the answer table) and (4) returns a `ChainStep`. The step runs inside the
PeTTa child through `metta_runtime`'s `after_load` hook: it reads every link off the loaded space with the knowledge base's own
functions (`Effect` atoms, `link-effect`'s record lift via `calibrate-tv`, and DrugAge rows via `drugage-effect`), then

* REFUSES (422) a fact of the caller's own that cannot be chained -- a strength or confidence outside [0, 1] or not a number, an
  unknown sign, a DrugAge row stated twice with different values -- naming the atom (`PlnInputError`): answered "no route" it
  would be a wrong answer. The same fault in a curated file is skipped and listed (`PLNSkipped`);
* works out WHICH SOURCES the program can ask about (`demand_of`): the literal sources of its direct `infer`/`explain` calls,
  none if nothing in it can reach them, and every source when it cannot tell (a consumer such as `score-candidate`, an unbound
  or computed source). Consumers bind the source while they evaluate and the space is read-only once the request starts, so
  only a program the analysis fully reads is narrowed -- to the same answers, never fewer;
* PLANS the derivation from the links alone (`plan_sources`): for each source the distinct walks the library will derive (each
  link used once -- far more than the node-simple routes on a cyclic graph), the beliefs they make and a modelled time; a source
  whose walks cannot fit the largest queue, or a table whose modelled time is over 80 percent of the request's budget, is
  REFUSED in milliseconds with the numbers (`PlnInputError`), not left to run into the deadline kill;
* loads the library, derives the answers of the sources (two `PLN.Derive` runs each, checking the clock between sources and
  queue sizes), and adds them to the space as facts, with the evidence table.

A caller's `extra_atoms` facts need no precompile: they are part of the text the step reads.

Control (decision.md section 3)
-------------------------------
`PLN_QUEUE_SIZE` sets both PLN.Derive queues to one value K; `PLN_MAX_STEPS` bounds the tasks expanded (default
4K). A run is `complete` iff the task queue ran empty and the belief queue never reached K - 1 (then nothing was
dropped; with the task queue smaller than the belief queue a drop goes unseen). An attempt that dropped something is
repeated at 8K and 64K; a source that still cannot finish FAILS THE REQUEST (`PlnChainerError`, HTTP 502). Only the
research lane (`ChainSettings(task_queue=.., belief_queue=..)`) may use different sizes or accept a `partial` table.
The priority function, the Unique order, the stamp policy and the rule set are the library's and are not exposed.

Evidence ids
------------
One id per output link (a curated Effect atom, a hazard-ratio record, a DrugAge entry or pooled group, a caller
fact), assigned in a deterministic order. An id is NEVER shared by two links: the library refuses to combine two
premises that share one, so a shared id on two serial links would silently lose the chain. What an id stands for
(file:line, rows, PMIDs, evaluated tier) is the side table, written to the space as `PLNEvidence` atoms and returned
with the table.

DrugAge
-------
DrugAge rows are lifted by the knowledge base's own `drugage-effect` into `compound -> Lifespan` links, handed to
`ontology.drugage_fold.fold_items` (the paper-level pooling: at most one protective and one harmful item per compound,
the evidence unit being the PMID) and chained through the materialised `Lifespan -> Mortality (stv 1.0 1.0)` link. Each
pooled item becomes ONE link with an evidence id of its own, so the same paper standing behind both directions of a
compound cannot make the library refuse a chain. `ChainSettings(drugage_fold=False)` (or a missing fold module) gives one
link per row, which reproduces the legacy per-row results. A DrugAge link is never an answer on its own (legacy: `infer
<compound> Lifespan` is empty), only the start of a chain.

A reported NULL (a row whose change is exactly 0.0, strength 0) counts in BOTH directions by default
(`ChainSettings.drugage_zero_rows`; ontology/drugage_fold.py explains why: filed on the protective side only, which is where the
sign convention puts a zero, it tilts every net toward harmful, and the sign of Resveratrol and Metformin was that tilt). A row
stated twice with identical content is one row; one whose copies disagree is refused as the caller's fault. The ranking built on
these links (`core.pln_runner.run_drugage_ranking`) lists a compound once, by THE NET of its protective and harmful items, using
every row of the compound (no row cap), and equals the pure-Python twin `ontology.drugage_fold.top_compounds`.

Research lanes (settings of `compute_table`, never of a request)
----------------------------------------------------------------
`ChainSettings(task_queue=.., belief_queue=..)` separate queue sizes and a `partial` table (the beam width and the memory
of a best-first search: the control demonstration); `count_selections` how many tasks each run expanded;
`mapping='literal'|'capped'` the fork's strength read literally (R0) and the superseded `min(1, p + e)` (a negative control);
`PLN_RULESET=full` all seventeen rules, uniform node probabilities, every sentence a task, the whole graph as beliefs, a fixed
150/100/400 budget, and a count of conclusions that no walk of the graph licenses (`SourceStatus.unattributed`, always 0 for
the restricted rules).

Nothing here imports hyperon, janus or petta: the step talks to whatever engine it is given (`.run(text)`,
`.load(text)`), so the adapter is testable against a pure-Python oracle (docs/petta_migration/fork_design/sim).
"""
from __future__ import annotations

import bisect
import functools
import math
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from core.metta_runtime import petta_refused_forms_in_kb, petta_strip_comments, petta_tokens, split_top_level_forms

__all__ = [
    "PlnChainerError", "PlnInputError", "ChainSettings", "Link", "Route", "SourceStatus", "SourcePlan", "ChainTable", "Prepared",
    "ChainStep", "ZERO_ROW_CONVENTIONS", "WalkStats", "walk_stats", "count_states", "plan_sources", "demand_of", "model_seconds",
    "chain_reaching_heads", "caller_skips_message", "request_budget_s", "out_edges",
    "ANSWER_HEAD", "AUDIT_HEADS", "CHAIN_FILE", "LIBRARY_FILE", "LIBRARY_FULL_FILE", "FULL_LANE_BUDGET",
    "chainer_mode", "audit_symbols", "is_pln", "settings_from_config",
    "answer_head_in", "answer_head_refusal", "filter_legacy_chain", "prepare", "library_text", "base_rate_text",
    "chainer_text", "parse_sexpr", "serialize", "fnum", "number",
    "collect_links", "assign_ids", "assert_ids_unique_along_paths", "evaluator_for_session",
    "reachable_links", "count_routes", "estimate_beliefs", "compute_table", "render_facts",
    "probe_library", "read_discount", "count_selections",
]

#: the head of the one table `infer` and `explain` read; nothing but the step may write it
ANSWER_HEAD = "PLNAnswer"
#: the audit atoms the step writes next to it: the side table of evidence ids, the per-source completeness status, and the
#: links that could not be chained. A caller may READ these (`/metta/run`); they are not an input, but a forged one would
#: only mislabel an audit, never change an answer
AUDIT_HEADS = ("PLNEvidence", "PLNStatus", "PLNSkipped")
REPO_ROOT = Path(__file__).resolve().parents[2]
#: the equations that read the table (appended to the knowledge base under pln)
CHAIN_FILE = REPO_ROOT / "pln_chainer.metta"
#: the vendored library, restricted rule set (generated and tested by scripts/make_pln_restricted.py)
LIBRARY_FILE = REPO_ROOT / "vendor" / "pln_fork" / "lib_pln_restricted.metta"
#: the vendored library, all seventeen rules: the research lane PLN_RULESET=full, never a request
LIBRARY_FULL_FILE = REPO_ROOT / "vendor" / "pln_fork" / "lib_pln_fork.metta"
RULESETS = ("restricted", "full")
MAPPINGS = ("affine", "literal", "capped")
#: how a DrugAge row that reports a change of exactly 0.0 (a reported NULL: strength 0) counts in the paper-level fold.
#: Kept equal to ontology.drugage_fold.ZERO_ROW_CONVENTIONS by a test (this module must not import the fold at load).
ZERO_ROW_CONVENTIONS = ("both", "neither", "protective")
#: the full lane's uniform node probabilities (LAUNCH.md's values; p in (0, 0.5] keeps the library's consistency check
#: satisfied for any strength, and p = 0 would be a division by zero)
FULL_LANE_BASE_RATE = (0.1, 0.9)
#: the full lane runs on a fixed budget (steps, task queue, belief queue), as the decision's reproduction did (150/100/400):
#: with every rule on, the derivation does not stop by itself, and a table made under it is flagged `partial` by design
FULL_LANE_BUDGET = (150, 100, 400)

#: the largest the strength 0.98**hops divisor may be read from (a leak this big is not the fork's)
_MAX_LEAK = 0.5
#: the PLN_MAX_STEPS clamp
_MAX_STEPS_CAP = 1_000_000
#: a probe tolerance (the fork prints (0.26 0.6400000000000001) for 0.8 * 0.8)
_PROBE_TOL = 1e-12
_SIGN_CROSSCHECK_MIN = 1e-9


class PlnChainerError(RuntimeError):
    """The chainer could not produce a complete, checked table. `str(error)` is safe to show a caller; the HTTP
    layer answers it as a service error (502, `runtime_error`), never from a partial table."""


class PlnInputError(PlnChainerError):
    """The same, and the program's own facts are the cause: a caller's `Effect` whose truth value is not a number in
    [0, 1], a DrugAge row stated twice with different values, a graph so dense or so cyclic that its walks cannot be
    derived in the time a request has. The HTTP layer answers it as a refused program (422, `refused_form`), not as a
    failure of the service; `str(error)` names the offending facts."""


# ── selection ──────────────────────────────────────────────────────────────────

def chainer_mode() -> str:
    """'legacy' or 'pln': PLN_CHAINER, read when called (a test can change it). `pln` on a runtime other than
    PeTTa is a configuration error."""
    import config
    mode = str(getattr(config, "PLN_CHAINER", "legacy") or "legacy").strip().lower()
    if mode not in ("legacy", "pln"):
        raise PlnChainerError(f"PLN_CHAINER must be legacy or pln; got {mode!r}")
    if mode == "pln":
        from core.metta_runtime import selected_runtime_name
        if selected_runtime_name() != "petta":
            raise PlnChainerError("PLN_CHAINER=pln needs PLN_RUNTIME=petta: patham9's PLN runs on PeTTa only")
    return mode


def is_pln() -> bool:
    return chainer_mode() == "pln"


@dataclass(frozen=True)
class ChainSettings:
    """The knobs of one table. `settings_from_config()` is what a request uses."""
    #: 'legacy' = library c x chain-discount^(hops-1); 'native' = library c
    chain_confidence: str = "legacy"
    #: tasks one run may expand; 0 = 4K of the rung
    max_steps: int = 0
    #: K of the first attempt (both queues)
    queue_size: int = 256
    #: each retry multiplies K by this, at most `rungs` attempts (256, 2048, 16384)
    growth: int = 8
    rungs: int = 3
    #: RESEARCH LANE ONLY: different queue sizes (the beam width and the memory of a best-first search). Any of the two
    #: switches the table to control mode: one attempt, no retry, and an incomplete table is returned flagged `partial`.
    task_queue: Optional[int] = None
    belief_queue: Optional[int] = None
    #: 'retry' = contract mode (retry at a larger K, then raise); 'partial' = return what was derived, flagged
    on_incomplete: str = "retry"
    #: count the selections of every run by chunked resumption (one PLN.Derive call per selection): for reports
    count_selections: bool = False
    #: pool the DrugAge rows of a compound by paper and direction (ontology.drugage_fold) before they become links;
    #: False = one link per row, which reproduces the legacy per-row results (the lift-fidelity check)
    drugage_fold: bool = True
    #: how a DrugAge row with a change of exactly 0.0 counts when the fold pools: 'both' = as a null in BOTH directions (the
    #: repo's own reading: a well-run null is evidence of no effect either way), 'neither' = it is dropped, 'protective' = the
    #: sign convention's side only (the earlier behaviour, kept to show the sign of Resveratrol and Metformin depends on it)
    drugage_zero_rows: str = "both"
    #: derive only the sources the program asks about (`demand_of`); False derives every source with a link out of it, as the
    #: first build did. A program the analysis cannot read (a consumer function, an unbound or computed source) always gets
    #: every source, so a narrower table can only ever be the same answers, never fewer
    demand: bool = True
    #: RESEARCH LANE: which library. 'restricted' (Revision, Modus Ponens, Negation-elimination) is the service's; 'full'
    #: (all seventeen rules, uniform base rates) shows what the other rules add (false chains) and is set by
    #: PLN_RULESET in the environment only, never by a request
    ruleset: str = "restricted"
    #: RESEARCH LANE: how a strength is stored and reported. 'affine' q = e + (1 - e) p, reported |s1 - s0| / (1 - e)^hops
    #: (exact for every p in [0, 1]; the service's); 'literal' q = p, reported |s1 - s0| (the fork read literally: every
    #: strength shrinks by (1 - e/p) per hop); 'capped' q = min(1, p + e), reported |s1 - s0| (the earlier mapping, whose
    #: difference is p per hop only while p <= 1 - e: a strength above 0.98 is shrunk to 0.98; kept as a negative control)
    mapping: str = "affine"

    def __post_init__(self):
        if self.ruleset not in RULESETS:
            raise ValueError(f"ruleset must be one of {RULESETS}; got {self.ruleset!r}")
        if self.mapping not in MAPPINGS:
            raise ValueError(f"mapping must be one of {MAPPINGS}; got {self.mapping!r}")
        if self.chain_confidence not in ("legacy", "native"):
            raise ValueError(f"chain_confidence must be legacy or native; got {self.chain_confidence!r}")
        if self.on_incomplete not in ("retry", "partial"):
            raise ValueError(f"on_incomplete must be retry or partial; got {self.on_incomplete!r}")
        if self.drugage_zero_rows not in ZERO_ROW_CONVENTIONS:
            raise ValueError(f"drugage_zero_rows must be one of {ZERO_ROW_CONVENTIONS}; got {self.drugage_zero_rows!r}")

    @property
    def control(self) -> bool:
        return self.task_queue is not None or self.belief_queue is not None


def settings_from_config() -> ChainSettings:
    import config
    return ChainSettings(
        chain_confidence=str(getattr(config, "PLN_CHAIN_CONFIDENCE", "legacy")),
        max_steps=int(getattr(config, "PLN_MAX_STEPS", 0)),
        queue_size=int(getattr(config, "PLN_QUEUE_SIZE", 256)),
        ruleset=str(getattr(config, "PLN_RULESET", "restricted")))


# ── s-expressions ──────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+')
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def parse_sexpr(text: str) -> Any:
    """One s-expression: a list becomes a tuple, an atom (symbol, number, "string") stays its text."""
    stack: list[list] = [[]]
    for tok in _TOKEN_RE.findall(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) < 2:
                raise PlnChainerError(f"unbalanced result from the engine: {text[:120]!r}")
            done = tuple(stack.pop())
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise PlnChainerError(f"expected one expression from the engine, got: {text[:120]!r}")
    return stack[0][0]


def serialize(tree: Any) -> str:
    if isinstance(tree, tuple):
        return "(" + " ".join(serialize(x) for x in tree) + ")"
    return str(tree)


def number(token: Any) -> Optional[float]:
    """The float a number token reads as, else None (a symbol, an unevaluated call, nan, inf)."""
    if isinstance(token, str) and _NUMBER_RE.match(token):
        value = float(token)
        return value if math.isfinite(value) else None
    return None


def fnum(x: float) -> str:
    """A float as MeTTa text that reads back to the same float: plain decimal (no exponent), always with a point."""
    x = float(x)
    if not math.isfinite(x):
        raise PlnChainerError(f"cannot write {x!r} into a sentence")
    r = repr(x)
    if "e" in r or "E" in r:
        r = format(Decimal(r), "f")
    return r if "." in r else r + ".0"


# ── the knowledge-base text: guard, filter, assembly ───────────────────────────

def audit_symbols() -> frozenset:
    """The audit heads a query may use: known to the validator under PLN_CHAINER=pln (the step writes them), nothing otherwise."""
    try:
        return frozenset(AUDIT_HEADS) if chainer_mode() == "pln" else frozenset()
    except PlnChainerError:
        return frozenset()


def answer_head_in(text: str) -> bool:
    """Does `text` (a query, a caller's atoms, a knowledge-base file) use the symbol PLNAnswer? Comments and
    strings aside, read the way PeTTa reads symbols."""
    return any(kind == "symbol" and tok == ANSWER_HEAD for kind, tok in petta_tokens(text))


def answer_head_refusal(text: str) -> Optional[str]:
    """The validator's message when a caller's text writes the answer table's head under PLN_CHAINER=pln, else None."""
    if not text or ANSWER_HEAD not in text:
        return None
    try:
        if chainer_mode() != "pln":
            return None
    except PlnChainerError:
        return None
    if answer_head_in(text):
        return (f"`{ANSWER_HEAD}` is the table PLN_CHAINER=pln writes its answers to (`infer` and `explain` read it); "
                f"a text that writes it would forge an answer, so it is refused")
    return None


_LEGACY_HEADS = ("infer", "explain")
_LEGACY_QUICK_RE = re.compile(r"\(=\s*\(\s*(?:infer|explain)[\s)]")


def filter_legacy_chain(text: str) -> tuple[str, int]:
    """`text` without its top-level `(= (infer ..) ..)` and `(= (explain ..) ..)` equations; and how many went.

    A form-level filter, not a hand edit of the files: under PeTTa an extra equation clause ADDS answers, so the
    legacy equations must not be loaded next to the ones that read the table. Everything else in the text, comments
    included, is kept byte for byte (pln_deduction.metta keeps its helpers, its type signatures and `link-effect`;
    drugage_calibration.metta keeps its lifts). A text with no such equation comes back as the same object."""
    if not _LEGACY_QUICK_RE.search(text):
        return text, 0
    pieces: list[str] = []
    last = removed = 0
    for start, end in split_top_level_forms(text):
        form = text[start:end]
        if form.startswith("!") or not _LEGACY_QUICK_RE.match(form):
            continue
        head = [tok for kind, tok in petta_tokens(form[:200]) if kind != "paren"][:2]
        if len(head) == 2 and head[0] == "=" and head[1] in _LEGACY_HEADS:
            pieces.append(text[last:start])
            last = end
            removed += 1
    if not removed:
        return text, 0
    pieces.append(text[last:])
    return "".join(pieces), removed


# ── what the program asks: which sources need a derivation ──────────────────────

#: a program that reads the answer table or its status itself wants it whole
_TABLE_HEADS = frozenset({ANSWER_HEAD, "PLNStatus"})
_PLAIN_SOURCE_RE = re.compile(r"[A-Za-z_][^\s()\"$&;#]*$")
_CHAIN_HEADS = frozenset(_LEGACY_HEADS)


@functools.lru_cache(maxsize=256)
def _definitions(text: str) -> tuple:
    """(head, symbols it uses) for every top-level `(= (head ..) body)` equation of `text`. A pure function of the text, so a
    cache by content keeps no state between requests."""
    if "(=" not in text:
        return ()
    out = []
    for start, end in split_top_level_forms(text):
        form = text[start:end]
        if form.startswith("!") or not form.startswith("(="):
            continue
        symbols = [tok for kind, tok in petta_tokens(form) if kind == "symbol"]
        if len(symbols) >= 2 and symbols[0] == "=":
            out.append((symbols[1], frozenset(symbols[2:])))
    return tuple(out)


def chain_reaching_heads(blocks: Sequence[tuple[str, str]]) -> frozenset:
    """The names of the functions that can call `infer` or `explain`, directly or through other functions: every equation
    whose body mentions the name of one (pattern variables and data included: an over-approximation, so the answer errs
    toward "yes"). `infer` and `explain` themselves are not in the set."""
    callers_of: dict[str, set] = {}
    for _, text in blocks:
        for head, uses in _definitions(text):
            for sym in uses:
                callers_of.setdefault(sym, set()).add(head)
    reaching: set = set()
    frontier = [sym for sym in _CHAIN_HEADS]
    while frontier:
        sym = frontier.pop()
        for head in callers_of.get(sym, ()):
            if head not in reaching and head not in _CHAIN_HEADS:
                reaching.add(head)
                frontier.append(head)
    return frozenset(reaching)


def demand_of(blocks: Sequence[tuple[str, str]], expressions: Optional[Sequence[str]]) -> Optional[frozenset]:
    """The sources the program can ask `infer`/`explain` about, or None when it cannot be said (then every source is derived).

    * `frozenset()`: nothing in the program can reach `infer` or `explain`; no derivation is needed (the audit atoms are
      still written: they depend on the links only);
    * a set of symbols: every use of `infer`/`explain` in the expressions is a direct call `(infer <space> <Source> <C>)`
      with a literal source, and no expression calls a knowledge-base function that reaches them;
    * None: anything else -- a consumer such as `score-candidate` or `rank-interventions-for-patient`, a source that is
      a variable, an expression or a number, `infer` passed as a value, no expressions at all.

    Consumers bind the source while they evaluate, and the space is read-only once a request starts, so a source the
    analysis fails to foresee would be answered "no". Hence the rule: when in doubt, every source."""
    if expressions is None:
        return None
    reaching = chain_reaching_heads(blocks)
    sources: set = set()
    for expr in expressions:
        text = petta_strip_comments(expr).strip()
        if text.startswith("!"):
            text = text[1:]
        symbols = [tok for kind, tok in petta_tokens(text) if kind == "symbol"]
        if any(sym in reaching or sym in _TABLE_HEADS for sym in symbols):
            return None
        mentions = sum(1 for sym in symbols if sym in _CHAIN_HEADS)
        if not mentions:
            continue
        try:
            tree = parse_sexpr(text)
        except PlnChainerError:
            return None
        direct = 0
        stack = [tree]
        while stack:
            node = stack.pop()
            if not isinstance(node, tuple):
                continue
            if node and node[0] in _CHAIN_HEADS:
                if len(node) != 4:
                    return None
                source = node[2]
                if not (isinstance(source, str) and _PLAIN_SOURCE_RE.match(source) and number(source) is None):
                    return None
                sources.add(source)
                direct += 1
            stack.extend(node)
        if direct != mentions:                 # `infer` somewhere else than at the head of a call
            return None
    return frozenset(sources)


def library_text(ruleset: str = "restricted") -> str:
    """The vendored library. The restricted one has to pass the engine sandbox's text scan (it holds no refused form),
    so a session of the production adapter can load it; the full one holds `Predicate` (a pattern in the three
    Evaluation-with-Inheritance rules, a PeTTa builtin the scan refuses) and nothing else, and only the research lane
    loads it, inside the child."""
    if ruleset not in RULESETS:
        raise ValueError(f"ruleset must be one of {RULESETS}; got {ruleset!r}")
    text = (LIBRARY_FILE if ruleset == "restricted" else LIBRARY_FULL_FILE).read_text(encoding="utf-8")
    refused = petta_refused_forms_in_kb(text)
    if ruleset == "full":
        refused = [r for r in refused if r != "Predicate"]
    if refused:
        raise PlnChainerError(f"the vendored library uses form(s) the PeTTa runtime does not run: {', '.join(refused)}")
    return text


def base_rate_text(links: Sequence["Link"]) -> str:
    """The full lane's node probabilities: `(= (STV (State n)) (stv p c))` for every node term (the restricted rules
    never look a node up; Deduction, Induction and the other rules do, and silently derive nothing for a term that has
    no equation)."""
    p, c = FULL_LANE_BASE_RATE
    names = sorted({link.src for link in links} | {link.dst for link in links})
    opaque = {key: f"n{i}" for i, key in enumerate(names)}
    return "\n".join(f"(= (STV ({kind} {opaque[key]})) (stv {fnum(p)} {fnum(c)}))" for key in names
                     for kind in ("State", "End")) + "\n"


def chainer_text() -> str:
    return CHAIN_FILE.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Prepared:
    """What `prepare` returns: the knowledge-base text to load, the step to run after it (None when the stack has no
    legacy chaining equation to replace), and how many equations were filtered out."""
    kb_text: str
    step: Optional["ChainStep"]
    removed: int


def prepare(blocks: Sequence[tuple[str, str]], settings: Optional[ChainSettings] = None,
            caller_blocks: Iterable[str] = ("extra_atoms",), expressions: Optional[Sequence[str]] = None) -> Prepared:
    """Assemble the knowledge base of a request under PLN_CHAINER=pln.

    `blocks` are (name, text) in load order, as `pln_runner` reads them (the files of the stack, then the caller's
    `extra_atoms` slice); `expressions` are the program's `!` expressions, from which the step works out which sources
    need a derivation (`demand_of`; None = unknown = every source). Raises MettaRuntimeError(kind="refused") if any block
    writes the answer table's head."""
    from core.metta_runtime import MettaRuntimeError
    caller = set(caller_blocks)
    for name, text in blocks:
        if ANSWER_HEAD in text and answer_head_in(text):
            raise MettaRuntimeError(
                f"{name} writes `{ANSWER_HEAD}`, the table PLN_CHAINER=pln derives its answers into; it would forge "
                f"an answer", kind="refused", detail=ANSWER_HEAD)
    filtered: list[str] = []
    removed = 0
    for name, text in blocks:
        if name in caller:                          # a caller's definitions are the caller's business
            filtered.append(text)
            continue
        new, n = filter_legacy_chain(text)
        filtered.append(new)
        removed += n
    if not removed:
        return Prepared("\n".join(filtered), None, 0)
    filtered.append(chainer_text())
    step = ChainStep(tuple(blocks), settings or settings_from_config(), caller_blocks=tuple(caller),
                     expressions=None if expressions is None else tuple(expressions))
    return Prepared("\n".join(filtered), step, removed)


# ── links ───────────────────────────────────────────────────────────────────────

@dataclass
class Link:
    """One traversable edge of the graph: a curated Effect atom, a hazard-ratio record, a DrugAge entry, a caller fact."""
    src: str                         #: node key (the node's printed text)
    dst: str
    sign: str                        #: 'Pos' | 'Neg'
    strength: float                  #: p, the legacy strength
    confidence: float                #: c
    terminal: bool = False           #: a hazard-ratio record: it can end a chain, never continue one
    kind: str = "curated"            #: curated | record | drugage | caller
    #: a DrugAge link is the start of a chain, never an answer: legacy answers `infer <compound> Lifespan` with nothing
    needs_continuation: bool = False
    source: str = ""                 #: file:line, or the DrugAge rows
    tier: Optional[str] = None       #: the evidence tier the atom's confidence was written with, if it names one
    detail: dict = field(default_factory=dict)
    id: int = 0                      #: the evidence id (1..n; 0 is the premise)

    def sort_key(self) -> tuple:
        return (self.src, self.dst, self.sign, self.terminal, self.kind, self.strength, self.confidence, self.source)


def assign_ids(links: Sequence[Link]) -> None:
    """Evidence ids 1..n in a deterministic order (by content, so the input order cannot change a result)."""
    for n, link in enumerate(sorted(links, key=Link.sort_key), start=1):
        link.id = n


def assert_ids_unique_along_paths(links: Sequence[Link]) -> None:
    """An evidence id may be shared only by parallel evidence of ONE link, never along a derivable path: the library
    refuses to combine two premises that share an id, so a shared id on two serial links loses the chain without a
    word. Here every link carries exactly one id of its own, so the check is that ids are unique and positive."""
    seen: dict[int, Link] = {}
    for link in links:
        if link.id <= 0:
            raise PlnChainerError(f"link {link.src} -> {link.dst} has no evidence id")
        other = seen.get(link.id)
        if other is not None and other is not link:
            raise PlnChainerError(f"evidence id {link.id} is shared by {other.src} -> {other.dst} and "
                                  f"{link.src} -> {link.dst}")
        seen[link.id] = link


def out_edges(links: Sequence[Link]) -> dict[str, list[Link]]:
    """The links grouped by their source node, in input order (built once per table, not once per source)."""
    out_by: dict[str, list[Link]] = {}
    for link in links:
        out_by.setdefault(link.src, []).append(link)
    return out_by


def reachable_links(links: Sequence[Link], source: str, out_by: Optional[dict] = None) -> list[Link]:
    """The links reachable from `source`, in input order: follow every link out of a visited node; a terminal
    link (a record) is kept but its target is not expanded. The library's cost is quadratic in the beliefs it is
    given, so the sub-graph is mandatory. `out_by` (from `out_edges`) saves rebuilding the adjacency per source."""
    if out_by is None:
        out_by = out_edges(links)
    seen = {source}
    frontier = [source]
    keep: set[int] = set()
    while frontier:
        nxt = []
        for node in frontier:
            for link in out_by.get(node, ()):
                keep.add(id(link))
                if not link.terminal and link.dst not in seen:
                    seen.add(link.dst)
                    nxt.append(link.dst)
        frontier = nxt
    return [link for link in links if id(link) in keep]


# ── reading the links off the loaded space ──────────────────────────────────────

Evaluate = Callable[[str], list]
"""Evaluate one MeTTa expression (no leading `!`) and return the result atoms as strings."""

_EFFECT_Q = "(match &self (Effect $a $b $g $t) (L $a $b $g $t))"
_RECORD_Q = ("(match &self (, (HasPredictor $r $a) (HasOutcome $r $c) (EffectValue $r $hr)) "
             "(R $r $a $c (signed (hr->sign $hr) (calibrate-tv &self $r))))")
_DRUGAGE_Q = ("(match &self (, (UsesIntervention $r $cmp) (AvgLifespanChangePercent $r $p)) "
              "(D $r $cmp $p (drugage-effect &self $r)))")
_PAPER_Q = "(match &self (, (UsesIntervention $r $cmp) (ReportedIn $r $p)) (P $r $p))"
#: what the fold needs to apply the repo's representative-row policy inside a paper (ontology.drugage_fold.POLICY_KEYS)
_SIGNIFICANCE_Q = "(match &self (, (UsesIntervention $r $cmp) (AvgLifespanSignificance $r $g)) (G $r $g))"
_ITP_Q = "(match &self (, (UsesIntervention $r $cmp) (IsITPStudy $r)) (I $r))"
_SPECIES_Q = "(match &self (, (UsesIntervention $r $cmp) (UsesSpecies $r $sp)) (S $r $sp))"
_HAS_CALIBRATE_RE = re.compile(r"\(=\s*\(\s*calibrate-tv[\s)]")
_HAS_DRUGAGE_RE = re.compile(r"\(=\s*\(\s*drugage-effect[\s)]")
_EVIDENCE_TIER_RE = re.compile(r"\(evidence-confidence\s+([A-Za-z0-9_\-]+)\)")


def _identity_fold(items: list[dict]) -> list[dict]:
    return items


def _fold_items(enabled: bool = True) -> Callable[[list[dict]], list[dict]]:
    """`ontology.drugage_fold.fold_items` (the paper-level pooling, owned by the DrugAge fold): at most one pooled
    item per compound and direction, the evidence unit being the paper. With `enabled=False` (or while that module
    does not exist) the identity: one item per row, which reproduces the legacy per-row results."""
    if not enabled:
        return _identity_fold
    try:
        from ontology.drugage_fold import fold_items            # type: ignore[import-not-found]
        return fold_items
    except ImportError:
        return _identity_fold


#: the DrugAge item handed to fold_items and received back (a list of these, one per direction after pooling):
#: {"compound", "target", "direction" ('Pos' | 'Neg': the direction of the lifespan change), "strength", "confidence",
#:  "rows" (tuple of row names), "papers" (tuple of PMIDs, '' when a row has none)}; the items handed IN also carry
#: "change" (AvgLifespanChangePercent), "significance", "itp" (bool) and "species", with which the fold applies the
#: repo's representative-row policy inside a paper. Extra keys of an item that comes back are kept in `Link.detail`.
DRUGAGE_ITEM_KEYS = ("compound", "target", "direction", "strength", "confidence", "rows", "papers")


def _atom_text(text: str, start: int, limit: int = 600) -> str:
    """The balanced parenthesised form that begins at `text[start]` (an atom may span lines), comments and strings
    aside; at most `limit` characters."""
    depth = 0
    in_string = in_comment = False
    for i in range(start, min(len(text), start + limit)):
        ch = text[i]
        if in_comment:
            in_comment = ch != "\n"
        elif in_string:
            in_string = ch != '"'
        elif ch == ";":
            in_comment = True
        elif ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:start + limit]


class _LineIndex:
    """Line numbers of offsets in one text, by bisection (a count per atom is quadratic on a large slice)."""

    def __init__(self, text: str):
        self._breaks = [m.start() for m in re.finditer("\n", text)]

    def line(self, pos: int) -> int:
        return bisect.bisect_left(self._breaks, pos) + 1


class _Occurrences:
    """Where the atoms of the knowledge-base text sit (file:line), to say what an evidence id stands for."""

    def __init__(self, blocks: Sequence[tuple[str, str]]):
        self.effects: dict[tuple[str, str, str], list[tuple[str, int, Optional[str]]]] = {}
        self.records: dict[str, list[tuple[str, int]]] = {}
        self.categories: dict[str, str] = {}
        #: DrugAge row -> the blocks that state `(UsesIntervention row compound)`; tells a caller's row from the ETL's
        self.row_blocks: dict[str, set] = {}
        effect_re = re.compile(r"^[ \t]*(\(Effect\s+(\S+)\s+(\S+)\s+([^\s()]+))", re.M)       # any sign: an unknown one is a skip, and has an origin
        record_re = re.compile(r"^[ \t]*\(EffectValue\s+(\S+)\s+", re.M)
        category_re = re.compile(r"^[ \t]*\(HasEvidenceCategory\s+(\S+)\s+([^\s()]+)\s*\)", re.M)
        uses_re = re.compile(r"^[ \t]*\(UsesIntervention\s+(\S+)\s", re.M)
        for name, text in blocks:
            lines = None
            if "(Effect" in text:
                lines = _LineIndex(text)
                for m in effect_re.finditer(text):
                    tier = _EVIDENCE_TIER_RE.search(_atom_text(text, m.start(1)))
                    self.effects.setdefault((m.group(2), m.group(3), m.group(4)), []).append(
                        (name, lines.line(m.start(1)), tier.group(1) if tier else None))
            if "(EffectValue" in text:
                lines = lines or _LineIndex(text)
                for m in record_re.finditer(text):
                    self.records.setdefault(m.group(1), []).append((name, lines.line(m.start())))
            if "(HasEvidenceCategory" in text:
                for m in category_re.finditer(text):
                    self.categories.setdefault(m.group(1), m.group(2))
            if "(UsesIntervention" in text:
                for m in uses_re.finditer(text):
                    self.row_blocks.setdefault(m.group(1), set()).add(name)
        self._taken: dict[Any, int] = {}

    def effect(self, a: str, b: str, sign: str) -> Optional[tuple[str, int, Optional[str]]]:
        key = (a, b, sign)
        n = self._taken.get(key, 0)
        self._taken[key] = n + 1
        found = self.effects.get(key, [])
        return found[n] if n < len(found) else None

    def record(self, rec: str) -> Optional[tuple[str, int]]:
        key = ("record", rec)
        n = self._taken.get(key, 0)
        self._taken[key] = n + 1
        found = self.records.get(rec, [])
        return found[n] if n < len(found) else None


def _truth(tv: Any) -> tuple[Optional[float], Optional[float]]:
    """(s, c) of an `(stv s c)` tree, None for a part that is not a number."""
    if isinstance(tv, tuple) and len(tv) == 3 and tv[0] == "stv":
        return number(tv[1]), number(tv[2])
    return None, None


def _truth_problem(s: Optional[float], c: Optional[float]) -> Optional[str]:
    if s is None or c is None:
        return "NonNumericTruthValue"
    if not 0.0 <= s <= 1.0:
        return "StrengthOutOfRange"
    if not 0.0 <= c <= 1.0:
        return "ConfidenceOutOfRange"
    return None


def _fold_for(settings_fold: Optional[Callable[[list[dict]], list[dict]]], enabled: bool, zero_rows: str) -> Callable[[list[dict]], list[dict]]:
    """The fold `collect_links` calls: the injected one (tests), else `_fold_items(enabled)` with the null convention."""
    if settings_fold is not None:
        return settings_fold
    fold = _fold_items(enabled)
    if fold is _identity_fold:
        return fold
    return functools.partial(fold, zero_rows=zero_rows)


def _sanitize_drugage(items: list[dict], skip: Callable[..., None]) -> list[dict]:
    """Items that the fold may take: a row stated twice with identical content is ONE item (the same atom pasted twice is not two
    studies); a row whose copies disagree -- two changes, two compounds, two significances -- cannot be lifted, so every item of it
    is skipped and listed. Legacy would answer both (with duplicate or contradictory derivations); the fold has to refuse."""
    by_row: dict[str, list[dict]] = {}
    for item in items:
        by_row.setdefault(item["rows"][0], []).append(item)
    keep: list[dict] = []
    for row, group in by_row.items():
        first = group[0]
        if all(g == first for g in group[1:]):
            keep.append(first)
            continue
        skip(first["compound"], first["target"], "RowStatedTwiceDifferently", "drugage", row=row,
             atom=f"DrugAge row {row} ({len(group)} readings that disagree)")
    return keep


def collect_links(ev: Evaluate, blocks: Sequence[tuple[str, str]], *, caller_blocks: Iterable[str] = ("extra_atoms",),
                  fold: Optional[Callable[[list[dict]], list[dict]]] = None, drugage_fold: bool = True,
                  zero_rows: str = "both") -> tuple[list[Link], list[dict], dict[str, Any]]:
    """Every link of the loaded space, with evidence ids: (links, skipped, nodes).

    The links are read off the space with the knowledge base's own functions: the `Effect` atoms (their truth values
    are literals by now, `reduce_stored_calls` having evaluated the stored tiers), the hazard-ratio records through
    `calibrate-tv` and `hr->sign` exactly as `link-effect` lifts them, DrugAge rows through `drugage-effect`. A link
    whose truth value is not a number in [0, 1] cannot be chained and is skipped, with the reason (`skipped`; each entry
    says whether it came from the caller's block: `origin`).
    `nodes` maps a node key to its parsed tree (an explain chain prints the original atom)."""
    caller = set(caller_blocks)
    all_text = "\n".join(text for _, text in blocks)
    where = _Occurrences(blocks)
    links: list[Link] = []
    skipped: list[dict] = []
    nodes: dict[str, Any] = {}
    row_origin: dict[str, str] = {}          # DrugAge row -> 'caller' | 'curated'

    def node(tree: Any) -> str:
        key = serialize(tree)
        nodes.setdefault(key, tree)
        return key

    def skip(a: Any, b: Any, reason: str, kind: str, *, origin: str = "", atom: str = "", row: str = "") -> None:
        skipped.append({"from": serialize(a), "to": serialize(b), "reason": reason, "kind": kind,
                        "origin": origin or row_origin.get(row, "curated"),
                        "atom": atom or f"{kind} {serialize(a)} -> {serialize(b)}", **({"row": row} if row else {})})

    for raw in ev(_EFFECT_Q):
        row = parse_sexpr(raw)
        if not (isinstance(row, tuple) and len(row) == 5 and row[0] == "L"):
            raise PlnChainerError(f"unexpected Effect row from the space: {raw[:120]}")
        _, a, b, sign, tv = row
        found = where.effect(serialize(a), serialize(b), str(sign))
        origin, line, tier = found if found else (None, 0, None)
        kind = "caller" if origin in caller else "curated"
        s, c = _truth(tv)
        atom = f"(Effect {serialize(a)} {serialize(b)} {sign} {serialize(tv)})"
        if sign not in ("Pos", "Neg"):
            skip(a, b, "UnknownSign", "effect", origin=kind, atom=atom)
            continue
        problem = _truth_problem(s, c)
        if problem:
            skip(a, b, problem, "effect", origin=kind, atom=atom)
            continue
        links.append(Link(node(a), node(b), sign, s, c, kind=kind,
                          source=f"{origin}:{line}" if origin else "", tier=tier))

    if _HAS_CALIBRATE_RE.search(all_text):
        for raw in ev(_RECORD_Q):
            row = parse_sexpr(raw)
            if not (isinstance(row, tuple) and len(row) == 5 and row[0] == "R"):
                raise PlnChainerError(f"unexpected record row from the space: {raw[:120]}")
            _, rec, a, b, signed = row
            found = where.record(serialize(rec))
            origin = "caller" if (found and found[0] in caller) else "curated"
            atom = f"record {serialize(rec)} ({serialize(a)} -> {serialize(b)})"
            if not (isinstance(signed, tuple) and len(signed) == 3 and signed[0] == "signed"):
                skip(a, b, "NonNumericTruthValue", "record", origin=origin, atom=atom)
                continue
            sign = signed[1]
            s, c = _truth(signed[2])
            problem = None if sign in ("Pos", "Neg") else "UnknownSign"
            problem = problem or _truth_problem(s, c)
            if problem:
                skip(a, b, problem, "record", origin=origin, atom=atom)
                continue
            links.append(Link(node(a), node(b), sign, s, c, terminal=True, kind="record",
                              source=f"{found[0]}:{found[1]}" if found else "",
                              tier=where.categories.get(serialize(rec)), detail={"record": serialize(rec)}))

    if _HAS_DRUGAGE_RE.search(all_text):
        papers: dict[str, str] = {}
        for raw in ev(_PAPER_Q):
            row = parse_sexpr(raw)
            if isinstance(row, tuple) and len(row) == 3 and row[0] == "P":
                papers.setdefault(serialize(row[1]), serialize(row[2]))
        significance: dict[str, str] = {}
        for raw in ev(_SIGNIFICANCE_Q):
            row = parse_sexpr(raw)
            if isinstance(row, tuple) and len(row) == 3 and row[0] == "G":
                significance.setdefault(serialize(row[1]), serialize(row[2]))
        itp_rows: set[str] = set()
        for raw in ev(_ITP_Q):
            row = parse_sexpr(raw)
            if isinstance(row, tuple) and len(row) == 2 and row[0] == "I":
                itp_rows.add(serialize(row[1]))
        species: dict[str, str] = {}
        for raw in ev(_SPECIES_Q):
            row = parse_sexpr(raw)
            if isinstance(row, tuple) and len(row) == 3 and row[0] == "S":
                species.setdefault(serialize(row[1]), serialize(row[2]))
        items: list[dict] = []
        for raw in ev(_DRUGAGE_Q):
            row = parse_sexpr(raw)
            if not (isinstance(row, tuple) and len(row) == 5 and row[0] == "D"):
                raise PlnChainerError(f"unexpected DrugAge row from the space: {raw[:120]}")
            _, rowname, cmp_, pct, eff = row
            if not (isinstance(eff, tuple) and len(eff) == 5 and eff[0] == "Effect"):
                continue                    # drugage-effect gave nothing (a row without a lift)
            _, comp, target, sign, tv = eff
            s, c = _truth(tv)
            r = serialize(rowname)
            row_origin[r] = "caller" if (where.row_blocks.get(r, set()) & caller) else "curated"
            problem = None if sign in ("Pos", "Neg") else "UnknownSign"
            problem = problem or _truth_problem(s, c)
            if problem:
                skip(comp, target, problem, "drugage", origin=row_origin[r], atom=f"row {r}", row=r)
                continue
            items.append({"compound": node(comp), "target": node(target), "direction": sign, "strength": s,
                          "confidence": c, "rows": (r,), "papers": (papers.get(r, ""),),
                          "change": number(pct), "significance": significance.get(r), "itp": r in itp_rows,
                          "species": species.get(r)})
        items = _sanitize_drugage(items, skip)
        # fold compound by compound: a compound whose items the fold refuses (a pooled confidence of 0, ...) is skipped and listed,
        # and one bad compound never costs another its answer
        folder = _fold_for(fold, drugage_fold, zero_rows)
        by_compound: dict[str, list[dict]] = {}
        for item in items:
            by_compound.setdefault(item["compound"], []).append(item)
        pooled: list[dict] = []
        for comp in sorted(by_compound):
            group = by_compound[comp]
            try:
                pooled.extend(folder(group))
            except (ValueError, ArithmeticError) as exc:
                reason = f"FoldRefused:{type(exc).__name__}"
                origin = "caller" if any(row_origin.get(g["rows"][0]) == "caller" for g in group) else "curated"
                skipped.append({"from": comp, "to": group[0]["target"], "reason": reason, "kind": "drugage", "origin": origin,
                                "atom": f"{len(group)} row(s) of {comp}: {exc}"[:200], "row": group[0]["rows"][0]})
        for item in pooled:
            rows = tuple(item.get("rows") or ())
            pap = tuple(p for p in (item.get("papers") or ()) if p)
            extra = {k: v for k, v in item.items() if k not in DRUGAGE_ITEM_KEYS}
            links.append(Link(item["compound"], item["target"], item["direction"], float(item["strength"]),
                              float(item["confidence"]), kind="drugage", needs_continuation=True,
                              source=", ".join(rows) + (f" ({', '.join(pap)})" if pap else ""),
                              detail={"rows": rows, "papers": pap, **extra}))
    assign_ids(links)
    assert_ids_unique_along_paths(links)
    return links, skipped, nodes


# ── the library's formulas, read from the library ───────────────────────────────

def _one(ev: Evaluate, expr: str) -> str:
    out = ev(expr)
    if len(out) != 1:
        raise PlnChainerError(f"expected exactly one result of {expr[:60]}, got {len(out)}")
    return out[0]


def _stv_of(text: str) -> tuple[float, float]:
    tree = parse_sexpr(text)
    s, c = _truth(tree)
    if s is None or c is None:
        raise PlnChainerError(f"expected (stv s c), got {text[:80]}")
    return s, c


def probe_library(ev: Evaluate) -> float:
    """Check the loaded library is the one the encoding was built for and return its Modus Ponens leak e.

    Compared in Python, never with MeTTa's `test`/`assert` (they halt the process on a mismatch):
    `Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8)` is (0.26, 0.64); `Truth_ModusPonens (stv 0 1) (stv 0 1)` reads
    the leak e = 0.02 (a library whose Modus Ponens changed would change the mapping); `Truth_Revision` with a
    confidence of 1.0 gives no result (a revision of a certain truth value is undefined, so a fold caps c below 1)."""
    s, c = _stv_of(_one(ev, "(Truth_ModusPonens (stv 0.5 0.8) (stv 0.5 0.8))"))
    if abs(s - 0.26) > _PROBE_TOL or abs(c - 0.64) > _PROBE_TOL:
        raise PlnChainerError(f"the library's Modus Ponens is not the fork's: (0.5 0.8)(0.5 0.8) gave ({s}, {c})")
    leak, cc = _stv_of(_one(ev, "(Truth_ModusPonens (stv 0.0 1.0) (stv 0.0 1.0))"))
    if not (0.0 < leak < _MAX_LEAK) or abs(cc - 1.0) > _PROBE_TOL:
        raise PlnChainerError(f"the library's Modus Ponens leak is not the fork's small constant: ({leak}, {cc})")
    if ev("(Truth_Revision (stv 0.5 1.0) (stv 0.5 0.5))"):
        raise PlnChainerError("the library revises a truth value of confidence 1.0; the fold's cap assumes it cannot")
    return leak


def read_discount(ev: Evaluate) -> float:
    """`(chain-discount)` of pln_deduction.metta (0.9): the legacy per-combination confidence discount."""
    out = ev("(chain-discount)")
    value = number(out[0]) if len(out) == 1 else None
    if value is None or not 0.0 < value <= 1.0:
        raise PlnChainerError(f"(chain-discount) did not give a number in (0, 1]: {out!r}")
    return value


# ── one source: two PLN.Derive runs, joined ─────────────────────────────────────

@dataclass
class Route:
    """One conclusion: a node-simple route from `source` to `target` and the truth value PLN.Derive gave it."""
    source: str
    target: str
    end: str                         #: 'State' (reached through an Effect) | 'End' (a hazard-ratio record)
    path: tuple[str, ...]
    sign: str                        #: 'Pos' | 'Neg' (parity of the Neg links)
    strength: float                  #: |s1 - s0| / (1 - e)^hops
    confidence: float                #: per PLN_CHAIN_CONFIDENCE
    native_confidence: float         #: the library's c
    hops: int
    stamp: tuple[int, ...]           #: the evidence ids of the route (sorted)
    level1: float                    #: the state strength with the intervention applied (an artefact; for audit)
    level0: float                    #: ... with it absent

    def key(self) -> tuple:
        return (self.hops, self.stamp)


@dataclass
class SourceStatus:
    source: str
    complete: bool
    queue: int                       #: K (the belief queue size) of the attempt that was kept
    task_queue: int
    max_steps: int
    links: int                       #: size of the sub-graph
    attempts: int                    #: how many K rungs were tried
    beliefs: tuple[int, int]         #: final belief count of the level-1 and level-0 runs
    tasks_left: tuple[int, int]
    selections: Optional[tuple[int, int]] = None
    routes: int = 0
    #: conclusions of the level-1 run that are not a route of the graph (0 for the restricted rule set: no false chain)
    unattributed: int = 0


@dataclass
class ChainTable:
    """Everything a request's `infer`/`explain` read, with how it was derived."""
    links: list[Link]
    nodes: dict[str, Any]
    leak: float
    discount: float
    settings: ChainSettings
    routes: dict[str, list[Route]] = field(default_factory=dict)
    status: dict[str, SourceStatus] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)
    derive_runs: int = 0
    derive_s: float = 0.0

    @property
    def complete(self) -> bool:
        return all(s.complete for s in self.status.values())

    def all_routes(self) -> list[Route]:
        return [r for src in sorted(self.routes) for r in self.routes[src]]

    def evidence(self) -> dict[int, dict]:
        """The side table: evidence id -> what it stands for."""
        out = {}
        for link in self.links:
            out[link.id] = {"kind": link.kind, "source": link.source, "tier": link.tier, "from": link.src,
                            "to": link.dst, "sign": link.sign, "strength": link.strength,
                            "confidence": link.confidence, "terminal": link.terminal,
                            **{k: list(v) if isinstance(v, tuple) else v for k, v in link.detail.items()}}
        return out

    def summary(self) -> dict:
        """A small JSON-able account of the table (comes back as `last_run["chain"]`)."""
        return {
            "links": len(self.links), "sources": len(self.status), "routes": sum(len(v) for v in self.routes.values()),
            "complete": self.complete, "derive_runs": self.derive_runs, "derive_s": round(self.derive_s, 4),
            "leak": self.leak, "discount": self.discount, "chain_confidence": self.settings.chain_confidence,
            "ruleset": self.settings.ruleset, "mapping": self.settings.mapping,
            "unattributed": sum(s.unattributed for s in self.status.values()),
            "queue_max": max((s.queue for s in self.status.values()), default=0),
            "attempts_max": max((s.attempts for s in self.status.values()), default=0),
            "skipped": self.skipped, "evidence": {str(k): v for k, v in self.evidence().items()},
            "status": {k: {"complete": v.complete, "queue": v.queue, "task_queue": v.task_queue,
                           "max_steps": v.max_steps, "links": v.links, "attempts": v.attempts,
                           "beliefs": list(v.beliefs), "selections": list(v.selections) if v.selections else None,
                           "routes": v.routes, "unattributed": v.unattributed}
                       for k, v in self.status.items()},
        }


class _Encoding:
    """The sentences of one graph: opaque node names, one sentence per link."""

    def __init__(self, links: Sequence[Link], leak: float, mapping: str = "affine"):
        names = sorted({link.src for link in links} | {link.dst for link in links})
        self.opaque = {key: f"n{i}" for i, key in enumerate(names)}
        self.key_of = {v: k for k, v in self.opaque.items()}
        self.leak = leak
        self.mapping = mapping
        self.link_of = {link.id: link for link in links}
        self.sentence: dict[int, str] = {}
        for link in links:
            tgt = f"({'End' if link.terminal else 'State'} {self.opaque[link.dst]})"
            if link.sign == "Neg":
                tgt = f"(Not {tgt})"
            if mapping == "affine":
                q = leak + (1.0 - leak) * link.strength
            elif mapping == "capped":
                q = min(1.0, link.strength + leak)
            else:
                q = link.strength
            self.sentence[link.id] = (f"(Sentence ((Implication (State {self.opaque[link.src]}) {tgt}) "
                                      f"(stv {fnum(q)} {fnum(link.confidence)})) ({link.id}))")

    def premise(self, source: str, level: float) -> str:
        return f"(Sentence ((State {self.opaque[source]}) (stv {fnum(level)} 1.0)) (0))"


def _parse_state(text: str) -> tuple[list, list]:
    tree = parse_sexpr(text)
    if not (isinstance(tree, tuple) and len(tree) == 2):
        raise PlnChainerError(f"PLN.Derive did not return (Tasks Beliefs): {text[:120]}")
    return [_parse_sentence(x) for x in tree[0]], [_parse_sentence(x) for x in tree[1]]


def _parse_sentence(tree: Any) -> tuple:
    """('Sentence' ((term (stv s c)) (ids..))) -> (term, s, c, ids)"""
    try:
        assert tree[0] == "Sentence"
        (term, tv), ids = tree[1], tree[2]
        s, c = _truth(tv)
        assert s is not None and c is not None
        return term, s, c, tuple(int(i) for i in ids)
    except (AssertionError, IndexError, TypeError, ValueError):
        raise PlnChainerError(f"a sentence of PLN.Derive's result is malformed: {serialize(tree)[:120]}") from None


def _derive(ev: Evaluate, tasks: str, beliefs: str, steps: int, maxsteps: int, tq: int, bq: int) -> tuple[list, list]:
    out = ev(f"(PLN.Derive {tasks} {beliefs} {steps} {maxsteps} {tq} {bq})")
    if len(out) != 1:
        # a failed pattern inside the library comes back as no result, which is a failure and not "no conclusions"
        raise PlnChainerError(f"PLN.Derive returned {len(out)} results; exactly one is a table")
    return _parse_state(out[0])


def _run(ev: Evaluate, enc: _Encoding, sub: Sequence[Link], source: str, level: float, steps_cap: int, tq: int, bq: int,
         count: bool, seed_all: bool = False) -> tuple[list, list, Optional[int]]:
    """One PLN.Derive run. The task is the intervention alone (the service); `seed_all` makes every sentence a task, which
    is what `PLN.Query kb term` does and what the rules that start from an implication (inversion, induction, ...) need:
    the full-rule-set research lane. It loses most pairs with the restricted rules (52 of 100), and is not a way to run."""
    app = enc.premise(source, level)
    beliefs = " ".join([enc.sentence[link.id] for link in sub] + [app])
    if not count:
        tasks_text = f"({beliefs})" if seed_all else f"({app})"
        tasks_left, beliefs_out = _derive(ev, tasks_text, f"({beliefs})", 1, steps_cap, tq, bq)
        return tasks_left, beliefs_out, None
    # chunked resumption with the 6-argument form: one selection per call, the state handed back each time
    tasks_text, beliefs_text = (f"({beliefs})" if seed_all else f"({app})"), f"({beliefs})"
    n = 0
    tasks_left: list = [None]
    beliefs_out: list = []
    while tasks_left and n < steps_cap:
        tasks_left, beliefs_out = _derive(ev, tasks_text, beliefs_text, n + 1, n + 1, tq, bq)
        n += 1
        tasks_text = "(" + " ".join(_sentence_text(x) for x in tasks_left) + ")"
        beliefs_text = "(" + " ".join(_sentence_text(x) for x in beliefs_out) + ")"
    return tasks_left, beliefs_out, n


def _sentence_text(sentence: tuple) -> str:
    term, s, c, ids = sentence
    return f"(Sentence ({serialize(term)} (stv {fnum(s)} {fnum(c)})) ({' '.join(str(i) for i in ids)}))"


def count_selections(ev: Evaluate, links: Sequence[Link], source: str, leak: float, *, level: float = 1.0,
                     queue: int = 256, max_steps: int = 0) -> int:
    """How many tasks PLN.Derive expands for `source` at one level (chunked resumption), for reports."""
    assign = list(links)
    enc = _Encoding(assign, leak)
    sub = reachable_links(assign, source)
    cap = max_steps or 4 * queue
    return _run(ev, enc, sub, source, level, cap, queue, queue, True)[2] or 0


def _trail(enc: _Encoding, source: str, ids: Sequence[int], end: Optional[str] = None,
           end_terminal: Optional[bool] = None, simple: bool = False) -> Optional[list[Link]]:
    """The links of a stamp in the order a walk from `source` uses them (each link once, a record last), or None when
    the stamp is not a walk of the graph. With `end` the walk has to finish at that node (through a record when
    `end_terminal`); with `simple` it may not visit a node twice, except that a closing record may name a node the walk
    already visited, the source included (the legacy chainer answers `explain A A` through a record that returns to A, and
    a record is terminal, so it can start no cycle). A node-simple route has exactly one order; a walk that comes back to
    a node (a feedback edge) can have several, any one is returned."""
    wanted = [i for i in ids if i != 0]
    used = [enc.link_of[i] for i in wanted if i in enc.link_of]
    if not used or len(used) != len(wanted) or len({link.id for link in used}) != len(used):
        return None
    by_src: dict[str, list[Link]] = {}
    for link in used:
        by_src.setdefault(link.src, []).append(link)

    def walk(node: str, left: frozenset, acc: list, seen: frozenset) -> Optional[list]:
        if not left:
            if end is not None and (node != end or acc[-1].terminal != bool(end_terminal)):
                return None
            return acc
        for link in by_src.get(node, ()):
            if link.id in left and not (link.terminal and len(left) > 1) and not (simple and link.dst in seen and not link.terminal):
                got = walk(link.dst, left - {link.id}, acc + [link], seen | {link.dst})
                if got is not None:
                    return got
        return None

    return walk(source, frozenset(link.id for link in used), [], frozenset([source]))


def _routes(enc: _Encoding, source: str, b1: list, b0: list, table_discount: float, st: ChainSettings,
            unattributed: Optional[list] = None) -> list[Route]:
    """The adapter's join: the level-1 and level-0 beliefs with the same stamp, one route each.

    A conclusion `(State C)`/`(End C)` of the level-1 run whose stamp is not a node-simple route of the graph (for
    instance one that Induction or Abduction derived) cannot be named a route; each is appended to `unattributed`
    (term, stamp). With the restricted rule set there is none: Modus Ponens and Negation-elimination conclude only
    along links, which is the no-false-chain property, checked on every run.

    A route may END at the source through a hazard-ratio record (`explain A A` over A -> B and a record B -> A): the legacy
    chainer answers it, and a record is terminal, so it starts no cycle. An `Effect` cycle back to the source is not a route
    (the legacy chainer does not terminate on one)."""
    lvl0: dict[tuple, float] = {}
    for term, s, c, ids in b0:
        if isinstance(term, tuple) and len(term) == 2 and term[0] in ("State", "End") and ids and ids[0] == 0:
            lvl0[(term[0], term[1], ids)] = s
    routes: list[Route] = []
    seen_conclusion: set = set()
    for term, s1, c1, ids in b1:
        if not (isinstance(term, tuple) and len(term) == 2 and term[0] in ("State", "End")):
            continue
        if (term, ids) in seen_conclusion:
            continue                    # only the full lane can conclude one term from one stamp twice; the first stands
        seen_conclusion.add((term, ids))
        if not ids or ids[0] != 0 or len(ids) == 1:          # not a derivation, or the premise itself (stamp (0))
            continue
        s0 = lvl0.get((term[0], term[1], ids))
        target = enc.key_of[term[1]]
        path_links = _trail(enc, source, ids, end=target, end_terminal=(term[0] == "End"), simple=True)
        if path_links is None:
            # not a node-simple route. A walk that comes back to a node (a feedback edge) is a legitimate derivation that
            # the node-simple contract does not answer; a stamp that is no walk of the graph ending at this node is a
            # conclusion the links do not license: a false chain.
            if unattributed is not None and _trail(enc, source, ids, end=target, end_terminal=(term[0] == "End")) is None:
                unattributed.append((term, ids))
            continue
        if s0 is None:
            continue
        if path_links[-1].needs_continuation:
            continue                    # a DrugAge link is the start of a chain, never an answer
        hops = len(path_links)
        ate = s1 - s0
        negatives = sum(1 for link in path_links if link.sign == "Neg")
        sign = "Neg" if negatives % 2 else "Pos"
        # the parity is checked against the library's own difference only where that difference IS the product (the affine
        # mapping, the service's). The research readings (`literal`: q = p, `capped`) are not: a link weaker than the leak e comes back
        # from the fork-literal reading with the wrong sign (a strength-0 link: s1 - s0 = -e), which is what that lane exists to show.
        if enc.mapping == "affine" and abs(ate) > _SIGN_CROSSCHECK_MIN and ((ate < 0) != (sign == "Neg")):
            raise PlnChainerError(
                f"sign cross-check failed on the route {' -> '.join(l.src for l in path_links)} -> {target}: "
                f"the parity of its Neg links says {sign} and the library's s1 - s0 is {ate:.3g}")
        # a product of numbers in [0, 1] is in [0, 1]; the difference of two levels can land one ulp above 1.0
        divisor = (1.0 - enc.leak) ** hops if enc.mapping == "affine" else 1.0
        strength = min(1.0, abs(ate) / divisor)
        conf = c1 if st.chain_confidence == "native" else c1 * (table_discount ** (hops - 1))
        nodes = tuple([source] + [l.dst for l in path_links])
        routes.append(Route(source, target, term[0], nodes, sign, strength, conf, c1, hops,
                            tuple(i for i in ids if i != 0), s1, s0))
    routes.sort(key=Route.key)
    return routes


def count_routes(sub: Sequence[Link], source: str, cap: int) -> int:
    """How many node-simple routes leave `source` in the sub-graph `sub`, counting up to `cap` (then `cap + 1`).
    The legacy chainer's derivations, counted without running anything. (The library derives MORE on a cyclic graph:
    every walk that uses each LINK once. That is `walk_stats`, and what the cost is made of.)"""
    out_by = out_edges(sub)
    count = 0
    stack: list[tuple[str, frozenset]] = [(source, frozenset([source]))]
    while stack:
        node, visited = stack.pop()
        for link in out_by.get(node, ()):
            if link.dst in visited:
                continue
            count += 1
            if count > cap:
                return count
            if not link.terminal:
                stack.append((link.dst, visited | {link.dst}))
    return count


@dataclass(frozen=True)
class WalkStats:
    """What `PLN.Derive` will hold for one source, counted without running it."""
    states: int          #: distinct conclusions: (end node, set of links used); `cap + 1` when there are more than `cap`
    stamp_total: int     #: the sum of their stamp sizes (links used)
    neg_final: int       #: conclusions whose last link is a Neg one: they also carry the `(Not ..)` sentence
    capped: bool

    @property
    def mean_stamp(self) -> float:
        return self.stamp_total / self.states if self.states else 0.0


def walk_stats(sub: Sequence[Link], source: str, cap: int, out_by: Optional[dict] = None) -> WalkStats:
    """The conclusions the library derives from `source` over `sub`: every walk that uses each LINK at most once (a record
    last), counted as distinct (end node, links used) -- the library keeps one sentence for each. On a graph without a cycle
    that is the number of node-simple routes; on a cyclic one it is far larger (a feedback edge doubles the walks, eight
    reversed edges multiply them by thousands), which is why a count of node-simple routes picked queue sizes that were
    hopeless. Counts up to `cap` states, then stops with `capped=True`."""
    if out_by is None:
        out_by = out_edges(sub)
    seen: set = set()
    total = neg = 0
    stack: list[tuple[str, frozenset]] = [(source, frozenset())]
    while stack:
        node, used = stack.pop()
        for link in out_by.get(node, ()):
            key = id(link)
            if key in used:
                continue
            nxt = used | {key}
            state = (link.dst, link.terminal, nxt)
            if state in seen:
                continue
            seen.add(state)
            total += len(nxt)
            neg += link.sign == "Neg"
            if len(seen) > cap:
                return WalkStats(len(seen), total, neg, True)
            if not link.terminal:
                stack.append((link.dst, nxt))
    return WalkStats(len(seen), total, neg, False)


def count_states(sub: Sequence[Link], source: str, cap: int, out_by: Optional[dict] = None) -> int:
    """`walk_stats(...).states`."""
    return walk_stats(sub, source, cap, out_by).states


def estimate_beliefs(sub: Sequence[Link], source: str, cap: int) -> int:
    """An upper bound of the beliefs one run ends with: the links, the premise, and at most two sentences (the
    conclusion and its negation form) per distinct conclusion. Used to pick the first queue size; the completeness test
    after the run is what proves nothing was dropped."""
    return len(sub) + 1 + 2 * count_states(sub, source, cap)


#: seconds of one `PLN.Derive` run per (belief count)^2 x (mean stamp size), measured on PeTTa (the library compares the
#: selected task with every belief, and its stamp checks and sorts walk the stamps): 0.86e-6 .. 1.12e-6 over 15 graphs
#: from 12 to 4,908 beliefs (chains, layered DAGs, rings with chords, a complete digraph); see docs/petta_migration/
#: fork_design/fix/COST_MODEL.md. A planning figure for refusing the hopeless, not a promise.
_SECONDS_PER_UNIT = 1.0e-6
#: a fixed cost per run (the call, the parse of the beliefs)
_SECONDS_PER_RUN = 0.002


def model_seconds(links: int, stats: WalkStats) -> float:
    """The modelled seconds for one source: two runs (the intervention at level 1 and at level 0) of
    `_SECONDS_PER_UNIT x beliefs^2 x mean stamp size`, beliefs = links + 1 + the conclusions and their `(Not ..)` forms."""
    beliefs = links + 1 + stats.states + stats.neg_final
    return 2.0 * (_SECONDS_PER_RUN + _SECONDS_PER_UNIT * beliefs * beliefs * max(1.0, stats.mean_stamp))


def _rungs(st: ChainSettings, need: int) -> list[int]:
    """The queue sizes to try: K, 8K, 64K ... starting at the first that exceeds `need` (what the run is
    expected to hold, so an attempt that cannot fit is not run). When none does, the request is refused with a
    named error (contract mode), or only the last rung is tried and its table flagged `partial` (`on_incomplete`)."""
    sizes = [st.queue_size * st.growth ** i for i in range(st.rungs)]
    start = next((i for i, k in enumerate(sizes) if k > need), None)
    if start is None:
        if st.on_incomplete == "partial":
            return sizes[-1:]
        raise PlnInputError(f"the derivation is expected to hold about {need:,} sentences, more than the largest queue "
                            f"(K={sizes[-1]:,}) can; the graph is too dense or too cyclic for the PLN chainer")
    return sizes[start:]


@dataclass(frozen=True)
class SourcePlan:
    """What one source costs, worked out from the links alone (before the library is loaded)."""
    source: str
    links: int                       #: the sub-graph reachable from it
    stats: WalkStats
    seconds: float                   #: modelled, both levels (model_seconds)

    @property
    def beliefs(self) -> int:
        return self.links + 1 + self.stats.states + self.stats.neg_final


#: a request is refused when the modelled derivation takes more than this share of its time budget
_BUDGET_SHARE = 0.8


def _clip(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[:n - 3] + "..."


def request_budget_s() -> float:
    """The seconds a request has (PLN_QUERY_TIMEOUT_SECONDS; 0 = none)."""
    import config
    return float(getattr(config, "PLN_QUERY_TIMEOUT_SECONDS", 60) or 0)


def plan_sources(links: Sequence[Link], settings: Optional[ChainSettings] = None, sources: Optional[Iterable[str]] = None, *,
                 out_by: Optional[dict] = None, budget_s: Optional[float] = None) -> list[SourcePlan]:
    """What deriving `sources` (default: every node with a link out of it) over `links` costs, and a refusal in milliseconds
    when it cannot be done in a request's time.

    Counted from the links alone (`walk_stats`), before the library is loaded: for every source the links reachable from
    it, the distinct walks the library will derive, the beliefs they make and the modelled seconds. Contract mode refuses
    (`PlnInputError`, naming the source and the numbers) a source whose walks cannot fit the largest queue, and a table
    whose modelled time exceeds 80 percent of `budget_s` (default: the request deadline). The research lanes (separate queue
    sizes, the full rule set, `on_incomplete='partial'`) are never refused: they choose their own budget."""
    st = settings or settings_from_config()
    link_list = list(links)
    out_by = out_edges(link_list) if out_by is None else out_by
    srcs = sorted({l.src for l in link_list}) if sources is None else sorted(set(sources))
    largest = st.queue_size * st.growth ** (st.rungs - 1)
    refuse = not (st.control or st.ruleset == "full" or st.on_incomplete != "retry")
    budget = request_budget_s() if budget_s is None else float(budget_s)
    plans: list[SourcePlan] = []
    for source in srcs:
        if source not in out_by:
            continue
        sub = reachable_links(link_list, source, out_by)
        if not sub:
            continue
        stats = walk_stats(sub, source, largest, out_by)
        plans.append(SourcePlan(source, len(sub), stats, model_seconds(len(sub), stats)))
        if refuse and (stats.capped or len(sub) + 1 + 2 * stats.states + 2 > largest):
            walks = f"more than {largest:,}" if stats.capped else f"{stats.states:,}"
            raise PlnInputError(
                f"the derivation from {_clip(source)} cannot be done: {walks} distinct walks over {len(sub)} reachable links "
                f"(each link used once) do not fit the largest queue (K={largest:,}); the graph is too dense or too cyclic for "
                f"the PLN chainer. Remove links or cycles, or ask about fewer sources")
    total = sum(p.seconds for p in plans)
    if refuse and budget > 0 and total > _BUDGET_SHARE * budget:
        worst = max(plans, key=lambda p: p.seconds)
        raise PlnInputError(
            f"the derivation would take about {total:,.0f} s for {len(plans)} source(s) (modelled), more than the {budget:g} s a "
            f"request has; the costliest is {_clip(worst.source)} ({worst.links} links, {worst.stats.states:,} distinct walks, "
            f"about {worst.seconds:,.0f} s). The graph is too dense, too cyclic or too long a chain for the PLN chainer; "
            f"remove links, or ask about fewer sources")
    return plans


def _check_clock(deadline_at: Optional[float], done: int, total: int) -> None:
    if deadline_at is not None and time.monotonic() > deadline_at:
        raise PlnChainerError(f"the derivation used up the request's time budget after {done} of {total} source(s); "
                              f"the answers would be incomplete, so the request is refused")


def compute_table(ev: Evaluate, links: Sequence[Link], *, leak: float, discount: float,
                  settings: Optional[ChainSettings] = None, sources: Optional[Iterable[str]] = None,
                  nodes: Optional[dict] = None, skipped: Optional[list] = None,
                  plans: Optional[Sequence[SourcePlan]] = None, deadline_at: Optional[float] = None) -> ChainTable:
    """The answers of `sources` (default: every node with a link out of it) on the graph `links`.

    `ev` evaluates one expression on an engine that has the vendored library loaded. For each source: the
    sub-graph reachable from it, then two `PLN.Derive` runs (the intervention at level 1 and at level 0) at a queue
    size K that is raised until the run is complete (task queue empty, belief queue below K - 1) -- or, in the
    research lane (`ChainSettings.control` / `on_incomplete='partial'`), once, with the status recorded.

    Before the first run the graph is planned (`plan_sources`; pass `plans` when the caller already did): a source whose walks
    cannot be derived in a request's time is refused in milliseconds. `deadline_at` (a `time.monotonic()` value) is checked
    between sources and between queue sizes, so a table that runs out of time fails with a named error instead of being killed."""
    st = settings or settings_from_config()
    link_list = list(links)
    assert_ids_unique_along_paths(link_list)
    enc = _Encoding(link_list, leak, st.mapping)
    table = ChainTable(link_list, nodes if nodes is not None else {k: k for k in enc.opaque}, leak, discount, st,
                       skipped=list(skipped or []))
    out_by = out_edges(link_list)
    srcs = sorted({l.src for l in link_list}) if sources is None else sorted(set(sources))
    if plans is None and not (st.control or st.ruleset == "full" or st.on_incomplete != "retry"):
        plans = plan_sources(link_list, st, srcs, out_by=out_by)       # contract mode only; the research lanes pick their own budget
    steps_clamp = lambda n: max(1, min(int(n), _MAX_STEPS_CAP))
    t0 = time.perf_counter()
    todo = [s for s in srcs if s in enc.opaque and s in out_by]
    for done, source in enumerate(todo):
        _check_clock(deadline_at, done, len(todo))
        # The service runs each source on the links reachable from it (the library's cost is quadratic in the beliefs). The
        # full lane needs the whole graph: inversion, induction and abduction conclude through links that are NOT reachable
        # from the source (a shared cause makes two siblings), and hiding those would hide the false chains it exists to show.
        sub = reachable_links(link_list, source, out_by) if st.ruleset != "full" else link_list
        if not sub or not reachable_links(link_list, source, out_by):
            continue
        L = len(sub)
        if st.ruleset == "full" and not st.control:
            plan = [(FULL_LANE_BUDGET[1], FULL_LANE_BUDGET[2])]
        elif st.control:
            k0 = st.queue_size
            plan = [(st.task_queue or k0, st.belief_queue or k0)]
        else:
            plan = [(k, k) for k in _rungs(st, estimate_beliefs(sub, source, st.queue_size * st.growth ** st.rungs) + 2)]
        attempts = 0
        kept = None
        for tq, bq in plan:
            if attempts:
                _check_clock(deadline_at, done, len(todo))
            attempts += 1
            cap = steps_clamp(st.max_steps or (FULL_LANE_BUDGET[0] if st.ruleset == "full" and not st.control
                                               else 4 * max(tq, bq)))
            seed_all = st.ruleset == "full"
            t1, b1, n1 = _run(ev, enc, sub, source, 1.0, cap, tq, bq, st.count_selections, seed_all)
            t0_, b0, n0 = _run(ev, enc, sub, source, 0.0, cap, tq, bq, st.count_selections, seed_all)
            table.derive_runs += 2
            # "tasks empty and fewer than K - 1 beliefs" proves nothing was dropped only if the task queue is at least as
            # large as the belief queue (tasks are then a subset of the beliefs until the first cut); with a smaller one
            # (research lane) a task can be cut without a trace, so no completeness is claimed.
            proven = tq >= bq
            complete = proven and not t1 and not t0_ and len(b1) < bq - 1 and len(b0) < bq - 1
            kept = (tq, bq, cap, t1, b1, n1, t0_, b0, n0, complete)
            if complete:
                break
        tq, bq, cap, t1, b1, n1, t0_, b0, n0, complete = kept
        if not complete and not (st.control or st.ruleset == "full" or st.on_incomplete == "partial"):
            raise PlnChainerError(
                f"the derivation for {source} did not finish with {attempts} queue size(s) up to K={bq} "
                f"({L} reachable links, {len(t1)} task(s) left, {len(b1)} beliefs); the answers for it would be "
                f"incomplete, so the request is refused. Raise PLN_QUEUE_SIZE or PLN_MAX_STEPS, or simplify the graph")
        stray: list = []
        routes = _routes(enc, source, b1, b0, discount, st, stray)
        table.routes[source] = routes
        table.status[source] = SourceStatus(
            source, complete, bq, tq, cap, L, attempts, (len(b1), len(b0)), (len(t1), len(t0_)),
            (n1, n0) if n1 is not None else None, len(routes), len(stray))
    table.derive_s = time.perf_counter() - t0
    return table


def evaluator_for_session(session: Any, deadline_s: Optional[float] = None) -> Evaluate:
    """An `Evaluate` for a `PettaSession` (tests and tools): the session holds the library, `compute_table` drives it."""
    return lambda expr: session.run("!" + expr, deadline_s)[0]


# ── writing the table into the space ────────────────────────────────────────────

def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_facts(table: ChainTable) -> str:
    """The table as facts: PLNAnswer (one per route, sorted by source then (hops, stamp)), PLNEvidence (the side
    table), PLNStatus (per source), PLNSkipped (a link that could not be chained, with the reason)."""
    lines: list[str] = []
    for source in sorted(table.routes):
        for r in table.routes[source]:
            chain = " ".join(serialize(table.nodes.get(n, n)) for n in r.path)
            lines.append(f"({ANSWER_HEAD} {serialize(table.nodes.get(r.source, r.source))} "
                         f"{serialize(table.nodes.get(r.target, r.target))} {r.sign} "
                         f"(stv {fnum(r.strength)} {fnum(r.confidence)}) ({chain}) ({' '.join(str(i) for i in r.stamp)}))")
    for link in sorted(table.links, key=lambda l: l.id):
        tier = link.tier if link.tier else "None"
        lines.append(f"(PLNEvidence {link.id} {link.kind} {_quote(link.source)} {tier})")
    for source in sorted(table.status):
        s = table.status[source]
        lines.append(f"(PLNStatus {serialize(table.nodes.get(source, source))} "
                     f"{'complete' if s.complete else 'partial'} {s.queue})")
    for sk in table.skipped:
        lines.append(f"(PLNSkipped {_quote(sk['from'])} {_quote(sk['to'])} {sk['reason']})")
    return "\n".join(lines) + "\n"


# ── the step that runs inside the PeTTa child ────────────────────────────────────

_REASON_TEXT = {
    "StrengthOutOfRange": "the strength is not in [0, 1]",
    "ConfidenceOutOfRange": "the confidence is not in [0, 1]",
    "NonNumericTruthValue": "the strength or the confidence is not a number",
    "UnknownSign": "the sign is neither Pos nor Neg",
    "RowStatedTwiceDifferently": "the row is stated twice with different values",
}


def caller_skips_message(skipped: Sequence[dict], limit: int = 5) -> Optional[str]:
    """The refusal text for the program's OWN facts that cannot be chained (a skipped link whose origin is the caller's block),
    or None. A link of a curated file is skipped and listed (`PLNSkipped`); one the caller wrote is a mistake the caller can fix,
    and answering "no route" for it would be a wrong answer, so the request is refused with the offending atoms named."""
    bad = [s for s in skipped if s.get("origin") == "caller"]
    if not bad:
        return None
    parts = []
    for s in bad[:limit]:
        reason = s["reason"]
        why = _REASON_TEXT.get(reason) or (f"the fold refused it ({reason.split(':', 1)[1]})" if reason.startswith("FoldRefused:") else reason)
        parts.append(f"{_clip(s.get('atom') or s['from'], 120)}: {why}")
    more = f" (and {len(bad) - limit} more)" if len(bad) > limit else ""
    return ("the program's own facts cannot be chained, so no route through them can be answered: " + "; ".join(parts) + more +
            ". A strength and a confidence are numbers in [0, 1] and a sign is Pos or Neg; correct or remove them")


class ChainStep:
    """`run_program(..., after_load=step)`: load the library, derive the answers the program can ask for, add them as facts."""

    def __init__(self, blocks: Sequence[tuple[str, str]], settings: ChainSettings,
                 caller_blocks: Sequence[str] = ("extra_atoms",), expressions: Optional[Sequence[str]] = None,
                 deadline_s: Optional[float] = None):
        self.blocks = tuple(blocks)
        self.settings = settings
        self.caller_blocks = tuple(caller_blocks)
        #: the program's `!` expressions (None = unknown: every source is derived)
        self.expressions = None if expressions is None else tuple(expressions)
        #: the seconds this request has (None: the configured deadline); the step refuses what the model says cannot be done in
        #: them and stops cleanly, with a named error, before the parent would kill it
        self.deadline_s = deadline_s
        self.table: Optional[ChainTable] = None        # set in the process that ran the step (the child)

    def __call__(self, engine: Any) -> dict:
        """Run the step on `engine` (`.run(text)`, `.load(text)`); returns the table's summary, plain JSON (it travels
        over the child's pipe, where a value json cannot write would kill the reply)."""
        import json
        summary = self._run(engine)
        return json.loads(json.dumps(summary, default=str))

    def _run(self, engine: Any) -> dict:
        t0 = time.perf_counter()
        started = time.monotonic()
        ev: Evaluate = lambda expr: [str(x) for x in engine.run("!" + expr)]
        if ev(f"(match &self ({ANSWER_HEAD} $a $c $g $t $n $i) found)"):
            raise PlnChainerError(f"the knowledge base already holds {ANSWER_HEAD} atoms; only the chainer writes them")
        # the links come off the space with the knowledge base's own functions: no library needed, and a knowledge base
        # with no link at all (a stack that never chains) costs no library load
        links, skipped, nodes = collect_links(ev, self.blocks, caller_blocks=self.caller_blocks,
                                              drugage_fold=self.settings.drugage_fold,
                                              zero_rows=self.settings.drugage_zero_rows)
        refusal = caller_skips_message(skipped)
        if refusal:
            raise PlnInputError(refusal)
        t1 = time.perf_counter()
        # which sources need a derivation: the ones the program names, or every one when that cannot be said
        demand = demand_of(self.blocks, self.expressions) if self.settings.demand else None
        have = {l.src for l in links}
        sources = sorted(have) if demand is None else sorted(demand & have)
        budget = request_budget_s() if self.deadline_s is None else float(self.deadline_s)
        plans = plan_sources(links, self.settings, sources, budget_s=budget) if sources else []
        t2 = time.perf_counter()
        if not sources:
            table = ChainTable(links, nodes, 0.0, 0.0, self.settings, skipped=list(skipped))
            if links or skipped:
                engine.load(render_facts(table))
            self.table = table
            summary = table.summary()
            summary["demand"] = None if demand is None else sorted(demand)
            summary["timing_s"] = {"library": 0.0, "links": round(t1 - t0, 4), "plan": round(t2 - t1, 4), "derive": 0.0,
                                   "facts": 0.0, "total": round(time.perf_counter() - t0, 4)}
            return summary
        engine.load(library_text(self.settings.ruleset))
        leak = probe_library(ev)
        discount = read_discount(ev)
        if self.settings.ruleset == "full":
            engine.load(base_rate_text(links))
        t3 = time.perf_counter()
        deadline_at = started + 0.85 * budget if budget > 0 else None
        table = compute_table(ev, links, leak=leak, discount=discount, settings=self.settings, sources=sources, nodes=nodes,
                              skipped=skipped, plans=plans, deadline_at=deadline_at)
        t4 = time.perf_counter()
        engine.load(render_facts(table))
        t5 = time.perf_counter()
        self.table = table
        summary = table.summary()
        summary["demand"] = None if demand is None else sorted(demand)
        summary["modelled_s"] = round(sum(p.seconds for p in plans), 4)
        summary["timing_s"] = {"library": round(t3 - t2, 4), "links": round(t1 - t0, 4), "plan": round(t2 - t1, 4),
                               "derive": round(t4 - t3, 4), "facts": round(t5 - t4, 4), "total": round(t5 - t0, 4)}
        return summary
