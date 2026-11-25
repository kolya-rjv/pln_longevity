"""The reasoning trace of a request under PLN_CHAINER=pln: which conclusions an answer read, and how each was derived.

What is in it
-------------
One entry per conclusion `infer`/`explain` returned while the program ran -- directly, or inside a ranking, a diagnosis or any
other function that calls them (the PeTTa child records the calls, `_PettaEngine.watch_calls`). Each entry carries:

* the route (nodes and evidence ids, in the order of the derivation);
* the steps. A step is one link, and holds the rule applications `PLN.Derive` made along it: Modus Ponens on (the state so far, the
  implication), and Negation-elimination when the link is Neg. The truth values are the ones `PLN.Derive` left in its belief
  list, for both runs (the intervention at level 1 and at level 0). They are read, not reconstructed: a step whose belief is not in
  the list says so;
* a check: every rule application is recomputed with the library's own `Truth_*` function from the traced premises and compared
  with the belief (`checked` true | false, or None when it could not be evaluated);
* the read-out, which is NOT a PLN rule: how this application turns the two levels into the reported (signed (stv s c)) --
  `|s1 - s0| / (1 - leak)^hops`, the sign of the Neg parity, and the confidence times the discount (PLN_CHAIN_CONFIDENCE);
* the evidence behind each link: file:line, tier, hazard-ratio record, DrugAge rows and papers.

What is not in it: anything after `infer`. The scoring of a ranking, the fusion of a diagnosis, the net of a DrugAge compound and a
plain `match` are the knowledge base's (or Python's) own arithmetic, and the trace does not claim to explain them. It also has no
search order (which task was expanded when).

Levels (`limit_trace`, `render_trace`): 0 off, 1 the route, 2 + steps and read-out, 3 + evidence.

A trace never changes or fails an answer: whatever goes wrong while it is built is reported in its `error`.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, Optional, Sequence

from core import pln_chainer as pc

#: conclusions a trace describes; the rest are counted in `omitted`
MAX_CONCLUSIONS = 50
#: pairs asked for and not connected that a trace lists (the rest are counted)
MAX_NO_ROUTE = 20
#: conclusions shown in a chat answer (the API and the log hold all of them)
MAX_RENDERED = 8
TRACE_LEVELS = (0, 1, 2, 3)
LEVEL_LABELS = (("Off", 0), ("Path", 1), ("Steps", 2), ("Steps + evidence", 3))
#: how far a recomputed rule application may be from the belief (the numbers travel as decimal text)
CHECK_TOL = 1e-12

SCOPE_NOTE = ("Covers the conclusions read through infer/explain. Scoring, sorting and fusion done with them afterwards, and plain "
              "lookups, are not PLN inference and are not traced.")


# ── building (in the PeTTa child, after the program ran) ─────────────────────────

def build_trace(table: Any, calls: Sequence[dict], ev_many: Callable[[Sequence[str]], Sequence[Sequence[str]]],
                overflow: bool = False) -> dict:
    """The trace of one run. `calls` are {"source", "target", "answered", "expressions"}; `ev_many` evaluates expressions on
    the engine (for the check), one result list each."""
    conclusions: list[dict] = []
    no_route: list[dict] = []
    outside: list[dict] = []
    omitted = 0
    for call in calls:
        routes = [r for r in table.routes.get(call["source"], ()) if r.target == call["target"]]
        where = {"source": call["source"], "target": call["target"], "expressions": sorted(i + 1 for i in call["expressions"])}
        if not routes:
            # answered, yet by nothing the chainer derived: an `infer`/`explain` equation of the caller's own
            (outside if call["answered"] else no_route).append(where)
            continue
        if not call["answered"]:
            continue
        for r in routes:
            if len(conclusions) >= MAX_CONCLUSIONS:
                omitted += 1
                continue
            conclusions.append(_conclusion(table, r, where))
    _check(conclusions, ev_many)
    applications = [a for c in conclusions for s in c["steps"] for a in s["rules"]]
    st = table.settings
    return {
        "chainer": "pln",
        "library": "patham9's PLN (vendor/pln_fork)",
        "ruleset": st.ruleset,
        "mapping": st.mapping,
        "chain_confidence": st.chain_confidence,
        "leak": table.leak,
        "discount": table.discount,
        "complete": table.complete,
        "scope": SCOPE_NOTE,
        "conclusions": conclusions,
        "omitted": omitted,
        "no_route": no_route[:MAX_NO_ROUTE],
        "no_route_omitted": max(0, len(no_route) - MAX_NO_ROUTE),
        "outside_chainer": outside[:MAX_NO_ROUTE],
        "calls_overflow": bool(overflow),
        "check": {"applications": len(applications),
                  "reproduced": sum(1 for a in applications if a["checked"] is True),
                  "mismatched": sum(1 for a in applications if a["checked"] is False)},
    }


def _beliefs(rows: Iterable[tuple]) -> dict:
    out: dict = {}
    for term, s, c, ids in rows:
        out.setdefault((term, ids), (s, c))
    return out


def _conclusion(table: Any, route: Any, where: dict) -> dict:
    enc = table.encoding
    links = pc._trail(enc, route.source, (0, *route.stamp), end=route.target, end_terminal=(route.end == "End"), simple=True) or []
    b1, b0 = table.beliefs.get(route.source, ((), ()))
    levels = (_beliefs(b1), _beliefs(b0))
    steps: list[dict] = []
    used: list[int] = [0]
    sign = "Pos"
    for hop, link in enumerate(links, start=1):
        premise_term, premise_ids = ("State", enc.opaque[link.src]), tuple(sorted(used))
        used.append(link.id)
        ids = tuple(sorted(used))
        end = ("End" if link.terminal else "State", enc.opaque[link.dst])
        target = ("Not", end) if link.sign == "Neg" else end
        sign = ("Neg" if sign == "Pos" else "Pos") if link.sign == "Neg" else sign
        rules = [_application("ModusPonens", [(premise_term, premise_ids)], (target, ids), levels, enc, link)]
        if link.sign == "Neg":
            rules.append(_application("NegationElimination", [(target, ids)], (end, ids), levels, enc, None))
        steps.append({"hop": hop, "from": link.src, "to": link.dst, "evidence_id": link.id,
                      "link": {"sign": link.sign, "strength": link.strength, "confidence": link.confidence},
                      "rules": rules, "so_far": _so_far(levels, (end, ids), sign, hop, table),
                      "evidence": _evidence(link)})
    product = 1.0
    for link in links:
        product *= link.strength
    factor = 1.0 if table.settings.chain_confidence == "native" else table.discount ** (route.hops - 1)
    divisor = (1.0 - table.leak) ** route.hops if table.settings.mapping == "affine" else 1.0
    return {
        **where,
        "sign": route.sign, "strength": route.strength, "confidence": route.confidence,
        "path": list(route.path), "evidence_ids": [link.id for link in links],
        "steps": steps,
        "readout": {
            "note": "this application's read-out, not a PLN rule",
            "level1": route.level1, "level0": route.level0, "hops": route.hops,
            "divisor": divisor, "library_confidence": route.native_confidence, "confidence_factor": factor,
            "product_of_link_strengths": product if links else None,
            "strength_equals_product": bool(links) and abs(product - route.strength) <= 1e-9,
        },
    }


def _so_far(levels: tuple, key: tuple, sign: str, hop: int, table: Any) -> Optional[dict]:
    """The read-out after this step, as if the chain ended here."""
    one, zero = levels[0].get(key), levels[1].get(key)
    if one is None or zero is None:
        return None
    st = table.settings
    divisor = (1.0 - table.leak) ** hop if st.mapping == "affine" else 1.0
    factor = 1.0 if st.chain_confidence == "native" else table.discount ** (hop - 1)
    return {"sign": sign, "strength": min(1.0, abs(one[0] - zero[0]) / divisor), "confidence": one[1] * factor}


def _application(rule: str, premises: list, conclusion: tuple, levels: tuple, enc: Any, link: Any) -> dict:
    """One rule application as PLN.Derive left it: premises and conclusion with their truth values at level 1 and 0."""
    prem = []
    for key in premises:
        prem.append({"term": _term_text(key[0], enc), "level1": _tv(levels[0].get(key)), "level0": _tv(levels[1].get(key))})
    if link is not None:
        tv = [enc.q(link), link.confidence]
        target = conclusion[0]
        prem.append({"term": f"(Implication (State {link.src}) {_term_text(target, enc)})", "level1": tv, "level0": list(tv)})
    out = {"rule": rule, "premises": prem,
           "conclusion": {"term": _term_text(conclusion[0], enc), "level1": _tv(levels[0].get(conclusion)),
                          "level0": _tv(levels[1].get(conclusion)), "stamp": list(conclusion[1])},
           "checked": None}
    if any(p[lv] is None for p in [*prem, out["conclusion"]] for lv in ("level1", "level0")):
        out["missing"] = "a truth value of this step is not in PLN.Derive's belief list"
    return out


def _tv(value: Optional[tuple]) -> Optional[list]:
    return None if value is None else [value[0], value[1]]


def _term_text(term: Any, enc: Any) -> str:
    """A library term with the opaque node name replaced by the knowledge base's own."""
    if isinstance(term, tuple):
        return "(" + " ".join(_term_text(t, enc) for t in term) + ")"
    return enc.key_of.get(term, term)


def _evidence(link: Any) -> dict:
    out: dict = {"kind": link.kind, "source": link.source or None, "tier": link.tier}
    for key in ("record", "rows", "papers"):
        if link.detail.get(key):
            value = link.detail[key]
            out[key] = list(value) if isinstance(value, (tuple, list)) else value
    return out


def _check(conclusions: list[dict], ev_many: Callable[[Sequence[str]], Sequence[Sequence[str]]]) -> None:
    """Recompute every rule application with the library's own function, both levels, each distinct call once."""
    def call_of(app: dict, level: str) -> Optional[str]:
        tvs = [p[level] for p in app["premises"]]
        if any(tv is None for tv in tvs):
            return None
        args = " ".join(f"(stv {pc.fnum(s)} {pc.fnum(c)})" for s, c in tvs)
        return f"({'Truth_ModusPonens' if app['rule'] == 'ModusPonens' else 'Truth_Negation'} {args})"

    apps = [a for c in conclusions for s in c["steps"] for a in s["rules"] if "missing" not in a]
    wanted = sorted({call for a in apps for lv in ("level1", "level0") if (call := call_of(a, lv))})
    if not wanted:
        return
    answers = ev_many(wanted)
    got: dict[str, Optional[tuple]] = {}
    for expr, out in zip(wanted, answers):
        try:
            got[expr] = pc._stv_of(out[0]) if len(out) == 1 else None
        except (pc.PlnChainerError, ValueError):
            got[expr] = None
    for a in apps:
        ok: Optional[bool] = True
        for lv in ("level1", "level0"):
            value, belief = got.get(call_of(a, lv) or ""), a["conclusion"][lv]
            if value is None:
                ok = None
                break
            if abs(value[0] - belief[0]) > CHECK_TOL or abs(value[1] - belief[1]) > CHECK_TOL:
                ok = False
                break
        a["checked"] = ok


# ── combining and cutting ────────────────────────────────────────────────────────

def merge_traces(traces: Sequence[Optional[dict]], positions: Optional[Sequence[Sequence[int]]] = None) -> Optional[dict]:
    """One trace for a program that ran as several parts in different spaces (`positions[k][j]` = the original index of part k's
    j-th expression). Evidence ids belong to a part, so each entry names its part."""
    have = [(k, t) for k, t in enumerate(traces) if t]
    if not have:
        return None
    out = copy.deepcopy(have[0][1])
    for key in ("conclusions", "no_route", "outside_chainer"):
        out[key] = []
    for key in ("omitted", "no_route_omitted"):
        out[key] = 0
    out["check"] = {"applications": 0, "reproduced": 0, "mismatched": 0}
    errors = []
    for k, trace in have:
        where = positions[k] if positions is not None and k < len(positions) else None
        for key in ("conclusions", "no_route", "outside_chainer"):
            for entry in copy.deepcopy(trace.get(key) or []):
                entry["part"] = k + 1
                if where is not None:
                    entry["expressions"] = sorted(where[i - 1] + 1 if 0 < i <= len(where) else i for i in entry["expressions"])
                out[key].append(entry)
        for key in ("omitted", "no_route_omitted"):
            out[key] += trace.get(key) or 0
        for key in out["check"]:
            out["check"][key] += (trace.get("check") or {}).get(key, 0)
        out["complete"] = bool(out.get("complete", True)) and bool(trace.get("complete", True))
        out["calls_overflow"] = bool(out.get("calls_overflow")) or bool(trace.get("calls_overflow"))
        if trace.get("error"):
            errors.append(trace["error"])
    out.pop("error", None)
    if errors:
        out["error"] = "; ".join(errors)
    return out


def limit_trace(trace: Optional[dict], level: int) -> Optional[dict]:
    """`trace` as a caller who asked for `level` gets it (None at 0)."""
    if not trace or level <= 0:
        return None
    out = copy.deepcopy(trace)
    out["level"] = min(int(level), 3)
    for c in out.get("conclusions", ()):
        if level < 2:
            c.pop("steps", None)
            c.pop("readout", None)
        elif level < 3:
            for s in c["steps"]:
                s.pop("evidence", None)
    return out


def unavailable_reason() -> str:
    """Why a run carries no trace, for the reader of the answer."""
    import config
    if getattr(config, "PLN_CHAINER", "legacy") != "pln":
        return ("not recorded: this server answers with the legacy chainer. It is recorded under "
                "`PLN_RUNTIME=petta PLN_CHAINER=pln`.")
    if not getattr(config, "PLN_TRACE", False):
        return "not recorded: `PLN_TRACE` is off on this server."
    return "none: this program ran no PLN derivation."


# ── rendering (the chat answer) ──────────────────────────────────────────────────

def _num(x: Optional[float]) -> str:
    return "?" if x is None else f"{x:.4g}"


def _tvs(tv: Optional[Sequence[float]]) -> str:
    return "?" if tv is None else f"({_num(tv[0])}, {_num(tv[1])})"


def _code(text: Any) -> str:
    """Text of the knowledge base (or a caller's) as inline code on one line, which cannot start markup of its own."""
    clean = " ".join(str(text).replace("`", "'").split())
    return f"`{clean[:157] + '...' if len(clean) > 160 else clean}`"


def _evidence_text(e: dict) -> str:
    bits = [_code(e["source"])] if e.get("source") and not e.get("rows") else []
    if e.get("record"):
        bits.append(_code(e["record"]))
    if e.get("tier"):
        bits.append(str(e["tier"]))
    if e.get("rows"):
        bits.append(f"{len(e['rows'])} DrugAge row(s), {len(e.get('papers') or ())} paper(s): {_code(', '.join(map(str, e.get('papers') or ())))}")
    return ", ".join(bits) or str(e.get("kind") or "")


_RULE_TEXT = {"ModusPonens": "Modus Ponens", "NegationElimination": "Negation elimination"}


def _render_conclusion(c: dict, level: int) -> list[str]:
    part = f" (part {c['part']})" if c.get("part") else ""
    lines = [f"**{_code(c['source'])} → {_code(c['target'])}**{part}: {_code(c['sign'])} "
             f"stv({_num(c['strength'])}, {_num(c['confidence'])})",
             "", " → ".join(_code(n) for n in c["path"])]
    if level < 2:
        return lines
    lines.append("")
    for s in c["steps"]:
        lines.append(f"{s['hop']}. {_code(s['from'])} → {_code(s['to'])} — link {s['link']['sign']} "
                     f"{_tvs([s['link']['strength'], s['link']['confidence']])}")
        for a in s["rules"]:
            con = a["conclusion"]
            check = {True: "✓", False: "✗ **differs from the library's formula**", None: "(not checked)"}[a["checked"]]
            lines.append(f"    - {_RULE_TEXT.get(a['rule'], a['rule'])} gives {_tvs(con['level1'])} with the intervention, "
                         f"{_tvs(con['level0'])} without {check}")
        so_far = s.get("so_far")
        lines.append("    - chain so far: " + (f"{so_far['sign']} {_tvs([so_far['strength'], so_far['confidence']])}" if so_far else "?"))
        if level >= 3:
            lines.append(f"    - evidence: {_evidence_text(s.get('evidence') or {})}")
    r = c["readout"]
    diff = f"abs({_num(r['level1'])} − {_num(r['level0'])})"
    strength = diff if r["divisor"] == 1.0 else f"{diff} / {_num(r['divisor'])}"
    lines += ["", f"Read-out (this app's, not a PLN rule): strength = {strength} = {_num(c['strength'])}; "
                  f"confidence = {_num(r['library_confidence'])} × {_num(r['confidence_factor'])} = {_num(c['confidence'])}"]
    return lines


def render_trace(trace: Optional[dict], level: int, unavailable: str = "") -> str:
    """Markdown for the chat, or "" at level 0. `unavailable` is said when a trace was asked for and there is none."""
    if level <= 0:
        return ""
    if not trace:
        return f"**PLN reasoning trace:** {unavailable}" if unavailable else ""
    if trace.get("error") and not trace.get("conclusions"):
        return f"**PLN reasoning trace:** not available ({_code(trace['error'])}). The answer above is unaffected."
    conclusions = trace.get("conclusions") or []
    total = len(conclusions) + (trace.get("omitted") or 0)
    check = trace.get("check") or {}
    head = f"**PLN reasoning trace** — {total} conclusion(s) read"
    if level >= 2 and check.get("applications"):
        head += f"; {check['reproduced']} of {check['applications']} rule applications reproduced with the library's own formulas"
        if check.get("mismatched"):
            head += f", **{check['mismatched']} differ**"
    parts = [head + ("." if trace.get("complete", True) else "; the derivation was cut short (partial).")]
    if not conclusions and not trace.get("no_route") and not trace.get("outside_chainer"):
        parts.append("This program made no PLN inference: it called neither `infer` nor `explain`, so its result is a lookup or "
                     "plain arithmetic.")
    for c in conclusions[:MAX_RENDERED]:
        parts.append("\n".join(_render_conclusion(c, level)))
    if total > MAX_RENDERED:
        parts.append(f"{total - MAX_RENDERED} more conclusion(s) not shown here.")
    if trace.get("no_route"):
        shown = ", ".join(f"{_code(x['source'])} → {_code(x['target'])}" for x in trace["no_route"])
        more = f" and {trace['no_route_omitted']} more" if trace.get("no_route_omitted") else ""
        parts.append(f"Asked and not connected (no route): {shown}{more}.")
    if trace.get("outside_chainer"):
        shown = ", ".join(f"{_code(x['source'])} → {_code(x['target'])}" for x in trace["outside_chainer"])
        parts.append(f"Answered by a definition of the request's own, not by the chainer: {shown}.")
    if trace.get("calls_overflow"):
        parts.append("The program made more inference calls than the trace records; the list is incomplete.")
    if trace.get("error"):
        parts.append(f"Part of the trace could not be built ({_code(trace['error'])}).")
    if conclusions:
        parts.append(f"_{trace.get('scope', SCOPE_NOTE)}_")
    return "\n\n".join(parts)
