"""An independent oracle for the paper-level DrugAge fold. Standard library only.

It is written from the SPEC (docs/petta_migration/fork_design/b2/notes.md and the module docstring of
pln_chat/ontology/drugage_fold.py), not from the module: it imports nothing from the repo, does its own
grouping, its own representative-row choice and its own net, and revises with a closed form of the
fork's Truth_Revision instead of the pairwise port. `tests/test_drugage_fold.py` loads it by path and
compares it with the module on every compound of the build, and with the fork simulator
(docs/petta_migration/fork_design/sim/fork_sim.py) on the folds.

A row is a plain dict:
    row_id, compound, clade ("Vertebrate" | "Invertebrate" | "Fungi" | "Protozoa" | None),
    itp (bool), sig ("Significant" | "NotSignificant" | "Unreported"), pct (float | None), pmid (str | None)

A change of exactly 0.0 is a reported null (strength 0). `fold(rows, zero_rows=...)` places it by the spec's convention, written
here in its own words: "both" = a null is a piece of evidence in each direction's pool (default); "neither" = it is in none
(a compound with nothing else keeps it, on both sides); "protective" = on the protective (Neg) side only, where the sign
convention (no zero) files it.

The constants are the calibration of drugage_calibration.metta / epistemic_calibration.metta as of this
writing; the test asserts they equal what `load_knobs()` reads, so a tuned constant fails loudly there.
"""
from __future__ import annotations

HALFSAT = 20.0
GATE = {"Significant": 1.0, "Unreported": 0.6, "NotSignificant": 0.4}
TIER = {"ITP_Positive": 0.9, "ITP_Negative": 0.9, "AnimalStudies_Single": 0.5, "InVitro": 0.35, "TraditionalUse": 0.2}
CLADE_TIER = {"Vertebrate": "AnimalStudies_Single", "Invertebrate": "InVitro", "Fungi": "TraditionalUse",
              "Protozoa": "TraditionalUse"}
CHAIN = 0.9
CAP = 0.999999


def lift(r):
    """(sign, strength, native confidence) of one row, or None when it reports no change."""
    if r["pct"] is None:
        return None
    m = abs(r["pct"])
    strength = m / (m + HALFSAT)
    if r["itp"]:
        native = TIER["ITP_Positive" if r["sig"] == "Significant" else "ITP_Negative"] * 1.0
    else:
        tier = TIER[CLADE_TIER[r["clade"]]] if r["clade"] else TIER["InVitro"]
        native = tier * GATE[r["sig"]]
    return ("Pos" if r["pct"] < 0 else "Neg", strength, native)


def evidence_key(r):
    """Higher is better: ITP first, then significance (the representative-row policy, rules 2 and 3)."""
    rank = 3 if r["itp"] else {"Significant": 2, "Unreported": 1, "NotSignificant": 0}[r["sig"]]
    return (rank, {"Significant": 2, "Unreported": 1, "NotSignificant": 0}[r["sig"]])


def representative(rows):
    best = max(evidence_key(r) for r in rows)
    top = sorted((r for r in rows if evidence_key(r) == best), key=lambda r: (r["pct"], r["row_id"]))
    return top[(len(top) - 1) // 2]            # the lower median (rule 4)


def revise_all(tvs):
    """Closed form of repeated Truth_Revision: weights c/(1-c) add, strengths average by weight, c = w/(w+1)."""
    if len(tvs) == 1:
        return tvs[0]
    ws = [min(c, CAP) / (1 - min(c, CAP)) for _, c in tvs]
    w = sum(ws)
    f = sum(wi * s for wi, (s, _) in zip(ws, tvs)) / w
    return (min(1.0, f), w / (w + 1))


def fold(rows, zero_rows="both"):
    """{'Neg': item, 'Pos': item, 'net': float} for the rows of ONE compound.

    item = dict(strength, native, confidence, papers (sorted evidence units), members (row ids)).
    """
    assert zero_rows in ("both", "neither", "protective")
    groups = {}
    nulls = []
    for r in rows:
        lifted = lift(r)
        if lifted is None:
            continue
        unit = r["pmid"] if r["pmid"] else "row:" + r["row_id"]
        if lifted[1] == 0.0 and zero_rows != "protective":
            nulls.append((unit, r))
            continue
        groups.setdefault((lifted[0], unit), []).append(r)
    if nulls and (zero_rows == "both" or not groups):
        for unit, r in nulls:
            for sign in ("Neg", "Pos"):
                groups.setdefault((sign, unit), []).append(r)
    out = {}
    for sign in ("Neg", "Pos"):
        units = sorted(u for (s, u) in groups if s == sign)
        if not units:
            continue
        reps = [representative(groups[(sign, u)]) for u in units]
        lifted = [lift(r) for r in reps]
        s, c = revise_all([(l[1], l[2]) for l in lifted])
        out[sign] = {"strength": s, "native": c, "confidence": c * CHAIN, "papers": units,
                     "members": [r["row_id"] for r in reps],
                     "rows": sum(len(groups[(sign, u)]) for u in units)}
    if zero_rows != "protective":
        # a direction whose pool pooled to strength exactly 0 (only nulls in it) is not an item; if every one is, the protective one stays
        real = {k: v for k, v in out.items() if v["strength"] != 0.0}
        if len(real) < len(out):
            out = real or {k: v for k, v in list(out.items())[:1]}
    net = 0.0
    for sign, item in out.items():
        net += (1.0 if sign == "Neg" else -1.0) * item["strength"] * item["confidence"]
    out["net"] = net
    return out
