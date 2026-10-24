"""Measure the paper-level DrugAge fold (task B2). Every number in b2/notes.md comes from this script.

    python docs/petta_migration/fork_design/b2/measure_fold.py [--out DIR]

It regenerates the ETL from data/drugage/dataset.zip into a temporary directory (the repo's build/ is not
read), then prints and writes (DIR/measure.json):

* the Ethanol / Resveratrol / Metformin / Rapamycin table: legacy representative row against the paper-level
  fold, and against the fold with every ROW as its own evidence, all under the one NET formula;
* the rank of Ethanol with the ETL build absent (the committed 201-row sample) and present (3,423 rows);
* the whole-build ranking summary: how many of the top 20 move, rank correlation, label changes;
* the size of the data (papers per compound, mixed-direction compounds) and the cost of a whole-build fold.

Standard library plus the repo (pandas for the ETL). No model, no network, no runtime.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "pln_chat"))

from ontology.drugage_fold import (DEFAULT_ZERO_ROWS, ZERO_ROW_CONVENTIONS, fold_compound, fold_rows, fold_truth_values,  # noqa: E402
                                   lift_row, top_compounds)
from ontology.drugage_scoring import load_knobs  # noqa: E402
from ontology.drugage_selector import SAMPLE_DRUGAGE, load_rows  # noqa: E402

COMPOUNDS = ["Resveratrol", "Metformin", "Rapamycin", "Ethanol"]
#: the null convention of every fold figure in this run (the module default unless --zero-rows says otherwise); the three
#: conventions side by side are in `null_conventions`
ZR = DEFAULT_ZERO_ROWS
#: the earlier design's (Design 3) paper-level figures, for the comparison column only: they came from another
#: within-paper pick (most confident row, then lower median STRENGTH) and are not reproduced by this module.
D3_EARLIER = {"Resveratrol": -0.0395, "Metformin": -0.0403, "Rapamycin": 0.3167}


def build_rows():
    spec = importlib.util.spec_from_file_location("drugage_etl_b2", REPO / "drugage_etl.py")
    etl = importlib.util.module_from_spec(spec)
    sys.modules["drugage_etl_b2"] = etl
    spec.loader.exec_module(etl)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "drugage.csv"
        with zipfile.ZipFile(REPO / "data" / "drugage" / "dataset.zip") as z:
            csv_path.write_bytes(z.read("drugage.csv"))
        out = Path(tmp) / "drugage_etl.metta"
        etl.generate(csv_path, out)
        return load_rows(out)


def item_row(it):
    return None if it is None else {
        "strength": it.strength, "native_confidence": it.native_confidence, "confidence": it.confidence,
        "papers": len(it.units), "rows": it.rows_covered, "score": it.score,
    }


def pooled_revision_net(fold):
    """The REJECTED alternative, for the comparison column only: Truth_Revision across the two directions on SIGNED
    strengths (+s for Neg, -s for Pos; weights c/(1-c) on the native confidences), score = signed strength x confidence
    x 0.9. It applies the revision formula outside its [0, 1] domain, and raises confidence when results conflict."""
    items = [i for i in fold.items]
    ws = [i.native_confidence / (1.0 - i.native_confidence) for i in items]
    total = sum(ws)
    x = sum(w * (i.strength if i.sign == "Neg" else -i.strength) for w, i in zip(ws, items)) / total
    c = total / (total + 1.0) * 0.9
    return {"signed_strength": x, "confidence": c, "score": x * c}


def table(rows):
    ranks = {}
    for label, kw in (("legacy", {"fold": False}), ("paper", {"zero_rows": ZR}), ("row", {"evidence_unit": "row", "zero_rows": ZR})):
        top = top_compounds(rows, n=10 ** 6, direction="any", **kw)
        ranks[label] = {e.compound: i for i, e in enumerate(top.entries, 1)}
    by = {}
    for r in rows:
        by.setdefault(r.compound, []).append(r)
    out = {}
    for c in COMPOUNDS:
        rs = by[c]
        legacy = fold_compound(rs, fold=False)
        paper = fold_compound(rs, zero_rows=ZR)
        per_row = fold_compound(rs, evidence_unit="row", zero_rows=ZR)
        out[c] = {
            "rows": len(rs), "papers": len({r.pmid for r in rs}),
            "legacy": {"score": legacy.net.score, "strength": legacy.net.strength, "confidence": legacy.net.confidence,
                       "sign": legacy.net.sign, "direction": legacy.net.direction, "row": legacy.net.row.row_id},
            "fold": {"Neg": item_row(paper.neg), "Pos": item_row(paper.pos), "net": paper.net.score,
                     "sign": paper.net.sign, "direction": paper.net.direction, "strength": paper.net.strength,
                     "confidence": paper.net.confidence, "lead_row": paper.net.row.row_id},
            "row_as_unit": {"Neg": item_row(per_row.neg), "Pos": item_row(per_row.pos), "net": per_row.net.score,
                            "direction": per_row.net.direction},
            "d3_earlier_net": D3_EARLIER.get(c),
            "rank_of_1035": {k: v[c] for k, v in ranks.items()},
            "pooled_revision_net_rejected": pooled_revision_net(paper) if len(paper.items) == 2 else None,
        }
    return out


def null_conventions(rows):
    """The three ways a reported null (a change of exactly 0.0) can count, side by side: the NET of the four named compounds, their
    rank among the 1,035, and over the whole build how many compounds have a null and how many NET signs the convention decides.
    The sign of Resveratrol and Metformin is the convention's (review finding M1)."""
    by = {}
    for r in rows:
        by.setdefault(r.compound, []).append(r)
    sign = lambda x: "harmful" if x < 0 else ("zero" if x == 0 else "protective")
    nets = {z: {c: f.net for c, f in fold_rows(rows, zero_rows=z).items() if f.net} for z in ZERO_ROW_CONVENTIONS}
    ranks = {}
    for z in ZERO_ROW_CONVENTIONS:
        top = top_compounds(rows, n=10 ** 6, direction="any", zero_rows=z)
        ranks[z] = {e.compound: i for i, e in enumerate(top.entries, 1)}
    named = {}
    for c in COMPOUNDS:
        rs = by[c]
        named[c] = {"rows": len(rs), "null_rows": sum(1 for r in rs if r.avg_change == 0),
                    **{z: {"net": nets[z][c].score, "direction": nets[z][c].direction, "rank_of_1035": ranks[z][c]}
                       for z in ZERO_ROW_CONVENTIONS}}
    with_null = [c for c, rs in by.items() if c in nets["both"] and any(r.avg_change == 0 for r in rs)]
    flips = {}
    for a in ZERO_ROW_CONVENTIONS:
        for b in ZERO_ROW_CONVENTIONS:
            if a < b:
                flips[f"{a} vs {b}"] = sorted(c for c in nets[a] if sign(nets[a][c].score) != sign(nets[b][c].score))
    top20 = {z: [e.compound for e in top_compounds(rows, n=20, direction="protective", zero_rows=z).entries] for z in ZERO_ROW_CONVENTIONS}
    return {"named": named, "scorable": len(nets["both"]), "compounds_with_a_null_row": len(with_null),
            "sign_flips": {k: {"count": len(v), "compounds": v} for k, v in flips.items()},
            "top20_overlap": {f"{a} vs {b}": len(set(top20[a]) & set(top20[b]))
                              for a in ZERO_ROW_CONVENTIONS for b in ZERO_ROW_CONVENTIONS if a < b},
            "top20": top20, "default": DEFAULT_ZERO_ROWS}


def d3_reproduction(rows):
    """The earlier design's within-paper pick (the most confident row, ties: the lower median STRENGTH) fed through
    this module's fold and net. If it gives the earlier figures (-0.0395, -0.0403, +0.3167), the earlier numbers and
    these differ ONLY in the within-paper pick, not in the aggregator."""
    out = {}
    for c in ("Resveratrol", "Metformin", "Rapamycin"):
        groups = {}
        for lr in (lift_row(r) for r in rows if r.compound == c and r.scorable):
            groups.setdefault((lr.sign, lr.unit), []).append(lr)
        net = 0.0
        for sign in ("Neg", "Pos"):
            reps = []
            for (sg, _unit), g in sorted(groups.items()):
                if sg != sign:
                    continue
                best = max(x.native_confidence for x in g)
                tied = sorted((x for x in g if x.native_confidence == best), key=lambda x: (x.strength, x.row.row_id))
                reps.append(tied[(len(tied) - 1) // 2])
            if reps:
                s_, c_ = fold_truth_values([(m.strength, m.native_confidence) for m in reps])
                net += (1 if sign == "Neg" else -1) * s_ * c_ * 0.9
        out[c] = net
    return out


def rank_of(top, compound):
    for i, e in enumerate(top.entries, 1):
        if e.compound == compound:
            return i
    return None


def ethanol(rows_by_source):
    out = {}
    for label, rows in rows_by_source.items():
        res = {}
        for direction in ("any", "protective"):
            for name, kw in (("legacy", {"fold": False}), ("paper", {"zero_rows": ZR}), ("row", {"evidence_unit": "row", "zero_rows": ZR})):
                top = top_compounds(rows, n=10 ** 6, direction=direction, **kw)
                res[f"{name}/{direction}"] = {"rank": rank_of(top, "Ethanol"), "of": len(top.entries),
                                              "first": top.entries[0].compound, "score": next(
                                                  (e.score for e in top.entries if e.compound == "Ethanol"), None)}
        out[label] = {"rows": len(rows), "compounds": len({r.compound for r in rows}), **res}
    return out


def spearman(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db)


def ranking_summary(rows):
    out = {}
    for direction in ("protective", "any"):
        legacy = top_compounds(rows, n=10 ** 6, direction=direction, fold=False)
        paper = top_compounds(rows, n=10 ** 6, direction=direction, zero_rows=ZR)
        lnames = [e.compound for e in legacy.entries]
        pnames = [e.compound for e in paper.entries]
        lrank = {c: i for i, c in enumerate(lnames, 1)}
        prank = {c: i for i, c in enumerate(pnames, 1)}
        common = [c for c in lnames if c in prank]
        res = {
            "universe_legacy": len(lnames), "universe_fold": len(pnames), "in_both": len(common),
            "top_overlap": {str(k): len(set(lnames[:k]) & set(pnames[:k])) for k in (10, 20, 50)},
            "same_position_top20": sum(a == b for a, b in zip(lnames[:20], pnames[:20])),
            "spearman_in_both": spearman([lrank[c] for c in common], [prank[c] for c in common]),
            "moved_more_than_100": sum(abs(lrank[c] - prank[c]) > 100 for c in common),
            "legacy_top20": [(i, e.compound, e.score, prank.get(e.compound)) for i, e in enumerate(legacy.entries[:20], 1)],
            "fold_top20": [(i, e.compound, e.score, e.strength, e.confidence, e.papers_neg, e.papers_pos,
                            e.rows_neg, e.rows_pos, lrank.get(e.compound)) for i, e in enumerate(paper.entries[:20], 1)],
        }
        out[direction] = res
    legacy_nets = {c: f.net for c, f in fold_rows(rows, fold=False).items() if f.net}
    fold_nets = {c: f.net for c, f in fold_rows(rows, zero_rows=ZR).items() if f.net}
    changed = {}
    for c, a in legacy_nets.items():
        b = fold_nets[c]
        key = f"{a.direction}->{b.direction}"
        changed[key] = changed.get(key, 0) + 1
    out["label_changes"] = dict(sorted(changed.items()))
    out["score_equal_to_legacy"] = sum(legacy_nets[c].score == fold_nets[c].score for c in legacy_nets)
    out["compounds_scored"] = len(legacy_nets)
    return out


def population(rows):
    folds = fold_rows(rows, zero_rows=ZR)
    scorable = [f for f in folds.values() if f.net is not None]
    return {
        "rows": len(rows), "compounds": len(folds), "scorable_compounds": len(scorable),
        "scorable_rows": sum(f.rows_scorable for f in scorable),
        "papers": len({r.pmid for r in rows}),
        "compounds_with_2plus_papers": sum(f.units >= 2 for f in scorable),
        "compounds_with_1_row": sum(f.rows_scorable == 1 for f in scorable),
        "compounds_with_both_directions": sum(len(f.items) == 2 for f in scorable),
        "compounds_with_a_pooled_item": sum(any(i.pooled for i in f.items) for f in scorable),
        "max_rows_for_a_compound": max(f.rows_total for f in folds.values()),
        "compounds_above_the_150_row_cap": sum(f.rows_total > 150 for f in folds.values()),
        "stamp_tokens": sum(len(i.stamp) for f in scorable for i in f.items),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--zero-rows", choices=ZERO_ROW_CONVENTIONS, default=DEFAULT_ZERO_ROWS, help="how a reported null counts in every figure")
    args = ap.parse_args()
    global ZR
    ZR = args.zero_rows
    args.out.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    sample = load_rows(SAMPLE_DRUGAGE)
    knobs = load_knobs()
    t0 = time.perf_counter()
    for _ in range(5):
        fold_rows(rows, zero_rows=ZR)
    fold_ms = (time.perf_counter() - t0) / 5 * 1000

    res = {
        "calibration": {"halfsat": knobs.halfsat, "chain_discount": knobs.chain_discount, "sig_gates": knobs.sig_gates},
        "table": table(rows),
        "ethanol": ethanol({"sample (build absent)": sample, "build (present)": rows}),
        "ranking": ranking_summary(rows),
        "population": population(rows),
        "zero_rows": ZR,
        "null_conventions": null_conventions(rows),
        "d3_within_paper_pick_through_this_fold": d3_reproduction(rows),
        "whole_build_fold_ms": fold_ms,
    }
    (args.out / "measure.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")

    print("== table (legacy representative vs paper fold vs row-as-unit; one NET formula)")
    for c, t in res["table"].items():
        f = t["fold"]
        print(f"{c:12s} rows {t['rows']:3d} papers {t['papers']:2d} | legacy {t['legacy']['score']:+.4f} ({t['legacy']['direction']})"
              f" | fold net {f['net']:+.4f} ({f['direction']}) Neg {f['Neg'] and (round(f['Neg']['strength'], 4), round(f['Neg']['confidence'], 4), f['Neg']['papers'])}"
              f" Pos {f['Pos'] and (round(f['Pos']['strength'], 4), round(f['Pos']['confidence'], 4), f['Pos']['papers'])}"
              f" | row-as-unit net {t['row_as_unit']['net']:+.4f} | D3 {t['d3_earlier_net']} | ranks {t['rank_of_1035']}"
              f" | pooled-revision (rejected) {t['pooled_revision_net_rejected'] and round(t['pooled_revision_net_rejected']['score'], 4)}")
    print("== Ethanol")
    for label, e in res["ethanol"].items():
        print(label, {k: (v["rank"], v["of"]) for k, v in e.items() if isinstance(v, dict)})
    print("== ranking")
    for d in ("protective", "any"):
        r = res["ranking"][d]
        print(d, {k: r[k] for k in ("universe_legacy", "universe_fold", "in_both", "top_overlap", "same_position_top20",
                                    "spearman_in_both", "moved_more_than_100")})
    print("label changes", res["ranking"]["label_changes"], "score unchanged:", res["ranking"]["score_equal_to_legacy"],
          "of", res["ranking"]["compounds_scored"])
    print("population", res["population"])
    nc = res["null_conventions"]
    print("== null conventions (a reported null counts: both sides | neither | protective side only)")
    for c, t in nc["named"].items():
        print(f"{c:12s} nulls {t['null_rows']:2d}/{t['rows']:3d} | " + " | ".join(
            f"{z}: {t[z]['net']:+.4f} {t[z]['direction']} rank {t[z]['rank_of_1035']}" for z in ZERO_ROW_CONVENTIONS))
    print("compounds with a null row", nc["compounds_with_a_null_row"], "of", nc["scorable"], "| sign flips",
          {k: v["count"] for k, v in nc["sign_flips"].items()}, "| top-20 overlap", nc["top20_overlap"])
    print("D3 within-paper pick through this fold", {k: round(v, 4) for k, v in res["d3_within_paper_pick_through_this_fold"].items()})
    print(f"whole-build fold {fold_ms:.1f} ms")


if __name__ == "__main__":
    main()
