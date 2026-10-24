"""Paper-level evidence fold for DrugAge (gate 6b of the PeTTa migration).

Why this exists
---------------
The legacy DrugAge ranking keeps ONE row per compound (the representative-row policy,
`drugage_selector.REPRESENTATIVE_POLICY`). That throws away 169 of Resveratrol's 170 rows, and it
hides the one thing a longevity ranking has to be careful about: how many INDEPENDENT studies stand
behind a compound. The obvious repair, pooling every row with PLN revision, is wrong in the other
direction, because rows are not independent: the 52 Ethanol rows come from ONE paper, and revising
them as 52 pieces of evidence lifts a tier-0.21 confidence to 0.93 and puts Ethanol first in the
whole build. So the evidence unit is the PAPER (the PMID), and a row with no PMID is its own unit.

What the function does
----------------------
`fold_compound(rows)` takes the DrugAge rows of ONE compound and

1. lifts each scorable row exactly as `drugage_calibration.metta` and `drugage_scoring.score_row` do
   (strength |pct| / (|pct| + halfsat); NATIVE confidence = evidence tier x significance gate;
   direction by the sign of the change). `native_confidence x chain_discount` is the legacy per-row
   confidence, bit for bit;
2. groups the lifted rows by (direction, evidence unit) and keeps ONE item per group, chosen by the
   existing representative-row policy (a paper that reports a male and a female result in the same
   direction contributes one item to that direction, not two);
3. folds the items of one direction with a Python PORT of the fork's `Truth_Revision` (patham9/PLN, vendored in
   `vendor/pln_fork`; `revision` below, the same expressions in the same order, held equal to the library's by a test that
   runs both): `w = c/(1-c)`, `f = sum(w f)/sum(w)`, `c = w/(w+1)`, left fold in a canonical order, every confidence
   capped at 0.999999 first because Revision has NO RESULT at c = 1.0;
4. returns at most one `Neg` (protective: lifespan went up, mortality goes down) and one `Pos`
   (harmful) pooled item, each with a stamp of evidence tokens that is unique to that item, and one NET.

THE NET (one formula)
---------------------
    net = sum over the pooled items of  (+1 if the item is Neg else -1) * strength * confidence

`strength * confidence` with a sign is exactly what the ranking already used for a single row
(`RowScore.score`). The net is that same quantity summed over the two directional items, so:

* a compound that has evidence in ONE direction scores what the legacy ranking gives that item (bit
  for bit when the item is a single row; the lift-fidelity tests say so for the whole build);
* evidence for and against subtracts instead of being pooled into one number, so a compound with
  equally strong evidence in both directions scores ~0 and cannot have its confidence raised by
  contradicting studies (pooling the two directions with Revision would do exactly that);
* swapping every direction negates the score.

The net's `sign` is `Neg` when `net >= 0` (the repo's convention has no zero), its `confidence` is
the confidence of the item that carries that sign, and its `strength` is `|net| / confidence` (the
item's own strength when only one direction exists). `direction` says "no_effect" when the net is
exactly 0 or when every row behind it is reported NotSignificant, which is the legacy label rule.

A row that reports a change of exactly 0.0 is a reported NULL (strength 0). The legacy ranking files it under the
protective side only because the sign convention has no zero; pooled that way, a null drags the protective mean toward 0
and leaves the harmful one alone, so every null pushes the net toward "harmful" (23 of Resveratrol's 170 rows are nulls:
-0.046 with the nulls on one side, +0.136 with them on both, +0.032 with them dropped). The repo's own reading is that a
well-run null is high-confidence evidence of NO effect, so it counts in BOTH directions, at strength 0 and its own
confidence (`zero_rows="both"`, the default); a direction whose pool holds nothing BUT nulls pools to strength 0, contributes nothing to
the net and is not emitted as an item (the nulls still dilute the other direction; a compound of only nulls keeps its protective
item, the legacy shape). `"neither"` drops the nulls from the pools; `"protective"` is the earlier
behaviour, kept so the sign of Resveratrol and Metformin can be seen to depend on this convention.

`fold=False` reproduces the legacy per-row results: one item per scorable row, and the net is the
representative row's score. `evidence_unit="row"` makes every row its own unit; it exists to show why
the paper is the unit (Ethanol), never as a setting a request can reach.

Three entry points, one fold underneath (`fold_lifted`):

* `fold_compound(rows)` / `fold_rows(rows)`: DrugAge rows in, `CompoundFold` out. The whole-build ranking
  (`top_compounds`, the counterpart of `core.drugage_router.drugage_top`) and the tests use these.
* `fold_items(items)`: the interface `core.pln_chainer` calls with the items the knowledge base's own
  `drugage-effect` lifted (dicts, lifespan direction). Same pooling, same net. The choice of ONE row inside a
  paper is the repo's representative-row policy when the items carry `change`, `significance` and `itp`, and a
  documented fallback (the most confident row) when they do not; every output item says which.
* `top_compounds(rows, ...)`: the ranking by net, with the filters of `drugage_top`.

Nothing here caps the number of rows: the hyperon-era `drugage_selector.MAX_ROWS = 150` is a
safety constant of the hyperon runtime, not a property of the evidence (Resveratrol has 170 rows).

The module is pure: no I/O besides `load_knobs()` (the same constants the Python twin reads out of
the .metta files), no randomness, no clock. It never imports a runtime.

See docs/petta_migration/fork_design/b2/notes.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional, Sequence

from ontology.drugage_scoring import ScoringKnobs, load_knobs
from ontology.drugage_selector import DrugAgeRow, _norm, select_rows

#: Revision has no result at c = 1.0 (`/safe` refuses a zero divisor); every confidence is capped here
#: before it is revised. The same constant as the decision's `Truth_Revision` probe.
CONFIDENCE_CAP = 0.999999

SIGN_NEG = "Neg"   # protective: the compound LOWERS mortality (it extended lifespan)
SIGN_POS = "Pos"   # harmful: the compound RAISES mortality (it shortened lifespan)

EVIDENCE_UNITS = ("paper", "row")

#: how a row with a change of exactly 0.0 (strength 0: a reported null) counts when the fold pools. See the module docstring.
ZERO_ROW_CONVENTIONS = ("both", "neither", "protective")
DEFAULT_ZERO_ROWS = "both"

#: How ONE row is chosen inside a (paper, direction) group, and for the lead row of a net.
#: PICK_REPRESENTATIVE is the repo's own policy (`drugage_selector.REPRESENTATIVE_POLICY`: ITP, then significance,
#: then the lower median change) and needs the row's ITP flag, significance and change. PICK_CONFIDENT is the
#: fallback for lifted items that carry only (strength, confidence, direction, row, paper): the most confident row,
#: ties to the lower median strength. The two agree whenever a group is one row; they can differ otherwise.
PICK_REPRESENTATIVE = "representative_row_policy"
PICK_CONFIDENT = "most_confident_row"

#: Returned verbatim next to a fold result, like `REPRESENTATIVE_POLICY` is next to a legacy one.
FOLD_POLICY: tuple[str, ...] = (
    "1. The independent evidence unit is the PAPER (PMID). Rows of one paper are one piece of "
    "evidence, not several: 52 Ethanol rows are ONE paper. A row with no PMID is its own unit.",
    "2. Rows are lifted exactly as in the legacy ranking: strength |change| / (|change| + 20), "
    "confidence = evidence tier x significance gate (the 0.9 chain discount is applied after the fold), "
    "direction by the sign of the change. A change of exactly 0.0 is a reported null (strength 0): it counts in BOTH "
    "directions, at its own confidence (a well-run null is evidence of no effect either way); the sign convention "
    "has no zero, and filing the null on the protective side only would tilt every net toward harmful. A direction "
    "whose pool holds only nulls has strength 0, adds nothing to the net and is not emitted as an item.",
    "3. Inside one paper and one direction the existing representative-row policy keeps one row "
    "(tier, then significance, then the lower median change).",
    "4. The items of one direction are pooled with a port of the fork's Truth_Revision (patham9/PLN): weights "
    "c / (1 - c), confidence-weighted mean strength, confidence w / (w + 1); confidences are capped at "
    "0.999999 because Revision has no result at 1.0.",
    "5. At most one protective (Neg) and one harmful (Pos) pooled item per compound. The two are NOT "
    "pooled with each other.",
    "6. net = sum over the pooled items of (+1 if protective, -1 if harmful) x strength x confidence. "
    "For one item this is the legacy score.",
)

NET_FORMULA = (
    "net = sum over the pooled items of (+1 if Neg else -1) * strength * confidence; "
    "the confidence includes the 0.9 chain discount"
)


class RevisionUndefined(ValueError):
    """`Truth_Revision` has no result for these inputs (a zero divisor in `/safe`)."""


# ── the fork's Truth_Revision, ported ────────────────────────────────────────

def _c2w(c: float) -> float:
    """(/safe $c (- 1 $c)): no result unless the denominator is positive."""
    d = 1 - c
    if not d > 0.0:
        raise RevisionUndefined(f"Truth_c2w({c!r}): the denominator 1 - c is not positive")
    return c / d


def _w2c(w: float) -> float:
    d = w + 1
    if not d > 0.0:
        raise RevisionUndefined(f"Truth_w2c({w!r})")
    return w / d


def revision(t1: tuple[float, float], t2: tuple[float, float]) -> tuple[float, float]:
    """Port of `Truth_Revision` (vendor/pln_fork/lib_pln_fork.metta, Formulas.metta of patham9/PLN).

    The expression order is the library's, so the result is the library's to the last digit:

        w1 = c1/(1-c1)   w2 = c2/(1-c2)   w = w1 + w2
        f  = (w1 f1 + w2 f2) / w          c = w/(w+1)
        (min 1.0 f), (min 1.0 (max (max c c1) c2))

    Raises `RevisionUndefined` where the library returns no result (c = 1.0, or total weight 0).
    """
    f1, c1 = t1
    f2, c2 = t2
    w1 = _c2w(c1)
    w2 = _c2w(c2)
    w = w1 + w2
    if not w > 0.0:
        raise RevisionUndefined("total evidence weight is 0")
    f = ((w1 * f1) + (w2 * f2)) / w
    c = _w2c(w)
    return (min(1.0, f), min(1.0, max(max(c, c1), c2)))


def fold_truth_values(tvs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Left fold of `revision` over (strength, confidence) pairs, in the order given.

    One pair is returned unchanged (nothing is revised, so nothing is capped). Two or more: every
    confidence is first capped at `CONFIDENCE_CAP`, because Revision has no result at c = 1.0.
    """
    if not tvs:
        raise ValueError("nothing to fold")
    if len(tvs) == 1:
        return (tvs[0][0], tvs[0][1])
    acc = (tvs[0][0], min(tvs[0][1], CONFIDENCE_CAP))
    for s, c in tvs[1:]:
        acc = revision(acc, (s, min(c, CONFIDENCE_CAP)))
    return acc


# ── lifting one row ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiftedRow:
    """One DrugAge row, lifted: the same numbers `drugage_scoring.score_row` computes."""
    row: DrugAgeRow
    unit: str                  # the evidence unit it belongs to ("PMID_13369", or "row:<row id>")
    sign: str                  # Neg protective | Pos harmful (the mortality axis)
    strength: float
    native_confidence: float   # evidence tier x significance gate; no chain discount
    confidence: float          # native x chain discount == RowScore.confidence
    tier_category: str

    @property
    def score(self) -> float:
        """The legacy `RowScore.score`: strength x confidence, signed so that higher is better."""
        return self.strength * self.confidence * (1.0 if self.sign == SIGN_NEG else -1.0)


def evidence_unit_of(row: DrugAgeRow, evidence_unit: str = "paper") -> str:
    """The id of the independent evidence a row belongs to.

    "paper": the PMID token the ETL wrote (`PMID_13369`); a row with no `ReportedIn` atom is its own
    unit (`row:DrugAgeRow_17`). "row": every row is its own unit.
    """
    if evidence_unit not in EVIDENCE_UNITS:
        raise ValueError(f"evidence_unit must be one of {EVIDENCE_UNITS}, got {evidence_unit!r}")
    if evidence_unit == "paper" and row.pmid:
        return row.pmid
    return f"row:{row.row_id}"


def lift_row(
    row: DrugAgeRow,
    knobs: Optional[ScoringKnobs] = None,
    *,
    evidence_unit: str = "paper",
) -> Optional[LiftedRow]:
    """Lift ONE row, or None when it reports no lifespan change (it has no score, as in the legacy path).

    This repeats the arithmetic of `drugage_scoring.score_row` because that function returns only the
    already-discounted confidence, and the fold has to revise the NATIVE one. The duplication is
    guarded by `tests/test_drugage_fold.py::test_the_lift_equals_score_row_on_every_row_of_the_build`
    (bit for bit on all 3,423 rows). The integrator can remove it by exposing `tier * gate` on
    `RowScore` (see the notes).
    """
    if row.avg_change is None:
        return None
    k = knobs or load_knobs()

    magnitude = abs(row.avg_change)
    strength = magnitude / (magnitude + k.halfsat) if (magnitude + k.halfsat) else 0.0

    if row.is_itp:
        category = "ITP_Positive" if row.significance == "Significant" else "ITP_Negative"
        tier = k.evidence_confidence.get(category, 0.9)
        gate = 1.0
    else:
        clade = k.species_clade.get(row.species or "")
        category = (
            k.clade_categories.get(clade, k.untaxonomised_category)
            if clade else k.untaxonomised_category
        )
        tier = k.evidence_confidence.get(category, 0.35)
        gate = k.sig_gates.get(row.significance or "Unreported", 0.6)

    native = tier * gate
    return LiftedRow(
        row=row,
        unit=evidence_unit_of(row, evidence_unit),
        sign=SIGN_POS if row.avg_change < 0 else SIGN_NEG,
        strength=strength,
        native_confidence=native,
        confidence=native * k.chain_discount,
        tier_category=category,
    )


# ── stamps ───────────────────────────────────────────────────────────────────

def stamp_token(compound: str, unit: str, sign: str) -> str:
    """The evidence token of one (compound, evidence unit, direction).

    Unique per LINK, not per paper: the same paper restated for the other direction, or for another
    compound, is a different token. That is what lets the chainer map a stamp id back to exactly one
    link (a shared id on two serial links would make StampDisjoint refuse the chain, and one id on two
    parallel links would make the path rebuild ambiguous). The paper itself stays in `Item.units`.
    """
    return f"{compound}|{unit}|{sign}"


# ── items and results ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    """A pooled item: all the evidence of one compound in one direction."""
    compound: str
    sign: str
    strength: float
    native_confidence: float    # what Truth_Revision produced (no chain discount)
    confidence: float           # native x chain discount: comparable to a legacy per-row confidence
    units: tuple[str, ...]      # the evidence units (papers) pooled, in the canonical order (numeric PMID first)
    stamp: tuple[str, ...]      # one token per unit, unique to this item (see stamp_token)
    members: tuple[LiftedRow, ...]   # the representative row of each unit, same order as `units`
    rows_covered: int           # scorable rows of this compound and direction inside those units
    covered: tuple[str, ...] = ()    # their row ids, sorted (a null row counts in both directions under zero_rows="both")

    @property
    def score(self) -> float:
        return self.strength * self.confidence * (1.0 if self.sign == SIGN_NEG else -1.0)

    @property
    def pooled(self) -> bool:
        """True when more than one evidence unit stands behind the item."""
        return len(self.units) > 1

    def provenance(self) -> list[dict]:
        """The side-table rows for this item: one per unit, in the order of `stamp`."""
        return [
            {
                "token": tok,
                "kind": "drugage_paper" if not m.unit.startswith("row:") else "drugage_row",
                "unit": m.unit,
                "pmid": m.row.pmid,
                "representative_row": m.row.row_id,
                "tier": m.tier_category,
                "significance": m.row.significance,
                "avg_lifespan_change_percent": m.row.avg_change,
                "species": m.row.species,
                "is_itp": m.row.is_itp,
                "strength": m.strength,
                "native_confidence": m.native_confidence,
            }
            for tok, m in zip(self.stamp, self.members)
        ]


@dataclass(frozen=True)
class NetScore:
    """The compound's NET. Field names follow `drugage_scoring.RowScore`, so the code that reads a
    ranking entry (`drugage_top`, GET /drugage/top) reads this too: `row` is the lead row."""
    compound: str
    score: float
    sign: str
    strength: float
    confidence: float
    direction: str                 # protective | harmful | no_effect
    row: DrugAgeRow                # the lead row: the best-evidence representative of the item carrying the sign
    tier_category: str
    folded: bool                   # False for the legacy per-row result
    papers_neg: int = 0            # evidence units pooled in the protective item (the row itself when folded=False)
    papers_pos: int = 0
    rows_neg: int = 0
    rows_pos: int = 0

    @property
    def protective(self) -> bool:
        return self.sign == SIGN_NEG


@dataclass(frozen=True)
class CompoundFold:
    compound: str
    folded: bool
    evidence_unit: str
    items: tuple[Item, ...]        # fold: at most one Neg then one Pos; legacy: one per scorable row
    net: Optional[NetScore]        # None when no row reports a lifespan change
    rows_total: int
    rows_scorable: int
    units: int                     # distinct evidence units among the scorable rows

    @property
    def neg(self) -> Optional[Item]:
        return next((i for i in self.items if i.sign == SIGN_NEG), None) if self.folded else None

    @property
    def pos(self) -> Optional[Item]:
        return next((i for i in self.items if i.sign == SIGN_POS), None) if self.folded else None


# ── choosing a representative ────────────────────────────────────────────────

def _representative(rows: Sequence[DrugAgeRow]) -> DrugAgeRow:
    """The legacy representative of rows of ONE compound: `drugage_selector.select_rows(best_per_compound)`.

    `limit` is the group's own size: the hyperon-era `MAX_ROWS` cap does not apply to a fold.
    """
    if len(rows) == 1:
        return rows[0]
    picked = select_rows(rows, best_per_compound=True, limit=len(rows))
    if len(picked) != 1:      # pragma: no cover - guarded by the single-compound check
        raise ValueError("rows of more than one compound")
    return picked[0]


def _most_confident(group: Sequence["LiftedRow"]) -> "LiftedRow":
    best = max(g.native_confidence for g in group)
    tied = sorted((g for g in group if g.native_confidence == best), key=lambda g: (g.strength, g.row.row_id))
    return tied[(len(tied) - 1) // 2]


def _choose(group: Sequence["LiftedRow"], pick: str) -> "LiftedRow":
    """The one lifted row that stands for `group` (rows of one compound)."""
    if len(group) == 1:
        return group[0]
    if pick == PICK_CONFIDENT:
        return _most_confident(group)
    if pick != PICK_REPRESENTATIVE:
        raise ValueError(f"unknown pick policy {pick!r}")
    rep = _representative([lr.row for lr in group])
    return next(lr for lr in group if lr.row.row_id == rep.row_id)


def _unit_key(unit: str) -> tuple:
    """A canonical, deterministic order for evidence units: numeric PMIDs by number first."""
    if unit.startswith("PMID_") and unit[5:].isdigit():
        return (0, int(unit[5:]), unit)
    return (1, 0, unit)


def net_of_signed(parts: Sequence[tuple[str, float, float]]) -> tuple[float, str, float, float]:
    """THE NET from bare (sign, strength, confidence) parts, at most one per direction.

    Returns (score, sign, strength, confidence). For callers that hold derivations rather than rows (a ranking in
    the engine): `confidence` is the one of the part that carries the sign, `strength` is that part's own when it is
    the only one and `|score| / confidence` otherwise. `compute_net` is this plus the label and the lead row.
    """
    by_sign: dict[str, tuple[float, float]] = {}
    for sign, strength, confidence in parts:
        if sign not in (SIGN_NEG, SIGN_POS):
            raise ValueError(f"sign must be Neg or Pos, got {sign!r}")
        if sign in by_sign:
            raise ValueError(f"two {sign} parts; the net takes one per direction")
        by_sign[sign] = (strength, confidence)
    if not by_sign:
        raise ValueError("no parts")
    ordered = [(sg, by_sign[sg]) for sg in (SIGN_NEG, SIGN_POS) if sg in by_sign]
    terms = [s * c * (1.0 if sg == SIGN_NEG else -1.0) for sg, (s, c) in ordered]    # Neg: +s c, Pos: -s c
    score = terms[0] if len(terms) == 1 else terms[0] + terms[1]
    sign = SIGN_NEG if score >= 0 else SIGN_POS
    if sign not in by_sign:                      # a lone Pos part whose product underflowed to 0.0
        sign = ordered[0][0]
    strength, confidence = by_sign[sign]
    other = [t for (sg, _), t in zip(ordered, terms) if sg != sign]
    if len(ordered) == 2 and any(t != 0.0 for t in other):
        strength = abs(score) / confidence if confidence > 0 else 0.0
    # (a part that contributes exactly 0, a reported null, leaves the other's own strength: the legacy score and strength to the bit)
    return score, sign, strength, confidence


def compute_net(items: Sequence[Item], *, folded: bool = True, pick: str = PICK_REPRESENTATIVE) -> NetScore:
    """THE NET of one compound's pooled items (at most one per direction). See the module docstring.

    A single item returns that item's own strength, confidence and score, so a compound with evidence
    in one direction scores exactly what the legacy ranking scores that item.
    """
    if not items:
        raise ValueError("no items")
    by_sign: dict[str, Item] = {}
    for it in items:
        if it.sign in by_sign:
            raise ValueError(f"two {it.sign} items for {it.compound}; the net takes one per direction")
        by_sign[it.sign] = it
    neg, pos = by_sign.get(SIGN_NEG), by_sign.get(SIGN_POS)
    ordered = [i for i in (neg, pos) if i is not None]

    score, sign, strength, confidence = net_of_signed([(i.sign, i.strength, i.confidence) for i in ordered])
    carrier = by_sign[sign]

    members = [m for i in ordered for m in i.members]
    if score == 0.0 or all((m.row.significance or "") == "NotSignificant" for m in members):
        direction = "no_effect"
    else:
        direction = "protective" if sign == SIGN_NEG else "harmful"

    lead = _choose(carrier.members, pick)
    return NetScore(
        compound=carrier.compound, score=score, sign=sign, strength=strength, confidence=confidence,
        direction=direction, row=lead.row, tier_category=lead.tier_category, folded=folded,
        papers_neg=len(neg.units) if neg else 0, papers_pos=len(pos.units) if pos else 0,
        rows_neg=neg.rows_covered if neg else 0, rows_pos=pos.rows_covered if pos else 0,
    )


# ── folding one compound ─────────────────────────────────────────────────────

def _row_item(lr: LiftedRow) -> Item:
    return Item(
        compound=lr.row.compound, sign=lr.sign, strength=lr.strength,
        native_confidence=lr.native_confidence, confidence=lr.confidence,
        units=(lr.unit,), stamp=(stamp_token(lr.row.compound, lr.unit, lr.sign),),
        members=(lr,), rows_covered=1, covered=(lr.row.row_id,),
    )


def _pool(compound: str, sign: str, groups: Mapping[str, list[LiftedRow]], chain_discount: float,
          pick: str = PICK_REPRESENTATIVE) -> Item:
    """One pooled item from {unit: lifted rows of that unit in this direction}."""
    units = sorted(groups, key=_unit_key)
    members = [_choose(groups[u], pick) for u in units]
    strength, native = fold_truth_values([(m.strength, m.native_confidence) for m in members])
    return Item(
        compound=compound, sign=sign, strength=strength, native_confidence=native,
        confidence=native * chain_discount,
        units=tuple(units), stamp=tuple(stamp_token(compound, u, sign) for u in units),
        members=tuple(members), rows_covered=sum(len(g) for g in groups.values()),
        covered=tuple(sorted(lr.row.row_id for g in groups.values() for lr in g)),
    )


def group_by_direction(lifted: Sequence[LiftedRow], zero_rows: str = DEFAULT_ZERO_ROWS) -> dict[str, dict[str, list[LiftedRow]]]:
    """{direction: {evidence unit: lifted rows}} for ONE compound, with a reported null (strength exactly 0) placed by the
    convention: 'both' = in each direction's pool (as a copy filed under that direction), 'neither' = in none (unless nothing
    else is left: a compound whose only evidence is nulls stays, as a null on both sides, and is never dropped from a
    ranking), 'protective' = in the protective pool only, where the sign convention (no zero) files it."""
    if zero_rows not in ZERO_ROW_CONVENTIONS:
        raise ValueError(f"zero_rows must be one of {ZERO_ROW_CONVENTIONS}, got {zero_rows!r}")
    by_dir: dict[str, dict[str, list[LiftedRow]]] = {SIGN_NEG: {}, SIGN_POS: {}}
    nulls: list[LiftedRow] = []
    for lr in lifted:
        if lr.strength == 0.0 and zero_rows != "protective":
            nulls.append(lr)
        else:
            by_dir[lr.sign].setdefault(lr.unit, []).append(lr)
    if nulls and (zero_rows == "both" or not any(by_dir.values())):
        for lr in nulls:
            for sign in (SIGN_NEG, SIGN_POS):
                by_dir[sign].setdefault(lr.unit, []).append(lr if lr.sign == sign else replace(lr, sign=sign))
    return by_dir


def fold_lifted(
    lifted: Sequence[LiftedRow],
    *,
    fold: bool = True,
    chain_discount: float,
    pick: str = PICK_REPRESENTATIVE,
    zero_rows: str = DEFAULT_ZERO_ROWS,
) -> tuple[tuple[Item, ...], Optional[NetScore]]:
    """The fold itself, on rows that are already lifted (one compound). Returns (items, net).

    fold=True   `lifted` carries the evidence units as given; at most one Neg and one Pos item. `zero_rows` says how a
                reported null counts (see `group_by_direction`).
    fold=False  the legacy result: pass rows lifted with evidence_unit="row"; one item per row, the net is the
                representative row's score.
    """
    if not lifted:
        return (), None
    compound = lifted[0].row.compound
    if any(lr.row.compound != compound for lr in lifted):
        raise ValueError("lifted rows of more than one compound")
    if not fold:
        lead = _choose(list(lifted), pick)
        net = compute_net([_row_item(lead)], folded=False, pick=pick)
        items = tuple(_row_item(lr) for lr in sorted(lifted, key=lambda x: x.row.row_id))
        return items, net
    by_dir = group_by_direction(lifted, zero_rows)
    items = tuple(
        _pool(compound, sign, by_dir[sign], chain_discount, pick)
        for sign in (SIGN_NEG, SIGN_POS) if by_dir[sign]
    )
    if zero_rows != "protective":
        # A direction whose pool holds only reported nulls pools to strength exactly 0: it adds nothing to the net (its term is 0)
        # and would show every consumer a second, empty derivation. It is not an item. The nulls still count: they sit in the OTHER
        # pool, diluting it, and a compound whose only evidence is nulls keeps its protective item (the legacy shape: one row, score 0).
        real = tuple(i for i in items if i.strength != 0.0)
        items = real or items[:1]
    return items, compute_net(items, pick=pick)


def fold_compound(
    rows: Iterable[DrugAgeRow],
    *,
    fold: bool = True,
    evidence_unit: str = "paper",
    knobs: Optional[ScoringKnobs] = None,
    zero_rows: str = DEFAULT_ZERO_ROWS,
) -> CompoundFold:
    """Fold the DrugAge rows of ONE compound. Pure and deterministic.

    fold=True   at most one Neg and one Pos pooled item (paper-level evidence, Truth_Revision) and the NET.
    fold=False  the legacy result: one item per scorable row, the net is the representative row's score.
    zero_rows   how a reported null counts when the fold pools ('both' | 'neither' | 'protective'; module docstring).

    `rows` may hold any number of rows (no cap). Raises ValueError for rows of more than one compound or
    for a repeated row id.
    """
    if evidence_unit not in EVIDENCE_UNITS:
        raise ValueError(f"evidence_unit must be one of {EVIDENCE_UNITS}, got {evidence_unit!r}")
    rows = list(rows)
    if not rows:
        raise ValueError("fold_compound needs at least one row")
    compounds = {r.compound for r in rows}
    if len(compounds) != 1:
        raise ValueError(f"rows of {len(compounds)} compounds; fold one compound at a time")
    compound = rows[0].compound
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError(f"a row id repeats among the rows of {compound}")
    k = knobs or load_knobs()

    unit_mode = evidence_unit if fold else "row"
    lifted = [lr for lr in (lift_row(r, k, evidence_unit=unit_mode) for r in rows) if lr is not None]
    units = len({lr.unit for lr in lifted})
    items, net = fold_lifted(lifted, fold=fold, chain_discount=k.chain_discount, zero_rows=zero_rows)
    return CompoundFold(compound, fold, evidence_unit, items, net, len(rows), len(lifted), units)


def fold_rows(
    rows: Iterable[DrugAgeRow],
    *,
    fold: bool = True,
    evidence_unit: str = "paper",
    knobs: Optional[ScoringKnobs] = None,
    zero_rows: str = DEFAULT_ZERO_ROWS,
) -> dict[str, CompoundFold]:
    """`fold_compound` for every compound among `rows` (no cap on the number of rows)."""
    k = knobs or load_knobs()
    by_compound: dict[str, list[DrugAgeRow]] = {}
    for r in rows:
        by_compound.setdefault(r.compound, []).append(r)
    return {
        c: fold_compound(rs, fold=fold, evidence_unit=evidence_unit, knobs=k, zero_rows=zero_rows)
        for c, rs in by_compound.items()
    }


# ── the whole-build ranking, with the filters of drugage_top ─────────────────

@dataclass
class FoldTop:
    """The counterpart of `core.drugage_router.DrugAgeTop`: a ranking over a set of rows."""
    entries: list[NetScore] = field(default_factory=list)
    total_compounds: int = 0
    total_compounds_in_source: int = 0
    unscorable_compounds: int = 0
    total_rows: int = 0
    scored_rows: int = 0
    unscorable_rows: int = 0
    filters: dict = field(default_factory=dict)
    fold: bool = True
    evidence_unit: str = "paper"
    zero_rows: str = DEFAULT_ZERO_ROWS


def top_compounds(
    rows: Iterable[DrugAgeRow],
    *,
    n: int = 20,
    species: Optional[str] = None,
    clade: Optional[str] = None,
    min_confidence: float = 0.0,
    itp_only: bool = False,
    significant_only: bool = False,
    direction: str = "protective",
    fold: bool = True,
    evidence_unit: str = "paper",
    knobs: Optional[ScoringKnobs] = None,
    zero_rows: str = DEFAULT_ZERO_ROWS,
) -> FoldTop:
    """Rank compounds by NET, with the same filters and the same ordering rules as `drugage_top`.

    With fold=False the entries equal `drugage_top`'s (the lift-fidelity test compares them bit for bit).
    The row filters (species, clade, itp_only, significant_only) act on ROWS before anything is pooled, so
    "mouse evidence only" pools mouse papers only.
    """
    k = knobs or load_knobs()
    rows = list(rows)
    if species:
        rows = [r for r in rows if r.species and _norm(r.species) == _norm(species)]
    if clade:
        rows = [r for r in rows if k.species_clade.get(r.species or "", "").lower() == clade.lower()]
    if itp_only:
        rows = [r for r in rows if r.is_itp]
    if significant_only:
        rows = [r for r in rows if r.significance == "Significant"]

    folds = fold_rows(rows, fold=fold, evidence_unit=evidence_unit, knobs=k, zero_rows=zero_rows)
    scored = {c: f for c, f in folds.items() if f.net is not None}
    nets = [f.net for f in scored.values()]
    scored_rows = sum(f.rows_scorable for f in folds.values())

    if min_confidence > 0:
        nets = [s for s in nets if s.confidence >= min_confidence]
    if direction == "protective":
        nets = [s for s in nets if s.direction == "protective"]
    elif direction == "harmful":
        nets = [s for s in nets if s.direction == "harmful"]
    elif direction == "none":
        nets = [s for s in nets if s.direction == "no_effect"]

    if direction == "harmful":
        nets.sort(key=lambda s: (s.score, s.compound))
    else:
        nets.sort(key=lambda s: (-s.score, s.compound))

    return FoldTop(
        entries=nets[: max(0, n)],
        total_compounds=len(scored),
        total_compounds_in_source=len(folds),
        unscorable_compounds=len(folds) - len(scored),
        total_rows=len(rows),
        scored_rows=scored_rows,
        unscorable_rows=len(rows) - scored_rows,
        filters={
            "species": species, "clade": clade, "min_confidence": min_confidence,
            "itp_only": itp_only, "significant_only": significant_only,
            "direction": direction, "n": n,
        },
        fold=fold,
        evidence_unit=evidence_unit,
        zero_rows=zero_rows,
    )


# ── the dict interface of the chainer ────────────────────────────────────────

#: A lifted DrugAge item as `core.pln_chainer` hands it over (its DRUGAGE_ITEM_KEYS): one item per row.
ITEM_KEYS = ("compound", "target", "direction", "strength", "confidence", "rows", "papers")
#: Optional keys. With all three on every item the within-paper pick is the repo's representative-row policy;
#: without them it is the fallback PICK_CONFIDENT, and every output item says which one was used.
POLICY_KEYS = ("change", "significance", "itp")


class FoldItemError(ValueError):
    """A lifted DrugAge item the fold cannot accept."""


def _lifted_from_item(it: Mapping, *, evidence_unit: str, chain_discount: float) -> LiftedRow:
    missing = [k for k in ITEM_KEYS if k not in it]
    if missing:
        raise FoldItemError(f"a DrugAge item needs the keys {ITEM_KEYS}; missing {missing}")
    rows = tuple(it["rows"] or ())
    papers = tuple(it["papers"] or ())
    if len(rows) != 1 or len(papers) > 1:
        raise FoldItemError(f"a lifted item stands for ONE row (rows={rows!r}, papers={papers!r})")
    direction = it["direction"]
    if direction not in ("Pos", "Neg"):
        raise FoldItemError(f"direction must be Pos or Neg (the lifespan direction), got {direction!r}")
    try:
        strength, confidence = float(it["strength"]), float(it["confidence"])
    except (TypeError, ValueError) as exc:
        raise FoldItemError(f"strength and confidence must be numbers: {exc}") from exc
    if not (0.0 <= strength <= 1.0 and 0.0 <= confidence <= 1.0):       # also refuses nan
        raise FoldItemError(f"strength and confidence must be in [0, 1], got ({strength!r}, {confidence!r})")
    name = str(rows[0])
    paper = str(papers[0]) if papers and papers[0] else ""
    change = it.get("change")
    change = None if change is None else float(change)
    sign = SIGN_NEG if direction == "Pos" else SIGN_POS                 # lifespan up is mortality down
    if change is not None and (change < 0) != (sign == SIGN_POS):
        raise FoldItemError(f"{name}: direction {direction} disagrees with change {change}")
    row = DrugAgeRow(
        row_id=name, compound=str(it["compound"]), species=it.get("species"), is_itp=bool(it.get("itp", False)),
        significance=it.get("significance"), avg_change=change,
        block=f"(ReportedIn {name} {paper})" if paper else "",
    )
    unit = paper if (paper and evidence_unit == "paper") else f"row:{name}"
    return LiftedRow(row=row, unit=unit, sign=sign, strength=strength, native_confidence=confidence,
                     confidence=confidence * chain_discount, tier_category=str(it.get("tier") or ""))


def fold_items(
    items: Sequence[Mapping],
    *,
    fold: bool = True,
    evidence_unit: str = "paper",
    knobs: Optional[ScoringKnobs] = None,
    zero_rows: str = DEFAULT_ZERO_ROWS,
) -> list[dict]:
    """Pool lifted DrugAge items (one per row) into at most one item per compound, target and direction.

    This is the interface `core.pln_chainer` calls (`ontology.drugage_fold.fold_items`). An input item is a dict with

        compound, target, direction ('Pos' | 'Neg': the direction of the LIFESPAN change, as `lifespan-sign`),
        strength, confidence (the lifted `tier x gate`, no chain discount), rows (a 1-tuple of the row name),
        papers (a 1-tuple of the PMID token, '' when the row has none)

    and optionally `change` (AvgLifespanChangePercent), `significance` and `itp`, which together select the repo's
    representative-row policy for the choice inside a paper (without them the fallback is the most confident row,
    ties to the lower median strength; the output says which was used in `within_paper_pick`). The policy is chosen
    PER (compound, target) GROUP: one compound's rows lacking those keys never change another compound's number. Extra
    input keys are ignored. `zero_rows` is the null convention (module docstring). The pooled items come back in the same shape: `strength`, `confidence` (the pooled NATIVE confidence;
    the chainer's 0.9 per-hop shim is applied afterwards, so the final answer is `confidence x 0.9` for the two hops
    compound -> Lifespan -> Mortality), `rows` (the representative row of each paper), `papers` (their PMIDs), plus

        units, stamp          the evidence units and one token per unit, unique to this link
        rows_covered, covered_rows   how many scorable rows, and which, stand behind the item
        pooled                True when more than one paper stands behind it
        native_confidence, chain_confidence, mortality_sign
        net, net_sign, net_direction   the compound's NET (the same on every item of a compound); the label
                              `net_direction` is None in the fallback, which does not know the significance
        within_paper_pick

    fold=False returns the items unchanged (the legacy per-row results).
    """
    if evidence_unit not in EVIDENCE_UNITS:
        raise ValueError(f"evidence_unit must be one of {EVIDENCE_UNITS}, got {evidence_unit!r}")
    items = list(items)
    if not fold:
        return [dict(i) for i in items]
    if zero_rows not in ZERO_ROW_CONVENTIONS:
        raise ValueError(f"zero_rows must be one of {ZERO_ROW_CONVENTIONS}, got {zero_rows!r}")
    k = knobs or load_knobs()

    groups: dict[tuple[str, str], list[LiftedRow]] = {}
    complete: dict[tuple[str, str], bool] = {}
    seen: set[tuple[str, str, str]] = set()
    for it in items:
        lr = _lifted_from_item(it, evidence_unit=evidence_unit, chain_discount=k.chain_discount)
        key = (lr.row.compound, str(it["target"]))
        ident = (*key, lr.row.row_id)
        if ident in seen:
            raise FoldItemError(f"row {lr.row.row_id} of {lr.row.compound} appears twice")
        seen.add(ident)
        groups.setdefault(key, []).append(lr)
        complete[key] = complete.get(key, True) and all(it.get(name) is not None for name in POLICY_KEYS)

    out: list[dict] = []
    for (compound, target), lifted in sorted(groups.items()):
        exact = complete[(compound, target)]
        pick = PICK_REPRESENTATIVE if exact else PICK_CONFIDENT
        pooled, net = fold_lifted(lifted, fold=True, chain_discount=k.chain_discount, pick=pick, zero_rows=zero_rows)
        for item in pooled:
            covered = list(item.covered)
            out.append({
                "compound": compound,
                "target": target,
                "direction": "Pos" if item.sign == SIGN_NEG else "Neg",
                "strength": item.strength,
                "confidence": item.native_confidence,
                "rows": tuple(m.row.row_id for m in item.members),
                "papers": tuple(m.row.pmid or "" for m in item.members),
                "native_confidence": item.native_confidence,
                "chain_confidence": item.confidence,
                "mortality_sign": item.sign,
                "units": item.units,
                "stamp": item.stamp,
                "pooled": item.pooled,
                "rows_covered": item.rows_covered,
                "covered_rows": tuple(covered),
                "within_paper_pick": pick,
                "net": net.score,
                "net_sign": net.sign,
                # the label needs each row's significance; without it the fallback cannot claim one
                "net_direction": net.direction if exact else None,
            })
    out.sort(key=lambda d: (d["compound"], d["target"], d["direction"]))
    return out


# ── stamp ids for the engine ─────────────────────────────────────────────────

def assign_stamp_ids(folds: Iterable[CompoundFold], *, first: int = 1) -> dict[str, int]:
    """Deterministic integer ids for the stamp tokens of every pooled item, sorted by token.

    The library's stamps are integers (id 0 is the intervention premise, so `first` must be >= 1 and
    must start above every id the curated links already use). Refuses a token that appears in two items:
    each item is one link, and a shared id on two links breaks both the stamp check and the path rebuild.
    """
    if first < 1:
        raise ValueError("stamp ids start at 1; 0 is the premise")
    seen: set[str] = set()
    for f in folds:
        for it in f.items:
            for tok in it.stamp:
                if tok in seen:
                    raise ValueError(f"stamp token {tok!r} is on two items")
                seen.add(tok)
    return {tok: first + i for i, tok in enumerate(sorted(seen))}


def semantics(zero_rows: str = DEFAULT_ZERO_ROWS) -> dict:
    """The fold's definitions as data, for a response's `semantics` block."""
    return {
        "evidence_unit": "paper (PMID); a row without a PMID is its own unit",
        "policy": list(FOLD_POLICY),
        "net": NET_FORMULA,
        "confidence_cap": CONFIDENCE_CAP,
        "zero_rows": zero_rows,
        "zero_rows_conventions": {
            "both": "a reported null (change exactly 0.0) counts in both directions, at strength 0 and its own confidence; "
                    "a direction that holds only nulls is not emitted as an item (it would add nothing to the net)",
            "neither": "a reported null is dropped from the pools (a compound with nothing else stays, as a null)",
            "protective": "a reported null counts on the protective side only: the sign convention has no zero, and this "
                          "tilts every net toward harmful",
        },
    }
