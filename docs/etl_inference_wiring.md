# ETL → Inference Wiring — DrugAge vertical slice (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`drugage_calibration.metta` (the lift + calibration + ranking wiring),
`pln_chat/ontology/drugage_selector.py` (row selection),
`pln_chat/core/pln_runner.py` (`run_drugage_ranking`, `DRUGAGE_STACK`, scoped
injection); regression tests: `tests/test_drugage_calibration.py` (16 tests).
**Scope:** make `infer` / `rank-interventions` operate over **real DrugAge rows**,
uncertainty-quantified, *without* loading the whole ETL dump — a data-driven
`pipeline.md` §4.2 Demo 2 over thousands of real compounds instead of three
curated ones.

> **Dependency / load order.** Sits on top of the whole inference stack —
> calibration → deduction → ranking. It loads **last**; the exact ordered file
> list is `pln_runner.DRUGAGE_STACK` and is repeated in the header of
> `drugage_calibration.metta`. It is a pure consumer: it adds a lift + a bridge +
> one `infer` equation, and reuses `rank-interventions` unchanged.

---

## 1. Why we need it

The repo had two disconnected halves. `scripts/run_etl.sh` extracts ~46k lines
of DrugAge/GenAge/CellAge evidence into `build/`, but inference could not touch
any of it, for two independent reasons:

1. **The rows are raw measurements, not `(Effect …)` links.** A DrugAge row is
   source-faithful and TV-free (the calibration-layer immutability invariant,
   `docs/calibration_layer.md` §9):
   ```metta
   (InstanceOf DrugAgeRow_1552 Experiment)
   (UsesIntervention DrugAgeRow_1552 Rapamycin)
   (UsesSpecies DrugAgeRow_1552 Mus_musculus)
   (IsITPStudy DrugAgeRow_1552)                    ; 164 rows — gold-standard tier
   (AvgLifespanChangePercent DrugAgeRow_1552 26.0)
   (AvgLifespanSignificance  DrugAgeRow_1552 Significant)
   (ReportedIn DrugAgeRow_1552 PMID_24341993)      ; provenance
   ```
   Nothing lifted these into the `(Effect <from> <to> <sign> (stv s c))` links
   deduction and ranking consume.

2. **hyperon 0.2.10 panics on a big space.** Loading the full DrugAge dump into
   one space and issuing *any* query aborts the process with a non-unwinding Rust
   panic (measured: ~420 rows loaded alone already trips it — see §6). The chat
   app only works today because it *excludes* files over 60 KB — i.e. it never
   loads the ETL data at all.

This layer closes both gaps for one vertical slice: **lift** DrugAge rows into
calibrated signed links on the fly, and rank real compounds inside a **scoped
space** that stays under the panic threshold.

## 2. Where it sits

```
build/drugage_etl.metta      selector picks a FEW matching rows
  (3,423 raw rows) ──────────▶  ┐
                                ├─ scoped space ─drugage-effect(lift)─▶ (Effect C Lifespan sign (stv s c))
DRUGAGE_STACK (inference   ────▶┘                    │
 stack, hand-written)                                └─ infer(C→Lifespan→Mortality) ─▶ rank-interventions ─▶ ranked signed list
```

Python (`drugage_selector`) decides *which rows* enter the space; MeTTa
(`drugage_calibration.metta`) does the lifting and the inference. Raw rows are
never rewritten — the lift is a pure function of `(space, row)`, exactly like
`link-effect` lifts a GrimAge empirical record in `pln_deduction.metta`.

## 3. The lift — one row → one signed, calibrated Effect

`(drugage-effect <space> <row>)` is the deliverable primitive:

```metta
!(drugage-effect &self DrugAgeRow_1552)
;; (Effect Rapamycin Lifespan Pos (stv 0.565 0.9))
```

`s` is the calibrated effect **magnitude**, `c` the evidence weight, `sign` the
direction. Provenance stays on the raw row (`ReportedIn … PMID_…`), reachable
from the compound/row for the audit trail. `infer` steps through this lift via
one added equation (a scoped realisation of `docs/deduction_layer.md` §7.3,
"stepping through empirical links mid-chain").

## 4. The calibration map

All of it lives in `drugage_calibration.metta`; every constant is in its §1
"knobs" block so tuning is local (`docs/calibration_layer.md` §8).

### 4.1 Strength ← `AvgLifespanChangePercent` (saturating)

A percent change is unbounded, so it is squashed to `[0,1)` by a saturating
transform on the **magnitude** (sign is a separate axis):

```
s = |pct| / (|pct| + k)          k = lifespan-halfsat = 20
```

`k` is the change at which `s = 0.5`. `k = 20` puts a strong replicated
extension (≈+20–25 %, e.g. ITP rapamycin) near 0.5–0.56 and a large +80 % effect
near 0.8. Coarse by design; raise `k` to be more conservative about large
reported effects.

### 4.2 Confidence ← MIN(evidence tier, significance gate)

Confidence reuses the single authority `epistemic_calibration.metta`
(`evidence-confidence`) and combines the applicable factors with **MIN — the
weakest link governs** (`docs/calibration_layer.md` §3.2).

| Factor | Row property | → EvidenceCategory / cap | value |
|---|---|---|---|
| **ITP** | `IsITPStudy` present | `ITP_Positive` (if Significant) / `ITP_Negative` | **0.90** |
| species: mammal | `UsesSpecies` → `Vertebrate` | `AnimalStudies_Single` | 0.50 |
| species: invertebrate | → `Invertebrate` (worm, fly) | `InVitro` *(borrowed rung)* | 0.35 |
| species: fungi/protozoa | → `Fungi` / `Protozoa` | `TraditionalUse` | 0.20 |
| significance gate (non-ITP) | `Significant` / `Unreported` / `NotSignificant` | cap 1.0 / 0.6 / 0.4 | — |

- **ITP is the gold-standard replicated-mouse program**, so an ITP row earns
  0.90 regardless of outcome; significance instead selects `ITP_Positive` vs
  `ITP_Negative` (both 0.90 — we trust a well-run *null* too). This is exactly
  what down-weights resveratrol through near-zero **strength**, not low
  confidence (§7).
- **Species clade** comes from `species_taxonomy.metta` (`Inheritance <sp>
  <clade>`); "shown only in a worm is weaker than shown in a mouse" lives here.
  The invertebrate/fungi rungs are *borrowed* for their numeric value pending a
  dedicated `ModelOrganism` tier — the same "no perfect tier yet, map
  conservatively" move `mechanistic_bridges.metta` makes for mechanistic
  consensus (`docs/calibration_layer.md` open-Q #2/#4).
- **Significance gate** caps a non-ITP row that did not reach significance.

### 4.3 Sign ← direction of the change — and THE SIGN DECISION

`Pos` = *from* raises *to* (`pln_deduction.metta` §0). A compound that **extends**
lifespan (`pct>0`) raises `Lifespan` → `Pos`; one that **shortens** it → `Neg`.

**The trap:** `Lifespan` is a *beneficial* outcome, so here `Pos` is *good* — the
opposite of the harmful CHD/mortality axis `rank-score` was written for (there
`Neg`=protective=good). Ranking on `Lifespan` directly would need an inverted
scorer, forking the sign convention — the exact footgun that silently inverts a
ranking.

**Decision (documented, test-guarded):** keep **one** global convention
(`Neg = beneficial`) by chaining through a curated bridge

```metta
(Effect Lifespan Mortality Neg (stv 1.0 1.0))
```

so `infer <compound> Mortality` composes *extend-lifespan* (`Pos`) with
*lifespan-reduces-mortality* (`Neg`) and `Pos·Neg = Neg = protective`, which
`rank-score` already scores as good. A life-shortener gives `Neg·Neg = Pos =
harmful`, ranked below zero. The bridge's TV is deliberately **neutral**
`(1.0 1.0)`: it is a definitional sign-adapter, constant across every compound,
so it cannot change the ranking *order* — the real cross-species surrogacy
uncertainty is already carried by the species tier in the compound→Lifespan
confidence. `tests/test_drugage_calibration.py` asserts a life-extender is `Neg`
at Mortality and a life-shortener is `Pos` — getting this backwards is what the
guard catches.

*Alternative considered and rejected:* rank on `Lifespan` directly with a
`rank-score-beneficial` that flips the sign. Rejected because two scoring
functions with opposite conventions are precisely the footgun the sign-trap
warning is about; a single convention is safer and reuses `rank-interventions`
verbatim.

## 5. Wiring into ranking (the demo)

`run_drugage_ranking(compounds)` assembles the scoped space and runs the existing
`rank-interventions` against `Mortality`. Over five real rows (the committed
`tests/fixtures/drugage_real_rows.metta`):

```
Rapamycin      +0.4578  Neg (stv 0.565 0.810)   ITP mouse, +26%  — replicated positive, top
Trimethadione  +0.2210  Neg (stv 0.701 0.315)   worm,     +47%  — big effect, low confidence
Acarbose       +0.1620  Neg (stv 0.200 0.810)   ITP mouse,  +5%  — small effect, high confidence
Resveratrol    +0.0000  Neg (stv 0.000 0.810)   ITP mouse,   0%  — ITP-NEGATIVE null
Metformin      +0.0000  Neg (stv 0.000 0.810)   ITP mouse,   0%  — ITP-negative null
```

Three reads an LLM cannot reproduce from the numbers:

1. **The ITP-negative down-weight (the bonus; `pipeline.md` App. B.3 / Demo 6).**
   Resveratrol failed the ITP. Its confidence stays *high* (0.90 — an ITP result),
   but its near-zero **strength** sinks it to ~0, far below the replicated ITP
   positive rapamycin. The honest statement is "we are confident resveratrol does
   *not* extend mouse lifespan", and that is exactly what the split (high `c`,
   ~0 `s`) encodes — not "we are unsure".
2. **Magnitude ⇆ confidence trade-off.** A +47 % *worm* effect (Trimethadione)
   outranks a +5 % *mouse ITP* effect (Acarbose): `0.701·0.315 > 0.200·0.810`.
   Debatable, but honest — the ranking weighs magnitude **and** confidence, never
   sign alone.
3. **Signed, provenance-bearing, deterministic.** Same inputs → same order, with
   the full `(signed … (stv s c))` retained per candidate.

**Aggregation:** ranking is per **compound**, one entry each. `best_per_compound`
(default) picks a compound's highest-evidence-tier row (ITP > significant >
unreported > null), tie-broken by the *median* change (a non-cherry-picked
representative). Merging a compound's *multiple* rows into one revised TV (and
rewarding replication *count*) is PLN revision — the documented follow-up
(§8, shared with `docs/deduction_layer.md` §7.4).

## 6. The scoped-space fix (the panic)

hyperon 0.2.10 aborts with a non-unwinding panic once one space exceeds ~a few
thousand atoms. Measured boundary (rows loaded alone, then one query):

| rows | atoms | result |
|---|---|---|
| 400 | ~4.7k | OK |
| 420 | ~4.9k | **panic (abort)** |

So we never build a big space. `run_drugage_ranking`:

- loads **`DRUGAGE_STACK`** — the minimal hand-written layers only (system types,
  logical predicates, epistemic calibration, species taxonomy, evidence
  calibration, deduction, ranking, this layer). It **excludes** grim_age /
  hallmarks / mechanistic_bridges: those are the CHD axis, add atoms, and would
  let a DrugAge compound name collide with a curated `Effect` bridge;
- injects only a **filtered slice** of rows (`drugage_selector.build_drugage_slice`
  → `run_query(..., extra_atoms=…)`), capped at `MAX_ROWS = 150` (well under the
  boundary; a real query injects a handful). The full 3,423-row dump is **never**
  loaded.

`tests/test_drugage_calibration.py::test_ranking_real_rows_downweights_itp_negative`
runs the whole path end-to-end; completing at all is the no-panic proof.

## 7. Hyperon gotchas hit here (every prior layer hit its own)

- **`(match &self …)` in an imported module reads only that module's space.** The
  space is threaded in explicitly and callers pass the top-level `&self`
  (`docs/calibration_layer.md` §10.1).
- **Grounded ops don't evaluate their args.** `+ / < == car-atom` see unreduced
  atoms, so every intermediate (magnitude, confidence, min) is forced through
  `let*` before it feeds arithmetic — else `(+ $m (lifespan-halfsat))` stays
  symbolic and the division misfires (same discipline as
  `pln_intervention_ranking.metta` §3).
- **`Effect` args are typed `Atom`, which *suppresses* their reduction.** A raw
  `(Effect .. (drugage-row-strength ..) ..)` keeps the call symbolic (exactly how
  curated bridges store `(evidence-confidence ..)` unevaluated). `drugage-effect`
  forces sign/strength/confidence through `let*` *first* so the emitted link
  carries concrete numbers.
- **`rank-interventions`'s insertion sort is ~O(n²) with a high constant**
  (measured: 5 candidates ≈1 s, 10 ≈6 s, 15 ≈16 s). The vertical-slice answer is
  architectural, not a rewrite: **rank compounds, not rows**, so the pool is
  Demo-2 scale (a handful) — the same constraint the existing ranking demo lives
  under. Ranking dozens at once needs a faster sort (§8).
- **`NotSignificant`, not `NonSignificant`.** The ETL emits `NotSignificant`;
  `system_types.metta` declares the label `NonSignificant` (a stale typo nothing
  matches). This layer matches the real data.
- **Never load the full KB into one space** (the panic, §6) — scope per query.

## 8. Open questions / next increments

1. **PLN revision over a compound's rows.** v1 collapses to one representative
   row; combining *all* of a compound's experiments into one revised TV (and
   rewarding replication *count* with higher confidence) is the honest upgrade,
   shared with `docs/deduction_layer.md` §7.4 and `docs/intervention_ranking.md`
   §5.1.
2. **A dedicated `ModelOrganism` evidence tier.** Invertebrate/fungi confidence
   currently *borrows* `InVitro` / `TraditionalUse` rungs; a first-class
   species-translatability tier (and a mammal-vs-fish split within `Vertebrate`)
   would be truer (`docs/calibration_layer.md` open-Q #2/#4).
3. **Faster ranking sort**, so a pool of dozens of real compounds is practical
   (§7).
4. **Strength normalisation across organisms.** A +40 % worm extension and a
   +40 % mouse extension are not equivalent biology; the transform currently
   treats percent uniformly. A per-clade scale is a natural knob.
5. ~~**Surface in the chat app.**~~ ✅ **Done (v1).** The translator now emits a
   **dedicated** NL-facing form `(rank-drugage-lifespan (C1 C2 …))` (three
   few-shots in `pln_chat/prompts/few_shot_examples.json` + a system-prompt
   section), and `pln_chat/app.py::chat()` routes it — BEFORE the generic
   `run_query` — to `run_drugage_ranking` (the scoped stack), feeding the result
   through the unchanged `format_bot_response`. Routing/parse/degrade live in the
   Gradio-free `pln_chat/core/drugage_router.py`; asserted by
   `tests/test_drugage_chat_routing.py` and demonstrated by ten executed queries
   in `docs/drugage_chat_test_queries.md`. This unblocks
   `docs/intervention_ranking.md` §5.4 for the DrugAge axis. Routing caveats hit:
   - **A distinct symbol is load-bearing.** Overloading `(rank-interventions …
     Mortality)` could not be routed unambiguously *and* would be mis-executed by
     the generic path (no DrugAge rows in `_ALL_KB_PATHS`, colliding CHD bridges,
     panic risk). A separate `rank-drugage-lifespan` symbol is what makes the
     branch a clean string check.
   - **The generic symbol validator is bypassed for the routed form.** Its
     compound names / `rank-drugage-lifespan` are not in the UI-selected registry,
     so `validate()` would flag them as "unknown"; the routed form is validated
     instead by the selector (does a real row match?). Everything else keeps
     `validate()`.
   - **Compound-name mismatch is the likeliest silent failure** (LLM string vs
     ETL token). Handled two ways: the selector matches case-/separator-insensitively
     (`rapamycin`→`Rapamycin`) and the few-shots model canonical names. A requested
     compound with no row is **omitted and named** (not mis-ranked); tested
     explicitly.
   - **`build/` is a soft dependency.** Missing `build/drugage_etl.metta` degrades
     to a "run `scripts/run_etl.sh`" message, never a crash. (The committed 5-row
     fixture has no non-null ITP variety, so the app path targets `build/`.)
6. **Other ETL verticals.** GenAge/CellAge rows can be lifted the same way (a
   gene→hallmark or senescence-effect link); this slice locks the pattern.

## 9. Non-goals

- Not numerically-validated constants — v1 locks the architecture; buckets are
  coarse (`docs/calibration_layer.md` §8).
- Not a modification of any raw ETL row — the lift is on the fly.
- Not a fix of hyperon's underlying panic — we scope *around* it.
- Not a general multi-row PLN revision or a faster sort — both are §8 follow-ups.
- The calibration/runtime *layer* itself is just the entry point
  (`run_drugage_ranking`); the chat-app routing that reaches it is a thin
  consumer added on top (§8.5, `pln_chat/core/drugage_router.py`), not part of
  this layer's inference logic.
