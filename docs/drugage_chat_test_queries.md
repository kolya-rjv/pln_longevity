# DrugAge lifespan→mortality ranking — 10 chat test queries (executed)

**What this is.** Ten natural-language chat queries that exercise the new
`rank-drugage-lifespan` route (docs/etl_inference_wiring.md §8.5). Each was run
through the **real** routing + inference pipeline against `hyperon 0.2.10` — the
outputs below are captured, not hypothetical.

**Pipeline exercised.**

```
NL --(translate; few-shot-pinned, low-temp)--> metta_query
   --parse_drugage_query--------------------->  [compounds]   (core/drugage_router.py)
   --route_drugage_ranking-------------------->  run_drugage_ranking  (scoped space)
   --format_bot_response---------------------->  chat output
```

`translate()` is an OpenAI call (needs a key), but it is a thin mapping pinned by
the three `rank-drugage-lifespan` few-shots in
`pln_chat/prompts/few_shot_examples.json`, so the `metta_query` shown for each is
what it emits. Everything downstream — the **routing decision** and the **scoped
DrugAge inference** — is executed here for real.

**Reproduce.**

```bash
pip install hyperon==0.2.10 pandas python-dotenv
bash scripts/run_etl.sh                                       # -> build/drugage_etl.metta (3,423 rows)
PLN_RUNTIME_AVAILABLE=true python scripts/drive_drugage_chat_queries.py
```

Scores below are over the **full** `build/drugage_etl.metta`. `best_per_compound`
(default) picks one representative row per compound — the highest evidence tier
(ITP > significant > unreported > null), tie-broken by the *median* change — so
the exact strength can differ from the 5-row fixture in the docs while the
ordering and the qualitative reads hold. Deterministic across runs.

> Notation: `score` is `rank-score` (higher = more protective on Mortality);
> `Neg` = beneficial (extends life → lowers mortality), `Pos` = harmful;
> `stv s c` = calibrated strength (magnitude) and confidence (evidence weight).

---

## Q1 — Ranking a varied real-compound set

- **NL:** *"Between rapamycin, curcumin and metformin, which one most lowers mortality by extending lifespan?"*
- **Emits:** `(rank-drugage-lifespan (Rapamycin Curcumin Metformin))` → routes ✅
- **Expected:** all three are real ITP DrugAge compounds; ranked most-protective
  first. Rapamycin (replicated positive) leads; metformin (ITP null) ~0.
- **Actual:**
  ```
  Rapamycin  score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, Mus musculus, +11%, ITP, Significant
  Curcumin   score=+0.1057 (Neg stv 0.130 0.810)   ← PMID_22451473, Mus musculus,  +3%, ITP, NotSignificant
  Metformin  score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_27312235, Mus musculus,  +0%, ITP, NotSignificant
  ```

## Q2 — The ITP-negative down-weight (the headline behavior)

- **NL:** *"Everyone hypes resveratrol — rank it honestly against rapamycin and metformin for real lifespan and mortality benefit."*
- **Emits:** `(rank-drugage-lifespan (Resveratrol Rapamycin Metformin))` → routes ✅
- **Expected:** resveratrol AND metformin both failed the ITP, so despite **high**
  (ITP-grade, 0.81) confidence their near-zero **strength** sinks them to ~0 — the
  honest "confidently no mouse-lifespan effect", far below the replicated positive
  rapamycin. Not "we're unsure".
- **Actual:**
  ```
  Rapamycin    score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, +11%, ITP, Significant
  Resveratrol  score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_20974732,  +0%, ITP, NotSignificant
  Metformin    score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_27312235,  +0%, ITP, NotSignificant
  ```
  Confidence stays 0.81 (an ITP result); strength is what collapses. ✅

## Q3 — Magnitude ⇆ confidence trade-off (big-effect worm vs small-effect ITP mouse)

- **NL:** *"A huge lifespan gain was reported for trimethadione in worms — does that outweigh acarbose's smaller but ITP-grade mouse result?"*
- **Emits:** `(rank-drugage-lifespan (Trimethadione Acarbose))` → routes ✅
- **Expected:** trimethadione (+39% worm) carries **low** confidence (invertebrate
  tier 0.315); acarbose (+5% mouse) is **high** confidence (ITP 0.81). The ranking
  weighs magnitude **and** confidence, so the big worm effect edges out — debatable
  but honest, never sign alone.
- **Actual:**
  ```
  Trimethadione  score=+0.2082 (Neg stv 0.661 0.315)   ← PMID_15653505, C. elegans, +39%, Significant
  Acarbose       score=+0.1620 (Neg stv 0.200 0.810)   ← PMID_30688027, Mus musculus, +5%, ITP, Significant
  ```
  `0.661·0.315 > 0.200·0.810` → worm wins on magnitude despite lower confidence. ✅

## Q4 — Single-compound lift / effect lookup

- **NL:** *"On its own, how much does rapamycin lower mortality through its lifespan effect?"*
- **Emits:** `(rank-drugage-lifespan (Rapamycin))` → routes ✅
- **Expected:** one-compound form is valid; returns rapamycin's signed, calibrated
  effect with its backing study (a `drugage-effect`-for-one-compound read).
- **Actual:**
  ```
  Rapamycin  score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, Mus musculus, +11%, ITP, Significant
  ```

## Q5 — An unconnected / absent compound (omitted, not mis-ranked)

- **NL:** *"Rank dasatinib, rapamycin and acarbose by lifespan-driven mortality benefit."*
- **Emits:** `(rank-drugage-lifespan (Dasatinib Rapamycin Acarbose))` → routes ✅
- **Expected:** dasatinib has **no** DrugAge lifespan row; it must be **omitted**
  from the ranking (not invented at 0 or mis-placed), and surfaced as such.
- **Actual:**
  ```
  Rapamycin  score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, +11%, ITP, Significant
  Acarbose   score=+0.1620 (Neg stv 0.200 0.810)   ← PMID_30688027,  +5%, ITP, Significant
  Omitted (no DrugAge lifespan rows matched): Dasatinib
  ```
  Dasatinib is absent from the ranking and explicitly noted. ✅

## Q6 — Provenance (which PMID backs this)

- **NL:** *"Rank metformin and acarbose for lifespan benefit and show which study each ranking rests on."*
- **Emits:** `(rank-drugage-lifespan (Metformin Acarbose))` → routes ✅
- **Expected:** every ranked compound rides with its backing `PMID_…` (read back
  from the untouched raw row's `ReportedIn`), plus species / % change / tier.
- **Actual:**
  ```
  Acarbose   score=+0.1620 (Neg stv 0.200 0.810)   ← PMID_30688027, Mus musculus, +5%, ITP, Significant
  Metformin  score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_27312235, Mus musculus, +0%, ITP, NotSignificant
  ```
  Acarbose ⇒ PMID 30688027, metformin ⇒ PMID 27312235 — the audit trail. ✅

## Q7 — Full canonical spread (leaderboard)

- **NL:** *"Give me a lifespan-benefit leaderboard across rapamycin, resveratrol, acarbose, metformin and trimethadione."*
- **Emits:** `(rank-drugage-lifespan (Rapamycin Resveratrol Acarbose Metformin Trimethadione))` → routes ✅
- **Expected:** the whole spread in one shot — replicated positive on top, the big
  low-confidence worm effect and the small high-confidence mouse effect in the
  middle, both ITP nulls at the bottom.
- **Actual:**
  ```
  Rapamycin      score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, +11%, ITP, Significant
  Trimethadione  score=+0.2082 (Neg stv 0.661 0.315)   ← PMID_15653505, C. elegans, +39%, Significant
  Acarbose       score=+0.1620 (Neg stv 0.200 0.810)   ← PMID_30688027,  +5%, ITP, Significant
  Resveratrol    score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_20974732,  +0%, ITP, NotSignificant
  Metformin      score=+0.0000 (Neg stv 0.000 0.810)   ← PMID_27312235,  +0%, ITP, NotSignificant
  ```
  ~1.8 s for 5 compounds — the O(n²) sort is why the pool stays a handful (§7). ✅

## Q8 — Casing robustness + a different compound set

- **NL:** *"compare CAFFEINE, aspirin and Taurine on lifespan and mortality"*
- **Emits:** `(rank-drugage-lifespan (CAFFEINE aspirin Taurine))` → routes ✅
- **Expected:** mixed casing (`CAFFEINE`, `aspirin`) still matches the canonical
  `Caffeine`, `Aspirin`, `Taurine` rows — the selector normalizes
  case/separators. Two worm effects outrank a tiny ITP mouse one (magnitude vs
  confidence again).
- **Actual:**
  ```
  Caffeine  score=+0.0955 (Neg stv 0.303 0.315)   ← PMID_24764514, C. elegans, +9%, Significant
  Taurine   score=+0.0727 (Neg stv 0.231 0.315)   ← PMID_25643626, C. elegans, +6%, Significant
  Aspirin   score=+0.0386 (Neg stv 0.048 0.810)   ← PMID_30916479, Mus musculus, +1%, ITP, NotSignificant
  ```
  Canonical names resolved from the loose input; ranking correct. ✅

## Q9 — Hyped senolytics across mixed evidence tiers (bonus: a net-harmful null)

- **NL:** *"Rank the senolytics fisetin, quercetin and spermidine by lifespan-extension benefit."*
- **Emits:** `(rank-drugage-lifespan (Fisetin Quercetin Spermidine))` → routes ✅
- **Expected:** these span tiers — spermidine has a strong *significant mouse*
  result, quercetin a *worm* result, fisetin an *ITP mouse null that came in
  slightly negative*. Fisetin's representative ITP row is −1%, so it lifts to
  `Pos` (life-shortening direction) and ranks **below zero** — the honest read
  that a hyped senolytic has no ITP lifespan benefit, not a fabricated positive.
- **Actual:**
  ```
  Spermidine  score=+0.2455 (Neg stv 0.545 0.450)   ← PMID_28386016, Mus musculus, +24%, Significant
  Quercetin   score=+0.1050 (Neg stv 0.333 0.315)   ← PMID_19043800, C. elegans, +10%, Significant
  Fisetin     score=-0.0386 (Pos stv 0.048 0.810)   ← PMID_38041783, Mus musculus, -1%, ITP, NotSignificant
  ```
  Fisetin sits **below** the protective compounds at a negative score — the
  sign trap working as designed (a life-shortening direction ⇒ Pos ⇒ harmful). ✅

## Q10 — Several absent compounds mixed with present ones

- **NL:** *"Which of sildenafil, lithium, aspirin and rapamycin actually has lifespan evidence, ranked?"*
- **Emits:** `(rank-drugage-lifespan (Sildenafil Lithium Aspirin Rapamycin))` → routes ✅
- **Expected:** sildenafil and lithium have no DrugAge lifespan rows → both
  omitted; the two with evidence rank normally. Answers the literal question
  ("which *actually* has evidence").
- **Actual:**
  ```
  Rapamycin  score=+0.2874 (Neg stv 0.355 0.810)   ← PMID_33145977, +11%, ITP, Significant
  Aspirin    score=+0.0386 (Neg stv 0.048 0.810)   ← PMID_30916479,  +1%, ITP, NotSignificant
  Omitted (no DrugAge lifespan rows matched): Sildenafil, Lithium
  ```
  Only the evidence-backed compounds are ranked; the rest are named as omitted. ✅

---

## Coverage summary

| Facet (from the task) | Query |
|---|---|
| Real-compound ranking, varied sets | Q1, Q7, Q8, Q9 |
| ITP-negative down-weight (→ ~0) | Q2, Q7 |
| Magnitude ⇆ confidence trade-off | Q3, Q8 |
| Single-compound lift/effect lookup | Q4 |
| Absent / unconnected compound omitted | Q5, Q10 |
| Provenance (which PMID) | Q6 (and the `←` trail on every row) |
| Casing / name normalization | Q8 |
| Net-harmful sign (life-shortener ⇒ Pos) | Q9 |

All ten route to `run_drugage_ranking` (never the generic `_ALL_KB_PATHS` space),
run without tripping the hyperon panic (scoped space), and are deterministic.
The routing decision + ITP-negative-to-~0 behavior are additionally asserted in
`tests/test_drugage_chat_routing.py`.
