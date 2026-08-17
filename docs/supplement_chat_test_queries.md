# Supplement Recommendation — Chat Test Queries (Demo 6, v1)

Ten natural-language chat queries that exercise the supplement-recommendation layer
(`supplement_evidence.metta` + `pln_supplement_recommendation.metta`) end-to-end
through the **real** pipeline: `translate()` → `validate()` → `run_query()` against
the live `hyperon 0.2.10` interpreter, loading the exact `_ALL_KB_PATHS` set the
Gradio app loads (every repo-root `*.metta` under `PLN_MAX_KB_FILE_BYTES` — 22 files
here, the two Demo-6 files included).

Reproduce:

```
PLN_RUNTIME_AVAILABLE=true python scripts/supplement_chat_demo.py
```

The **ACTUAL** blocks below are transcribed verbatim from that run — real hyperon
output, no hypothetical values. (Floats carry trailing precision noise; the
regression tests in `tests/test_supplement_recommendation.py` compare with tolerance.)
In this sandbox `OPENAI_API_KEY` is unset, so `translate()` reports its "key not set"
error and the driver falls back to the reference MeTTa each few-shot targets (labelled
in the script); every query still `validate()`s clean and executes for real. The
queries are deliberately non-trivial and spread across the demo's headline behaviors.

---

### Q1 — full tiered recommendation (the deliverable)
**NL:** "What supplements should Patient001 consider, and how strong is the evidence for each?"
**MeTTa:** `(recommend-supplements-patient &self Patient001)`
**Expected:** Tier1 Omega3 (MultipleHumanTrials); Tier2 NMN then Fisetin; NotRecommended Resveratrol; Berberine **omitted** (metabolic, irrelevant); no interactions (no meds).
**Actual:**
```
(SupplementRecommendation Patient001
  (Tier1HighConfidence ((SuppRec Omega3 (tier Tier1HighConfidence) (score 0.4326…)
      (evidence MultipleHumanTrials 0.85) (safety GenerallyWellTolerated 1.0)
      (Targets (CRP DNAmGDF15)))))
  (Tier2Promising ((SuppRec NMN (tier Tier2Promising) (score 0.16892)
      (evidence SingleHumanTrial 0.7) (safety GenerallyWellTolerated 1.0)
      (Targets (DNAmGDF15)))
    (SuppRec Fisetin (tier Tier2Promising) (score 0.13434)
      (evidence AnimalStudies_Single 0.5) (safety GenerallyWellTolerated 1.0)
      (Targets (CRP DNAmGDF15 DNAmPAI1)))))
  (NotRecommended ((SuppRec Resveratrol (tier NotRecommended) (score 0.07234)
      (evidence ITP_Negative 0.9) (safety GenerallyWellTolerated 1.0)
      (Targets (CRP DNAmGDF15 DNAmPAI1)))))
  (Interactions ()))
```

### Q2 — the negative-evidence veto (the killer)
**NL:** "Should Patient001 take resveratrol for their elevated GrimAge?"
**MeTTa:** `(supplement-for-patient &self Patient001 Resveratrol)`
**Expected:** Resveratrol IS relevant by mechanism (`Targets` the senescence markers) but `ITP_Negative` → `tier NotRecommended`. Declined on the negative trial, not a missing mechanism — the restraint an LLM lacks.
**Actual:**
```
(SuppRec Resveratrol (tier NotRecommended) (score 0.07234)
  (evidence ITP_Negative 0.9) (safety GenerallyWellTolerated 1.0)
  (Targets (CRP DNAmGDF15 DNAmPAI1)))
```

### Q3 — personalization + interaction flag
**NL:** "Recommend supplements for Patient002 and flag any interactions with their medications."
**MeTTa:** `(recommend-supplements-patient &self Patient002)`
**Expected:** Berberine → Tier1 (metabolic markers); the senescence supplements omitted; the Berberine↔Metformin interaction **flagged** against Patient002's metformin.
**Actual:**
```
(SupplementRecommendation Patient002
  (Tier1HighConfidence ((SuppRec Berberine (tier Tier1HighConfidence) (score 0.38786)
      (evidence MultipleHumanTrials 0.85) (safety UseWithCaution 0.8)
      (Targets (HbA1c FastingGlucose)))))
  (Tier2Promising ()) (NotRecommended ())
  (Interactions ((InteractionFlag Berberine Metformin
      "Shared AMPK activation — additive glucose lowering; monitor and consult MD"))))
```

### Q4 — the same pool, different advice (personalization reorders)
**NL:** "Does the same supplement pool give Patient002 different advice than Patient001?"
**MeTTa:** `(recommend-supplements-patient &self Patient002)` and `(recommend-supplements-patient &self Patient001)`
**Expected:** Same five candidates, **different** output — Patient002 gets Berberine (metabolic), Patient001 gets Omega3/Fisetin/NMN (senescence/inflammation). Relevance reorders, it does not merely amplify.
**Actual:** two results — the Patient002 blob from Q3 and the Patient001 blob from Q1, side by side (verbatim in the script output).

### Q5 — an irrelevant supplement is omitted, not mis-placed
**NL:** "Would berberine help Patient001?"
**MeTTa:** `(supplement-for-patient &self Patient001 Berberine)`
**Expected:** Berberine acts on the metabolic axis; Patient001 has no metabolic marker elevated → irrelevant → **omitted** (empty), never a fabricated placement.
**Actual:** `(empty — no result)`

### Q6 — provenance: which markers a recommendation targets
**NL:** "Which of Patient001's elevated markers would omega-3 actually address?"
**MeTTa:** `(supplement-for-patient &self Patient001 Omega3)`
**Expected:** Omega3 is Tier1 and its `(Targets …)` names CRP and DNAmGDF15 — the inflammation markers it protectively reaches. Auditable.
**Actual:**
```
(SuppRec Omega3 (tier Tier1HighConfidence) (score 0.4326…)
  (evidence MultipleHumanTrials 0.85) (safety GenerallyWellTolerated 1.0)
  (Targets (CRP DNAmGDF15)))
```

### Q7 — explicit candidate pool
**NL:** "Rank omega-3, fisetin and NMN for Patient001 by how well they fit their biomarker profile."
**MeTTa:** `(recommend-supplements &self Patient001 (Omega3 Fisetin NMN))`
**Expected:** Omega3 Tier1; NMN, Fisetin Tier2 (sorted by score). Each carries evidence tier + confidence + targeted markers. (Resveratrol not in the pool → NotRecommended block empty.)
**Actual:**
```
(SupplementRecommendation Patient001
  (Tier1HighConfidence ((SuppRec Omega3 … (Targets (CRP DNAmGDF15)))))
  (Tier2Promising ((SuppRec NMN … (Targets (DNAmGDF15)))
                   (SuppRec Fisetin … (Targets (CRP DNAmGDF15 DNAmPAI1)))))
  (NotRecommended ()) (Interactions ()))
```

### Q8 — supplement vs pharmaceutical confidence (Layer-4 honesty)
**NL:** "Is the evidence for Fisetin as strong as for a pharmaceutical like metformin?"
**MeTTa:** `(supplement-for-patient &self Patient001 Fisetin)`
**Expected:** Fisetin's evidence is `AnimalStudies_Single` (conf 0.5), well below a pharmaceutical's trial-grade confidence — supplements are tiered lower, not conflated with RCT evidence.
**Actual:**
```
(SuppRec Fisetin (tier Tier2Promising) (score 0.13434)
  (evidence AnimalStudies_Single 0.5) (safety GenerallyWellTolerated 1.0)
  (Targets (CRP DNAmGDF15 DNAmPAI1)))
```

### Q9 — metabolic-patient plan with provenance
**NL:** "Give Patient002 a supplement plan and show which markers each recommendation targets."
**MeTTa:** `(recommend-supplements-patient &self Patient002)`
**Expected:** Berberine Tier1 with `(Targets (FastingGlucose HbA1c))`; Interactions flags Berberine↔Metformin. A wholly different plan than Patient001's.
**Actual:** identical to Q3's blob (Berberine Tier1, `(Targets (HbA1c FastingGlucose))`, the Berberine–Metformin flag).

### Q10 — head-to-head within a tier
**NL:** "Between fisetin and NMN, which fits Patient001's profile better?"
**MeTTa:** `(recommend-supplements &self Patient001 (Fisetin NMN))`
**Expected:** Both Tier2, sorted by personalized score: NMN (single mito hop → higher-confidence transmission to DNAmGDF15) ranks above Fisetin; each lists its targeted markers.
**Actual:**
```
(SupplementRecommendation Patient001
  (Tier1HighConfidence ())
  (Tier2Promising ((SuppRec NMN (tier Tier2Promising) (score 0.16892)
        (evidence SingleHumanTrial 0.7) (safety GenerallyWellTolerated 1.0)
        (Targets (DNAmGDF15)))
      (SuppRec Fisetin (tier Tier2Promising) (score 0.13434)
        (evidence AnimalStudies_Single 0.5) (safety GenerallyWellTolerated 1.0)
        (Targets (CRP DNAmGDF15 DNAmPAI1)))))
  (NotRecommended ()) (Interactions ()))
```

---

## What these show that an LLM cannot

1. **Q2/Q1 — the veto.** Resveratrol has a senescence-clearing mechanism relevant to
   Patient001's markers (an LLM recommends it on that basis) — PLN puts it in
   `NotRecommended` on the ITP-negative trial, regardless of the mechanism.
2. **Q3/Q4/Q5 — personalization reorders.** The identical five-supplement pool yields
   Berberine for the metabolic Patient002 and omits it for Patient001; the senescence
   supplements do the reverse. Not amplification — reordering.
3. **Q3/Q9 — interactions.** The Berberine–Metformin flag fires only for the patient
   actually on metformin.
4. **Q6/Q8 — provenance + calibrated honesty.** Every recommendation names the exact
   markers it targets and carries an evidence confidence tiered below RCT-grade.
5. **Determinism.** Identical queries return byte-identical output (Q1 vs the Q4
   Patient001 blob).
