# Abductive Diagnosis — Demo 1 (v1)

**Status:** v1, verified against `hyperon 0.2.10`. Implementation:
`pln_abductive_diagnosis.metta`; cause→biomarker bridges added to
`mechanistic_bridges.metta`; regression tests in `tests/test_pln_deduction.py`.
**Scope:** pipeline.md Demo 1 (§4.1) — given a patient's observed (elevated)
biomarkers, return **ranked hypotheses** for the upstream causes, each with a
truth value and the markers it explains.

> **Dependency.** Pure consumer of the deduction layer (`pln_deduction.metta` →
> `infer`) and the curated bridges. Load last (see the file header). Adds no new
> inference power — it *reuses* forward deduction and reads it backward.

---

## 1. Deduction forward, abduction backward

Demo 2 ran the causal graph forward (intervention → outcome). Demo 1 runs the
**same graph backward**: a hypothesis cause `H` *explains* an observed marker `O`
iff forward deduction `H → O` is **positive** — i.e. `H`, when elevated, *raises*
`O`, which is exactly what an observed *elevated* `O` is evidence for. So
abduction is built entirely on `infer`:

```metta
(explains-obs space H O) := keep (infer space H O) only when it is (signed Pos …)
```

A `Neg` chain (the cause would *lower* the marker) fails the match — a cause is
never credited for a finding it would reduce — and `infer` already returns
nothing for unconnected pairs, so there are no false explanations.

## 2. Scoring & ranking — the abductive parsimony principle

For each candidate cause we collapse everything it explains, then aggregate:

| Field | Rule | Meaning |
|---|---|---|
| **coverage** | count of explained observations | how much of the picture this one cause accounts for |
| **mass** | Σ (strength · confidence) | total supporting evidence |
| reported strength `s` | max over explanations | the strongest single causal pathway |
| reported confidence `c` | noisy-OR `1 − Π(1 − cᵢ)` | confidence the cause is present, rising with each independent corroborating marker |
| SupportedBy | the explained markers | provenance — *which* findings it accounts for |

Hypotheses are ranked by **coverage first, then mass**: one root cause explaining
many findings beats causes that each explain a subset. That parsimony — *prefer
the single explanation that covers the most* — is the diagnostic judgment an LLM
states only qualitatively; here it is a deterministic function of calibrated
evidence.

## 3. The headline demo

Synthetic patient: **elevated DNAmPAI1, DNAmGDF15, CRP**. Candidate causes: three
hallmarks. The cause→biomarker bridges (in `mechanistic_bridges.metta`, grounded
in pipeline.md App. A.2/A.6) give:

```metta
!(diagnose &self
    (CellularSenescence MitochondrialDysfunction ChronicInflammation)
    (DNAmPAI1 DNAmGDF15 CRP))
=>
( (Hypothesis CellularSenescence       (stv 0.595 0.63) 3 0.459
      (SupportedBy (DNAmPAI1 CRP DNAmGDF15)))      ; explains ALL three findings
  (Hypothesis ChronicInflammation      (stv 0.80  0.88) 2 0.943
      (SupportedBy (CRP DNAmGDF15)))               ; explains two
  (Hypothesis MitochondrialDysfunction (stv 0.75  0.65) 1 0.488
      (SupportedBy (DNAmGDF15))) )                 ; explains the lone GDF15 signal
```

Three reads no LLM produces from first principles:

1. **CellularSenescence wins on coverage** — via the SASP it reaches DNAmPAI1
   (thrombosis surrogate), CRP and DNAmGDF15 (through chronic inflammation), so
   one upstream cause accounts for the entire presentation.
2. **ChronicInflammation is second and is itself downstream of senescence** — a
   real mediator that explains two findings; abduction surfaces it without
   conflating it with the root cause.
3. **MitochondrialDysfunction explains only the isolated GDF15 signal** — it is
   not dismissed, just ranked by how little of the picture it covers.

Pass any cause with no chain to the findings (e.g. `GenomicInstability`) and it
is simply **omitted** — abduction declines to place it rather than inventing a
plausible-sounding rank.

## 4. Implementation notes (hyperon 0.2.10)

- `superpose`/`collapse` fan the observation set through `infer` and gather the
  per-hypothesis evidence; the folds (coverage/mass/max-strength/noisy-OR/
  marker-list) and the coverage-then-mass insertion sort are all `let`/`let*`-
  forced, because hyperon grounded ops (`car-atom`/`cdr-atom`/`cons-atom`/`==`/
  `+`/`*`) do not evaluate their arguments. (Same discipline as the deduction and
  ranking layers.)
- Coverage prints as a float (`3.0`) — hyperon `+` promotes to float; harmless,
  the ranking only compares it.

## 5. Open questions / next increments

1. **Negative / contradicting evidence** — a hypothesis whose chain would *lower*
   an observed-elevated marker is currently just not credited; it could instead
   *reduce* its support (evidence against). Likewise an *unexpectedly normal*
   marker that a hypothesis predicts should be elevated.
2. **Base rates** — without node priors (deduction_layer.md §7.2) the score is a
   coverage/evidence proxy, not a calibrated posterior `P(cause | findings)`.
   Adding base rates upgrades this to a proper Bayesian abduction.
3. **Patient grounding** — observations are passed explicitly; wiring them to a
   real NHANES methylation + blood profile (still-absent layer) turns the demo
   from synthetic to personalized.
4. **Auto-candidate causes** — default the hypothesis set to all twelve hallmarks
   (unconnected ones already drop out) instead of an explicit list.
