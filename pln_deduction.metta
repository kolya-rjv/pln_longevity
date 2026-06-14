;; pln_deduction.metta
;; First PLN inference layer: transitive deduction with uncertainty
;; propagation over truth-valued links.                         [v0]
;;
;; This is the keystone that turns the otherwise-inert knowledge base "live":
;; it consumes the (stv s c) truth values produced by the calibration layer
;; (and curated mechanistic priors) and CHAINS them, propagating both strength
;; and confidence so a multi-step conclusion carries appropriately reduced
;; confidence. This is the capability the project exists to demonstrate
;; (pipeline.md Demos 2 & 3: intervention ranking + novel cross-source
;; inference).
;;
;; Design rationale and open questions: docs/deduction_layer.md
;;
;; Load AFTER:
;;   system_types.metta
;;   logical_predicates.metta
;;   epistemic_calibration.metta        (evidence-confidence, used by bridges)
;;   grim_age_core.metta
;;   grim_age_lu2019_evidence.metta     (empirical records, lifted below)
;;   evidence_calibration.metta         (provides calibrate-tv)
;;   mechanistic_bridges.metta          (curated Implication bridge priors)
;;
;; STATUS: v0, verified against hyperon 0.2.10 (smoke tests in §5). The TV
;; combination rule is deliberately conservative (see §1); the point of v0 is
;; to lock the architecture and make the KB inference-ready so the rule can be
;; tuned against real pipeline behaviour (calibration_layer.md §8), not to be
;; the final numerically-tuned PLN formula.

;; ================================================================
;; 0. Canonical truth-valued link  (resolves calibration open-Q#1)
;; ================================================================
;; The calibration layer left the "how is (stv s c) attached to a link?"
;; question open, to be settled together with the first inference rule. This is
;; that rule, so we settle it: the inference layer chains over a single,
;; dedicated, TV-bearing directed link
;;
;;     (Implication <from> <to> (stv s c))
;;
;; Why a dedicated link rather than annotating the raw predicates:
;;   - The raw evidence/ETL atoms stay UNTOUCHED and TV-free (the calibration
;;     layer's invariant — measurements remain immutable and auditable).
;;   - Deduction stays generic: it matches ONE link shape regardless of whether
;;     a fact came from GrimAge, hallmarks, DrugAge, or an LLM extraction.
;;   - The endpoints are plain Atoms, so any concept can be chained.
;; Domain predicates (Predicts, Causes, …) are *lifted* into this canonical form
;; (see §2); the raw atoms are never rewritten.
(: Implication (-> Atom Atom Atom Atom))

;; ================================================================
;; 1. Truth-value combination for a two-link chain        [STUB: v0]
;; ================================================================
;; PLN deduction A->B, B->C  |-  A->C.
;;
;; strength:   v0 uses the conservative independence product  s = sAB * sBC.
;;             The full PLN deduction formula also folds in the node base rates
;;             P(B), P(C):  sAC = sAB*sBC + (1-sAB)*(sC - sB*sBC)/(1 - sB).
;;             We don't yet store node strengths, so v0 omits that term; adding
;;             base rates is the natural v1 (docs/deduction_layer.md).
;; confidence: each hop multiplies confidences and applies a fixed per-hop
;;             discount, so confidence necessarily DECAYS with chain length
;;             ("uncertainty increases with chain length", pipeline.md Table 2).
;;
;; chain-discount is factored out so the whole tuning loop (calibration_layer.md
;; §8 — "does confidence vanish too fast with length?") is a one-line change.
(= (chain-discount) 0.9)

(= (combine-tv (stv $sab $cab) (stv $sbc $cbc))
   (stv (* $sab $sbc)
        (* (* $cab $cbc) (chain-discount))))

;; ================================================================
;; 2. Direct-link truth value  (the "lift")
;; ================================================================
;; A single directed step A->C gets its TV from either source, normalised to
;; one return shape (stv s c). The space $s is threaded explicitly — a bare
;; (match &self ...) inside this imported module would read only this module's
;; space (the cross-module &self trap documented in evidence_calibration.metta
;; §10.1 / calibration_layer.md §10). Callers pass the top-level &self.
;;
;;   (a) a curated mechanistic Implication prior (mechanistic_bridges.metta);
;;   (b) a calibrated empirical association (HasPredictor/HasOutcome record),
;;       whose TV is DERIVED on the fly via calibrate-tv — never hand-typed.
(= (link-tv $s $a $c)
   (match $s (Implication $a $c $tv) $tv))
(= (link-tv $s $a $c)
   (match $s (, (HasPredictor $r $a) (HasOutcome $r $c))
          (calibrate-tv $s $r)))

;; ================================================================
;; 3. Transitive inference with uncertainty propagation
;; ================================================================
;; (infer space A C) returns the propagated (stv s c) for every derivation of
;; A->C, attenuating confidence per hop. It is non-deterministic: one result
;; per distinct path (none if A and C are unconnected — no false chains).
;;
;; Base case:       any direct link A->C (curated or calibrated-empirical).
;; Transitive case: step through one curated Implication bridge A->B, then
;;                  recurse B->C. v0 chains THROUGH curated bridges and bottoms
;;                  out at a direct link; stepping through empirical links
;;                  mid-chain is a documented v1 extension.
(= (infer $s $a $c) (link-tv $s $a $c))
(= (infer $s $a $c)
   (match $s (Implication $a $b $tvab)
          (combine-tv $tvab (infer $s $b $c))))

;; ================================================================
;; 4. Provenance: the auditable inference chain
;; ================================================================
;; Same recursion as `infer`, but also returns the node path, so every
;; conclusion traces to its intermediate steps — the auditability that
;; LLMs cannot provide (pipeline.md §1.2 "Complete provenance"). Returns
;;   (Chain (n0 n1 ... nk) (stv s c)).
(= (explain $s $a $c)
   (match $s (Implication $a $c $tv) (Chain ($a $c) $tv)))
(= (explain $s $a $c)
   (match $s (, (HasPredictor $r $a) (HasOutcome $r $c))
          (Chain ($a $c) (calibrate-tv $s $r))))
(= (explain $s $a $c)
   (match $s (Implication $a $b $tvab)
          (prepend-node $a $tvab (explain $s $b $c))))

(= (prepend-node $a $tvab (Chain $nodes $tvrest))
   (Chain (cons-atom $a $nodes) (combine-tv $tvab $tvrest)))

;; ================================================================
;; 5. Smoke-test queries (verified, hyperon 0.2.10)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;;             mechanistic_bridges pln_deduction
;; (Confidences below are the intended exact values; live float output may carry
;;  trailing noise, e.g. 0.29089125 prints as 0.29089125000000005. Tests in
;;  tests/test_pln_deduction.py compare with tolerance.)
;;
;; Direct, calibrated-empirical link (Lu 2019, HR 1.31):
;; !(link-tv &self DNAmPAI1 CoronaryHeartDisease)        ; (stv 0.6 0.85)
;;
;; One-hop chain through a curated bridge into Lu 2019 evidence:
;; !(infer &self SASP CoronaryHeartDisease)              ; (stv 0.42 0.49725)
;;
;; NOVEL cross-source inference — stated by NO single source
;; (Hallmarks concept -> curated SASP/PAI-1 mechanism -> Lu 2019 GrimAge):
;; !(infer &self CellularSenescence CoronaryHeartDisease)
;;     ; (stv 0.357 0.29089125)   strength & confidence attenuated over 3 hops
;;
;; Auditable derivation (path + propagated TV):
;; !(explain &self CellularSenescence CoronaryHeartDisease)
;;     ; (Chain (CellularSenescence SASP DNAmPAI1 CoronaryHeartDisease)
;;     ;        (stv 0.357 0.29089125))
;;
;; No false chains when unconnected:
;; !(infer &self CellularSenescence AllCauseMortality)   ; (empty)
