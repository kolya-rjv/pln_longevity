;; pln_deduction.metta
;; PLN inference layer: transitive deduction with uncertainty propagation
;; over SIGNED truth-valued links.                              [v1]
;;
;; This is the keystone that turns the otherwise-inert knowledge base "live":
;; it consumes the (stv s c) truth values produced by the calibration layer
;; (and curated mechanistic priors) and CHAINS them, propagating strength,
;; confidence AND direction so a multi-step conclusion carries appropriately
;; reduced confidence and the correct sign. This is the capability the project
;; exists to demonstrate (pipeline.md Demos 2 & 3: intervention ranking + novel
;; cross-source inference).
;;
;; v1 adds polarity (Pos/Neg) on top of v0's magnitude/confidence math, so an
;; intervention that *reduces* an outcome (a protective effect) chains to a net
;; Neg result instead of being conflated with a harmful one. This is what
;; intervention ranking needs. (Represents effect direction — the "Direction"
;; point in calibration_layer.md §3.1 — and gives protective/negative effects a
;; sign, calibration open-Q#5.) The v0 magnitude rule — combine-tv /
;; chain-discount — is unchanged.
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
;;   mechanistic_bridges.metta          (curated Effect bridge priors)
;;
;; STATUS: v1, verified against hyperon 0.2.10 (smoke tests in §5; regression-
;; guarded in tests/test_pln_deduction.py). The magnitude rule is deliberately
;; conservative (see §1) — v0/v1 lock the architecture and make the KB
;; inference-ready so the policy can be tuned against real pipeline behaviour
;; (calibration_layer.md §8), not to be the final numerically-tuned PLN formula.

;; ================================================================
;; 0. Canonical SIGNED truth-valued link  (resolves calibration open-Q#1)
;; ================================================================
;; The inference layer chains over one dedicated, TV-bearing directed link
;;
;;     (Effect <from> <to> <sign> (stv s c))      sign ∈ { Pos, Neg }
;;
;;   Pos = <from> raises / increases <to>;  Neg = <from> reduces / lowers <to>.
;;   strength s   = MAGNITUDE of the effect (how strong), independent of sign;
;;   confidence c = evidence weight (as in calibration).
;;
;; Why a dedicated link rather than annotating the raw predicates:
;;   - The raw evidence/ETL atoms stay UNTOUCHED and TV-free (the calibration
;;     layer's invariant — measurements remain immutable and auditable).
;;   - Deduction stays generic: it matches ONE link shape regardless of whether
;;     a fact came from GrimAge, hallmarks, DrugAge, or an LLM extraction.
;;   - Endpoints are plain Atoms, so any concept can be chained.
;; Domain predicates (Predicts, Causes, …) are *lifted* into this canonical form
;; (see §2); the raw atoms are never rewritten.
(: Effect (-> Atom Atom Atom Atom Atom))

;; Effect-sign enumeration.
(: EffectSign Type)
(: Pos EffectSign)
(: Neg EffectSign)

;; ================================================================
;; 1. Combination for a two-link chain        [magnitude: STUB v0]
;; ================================================================
;; PLN deduction A->B, B->C  |-  A->C, split into independent axes:
;;
;; SIGN:       the net direction is the product of the hop signs — a protective
;;             step (Neg) into a harmful axis (Pos…) yields a net protective
;;             (Neg) conclusion. (Neg·Neg = Pos: reducing a reducer raises.)
;; strength:   v0 conservative independence product  s = sAB * sBC. The full PLN
;;             deduction formula also folds in node base rates P(B), P(C):
;;             sAC = sAB*sBC + (1-sAB)*(sC - sB*sBC)/(1 - sB). We don't store
;;             node strengths yet, so v0 omits that term (natural v1+).
;; confidence: confidences multiply with a fixed per-hop discount, so confidence
;;             necessarily DECAYS with chain length ("uncertainty increases with
;;             chain length", pipeline.md Table 2).
;;
;; chain-discount is factored out so the whole tuning loop (calibration_layer.md
;; §8 — "does confidence vanish too fast with length?") is a one-line change.
(= (sign-product Pos Pos) Pos)
(= (sign-product Pos Neg) Neg)
(= (sign-product Neg Pos) Neg)
(= (sign-product Neg Neg) Pos)

(= (chain-discount) 0.9)

(= (combine-tv (stv $sab $cab) (stv $sbc $cbc))
   (stv (* $sab $sbc)
        (* (* $cab $cbc) (chain-discount))))

;; Combine one curated hop (sign + tv) with the recursively-inferred rest.
(= (combine-hop $sgnAB $tvAB (signed $sgnBC $tvBC))
   (signed (sign-product $sgnAB $sgnBC) (combine-tv $tvAB $tvBC)))

;; ================================================================
;; 2. Direct signed effect  (the "lift")
;; ================================================================
;; A single directed step A->C, normalised to one return shape
;; (signed <sign> (stv s c)). The space $s is threaded explicitly — a bare
;; (match &self ...) inside this imported module would read only this module's
;; space (the cross-module &self trap documented in evidence_calibration.metta
;; §10.1 / calibration_layer.md §10). Callers pass the top-level &self.
;;
;;   (a) a curated mechanistic Effect prior (mechanistic_bridges.metta);
;;   (b) a calibrated empirical association (HasPredictor/HasOutcome record):
;;       magnitude is DERIVED on the fly via calibrate-tv (never hand-typed),
;;       and the sign is read off the hazard-ratio direction (HR>1 harmful=Pos,
;;       HR<1 protective=Neg).
(= (hr->sign $hr) (if (< $hr 1.0) Neg Pos))

(= (link-effect $s $a $c)
   (match $s (Effect $a $c $sgn $tv) (signed $sgn $tv)))
(= (link-effect $s $a $c)
   (match $s (, (HasPredictor $r $a) (HasOutcome $r $c) (EffectValue $r $hr))
          (signed (hr->sign $hr) (calibrate-tv $s $r))))

;; ================================================================
;; 3. Transitive inference with uncertainty + sign propagation
;; ================================================================
;; (infer space A C) returns the propagated (signed <sign> (stv s c)) for every
;; derivation of A->C, attenuating confidence per hop. It is non-deterministic:
;; one result per distinct path (none if A and C are unconnected — no false
;; chains).
;;
;; Base case:       any direct link A->C (curated or calibrated-empirical).
;; Transitive case: step through one curated Effect bridge A->B, then recurse
;;                  B->C. v0 chains THROUGH curated bridges and bottoms out at a
;;                  direct link; stepping through empirical links mid-chain is a
;;                  documented v1+ extension.
(= (infer $s $a $c) (link-effect $s $a $c))
(= (infer $s $a $c)
   (match $s (Effect $a $b $sgn $tv)
          (combine-hop $sgn $tv (infer $s $b $c))))

;; ================================================================
;; 4. Provenance: the auditable inference chain
;; ================================================================
;; Same recursion as `infer`, but also returns the node path, so every
;; conclusion traces to its intermediate steps — the auditability that LLMs
;; cannot provide (pipeline.md §1.2 "Complete provenance"). Returns
;;   (Chain (n0 n1 ... nk) (signed <sign> (stv s c))).
(= (explain $s $a $c)
   (match $s (Effect $a $c $sgn $tv) (Chain ($a $c) (signed $sgn $tv))))
(= (explain $s $a $c)
   (match $s (, (HasPredictor $r $a) (HasOutcome $r $c) (EffectValue $r $hr))
          (Chain ($a $c) (signed (hr->sign $hr) (calibrate-tv $s $r)))))
(= (explain $s $a $c)
   (match $s (Effect $a $b $sgn $tv)
          (prepend-node $a $sgn $tv (explain $s $b $c))))

(= (prepend-node $a $sgnAB $tvAB (Chain $nodes (signed $sgnBC $tvBC)))
   (Chain (cons-atom $a $nodes)
          (signed (sign-product $sgnAB $sgnBC) (combine-tv $tvAB $tvBC))))

;; ================================================================
;; 5. Smoke-test queries (verified, hyperon 0.2.10)
;; ================================================================
;; Load order: system_types logical_predicates epistemic_calibration
;;             grim_age_core grim_age_lu2019_evidence evidence_calibration
;;             hallmarks_core hallmarks_lopezotin2023_intervention_evidence
;;             mechanistic_bridges pln_deduction
;; (Confidences below are the intended values; live float output may carry
;;  trailing noise. tests/test_pln_deduction.py compares with tolerance.)
;;
;; Direct, calibrated-empirical link (Lu 2019, HR 1.31, harmful):
;; !(link-effect &self DNAmPAI1 CoronaryHeartDisease)     ; (signed Pos (stv 0.6 0.85))
;;
;; HARMFUL cross-source chain — senescence raises CHD risk (3 hops):
;; !(infer &self CellularSenescence CoronaryHeartDisease)
;;     ; (signed Pos (stv 0.357 0.29089125))
;;
;; PROTECTIVE cross-source chain — the senolytic D+Q *reduces* CHD risk (4 hops).
;; The sign flips at the senolytic hop (Neg·Pos·Pos·Pos = Neg). This is the
;; intervention-ranking signal LLMs cannot quantify (pipeline.md Demo 2):
;; !(infer &self DasatinibPlusQuercetin CoronaryHeartDisease)
;;     ; (signed Neg (stv 0.2499 0.17017138125))
;;
;; Auditable derivation (path + signed propagated TV):
;; !(explain &self DasatinibPlusQuercetin CoronaryHeartDisease)
;;     ; (Chain (DasatinibPlusQuercetin CellularSenescence SASP DNAmPAI1
;;     ;         CoronaryHeartDisease) (signed Neg (stv 0.2499 0.17017138125)))
;;
;; No false chains when unconnected:
;; !(infer &self DasatinibPlusQuercetin AllCauseMortality)   ; (empty)
