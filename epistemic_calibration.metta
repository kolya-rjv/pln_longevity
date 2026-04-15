;; ==========================================
;; EPISTEMIC CALIBRATION: Confidence Mapping
;; ==========================================
;; These map raw evidence types to their fixed confidence (c) values in the PLN.

(: evidence-confidence (-> EvidenceCategory Number))

;; High Confidence (Clinical/Standardized)
(= (evidence-confidence RCT_Human) 1.0)
(= (evidence-confidence ITP_Positive) 0.90)
(= (evidence-confidence ITP_Negative) 0.90)     ; High confidence in negative results
(= (evidence-confidence MultipleHumanTrials) 0.85)

;; Moderate Confidence
(= (evidence-confidence SingleHumanTrial) 0.70)
(= (evidence-confidence AnimalStudies_Replicated) 0.65)
(= (evidence-confidence Epidemiological) 0.60)

;; Low/Preliminary Confidence
(= (evidence-confidence AnimalStudies_Single) 0.50)
(= (evidence-confidence Preprint) 0.40)
(= (evidence-confidence InVitro) 0.35)
(= (evidence-confidence TraditionalUse) 0.20)

;; Helper function: Assigns STV based on extracted strength and evidence type
;; Usage: (apply-tv 0.85 AnimalStudies_Replicated) -> (stv 0.85 0.65)
(= (apply-tv $strength $evidence-type)
   (stv $strength (evidence-confidence $evidence-type)))