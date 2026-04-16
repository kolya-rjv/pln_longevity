;; ==========================================
;; EPISTEMIC CALIBRATION: Confidence Mapping
;; ==========================================
;; Maps raw evidence types to their fixed confidence (c) values in PLN.
;; EvidenceCategory instances are declared here as the single authority.

;; ------------------------------------------
;; EvidenceCategory enumeration
;; ------------------------------------------
(: RCT_Human EvidenceCategory)
(: ITP_Positive EvidenceCategory)
(: ITP_Negative EvidenceCategory)
(: MultipleHumanTrials EvidenceCategory)
(: SingleHumanTrial EvidenceCategory)
(: AnimalStudies_Replicated EvidenceCategory)
(: Epidemiological EvidenceCategory)
(: AnimalStudies_Single EvidenceCategory)
(: Preprint EvidenceCategory)
(: InVitro EvidenceCategory)
(: TraditionalUse EvidenceCategory)

;; ------------------------------------------
;; Confidence mapping function
;; ------------------------------------------
(: evidence-confidence (-> EvidenceCategory Number))

;; High Confidence (Clinical / Standardized)
(= (evidence-confidence RCT_Human) 1.0)
(= (evidence-confidence ITP_Positive) 0.90)
(= (evidence-confidence ITP_Negative) 0.90)        ; High confidence in negative results too
(= (evidence-confidence MultipleHumanTrials) 0.85)

;; Moderate Confidence
(= (evidence-confidence SingleHumanTrial) 0.70)
(= (evidence-confidence AnimalStudies_Replicated) 0.65)
(= (evidence-confidence Epidemiological) 0.60)

;; Low / Preliminary Confidence
(= (evidence-confidence AnimalStudies_Single) 0.50)
(= (evidence-confidence Preprint) 0.40)
(= (evidence-confidence InVitro) 0.35)
(= (evidence-confidence TraditionalUse) 0.20)

;; ------------------------------------------
;; Helper: build an STV from strength + evidence type
;; ------------------------------------------
;; Usage: (apply-tv 0.85 AnimalStudies_Replicated) -> (stv 0.85 0.65)
(= (apply-tv $strength $evidence-type)
   (stv $strength (evidence-confidence $evidence-type)))