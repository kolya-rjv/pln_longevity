;; cellage_calibration.metta
;; CellAge CellSenescenceRecord rows -> calibrated, SIGNED senescence Effect links. [v1]
;;
;; This is the ETL->inference bridge for the CellAge vertical slice, and the
;; direct answer to the 2026-09-18 evaluation §4: "Gene data is present on disk
;; but not queryable. 'Which genes drive cellular senescence?' returned four
;; hallmark components, not the CellAge gene set."
;;
;; cellage_etl.py emits 927 raw, source-faithful curated rows:
;;
;;   (InstanceOf               CellAgeRow_869 CellSenescenceRecord)
;;   (InvolvesGene             CellAgeRow_869 Gene_TP53)
;;   (GeneSymbol               Gene_TP53 "TP53")
;;   (EntrezID                 Gene_TP53 7157)
;;   (ReportedIn               CellAgeRow_869 PMID_9275177)      ; provenance
;;   (HasSenescenceType        CellAgeRow_869 ReplicativeSenescence)
;;   (UsesCellContext          CellAgeRow_869 CancerCellContext)
;;   (HasSenescenceEffectLabel CellAgeRow_869 Induces)
;;   (HasSenescenceEffect      CellAgeRow_869 (Increases CellularSenescence))
;;
;; Nothing lifted those into the (Effect ...) links deduction consumes. This
;; layer is that bridge: it LIFTS one row on the fly into
;;
;;   (Effect <gene> CellularSenescence <sign> (stv s c))
;;
;; exactly as drugage_calibration.metta §7 lifts a DrugAge row. Raw rows are
;; never rewritten (the calibration-layer immutability invariant).
;;
;; ----------------------------------------------------------------
;; HONESTY CONTRACT  (read this before touching §1)
;; ----------------------------------------------------------------
;; 1. CellAge RECORDS A DIRECTION, NOT A MAGNITUDE. The curated table has one
;;    effect column with three values (Induces / Inhibits / Unclear) and no
;;    effect size anywhere. There is therefore NO reported quantity to transform
;;    into a strength, and none is manufactured: strength is ONE curated prior,
;;    identical for every curated row (§1). A per-row gradation here would be
;;    invented precision — it would rank genes by a number no source states.
;; 2. CONFIDENCE COMES FROM THE AUTHORITY. It is `(evidence-confidence InVitro)`
;;    from epistemic_calibration.metta, written as that lookup and not as a
;;    literal, so retuning the tier retunes this layer. CellAge curation is
;;    cell-line experimental evidence; InVitro is what that is.
;; 3. THE ETL'S OWN TRUTH VALUES ARE NOT USED. build/cellage_genes.metta also
;;    carries `(Causes Gene_TP53 (Increases CellularSenescence) (stv 0.82 0.70))`.
;;    Those numbers are computed by `cellage_etl.calibrated_stv` from the
;;    senescence type and the cancer-cell flag; they do NOT come from
;;    epistemic_calibration.metta and they are not calibrated. This layer
;;    ignores them entirely. If a caller surfaces them, they must be labelled as
;;    ETL-side, uncalibrated numbers.
;; 4. AN `Unclear` ROW PRODUCES NOTHING. The ETL emits `HasSenescenceEffect`
;;    only for Induces/Inhibits, and the lift matches on that atom, so a row
;;    whose curator declined to state a direction yields no link. Silence is the
;;    correct output for "the source does not say".
;; 5. EXPRESSION SIGNATURES ARE DELIBERATELY NOT LIFTED. cellage_expression.metta
;;    emits `(CorrelatedWith <gene> (Increases CellularSenescence) ...)` for
;;    1,259 genes differentially expressed in senescent cells. That is a
;;    CORRELATION — a gene going up in senescent cells is not a gene that drives
;;    senescence. Lifting it into (Effect ...) would silently convert an
;;    association into a causal claim, at scale. It stays out of this layer.
;;
;; Load AFTER (the CellAge inference stack — see pln_runner.CELLAGE_STACK):
;;   system_types.metta            (CellSenescenceRecord, EvidenceCategory, ...)
;;   logical_predicates.metta      (Inheritance / InstanceOf predicates)
;;   epistemic_calibration.metta   (evidence-confidence — the confidence authority)
;;   evidence_calibration.metta    (calibrate-tv — kept loadable for link-effect eq2)
;;   pln_deduction.metta           (Effect, Pos/Neg, infer, combine-hop)
;;   cellage_calibration.metta     (this file — loads LAST)
;;   + a FILTERED slice of CellAge rows (NEVER the full dump — hyperon aborts;
;;     ontology.cellage_selector selects the rows, capped at MAX_ROWS = 100)
;;
;; `CellularSenescence` is deliberately NOT re-declared here: it is the hallmark
;; symbol from hallmarks_core.metta, and Effect's arguments are typed Atom, so
;; this layer uses it without asserting a second, possibly conflicting type for
;; it.
;;
;; STATUS: v1, verified against hyperon 0.2.10
;; (tests/test_gene_queries.py::test_the_lift_*). Constants are coarse by
;; design: v1 locks the architecture, not the numbers, and every knob is in §1
;; so tuning stays local.

;; ================================================================
;; 1. KNOBS — every tunable constant lives here
;; ================================================================
;; CURATED PRIOR (magnitude). CellAge states direction only (honesty contract
;; §1), so this is a single labelled prior standing for "a curator read a paper
;; reporting this gene inducing or inhibiting senescence in a cell line". It is
;; deliberately below the ~0.85 this repo's curated mechanistic bridges carry
;; (mechanistic_bridges.metta), because a bridge there is a reviewed mechanism
;; and this is one database row. It is IDENTICAL for every row on purpose: any
;; spread between rows would be a magnitude CellAge does not report.
(= (cellage-curated-strength) 0.70)

;; The EvidenceCategory whose confidence this layer claims. Curated CellAge rows
;; are cell-line experiments, which is what InVitro means in
;; epistemic_calibration.metta (0.35). Changing the tier here is the ONLY
;; supported way to change this layer's confidence — never write a number.
(= (cellage-evidence-tier) InVitro)

;; ================================================================
;; 2. Sign  <-  direction of the curated effect   (THE SIGN DECISION)
;; ================================================================
;; Effect sign convention (pln_deduction.metta §0): Pos = <from> RAISES <to>.
;; The target here is CellularSenescence itself, so a gene that INDUCES
;; senescence raises it -> Pos, and one that INHIBITS it lowers it -> Neg.
;;
;; NB: unlike drugage_calibration §6 there is no sign trap to dodge, because
;; nothing is chained onto a beneficial axis. CellularSenescence is a HALLMARK,
;; not an outcome: "more senescence" is not automatically "worse lifespan", and
;; this layer does NOT bridge CellularSenescence -> Mortality. Senolytic biology
;; is context-dependent (senescence is tumour-suppressive in a cancer cell and
;; damaging in an ageing tissue), and the KB holds no evidence that fixes that
;; sign. Inventing the bridge would let every senescence-inducing gene fall out
;; of a mortality ranking with a confident, unsupported sign.
(= (senescence-sign Increases) Pos)
(= (senescence-sign Decreases) Neg)

;; ================================================================
;; 3. THE LIFT — one row -> one signed, calibrated Effect  (the deliverable)
;; ================================================================
;; Pure function of (space, row): reads the row's gene and its curated direction
;; (both must be present) and emits the canonical link. Provenance
;; (ReportedIn PMID) stays in the raw row, reachable from <row> for the audit
;; trail; nothing is materialised or rewritten.
;; NB: Effect's args are typed Atom (pln_deduction §0), which SUPPRESSES their
;; reduction — so sign/strength/confidence are forced through let* FIRST and the
;; emitted link carries concrete numbers, exactly as drugage-effect does.
(= (cellage-effect $s $row)
   (match $s (, (InvolvesGene $row $gene)
                (HasSenescenceEffect $row ($dir CellularSenescence)))
      (let* (($sgn (senescence-sign $dir))
             ($str (cellage-curated-strength))
             ($c   (evidence-confidence (cellage-evidence-tier))))
         (Effect $gene CellularSenescence $sgn (stv $str $c)))))

;; ================================================================
;; 4. Gene-level entry points
;; ================================================================
;; Every lifted link for one gene. A gene with several curated rows (TP53 has
;; three — replicative, stress-induced and oncogene-induced) yields one link per
;; row, NOT a merged verdict: merging is evidence revision, which this repo does
;; not do yet, and averaging three rows would report a number none of them says.
(= (cellage-gene-effects $s $gene)
   (match $s (InvolvesGene $row $gene) (cellage-effect $s $row)))

;; "Which genes drive cellular senescence?" — the evaluation's question, as a
;; query. $dir is Increases (drives it) or Decreases (restrains it).
;; DANGER: this is a variable-in-the-row-slot match, the exact shape that aborts
;; hyperon 0.2.10 past a few hundred data rows. It is only safe because the
;; space holds a SELECTED SLICE (ontology.cellage_selector, capped at 100 rows).
;; Never run it against the full 927-row dump.
(= (genes-affecting-senescence $s $dir)
   (match $s (, (HasSenescenceEffect $row ($dir CellularSenescence))
                (InvolvesGene $row $gene))
      $gene))

;; Same question, narrowed to one senescence type or one cell context — the two
;; qualifiers CellAge actually records, kept as separate arguments so a caller
;; cannot conflate "induces senescence in a cancer line" with "induces
;; senescence".
(= (genes-affecting-senescence-in $s $dir $context)
   (match $s (, (HasSenescenceEffect $row ($dir CellularSenescence))
                (, (UsesCellContext $row $context)
                   (InvolvesGene $row $gene)))
      $gene))

;; ================================================================
;; 5. Smoke-test queries (verified, hyperon 0.2.10 — see the test module)
;; ================================================================
;; With the CellAge stack + a selected slice loaded into one space:
;;
;; ONE curated inducing row (TP53, replicative, cancer context):
;; !(cellage-effect &self CellAgeRow_869)
;;    ; (Effect Gene_TP53 CellularSenescence Pos (stv 0.7 0.35))
;;
;; ONE curated inhibiting row (EZH2):
;; !(cellage-effect &self CellAgeRow_270)
;;    ; (Effect Gene_EZH2 CellularSenescence Neg (stv 0.7 0.35))
;;
;; Every link for one gene (TP53 -> one per curated row):
;; !(cellage-gene-effects &self Gene_TP53)
;;
;; The evaluation's question, against a slice:
;; !(genes-affecting-senescence &self Increases)
;;    ; Gene_TP53, Gene_CDKN1A, Gene_CDKN2A, ...
