# Phase 3 (curation): what I propose to write, for your decision

> **Status, 2026-10-05: decided and implemented; this is the proposal as it was put to the user.** Where it differs
> from what was built, `REPORT.md` ("Status of the implementation") and the commits are the record: decision 1 was
> **A** (LinAge2's young reference), not the recommended B; the node is `LowSerumAlbumin`; white cell count and the urate
> edge were not built; triglycerides are a fasting value of 150 mg/dL or more; the CHD observation is the narrow form.
> The statements below that say "nothing is written yet" or recommend B are the state at proposal time.

Nothing below is written yet. Per the brief, thresholds, gates and the albumin node name come to you first. After
each decision I change `mechanistic_bridges.metta` (edges), `linage2_core.metta` §5 (readouts) and
`patient_builder` / `patient_text` (witnesses), run `tests/test_patient_stack.py` (8 must-answer, 3 must-abort,
built-ins byte-identical), and commit one item at a time.

Strengths below are **curated priors** in the sense of the contract at the top of `mechanistic_bridges.metta`:
the confidence is the tier's lookup (`Epidemiological` = 0.60), the strength is a judgment I label as such.
"Impact" figures are the report's probes S1/S3 (`docs/kb_quick_wins/REPORT.md`); I have **not** re-run them.

## 1. What the literature supports (every record below was opened in PubMed)

| Edge | Anchor | What it shows | What it does NOT show |
|---|---|---|---|
| inflammation → RDW | Lippi 2009, PMID 19391664, [doi](https://doi.org/10.5858/133.4.628): n = 3,845 outpatients, retrospective | hsCRP and ESR predict RDW independently of age, sex, MCV, Hb, ferritin; hsCRP > 3 mg/L in 28 % of the lowest RDW quartile vs 63 % of the highest | direction (cross-sectional); the authors say the mechanism is unknown |
| | Förhécz 2009, PMID 19781428, [doi](https://doi.org/10.1016/j.ahj.2009.07.024): n = 195, heart failure | RDW tracks IL-6, TNF receptors, CRP **and** iron-metabolism, nutrition and renal markers | a single cause: the strongest correlates were soluble transferrin receptor and TNF receptors |
| | Patel 2009, PMID 19273783, [doi](https://doi.org/10.1001/archinternmed.2009.11): NHANES III, n = 8,175 | RDW predicts death even in non-anaemic people inside the 11-15 % reference range with no iron/folate/B12 deficiency | an inflammation → RDW edge (it is a mortality study) |
| inflammation → low albumin | Soeters 2018, PMID 30288759, [doi](https://doi.org/10.1002/jpen.1451): review | hypoalbuminaemia results from and reflects inflammation (capillary leak, larger distribution volume, shorter half-life) | that nutrition is irrelevant |
| | Don & Kaysen 2004, PMID 15660573, [doi](https://doi.org/10.1111/j.0894-0959.2004.17603.x): review, renal failure | inflammation **and** low protein/calorie intake both lower albumin; inflammation also raises catabolism | specificity: nutrition is a co-driver |
| insulin resistance → uric acid | McCormick 2021, PMID 33982892, [doi](https://doi.org/10.1002/art.41779): bidirectional Mendelian randomisation, n = 288,649 (urate), 153,525 (fasting insulin) | genetically higher fasting insulin raises urate (0.37 mg/dL per log-unit; 0.69 after BMI adjustment); genetically higher urate does not raise insulin | a size large enough to matter on its own for a person |
| | Facchini 1991, PMID 1820474: n = 36 | insulin resistance tracks urate (r = 0.69) and falls with urinary urate clearance (r = -0.49) | causation (cross-sectional) |
| | Quiñones Galvan 1995, PMID 7840165, [doi](https://doi.org/10.1152/ajpendo.1995.268.1.E1): clamp, n = 20 | hyperinsulinaemia lowers urinary urate excretion | an acute rise in serum urate (it did not change) |
| WBC ← inflammation | Chmielewski 2018, PMID 29064542, [doi](https://doi.org/10.5603/FM.a2017.0101): review | leukocyte count rises with infection, trauma, inflammation and correlates with CRP and IL-6; smoking is a named determinant | a primary study of inflammation raising WBC (I found none) |
| | Smith 2003, PMID 12921986, [doi](https://doi.org/10.1016/s0021-9150(03)00200-4): EPIC-Norfolk, n = 15,307 | age/BMI-adjusted WBC in men 7.8 (current) vs 6.4 (former) vs 6.2 (never) ×10³/µL; it falls with time since quitting | independence of smoking from inflammation (the authors read it as smoking-induced inflammation) |
| triglycerides ← insulin resistance | Ginsberg 2005, PMID 15925013, [doi](https://doi.org/10.1016/j.arcmed.2005.01.005): review | in insulin resistance the fatty-acid flux and de novo lipogenesis raise hepatic VLDL output, hence triglycerides | a per-person size |
| | McLaughlin 2003, PMID 14623617, [doi](https://doi.org/10.7326/0003-4819-139-10-200311180-00007): n = 258 overweight non-diabetic | triglyceride ≥ 130 mg/dL identified insulin-resistant people with 67 % sensitivity, 71 % specificity | insulin resistance causing the rise (it uses TG as a marker) |

**Anchors that do not support what they were listed for** (the report listed them, so you should know): Don & Kaysen
is about albumin, not RDW. Ruggiero 2007 (PMID 17481443) and Danesh 1998 (PMID 9600484, [doi](https://doi.org/10.1001/jama.279.18.1477))
test WBC → mortality / CHD, not inflammation → WBC; **I will not cite either for the WBC edge.** Danesh 1998 also
reports an inverse albumin–CHD association (risk ratio 1.5 for 38 vs 42 g/L), a different edge.

**Metformin and urate**: Dai 2023 (PMID 37807832, [doi](https://doi.org/10.1111/dom.15310)): metformin users −4.3 µmol/L,
genetically proxied −12.5 µmol/L, a third of it through BMI. Marrugo 2024 (PMID 38749572, [doi](https://doi.org/10.1136/ard-2024-225652)):
no relationship between metformin and a change in serum urate in people with pre-diabetes. Mixed and small.

## 2. #7 — RDW and low albumin (LinAge2 side)

Proposed atoms (strength = prior; tier `Epidemiological`):

```
;; mechanistic_bridges.metta, after the ChronicInflammation edges (:129-143)
(: RDW Biomarker)  (: LowSerumAlbumin Biomarker)
(Effect ChronicInflammation RDW             Pos (stv 0.50 (evidence-confidence Epidemiological)))
(Effect ChronicInflammation LowSerumAlbumin Pos (stv 0.55 (evidence-confidence Epidemiological)))
;; linage2_core.metta §5
(MeasuresBiomarker RedCellDistributionWidth RDW)
(MeasuresBiomarker SerumAlbumin             LowSerumAlbumin)
```

0 new head symbols (new non-head symbols are safe inside the file; they abort as extra atoms, which I will not do).
SASP → ChronicInflammation already exists, so senescence is credited through the chain as well.

**Decision 1: what counts as an elevated RDW or a low albumin (the witness).**

| | RDW | albumin | the 3 tab examples (smoker / healthy woman / six labs: RDW 14.1 / 12.6 / not given; albumin 4.1 / 4.5 / 3.8 g/dL) |
|---|---|---|---|
| **A. LinAge2 young reference, z > 1** (the rule every witness uses; sex-specific) | > 13.09 % (men), > 13.27 % (women) | < 43 g/L (men), < 41 g/L (women) | the smoker and the six-labs patient gain inflammation levers (the report's probes: the smoker's 0 → −2.46 y and −1.77 y; the six-labs patient −2.56 y and −1.85 y from albumin alone); the healthy woman is unchanged; **normal-range values are called abnormal** |
| **B. Clinical limits** | > 15 % (the upper limit of the 11-15 % reference range in Patel 2009) | < 35 g/L (3.5 g/dL; the usual report lower limit, **not sourced here**) | none of the three changes; only RDW > 15 % or albumin < 3.5 g/dL is explained |

I recommend **B**. You chose "status quo" for #6 so that normal-range values are not called abnormal; A does exactly that
for RDW and albumin. The cost of B is that #7 changes nothing visible for the three examples.

**Decision 2: the deficit node's name.** `LowSerumAlbumin` (recommended). `Hypoalbuminemia` is a clinical diagnosis
(< 3.5 g/dL) and would be wrong under A.

**Gate for RDW (recommended):** do not credit inflammation when the person typed an anaemia or deficiency the reader
already parses: hemoglobin, ferritin, vitamin B12 or folate below its lower limit (conventional limits, not sourced here;
I will take the numbers from you or from the reader's own table). Bessman 1983 (PMID 6881096) uses RDW to classify iron- and
folate-deficiency anaemia, which is why this gate exists. **Gate for albumin:** a note, not a block: nutrition (Don & Kaysen),
the last meal (albumin is lower up to 3-5 h after eating: Langsted 2008, PMID 18955664) and smoking are named as confounders.

## 3. #12 — uric acid, white cell count, triglycerides

| marker | edge | readout | witness / threshold (proposed) | gates | evidence |
|---|---|---|---|---|---|
| **uric acid** | `(Effect InsulinResistance UricAcid Pos (stv 0.50 …Epidemiological))` | `(MeasuresBiomarker UricAcid UricAcid)` | per-sex; **B**: > 420 µmol/L (7.0 mg/dL) men, > 360 µmol/L (6.0 mg/dL) women (the usual hyperuricaemia limits, **not sourced here**) | CKD diagnosis or high creatinine (Johnson 2018, PMID 29496260, [doi](https://doi.org/10.1053/j.ajkd.2017.12.009): subtle changes in kidney function move urate); diuretics (Choi 2012, PMID 22240117, [doi](https://doi.org/10.1136/bmj.d8190): RR 2.36 for gout, an indirect anchor) are **not readable** (the medication reader only knows KB drugs), so they become a note | strongest of the five: a Mendelian-randomisation result in the stated direction |
| **WBC** | `(Effect ChronicInflammation WhiteBloodCellCount Pos (stv 0.40 …))` | `(MeasuresBiomarker WhiteBloodCellCount WBC)` | **B**: > 11 ×10³/µL | **not a witness for a current smoker** (the default patient is one; Smith 2003); infection and steroids unreadable → note | weakest: reviews only, no primary study of the direction |
| **triglycerides** | `(Effect InsulinResistance Triglycerides Pos (stv 0.50 …))` | none (not a LinAge2 input) | **B**: ≥ 150 mg/dL (1.7 mmol/L), counted only when "fasting" is typed, like glucose | non-fasting: triglycerides rise up to 0.3 mmol/L (27 mg/dL) after a meal (Langsted 2008) | mechanism review + a marker study |

Every "B" threshold in this table is the usual report limit; none of them is sourced from a paper here.

The name `Triglycerides` is already reserved at `nhanes_reference_etl.py:405-414`.

**Decision 3, WBC:** include it with the smoker gate, or leave it out. It has the weakest anchor and its default patient
is gated off, so it adds little. I lean to leaving it out until a primary anchor exists.

**Decision 4, uric acid:** the chain implies "metformin lowers urate" (Metformin → insulin resistance → urate). The two
studies above disagree. Include the edge (the claim is derived, low-confidence, and labelled) or leave it out.

**Decision 5, triglycerides:** 150 mg/dL fasting-only (recommended), or 130 mg/dL (McLaughlin's cut, for overweight
non-diabetic people only), or the NHANES-percentile prior (≈ 210 mg/dL; the report found the tab example's 190 mg/dL then changes nothing).

## 4. #11 — CHD, angina and heart attack as an observation

Mechanics: a new head `PatientCondition` (1 head, only ever in the patient stack, which the stack guarantees); a
4-argument `patient-relevance` that skips `$outcome`, so the CHD ranking does not count the condition against itself
(without that exclusion every CHD score exactly doubles). `DIQ010` is not mapped.

**Honesty conflict you should rule on.** The KB's own policy says these NHANES items are *prevalence*, "usable only as
exclusion" (`nhanes_baseline.metta:38-39`, `docs/nhanes_integration.md:212-218`). Using them as a positive observation
reverses that. It also credits a person's CHD to insulin resistance or senescence, because smoking has no edge to CHD in
`lifestyle_evidence.metta`: an association-level statement presented as a diagnosis of cause.

**Decision 6:** (a) skip #11; (b) **narrow**: the observation feeds `diagnose-patient` only, labelled "prevalence item, not a
measured value", and does not change the supplement tiers or the ranking; (c) full, as in the report (supplement tiers change:
Berberine 0.181 → 0.242 for the test caller). I recommend (b).

## 5. What I will not do

No `MeasuredZ` is invented for a diagnosis. No edge goes in through `extra_atoms` or a separate appended file. The two
clocks stay apart. RDW, albumin and the rest never get a `LinAge2` lever token (rule 16 already says so, and the new test
reads the rule's marker list from the KB, so each bridge makes that test fail until the rule is updated, which is wanted).

## 6. After your decisions

Order: #7, then #12, then #11, one commit each with its tests; the rule-16 marker list and the default prompt size are
updated with each; then #13 (battery entries, one rebuild of the PDF, because D3's albumin entry changes with #7).
