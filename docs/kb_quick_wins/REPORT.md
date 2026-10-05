# What the My Patient tab could feed into the rest of the knowledge base — research report

Branch `linage2`, measured at `2fdcad1` (after the routing fix `282c854`). Produced in the
cloud session: 5 scouts with live MeTTa probes found 37 leads; each lead was re-checked by a
skeptic who re-ran its probe (35 held, 2 did not); a synthesizer ran ~126 more probes to
settle disagreements and rank them. Full lead-by-lead evidence, probes and verdicts:
`scouting.json`. Reproducible probes: `probes/`. The question and constraints:
`RESEARCH_BRIEF.md`.

## Summary

- **Today only CRP, HbA1c and fasting glucose can change a shared-layer answer.** No other
  typed lab has a causal edge in the KB, typed diagnoses reach only LinAge2's comorbidity
  score (≈0 years), and low values are never read. "Add markers for labs that already have
  edges" is therefore already done; more needs new curated edges.
- **The cheapest wins need no new KB content**: a default cause list so "what drives my
  abnormal labs?" stops returning empty for the tab's own example (#1); honest notes when the
  shared layers have nothing to work from (#4) and when the heart-risk model cannot apply
  (#5, #9); and reading "takes metformin" so the supplement plan flags Berberine (#3).
- **One line removes the latency wall** (#2): de-duplicating patient markers natively in
  `patient_profile.metta:212` takes the supplement plan with 16 extra markers from 104 s to
  10 s, every answer byte-identical — the enabler for any new witness.
- **The witness cut-points are a clinical decision for you** (#6): today HbA1c must exceed
  6.0 %, fasting glucose 107 mg/dL and CRP 5.44 mg/L (both sexes) before anything happens; the
  options and their effect on each example are in table S4.
- **The biggest gain on the LinAge2 side is two curated edges** (#7): RDW and low albumin →
  chronic inflammation, written inside `mechanistic_bridges.metta` (safe there — any other
  placement aborts the shared space). The smoker example's two largest unexplained drivers
  then get a cause, and the inflammation levers go from 0 to −2.5 y and −1.8 y.

## Recommended plan

1. **No decision needed, small (S):** #2 latency fix → #1 default cause list → #4 "nothing
   to work from" note → #5 heart-risk-without-GrimAge note → #9 prevalent-CHD caveat → #3
   medications. Each with the tests named in its row; re-run `tests/test_patient_stack.py`
   after any `.metta` edit (8 must-answer + 3 must-abort controls).
2. **Your decision, then small:** #6 witness cut-points (status quo / clinical floors /
   LinAge2 references — see S4 and open question 1).
   **Decided 2026-10-05: status quo (option A).** The coarse pooled prior stays (HbA1c > 6.0 %, fasting glucose
   > 107 mg/dL, CRP > 5.44 mg/L, both sexes); nothing changes in `patient_builder`. The implied cut points of the
   alternatives, for the record: clinical floors 5.7 % / 100 mg/dL / 3 mg/L; LinAge2 young reference HbA1c 5.74 (M) /
   5.64 (F), glucose 101 / 97, CRP 4.6 (M) / 10.2 (F); LinAge2 age-median at 58 HbA1c 5.94, glucose 104 (M) / 100 (F),
   CRP 7.4 (M) / 14.6 (F). The gap band (HbA1c 5.7-6.0, glucose 100-107, CRP 3-5.4) therefore still reads as normal.
3. **Curation (M each), after #2:** #7 RDW + albumin bridges (verify anchors, choose
   thresholds and gates, rename the deficit node), then #12 (urate, WBC, triglycerides) and
   #11 (CHD-family diagnoses as an observation, with the ranking double count excluded).
4. **Visibility:** #8 patient-dependent suggested buttons, #10 translator rule (needs the live
   translator to check), #13 battery entries guarding all of the above.

Items for other threads, found on the way (§3 items 9-10): reader defects (". " is not a
statement separator; "<condition> on <drug>" and "I have type 2 diabetes, hypertension and…"
block the build) belong with the model-extraction work; `!(human-evidence &self Metformin)`
and any full-stack query naming an unknown symbol abort today; the tab's download header
fails validation (`validate()` tokenizes `;;` comments).

Every MeTTa probe below ran in a subprocess. The scouts and skeptics ran about 600 probes. I ran 126 more as synthesizer, and none of mine aborted unexpectedly; the only aborts were the three controls that are supposed to abort. No repository file was edited. The harness and outputs are in `docs/kb_quick_wins/probes/` (`run1.py` runs one job in a subprocess; `drive.py jobs.json [parallel]` runs a job file; `mb_all.metta` is `mechanistic_bridges.metta` with the five candidate edges; `pp_ua.metta` is `patient_profile.metta` with the S2 fix; `jA`–`jD.json` are the job files with their `*_out.json` results). Run from that directory.

## 0. New synthesizer probes (these settle the scouts' disagreements)

**S1. All five candidate bridges written inside `mechanistic_bridges.metta` break nothing.** I made a scratch copy that adds:
- after `:138`: `(Inheritance Hypoalbuminemia Biomarker)` and the edges ChronicInflammation → RedCellDistributionWidth / WhiteBloodCellCount / Hypoalbuminemia;
- after `:185`: the edges InsulinResistance → UricAcid / Triglycerides.

Results:
- **Full stack:** all 8 must-answer controls from `tests/test_patient_stack.py` gave byte-identical answers, and so did 5 generic queries (`infer Metformin CHD`, population `rank-interventions`, population `diagnose`, `infer Omega3 CRP`, `match Inheritance CHD`). The 3 must-abort controls (P001 rank, P001 supplements, P002 supplements) still exit with rc -6.
- **Patient stack:** P001 diagnosis, rank (0.2306046747621094) and supplements, P002 supplements and P003 rank are byte-identical.
- **LinAge2 stack:** the smoker decomposition is byte-identical.

So the linage2-witnesses scout's rule "never put a bridge in `mechanistic_bridges.metta`" is **wrong**. It came from appending the edge after all the files. Placement is what decides it:
- inside `mechanistic_bridges.metta`: safe;
- `extra_atoms` in the full stack: aborts;
- a new file appended to the stack: aborts.

**S2. A one-line fix removes the latency ceiling.** In `patient_profile.metta:212`, replace `(unique-tuple (append-tuple $zs $rs))` with `(let $all (append-tuple $zs $rs) (unique-atom $all))`.

Timing with the tab-default patient plus N inert extra labs, run one at a time:

| Query | Before | After |
|---|---|---|
| `recommend-supplements-patient`, +0 labs | 2.7 s | 1.6 s |
| +8 labs | 26.6 s | 4.9 s |
| +12 labs | 55.7 s | 7.2 s |
| +16 labs | 104.1 s | 10.1 s |
| +24 / +32 labs | not run | 18.4 / 30.6 s |
| Ranking, +16 labs | 51.4 s | 5.1 s |
| P001 supplements | 14.4 s | 6.0 s |
| P002 supplements | 10.4 s | 3.1 s |

Every answer is byte-identical: the full-stack controls, the patient-stack built-ins, all 3 tab examples × diagnosis/supplements/ranking, and every timed run. The must-abort controls still abort.

Two cautions:
- **The let-forcing is required.** My first attempt, an un-forced `(unique-atom (append-tuple …))`, silently returned empty observations, and that also made the must-abort controls "answer". Those controls only stay meaningful while the observations are right.
- **Order can change in one case.** It differs only when a marker appears as both MeasuredZ and MeasuredRaw: `(HbA1c CRP)` becomes `(CRP HbA1c)`. The diagnosis was identical in that probe.

**S3. Combined demo on the tab's "58-year-old smoker"** (S1 bridges + S2 fix). Realistic witness z values: RDW 2.698 and Hypoalbuminemia 1.686; WBC 0.73 and TG 0.81 stay Normal. The patient stack has 32 padding heads, and the answers are identical with or without them.
- **Diagnosis** adds ChronicInflammation `(stv 0.5 0.84)` with coverage 2, `SupportedBy (Hypoalbuminemia RedCellDistributionWidth)`, plus CellularSenescence. InsulinResistance stays on top.
- **Supplements:** Omega3 joins Tier 1 (0.275), Fisetin enters Tier 2, Resveratrol goes to NotRecommended. Takes 7.2 s with S2, 18.8 s without.
- **Ranking:** D+Q 0.0425 → 0.103.
- **LinAge2** (plus the readouts `(MeasuresBiomarker RedCellDistributionWidth RedCellDistributionWidth)` and `(MeasuresBiomarker SerumAlbumin Hypoalbuminemia)`):
  - ChronicInflammation lever 0.0 → **−2.456 y** (confidence 0.6)
  - CellularSenescence lever 0.0 → −1.774 y (confidence 0.205), Via (RedCellDistributionWidth SerumAlbumin)
  - RDW (+2.553 y) and albumin (+2.359 y) are now DrivenBy (ChronicInflammation CellularSenescence).
- **"Six labs only" + Hypoalbuminemia 2.698:** ChronicInflammation coverage 1→2 and mass 0.52→0.82; Omega3 0.239→0.376; D+Q 0.095→0.126.

**S4. Witness z under each reference policy** (Elevated means z > 1.0, strict):

| Patient / value | coarse prior (today) | LinAge2 young reference (≤50, same sex) | LinAge2 age-median |
|---|---|---|---|
| 58M smoker: HbA1c 6.4 / CRP 3.1 / FPG 112 | 1.80 / **0.44** / 1.42 | 2.47 / 0.64 / 2.04 | 2.02 / 0.23 / 1.75 |
| 45F healthy: HbA1c 5.2 / CRP 0.6 / FPG 88 | −0.60 / −1.20 / −0.58 | 0.00 / −1.19 / 0.01 | 0.00 / −1.22 / 0.01 |
| 66M six labs: HbA1c 7.1 / CRP 6.5 | 3.20 / 1.18 | 4.05 / 1.30 | 3.37 / **0.78** (CRP would be lost) |
| 58M "gap": HbA1c 5.9 / CRP 5 / FPG 105 | 0.80 / 0.92 / 0.83 | 1.35 / 1.07 / 1.36 | 0.90 / 0.65 / 1.07 |
| 58F "gap": same values | 0.80 / 0.92 / 0.83 | 1.57 / **0.45** / 1.92 | 0.90 / 0.17 / 1.58 |
| 58F CRP 10 mg/L | **1.61** | 0.98 | 0.71 |
| 70M HbA1c 5.9 | 0.80 | 1.35 | 0.67 |

Today's cut points (patient_builder.py:137-166):
- HbA1c Elevated only above 6.0 %
- fasting glucose above 107 mg/dL
- CRP above 5.44 mg/L, for both sexes

## 1. Ranked quick wins (verified, corrections applied)

| # | Name | What changes for the person (probed) | What to change, where | New heads | Effort | Evidence | Probe result |
|---|---|---|---|---|---|---|---|
| 1 | **Default broad cause list for "what drives my abnormal labs?"** | On the tab default, the suggested question (patient_tab.py:52), translated with the only few-shot list, returns `()`. The broad list returns **InsulinResistance (stv 0.8 0.8775) coverage 2 SupportedBy (HbA1c FastingGlucose)**, then DeregulatedNutrientSensing. In "six labs", the HbA1c z of 3.2 also gets explained. | (a) A 2-argument overload after patient_profile.metta:302: `(= (diagnose-patient $space $patient) (diagnose-patient $space $patient (CellularSenescence ChronicInflammation MitochondrialDysfunction InsulinResistance DeregulatedNutrientSensing SmokingPackYears)))`. (b) Rule 17 after rule 16 (system_prompt.txt:184-213), separating "drivers of my biological age" (linage-drivers-patient) from "cause of my abnormal labs" (diagnose-patient). (c) A diagnose line in the hint at patient_context.py:113-131. (d) Few-shot few_shot_examples.json:185. (e) Fix the comment at patient_tab.py:44-45. (f) One test for the 2-argument form. No test reads the few-shot file. | 0 (the overload reuses the name) | S | few_shot_examples.json:185; no "diagnos" rule anywhere in system_prompt.txt; pln_abductive_diagnosis.metta:147-149; mechanistic_bridges.metta:180-185 | 27 skeptic probes, all rc 0. The 3-argument form is unchanged when the overload is present. Full-stack predict-risk P001 is unchanged with +4 and +5 definitions. Caveats: hallmark-only abduction, and the size of z is ignored. |
| 2 | **Latency: native dedupe in `patient-markers`** (enabler) | Same answers, and the 60 s timeout is no longer an issue: about 30 witnesses become affordable instead of 3-5. P001's supplement plan drops from 14 s to 6 s. | patient_profile.metta:212, one line (S2), and it must stay let-forced. Keep `unique-tuple`, which pln_linage2.metta:120 uses. Add a regression test: built-in byte identity plus a wall-time bound on supplements with 16 markers. tests/test_nhanes_integration.py:266-269 (each marker once) still holds. | 0 (`unique-atom` is grounded stdlib) | S | Cost is the recursive dedupe at patient_profile.metta:191-212, recomputed per candidate via :335-339 and pln_supplement_recommendation.metta:101-106 and 116-129; timeout at pln_chat/config.py:132 | S2 |
| 3 | **Read "takes metformin" into `(CurrentMedication Caller_Me Metformin)`, and make the single-supplement form flag interactions** | Smoker example plus "takes metformin": Interactions `()` becomes `(InteractionFlag Berberine Metformin "Shared AMPK activation — additive glucose lowering; monitor and consult MD")`. With the extra equation, "should I take berberine?" also carries the flag; today it returns a bare SuppRec, even for built-in Patient002. Today the medication line lands in not_understood. | A reader rule in patient_text.py: allow-list Metformin plus brand/salt names from ontology/compound_names.py:227-228 (not wired blindly, it is keyed to DrugAge), and negation ("stopped/no metformin"). Split "<condition> on <drug>", which **blocks the build today**, and update tests/test_patient_text_corpus.py:157,206 and docs/linage2_integration.md:409. Add the field in to_patient/as_dict (patient_text.py:314-347). build_patient validates the list and appends to `shared_lines` only (patient_builder.py:666-679). Add the medication as prose in patient_prompt_section (patient_context.py:134-149). Add the field to PatientIn (api.py:1066-1084, `extra="forbid"`; without it /patients/from-text returns 500) and to PatientOut/PatientPreviewResponse. After pln_supplement_recommendation.metta:232: `(= (supplement-for-patient $space $patient $supp) (let $r (supplement-rec $space $patient $supp) (interaction-flag $space $patient (rec-supp $r))))`. Update the result prose at system_prompt.txt:113-119 and API.md:1286. | 0 in the patient stack (the head exists at supplement_evidence.metta:151); +1 in the LinAge2 stack if put in `atoms`, so don't | S+ | supplement_evidence.metta:68, 136-141 (the KB's **only** Interaction fact), 151; pln_supplement_recommendation.metta:186-199, 212-218, 231-232 | Flag appears with 32 padding heads; Lisinopril and Rapamycin are inert; the equation leaves P001 Berberine/Resveratrol and P002 Omega3 unchanged. Not modelled: "already taking". Metformin is still ranked first (0.648), and the LinAge2 Metformin counterfactual is still −0.489 y. |
| 4 | **Say when the shared layers have nothing to work from, and flag the ranking as unpersonalized** | A patient with diabetes, hypertension, CKD, albumin 4.0, creatinine 1.8, SBP 150 and RDW 15.2 is told "Diagnosis, supplement ranking and intervention ranking still work". In fact diagnosis returns `()`, every tier is empty, and the ranking is byte-identical to an unknown id. The same happens to the healthy example, to a diabetic with HbA1c 5.5 typed, and to elevated markers with no edge (e.g. AgeAccelGrim 2.0). | patient_builder.py:583-588: make the last sentence conditional on `any(m.status=="Elevated" and m.name in EDGE_SET)`. Status is computed at :408 with the KB threshold from patient_knobs (patient_context.py:26-48). Read EDGE_SET from the KB by regex on `(Effect \S+ <marker>`; it is {CRP, FastingGlucose, HbA1c, DNAmPAI1, DNAmGDF15, DNAmPACKYRS}. The "inert" flag at :411-417 cannot stand in, because AgeAccelGrim has no note yet no edge. Map both variants in the _TAB_WORDING prefix (patient_tab.py:212-215). Add one shared warning helper for api.py (~3405) and app.py:459-465, keeping tests/test_ui_api_parity.py intact. Word it "may be unpersonalized", since an elevated DNAmPACKYRS still gives a population-identical CHD rank. Say that glucose counts only when "fasting" is typed (patient_text.py:296-298). | 0 | S | patient_profile.metta:277-287, 347-348, 363-365; api.py:592-616 (the warning fires only for unknown ids) | 22 skeptic probes. The boundary is strict: CRP z 1.0 gives `()`, z 1.01 gives hypotheses. |
| 5 | **"What's my heart risk?" without a GrimAge line** | For the tab default, predict-risk-patient returns empty. In a mixed program the CHD part silently drops out, and the UI shows only the hazard with no note. With the fix: a deterministic note "no heart-specific model without a GrimAge clock", plus `LinAgeHazard … AllCauseMortality (hazard-multiplier 3.2568) (confidence 0.54)`, labelled all-cause. | Rewrite system_prompt.txt:210-213 ("a patient with no **AgeAccelGrim** gets nothing…") and pair the heart line in rule 12 (:93-94). Add a hint at patient_context.py:126-130. Add a deterministic warning modelled on linage2_form_warnings (linage2_router.py:169-193), appended at app.py:460-463 and api.py:3424-3427. Use linage-hazard-patient only: linage-risk-patient is **empty** because no baseline file ships (pln_runner.py:222-233). It must not fire for battery C2 (linage2_battery.py:508-509). Add a P1 heart entry to the battery. | 0 | S | linage2_fong2025_evidence.metta:43-50, 57-58; pln_linage2.metta:194; docs/linage2_integration.md:267-271 | P1 + AgeAccelGrim 1.07143 → CHD point 0.1085. POST /query with the pair routes linage2+generic and returns only the hazard. **Never multiply the two clocks.** |
| 6 | **Witness z policy — needs your decision** | The gap band only: HbA1c 5.7-6.0 %, FPG 100-107 mg/dL, CRP 3-5.4 mg/L. On the three examples, only the smoker's CRP 3.1 flips under a >3 mg/L floor. It then gains ChronicInflammation and CellularSenescence, Omega3 Tier 1, Fisetin Tier 2, Resveratrol NotRecommended, and D+Q 0.0425 → 0.0952 (still 3rd). For the gap patient (58M, HbA1c 5.9, CRP 5, FPG 105), today's observations are `()`. With all three re-referenced: IR coverage 2 (stv 0.8 0.8775), ChronicInflammation, Berberine 0.388 + Omega3 0.239, Metformin 0.648. HbA1c alone gives IR coverage 1, Berberine 0.181, Metformin 0.350. | Option B (clinical floors: HbA1c ≥ 5.7, FPG ≥ 100, hs-CRP > 3) belongs in patient_builder (MarkerSpec/Reference, :137-166), **not** kb_markers. kb_markers' `{value, unit}` output is pinned by tests/test_patient_text.py:221-225 and tests/test_patient_tab.py:53. Floors must carry a witness note, take the threshold from the KB (patient_context.py:22,42), and update the warning at patient_builder.py:635-656. THE_REPORTED_PATIENT (tests/test_caller_patients.py:53-63, CRP 4 mg/L) flips. The principled form is a per-marker threshold (docs/patient_grounding.md:185-187), about M and 1 head. | 0 (Python) | S once chosen | patient_profile.metta:151, 157-161, 277-283; S4 table | Options: **(A) LinAge2 age-median** is *stricter* than today for HbA1c at ≥58 and **loses six-labs' CRP** (0.78). **(Young reference)** matches ADA 5.7 % but overcalls older people (cohort mu_z ≈ 0.97) and **hides CRP in women up to ~10 mg/L**; it also uses a different assay (NHANES latex CRP vs hs-CRP). **(B) Floors** encode guideline cuts that are not in the KB (the ADA 100 mg/dL cut; WHO uses 110). |
| 7 | **RDW and low-albumin bridges** (biggest gain on the LinAge2 side) | See S3. The smoker's two largest unexplained measured drivers get a cause, the "what could I do" inflammation levers go from 0 to −2.46 y and −1.77 y, and Omega3 enters Tier 1. RDW alone gives −2.73 y on the ChronicInflammation lever (not −6.1). Albumin alone, on six labs, gives ChronicInflammation −2.56 y and CellularSenescence −1.85 y. | (a) In mechanistic_bridges.metta after :138: the RDW and deficit-node edges with anchors, tier Epidemiological (epistemic_calibration.metta:36 = 0.60). (b) In linage2_core.metta §5 (:250-268): two MeasuresBiomarker facts. Readouts must live there, because linage2_builder.py:85-92 reads only this file. (c) z-only MarkerSpecs plus kb_markers computing a **sex-specific** z from data/linage2/linage2_model.json median/MAD, negated for albumin. Route them through the Z_LIMIT cap (patient_text.py:298-311), otherwise RDW ≥ 19.6 % fails the build with implausible_z. Add them to the witness set at patient_builder.py:565 and add a provenance note: a `{"z":…}` input skips the "not age-adjusted" warning. (d) Update tests/test_linage2.py:106, 490, 715; the linage2_core.metta:245 comment; docs/linage2_integration.md:130-134, 264; battery c_decomp_top (linage2_battery.py:269-274) and D3 (:534-536). | 0 (Hypoalbuminemia is a new non-head symbol: safe in-file, fatal as extra atoms in the full stack) | M | linage2_core.metta:200, 208; pln_linage2.metta:99-110, 132-143; PubMed-checked anchors: Lippi 2009 PMID 19391664 (hsCRP and ESR predict RDW; cross-sectional); Don & Kaysen 2004 PMID 15660573 (CKD review; malnutrition also lowers albumin) | S1 and S3. Risks: on the young reference, RDW counts as Elevated above 13.09 % (M) / 13.27 % (F) and the albumin deficit fires below 43 g/L (M) / 41 g/L (F). Both are inside normal ranges, so both tab examples (albumin 4.1 and 3.8 g/dL) would read "Hypoalbuminemia". Rename the node (e.g. LowSerumAlbumin) or gate at a clinical cut. Gate RDW on ferritin, B12 and folate, which the reader already has (patient_text.py:79-83). LinAge2 years = strength × contribution years, so the 0.5 prior sets the headline. |
| 8 | **Suggested buttons that depend on the patient** | For the witness-less patient and the healthy example, 4 of the 6 buttons return `()` or zero today. Gate diagnosis, supplements and scenarios on an elevated witness that has an edge, and the smoking button on a smoker. The gate matched which shared forms answered for all 5 probed patients. | patient_tab.py:46-53, :261, :295, :303-304, :308-310 (closures → `inputs=btn`), on_clear :264-267; tests/test_patient_tab.py:31, 71; battery output indices linage2_battery.py:206-207, 231, 253, 263 (append new outputs at the end). Drop the templated "why does my <lab> add N years?": it has no translator rule, and naive translations return empty or a 422. If kept, word it "What does the knowledge base credit for my RDW's +N y?" and add a rule-16 line. | 0 | S (S+ with the template) | linage2_builder.py:111-116 (`reads_out` already on BuiltPatient) | Battery: the "honest zero" answers it hides are a documented feature (few-shot, D3). Keep at least one. |
| 9 | **Caveat on the 10-year CHD risk for someone who reports CHD** | CHD, MI or angina plus a GrimAge line gives the same incident-CHD number either way: 0.10416 with or without "diagnoses: coronary heart disease, heart attack, angina". Add an explicit "first-event model, does not apply" note. | Tab-only: about 5 lines in on_build (patient_tab.py:239). Full version, about 30-50 lines: a flag through the to_patient payload (patient_text.py:326-330) → build_patient → a BuiltPatient field (patient_builder.py:226-247) → PatientIn → patient_prompt_section. Only when can_predict_risk holds (patient_builder.py:249-252). Exclude MCQ160B. Also patient_text.py:895-896 and system_prompt.txt:91-98. | 0 | S / S+ | pln_risk_prediction.metta:96-104, 218, 399-415; grim_age_lu2019_evidence.metta:28; nhanes_baseline.metta:38-39 | Reach is narrow: only people who type a GrimAge value. |
| 10 | **Translator rule: a lab or condition with no KB relation** | "What should I do about my kidney function?", "what if my BP were normal?" and similar: today a lab-named linage-counterfactual returns empty with no warning, a `LinAgeContribution` filter gets a 422, and a diagnosis-named lever fails validation. The rule maps these to the decomposition (+ recommend-supplements-patient if asked) plus "no curated cause or lever". Diagnoses enter only ComorbidityScore, which is about 0 y (−0.0046 → +0.0007 y). | Rule 16 (system_prompt.txt:184-213): 3 lines and 1 few-shot; hint at patient_context.py:113-131. **Rewrite battery D3** (linage2_battery.py:534-536), which expects the empty lab-lever answer, and rebuild the PDF and results.json. Prompt headroom is about 5.9k chars (API.md:161; tests/test_prompt_size.py:47). Never call the dropped years a counterfactual to "normal". | 0 | S | pln_linage2.metta:86, 274-281, 299-301; docs/linage2_integration.md:264-266 | The decomposition plus supplements route through POST /metta/run returns ok. LLM-graded, so it cannot be verified offline (no OPENAI_API_KEY). |
| 11 | **CHD-family diagnoses as an observation** (MCQ160C/D/E → CoronaryHeartDisease) | Test caller: IR and CellularSenescence reach coverage 2 and overtake ChronicInflammation; Berberine 0.181 → 0.242 (Tier 1 #1); Fisetin and Resveratrol are added. Healthy example: `()` → 3 hypotheses. **The ranking for CHD is unchanged once the outcome is excluded.** Without the exclusion, every score exactly doubles. | Body edit at patient_profile.metta:286-287 (union in conditions). Add `(: PatientCondition (-> PatientProfile Outcome Atom))` after :93. Add a 4-argument `patient-relevance` overload that skips `$outcome`, called from personalized-score (:346-348). Add the payload key, builder emission into `shared_lines`, the PatientIn field, and `_PATIENT_FACT_RE` (patient_context.py:72-74). Do **not** map DIQ010. | 1 (PatientCondition; only ever in the patient stack, guaranteed since 282c854) or 0 via `InstanceOf` (type abuse, logical_predicates.metta:11) | M | grim_age_lu2019_evidence.metta:11, 44-49; mechanistic_bridges.metta:182, 191; pln_deduction.metta:115-117 | Margin: 64 padding heads OK in the patient stack. The full stack aborts on **one** PatientCondition atom (rc -6), and **appending** the rule after all files aborts 6 full-stack forms; inside patient_profile.metta it is byte-identical. Honesty: the KB treats these NHANES items as prevalence, "usable only as exclusion" (nhanes_baseline.metta:38-39; docs/nhanes_integration.md:212-218). CHD gets credited to IR or senescence because smoking does not reach CHD (lifestyle_evidence.metta:44-49). |
| 12 | **More bridges: uric acid → IR, WBC → ChronicInflammation, triglycerides → IR** | Uric acid (62F, 7.5 mg/dL): IR lever 0 → −0.857 y, Via (UricAcid). With the edge also in the patient stack: DeregulatedNutrientSensing is added and Berberine reaches Tier 1 on urate alone. WBC: with CRP, ChronicInflammation coverage 2 (conf 0.86), Omega3 0.239 → 0.404. TG: under an NHANES-percentile prior only ≥ about 2.4 mmol/L counts, so the tab example (190 mg/dL, z ≈ 0.81) changes nothing. | Same pattern as #7. Use the name `Triglycerides`, already reserved at nhanes_reference_etl.py:405-414. Urate needs a per-sex reference; medians differ by 83 µmol/L. TG has no MAD (not a LinAge2 feature) and needs a non-fasting caveat. | 0 | M each | Facchini 1991 PMID 1820474 (n=36, cross-sectional); Ruggiero 2007 PMID 17481443 (WBC as an inflammation marker; Danesh 1998 PMID 9600484 supports WBC→CHD, **not** ChronicInflammation→WBC); McLaughlin 2003 PMID 14623617. No anchor yet for neutrophil count. | Included in S1. Gates needed: urate (diuretics, CKD, diet); WBC (**smoking** — the default patient smokes — infection, steroids); TG (non-fasting). The urate chain implies "metformin lowers urate", which no anchor shows. |
| 13 | **Battery entries for the gaps** (regression guard) | Nothing visible in the app. It covers a witness-less abnormal patient (P6), RDW decomposition, the kidney lever returning empty, hypertension leaving the plan unchanged, and metformin (after #3). | scripts/linage2_battery.py: P6 text, PATIENT_LABEL :56-62, render prose :777-785. The hypertension comparison needs P1 **without** its existing "diagnoses: hypertension" line (patient_text.py:1118), as a special case like F2 (:641-647). | 0 | M | docs/linage2_battery/results.json (no RDW, kidney, BP or medication questions) | The battery is not run by pytest or CI. |

**Held but not quick** (deferred, not dropped):
- **Diabetes → InsulinResistance or Type2Diabetes.** The abort the scout reported does not happen when the edge is written in-file. But the edge is new curation, and the reader collapses type 1 and type 2 diabetes (patient_text.py:471). It also double-counts: an IR observation alone lifts Berberine 0.388 → 0.830, more than both labs together, and flips the top hypothesis.
- **A general low-direction stream** (sign-aware explains-obs). A 3-equation prototype works. It needs orientation tags (nhanes_common.py:440-460) and a separate stream so that tests/test_patient_grounding.py:117-122 still holds. Without orientation, CRP z −2 credits Autophagy.

## 2. Dropped candidates

- **Route generic patient-predicate programs to the patient stack** (skeptic: does not hold). Already shipped in `282c854`: patient_context.py:72-88, api.py:3088-3093, tests/test_patient_stack.py:172-199 (13 tests pass). Small residuals:
  - `(match &self (InstanceOf $x $t) …)` silently omits Caller_Me (asserted False at test :178).
  - /metta/run with a caller's own patient `extra_atoms` plus such a program still aborts (api.py:3573).
  - docs/linage2_integration.md:222-223 is stale.
- **Diagnoses as `;;` comment lines in the patient atoms** (skeptic: does not hold). validate() tokenizes comments (metta_validator.py:57, 161-182). Every patient query would show Validation Issues, and /metta/run would return 422. The payload does not carry diagnoses at all (patient_text.py:326-330). Use prose in patient_prompt_section instead (#9/#10), or make validate() strip comments. That would also fix the tab download header (patient_tab.py:190-192), which already fails validation.
- **Rejected by design** (the skeptics confirmed the rejection):
  - MeasuredZ for every lab without an edge: no answer changes, the builder refuses them (patient_builder.py:332-340, MAX_MARKERS 40). S2 removes the latency objection, but the "no edge" objection stands.
  - Diabetes as an implied HbA1c z: a fabricated measurement.
  - Generic condition facts: inert except CHD.
  - Cotinine in the shared layer: the KB refuses cotinine → pack-years (linage2_core.metta:262-266).
  - LinAge2 per-sex witness z as a standalone win: it is only a prerequisite, folded into #6 and #7.

## 3. Constraints any implementation must respect

1. **Edge placement.** New Effect edges go **inside** `mechanistic_bridges.metta`; S1 measured the combined set. Never put them in `extra_atoms` in the full stack: one edge with one new symbol aborts `infer`, `rank-interventions` and `decompose-grimage` (rc -6, trie.rs:179). Never put them in a separate appended file either (aborts). No stack exists only for the patient (pln_runner.py:149-151). After any KB edit, re-run tests/test_patient_stack.py: 8 answer controls and 3 abort controls.
2. **Readouts.** MeasuresBiomarker facts go in linage2_core.metta §5. They are pinned by tests/test_linage2.py:106 and :715 and the comment at :245.
3. **What the shared forms read.**
   - Only z > 1.0 (strict) and Pos chains count; the size of z beyond the threshold is ignored.
   - Labs that are harmful when low need a sign-flipped node.
   - Risk reads AgeAccelGrim only; decomposition and the GrimAge counterfactual read the DNAm components only. No tab lab reaches these three forms.
4. **Builder and API.**
   - MARKERS refuses unknown names (patient_builder.py:332-340).
   - `Reference` has no sex field (:58-90).
   - kb_markers emits only CRP, HbA1c and fasting glucose (patient_text.py:279-312).
   - Any new payload key also goes into PatientIn (`extra="forbid"`, api.py:1066-1084).
   - Any new patient head goes into `_PATIENT_FACT_RE` (patient_context.py:72-74).
5. **Head budget.**
   - Patient stack: +32 padding OK, +64 OK, 96 new data heads abort. 400 new `=` function names are fine.
   - LinAge2 stack: margin is +16/+32. CurrentMedication would be new there, so keep it in `shared_atoms`.
6. **Latency.** Without S2, about 12-13 extra markers exceed the 60 s timeout. With S2, +32 markers take 30.6 s. Never send labs that have no edge.
7. **Honesty.**
   - Never fabricate a MeasuredZ.
   - A z sent as `{"z":…}` skips the "not age-adjusted" warning (patient_builder.py:635-655), so a young-reference z needs a provenance note.
   - Strengths are labelled curated priors (mechanistic_bridges.metta:11-21).
   - LinAge2 lever years are strength × contribution years.
   - The two clocks are never combined.
8. **Double counting.** A condition observation double-counts the ranked outcome. An IR observation double-counts HbA1c and glucose.
9. **Reader defects found.**
   - `". "` is not a statement separator (patient_text.py:549-577), so a typed glucose can be lost.
   - "<condition> on <drug>" blocks the build.
   - "I have type 2 diabetes, hypertension and…" blocks the build.
10. **Things that already abort or fail on HEAD.**
    - `!(human-evidence &self Metformin)` aborts the full stack.
    - A full-stack query naming any unknown symbol (e.g. RedCellDistributionWidth) aborts.
    - The tab download header fails validation.
    - The docstring at tests/test_smoking_lever.py:380-381 ("+5 defs abort") is stale.

## 4. Open questions for the deeper research

1. **The z policy (your call).** Status quo, clinical floors (cite AHA/CDC 2003 for CRP and ADA vs WHO for FPG), young reference, or age-median, per marker and per sex. See S4: age-median drops six-labs' CRP; the young reference hides CRP in women.
2. **Bridge strengths and gates.** The 0.5/0.6 priors directly set the LinAge2 years. Choose clinical thresholds for RDW (> 14.5 %?) and albumin (< 35 g/L?), and a name for the deficit node. Decide the anaemia gate for RDW, the smoking gate for WBC, and the diuretic/CKD gate for urate. Find an anchor for neutrophil count.
3. **The S2 fix.** Confirm `unique-atom` semantics are stable across hyperon versions. Run the NHANES MeasuredRaw path with duplicate markers through the full test suite: order changes, and SupportedBy order might matter to a consumer.
4. **Conditions.** PatientCondition (1 head) or InstanceOf (0 heads, type abuse)? How should the answer disclose hallmark-only abduction? Should an IR observation be dropped when a downstream marker already witnesses it? How should type 1 and type 2 diabetes be split before any DIQ010 use?
5. **"Already taking".** Should rank-interventions-for-patient and the LinAge2 counterfactual demote or annotate an intervention the person already takes?
6. **Live translator checks** for #1, #5 and #10. They need an OPENAI_API_KEY; the battery stubs the translator.
7. **The low-direction stream.** Orientation tags (HigherIsWorse/HigherIsBetter, nhanes_common.py:458-460) and how they meet `explains-obs` and `relevance-on`.
8. **The MeasuredZ type.** It is declared `(-> PatientProfile Biomarker Number Atom)` (patient_profile.metta:70), but new lab symbols are not Biomarkers in the patient stack. It is unenforced today. Should `(Inheritance X Biomarker)` be added with each bridge?