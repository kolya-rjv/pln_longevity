# Research brief: what can the "My Patient" tab feed into the rest of the knowledge base?

Run this in a local Claude Code thread on branch `linage2` (after applying the handover
bundle; tip includes `282c854`). It is a **research task with small prototypes**, not a
build-out: the output is a ranked, evidence-backed recommendation, plus the two or three
cheapest wins prototyped behind tests if the user agrees.

**Be economical.** A scouting pass already ran (5 scouts with live MeTTa probes, 22 of 37
leads re-checked by a skeptic who re-ran the probe). Its full output is
`docs/kb_quick_wins/scouting.json` — every lead with evidence (file:line), the probe it ran
and the result, what is missing, risks, and the skeptic's verdict and corrections. **Start
from it; do not re-scout from scratch.** Re-probe only what you are about to recommend.

## 1. The question

A person types a few lines (age, sex, smoking, ~57 lab values, diagnoses) into the My Patient
tab (`pln_chat/patient_tab.py`, reader `pln_chat/core/patient_text.py`). Today only a sliver
reaches the shared KB that the non-LinAge2 layers reason over — `(PatientAge …)`,
`(PatientSex …)`, `(PatientSmoking …)` and `(MeasuredZ P <marker> z)` for CRP, HbA1c,
FastingGlucose (only when "fasting" is typed), and AgeAccelGrim (only if typed). Everything
else — albumin, RDW, creatinine, NT-proBNP, lipids, blood counts, BP, BMI, every typed
diagnosis — feeds only LinAge2. **Which small changes would let the shared layers (diagnosis,
supplement plan, intervention ranking, risk) say something more, and honestly, about a typed
patient?**

## 2. What the scouting established (re-check before relying on it)

How patient data is consumed:
- Diagnosis, supplements and ranking read one thing: `patient-observations`
  (`patient_profile.metta` ~277-287) = markers whose z > `elevated-z-threshold` (1.0). Only
  **Pos** causal chains explain an observation (`pln_abductive_diagnosis.metta` ~43-45). **Low
  values are never used**, and **the size of z above 1.0 does not matter** (z 1.07 and 2.0
  gave byte-identical answers). Risk reads AgeAccelGrim only; GrimAge decomposition and
  counterfactuals read the DNAm components only — no typed lab reaches those three.
- Of the tab's labs, **only CRP, HbA1c and FastingGlucose have any path** to a conclusion.
  None of the other ~54 has an edge in the patient stack. So "add markers for labs that
  already have cause edges" is already done; anything more needs **new curated edges**.
- Injecting `MeasuredZ` for labs without edges changes nothing but query time (probed).

Binding constraints any recommendation must respect:
- **Latency, not head symbols, is the first wall.** `recommend-supplements-patient` took
  ~5 s with the tab's 3 markers, then ~17 / 40 / 66 s with +4 / +8 / +12 markers (app timeout
  `PLN_QUERY_TIMEOUT_SECONDS=60`). The cost sits in the recursive tuple helpers
  (`patient_profile.metta` ~191-211) and observations recomputed per supplement. Keep added
  witnesses to ~3-5, or first make `patient-observations` computed once per query.
- **Head-symbol budget.** In the *patient stack* a new head and 32 padding heads were fine.
  In the *full shared stack*, any atom appended via `extra_atoms` can abort (hyperon
  `trie.rs:179`), and a new node wired into the causal graph (e.g. a `Type2Diabetes` node
  with an Effect edge) aborts the most common generic questions (`infer Metformin CHD`).
  New edges must be written **in the .metta files** and tested **in the full stack**
  (`tests/test_patient_stack.py` shows the subprocess pattern); never validate via
  `extra_atoms` there. Since `282c854` a session patient never enters the full stack.
- **Honesty.** `MeasuredZ` is documented as a standardised measurement; never fabricate one
  (e.g. diabetes → fake HbA1c z). New edges follow the honesty contract at the top of
  `mechanistic_bridges.metta` (strength + evidence tier, anchored).

## 3. Leads, by kind (✓ = re-checked and held; · = not re-checked)

| lead | kind | new heads | effort | status |
|---|---|---|---|---|
| Witness **z policy**: today HbA1c 5.9 %, fasting glucose 105 mg/dL, CRP 5 mg/L all read Normal (coarse priors in `patient_builder.MARKERS`); at z>1 the metabolic / inflammation axes light up | policy | 0 | S | ✓ (×3: HbA1c, FPG, CRP) |
| **Default cause list** for "what drives my abnormal labs?": the only diagnose few-shot has no metabolic cause, so the answer is empty for the tab's own example | prompt / rule | 0 | S | ✓ |
| **Medications** → `(CurrentMedication P Metformin)` (head exists): the supplement plan then flags Berberine + Metformin | reader + builder | 0 | S | ✓ |
| **Prevalent-CHD guard**: a person who typed CHD / angina / MI still gets an *incident* 10-year CHD risk | Python note | 0 | S | ✓ |
| **CHD-family diagnoses as an observation** (MCQ160C/D/E → CoronaryHeartDisease), via existing heads or a patient-stack-only `PatientCondition`; must exclude the ranked outcome (double count found) | rule edit + builder | 0-1 | M | ✓ |
| **New curated bridges** for RDW, WBC → ChronicInflammation; uric acid, triglycerides → InsulinResistance; + MARKERS entry, reference, `kb_markers` mapping, and an identity `MeasuresBiomarker` so LinAge2 years get a cause | KB edge + builder | 0 | M each | ✓ (corrections in scouting.json: e.g. RDW alone moves the CI lever −2.7 y, not −6.1) |
| **Low albumin** as an elevated `Hypoalbuminemia` witness (sign-flip convention) | KB + builder | 0 | M | ✓ (with design caveats) |
| **Low-direction findings** in general (albumin, HDL, Hb, lymphocyte %) | shared rules | 0 | L | ✓ structural, not quick |
| Patient-aware suggested questions; "why does my <top unexplained lab> add years?"; translator rules for labs/conditions with no KB relation | UX / prompt | 0 | S | · |
| Witness z from LinAge2's own per-sex reference | policy | 0 | S | · (scouts disagree: young ≤50 reference overcalls older people; non-fasting glucose reference; undercalls CRP in women) |
| NOT: MeasuredZ for every lab; diabetes as an implied HbA1c; a Type2Diabetes node in shared files; diagnoses as comment lines (breaks validation) | — | — | — | rejected |

## 4. Questions to answer

1. **Rank** the leads for a *demo* by: value to the person (a question that was empty now
   answers, or a LinAge2 year gets a cause) × effort × risk. Re-probe each recommended lead on
   current HEAD; quote the before/after answers.
2. **Witness z policy** — the single cheapest lever, but a clinical decision: lay out the
   options with numbers (clinical floors such as HbA1c ≥ 5.7 %, FPG ≥ 100 mg/dL, hs-CRP > 3
   mg/L; LinAge2 per-sex/per-age references; status quo) and what each does to the three
   example patients and to women vs men. **Ask the user to choose; do not pick silently.**
3. **Latency**: can `patient-observations` be computed once per query (memoised / passed
   down) to lift the ~3-5 marker ceiling? Measure before and after on Patient001/002 and the
   tab examples; built-in patients' answers must stay byte-identical.
4. **Each new bridge**: verify its anchor in PubMed (the scouts proposed PMIDs — check them),
   set strength and tier per the honesty contract, and decide whether it needs gating
   (RDW also rises with iron/B12/folate deficiency; WBC with infection and smoking; urate with
   diuretics and CKD). Prove it is byte-neutral for built-in patients and does not abort the
   full stack (file edit, full-stack subprocess test).
5. **Conditions**: choose the encoding (existing heads vs a patient-stack-only
   `PatientCondition`), fix the ranking double count, and decide what to say about
   hallmark-only abduction (CHD gets credited to InsulinResistance / senescence, not to
   smoking or LDL).
6. **Make it visible**: which suggested questions, translator rules (`pln_chat/prompts/
   system_prompt.txt`, few-shots) and battery entries (`scripts/linage2_battery.py`) show each
   win to a person?

## 5. Deliverables

- `docs/kb_quick_wins/REPORT.md`: the ranked recommendation, each with probe evidence
  (query, atoms, before/after), files to change, tests to add, risks, and an effort estimate.
- With the user's go-ahead only: prototypes of the top 2-3 **S-effort** wins on a branch off
  `linage2` (e.g. `linage2-kb-quickwins`), each with tests and, where visible to a person, a
  battery entry. Nothing in shared rules or `mechanistic_bridges.metta` without the full-stack
  test and built-in byte-identity.

## 6. Coordination

A separate thread is implementing model extraction for the same tab
(`docs/patient_extraction/design.md`); both touch `pln_chat/core/patient_text.py`
(`kb_markers`) and `pln_chat/core/patient_builder.py`. Keep this research's code changes on
its own branch and small, so the two merge cleanly.

## 7. Practicalities

- Probe pattern: run MeTTa in a **subprocess** (an abort must not kill the session) with
  `patient_stack(api._runtime_kb_paths())` for patient questions and the full runtime stack
  for generic ones — see `_run` in `tests/test_patient_stack.py`. Probes take ~1-15 s.
- Tests: `pytest tests/test_patient_stack.py tests/test_patient_tab.py
  tests/test_patient_text_corpus.py -q`; full suite with
  `--deselect tests/test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move`
  (33 failures pre-exist on a clean `909ee40` in the cloud container — check yours).
- Battery: `python scripts/linage2_battery.py --date <date>` (47 pass, 0 fail, 5 live today).
