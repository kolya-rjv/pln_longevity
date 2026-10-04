# Reading the My Patient text with a model — design (v3)

Status: **designed, not built.** Committed so far: `pln_chat/core/patient_vocabulary.py`
(the vocabulary, generated from the KB, with `drift()`, tested in
`tests/test_patient_vocabulary.py`). Everything else below is the plan. Two earlier
versions were critiqued adversarially before any extractor code was written
(`design_critique.json`); §10 says what each critique changed.

## 0. Why

`pln_chat/core/patient_text.py` (`read_patient_text`) reads the tab's free text with
regular expressions. Four adversarial review rounds found a long tail of phrasings it reads
as a confident, wrong patient, or refuses although they are clear — round 4 alone confirmed
61 findings (47 major) plus 12 from a completeness pass (`review_round4.json`). Smoking alone
is ~8.8 LinAge2 years (cotinine level 0 vs 3). Hand-written rules will not close that tail;
a language model reads free text far better. But the KB must stay consistent and the tab's
contract — **refuse rather than guess** — must hold.

## 1. Principle: the model rewrites, the rules read

The model never produces a value. For each statement the rules did not understand, it
proposes a **canonical line in the rules' own grammar** ("albumin 4.1 g/dL", "former smoker,
quit 2010", "diagnoses: hypertension", "no diabetes", "58 year old", "female"), built by
code from the model's enum choices plus numbers and units **copied verbatim from the quote**.
The rules then read the resulting **"read as" text** — the only path from text to values,
exactly as today. The person sees the "read as" text before anything is built.

Consistency with the .metta files is therefore by construction:
- every value that reaches MeTTa was read by the rules (`read_patient_text`) and built by
  `core.patient_builder.build_patient`, unchanged;
- every enum the model may choose comes from `core.patient_vocabulary` (smoking statuses from
  `patient_profile.metta`; the 59 LinAge2 inputs from `linage2_core.metta`; lab name-groups
  and the 23 conditions from the reader's tables and `data/linage2/linage2_model.json`), and
  `drift()` fails a test if those sources part;
- a **round-trip test** over the whole vocabulary asserts `rules(render(item)) == item`, so a
  canonical line always means, to the rules, exactly what the model chose.

## 2. Flow

1. **Rules first.** `read_patient_text(text)` as today, extended so every statement and every
   problem carries its span `(line, start, end)` in the original text and every problem a
   kind: `not_understood | ambiguous | contradiction | someone_else | vaping | unit |
   lost_condition | missing` (`all_problems()` still returns strings; tests and API unchanged).
2. **Model (only from the Read button, only if configured).** The model receives the text as
   numbered statements (with the rules' reading of each, if any) and the vocabulary, and
   returns, per statement it can read, items: `{statement_id, quote, kind, <enum fields>}` —
   no numbers, no units, no MeTTa.
3. **Verify and render (code).** For each item, code checks (§4) and renders canonical lines,
   copying the number and unit text out of the quote with the rules' own regexes.
4. **Substitute, suggest, or flag.**
   - Statement the rules marked `not_understood` → replaced by the canonical line(s) in the
     "read as" text, marked "· model" in the read table.
   - Statement the rules refused with a **judgement** (ambiguous, contradiction, unit,
     vaping, someone_else, lost_condition) → never replaced; the canonical line is offered as
     a **[Use this wording]** suggestion that writes it into the textbox and reads again.
   - **Any smoking status from the model alone** → a suggestion, never a silent substitution.
   - Statement the rules read and the model reads differently → smoking, age, sex: blocking,
     with both canonical options as buttons; labs and conditions: a note. A statement whose
     text already equals its canonical form is authoritative (typing the canonical phrase
     always resolves a disagreement).
5. **Rules read the "read as" text** → the final `ParsedPatient`. Text, "read as" text and
   `ParsedPatient` go into the tab's `gr.State`.
6. **Build** uses that state, never calls the model, and refuses ("the text changed since
   Read — press Read") if the textbox differs. The download header carries the "read as"
   text, so the patient can be rebuilt offline by the rules alone.
7. **Failure** (no key, network, timeout, `message.refusal`, `finish_reason != "stop"`, schema
   error) → no substitutions, and a header line names the reason: "Read by rules only — no
   OPENAI_API_KEY" / "— model timed out after 20 s" / "— model declined". A 400 on
   `response_format` or `temperature` is shown as a **configuration** error, not a fallback.

## 3. The model's output schema (OpenAI structured outputs, strict)

- every object: all properties required, `additionalProperties: false`; nullable enums list
  `null` inside the enum; **`quote` is the first property of every object** (keys are
  generated in schema order: copy first, claim second); readable enum keys mapped to codes
  in Python;
- item kinds and their enum fields:
  - `lab {lab_group}` — one key per alias group of the reader's `_ALIAS_INDEX` ("urea",
    "urea nitrogen (BUN)", "lymphocytes" …), never an NHANES code: code resolves with the
    group's whole spec list, so %/count and urea/BUN ambiguity refusals still apply;
  - `cotinine {}` — its own path (§4);
  - `smoking {status: NeverSmoker|FormerSmoker|CurrentSmoker|unclear, occasional: bool,
    other_nicotine: none|vaping|nicotine_replacement|smokeless|cannabis|secondhand}`;
  - `condition {item: <23 keys>, answer: yes|no|borderline}`; `no_other_conditions {}`;
  - `sex {value: male|female}`; `age {}`; `weight {}`; `height {}`;
  - `self_rated_health {value}`, `health_vs_year_ago {value}`, `healthcare_visits {period:
    year|month|week|unstated}`; `grimage {direction: older|younger|signed|unstated, wording:
    acceleration|clock_age|unstated}`;
  - `someone_else {}`; `unclear {topic: smoking|condition|lab|other, why}`.
- the prompt carries, per enum value, the reader's label, aliases and accepted units; per
  condition the NHANES 1999-2002 question wording, exclusions and time window (DIQ010
  excludes gestational diabetes; KIQ020 is weak/failing kidneys, not stones or infection;
  MCQ160K is chronic bronchitis; MCQ160F excludes TIA; OSQ060 excludes osteopenia; HUQ070 is
  an overnight hospital stay in the past 12 months); cotinine as "a serum cotinine test
  result in ng/mL, only when a value is written" (never the KB's digitised 0-3 wording);
  static instructions and vocabulary first, the person's text last (prefix caching).

## 4. Verification rules (code, before rendering)

- **Grounding:** normalise text and quote the same way (casefold; unicode dashes, quotes,
  apostrophes; spaces within a line; ignore punctuation-only differences). The quote must
  have ≥3 non-space characters, occur **exactly once**, inside **one line and the item's own
  statement**. Otherwise the item is discarded (listed in a collapsed debug section) and
  counts as "the model said nothing" — which never blocks.
- **Someone else:** a quote in a statement the rules set aside, matching `_SOMEONE_ELSE`, or
  overlapping a `someone_else` item, is discarded; a rules reading inside a span the model
  marks someone_else becomes a note-level disagreement (blocking for smoking).
- **Lab:** an alias of the chosen group must be in the quote; exactly one number after it
  (only `[:=]`/is/of/was/level between); the unit text right after it is copied verbatim.
  A second number for the analyte that is not an `a-b` reference range → no rendering.
- **Cotinine:** a value with `ng/mL|µg/L|ug/L` directly after it, or the word 'level' with
  0-3; any comparator (<, >, less than, undetectable, negative) → no rendering (or `<N` with
  N ≤ 10 → "cotinine 0 ng/mL"). A rules cotinine refusal is never replaced.
- **Smoking:** the quote must contain a smoking-topic word; `unclear` or
  `other_nicotine != none` → a smoking problem, no suggestion; the rendered phrase carries
  intensity only as "occasional smoker" when an occasional word is in the quote.
- **Condition:** the quote must contain one of the item's reviewed terms (the rules'
  `_DIAGNOSES` pattern plus synonyms, generated in `patient_vocabulary`) and must not be a lab
  or vital line; polarity is computed by code (leading negation over the whole comma/'or'
  list: no|not|never|denies|negative for|without|free of; trailing: ruled out|excluded|
  resolved|negative|?|: no) and must equal the model's answer; per-item exclusions and time
  windows block; `borderline` only for DIQ010 and only with `pre-?diabet|borderline diabet`
  in the quote; an inferred item (from a lab or medicine) becomes a note: "the model
  inferred prediabetes from 'HbA1c 6.4 %' — write 'prediabetes' if a doctor told you so".
- **Sex:** only an explicit token (`_SEX_RES` words, `sex: m/f`, `58M`, `58 y/o F`), never
  inferred from husband/wife/PSA/pregnancy. **Age:** an age phrase; number copied.
- **Weight/height:** number followed by a literal unit; height with feet/inches parsed by the
  rules' own arithmetic. **Health/trend:** the scale word in the quote, not negated.
  **Visits:** the count (or once/twice/none) in the quote; per-month/week converted.
  **GrimAge:** 'accel'/'AgeAccelGrim'/explicit sign/older|younger required; else no rendering.
- **Unclear:** blocks only for topic smoking, or a condition among the 23; otherwise a note.

## 5. Rules hardening still needed

The rules stay the only reader, so their confident-wrong readings still matter — but the
cheap fix is now to make them **stricter**, not cleverer: turn each confident-wrong case into
`not_understood` (the model then offers the canonical line) or into a judgement refusal.
From `review_round4.json`, at least: a non-tobacco object of "smoke" (marijuana, weed,
cannabis, pot, CBD); a zero count ("0 cigarettes a day"); past-tense quantities ("smoked a
pack a day for 40 years"); unrealised quitting (tried/need/should/never quit, on patches to
quit); form answers meaning No (Nil, negative, denies, false, 0, '-'); "lives/works with a
smoker"; a smoking clause with leftover words after `_SMOKING_DETAIL` (block, don't file as
not understood); a leading negation over a whole condition list ("negative for diabetes,
hypertension"); "No diabetes. Hypertension." sentence style; set-aside lines that also state
the person's own facts; age phrases inside other clauses. Add each to
`tests/test_patient_text_corpus.py` with its expected outcome.

## 6. What the person sees

- No new column: rows read by the rules alone stay byte-identical (the battery pins them). A
  row that came from a model rewrite gets "· model" in the Check cell, with its quote in "You
  typed"; the who line names the source for smoking, age and sex.
- One header line: "Read by rules + <model>" / "Read by rules only — <reason>".
- Suggestions and disagreements name both readings and offer **[Use this wording]**.
- The initial render and the example buttons are rules-only; only the Read button calls the
  model ("Read sends your text to OpenAI (<model>)" next to it). Own client:
  `PLN_EXTRACT_TIMEOUT_SECONDS` ≈ 20, `max_retries=0`; `read_btn.click(concurrency_limit=4)`.

## 7. API

`PatientTextIn` gains `reader: Literal['rules','model'] = 'rules'` (old bodies stay valid);
the response adds `read_as_text`, `reader_used`, `model_error` and a per-statement `source`.
`reader='model'` needs a configured key and text ≤ ~4,000 characters (else 413). Update the
"no LLM" wording in the field description, `pln_chat/API.md`, and bump `PLN_API_VERSION`.

## 8. Model configuration

`PLN_EXTRACT_MODEL` with its own default, checked at startup against an allowlist of models
that support structured outputs (gpt-4-turbo does not); omit `temperature` for GPT-5-family
reasoning models; check `message.refusal` first and treat `finish_reason != "stop"` as
truncated; pin `openai>=1.40`; cache by sha256(model | prompt hash | schema hash | text),
failures included, LRU ~128 — a cost saver only, never the Read/Build guarantee.

## 9. Verification

1. **Round-trip test** over the vocabulary: `rules(render(item)) == item`.
2. **Schema test:** walk the generated schema and assert the strict-mode rules; validate a
   recorded all-null and a full extraction with `jsonschema`.
3. **Recorded extractions** (JSON fixtures) for grounding, rendering and substitution.
4. **Hostile-model property test:** for every corpus entry expected refused (X / ok=False),
   every schema-valid grounded output over that statement leaves it refused without a click;
   for every ok entry, a grounded but disagreeing output gives a note or a block, never a
   flipped value.
5. **No live calls in tests:** keep `read_patient_text` rules-only and pure; add
   `read_patient(text, extractor)` used only by the Read button and `reader='model'`; an
   autouse conftest fixture makes the OpenAI extractor raise unless `PLN_LIVE_EXTRACT=1`. The
   battery stays rules-only and prints "reader: rules".
6. **Live eval** `scripts/eval_patient_extraction.py` (needs `OPENAI_API_KEY`): one
   schema-acceptance call first (exit non-zero on a 400); then the corpus and every
   reproduction in `review_round4.json`; reports rewrite rate, agreement with expected
   outcomes, p50/p95 latency and tokens; FAILS if any expected-refused text becomes usable
   without a click, or any text came back rules-only.

## 10. Data fixes the vocabulary work exposed

- `linage2_core.metta`: the URXUCRSI structured comment says mmol/L; NHANES and the reader use
  µmol/L. `linage2_model.json` descriptions: URXUCRSI (mmol/L → µmol/L), LBXCRP (mg/L →
  mg/dL), LBXCOT (0-2 → 0-3); its `nhanes_range` for LBXCOT is the digitised 0-2.
- Extend `drift()`: each KB description's unit normalises into the reader's canonical unit;
  `cotinine_levels` keys == {0,1,2,3}; `_HEALTH`/`_TREND`/`_visits_category` values ⊆
  `linage2_model._ANSWER_CODES`; `HEALTH_ITEMS == FS2_ITEMS + FS3_ITEMS`;
  `set(linage2_model.FS1_ITEMS) | {'MCQ160B'} == vocabulary().conditions` (23 read, 22
  counted — MCQ160B is read but not counted, as upstream); one vocabulary lab entry per alias
  group, every alias and unit present.

## 11. What the critiques changed

- v1 (five-row merge of two readers): a grounded model reading could replace any rules
  refusal; substring grounding; NHANES-code lab enum; cotinine through the lab path; a model
  intensity; condition answers, sex, fasting, GrimAge sign and health scale taken from the
  model; immediate Build; tests only with a careful model.
- v2 fixed each of those inside the merge (span-local grounding, typed problems, term lists,
  computed polarity, confirmation, property test).
- v3 (this) takes the critique's simpler route: the model only rewrites unread statements
  into the rules' grammar; the rules are the single reader; judgement refusals and
  model-only smoking become suggestions; Build reads the stored "read as" text. All of v2's
  per-field checks survive as the verification rules of §4.
