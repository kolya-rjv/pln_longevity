# Model reader — how the evaluation moved the build

`scripts/eval_patient_extraction.py` reads every corpus entry
(`tests/test_patient_text_corpus.py`, each with its expected outcome) and every
reproduction quoted in `review_round4.json`, by the rules alone and by
`read_patient` with the live model. It fails if an expected-refused text becomes usable
without a click, or if any text comes back read by the rules only. The latest full report
is `eval_gpt-6-luna.md` (and `.json`); the raw extractions of that run are
`tests/fixtures/patient_extractions.json.gz`, which `tests/test_patient_read.py` replays
offline. All runs: `gpt-6-luna`, `reasoning_effort: low`, 8 workers.

| run | code | texts | gates | corpus usable read as expected | corpus refused kept | p50 / p95 |
|---|---|---|---|---|---|---|
| 1 | verification + first prompt (`baaf99f`) | 1,302 | pass | 166/185 (90%) | 52/52 | 2.3 s / 3.5 s |
| 2 | prompt tuned, rules hardened (`3414843`) | 1,429 | pass | 212/217 (98%) | 147/147 | 2.4 s / 3.5 s |
| 3 | run-2 catches closed in the rules, prompt tuned (`05d1387`) | 1,463 | pass | 218/218 (100%) | 180/180 | 2.5 s / 4.0 s |

**Run 1.** Both gates held, but the model blocked 19 of 185 usable corpus entries: it
answered `unclear` for settled conventions ("smoker since 1990", "trying to quit",
"doesn't smoke", "denies smoking", "tobacco: none") and read "never vaped" as vaping. The
prompt now states the reader's conventions; the check treats a negated nicotine word as
none. The run also showed why the rules had to change: they read 393 of 815 round-4
texts as usable while the model blocked them, nearly all correctly ("I never quit
smoking", "smoked a pack a day for 40 years", "Smoker: Nil").

**Rules hardening (step 8).** Each confident-wrong round-4 case became a refusal, or a
correct reading, and was pinned in the corpus (§5 of the design).

**Run 2.** 5 over-blocks left, all the model being pedantic ("hospitalized" without
"overnight", "broken wrist" without a doctor, COPD not named as emphysema, "never vaped",
"Smokeless tobacco: never used"); fixed in the prompt (stated conditions count, common
names per condition) and the check (an `unclear` smoking item needs a tobacco word). The
model blocked 64 round-4 texts the rules still read confidently wrong ("smoker, quit
approximately 2010", "kicked the habit in 2010", "smoke free 10 yrs", "Light tobacco
smoker", "about to quit smoking"); each was closed in the rules and pinned in the corpus.

**Run 3.** No usable corpus entry blocked, every expected refusal kept; the model
rewrote 46 of the 226 texts with a statement the rules did not fully understand (the
rest are refusals, where its reading is a wording to click — 656 texts got one — or
statements with nothing about health), and blocked 9 round-4 texts the rules still read
as usable, most of them rightly ("smoker, quit during covid", "former smoker, quit 2010
by vaping", "patches"). 142 of 4,612 items failed the checks, mostly quotes about someone
else or without a smoking word. This run's extractions are the replay fixture.

**After the adversarial review** (`46be389` and the commit after it). Three independent
reviewers, each playing a hostile model, got confident wrong patients through the model
route — a rewrite that dropped words its quote did not hold ("Non-HDL", "retired at 65
years old", "… but is 6.0 % now"), quotes cut inside a number or word ("glucose 10" in
"glucose 105", "diabetic" in "nondiabetic"), a lenient second item covering those words, a
combination of lines flipping a combined answer, a missing age filled from "male, 82 kg" —
and one hang. All are fixed and pinned (REVIEW_A / REVIEW_B in
`tests/test_patient_read.py`): every quote must be accounted for by its kind's grammar, a
rewrite needs its quotes to hold the whole statement, a partial reading is a note and never a
button, and a final check refuses any rewrite that changes a value the rules read. The prompt
did not change, so run 3's recorded extractions replayed under the final code show what the
fixes cost: all 218 usable corpus entries still read as expected; rewrites 40 (46); texts the
rules read as usable that the model blocks 6 (9); texts with a wording to click 341 (576),
the rest of the partial readings shown as notes.

**Cost per Read.** About 5,400 prompt tokens, of which about 99% are a cached prefix (the
instructions and the vocabulary come first, the person's text last), and about 165
completion tokens (about 100 of them reasoning).

## Test baseline on this machine

Full suite at `7b8f15d` (the hand-over commit), before any of this work:
**33 failed, 1100 passed, 3 skipped** (one test deselected:
`tests/test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move`, which
aborts the process). The 33 are the same 33 the cloud session saw fail on a clean
`909ee40`: test_api.py 16, test_api_operations.py 12, test_combined_app 1,
test_human_evidence 1, test_prompt_size 1, test_runtime_grounding 2. 27 fail with
`TypeError: …run_inline() got multiple values for argument 'function'` — the tests'
inline-threadpool monkeypatch against this FastAPI's `run_in_threadpool`;
test_human_evidence on hyperon's trie panic aborting the interpreter (exit -6, the known
abort); test_prompt_size on API.md examples answered with HTTP 500 (the same abort);
test_runtime_grounding on a missing `genes-affecting-senescence` form in the inventory
and a 422 that follows from it. None touches the patient reader.
