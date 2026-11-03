# W2 checkpoint: draft rows, the query slate, the cost model, and what W3 would run

Written 2026-10-26 at the checkpoint of LAUNCH_ULTRACODE.md W2; repo citations are to tip `d21ae14` (work tree
`atomspace-expansion-workflow-2cc380`). Drafts for the owner, not decisions. W2 used 29 agents: 3 slate proposers,
4 KB checkers, 3 judges, 1 synthesizer; 1 cost counter, 2 recomputers (no third: no expected-scenario line differed
by more than 20%); a budget proposer and a skeptic; for rows 2, 3, 6 and 7 two proposers and one judge-synthesizer
each (the brief's separate judge and synthesizer were merged to save agents); 1 harness drafter. No LLM was called;
nothing was installed into the project venv; the KB checks ran the repo's own code offline on hyperon with keys unset.

## §4 item 1: the 10 queries (draft for sign-off)

28 distinct candidates from the 131 collected queries; 10 slate, 5 alternates, 13 dropped with reasons. Judges'
scores are Ben's intent / measurable growth / honesty and coverage, each 0 to 5. "Adapted" means the recorded query
was patient-specific and is lifted to a population-level question with the same route.

| # | Source | Query | Form | Today | Scores |
|---|---|---|---|---|---|
| 1 | `docs/pipeline.md:288-303` demo 3, adapted | Could rapamycin lower GrimAge through DNAmGDF15, by way of mTORC1 inhibition, mitophagy and mitochondrial function? Show the chain. | `(explain &self Rapamycin DNAmGDF15)` | empty: three of four Effect legs missing | 5/5/5 |
| 2 | `docs/counterfactual_chat_test_queries.md:114` CF4, adapted | Does clearing senescent cells lower CHD risk more than reducing chronic inflammation does, and which is more certain? | `(infer &self CellularSenescence CoronaryHeartDisease)`; `(infer &self ChronicInflammation CoronaryHeartDisease)` | senescence (stv 0.357 0.291); inflammation empty, no warning | 4/4/5 |
| 3 | `docs/counterfactual_chat_test_queries.md:236` CF9 | If Patient001 took elamipretide, what GrimAge change would you predict? (with the population analogue `(explain &self Elamipretide DNAmGDF15)`) | `(counterfactual-patient &self Patient001 Elamipretide)` | empty: no Effect edge from Elamipretide | 4/5/3 |
| 4 | `docs/pipeline.md:255-270` demo 1, adapted | Given elevated GrimAge with elevated DNAmPAI1, DNAmGDF15 and CRP, rank the upstream causes by how well each explains them | `(diagnose &self (CellularSenescence InsulinResistance MitochondrialDysfunction ChronicInflammation) (DNAmPAI1 DNAmGDF15 ...))` | answers; cross-links absent | 5/3/4 |
| 5 | `docs/pipeline.md` demo 2, adapted | Rank rapamycin, metformin, D+Q and acarbose by expected effect on CHD, showing the chains | `(rank-interventions &self (Rapamycin Metformin DasatinibPlusQuercetin Acarbose) CoronaryHeartDisease)` | Acarbose fails the validator; candidates drop silently | 5/4/3 |
| 6 | CF6, adapted | Does metformin change GrimAge, and through which component, if any? | `(explain &self Metformin DNAmGDF15)`; `(infer &self Metformin DNAmPAI1)` | thin | 4/3/4 |
| 7 | SU8, adapted | Is the evidence that fisetin protects against CHD as strong as for metformin, and where does each chain get its confidence? | `(explain &self Fisetin CoronaryHeartDisease)`; `(explain &self Metformin CoronaryHeartDisease)` | one side from a curated EvidenceLevel only | 4/2/5 |
| 8 | CF8 | Which GrimAge components would a senescence-clearing intervention act on for Patient001, and through what route? | `(counterfactual-patient &self Patient001 CellularSenescence)`; `(explain &self CellularSenescence DNAmPAI1)` | answers (the only recorded `explain`) | 4/2/4 |
| 9 | Jasmine Smith (LAUNCH.md §1), adapted | Which supplements have human evidence for building or keeping muscle and bone density in a 35-year-old woman before menopause? | `(human-evidence &self Creatine)` and likewise; `(infer &self Creatine MuscleMass)` | no muscle, bone or menopause node, predicate or edge | 3/3/4 |
| 10 | LinAge2 battery LB36 | Why does my RDW add years? (with the population analogue) | `(linage-decomposition-patient &self Caller_Me)` | answers; causal side thin | 3/3/3 |

Alternates: 11 SU5 (berberine for Patient001; 3/4/2), 12 RP8 (risk certainty, senescence vs inflammation; 3/2/3),
13 CF5 (lower CRP, effect on GrimAge; 3/2/3; the swap for #2 if the gate refuses EmpiricalAssociation records),
14 RP5 adapted (senescence to CHD route; 4/3/3), 15 DA3 (trimethadione vs acarbose DrugAge ranking, kept as a negative
control for the DrugAge route; 1/0/3).

Open before sign-off: whether `check_entry` admits `EmpiricalAssociation` records (their heads are outside
`CANONICAL_PREDICATES` and `SCHEMA_EXEMPLAR`), on which #2, #14 and part of #5 depend; staged hallmark cross-links can
create cycles that hang legacy `infer`/`explain` (no visited set), so each staged edge needs a cycle check. DrugAge
ranking queries cannot show growth: staged rows are invisible to `drugage_selector`.

## §4 items 2 and 8: cost (gpt-5.4-mini; list prices fetched 2026-10-25/26)

Prices: OpenRouter and OpenAI list the same for gpt-5.4-mini ($0.75 / M input, $0.075 cached, $4.50 output);
gpt-6-luna $0.10 / $0.01 / $0.50; text-embedding-3-large $0.13. OpenRouter's credit-purchase fee was not fetched
(row 2's judge cites 5.5%, not verified here). Tokens by tiktoken 0.14.0, o200k_base, in a scratch venv.

| Line | Low | Expected | High | gpt-6-luna expected |
|---|---|---|---|---|
| L1 parser, one sentence | | $0.056 | | $0.0065 |
| L2 parser, one abstract | $0.066 | $0.39 | $2.99 | $0.045 |
| L3 parser, one full text (250 sentences expected) | | $14.08 | | $1.66 |
| L4 expander, one abstract | $0.003 | $0.013 | $0.038 | $0.0015 |
| L6 stage-1 recommendation calls | $0.22 | $0.84 | $2.49 | $0.10 |
| L7 stage 1, abstracts only (356 papers expected) | $19 | $144 | $1,502 | $17 |
| L8 stage 1, full text where open | $158 | $2,583 | $52,704 | $304 |
| L9 one stage-2 iteration (2 gaps, 15 abstracts) | $0.46 | $6.08 | $82 | $0.71 |

Drivers: reasoning tokens (the parser sends effort `high`; about 9,200 of the expected 10,868 output tokens per
sentence, about 85% of L1); the correction loop (numeric sentences cannot pass the format check, W1 finding 10);
parsing is about 30 times the expander per paper. Recomputers matched within 0.2% (the script compared recomputer 2
by line id; recomputer 1 used longer ids and was compared by hand). 27 assumptions (A1 to A27) and 9 unknowns are in
the scratch model; the largest unknowns are real reasoning-token counts, correction rates, cache-hit rates, and
whether the models accept the expander's and translator's temperature.

Budget proposal (draft): a dry run of one query and two papers, expected $0.91 (envelope $0.21 to $6.24), hard cap $3
measured from usage; stage 1 on abstracts with Layer B reading the 10,000-char full-text cut where open, expected
$144, hard cap $200, run one query per tranche and re-project after each; stage 2 at most 3 gaps and 30 papers per
iteration, 3 iterations and $25 per query. The skeptic found 12 problems that must be fixed before the dry-run ask
goes out, chiefly: a usage ledger outside the parser cannot see its 5 billed retries or its embedding calls, so Layer
A has no usage-based guard without changing the parser; two stop rules read signals today's code does not produce
(Layer A reaches no chainer source, and the HTTP response carries no chains, per-edge TVs or tiers); and the "no new
gaps" key rarely fires on paraphrased gap text.

## Draft rows

| Item | Draft decision | Owner question |
|---|---|---|
| 2 provider | Split: the parser runs unmodified on OpenRouter with a new prepaid, credit-capped key ($2 for the dry run), provider order OpenAI only, set only in the parser's process; translator, expander and gap call stay on `OPENAI_API_KEY`. Provider moves cost about 5.5%; model and effort move it about 8.5 times. | New prepaid OpenRouter key for the parser, $2 dry-run cap, translator and expander unchanged: yes / no / edit |
| 3 parser intake | Pinned checkout `d00331d` in a gitignored `.semparser/`, set up by a `scripts/setup_semparser.sh` like `setup_petta.sh`, its own Python 3.12 venv from a hash-locked trim of `requirements.txt` (no spaCy, no PeTTaChainer, no datasets, no pubmed-parser), called only through one driver as a child process (JSONL in and out); a stub `chaining` module so PeTTaChainer is never imported; sentences from PubMed's AbstractText by a deterministic splitter | May I clone, build that venv and run only an offline, keyless import smoke test: yes / no / edit |
| 6 equivalences | None in Layer A or Layer B: the wrapper calls `nl2pln` with `runs=1`, never `generate_equivalences` or `assisted_qa`, and logs them as off per run; similarity may at most rank candidate canonical names for the bridge LLM | Keep them out, with any paid precision test waiting for a separate yes: yes / no / edit |
| 7 staging | Not `CUSTOM_ONTOLOGY_DIR` (the shadowing hazard): per-loop append-only files plus `manifest.jsonl` under `build/atomspace_staging/<run_id>/`, rolled before 60,000 bytes and never dropped; written through `guard_ontology_write` with the schema gate; executed only in a PeTTa-only scoped stack built from the literal `_INFERENCE_STACK` minus `GENERIC_STACK_EXCLUDED`, re-anchored to the repo root; hyperon never sees them; promotion only by the owner | That layout and execution path: yes / no / edit |

Rows 4 and 5 are W3 and W4.

## What W3 would run after the go (drafted, never executed)

Harness in the scratchpad (`w3/harness/`: `w3common.py`, `gen_layer_a.py`, `measure_petta.py`, `probe_hyperon.py`,
`runner.py`; all parse with `ast`). It generates synthetic Layer A for 100, 300 and 1,000 abstracts from the
exemplar's statistics with planted 1-, 2- and 3-hop chains and a 0.7-cosine Equivalence variant, in the parser shape
and in the restricted library's shape; loads each on PeTTa through `pln_chat/core/metta_runtime.py` with
`PETTA_MEMORY_LIMIT_MB=16384`; times `match` and `PLN.Derive`; probes parser-shape hazards on PeTTa first; counts key
atoms and runs a hyperon subprocess probe with a bisection to the abort edge. Every child runs from the scratchpad
with `LOGS_DIR` and `TMPDIR` redirected, so nothing lands in either tree. Smoke run first (N=100, about 10 to 25
minutes), then the full run (runner, re-runner from the recorded commands, a third run on a gap over 10%, about 70 to
150 minutes):

```
cd <workflow worktree> && env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/w3/harness/runner.py all --sp $SP --sizes 100 --runs $SP/w3/smoke --no-third
cd <workflow worktree> && env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/w3/harness/runner.py all --sp $SP
```

Cannot be measured without the parser and a key: real parse sizes and vocabulary spread, correction-loop rates,
real cost per abstract, and how many real statements render to the library's shape.
