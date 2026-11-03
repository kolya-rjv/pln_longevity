# Atomspace expansion: an agentic loop of LLM search, semantic parsing and PLN reasoning

Launch prompt for a Claude Code thread on repo `pln_longevity`. Written 2026-10-25 from Ben Goertzel's
comment below and from a read of the code at `petta-migration` tip `d21ae14`; §1 and §2 were then checked
against the code by independent review on 2026-10-25 and eleven claims tightened. Start from that branch
(work tree `/Users/kolya/PycharmProjects/pln_longevity_petta`, or make a new work tree from it); it holds
linage2, the HTTP API, the PeTTa runtime adapter and patham9's PLN chainer. `main` on origin is stale.

## 0. The stakeholder's request (Ben Goertzel, verbatim)

> I would say it goes in two stages,
> 1) we need a seed of some texts giving enough basic information on the class of queries we are asking.
> Take a list of 10 queries we care about and ask an LLM, for each query, to recommend 10 papers that should
> be downloaded to help a reasoning system answer the query. That will give 100 papers to start with. Then
> for each of those 100 papers, ask the LLM what materials to download for the AI reasoning system to use as
> background for reasoning about what is in the paper. This might yield another 500 papers or something.
> 2) after that, the knowledge base should be expanded dynamically based on usage. Have the PLN system try
> to reason based on a query, and then after seeing what it comes up with, show an LLM the reasoning traces
> and ask it what additional information the reasoning system should have in order to come to better
> conclusion on that query. Then ask it to find 10 papers that may give it some of this missing
> information, and parse them into the Atomspace.
> So, more of an agentic workflow where LLM-driven web search, semantic parsing and PLN reasoning are
> wrapped in an agentic loop that is trying to understand things... rather than trying to identify a
> partitioned-off knowledge base that is going to tell us everything we need once and for all..

The semantic parser he has in mind: https://github.com/leungmanhin/semantic-parser (MIT, Python, pushed
2026-10-10, no README). Read it before designing; what it is, as of that date, is summarised in §2.

## 1. What already exists in this repo (do not rebuild it; read these first)

- **A paper-to-PLN expander with a schema gate.** `pln_chat/ontology/expander.py` turns an uploaded PDF
  (through pypdf) or any text file into canonical MeTTa with one OpenAI `chat.completions` call in
  `call_extraction_llm` (JSON-object response; the text is cut at 10,000 characters). `run_expansion_pipeline`
  takes the model and the temperature as required arguments and never reads `PLN_MODEL` itself; its two
  callers default to `config.DEFAULT_MODEL` (`PLN_MODEL`, else `gpt-5.4-mini`) at temperature 0.1:
  `POST /ontology/expand` (pasted `paper_text` only, no upload) and the Gradio "Ontology Expander" tab (the
  only surface that takes a PDF). `pln_chat/ontology/expansion_schema.py` gates every entry (`check_entry`;
  the target shapes are `SCHEMA_EXEMPLAR`). Its vocabulary is `allowed_predicates(inventory)`: the closed
  `CANONICAL_PREDICATES` list PLUS every head the runtime inventory already grounds, so a minted head such as
  `increases-life-span` is refused but a non-canonical head with existing facts (`PubMedID`, `HumanEvidence`,
  `Causes`; about 35 today) passes, and every production path passes an inventory. It also refuses a
  hand-typed two-float `(stv s c)`, a redefinition of a `CALIBRATION_FUNCTIONS` name or a bare number under a
  confidence-looking name, a tier `epistemic_calibration.metta` does not declare, and an entry with no
  PMID/DOI in its atoms or in the caller's identifier text (a `;;` comment does not count). Confidence is
  never proposed: `normalise_truth_values` rewrites every `Effect` truth value to
  `(stv s (evidence-confidence <Tier>))`, left unevaluated, so `epistemic_calibration.metta` stays the single
  authority. API.md section "Expanding the KB from a paper". Tests: `tests/test_ontology_expansion.py` (the
  taurine block that was once produced and is now refused is the fixture).
- **One write gate, which refuses but does not route.** `pln_chat/ontology/write_gate.py`'s
  `guard_ontology_write` only raises; the callers pick the target. `api._resolve_target_path`,
  `POST /ontology/apply` and the Gradio "create new file" box put a new file under `CUSTOM_ONTOLOGY_DIR`
  (`pln_chat/ontology/metta_files/`, absent on this branch) but can also name an existing root file. Without
  `PLN_ALLOW_CURATED_WRITES=1` the gate refuses (403) any file directly in the repo root, existing or new; it
  refuses a block over `PLN_MAX_APPLY_BYTES` (32,000 bytes) and an append that would take a file past
  `PLN_MAX_KB_FILE_BYTES` (60,000; a stack file over that is dropped from execution with no query-time error,
  though `GET /ontology/files` lists it as `excluded_from_runtime`). It re-runs `check_entry` over the block
  only for the caller-supplied block of `POST /ontology/apply`; the `/ontology/expand?apply=true` path and
  the Gradio "Apply to Ontology" button pass `schema_checked=True` and are re-checked for target and size
  only, and `run_expansion_pipeline(apply=True)` writes with no gate at all (both shipped callers pass
  `apply=False`). Two known defects to carry, not fix: the Gradio "create new file" name is not confined to
  the two ontology roots, and `write_gate.py` and API.md tell a caller that a custom file "is loaded the same
  way", which is false. Know what a custom file reaches: `_discover_metta_files` (api.py, app.py) lists root
  and custom files for the translator's context and as write targets, but execution loads only the names in
  `_INFERENCE_STACK` plus the query-scoped stacks in `core/pln_runner.py`, every one built from the repo
  root (`api._runtime_kb_paths` re-discovers on each call; `app.py` resolves against a `_METTA_FILES`
  snapshot taken at import, so the Gradio chat needs a restart to see a new file;
  `tests/test_ui_api_parity.py` pins the two stack lists equal). A staged file under `CUSTOM_ONTOLOGY_DIR` is
  therefore never executed until a stack names it.
- **A runtime inventory and schema card.** `pln_chat/ontology/inventory.py` records which predicates have
  data and how much; `scoped_forms.py` tells a caller when a form is well-formed but data-less. This is the
  "what the reasoner knows" view an LLM should see in stage 2.
- **The reasoning trace.** `(explain &self A C)` in `pln_deduction.metta` returns
  `(Chain (n0 ... nk) (signed <sign> (stv s c)))`, one per derivation; `infer` returns the signed TV only;
  the legacy chainer has no cycle guard. Under `PLN_RUNTIME=petta PLN_CHAINER=pln` (config.py refuses `pln`
  on any other runtime) the same seam is answered by patham9's `PLN.Derive` (`pln_chat/core/pln_chainer.py`,
  `pln_chainer.metta`, library in `vendor/pln_fork`). It terminates on cycles because a derivation uses each
  link once, but it enumerates every walk, so a cyclic graph costs far more than its routes. Time is handled
  in three layers: before any derivation, `plan_sources` refuses as `422 refused_form`, in milliseconds, a
  source whose walks cannot fit the largest queue (16,384) or whose modelled time exceeds 80% of
  `PLN_QUERY_TIMEOUT_SECONDS`; a derivation that runs and still does not finish fails as `502 runtime_error`,
  never a partial table; the worker deadline kill (`504 pln_timeout`) still stands. `/query` and `/metta/run`
  also return `ungrounded_predicates` and `validation_warnings` from the validator and the data-less-form
  warnings from `scoped_forms.py` under `warnings`; `/ontology/expand` returns `rejected_entries` with the
  gate's codes and reasons. Every HTTP failure carries a `code`, but only validation and schema refusals are
  422 (a curated-file write is 403, an extraction failure 400, an oversize prompt 413). Together these are the
  trace to show the LLM.
- **Evidence shapes the rules consume.** `(Effect A B Pos|Neg (stv s (evidence-confidence Tier)))` is static
  data in four curated root files under one honesty contract (strength a labelled curated prior, confidence
  an unevaluated tier lookup): `mechanistic_bridges.metta` (19 links), `lifestyle_evidence.metta` (2, the
  smoking lever), `supplement_evidence.metta` (3: Omega3, NMN, Resveratrol) and `linage2_core.metta` (1, loaded
  only by the LinAge2 stack); plus one neutral sign adapter `(Effect Lifespan Mortality Neg (stv 1.0 1.0))` in
  `drugage_calibration.metta`, a definitional flip, not evidence. Everything else is derived at query time
  and never stored as an Effect atom: DrugAge rows through `drugage-effect` (which also lets `infer` step
  through the lift mid-chain), CellAge rows through `cellage-effect`, and the GrimAge (Lu 2019) and LinAge2
  (Fong 2025) `EmpiricalAssociation` records (`HasPredictor`/`HasOutcome`/`EffectValue`), which `link-effect`
  and `explain` read straight into a signed TV through `calibrate-tv` as a chain's terminal link only, since
  the transitive step matches `(Effect ...)` atoms alone. Alongside: review-level `HallmarkInterventionEvidence`
  records, and `human_evidence.metta` provenance records with PMIDs and deliberately no truth value (positive
  / null / no record / not-yet-reported are four distinct states). Read the honesty contracts at the top of
  `mechanistic_bridges.metta`, `lifestyle_evidence.metta`, `linage2_core.metta` §6 and
  `human_evidence.metta`, and `docs/calibration_layer.md`.
- **Runtime limits that shape the design.** hyperon 0.2.10 mis-decodes every key atom stored past about the
  1,024th distinct one (every distinct symbol, variable, number and string counts, so each distinct sentence
  string costs one; upstream issues #1076/#1095, fix PR #1081 still unmerged). Such a space loads and answers
  exact lookups; what aborts the process (an uncatchable Rust panic at `trie.rs:179`, exit -6) is any query
  that reads back a late-stored atom, so a load test proves nothing and the probe is a variable match in a
  subprocess (`tests/test_generic_stack.py`, `tests/test_kb_head_symbol_budget.py`). The full shared stack
  holds about 1,064 key atoms and is never queried; the app survives by query-scoped stacks
  (`core/pln_runner.py`: generic with about 804, patient, human-evidence, DrugAge, CellAge, NHANES and LinAge2
  stacks; `api._INFERENCE_STACK` is the curated list, `api._generic_kb` picks the stack) and by Python
  choosing which rows enter a space (`ontology/drugage_selector.py`, 150 rows at most;
  `ontology/cellage_selector.py`, 100). PeTTa has no such edge (the full stack answers on it every program
  that aborts hyperon) and loads the shipped stacks in 0.04 to 0.32 s, but its behaviour on a space of
  hundreds of papers is unmeasured. Latency, not atoms, was the first wall in the patient forms
  (`docs/kb_quick_wins/REPORT.md`). Hyperon stays the default runtime; petta is opt-in.
- **Where the "10 queries we care about" can come from.** `SUGGESTED_QUESTIONS` in `pln_chat/patient_tab.py`
  (6 patient questions); `docs/*_chat_test_queries.md` (10 each for DrugAge ranking, supplements,
  counterfactuals, risk); `docs/linage2_battery/`; the six demos in `docs/pipeline.md` (Ben's January 2026
  design); Jasmine Smith's test question (35-year-old woman, muscle and bone before menopause, which
  supplements: the KB has no entity, predicate, rule or Effect edge about muscle, bone density or menopause,
  only two free-text `EvidenceOutcomeText` strings that no rule reads); the 34 questions in the 28 Sep to 5 Oct server
  log (7 DrugAge rankings, 17 plain match queries, 2 lookups; the log is not in the repo, ask the owner).
- **Test baseline and commands.** From the repo root with `/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -m pytest`
  (that venv has hyperon 0.2.10, janus-swi 1.5.3 imported as `janus_swi`, and sexpdata). The PeTTa checkout
  is `.petta/PeTTa` (gitignored; `scripts/setup_petta.sh` clones the pinned commits, `config.PETTA_HOME`
  defaults to it); it exists in `/Users/kolya/PycharmProjects/pln_longevity_petta/.petta` and not in a fresh
  work tree, where `-m petta` tests skip until the script has run or `PETTA_HOME` points at that checkout.
  Always `--deselect tests/test_hallmark_targeting.py::test_the_patient_facing_outputs_did_not_move`. The
  known failure set is the 32 ids in `docs/petta_migration/baseline/hyperon_failures_47130aa.txt`; it must
  stay identical. Only the patient reader is guarded suite-wide (`tests/conftest.py` raises unless
  `PLN_LIVE_EXTRACT=1`); the translator and the expander build a real OpenAI client whenever `OPENAI_API_KEY`
  is set, so a test that touches them stubs `openai.OpenAI` or `call_extraction_llm`, as
  `tests/test_ontology_expansion.py` does. Use `-m petta` to select PeTTa tests.

## 2. What leungmanhin/semantic-parser is (read the code; this is the 2026-10-10 state)

- Flat modules, no package, no README: `pipelines.py`, `prompts.py` (46 KB of guidelines), `llm.py`,
  `chaining.py`, `graphs.py`, `vector_index.py`, `checker_functions.py`, `built_in.py`, `utils.py`,
  `batch_parse.py`, `qa.py`, `experiments/pubmed/`.
- `nl2pln(sentence, mode="parsing"|"querying")` sends one sentence to an LLM through **OpenRouter**
  (`OPENROUTER_API_KEY`, default `gpt-5.4-mini`, JSON-schema structured output, 128k max tokens) and returns
  `(type_defs, stmts, queries, extra_exprs, sent_links)`. Statements are PLN expressions
  `(: <proof_name> <body> (STV s c))` in an open, event-reified vocabulary: `IsA`, `HasAttribute`, `Has`,
  UpperCamelCase verb predicates, `Implication`/`Equivalence`/`Similarity`, temporal operators, numeric
  distribution truth values. Instances are `name_1`, stamped with UUIDs by `utils.stamp_parse_with_uuids`.
  **The LLM assigns the STVs.** A `(SentenceLink <proof_name> "<sentence>")` keeps provenance to the sentence.
- Checks are format and graph checks with a correction loop (`format_check_correct`: sexpdata parse, STV
  shape, unused/undefined predicates, connectivity; MeTTa type checking is commented out). A FAISS index of
  predicate embeddings (`text-embedding-3-large` via OpenRouter) drives `generate_equivalences` (cosine above
  0.7 gives `(Equivalence (P ..) (Q ..)) (STV sim 0.9)`); previously parsed sentences are retrieved as RAG
  context; parsing the same sentence several times and taking the maximum common subgraph yields LLM-written
  "bridging rules".
- Reasoning runs on **PeTTaChainer** (`rTreutlein/PeTTaChainer`, PeTTa underneath) in a `multiprocessing`
  child (`chaining.py`): `chaining(kb, query, timeout=30, max_steps=100)` waits 30 s, then kills the child
  and returns `None` (an exception inside yields `[]`; both callers treat the two alike), and
  `PeTTaChainer.query` has its own 10 s default the parser never overrides. Before loading, `Equivalence`
  statements become two `Implication`s with the same STV and instances are UUID-stamped. The chainer answers
  `(: <proof-term> <body> <tv>)`; the proof term is the only derivation trace, and `pln2nl` drops it, keeping
  the derived conclusion for the LLM to verbalise. Nothing in the parser verbalises or shows a reasoning
  trace, so Ben's stage-2 "show an LLM the reasoning traces" component does not exist there. `assisted_qa` is
  a one-step version of his stage 2: when chaining fails it shows the LLM the knowledge and the query and
  asks for the missing rules (`add_missing_knowledge_system_prompt`), with an explicit "return empty rather
  than overfit" instruction.
- `experiments/pubmed/export_pubmed.py` downloads one PubMed baseline file, splits abstracts into sentences
  with spaCy; `batch_parse_pubmed.py` parses every sentence of an abstract with the preceding sentences as
  context and writes one JSON per abstract (`type_defs`, `stmts`, `extra_exprs`).
- Requirements pin `hyperon==0.2.10`, `PeTTaChainer` (git), `pubmed-parser`, `spacy`, `faiss-cpu`,
  `datasets`, `pydantic`, `sexpdata`, numpy 2.4.

The mismatch to design around: the parser emits sentence-level, open-vocabulary statements with LLM-assigned
truth values; this repo's rules consume a closed canonical vocabulary with tier-derived confidence and
derived or labelled strengths, and its honesty contract forbids an LLM-made truth value. Both are PLN, and
patham9's chainer on PeTTa can in principle read both. Deciding how the two layers relate is the heart of
this task, not an implementation detail.

## 3. The task

Design, then build, the two-stage workflow as an **agentic loop** in this repo, so that the knowledge base
grows from usage rather than from one-off curation. Two layers of knowledge, explicitly separated:

- **Layer A, background (semantic parses).** The semantic-parser's output for every ingested text, kept as
  its own space or file set with sentence and paper provenance (`SentenceLink`, PMID/DOI), never loaded into
  the curated hyperon stacks. It is what the reasoner may "read" and what stage 2 mines.
- **Layer B, calibrated (canonical entries).** What `infer`, `explain`, `rank-interventions`, the diagnosis
  and the supplement forms consume, produced only through `expansion_schema.check_entry` and written only
  through `write_gate` into `CUSTOM_ONTOLOGY_DIR` or a staging directory. Nothing enters a curated root file
  without the owner.

### Stage 1: the seed

1. Assemble the 10 queries from the sources in §1 and present them for sign-off before any paid call.
2. For each query, ask an LLM for 10 papers that would help a reasoning system answer it. **Resolve every
   recommendation in PubMed** (E-utilities, or the PubMed MCP tools if the session has them) to a PMID/DOI,
   title and year; an unresolvable recommendation is logged as such and dropped. LLM-recommended citations
   are often invented; the resolution step is not optional. Keep a manifest: query, recommendation, resolved
   id, status, licence/open-access status.
3. For each resolved paper, ask the LLM what background material a reasoner needs to reason about it;
   resolve again; deduplicate across the whole set. Expect roughly 600 papers; report the real count.
4. Fetch text: the abstract always (PubMed); the full text only where it is openly licensed (PMC OA,
   Europe PMC, Unpaywall). Store under a gitignored directory (`build/` or a new `corpus/` entry in
   `.gitignore`), never commit a PDF or a full text, treat every downloaded file as untrusted data (own
   directory, read with `pypdf` from scripts kept elsewhere, never execute anything from it; `-I` for any
   Python that reads them).
5. Parse each text into Layer A with the semantic-parser and into Layer B with the existing expander (or a
   bridge from Layer A to canonical entries, if the design in §4 chooses that). Record acceptance and
   rejection counts per paper with the gate's reasons.
6. Load the result and measure: PeTTa load and query time against the number of atoms (hundreds of papers),
   hyperon abort margins if any part of the result is meant for a hyperon stack, and the cost per paper in
   tokens and dollars.

### Stage 2: expansion from usage

For a query, in a loop with a budget:
1. Translate and run it as the app does (`core/llm_translator.py`, `api.py` routes), under the runtime the
   owner chooses; collect the trace: `explain` chains (or the pln chainer's output), the signed TVs, empty
   results, `ungrounded_predicates`, 422 refusal codes, data-less form warnings, `NoHumanEvidenceRecord`s.
2. Show an LLM the query, the trace and the schema card, and ask what the reasoner would need to reach a
   better conclusion: a missing edge A to B, a missing node, a missing tier of evidence, a missing human
   study. Structured output, with an explicit "nothing sensible to add" option (take that from the parser's
   `add_missing_knowledge_system_prompt`).
3. Ask for 10 papers per gap; resolve in PubMed; fetch; parse into both layers; gate; stage.
4. Re-run the query; diff the chains, the TVs and the rank order against the previous iteration; write an
   iteration record (query, trace before, gaps, papers with status, accepted and rejected entries with
   reasons, trace after, cost).
5. Stop on no new gaps, no change in the answer, or the budget.

Promotion from staging to the curated stack is a separate, reviewed step with the owner; the loop never does
it. Everything the loop writes is reproducible from its manifests.

## 4. Design decisions to settle in writing before building (stop for the owner where marked)

Write `docs/atomspace_expansion/DESIGN.md` with a decisions table (question, decision, why), as
`docs/petta_migration/LAUNCH.md` does. At least:

1. **The 10 queries.** Owner sign-off.
2. **LLM provider.** The repo uses OpenAI directly (`OPENAI_API_KEY`, `openai>=1.40`, `PLN_MODEL`,
   `PLN_EXTRACT_MODEL=gpt-6-luna` for structured extraction); the parser uses OpenRouter. Either port the
   parser's `to_openrouter` to the repo's client or add `OPENROUTER_API_KEY` to `.env`. Owner decision; it is
   their money. State the expected cost of stage 1 before running it, from a measured dry run of one query
   and two papers.
3. **How the parser is brought in.** It has no package structure and pins `hyperon==0.2.10`, PeTTaChainer,
   spaCy and FAISS. Options: a pinned checkout in a gitignored directory with a `scripts/setup_semparser.sh`
   like `setup_petta.sh`, or vendoring the few modules needed (`pipelines`, `prompts`, `checker_functions`,
   `utils`, `built_in`, `llm`) under `vendor/` with the MIT notice. Check that its dependencies do not break
   the project's venv; if they do, give it its own.
4. **Layer A's runtime and reasoner.** Does patham9's PLN (`PLN_CHAINER=pln` on PeTTa) read Layer A directly,
   does PeTTaChainer stay the parser's own reasoner over Layer A, or is Layer A only mined by the LLM for
   Layer B entries? Measure before deciding: load a few hundred parsed abstracts into PeTTa and time
   `explain`-style queries. Parsing real abstracts needs the parser running and an LLM key, both owner
   authorisations; until they are given, measure with hand-made atoms of the parser's output shape, which
   costs nothing, and write the plan for the real measurement. Ask Ben which he means; propose a default.
5. **The bridge from Layer A to Layer B.** How a parsed statement becomes a canonical entry: which
   predicates map to `Effect`/`EvidenceHallmark`/`human_evidence` records, how the study tier is read from the
   paper (not from the parser's STV), how strength is derived or labelled as a prior. The parser's
   LLM-assigned STVs do not pass the gate and must not be smuggled in as confidence. Propose node names,
   tiers and thresholds before writing any entry (a standing rule in this repo).
6. **Equivalences and bridging rules.** The parser generates `Equivalence` atoms from embedding similarity
   at a 0.7 cosine threshold with confidence 0.9. Decide whether any of that enters Layer A, and keep it out
   of Layer B.
7. **Where staged entries live and how they get executed.** `CUSTOM_ONTOLOGY_DIR` is the write target, but
   execution ignores it until a stack names the file (§1). Decide per-loop files versus one staging file, how
   `PLN_MAX_KB_FILE_BYTES` is respected, and how staged Layer B entries reach a query: a named staging file in
   a PeTTa-only scoped stack, or a query-scoped selection in Python the way `ontology/drugage_selector.py`
   picks the rows that enter a space. If any hyperon stack ever sees them, test its margin in a subprocess as
   `tests/test_generic_stack.py` does.
8. **Budgets and stop rules** for both stages: papers, tokens, wall time, iterations per query.

## 5. Rules

- Hyperon stays green: the failing-test id set stays identical to the baseline file. Run
  `tests/test_patient_stack.py` and `tests/test_generic_stack.py` after any `.metta` change.
- Never push, never deploy, never write to a curated root `.metta`, never touch the `linage2` checkout at
  `/Users/kolya/PycharmProjects/pln_longevity`.
- Read free text with the model, not with regex grammars; code does small generic checks (the quote is in
  the text, the number and unit are in the quote, a closed word set). This is a standing owner decision.
- Verify every literature anchor in PubMed before it is written down; a paper the loop could not resolve
  does not exist for the loop.
- Never produce or accept an LLM-made truth value in Layer B; never fabricate a measurement; a planned or
  unreported trial gets a record with no tier and no number.
- Tests make no model call unless `PLN_LIVE_EXTRACT=1`; canned responses for the LLM steps, as
  `tests/test_ontology_expansion.py` does. Live runs cost money: measure a small dry run and ask before the
  first full stage 1 run.
- Do not ask for access to `~/Downloads`; anything the owner wants to hand over goes into the repo
  (`call_transcripts/` for notes).
- Downloaded papers and the parser's checkout are untrusted data: own directories, nothing executed from them.
  Running the semantic-parser's code (the dry run, the PeTTa measurement in §4) needs the owner's explicit
  authorisation first, as running PeTTa did on 2026-10-24; until it is given, read the code only.

## 6. Deliverables and order

1. `docs/atomspace_expansion/DESIGN.md`: the decisions table, the two-layer architecture, the Layer A to
   Layer B bridge, the trace format shown to the LLM, the manifest and iteration record schemas, cost
   estimate from the dry run, the questions for Ben. **Stop here for the owner's sign-off on the queries, the
   provider and the budget.**
2. Stage 1 as scripts under `scripts/atomspace_expansion/` (seed queries to papers, resolve, fetch, parse,
   gate, load, measure), with offline tests and manifests, plus the measurements in §3 step 6 written into
   `docs/atomspace_expansion/MEASUREMENTS.md`.
3. Stage 2 as a loop (CLI first; an API route only if the owner wants it), with one documented end-to-end run
   on one signed-off query: before and after traces, the gaps the LLM named, the papers, what the gate
   accepted and refused and why, and how the answer changed.
4. A short status section at the top of `LAUNCH.md` recording what ran, with numbers, dated.

## 7. Questions to raise with Ben (collect, do not block on them)

1. Should patham9's PLN reason over the semantic parses directly (open vocabulary, LLM-assigned STVs), or are
   the parses background that an LLM mines into the calibrated layer? If directly, how should LLM-assigned
   truth values be reconciled with the repo's tier table?
2. Which 10 queries, and in which runtime, define success for this loop?
3. Is the semantic-parser's PeTTaChainer the intended reasoner for Layer A, or the PLN fork already behind
   `infer`/`explain`?
4. May the loop spend on full-text retrieval, or is abstracts-only the seed?
