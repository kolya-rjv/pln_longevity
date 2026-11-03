# W1 ground truth: what the readers found against LAUNCH.md §1 and §2

All repo citations are to tip `d21ae14` (work tree `atomspace-expansion-workflow-2cc380`, branch
`claude/atomspace-expansion-workflow-2cc380`), not to the commit this note sits on. Parser clone
`leungmanhin/semantic-parser` main `d00331d` (2026-04-05; GitHub's 2026-10-10 `pushed_at` is the dependabot branch
`c53aee6`, datasets 5.0.1, so main has not moved since the brief's state). Run 2026-10-25/26: 16 fact sheets, each
refuted by two skeptics, disputes adjudicated, one completeness critic; 36 agents, 698 facts, 594 confirmed by both
skeptics, 232 disputed items adjudicated (121 stand, 104 corrected, 4 wrong, 3 unverifiable). The sheets are scratch
data. Nothing called an LLM.

## Contradictions and deltas that change the design

| # | Brief | What the code says | Evidence |
|---|---|---|---|
| 1 | §1 b2: a staged file under `CUSTOM_ONTOLOGY_DIR` is never executed until a stack names it | A custom file named like a stack file runs **in place of** the curated one: `_discover_metta_files` lets the custom root overwrite the curated name and `_runtime_kb_paths` resolves `_INFERENCE_STACK` through it. `_INFERENCE_STACK` names `cellage_metadata.metta`, absent on disk, so a custom file of that name runs at once. A `target_file` without `.metta` misses the dict and creates such a shadow. | `pln_chat/api.py:175-178`, `:195`, `:274-275`, `:436-442` |
| 2 | §1 b2: `excluded_from_runtime` lists oversized stack files | It lists every discovered file outside the stack, every custom file included | `pln_chat/api.py:1918-1919` |
| 3 | §1 b4: the 80% test is per source against `PLN_QUERY_TIMEOUT_SECONDS` | The queue test is per source; the time test sums the modelled seconds of every planned source, against the request's remaining deadline under PeTTa; a separate 85% clock check fails an overrun as 502 | `pln_chat/core/pln_chainer.py:1419-1433` |
| 4 | §1 b4: every HTTP failure carries a `code` | A pydantic request violation returns FastAPI's default 422 with no `code` (no `RequestValidationError` handler). 422 also covers cost refusals (`refused_form` for a too-dense graph) and the write gate's size refusals | `pln_chat/api.py:986-1016`, `:944-947`; `pln_chat/ontology/write_gate.py:62` |
| 5 | §4 item 2: `PLN_EXTRACT_MODEL=gpt-6-luna` for structured extraction | It is the My Patient reader's model only; the expander uses the caller's model, `DEFAULT_MODEL` (`PLN_MODEL`, else `gpt-5.4-mini`) | `pln_chat/core/patient_extract.py:334-336`; `pln_chat/config.py:23-28` |
| 6 | §1 b1: `/ontology/expand` reports every refusal | Entries that are not dicts, have an empty name or MeTTa, or contain no `(` are skipped before the gate and counted nowhere (the route docstring says they are never dropped quietly) | `pln_chat/ontology/expander.py:717-730`; `pln_chat/api.py:3875` |
| 7 | §2 b1: `prompts.py` holds 46 KB of guidelines | 46,373 bytes is the file; the guidelines (`base_instructions`) are 33,120 chars; each LLM call sends 33.7k to 36.8k chars (8.2k to 9.1k tokens, o200k_base) of system prompt | `parser:prompts.py:72-189` |
| 8 | §2 b2: a `SentenceLink` keeps provenance | No shipped script persists `sent_links` (`batch_parse.py`, `batch_parse_pubmed.py`, `batch_parse_sew.py` unpack and drop them; `qa.py` discards them). They name the unstamped proof name while the loaded KB carries stamped names | `parser:batch_parse.py:62-71`; `parser:experiments/pubmed/batch_parse_pubmed.py:86-97`; `parser:qa.py:15`; `parser:pipelines.py:176`; `parser:chaining.py:84` |
| 9 | §2 b2: UUID-stamped instances | Stamping is one mapping per call, so equal instance names from different sentences or papers merge when a KB is stamped at once; concept names ending in `_<digits>` (`il_6`, `igf_1`) are stamped too | `parser:utils.py:24-34`; `parser:chaining.py:84` |
| 10 | §2 b2: numeric-distribution truth values | The prompt asks for them, but `stmt_format_check` accepts only `(STV s c)` in the TV slot, so such a statement never leaves the correction loop | `parser:checker_functions.py:33-37`; `parser:prompts.py:168-176` |
| 11 | §2 b3, §4 item 6: the parser generates Equivalences and bridging rules | Both exist; neither is wired: `generate_equivalences` has no caller, no caller passes `runs>1` | `parser:pipelines.py:113`, `:194-201` |
| 12 | §2 b4: `assisted_qa` is a one-step stage 2 | Accurate in behaviour, but it has had no caller since `4e01041` (2026-03-11), and its rules skip the format check | `parser:pipelines.py:269-308`; `parser:qa.py:66` |
| 13 | §2 b4: `chaining(kb, query, timeout=30, max_steps=100)` | `chaining(kb, query, handler=None, timeout=30, max_steps=100)`; `pln2nl` drops the truth value as well as the proof term | `parser:chaining.py:127`; `parser:pipelines.py:258` |
| 14 | §2 b6: requirements | Also pins `networkx==3.6.1`. The `PeTTaChainer` git URL (`rTreutlein/PeTTaChainer`) returns 404; live copies at `leungmanhin/PeTTaChainer` (`b5182ab`, 2026-10-20) and `MesTTo/PeTTaChainer` (`36b3fe6`, 2026-07-17), neither a GitHub fork of the other. The clone also holds `fsenf/` and `experiments/simple_eng_wiki/` | `parser:requirements.txt:4`; GitHub API, 2026-10-25 |
| 15 | §1 b6: fix PR #1081 still unmerged | The repo says only "not released"; merge state is not recorded and was not checked online | `pln_chat/core/pln_runner.py:164-165` |
| 16 | §1 b7: the server log holds 34 questions (7 + 17 + 2) | The breakdown sums to 26; the log is in neither tree | `docs/atomspace_expansion/LAUNCH.md:121-122` |

## Facts the critic added that the design leans on

- The default `/query` translator system prompt is 309,635 chars (77,201 tokens, o200k_base), rebuilt per request; it
  never shows an `explain` form, and the 46 few-shots hold no `explain` and one `infer`
  (`pln_chat/core/context_builder.py:54-80`, `pln_chat/prompts/few_shot_examples.json`). Stage 2 cannot rely on the
  translator to produce a chain.
- patham9's restricted library keeps only Revision, Modus Ponens over `(Implication $A $B)` and Not-elimination
  (`vendor/pln_fork/lib_pln_restricted.metta:104-115`); Equivalence and Similarity rules were dropped
  (`vendor/pln_fork/PROVENANCE.md`). The parser's `(Implication (Premises ..) (Conclusions ..))` matches neither the
  library's `(Sentence ((Implication A B) (stv f c)) (id))` shape nor any `pln_chainer` link source.
- The expander's gate vocabulary comes from every root and custom file of at most 60,000 bytes, the write gate's
  from the runtime stack: two different predicate sets (`pln_chat/ontology/expander.py`; `pln_chat/api.py:271-285`).

## Not covered in W1

PMC open-access licence status of the samples (needs a network call beyond PubMed); spaCy sentence counts (spaCy is
not installed; a naive split gives 9, 6 and 7 for the three samples); the 28 Sep to 5 Oct server log.

## Side effect to record

Running the repo's own code offline (the KB-check recipe, once in W1 and by the W2 checkers) wrote the gitignored
debug logs `pln_chat/logs/last_query.metta` and two `pln_chat/logs/session_*.jsonl` in the workflow work tree. No
tracked file changed there (`git status --short` shows only `?? docs/atomspace_expansion/`). The W3 harness redirects
`LOGS_DIR` into the scratchpad.
