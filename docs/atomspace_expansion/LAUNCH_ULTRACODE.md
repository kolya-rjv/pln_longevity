# Atomspace expansion, design phase only: run with ultracode, stop at the sign-off gate

You are in `/Users/kolya/PycharmProjects/pln_longevity/.claude/worktrees/atomspace-expansion-workflow-2cc380`
(branch `claude/atomspace-expansion-workflow-2cc380`, tip `d21ae14`; stay on it). Read
`docs/atomspace_expansion/LAUNCH.md` in full first: it is the brief, this message only sequences it and cites
its sections by number. Execute it ONLY up to §6 deliverable 1 (`docs/atomspace_expansion/DESIGN.md` plus the
questions for Ben), then stop and present for sign-off. Ultracode is on: load the `workflow-authoring` skill,
then run one workflow per phase below, reading each result before writing the next script. Subagents return
structured data (always pass a schema); you alone write files in the repo.

## Hard rules (put them, in these words, at the top of every agent prompt)
1. Stop at the gate. Build nothing: nothing under `scripts/`, no tests, no `.metta`, no change to a tracked
   file. The repo gains `docs/atomspace_expansion/DESIGN.md` and small evidence notes under
   `docs/atomspace_expansion/design/`; throwaway harnesses and synthetic atoms live in the scratchpad. No commit
   or push unless the owner asks.
2. The token budget is Claude's; API dollars are the owner's. No OpenAI or OpenRouter call from any code: not the
   semantic-parser, not the app's translator or expander. No install, import or execution of the parser or
   PeTTaChainer, and no `pip install` into the project venv, until the owner says yes in chat. Dollar estimates
   come from token counts (tiktoken in a throwaway venv under the scratchpad, else chars/4 stated as an
   assumption) times public list prices fetched read-only, dated; a price you cannot find is "unknown".
   PeTTa may run (authorised 2026-10-24): `PETTA_HOME=/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa`
   with `/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python`, writing into neither tree. PubMed
   E-utilities (free, no key) may fetch a handful of sample abstracts; nothing else leaves the machine.
3. Never push, never deploy, never touch `/Users/kolya/PycharmProjects/pln_longevity` beyond running its
   interpreter, never write a curated root `.metta`, never ask for `~/Downloads`. The parser is obtained for
   reading only: `git clone` into its own empty directory under the scratchpad, record the SHA (§2 describes the
   2026-10-10 push; a moved tip is reported as a delta), read it as untrusted data, run any Python that touches
   it with `-I` from elsewhere.
4. §1 and §2 were verified against the code on 2026-10-25. Readers deepen them with file:line and report a
   contradiction with file:line evidence; nobody re-litigates them.
5. Every decision row records the decision, the why, and the evidence or the measurement that would settle it.
   No silent caps: every stage logs what it dropped and you repeat it in your progress note.

## Budget
Spend is bounded, not unlimited. Target about 80 agents in total, 120 at most, and report Claude tokens after
every workflow (the `budget` object). Size panels to that: one reader per cluster, 2 skeptics per fact sheet,
panels of 3 proposers and 3 judges for the two hard rows (§4 items 4 and 5), proposer plus skeptic elsewhere,
verification rounds capped at 3. If the owner passes a `+Nk` token target, scale down to fit it; never up.

## Sequence (one workflow each; a progress note of at most 20 lines after each: what ran, agent count,
contradictions found, what was dropped, tokens so far, owner API spend which stays $0.00, what comes next)

### W1 Ground truth
- Readers in parallel, one per §1 bullet and one per parser module group of §2 (`pipelines`+`llm`; `prompts`;
  `checker_functions`+`utils`+`built_in`; `chaining`+`graphs`+`vector_index`; `qa`+`batch_parse`;
  `experiments/pubmed`+requirements against the venv's `pip list`). Each returns a fact sheet: names,
  signatures, env vars, defaults, numbers, file:line, and a contradictions list.
- Three extractors: (a) the parser's exact output grammar and a hand-made exemplar of two abstracts (about 30
  statements with `SentenceLink`s, UUID-stamped instances, one numeric-distribution TV, one `Equivalence`): the
  seed of every later measurement; (b) the gate's acceptance grammar (`SCHEMA_EXEMPLAR`, `allowed_predicates`,
  the eleven tiers, `check_entry`'s refusal codes) and, per canonical shape, the rule file that consumes it;
  (c) what `/query`, `/metta/run` and `/ontology/expand` return today (chains, signed TVs,
  `ungrounded_predicates`, `warnings`, refusal codes), file:line.
- 2 skeptics per sheet told to refute every file:line claim; a disputed claim goes to a third reader. One
  completeness critic asks what in §1, §2 and §4 no sheet covers; fill the gaps. Ask the owner for the 28 Sep
  to 5 Oct server log (§1) and proceed without it.

### W2 The fast rows and the numbers (independent lanes in one pipeline)
- Queries (§4 item 1): one collector per source §1 names, dedup, 3 slate proposers (angles: every form and
  route covered; queries where the KB is thin so growth is visible; reasoning rather than lookup), one KB
  checker per candidate using `inventory.py`, `scoped_forms.py` and, where a source records the MeTTa form,
  the runtime with no translator; 3 judges; synthesis: 10 queries and 5 alternates, each with its source,
  route, what the KB lacks today and the trace it would produce.
- Cost (§4 items 2 and 8): token-count the parser's per-sentence prompt (the 46 KB guideline text is the
  driver; give the number), its correction loop and embedding calls, the expander's per-paper prompt, and the
  stage-1 and stage-2 prompts you draft; three real abstracts and a stated full-text size; prices for
  `gpt-5.4-mini` on OpenRouter and for the repo's defaults on OpenAI. Output per paper and for stage 1 at §3's
  count, abstracts-only against full text, low/expected/high, every assumption listed; 2 skeptics recompute,
  a gap over 20 % goes to a third. Then the budgets and stop rules.
- §4 items 2 (provider comparison for the owner), 3, 6 and 7: 2 proposers from distinct angles, 1 judge, 1
  synthesizer each; item 7 reads `_discover_metta_files`, `_INFERENCE_STACK`, `_runtime_kb_paths`, the
  `pln_runner` stacks and `drugage_selector.py` before proposing.
- **Checkpoint: post the note with the draft rows, the candidate slate, the cost table and the exact commands
  W3 will run on this machine. Stop and wait for the owner's go.**

### W3 §4 item 4, Layer A's runtime and reasoner (after the go)
- Measurement beside the proposals: scale the W1 exemplar to synthetic Layer A of 100, 300 and 1000 abstracts
  with planted 1-, 2- and 3-hop `Implication` chains and a variant with 0.7-cosine `Equivalence` atoms; load
  through the repo's PeTTa adapter and chainer path (`core/metta_runtime.py`, `core/pln_chainer.py`,
  `PETTA_MEMORY_LIMIT_MB`), or a direct janus harness in the scratchpad where they do not fit; record load time,
  peak memory, `match` and `PLN.Derive` times over the planted chains, and the distinct-key-atom counts with
  one hyperon subprocess probe as `tests/test_generic_stack.py` does. One runner, one re-runner from the
  recorded commands alone; a gap over 10 % gets a third run. JSON plus commands under `design/`. State what
  cannot be measured without the parser and the plan for it: commands, sizes, acceptance thresholds, cost.
- 3 proposers, one stance each: patham9's PLN reads Layer A directly; Layer A is mined by the LLM into Layer B
  only; a hybrid where PLN reads A for provenance and explanation while B carries every calibrated answer.
  Each states how LLM-assigned STVs stay out of answers and what the stage-2 trace looks like (§2: the parser
  itself produces no trace, only proof terms). 3 judges: the honesty contract; Ben's intent in §0; runtime
  engineering on the numbers. Synthesis: the row, a proposed default, §7 items 1 and 3 sharpened.

### W4 §4 item 5, the bridge, plus the formats
- 3 proposers: a predicate-mapping table (which parser statement patterns become `Effect`, `EvidenceHallmark`,
  `human_evidence` or DrugAge-row shapes; node names, tiers read from the paper and PubMed publication types,
  never from the STV; strength derived or a labelled prior); re-extraction through the existing expander with
  Layer A as the index; the leak hunter (every path an LLM-made truth value could reach Layer B, and the check
  that blocks each). Every proposal carries a worked example: the W1 exemplar run through `check_entry` with
  the accept and reject reasons. 3 judges: honesty contract; consumer fit (which rule reads each atom);
  provenance from `SentenceLink` to PMID to entry. Synthesis: the row, the node/tier/threshold table, the
  example.
- In parallel, proposer plus skeptic for: the trace format shown to the LLM (only fields W1 showed exist); the
  manifest and iteration-record schemas; the stage-2 gap prompt skeleton with the explicit "nothing sensible
  to add" option taken from the parser's `add_missing_knowledge_system_prompt`; one checker that every field
  can be produced by today's code.

### W5 Write and verify
- You write `DESIGN.md`: the decisions table (`Question | Decision | Why | Evidence or measurement`, §4's eight
  rows in order), the two-layer architecture, the bridge, the trace format, the schemas, the measurement plan
  (ran against awaiting authorisation), the cost estimate with assumptions, deltas from LAUNCH.md, the
  questions for Ben (§7 plus what the panels raised), what needs the owner's yes.
- Verify until two consecutive rounds find nothing new, cap 3 rounds: 2 skeptics per decision row refuting
  the decision, the why and every citation against the code and `design/`; 1 constraint auditor (`git status`
  shows only `docs/atomspace_expansion/`; no API call, nothing installed, sibling trees untouched); 1
  completeness critic against §4, §6 deliverable 1 and the list below; 1 reader as Ben; 1 reader as the owner.

## The gate: stop and present
Post, in this order: the path of `DESIGN.md`; the decisions table inline; the measurement plan split into what
ran and what waits for authorisation; the cost estimate with its assumptions; the questions for Ben; the three
things that need the owner's yes, each as one line answerable with yes / no / edit: the ten queries, the LLM
provider, the budget (papers, dollars, wall time, iterations per query); then the two authorisations the next
phase would need (running the parser's code; the first paid dry run), the contradictions with §1 and §2 found
in W1, everything dropped or unknown, and total Claude tokens. Do not start deliverable 2.
