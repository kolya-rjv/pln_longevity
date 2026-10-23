# PeTTa migration: launch brief

Written 2026-10-23 from a read-only feasibility study (seven agents, two of them adversarial reviewers).
Launch with:

    ultracode: run the PeTTa migration in docs/petta_migration/LAUNCH.md

## Goal

Move the app's MeTTa runtime from hyperon-experimental 0.2.10 to PeTTa (MeTTa compiled to SWI-Prolog), and put
trueagi-io/PLN (lib_pln) behind the `infer`/`explain` seam as a selectable chainer. This answers Ben Goertzel's
request to use "the PLN version from patham9 github on the PeTTa Atomspace". It is complementary to the
existing architecture: the LLM translator, the procedures and the HTTP/Gradio surfaces stay.

## Inputs

- `docs/petta_migration/feasibility/01_verdict.md` is the synthesis: compatibility table, phases, risks.
- `02_review_engineering.md` and `03_review_upstream.md` are the adversarial reviews. **Where they conflict with
  the verdict, the reviews win**; their corrections are folded into the phase plan below.
- `10_metta_constructs.md`, `11_python_coupling.md`, `12_petta_facts.md`, `13_pln_lib_facts.md` are the
  file:line maps of the KB dialect, the Python coupling, PeTTa and lib_pln.
- Background memory notes (read when relevant), under
  `/Users/kolya/.claude/projects/-Users-kolya-PycharmProjects-pln-longevity/memory/`:
  `linage2-shared-stack-pair-abort.md` (why the scoped stacks exist), `deployed-server-root-etl-abort.md`,
  `server-log-replay-method.md`, `stakeholder-ben-pln-inference.md`.

## Authorized by launching this brief

- Create branch `petta-migration` from linage2 @ 47130aa and commit to it, one or more commits per phase.
- `brew install swi-prolog` (Homebrew ships 10.0.2; PeTTa needs >= 9.3).
- `.venv/bin/pip install janus-swi sexpdata`.
- A setup script, `scripts/setup_petta.sh`, that clones PeTTa @ `fea83178af3722666d2948d074591d10a3134f56`
  and trueagi-io/PLN @ `4405956947c4b53c7ff01bd565aa3b114bc970a1` into a gitignored `.petta/` directory,
  with PLN at `.petta/PeTTa/repos/PLN` as its README expects. Run that script.

## Not authorized (stop and ask)

- Touching linage2, main or any other branch; merging; pushing.
- Deploying to the server or changing it. The deployed UI is used by stakeholders.
- Making lib_pln the default chainer. That needs the owner's and Ben's sign-off on the number table.
- Any install beyond the list above, including Docker.

## Hard rules

- **Dual runtime until the owner retires hyperon.** Every commit keeps the full suite green on hyperon.
  `PLN_RUNTIME=hyperon|petta` selects the runtime; hyperon stays the default on this branch.
- **No pinned number changes under `PLN_CHAINER=legacy`.** lib_pln numbers get their own pins under
  `PLN_CHAINER=pln`.
- **PeTTa state is process-global.** One KB load per process, always. See Phase 3.
- Never execute upstream code except the pinned PeTTa runtime, lib_pln and their own test scripts.

## Phases and gates

Proceed automatically when every gate passes. Stop and report when a gate fails or a decision is listed.

### 1. Spike (sequential, one agent)

Install as authorized. Then:

1. Run lib_pln's own examples and `test.sh` through `run.sh`, and through Python `PeTTa().process_metta_string`.
2. Load each stack (generic, patient, human evidence, LinAge2, one DrugAge slice) as one string and as a temp
   file through `load_metta_file_cached`. Record the first error per stack.
3. In a scratch copy only, apply the Phase 2 load fixes, then run the five smoke queries in
   `pln_deduction.metta` section 5 and compare with the expected answers written there.
4. Run every taught translator shape (the 46 entries of `pln_chat/prompts/few_shot_examples.json` and the form
   families in `system_prompt.txt`) through a throwaway adapter. Record each failure with PeTTa's error text.
5. Measure per stack: `consult(main.pl)` time, cold load, cached replay, RSS, and per-query latency next to
   hyperon on the same probes. Also measure fork after a child-only import, the janus GIL behaviour, how
   floats, ints and booleans print, and answer order at the seven first-answer sites (verdict section 2, item 7).

**Gate.** Janus-backed PeTTa runs on this Mac. The smoke answers appear after only the Phase 2 fixes. lib_pln
main's own examples run on the pinned PeTTa, or each failure has a fix. No routed probe is more than five times
slower than on hyperon with the load included; if one is, stop for the owner's decision.

### 2. Load-unblock sweep (sequential, one agent)

PeTTa aborts a whole load at the first form it cannot compile, and every test module loads files from several
clusters. So these fixes land before any parallel work:

- Replace the 20 code lines of `(superpose ())` in 8 root files with `(empty)`. Both runtimes have `(empty)`.
- Pick one strategy for the 25 facts that store an unevaluated `(stv s (evidence-confidence T))` call
  (mechanistic_bridges 19, supplement_evidence 3, lifestyle_evidence 2, linage2_core 1) and for their writer
  in `pln_chat/ontology/expansion_schema.py`. PeTTa does not evaluate values bound from a space. Either reduce
  the call in the three lifter rules or store numbers generated from the tier table, with a test that keeps the
  tier table the source of truth.
- Flatten the nested conjunction at `cellage_calibration.metta:160-163`.

**Gate.** The full suite is green on hyperon, and every stack loads on PeTTa without an exception.

### 3. Runtime adapter and test harness (sequential, one or two agents)

- `pln_chat/core/metta_runtime.py`: a protocol both runtimes satisfy, `MettaRuntimeError`, and
  `PLN_RUNTIME`. `_hyperon_run` becomes `_runtime_run`, the single execution site.
- **Fresh process per KB load under PeTTa.** `run_query_parts` (`pln_runner.py:709-723`) loads two stacks back to
  back in one process; offload each part to its own child. Inline mode (`PLN_WORKER_POOL_SIZE=0`) must fork
  internally or be refused under PeTTa. A second load in one process duplicates every clause and multiplies
  recursive answers.
- **Load verification.** A form that fails to compile raises through janus, but the forms before it stay
  loaded. Treat any exception during a load as a failed load and discard the process. Check that the expected
  function heads exist. A query that fails without raising comes back as `[]` and is an empty answer.
- One call per `!` expression, to keep per-expression result groups.
- Normalise result strings. PeTTa prints `true`/`false` in lowercase, keeps ints as ints and writes `1.0e-5`
  style floats. Every parser accepts both renderings.
- Cached loads: write each stack's concatenation to a temp file keyed by a content hash, load it with
  `load_metta_file_cached`, then add per-query atoms with `process_metta_string`.
- Validator under PeTTa: refuse `add-atom`, `remove-atom` and `(= …)` from `/metta/run`, plus the `assert` and
  `test` heads, which can `halt(1)` the process. Re-check the three builtin lists: `new-space` is absent,
  `unify` exists only in lib_he, and `fst`/`snd` exist only in lib_roman.
- Do not build a long-lived warm worker that adds and removes per-query atoms. PeTTa's `remove-atom` retracts
  every identical copy, so it can delete KB facts.
- Test harness: one runtime fixture, per-call process isolation under PeTTa, and runtime markers, so that
  `PLN_RUNTIME=petta pytest` runs end to end.

**Gate.** The suite is green on hyperon. Under PeTTa it runs end to end without hangs and produces a written
failure list. `/health` keys and `routed` values are unchanged.

### 4. Dialect fixes (parallel, up to nine agents in worktrees)

Use the nine clusters in verdict section 3, Phase 3. Each cluster owns semantic fixes and its own pins, not
loadability, which Phase 2 settled. Each is verified by its test modules on both runtimes.

**Gate.** All 38 test modules are green on PeTTa, except the hyperon-only abort probes, which are quarantined
behind a marker. UI/API parity tests are green, and the spike's breakage table is empty.

### 5. Retire the scoped-stack workaround (sequential, judgment)

As verdict section 3, Phase 4. Before deleting the abort-probe modules, move the routing-agreement tests out of
them (`tests/test_generic_stack.py:126-136, 165-172`, `tests/test_patient_stack.py:93-106`). Keep the DrugAge
and CellAge slices, which exist for semantic reasons.

**Gate.** The routing-agreement tests give byte-identical answers from one stack, within the Phase 1 latency
threshold. If not, keep the stacks, but note that under PeTTa a mixed request then costs two loads.

### 6. lib_pln at the infer/explain seam (design sequential, build parallel)

- Vendor lib_pln @ 4405956 with the per-step `println! SELECTED` removed. Add `PLN_CHAINER=legacy|pln`,
  with legacy as the default.
- Map links with an up/down node split, `A-up` and `A-down`, so the Pos/Neg sign survives. Simpler mappings fail:
  `Implication A (Not B)` never chains.
- Use one evidence id per source record (curated Effect atom, hazard-ratio record, DrugAge row), so that revision
  merges a compound's rows instead of keeping one.
- Give every node a base rate strictly above zero, uniform to start. Without one, a node takes part in no
  syllogistic inference. A zero base rate raises a division error.
- **Build two rule-set modes and compare them.** Full lib_pln runs inversion, induction and abduction on every
  task. Those beliefs can outrank deduction in `PLN.Query` and create conclusions between unconnected pairs, for
  example DasatinibPlusQuercetin to AllCauseMortality, which `test_no_false_chain` pins as empty. Upstream's own
  `examples/DeductionRevision.metta` returns a non-deduction answer. Deduction-plus-revision mode is the
  restricted alternative.
- Derive once per request or per child, then look terms up. `PLN.Query` reruns the whole closure on every call,
  and the procedures call `infer` dozens of times per request.
- Write adapters that return today's `(signed Pos|Neg (stv s c))` and `(Chain …)`. Rebuild the chain from stamps
  only in deduction-only mode: in full mode a stamp cannot tell deduction from induction.
- Write a number table: every pinned (s, c) before and after, in both modes. Cover orderings, the no-false-chain
  pair, a feedback edge that must terminate, a compound with mixed-sign rows, and a multi-row DrugAge compound.
  lib_pln deduction is not associative, so record the step and queue settings next to every number.

**Gate.** Stop for the owner's and Ben's sign-off before making `pln` the default.

## Out of scope for this run

- Deployment, which needs server access and the owner's go.
- The large network Ben's point is about: DrugAge rows, hallmark targeting and GenAge/CellAge links as PLN
  sentences, with their base rates. That is curation and waits for Ben's choice of network.

## Report at every stop

Report what passed, and what failed with the exact error text. Include a table of measurements next to the
hyperon baseline, the commits made, and the decisions needed from the owner.

## Questions to send Ben now

1. Which PLN: trueagi-io/PLN main (March 2026), the older lib_pln bundled with PeTTa, or patham9's fork?
2. Should inversion, induction and abduction run on a causal effect graph, given that they create conclusions
   between unconnected pairs and can outrank deduction? Or should the chainer be deduction plus revision?
3. Is a uniform base rate acceptable to start? Every node needs one, and non-uniform priors make the inversion
   rule's copied strength wrong.
4. Which large network should the control be shown on, and is lib_pln's confidence-ranked queue the control he
   means?

## Effort

| Phase | Agent-phases |
|---|---|
| 1 Spike | 1 |
| 2 Load-unblock sweep | 1 |
| 3 Adapter and harness | 3–4 |
| 4 Dialect clusters | 9, in parallel |
| 5 Retire scoped stacks | 2 |
| 6 lib_pln seam | 4–5 |

Deployment adds one or two agent-phases. The large network is open-ended and bound by curation.
