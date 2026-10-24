# PeTTa migration: plan and status

Branch `petta-migration` (a git work tree at `../pln_longevity_petta`, derived from linage2 @ 47130aa). linage2 is untouched.
Updated 2026-10-24. The first version of this brief was written before anything ran; this one records what ran.

## Goal

Run the app's MeTTa on PeTTa (MeTTa compiled to SWI-Prolog) instead of hyperon 0.2.10, and put patham9's PLN behind the
`infer`/`explain` seam as a selectable chainer. This answers Ben Goertzel's request to use "the PLN version from patham9
github on the PeTTa Atomspace", complementary to the existing architecture: the LLM translator, the procedures and the
HTTP/Gradio surfaces stay.

## Decisions (the owner chose the first; the rest were delegated on 2026-10-24)

| Question | Decision | Why |
|---|---|---|
| Which PLN | patham9/PLN main (55f1751, 15 Aug 2025), in the form that runs on PeTTa: PeTTa's `lib/lib_pln.metta` @ fea83178, which is the fork's code ported to PeTTa by its author (63 of 81 forms identical, formulas identical). The newer trueagi-io/PLN rewrite (March 2026) is cloned for comparison only. | The fork's own examples cannot run on PeTTa: its `PLN.Init` is a Python hook that needs the hyperon CLI and a Chicken Scheme compiler. |
| Runtime default | hyperon stays the default (`PLN_RUNTIME=hyperon`); petta is opt-in until the owner retires hyperon. | Every commit keeps the known 32 failures and nothing else. |
| Rule set | Restricted: Modus Ponens, Not-elimination, Revision. Full 17 rules only as a comparison lane. | Inversion, induction and abduction create conclusions between unconnected pairs and can outrank deduction. |
| Base rates | None for the restricted set. Uniform p = 0.1, confidence 0.9 only in the comparison lanes. | Modus Ponens never looks up a node probability. |
| Confidence | Keep the legacy 0.9 per extra hop as a shim (`PLN_CHAIN_CONFIDENCE=legacy`); the fork's native c1*c2 is a switch. | Native confidence flips two pinned orderings. |
| DrugAge evidence unit | The paper (PMID), not the row. | Fifty-two ethanol rows come from one paper and would pool to confidence 0.93. |
| Feedback edges | Allowed in curated data under `pln` (it terminates); legacy still hangs. | |
| Large network | Not curated in this run. The control is shown on the shipped network plus a feedback edge and on a labelled synthetic layered network. Ben picks the real one. | The shipped 30-node graph never binds the control. |

## What ran (2026-10-24, on this Mac)

- SWI-Prolog 10.0.2 (Homebrew), `janus-swi` 1.5.3 in the project venv. PeTTa's Python bridge starts in 0.15 s.
- PeTTa's bundled PLN examples reproduce their pinned values exactly (pln_tuffy, pln_roman, plntest, plntestdirect).
- Every stack loads on PeTTa in at most 0.31 s with 40 to 80 MB. A fresh process per query costs about 0.2 s of start-up.
  The study's worry about load cost and a warm-worker redesign does not apply.
- Three source changes make the knowledge base load and run (commit 4fff9d8), with the hyperon suite unchanged:
  `(superpose ())` to `(empty)` on 20 lines, one nested conjunction flattened, and `%Undefined%` signatures for the
  polymorphic functions `link-effect`, `infer`, `explain`, `supp-evidence-cat`, `supp-safety-label`.
- 25 stored `(evidence-confidence T)` calls are reduced to numbers at load time by the PeTTa adapter, not in the sources.
- 145 recorded query shapes, PeTTa against hyperon: 103 identical; the rest differ in result order inside a tuple
  (PeTTa returns facts in assertion order, hyperon in reverse) or in integer rendering (`3` against `3.0`); 3 probe errors
  are edge cases the knowledge base already guards (`log 0`, `car-atom ()`, `(superpose ())`).
- The suite on the final state (measured by the orchestrator, 2026-10-24): hyperon 2201 passed and 32 failed, the failing ids identical to
  `docs/petta_migration/baseline/hyperon_failures_47130aa.txt`; `PLN_RUNTIME=petta` 2209 passed and 31 failed, the same ids minus one
  hyperon-abort probe that passes on petta by nature. Run time on petta is shorter (291 s against 425 s).
- On the heavy stacks PeTTa answers faster than hyperon (patient stack 0.38 s against 1.05 s, LinAge2 0.48 s against 2.10 s); a trivial
  program costs 0.14 s of fixed start-up per fresh process.
- Design of the chainer: `docs/petta_migration/fork_design/` (decision, three designs, two adversarial reviews).
- The `pln` chainer (Phase 6) is built, was reviewed twice (engineering and semantics lenses) and the 18 findings were fixed (2026-10-24, `docs/petta_migration/fork_design/fix/NOTES.md`):
  `PLN_CHAINER=pln` (opt-in, PeTTa only, never a default) answers `infer`/`explain` through patham9's `PLN.Derive`; its answers equal the legacy ones to 2.2e-16 on the 100 patient, 86 LinAge2 and
  101 generic pairs, on the 109 pairs of a feedback edge and on the 50 composite queries (`fork_design/NUMBER_TABLE.md`, six lanes). The mapping is built to make them equal, so what the library contributes is
  the route enumeration, the stamp rule, the bounded queues and the completeness proof, not new numbers. A request pays +0.014 s (no `infer`), +0.06 s (one source) or +0.125 s (every source) over the legacy
  chainer on the same runtime; a program whose graph the library cannot derive in a request's time is refused in milliseconds (422) instead of being killed at the deadline (`fork_design/fix/COST_MODEL.md`);
  a caller fact that cannot be chained is a 422, not a silent "no route". The DrugAge paper-level fold is wired into the ranking (one entry per compound, by the NET of its two directions, every row, no cap).
- The one decision of the fix phase the owner and Ben should look at: a DrugAge row that reports a change of exactly 0.0 (a null) counts on BOTH sides by default. Filed on the protective side only, as
  the first build did, it made Resveratrol and Metformin slightly harmful (-0.0460, -0.0570); counted on both sides they are +0.1363 and +0.1182, dropped +0.0321 and +0.0629 (`fork_design/b2/notes.md`).
- Suites on this state (measured 2026-10-24): hyperon default (`PLN_RUNTIME=hyperon PLN_CHAINER=legacy`) 2552 passed and 32 failed, 22 skipped, the failing ids identical to `docs/petta_migration/baseline/hyperon_failures_47130aa.txt`; `PLN_RUNTIME=petta` (legacy chainer) 2560 passed and 31 failed, the same ids minus the hyperon-only abort probe; the pln-lane tests (`test_pln_chainer`, `test_pln_fixes`, `test_drugage_fold`, `test_pln_vendor`, `test_pln_number_table`, `test_prompt_size`, `test_drugage_discovery`) 375 passed, in the default environment and with `PLN_RUNTIME=petta PLN_CHAINER=pln` exported; the 145 query shapes through the legacy chainer: classes unchanged (103 identical, 8 number_format_only, 4 ordering_only, 25 different, 3 error, 1 no_baseline), 0 of 177 raw strings differ from the kit's own run. A 40-mutant check of the fix tests: 40 killed.

## Phases

1. Spike. Done. Gate passed (commit 4fff9d8 and the kit in `scripts/petta_spike`).
2. Load fixes. Done for the hyperon-neutral ones. The stored-call reduction lives in the adapter.
3. Runtime adapter `pln_chat/core/metta_runtime.py`. Done (93fb137 and the parity commit): `PLN_RUNTIME=hyperon|petta|auto`, a fresh
   forked process per KB load, load verification, deadline kill, result normalisation, `/health.runtime_name`.
   Caller expressions under petta pass three layers: a text refusal, an engine allowlist of builtins (everything PeTTa registers that is
   not on the list is replaced by a thrower in the child), and a scrubbed child (no API keys, a private scratch directory, closed
   descriptors, resource limits). Two adversarial reviews found real bypasses; all blocking and major findings were fixed and tested.
   A third independent review (commit 2b641c9) found no route to the environment, files, network or another request's space, and one
   denial-of-service lever (a tiny allowed program grew a child past 1 GB); that is now bounded by `PETTA_MEMORY_LIMIT_MB` (an in-child
   watchdog; `RLIMIT_AS` where the kernel honours it, never run on a real Linux kernel). **petta is opt-in. Before it faces callers the
   operator does not trust, put a container or cgroup memory cap around the whole service and test `RLIMIT_AS` on Linux.**
3b. Test harness. Done: `tests/runtime_harness.py` and `tests/runtime_compare.py`; 15 modules that drove hyperon directly now run on both
   runtimes; 13 hyperon-only crash probes are skipped under petta, each with a petta-only companion.
4. Dialect and order differences. Done at the test level. Result order inside tuples and last-bit float sums differ between runtimes; where
   a test pinned them, the hyperon assertion is untouched and the petta side compares canonically with a justified allow-list.
   Open finding: grounded `log-math` raises on petta where hyperon returns -inf/NaN (the knowledge base already guards its inputs).
5. Retire the scoped-stack workaround: not done, and not needed while hyperon stays the default.
6. patham9's PLN at the seam (`PLN_CHAINER=pln`), per `fork_design/decision.md` with the reviewers' corrections: strengths stored as
   `q = 0.02 + 0.98 p` and reported as `|s1 - s0| / 0.98^hops`; the DrugAge net of the two directions defined once; shared evidence ids only
   for parallel evidence of one link; the generic stack in the equality gate; selections read by chunked resumption. Built (G0, 6a, 6b, 6c), reviewed twice, fixed (the 18 findings,
   `fork_design/fix/NOTES.md`), number table and control demonstration regenerated. Open: the owner's and Ben's decisions (null convention, confidence discount, which large network).
   Gate: stop for the owner's and Ben's sign-off before `pln` becomes the default.
7. Deployment: out of scope until the owner says so. The server needs SWI-Prolog >= 9.3 and janus.

## Rules

- Hyperon stays green: the failure set stays identical to `docs/petta_migration/baseline/hyperon_failures_47130aa.txt`.
- Never push, never deploy, never touch linage2 or the main checkout.
- The owner authorised running PeTTa and the pinned checkouts in `.petta/` on 2026-10-24. Nothing else downloaded is executed.

## Questions for Ben

1. Is confidence ranking the control you mean, or should the application supply its own priority (effect times confidence)?
2. Is the per-hop 0.9 discount a PLN concept, or is Modus Ponens' c1*c2 (and Deduction's minimum) the intended behaviour?
   Is the fixed 0.02 in Modus Ponens the intended floor for P(B | not A)?
3. Which large network should the control be shown on?
4. Induction and Abduction call sites pass the shared node's probability in the third slot while the formula expects it in
   the second. Intentional?
5. A DrugAge row with a change of exactly 0.0 is a reported null. Does it count as evidence of no effect in both directions (the default), is it dropped, or does it count on the protective side only (the
   sign convention's accident, which made Resveratrol and Metformin harmful)? And is pooling by `Truth_Revision` (the weight of evidence, not its agreement; papers as independent units) what you mean for DrugAge?
