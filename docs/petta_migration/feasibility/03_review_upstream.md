# Adversarial review of the feasibility verdict: upstream reality

Lens: the cloned PeTTa (fea83178) and trueagi-io/PLN (4405956) sources under
`scratchpad/feasibility/upstream/`. Each claim the verdict makes about them was checked against
the source. Nothing was executed from the clones. My own read-only scan scripts are in
`review-upstream/scripts/` (a small s-expression reader plus four scans over the 32 root .metta
files). Arithmetic checks were run with plain Python over lib_pln's formulas, transcribed from
`PLN/lib_pln.metta:84-153`.

Path prefixes: `PeTTa/` and `PLN/` mean
`/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/feasibility/upstream/{PeTTa,PLN}/`.
Repo paths are relative to `/Users/kolya/PycharmProjects/pln_longevity` (linage2 @ 47130aa).

## 1. Claims checked and confirmed (no objection)

- `(superpose ())` does not compile. `build_superpose_branches([], _, [])` succeeds, then
  `disj_list([], _)` has no clause (PeTTa/src/translator.pl:637-640, 1533-1540). No rewrite rule
  catches the form first (translator.pl:410-447).
- Nested `(, A (, B C))`: the inner list becomes a lookup of a 3-arity `,` fact (PeTTa/src/spaces.pl:218-223).
- `match` on a runtime space symbol works (spaces.pl:232-240). Match patterns are not translated for
  untyped spaces (translator.pl:880-897; type_lang.pl:303-312). Templates are translated, and their
  goals run after the match.
- Values bound from the space are not re-evaluated (PeTTa/examples/eval.metta:5-9;
  he_atomspace.metta:3-6). So the stored `(stv s (evidence-confidence T))` facts reach
  `combine-tv` unreduced. The `*` call then runs `is/2` on a list and raises.
- Bare arrow declarations only cache a type. They never call `register_fun`
  (decl_store.pl:107-131, 541-545). `fun/1` is asserted only from definitions, at
  filereader.pl:110-112 and spaces.pl:56.
- `car-atom ()` throws (metta.pl:186). `unique-atom` is `list_to_set/2`, which keeps the left-most copy
  (metta.pl:148-149). `log-math` is `log(X)/log(Base)` (metta.pl:104). `pow-math` is `**`
  (metta.pl:101). `and`/`or`/`not` are eager and Bool-only (metta.pl:129-136). `True`/`False` are
  read as `true`/`false` (parser.pl:82-84).
- `if` reaches its then-branch only on `true`. Comparisons that feed an `if` compile to native
  `<`/`>`/`==`, which raise on non-numbers (translator.pl:1358-1373).
- 5-argument `foldl-atom` is a special form (translator.pl:817-825).
- Python gets one flat list per `process_metta_string` call (filereader.pl:100; python/helper.pl:6-10).
  The engine is global per process, with an 8 GB stack (python/petta/__init__.py:5-60).
  `test`/`assert` call `halt(1)` (metta.pl:256-265).
- `janus` is loaded unconditionally (metta.pl:27). CI is ubuntu-only and runs both the Python tests
  and test.sh (.github/workflows/ci.yml:117-145).
- lib_pln: the deduction confidence is `s_AB*s_BC*c_AB*c_BC` (PLN/lib_pln.metta:97). The STV
  catch-all is `(empty)` (:67-68). An inconsistent case returns `(stv 1 0)` (:42-54, 99). Revision
  uses the w-sum and returns nothing when c = 1 (:71-72, 136-143). Inversion gives
  c = c_B·c_AB·0.6 and copies the strength (:150-153, 290-292). The defaults are 100 steps and
  queues of 10/100 (:326-332). `LimitSize` keeps N-1 (:370-374). `println!` runs on every step (:391).
  `PLN.Query` takes the best-confidence belief whose term is `==` the query term (:412-421).
  `git-import!` with a SHA is at PeTTa/lib/lib_import.pl:98-108.

## 2. Objections

### O1 (major): lib_pln changes which conclusions exist, and which rule's number is returned, not only the numbers

The verdict says that on the curated graph "PLN changes the numbers, not which conclusions exist",
that chain confidences "fall 2-11x", and that the derivation record is "rebuildable from stamps".
lib_pln's rule set contradicts all three:

- Implication inversion is a single-premise rule. It fires on every selected Implication task
  (PLN/lib_pln.metta:290-292, called through `(|- $x)` at :390). Induction (:245-251) and
  abduction (:253-259) fire on any two Implications that share a source or a target.
  The only guard is `SyllogisticRuleGuard`, which is shared with deduction (:232-233).
  You cannot turn induction or abduction off without editing the library.
- `PLN.Query` returns the highest-confidence belief for the term, whichever rule produced it
  (:412-421). A Sentence stores only the term, the stv and the stamp (:362, :380-390), so the
  rule that produced it cannot be recovered.
- **Upstream proof.** `PLN/examples/DeductionRevision.metta:5-18` is named for deduction plus
  revision. It pins `(stv 0.125 0.30429942952553224) (1 2 3 4)`. That value is the revision of two
  INDUCTION beliefs built on inverted links: (B→A inverted, B→D) and (C→A inverted, C→D) give
  s = 0.125 and c = 0.17945 each, and revising the two gives c = 0.3042994295255322. The deductive
  route gives s = 0.5 with c = 0.10125 per path, or c = 0.184 revised. So the answer has 4x the
  wrong strength for "deduction", and it carries the same stamp (1 2 3 4) as the two deductive
  paths. A Chain rebuilt from that stamp would find both A→B→D and A→C→D and label an induction
  result as deduction.
- **This KB's DrugAge mapping** (the place the verdict wants to demonstrate PLN). Take X-up→Lifespan-up
  (0.2, 0.9) and the bridge Lifespan-up→Mortality-down (1.0, 1.0), with uniform p = 0.1.
  Deduction gives (0.2, 0.180). Induction over the inverted row (Lifespan-up→X-up, c = 0.486) and
  the bridge gives (0.2, 0.327), with the same stamp. PLN.Query therefore returns the induction
  value, and its confidence RISES rather than falls.
- **Curated chain.** For a final link with s = 0.5 (the calibrate-tv bucket of the Lu2019
  AgeAccelGrim HR records), abduction through the inverted last link reaches c = 0.06475 against
  deduction's c = 0.06410, with the same stamp. Deduction wins only when the last link has s > ~0.54.
- **New conclusions for unconnected pairs.** CHD-down → AgeAccelGrim-down exists by inverting the
  Lu2019 AgeAccelGrim→CHD record (grim_age_lu2019_evidence.metta:28-38). Induction from the common
  cause AgeAccelGrim (→CHD, →AllCauseMortality) gives CHD → AllCauseMortality. Either route makes
  D+Q → AllCauseMortality derivable. tests/test_pln_deduction.py:150-155 (`test_no_false_chain`)
  pins that pair as empty. Compounds that share a target (D+Q, Fisetin, Resveratrol →
  CellularSenescence; pln mechanistic_bridges.metta:71-106, supplement_evidence.metta:106) also get
  abduced links between each other.
- **The sign split leaks once the signed graph is unbalanced.** Take a compound with one
  life-extending row and one life-shortening row: X-up→Lifespan-up plus X-up→Lifespan-down.
  Induction (:245-251) derives Lifespan-up→Lifespan-down. Deduction through that link then gives
  every other compound a spurious opposite-sign answer with a small confidence. "The sign is never
  lost" holds only without induction and abduction.

Consequence: the up/down split is necessary but not sufficient. Phase 5 needs one of two
decisions. Either keep a vendored lib_pln restricted to Deduction and Revision, or tag each
Sentence with its rule. Both are forks, which contradicts the "no change to lib_pln" pro in the
PLN facts. The alternative is to accept inversion, induction and abduction answers in a causal KB.
That gives up the "no false chains" contract and the ability to rebuild provenance from stamps.
This belongs in the questions for Ben. The Phase 5 gate "(Chain …) rebuilt from stamps", the number
table, and the test_no_false_chain pin all depend on that decision.

### O2 (major): the process-global PeTTa state breaks `run_query_parts` and inline mode

`run_query_parts` (pln_runner.py:709-723) runs several programs, each with its own KB, "one after
another in THIS process". Hyperon gave each one a fresh `MeTTa()`. In PeTTa the space is a set of
global dynamic predicates (spaces.pl:2-10). There is one engine per process and no reset
(python/petta/__init__.py:29-60). `process_metta_string` recompiles every `(=…)` on every call:
mode `compile_functions` (filereader.pl:72-74). Only file loads dedupe, through
`claim_source_compile` (filereader.pl:44-47). So a second KB load in the same child adds
duplicate facts and duplicate clauses. Results are duplicated, recursive `infer` multiplies them,
and the scoped LinAge2 stack sees the other stack's atoms.

The same thing happens with `PLN_WORKER_POOL_SIZE=0` (inline mode), which reloads the KB on every
request in a long-lived process. janus also gives each non-main Python thread its own engine, and
PeTTa keeps its function metadata in thread-local `nb_` globals (translator.pl:83-84).

The verdict's row #29 says "one KB load per process, which the fork-per-query executor already
provides". Phase 4's fallback says "keep the stacks (they are harmless)". Neither holds for
run_query_parts or inline mode. Phase 2 must run one child per part and disable inline mode.
That means two loads and two deadlines for every split LinAge2 program.

### O3 (major): an upstream mitigation for per-process load cost was dismissed with a wrong reason

The verdict dismisses `load_metta_file_cached` because it "applies to files, not strings"
(§2.3, risk #8). Upstream built and tested it for exactly the fork-per-query model.
python/tests/test_load_cache.py:6-7 says "Each load runs in a fresh process, as a snapshot only
replays onto the state it was taken from". The tests then show a warm replay, invalidation when
the inputs change, and replay of a 2,500-clause predicate (:38-100).

The snapshot covers dynamic predicates, `nb_` globals and flags, and leaves out the pid
(PeTTa/src/load_cache.pl:1-30, 76-104). The KB stacks are fixed file lists, so each stack can be
written to one file and loaded cached in the child. Per-query `extra_atoms` can then go through
`process_metta_string`.

Phase 1 should measure cold load, cached replay, and `consult(main.pl)` separately.
`consult(main.pl)` is about 13.7k lines of Prolog: src/*.pl is 3,350 lines and src/typecheck is
10,311. Risk #8 ("medium-high") and the warm-worker contingency should be re-rated on those
numbers.

### O4 (minor): lib_pln deduction is not associative, so the pinned numbers depend on control parameters

With p = 0.1, the D+Q→CHD chain gives c = 0.01517 associated left and c = 0.00955 associated
right, both with the stamp {1,2,3,4}. Revision cannot merge them because their stamps overlap.
Which association orders exist depends on MaxSteps and the queue sizes. The Phase 5 number table
is therefore a function of those settings and must be regenerated after any Phase 5 or Phase 7
tuning.

### O5 (minor): the failure mode of a form that does not compile is settled by the source

When `translate_clause` fails, as it does on `(superpose ())`, the catch-all `process_form` clause
throws `syntax_error("failed to process form: …")` (filereader.pl:185). By then `add_sexp` has
already stored the `(=…)` atom (filereader.pl:167), and every earlier form stays asserted, with no
rollback. From Python this is an exception, not a silent partial load.

Only a FAILED query (as opposed to one that throws) comes back as `[]`: `_run_helper` returns
`[]` for a None result and falls back to `.get("Results", [])` (python/petta/__init__.py:71-73).

## 3. Corrections (short)

- filereader.pl:163 → 166-167, the line where `add_sexp` runs before compilation.
  parser.pl:32 writes numbers; atoms are written at parser.pl:38.
- `sin-math` is present (metta.pl:109, 459). `fst`/`snd` exist only in lib/lib_roman.metta:150-153.
- PeTTa's test.sh also runs examples/pln_roman.metta. PLN's own RavenInduction.metta imports PeTTa's
  bundled `(library lib_pln)` and `lib_patrick`. A passing PLN test.sh is therefore not a clean
  signal for lib_pln main.
- "No macOS mention upstream": PeTTa.wiki/Smart-dispatch.md:28 times `(fib 100000)` on an Apple M4,
  so the CLI has run on macOS. The janus path on macOS is still unverified. The CI image is rebuilt
  only when .github/ci/Dockerfile changes (ci.yml:3-9, 147-169), so CI's SWI version is frozen and
  current SWI 10.x is not tested continuously.
- Stored-call facts in the runtime stacks number 25, not 46: mechanistic_bridges 19,
  supplement_evidence 3, lifestyle_evidence 2, linage2_core 1. The other 21 are in
  docs/kb_quick_wins/probes/mb_all.metta. Of the 25 `(superpose ())` grep lines, 5 are comments.
- PLN.Derive takes no query term (lib_pln.metta:377-406). One Derive per request, followed by term
  lookups, avoids a closure recomputation per PLN.Query.
- The facts' claim that "mirror sentences are inert unless a -down source is queried" is wrong:
  Derive forward-chains every task.
- Base rates must be strictly positive. `and` is eager (metta.pl:131), so
  `smallest-intersection-probability` divides by P(A) even when P(A) = 0 (lib_pln.metta:43-44, 52).
- Inversion copies s_AB, which equals Bayes only when base rates are uniform. Prevalence priors
  would make inverted links wrong.

## 4. Scans that found nothing (supports the verdict)

- No KB function name collides with a PeTTa builtin, and no unintended builtin head appears in any
  body (scripts/scan_builtin_heads.py).
- No KB function is called at an arity it does not define (scripts/scan_arity.py). PeTTa would
  otherwise turn the call into `partial(F, Args)` or throw overapplication (translator.pl:284-292,
  1383-1393).
- No Atom-typed constructor position holds a call that hyperon would leave unevaluated
  (scripts/scan_atom_positions.py).
- Of the 119 pattern variables typed from declarations, none receives two different types
  (scripts/scan_pattern_types.py). The 153 arrow declarations do still feed the checker, through
  match-pattern typing (type_lang.pl:309-311).
- Facts store no builtin arithmetic calls. No `==` compares a number. `$_` is never used.
