# Feasibility verdict: full migration of pln_longevity (linage2) from hyperon-experimental 0.2.10 to PeTTa, with trueagi-io/PLN (lib_pln) at the infer/explain seam

Written 2026-10-23. Read-only study. Nothing was executed (SWI-Prolog is deliberately absent); every
claim is from the four fact-finding reports or from my own reads of the repo at
`/Users/kolya/PycharmProjects/pln_longevity` (branch linage2 @ 47130aa) and the upstream checkouts under
`scratchpad/feasibility/upstream/` (PeTTa @ fea83178, 2026-10-15; trueagi-io/PLN @ 4405956, 2026-03-15).
"unknown" means: cannot be settled without running something.

Inputs (their full notes are the authoritative detail):
- INVENTORY: `scratchpad/feasibility/inventory/metta_constructs.md` (37 .metta files, 10,486 lines; 32 root files / 9,364 lines form the runtime stacks)
- COUPLING: `scratchpad/feasibility/coupling/python_coupling.md` (one execution site, string results, 16 test modules drive hyperon directly, 8 abort-probe modules)
- PETTA: `scratchpad/feasibility/upstream/petta_facts.md`
- PLN_LIB: `scratchpad/feasibility/upstream/pln_lib_facts.md`

---

## 0. Verdict

**Doable: yes, with three conditions. Confidence: medium.**

1. The runtime port is a bounded engineering job. The app touches hyperon at exactly one site
   (`pln_chat/core/pln_runner.py:734-831`), every downstream consumer parses result STRINGS, and PeTTa
   supports the KB's core (match incl. flat conjunctions and evaluated templates, superpose/collapse,
   let/let*, if, multi-clause `=`, nested-pattern heads, car/cdr/cons-atom, 5-arg foldl-atom,
   unique-atom, pow-math/log-math, strings, special-character symbols, one-space text concatenation).
   The incompatibilities I could establish from source all have mechanical fixes:
   `(superpose ())` does not compile, 46 facts store an unevaluated `(evidence-confidence T)` call that
   PeTTa will not reduce, one nested `(, A (, B C))` conjunction matches nothing, `True`/`False` print
   lowercase, `car-atom ()` and `log(0)` raise instead of returning values, a known function with no
   matching clause yields nothing instead of itself, and Python gets one flat result list per call.
   Five constructs plus the whole platform/performance story are unknown until something runs, so the
   week-1 spike is a real go/no-go, not a formality.
2. Putting lib_pln at the `infer`/`explain` seam is feasible only through the up/down node split
   (`A-up`/`A-down`, one evidence id per source record); the two simpler mappings are ruled out by
   lib_pln's own code. It requires a base-rate STV for every node (none exist today), changes every pinned
   (s, c) (chain confidences fall 2-11x and go to zero through a null link; strengths rise slightly),
   turns each query into a closure recomputation under a confidence-ranked queue whose defaults are too
   small for this KB, and loses `explain`'s derivation record (rebuildable from stamps on a unique-path
   DAG). Those are semantics decisions for Ben and the owner, not engineering ones.
3. "A more sophisticated statistical/heuristic control mechanism on a large network" is mostly a
   KB-curation job (base rates, sentence modelling of DrugAge/GenAge/CellAge/hallmark rows, evidence
   ids, synonyms). lib_pln's list-based queues are quadratic and every PLN.Query recomputes from scratch,
   so whole-KB inference stays out of reach; per-query scoping (today's DrugAge/CellAge slices) remains,
   repurposed from "avoid the hyperon panic" to "fit the step/queue budget". On the curated 30-node graph
   (unique paths) PLN changes the numbers, not which conclusions exist; the control only matters once
   several paths or rows exist (DrugAge), which is where the demonstration should be.

What the migration buys, honestly: catchable errors instead of process aborts, no ~1,024-key-atom cap
(so ~1,050 app lines and ~700 test lines of scoped-stack machinery can go), a terminating chainer with
revision across rows/paths, and alignment with Ben's stack. What it costs: a second runtime (SWI-Prolog
>= 9.3 + janus + a 0.1.0-versioned PeTTa with 8 open issues and Linux-only CI), a re-pin of every number,
a redesign of provenance and the hand-composed confidences, and an unknown per-process load cost that
may force a warm-worker redesign of the deadline mechanism.

---

## 1. Compatibility table (every construct the KB or the translator relies on vs PeTTa @ fea83178)

Risk legend: **none** = supported with the semantics the KB needs; **needs rewrite** = a known
incompatibility with a known fix; **unknown-until-run** = only a run settles it.

| # | Construct (INVENTORY) | Uses | PeTTa status (evidence) | Risk |
|---|---|---|---|---|
| 1 | `match` on a threaded space variable (`$s`/`$space` bound to `&self`) | 84 | supported: `Space` is whatever symbol the arg is bound to, `Term =.. [Space, Rel \| PatArgs]` (PeTTa/src/spaces.pl:232-240); `&self` passed as an argument is a plain atom; all files load into `'&self'` (filereader.pl:72) | none |
| 2 | `match` conjunction `(, …)` | 14 | flat conjunctions supported (spaces.pl:218-223); the ONE nested form `(, A (, B C))` (cellage_calibration.metta:160-163) turns the inner `(, B C)` into a lookup of a `,`-headed fact (`append([Space], Head, List), Term =.. List`, spaces.pl:220) and matches nothing | needs rewrite (1 site: flatten) |
| 3 | `match` template evaluated after binding | 14 | supported: the template is translated like any expression, known function heads are called (translator.pl:987-992 smart dispatch) | none |
| 4 | `match` on `(: x T)` / `(= …)` atoms, bare-variable pattern | 4 | supported: value declarations are space atoms; `(= …)` forms are `add_sexp`-ed BEFORE being compiled (filereader.pl:163) so `(match &self (= (evidence-confidence $e) $c) …)` finds them; a variable pattern enumerates (spaces.pl:226-229) | none |
| 5 | variable in head position of a sub-pattern; expression-headed patterns | 5 | supported by construction: patterns are Prolog lists, `[Dir, CellularSenescence]` unifies; an expression head becomes the first ARGUMENT of the space predicate (spaces.pl:234) | none (low unknown) |
| 6 | `superpose` | 41 | over a bound list: supported (`superpose(L,X) :- member(X,L)`, metta.pl:139; literal list -> disjunction translator.pl:637-640). **`(superpose ())` does not compile**: `build_superpose_branches([], _, [])` then `disj_list([], Disj)` has no clause (translator.pl:1533-1538); no PeTTa example uses the form (grep examples/ lib/ src/: none). 25 matching lines in 8 root files (grep -c) | needs rewrite (replace with `(empty)`, which both runtimes have) |
| 7 | `collapse` | 59 | supported: `findall` (translator.pl:641-650); nothing -> `()`; order = clause/fact assertion order | none (see #21 for order) |
| 8 | `let` (variable; pattern as filter; fan-out over nondeterministic values) | 84 | supported: `(Pv=V)` unification, failure = no result, backtracking multiplies (translator.pl:769-778; examples/letext.metta, tests.metta) | none |
| 9 | `let*` | 77 | supported: rewritten to nested `let` (translator.pl:1428-1429) | none |
| 10 | `if` | 109 | supported; divergence: any guard other than the atom `true` silently takes the else branch (translator.pl:1358-1361). KB guards are `==`/`<`/`>`/`and`/`not`/user Bool predicates, which yield true/false; a guard that would be an unreduced expression or NaN in hyperon behaves differently | unknown-until-run (low) |
| 11 | `=` definitions: zero-arity constants, literal/string dispatch, nested-pattern heads, multi-equation union, arity overloading, cross-file extension of `infer` | 282 | supported: one clause per equation in source order (filereader.pl:160-171); string heads unify (SWI strings); arity overloads are distinct predicates; cross-file extension works because the FIRST PASS registers every `(= (f …))` head over the whole concatenated string before compiling anything (filereader.pl:72-82, 126-131), so the "defined after use" warning (filereader.pl:113-116) does not fire for the concatenated load. A second caller `(= …)` still ADDS a clause, so `_guard_extra_atoms` (api.py:528-596) stays | none |
| 12 | unmatched function call reduces to itself | 3 design sites + 1 test pin | divergent: an UNKNOWN head compiles as data and comes back unreduced (same as hyperon), but a KNOWN function with no matching clause fails -> zero results (PeTTa/README.md:670). The presence tests (pln_risk_prediction.metta:246-252, nhanes_baseline.metta:79-85) guard the unknown-layer case and still work; tests/test_nhanes_common.py:732-754 pins the other case | needs rewrite (re-pin the test; audit the 3 sites) |
| 13 | recursion (insertion sorts, tuple folds, graph recursion with no cycle guard) | 20 heads | supported (Prolog recursion, last-call optimisation); a cycle loops forever as today -> deadline | none (speed unknown) |
| 14 | type declarations `(: X Type)`, `(: x T)`, `(: NotStated Atom)` | 225 | supported: value declarations are ordinary space atoms (README.md:782-783); duplicates stored twice, harmless | none |
| 15 | arrow types `(: P (-> … Atom))` on 143 fact predicates + 2 typed functions (`-> EvidenceCategory Number`) | 153 | a bare `(: f (-> …))` does NOT register `f` as a function (register_fun is called only from definitions: filereader.pl:110-131, spaces.pl:56, translator.pl:860; precache_fn_type_decl in decl_store.pl:541 only caches), so `Effect`, `scored`, `HasPredictor` … stay data and the "Atom-return = unevaluated body" rule (translator.pl:329-330) bites no function in this KB; facts are type-checked at load only in declared typed spaces (spaces.pl:40-44, 197-201). The two typed functions get strict literal-argument checks (README.md:69-83). Unknown: the checker's reaction to duplicate declarations (logical_predicates.metta:109-112), `Type`/`Expression` as types, and data constructions in rule bodies against constructor types | unknown-until-run (medium) |
| 16 | arithmetic `+ - * /` (binary) | 80 | supported (`R is A+B`, metta.pl:52-56); ints stay ints (`3` where hyperon printed `3.0`, docs/abductive_diagnosis.md:92); `(* $cab …)` on a `$cab` bound to a stored `(evidence-confidence T)` call is a list -> `is/2` type error -> the query raises (see #25) | needs rewrite (via #25; re-pin int/float prints) |
| 17 | `pow-math` / `log-math` | 4 | `pow-math` = `**` (metta.pl:101); `log-math` = `log(X)/log(Base)` (metta.pl:104), so the base-10 call is right; `log(0)`/`log(-5)` raise evaluation errors in SWI by default where hyperon printed `-inf`/`NaN` (tests/test_nhanes_common.py:757-774); the input guards nhanes_reference.metta:253-270 already keep them out | needs rewrite (test pins; SWI float flags unknown) |
| 18 | comparisons `== < > >=` | 107 | supported (metta.pl:57-80); `==` is structural identity: `()`=`[]` and symbols fine; `(== 1 1.0)` is False but the KB never `==`-compares numbers | none |
| 19 | `and`/`not`, `True`/`False` literals | 16 | supported, eager and Bool-only (metta.pl:129-136); `True`/`False` are READ as `true`/`false` and PRINT lowercase (parser.pl:82-84) -> `(witnessed True)`, the `True` existence template (few_shot_examples.json:134) and `linage2_router.py:259-266` string compares change | needs rewrite (small: parsers accept both) |
| 20 | numbers: floats, ints, negatives, scientific | 861 | parser accepts `2.0e-75` (dcg/basics number//1, parser.pl:64); output via SWI `write/1`: shortest round-trip, `1.0e-10` style, `-0.0`, ints as ints (tests/test_parser_swrite.pl:57-73); `_STV_RE` (pln_runner.py:588-591) already accepts exponents; whether `1e-05` without a mantissa dot parses is unknown (the ETL writes fixed notation, nhanes_common.py:985-998); digit pins (tests/test_patient_stack.py:115,124) may or may not survive | unknown-until-run (low) |
| 21 | `car-atom` / `cdr-atom` / `cons-atom` | 82 | supported (metta.pl:143-144, 180-189); `(car-atom ())` THROWS (metta.pl:186; open #237 item 4) where hyperon returned an Error atom; the collapse sites guard with `(== $x ())` first (pln_counterfactual.metta:109, patient_profile.metta:262, pln_risk_prediction.metta:261, human_evidence.metta:281) and the folds have base cases. The FIRST answer of a collapse is now assertion order, which may differ from hyperon's at the 7 order-dependent sites (INVENTORY §4.11) and change the chosen derivation | unknown-until-run (medium: order) |
| 22 | `foldl-atom` 5-arg `(foldl-atom $xs init $acc $x body)` | 3 | supported: `T = [List, Init, AccVar, XVar, Body]` -> Prolog `foldl` with a lambda (translator.pl:817-825) | none |
| 23 | `unique-atom` keeping the first occurrence | 3 | supported: `list_to_set/2` keeps the left-most copy (metta.pl:148-149) | none |
| 24 | empty tuple `()` as value/pattern/compare target | 85 | `[]`, prints `()` | none |
| 25 | ground facts with nested expression arguments, incl. 46 facts storing `(stv s (evidence-confidence T))` | 368 (46 stored calls) | facts load as data (filereader.pl:152-156, spaces.pl:2-10); PeTTa does NOT re-evaluate values bound from a space ("Match does not automatically interpret the returned pattern as code", examples/eval.metta:5-9; he_atomspace.metta:3-6), so `combine-tv` (pln_deduction.metta:89-91) multiplies a list and the query raises; results must still print numbers for `_STV_RE` | needs rewrite (46 facts in mechanistic_bridges/supplement_evidence/lifestyle_evidence/linage2_core + expansion_schema.py:232-240 writer, or reduce in the 3 lifter rules via `eval`/a dispatch clause) |
| 26 | symbols with `%`, `/`, `->`, `?`, mixed case | 40 | Prolog atoms hold any text; printed unquoted by `write/1` (parser.pl:32) | none (low unknown: tokenizer on `mg/kg/week`) |
| 27 | strings with `;`, parentheses, apostrophes, em dashes | 163 | SWI strings with escapes (parser.pl:91-94); comment stripper is string-aware (filereader.pl:216-222); printed with quotes (human_evidence.py:51 already strips) | none |
| 28 | comments, multi-expression lines, structured `;; CODE — desc` comments read by Python regex | 235 | `;` comments stripped, several forms per line accepted (top_forms parser); the Python regexes read the file TEXT, unchanged | none |
| 29 | `&self` and one-space composition by text concatenation (no `import!`) | 1 | `process_metta_string` loads into `'&self'` (filereader.pl:72); no module scoping issue because nothing is imported. BUT the space and compiled clauses are process-global with no clear (python/petta/__init__.py:29-60; filereader.pl:160-171) -> one KB load per process | none (executor already forks per query) |
| 30 | top-level `!` per query expression and per-expression result groups | 1 site | `!` forms run as `findall` (filereader.pl:160-165) but Python receives ONE flat list per call (filereader.pl:100; python/helper.pl:6-10) -> the adapter must call once per expression to keep `expr_index` grouping | needs rewrite (adapter, ~95 lines) |
| 31 | unused-but-whitelisted builtins callers may send (`case`, `empty`, `quote`, `unify`, `new-space`, `get-type`, `min/max-atom`, `sqrt/abs-math`, `xor`, `<=`, `!=`, `fst`/`snd` …) | 0 in KB | present: case, empty, quote, get-type, get-atoms, add/remove-atom, size/index/min/max/decons-atom, sqrt/abs-math, xor, <=, != (metta.pl:449-463); absent/partial: `new-space` (undefined), `unify` (lib_he only), `fst`/`snd` and `sin-math` unknown, `function`/`return`, `format-args` | needs rewrite (re-check `_BUILTINS` metta_validator.py:23-55, `_STRUCTURAL_HEADS` linage2_router.py:122-129, registry.py:43-54) |
| 32 | `py-call`/`py-atom`/`get-metatype` | 0 | not used | none |
| 33 | scoped-stack and budget workarounds for the 1,024-key-atom abort (generic/patient/HE/LinAge2 stacks, 60 KB file cap, row caps, flat records, Python lever gate) | 12 | no cap on space size found: a space is dynamic Prolog predicates (spaces.pl:2-10); only the 8 GB stack limit (python/petta/__init__.py:8); but per-process consult + compile of ~9.4k lines and query latency are unmeasured | unknown-until-run (performance; removal optional) |
| 34 | translator templates with undefined heads: `(pair $x $y)`, `(if (> $c 0.7) (pair $e $c) (empty))` | 5 few-shots | an unknown head compiles as data (translator.pl:999-1003) and survives unreduced, as in hyperon; `if`/`>`/`empty` inside a template evaluate | none |
| 35 | `(rank-drugage-lifespan (…))` intercepted by regex before MeTTa | 3 few-shots | Python-side, unchanged (drugage_router.py:66-70) | none |
| 36 | ~30 space-threaded procedure forms `(infer &self A C)`, `(diagnose-patient &self P)`, `(linage-… &self P)` … | ~30 | plain calls of compiled predicates once the KB loads | none |

**Counts: 36 rows; none 22; needs rewrite 9 (#2, #6, #12, #16, #17, #19, #25, #30, #31); unknown-until-run 5 (#10, #15, #20, #21, #33).**

Translator-emitted shapes (INVENTORY §6, 46 few-shots + 13 form families) map onto rows 1-5, 19, 30, 34-36;
none needs a prompt change beyond the "OpenCog Hyperon" wording and the lowercase `true` in the existence template.

---

## 2. Blocking unknowns (only an actual run settles them)

1. **PeTTa on the developer Mac.** Upstream CI is Linux-only (`swipl:latest` Docker, PeTTa/.github/workflows/ci.yml:117-121); the README asks for SWI-Prolog >= 9.3.x (PeTTa/README.md:9) while Homebrew ships 10.0.2 with no Python dependency listed; whether that build has `library(janus)` (loaded unconditionally, PeTTa/src/metta.pl:27) and whether `pip install janus-swi` builds and loads (`DYLD_LIBRARY_PATH` caveat) are unknown. No macOS mention anywhere upstream.
2. **lib_pln main on the current PeTTa.** PeTTa's own test.sh exercises only its BUNDLED older lib_pln (examples/pln_tuffy.metta, pln_direct.metta, plntest*.metta; PeTTa/test.sh:85-93), which has different confidence formulas (min-based) and pinned numbers. trueagi-io/PLN main (4405956, 2026-03-15) predates PeTTa's #209/#237 behaviour changes (specializer, `catch` normalisation, `car-atom ()` raising); whether `repos/PLN/test.sh` passes on fea83178 is unknown.
3. **Per-process cost.** Each forked child pays `janus.consult(main.pl)` + the unified-checker analysis + compilation of the concatenated stack (patient stack 18 files; full root 32 files, 9,364 lines) before its one query. hyperon queries take 10-600 ms today. If PeTTa's load is seconds, fork-per-query dies and a warm worker per stack (KB loaded once; per-query atoms via `add-atom`/`remove-atom`; restart on kill) is needed, which reopens the un-aimable-kill problem measured in executor.py:170-188. `load_metta_file_cached` (snapshot replay) applies to files, not strings.
4. **Type checker verdict on this KB.** 153 arrow declarations, two typed functions, duplicate declarations, `Type`/`Expression` as types, constructor-shaped data in rule bodies. A static error aborts the whole load with no per-form recovery (filereader.pl:98-100, 136-149).
5. **Failure mode of a form that does not compile** (e.g. `(superpose ())`): whether `process_metta_string` raises, returns None (Python sees `[]`), or leaves a PARTIAL space (clauses asserted before the failing form stay). A silent partial load is the dangerous case; the adapter needs a load-verification step.
6. **Result strings vs the pins:** float digits (tests/test_patient_stack.py:115,124), `3` vs `3.0` (coverage ` 2.0 `, :289,363), `-inf`/`NaN` vs exceptions, `$_N` variables, lowercase booleans, `[]` vs `[[]]` empty shape.
7. **Answer order** at the car-atom-of-collapse and unique-atom sites (pos-transmission, resolve-lever, cf-component, species-tier-conf, patient-z, patient-baseline, human-evidence-absent) — assertion order vs hyperon's order decides which derivation is "first".
8. **Deadline and abort story under janus:** fork safety with child-only import (today's discipline), whether janus releases the GIL (inline-mode deadline), memory growth under the 8 GB default stack limit, the two `halt(1)` paths (`assert`/`test`, PeTTa/src/metta.pl:255-265) reachable from `/metta/run`, and whether SIGKILL/EOF handling behaves as today.
9. **lib_pln control cost at the sizes this KB needs:** closure 100 (256 sentences after the split) plus inversion/induction/abduction by-products, queue sizes in the hundreds, list-based quadratic `LimitSize`/`ConcatUnique` (lib_pln.metta:29-34, 370-374), recomputation per query (:416-421), `println! SELECTED` per step (:391) — whether one PLN.Query fits the 60 s deadline for the 30-node graph and for a 170-row compound.
10. **Reader edge cases:** `1e-05` without a mantissa dot (LLM-emitted), `%` and `mg/kg/week` tokens, em dashes in strings, UTF-8 comments.

---

## 3. Phases with go/no-go gates

Branch: `petta-migration` derived from linage2 (new branch; linage2 untouched). Every phase keeps hyperon
green behind a `PLN_RUNTIME=hyperon|petta|auto` switch until Phase 6 retires it.

| # | Phase | Work | Gate (go/no-go) | Parallel? |
|---|---|---|---|---|
| 1 | **Spike: install and run** | Install SWI-Prolog (Homebrew; fall back to the swi-prolog.org macOS build or a Linux Docker image), `pip install janus-swi`, clone PeTTa @ fea83178 and trueagi-io/PLN @ 4405956 (pinned). Run: (a) `repos/PLN/test.sh` and `examples/Smokes.metta`/`FlyingRaven.metta` both via `run.sh` and via `PeTTa().process_metta_string`; (b) the patient stack (pln_runner.patient_stack order) and the 10-file load order of pln_deduction.metta:160-164 concatenated as one string, then the five section-5 smoke queries (:167-186: link-effect, infer x2, explain, the unconnected pair) — expected `(signed Pos (stv 0.6 0.85))`, `(signed Pos (stv 0.357 0.29089125))`, `(signed Neg (stv 0.2499 0.17017138125))`, the Chain, and nothing; (c) one query per taught shape: all 46 `few_shot_examples.json` entries plus the 13 system_prompt form families, through a 30-line throwaway adapter, with the PeTTa error text per failure. Also measure: load time and RSS per stack and for the full 32-file root, `(superpose ())` failure mode, type-checker output, car-atom order, float/int/bool strings, `log-math 10 0`, fork-after-child-import, a janus GIL probe. Write the breakage table. | GO if: janus-backed PeTTa runs on the Mac (or in Docker with the Mac as editor); lib_pln's own examples pass or fail for understood reasons; the four smoke answers appear after at most the mechanical edits of table rows #6/#25; per-process load is seconds, not tens of seconds; no silent partial loads. NO-GO otherwise: stop, or re-plan around a Linux dev box and a warm-worker design before Phase 2. | sequential (1 agent + the owner for installs) |
| 2 | **Runtime adapter + test harness** | `pln_chat/core/metta_runtime.py` (the COUPLING §6 Protocol: `available()`, `run_program(kb_text, expressions) -> list[list[str]]`, `MettaRuntimeError`); hyperon and petta implementations (petta: one `process_metta_string` for the KB, one per `!` expression, stdout capture, `stack_limit` lowered, load verification); `_hyperon_run -> _runtime_run`; config `PLN_RUNTIME`; `/health.runtime_name` with the existing keys kept; executor comments/messages, child-only import, spawn fallback; requirements split (`hyperon` optional, `petta`+`janus-swi` + documented SWI dependency); one shared pytest runtime fixture replacing the 16 direct `MeTTa()` fixtures and `importorskip("hyperon")` sites, with per-module process isolation (pytest-forked/`--forked` or a subprocess runner) because PeTTa is one global space per process; `test_caller_patients.py:232-249` and the other multi-load tests restructured. | Full suite green on hyperon (zero regression); the suite RUNS end to end on PeTTa with a written failure list (no hangs, no crashes); `pln_mode`, `/health` keys and the `routed` values unchanged. | sequential (1-2 agents) |
| 3 | **Dialect fixes, file by file** | Nine clusters, each in its own worktree, each verified by its test module on BOTH runtimes: (a) pln_deduction + evidence/epistemic_calibration + mechanistic_bridges (stored calls, combine-tv, decls) ↔ test_pln_deduction; (b) pln_intervention_ranking + pln_abductive_diagnosis ↔ test_hallmark_targeting, test_drugage_calibration; (c) patient_profile + pln_supplement_recommendation + supplement_evidence (`(superpose ())`, car-atom, booleans) ↔ test_patient_grounding, test_supplement_recommendation; (d) pln_counterfactual + pln_risk_prediction + lifestyle_evidence (`(superpose ())`, presence tests, hand-composed confidences) ↔ test_counterfactual, test_risk_prediction, test_smoking_lever; (e) human_evidence + hallmark_targeting + hallmarks_* ↔ test_human_evidence, test_gene_queries; (f) drugage_calibration + cellage_calibration + ETL writers (nested conjunction, stored calls, row slices) ↔ test_drugage_*, test_compound_names; (g) nhanes_baseline + nhanes_reference + nhanes_common.py (log-math, budgets) ↔ test_nhanes_*; (h) pln_linage2 + linage2_core + linage2_router.parse_sexpr (lowercase booleans, foldl) ↔ test_linage2, test_patient_tab; (i) Python parsers, validator builtin lists, registry, prompts wording, docs ↔ test_api, test_error_surfacing, test_runtime_grounding, test_prompt_size. Each cluster re-pins only its own strings; semantics stay hyperon-compatible (dual-runtime rule). | All 40 test modules green on PeTTa except the hyperon-only abort probes, which are quarantined behind a runtime marker (not deleted yet); UI/API parity tests green; the breakage table from Phase 1 is empty. | PARALLEL (up to 9 agents) |
| 4 | **Retire the scoped-stack workaround** | Measure the full root (32 files + ETL build outputs) on PeTTa: load time, RSS, latency of every routed probe in test_generic_stack/test_patient_stack/test_human_evidence_route. If clean: collapse `_generic_kb` (api.py:3189-3200, app.py:244-251) to one stack, drop `BuiltPatient.shared_atoms`, `patient_atoms_for`, the LinAge2 split/merge (`run_query_parts`, `merge_run_results`, `routed="linage2+generic"` kept as a documented alias), the NHANES/LinAge2 exclusions from `_INFERENCE_STACK`, `PLN_MAX_KB_FILE_BYTES` and its write-gate refusal, move the smoking-lever gate back into MeTTa (pln_counterfactual.metta §3b), delete the 8 abort-probe modules. KEEP: DRUGAGE_STACK/CELLAGE_STACK slices (semantic exclusions, provenance), `drugage_top` in Python, `route_drugage_ranking`, `_guard_extra_atoms`, caps as cost bounds (re-measured). | Routing-agreement tests give byte-identical answers from the single stack; p95 latency within `PLN_QUERY_TIMEOUT_SECONDS`; API.md version note written. If load/latency is bad: keep the stacks (they are harmless) and move on. | sequential (1 agent; judgment) |
| 5 | **lib_pln at the infer/explain seam** | Design (owner + Ben sign-off): vendor/pin trueagi-io/PLN 4405956 (or `git-import!` with SHA, lib_import.pl:98-108), remove `println! SELECTED`; up/down node split with one evidence id per source record (curated Effect atom, HR record, DrugAge row); STV base-rate table (uniform p=0.1, c=0.9 to start; outcome prevalences later, respecting `s_BC <= P(C)/P(B)`); `infer-pln`/`explain-pln` adapters returning today's `(signed Pos\|Neg (stv s c))` and a `(Chain …)` rebuilt from stamps (walk the id->link graph); config for MaxSteps/queue sizes; `PLN_CHAINER=legacy\|pln`. Build (parallel): sentence builder (Python pre-pass or MeTTa), adapters, base-rate file, the replacement of the hand-composed confidences in pln_counterfactual.metta:282-284 and pln_linage2.metta:312-314 by direct `Lever-up -> M-down` queries, DrugAge all-rows-per-compound injection with revision, re-pinned tests with a before/after number table, docs. | Pinned ORDERINGS hold (D+Q > Fisetin > Spermidine; Metformin > Berberine); every changed (s, c) is listed and signed off; a feedback edge terminates; a two-path/multi-row revision case demonstrably merges; one PLN.Query for the patient stack and for the largest compound fits the deadline. | design sequential; build PARALLEL (2-3 agents) |
| 6 | **Deployment** | Server gets SWI-Prolog >= 9.3 + janus, pinned PeTTa/PLN checkouts, `PLN_RUNTIME=petta` (a Dockerfile or install script — none exists today; `git ls-files` shows only two requirements files); dual-runtime rollback kept; `.env.example`, API.md, docs, prompt wording; replay the 28 Sep-5 Oct session log (34 questions; server-log-replay method in memory) and diff answers; clean the root of ETL files before deploying (deployed-server note). | `/health` reports `runtime_name=petta` on the server; the replayed log answers match or every difference is explained by the Phase 5 number table; rollback to hyperon exercised once. | sequential (1 agent + ops access) |
| 7 | **The large network (Ben's point)** | Curation (sequential, domain review): base rates with citations, DrugAge rows as sentences (3,423 rows / 1,027 compounds; synonyms; one id per row), TargetsHallmark and hallmark anchors as Implication sentences, GenAge/CellAge gene -> senescence -> hallmark links, evidence-id policy, canonical ordering for determinism. Engineering (parallel): per-query sentence selection sized to the step/queue budget, control-parameter sweeps, performance measurement, optional MORK spaces (Linux only). | A multi-path/multi-row inference that today's `infer` cannot produce, answered within the deadline with provenance, on >1,000 sentences; or an honest report that lib_pln's list queues cannot run the control at that scale and the control itself (Ben's/patham9's side) needs rewriting. | curation sequential; engineering PARALLEL |

Sequential judgment work: Phases 1, 2, 4, 5-design, 6, 7-curation. Parallel agent work: Phase 3 (9 clusters), 5-build, 7-engineering.

---

## 4. Risks, ranked, with mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **PeTTa maturity and macOS/Python support**: version 0.1.0, 8 open issues incl. behaviour changes (#237), regressions (#259), non-reproducible builds (#263); CI Linux-only; macOS undocumented; janus build against Homebrew SWI 10.x unknown; load aborts on the first error with no recovery. | high | Phase 1 spike before any port work; pin PeTTa and PLN SHAs and vendor them; Docker/Linux fallback for dev; `PLN_RUNTIME` switch keeps hyperon runnable through Phase 6; a load-verification step in the adapter (count expected heads) against silent partial loads. |
| 2 | **lib_pln correctness and the STV question**: every node needs a base-rate `(= (STV node) …)` (the catch-all `(= (STV $stv) (empty))`, lib_pln.metta:67-68, means "no base rate -> no syllogistic inference"); inconsistent base rates yield `(stv 1 0)` (:42-54, :99); confidence `s_AB*s_BC*c_AB*c_BC` (:97) collapses chain confidences 2-11x and zeroes through null links; no 0.9 knob; two lib_pln variants with different formulas; defaults 10/100/100 keep at most 9 tasks / 99 beliefs (:371); no derivation record; `println!` per step. | high | Up/down mapping with uniform base rates first; pin the commit (4405956) and document the formula choice; raise queue sizes/MaxSteps per stack; optional post-scaling `0.9^(|stamp|-1)` for single-path stamps; provenance from stamps; `PLN_CHAINER=legacy` kept for comparison; put the formula/base-rate choice to Ben (questions below). |
| 3 | **Loss of the Pos/Neg sign**: `Implication A (Not B)` never chains (middle-term unification lib_pln.metta:236-237; negation rule :284 is top-level only); "sign outside PLN" loses the sign under revision. | high if mis-mapped, low with the split | Mandatory up/down node split; tests with mixed-sign compounds (e.g. a compound with life-extending and life-shortening rows) asserting both a Pos and a Neg answer. |
| 4 | **Changed numbers break pinned tests and stakeholder-facing outputs**: every (s, c) in test_pln_deduction/test_patient_grounding/test_drugage_calibration/test_drugage_ranking_contract/test_counterfactual moves; `confidence_threshold` semantics shift; the LLM shows confidences to users; "confident null" ITP results become unranked. | high | Phase 5 number table (before/after, per pinned value) signed off by the owner and Ben; API answers carry `chainer`/`runtime_name`; orderings tested rather than digits where possible; keep the legacy chainer selectable for the demo until sign-off. |
| 5 | **Dialect incompatibilities at load/run time** (table rows #2, #6, #12, #16, #17, #19, #25; `car-atom ()` throwing; type checker). All have known fixes; the dangerous mode is a silent partial load. | medium | Phase 1 catalogue; Phase 3 per-file clusters on both runtimes; `(superpose ()) -> (empty)` mechanical; stored-call facts reduced at lift time; nested conjunction flattened; booleans accepted in both spellings. |
| 6 | **Result-format changes break the formatter and the HTTP contract**: flat list per call; `1.0e-5`/ints/`true`/`$_N`; `[]` vs `[[]]`; "known function, no clause" arrives as `status=empty` instead of an unreduced atom. | medium | Adapter restores per-expression grouping; parsers accept both renderings; contract tests pin keys not digits; `ungrounded_predicates`/warnings already explain empty answers. |
| 7 | **Worker deadline mechanism**: PeTTa/janus have no timeout; 8 GB default stack limit; `halt(1)` from `assert`/`test`; fork-after-import unknown; inline mode has no deadline. | medium | Keep fork-per-query with child-only import and SIGKILL; lower `stack_limit` after construction; validator rejects `assert`/`test` heads from `/metta/run`; measure GIL behaviour; spawn fallback exists (`_child_init`). |
| 8 | **Per-process load cost forces a warm-worker redesign** (consult + analysis + compile of 9.4k lines per query). | medium-high (unknown) | Measure in Phase 1; if slow: one long-lived PeTTa worker per stack with `add-atom`/`remove-atom` of per-query atoms and restart-on-kill, accepting the re-warm cost; or snapshot replay if it applies to strings. |
| 9 | **Two runtimes during transition** (requirements, double test runs, docs, server). | medium | `PLN_RUNTIME` switch + pytest runtime markers; time-box the dual period to Phase 6; one CI-like script that runs both. |
| 10 | **"Large network" is KB curation, not engineering**: base rates, sentence modelling, evidence ids, synonyms; lib_pln's quadratic list queues and per-query recomputation keep whole-KB inference out of reach; the hyperon-era caps become budget caps. | high (for expectations) | Scope Phase 7 explicitly as curation-first with a small engineering core; keep per-query scoping; demonstrate on DrugAge rows (where revision has something to merge). |
| 11 | **Ben's expectations vs what a 30-node graph shows**: unique paths mean PLN changes numbers, not conclusions; the control only decides what is REACHED under small queues; by-products (inversion at c 0.27-0.46, induction/abduction <= 0.5) outrank 3-4-hop deductions (0.015-0.08). | medium | Say so up front; pick the demonstration where multiple paths exist; ask which control he means (questions below). |
| 12 | **Determinism/order**: findall order and sentence tuple order decide first-answer sites, `BestCandidate` ties (:356) and eviction. | low-medium | Canonical sort of generated sentences; order-independent assertions; document the 7 first-answer sites. |
| 13 | **Validator/prompt drift**: whitelisted names PeTTa lacks (`new-space`, `unify` without lib_he, `fst`/`snd`); the `True` existence template. | low | Re-check the three builtin lists; prompt wording. |

---

## 5. Effort

Unit: one agent-phase = one agent working one gated task to review (roughly a day of agent time incl. review).

| Phase | Agent-phases | Calendar |
|---|---|---|
| 1 Spike | 1 (+ owner for installs) | 1-2 days (week 1) |
| 2 Adapter + harness | 2 | 2-3 days |
| 3 Dialect fixes (9 parallel) | 9 | 2-3 days with review |
| 4 Retire scoped stacks | 2 | 2-3 days |
| 5 lib_pln seam (design 1, build 2-3, re-pin/review 1) | 4-5 | 5-8 days incl. sign-off |
| 6 Deployment | 1-2 (+ ops) | 1-2 days |
| 7 Large network | 4-6, open-ended | 2-4 weeks, curation-bound |

Total ≈ 23-27 agent-phases. Calendar: 4-6 weeks to a deployable PeTTa + lib_pln branch (Phases 1-6) if
the week-1 spike passes; Phase 7 adds 2-4 weeks and is gated on curation and on Ben's choice of network.
If the spike fails on macOS, add ~1 week for a Linux dev setup before Phase 2.

---

## 6. Questions for Ben (before or during the work)

1. Which PLN exactly: trueagi-io/PLN main (4405956; deduction confidence `s·s·c·c`, no length discount), PeTTa's bundled older lib_pln (min-based confidence), or patham9's fork (not inspected)? Is the resulting confidence collapse on chains (and zero through a null link) intended, or should we pre-/post-scale to keep a per-hop discount?
2. Base rates: every node needs an STV. Is uniform p≈0.1 acceptable for a first cut, or does he want prevalence-style priors per outcome (CHD, mortality), which the consistency check then constrains against every incoming link's strength?
3. Which "large network" and which demonstration: all DrugAge rows revised per compound (3,423 rows / 1,027 compounds), the hallmark/gene graph (TargetsHallmark, GenAge/CellAge), or the curated graph with cycles? And is lib_pln's confidence-ranked queue the control he means, or will he/patham9 supply a different control (attention/priorities) that we should leave a seam for?
4. Deployment: is PeTTa-only acceptable on the server (SWI-Prolog + janus, Linux where PeTTa is tested), or must hyperon remain as a fallback? Does he need the branch runnable on macOS at all?

---

## 7. What I verified myself in PeTTa's source (beyond the four reports)

- First pass registers every `(= (f …))` head and precaches every arrow declaration over the whole string before compiling any form (PeTTa/src/filereader.pl:72-82, 126-131), so cross-file forward references in the concatenated load are safe; the "defined after use" warning (filereader.pl:113-116) concerns later loads only.
- `(= …)` forms are `add_sexp`-ed into the space before compilation (filereader.pl:163), so `match` on equation atoms works (few_shot_examples.json:44,53).
- Plain facts are type-checked at `add-atom` only for declared typed spaces (PeTTa/src/spaces.pl:40-44, 197-201); `&self` is not one.
- A bare `(: f (-> …))` does not call `register_fun` (callers: filereader.pl:110-131, spaces.pl:56, translator.pl:860, specializer.pl:47), so data heads with arrow types stay data under smart dispatch (translator.pl:987-1003).
- Nested conjunction: the inner `(, B C)` is built as a `,`-relation fact lookup (spaces.pl:218-223) and matches nothing.
- `(superpose ())`: `build_superpose_branches([], _, [])` then `disj_list([], _)` has no clause (translator.pl:1533-1538); no example in examples/, lib/ or src/ uses the form.
- `foldl-atom` 5-argument pseudo-lambda is a special form (translator.pl:817-825).
- `unique-atom` = `list_to_set/2` (metta.pl:148-149), left-most copy kept; `car-atom ()` throws (metta.pl:186); `log-math` = `log(X)/log(Base)` (metta.pl:104); `pow-math` = `**` (metta.pl:101); `True`/`False` read as `true`/`false` (parser.pl:82-84).
- PeTTa's test.sh runs its bundled lib_pln examples (examples/pln_tuffy.metta, pln_direct.metta, plntest.metta, plntestdirect.metta; test.sh:85-93, none excluded), not trueagi-io/PLN main.
- lib_pln.metta: `println! (SELECTED …)` every step (:391); defaults MaxSteps 100 / TaskQueueSize 10 / BeliefQueueSize 100 (:326-332); `LimitSize` uses `<` so a queue of size N keeps N-1 (:370-374); `PLN.Query` recomputes the closure and returns `((stv s c) (ids))` for an `==`-identical term (:416-430).
- Repo: pln_deduction.metta §5 smoke queries at :158-186 with the 10-file load order at :160-164; `(superpose ())` on 25 lines in 8 root files; 46 stored-call facts (e.g. mechanistic_bridges.metta:53-54, comment :50-52); the presence-test idiom pln_risk_prediction.metta:230-252; 40 test modules / 14,816 lines; 32 root .metta files.
