# patham9/PLN (the fork, old architecture) on PeTTa fea83178: static compatibility study

Written 2026-10-24. STATIC: nothing under `.petta/` was executed. Labels used below:
[read] = a line of upstream source or history that I read; [computed] = I re-implemented the truth functions in my own
Python (`work/numcheck*.py`) and compared with pins; [predicted] = my inference from the source, to be confirmed by the spike.
Companion file: `spike_runbook.md`. Generated helper files: `artifacts/` (index in section 7).

## 0. Pins and method

| Item | Pin | Evidence |
|---|---|---|
| PeTTa | `fea83178af3722666d2948d074591d10a3134f56` (2026-10-15, 1183 commits, not shallow) | `git rev-parse HEAD` in `.petta/PeTTa` |
| The fork | patham9/PLN main `55f1751d993f71b8a24da03e3aec94ab40789a59` (2025-08-15, 144 commits), at `.petta/PeTTa/repos/PLN` | `git rev-parse HEAD`, origin `https://github.com/patham9/PLN.git` |
| Bundled lib | `PeTTa/lib/lib_pln.metta`, 465 lines, sha256 `d864c3cf376ad1f9252d153306d92b7aae72a184701c2eb4805c9661eed27ece` | `shasum -a 256` |
| Newer lib (comparison only) | trueagi-io/PLN main `4405956947c4b53c7ff01bd565aa3b114bc970a1` (2026-03-15), `lib_pln.metta` 430 lines | `.petta/PLN-trueagi` |
| Wikis | PeTTa.wiki `f99c898` (2026-09-30), PLN.wiki `91de324` | `scratchpad/feasibility/upstream/` |

Method: `cat`/`grep`/`sed`/`git log`/`git show` on the pinned trees. Own tools only: `work/cmp_forms.py` (diff by top-level
form), `work/numcheck.py`, `numcheck2.py`, `numcheck3.py` (truth functions re-implemented from `Formulas.metta`), `work/mk_artifacts.py`
(patched copies), `artifacts/to_petta.py`, `artifacts/check_pins.py`, all run with `python3 -I`, all text-only. I ran the
system binary `swipl --version` once (prints 10.0.2; loads no upstream file). I did not run `run.sh`, any upstream script,
hyperon on fork code, or PeTTa. One mistaken redirect wrote a copy of an old PeTTa example to `/tmp/_nope`; I deleted it.

## 1. Conclusions

1. **The bundled `lib_pln.metta` is the fork's code made to run on PeTTa, by the fork's author.** 63 of the fork's 81 top-level
   forms are identical modulo whitespace and comments (every `Truth_*` formula, the consistency constraints, every `|-` rule
   and guard, `StampDisjoint`, `BestCandidate`, `LimitSize`, `PLN.Derive`/`PLN.Query` logic). The other forms are a port done in
   PeTTa's first PLN commit `f5a4f72` (2025-09-29, Patrick Hammer, "minpln example"), plus renames and a forward declaration
   later. What it drops is the hyperon + metta-morph glue: `Translator.metta`, `PLN.Init` (a Python extension, `examples/PLN.py`) and the fork's tests.
2. **Truth formulas are textually identical, so numerically identical for equal inputs** ([computed]: my re-implementation
   reproduces PeTTa's pins `pln_roman` (0.5, 0.9473684210526316), `pln_tuffy` (0.5199204545454544, 0.8290782204129108),
   `pln_direct` (0.51, 0.8432620011567381) and the fork's FlyingRaven pin (0.97052375, 0.81)). What differs is control flow and
   printing, section 2.3.
3. **The fork's `src/` built by `build.sh` is not directly usable on PeTTa.** Three defects of its own (unary minus in
   `PriorityRankNeg`, `STV` not forward-declared, stdout trace in the deriver), one PeTTa collision shared with the bundled lib
   (the builtin `Predicate`), and the Translator cannot run (its `trace!` does not emit equations in PeTTa). All have small fixes
   (section 4.2); the fixed files are in `artifacts/`.
4. **None of the fork's 11 example/ruletest files runs unmodified** [predicted]: each stops at line 1, `(import! &self PLN)`
   (hyperon resolves `PLN.py` as a Python module; PeTTa wants `PLN.metta` next to the importing file).
5. **Recommendation:** vendor PeTTa's bundled `lib_pln.metta` (sha256 above) with a 1-hunk patch (drop the per-step stdout trace)
   and a provenance header, plus both MIT licences; do not vendor the Translator or `PLN.py` in v1. Use the fork-faithful
   `artifacts/fork_PLN_A1.metta` only inside the spike, to prove the port is faithful and to reproduce the fork's own pins.
6. **The fork's Translator is not just a speed-up; it changes numbers** [computed]: it compiles `(Implication A (Implication B C))` into
   `MP(MP(a,b),imp)` while generic modus ponens gives `MP(b, MP(a,imp))`. They agree when the premise strength is 1.0 (Smokes)
   and differ otherwise (Toothbrush 0.9051004676960268 vs 0.9013534742103302). The restricted rule set (no Translator) therefore
   reproduces the newer trueagi-io lib's Toothbrush strength, not the fork's pin.

## 2. Fork versus the bundled lib

### 2.1 History [read: `git log`, `git show` in `.petta/PeTTa`]

| Commit | Date | Author | Message (as written) | What it did to the PLN file |
|---|---|---|---|---|
| fork `55f1751` | 2025-08-15 | P. Hammer | Merge pull request #54 from trueagi-io/Negation | last fork commit; Translator + Deriver + Rules + Formulas as in section 2.2 |
| PeTTa `f5a4f72` | 2025-09-29 | P. Hammer | "Update: minpln example" (no body) | **creates `examples/minpln.metta`**: the fork's concatenated `src/` with the port below, plus the Tuffy smokes example with `(= (STV (Concept Anna)) ...)` clauses |
| `aef7756` | 2025-10-14 | P. Hammer | "Update: PLN code renamed" | rename to `examples/pln.metta` |
| `b3ae36f` | 2025-10-26 | P. Hammer | "excludefast renamed to exclude-item, memberfast to is-member" | follows the builtin renames in `src/metta.pl` |
| `ccf834f` | 2025-10-29 | P. Hammer | "forward declaration of STV function in pln.metta" | adds `(= (STV $stv) (empty))` |
| `bcbcc7f` | 2025-10-30 | P. Hammer | "Update: lib_pln" | becomes `examples/lib_pln.metta`; the Tuffy example is split out to `pln_tuffy.metta`, a new `pln_roman.metta`; `Truth_c2w`/`Truth_w2c` declarations move above first use |
| `69e7625` | 2025-10-30 | P. Hammer | "MeTTa libraries moved to lib folder" | now `lib/lib_pln.metta` |
| `703b7e2` | 2026-07-28 | R. Treutlein (Co-Authored-By Claude) | "Library-trusted declarations, typed libs, and effect-polymorphic arrows" | `->` becomes `-[det]->` / `-[semidet]->` on 5 declarations |
| `5d3f457`, `3d35b16` | 2026-02-19/20 | R. Treutlein | "port queue and dedup improvements from benchmark variant" / "fix PLN tests by removing incompatible heap queue path" | **not merged**: only on `origin/task2-libpln-testfix` (heap queue, `ConcatUnique`) |

The reasons for the port are not stated in any commit message (first commit has an empty body). Reasons below are inferred.

### 2.2 Function by function

Identical (63 forms; only whitespace/comments differ): `clamp`, `and5`, `min5`, `/safe`, `negate`, `invert`, `InsertSorted`;
`smallest-intersection-probability`, `largest-intersection-probability`, `conditional-probability-consistency` (bodies),
`Consistency_ImplicationImplicantConjunction`; `Truth_Deduction`, `Truth_Induction`, `Truth_Abduction`, `Truth_ModusPonens`,
`Truth_SymmetricModusPonens`, `Truth_c2w`, `Truth_w2c`, `Truth_Revision`, `Truth_Negation`, `Truth_inversion`,
`Truth_equivalenceToImplication`, `TransitiveSimilarityStrength`, `Truth_transitiveSimilarity`, `simpleDeductionStrength`,
`Truth_evaluationImplication`; all 17 `|-` rules and the two guards (`SymmetricModusPonensRuleGuard`, `SyllogisticRuleGuard`);
`StampDisjoint`, `BestCandidate`, both `PriorityRank` clauses, `LimitSize`, the 5/3/2-argument `PLN.Derive` wrappers, both `ConfidenceRank` clauses, all four `PLN.Query` arities.

Different:

| Item | Fork `src/` | Bundled lib | Effect |
|---|---|---|---|
| `TupleConcat` | `collapse`/`superpose` (Utils.metta:18-19) | `(append $Ev1 $Ev2)` (lib_pln:5-7) | same result and order; faster |
| `TupleCount` | recursion with `cdr-atom` (Utils:22-23) | `(length $tuple)` (10-12) | same |
| `InsertionSort` | insertion sort using `<` and the 2nd argument (Utils:62-68) | `(msort $L)`, 2nd argument ignored (51-52) | same on integer stamps; fork throws on non-numeric ids (`<` raises), `msort` sorts any term |
| `Without` | `collapse` + `==` (Utils:71-75) | `(exclude-item $a $Tuple)` = `exclude(==(A))` (61-62; metta.pl:204) | same |
| `ElementOf` | `==` membership (Utils:78-82) | `(is-member ...)` = unification `member` (69-70; metta.pl:195-196) | differs for non-ground items; unused elsewhere in the bundled file |
| `Unique` | prepends while scanning (Utils:85-92) | `(list_to_set $L)`, 2nd argument ignored (77-78) | **order**: fork returns first occurrences in REVERSED order ([computed] `Unique (3 1 2 1 3) ()` = `(2 1 3)`), bundled keeps order `(3 1 2)`; changes tie-breaking in `BestCandidate`/`LimitSize` |
| `Test` | `Test` (Utils:48-49) | `Test2` (37-38) | rename; PeTTa has a builtin `test`; reason not stated |
| `PriorityRankNeg` | `(- $c)` (Deriver:23) | `(- 0.0 $c)` (406) | PeTTa has no unary minus, see 3.3 |
| `PLN.Derive` | `(trace! (SELECTED $steps (Sentence $x $Ev1) (DERIVED $derivations)) (PLN.Derive ...))` (Deriver:46) | `(let $temp (trace! (SELECTED $steps (Sentence $x $Ev1)) 42) (PLN.Derive ...))` (430) | both print one stdout line per step in PeTTa (3.1); bundled drops the `DERIVED` list |
| `PLN.Config.*` | 20 / 20 / 200 (Config.metta:4,7,10) | 100 / 10 / 100 (370,373,376) | different defaults (steps / task queue / belief buffer), changes results when the budget binds |
| `STV` | not in the library; the Translator emits `(= (STV t) tv)` for each static concept sentence | forward declaration `(= (STV $stv) (empty))` (116); callers add clauses | see 3.3 (load order) |
| Declarations | plain `->` (Constraints:3,7,11; Formulas:58,62) | `-[det]->`, `-[semidet]->` (90,94,98,118,122) | PeTTa-only syntax; hyperon cannot read it |
| `StampConcat` | absent | present (385-388), unused | none |
| `Truth_Identity` | Formulas:138 | absent | only the Translator needs it |
| `Translator.metta`, `PLN.Init`, `examples/PLN.py` | present | absent | static KB compilation (and `Not` inside implications) is lost |

### 2.3 Where the two differ numerically

* **Formulas: nowhere.** All 15 truth functions are textually identical [read], and they reproduce the pins [computed].
* **Printing / equality.** PeTTa prints shortest-round-trip floats (e.g. `0.5199204545454544`, metta.pl `write/1` via parser.pl:32) and `==` is Prolog
  identity (metta.pl:61). The fork's pins carry 15 significant digits: its Smokes pin `(stv 0.519920454545454 0.829078220412911)` is exactly
  `%.15g` of the doubles `0.5199204545454544`, `0.8290782204129108` that the formulas give [computed]. So a correct run prints `(Passed: false)`
  for the fork's own tests. `artifacts/check_pins.py` compares with a relative tolerance (default 1e-12) and labels such a line NUMERIC.
* **Control flow.** `Unique` order (above); `Config` defaults; `LimitSize` keeps at most `size-1` items in both.
* **Translator versus generic modus ponens** (above; `work/numcheck3.py`): fork-translated 0.9051004676960268, generic 0.9013534742103302.
* **Versus the newer trueagi-io lib** (comparison only): different confidence formulas (e.g. deduction confidence `(* (* PQs QRs) (* PQc QRc))`
  instead of `min`; modus ponens confidence `(* (* Ps PQs) (* Pc PQc))` instead of `c1*c2`), so its pins differ: Smokes (0.6, 0.486) vs the fork's (0.5199.., 0.8290..);
  `memberDeductionA` (0.75, 0.375) vs (0.75, 0.5); `transitiveSimilarity` (1.0, 0.445) vs (1.0, 0.5).
* **Engine arithmetic.** `/` raises on a zero divisor (metta.pl:55, `R is A/B`); `/safe` guards only some calls. Node strengths of exactly 0 reach
  `smallest-intersection-probability` (`(/ ... $As)`, Constraints:5) and raise in the fork, the bundled lib and PeTTa alike (the project's LAUNCH.md already requires a base rate above zero).

## 3. Construct status on PeTTa fea83178

Status: supported / partial / unsupported. Evidence is `file:line` in `.petta/PeTTa` (fork lines in `.petta/PeTTa/repos/PLN/src`).

| Construct (fork use) | Status | Evidence and behaviour |
|---|---|---|
| `trace!` (Translator:3,7,40; Deriver:46) | **partial** | `translator.pl:412-413` rewrites `(trace! A B)` to `(progn (println! A) B)`. `progn` evaluates every argument (translator.pl:681-685), `println!` writes the serialised value to **stdout** (metta.pl:250-251), not stderr, and does not obey `silent`. Hyperon's `trace!` writes to stderr (examples/PLN.py:35 captures it with `2>>`) and, judging by how the Translator uses it (unbound variables, `$ResultFn` heads inside the printed equation), takes its first argument unevaluated [inferred; hyperon source not read]. |
| Equations produced at runtime (Translator's `(= (\|- ...) ...)`) | **unsupported as written; supported through `add-atom`** | A `(= ...)` that is a *value* is the builtin unify-and-answer-bool (`'='/3`, metta.pl:63), so `(trace! (= (\|- ($Link1 $TV1) ($Link3 $TV3)) ...) msg)` would CALL `\|-` with the unbound premises and print `true`/`false` or nothing: **not asserted, not printed as source, effectively ignored** (and `($ResultFn ...)` variable heads would be dispatched, translator.pl:1008-1027). Equations are installed only (a) as top-level forms of a loaded file or string (filereader.pl:166-184) and (b) by `add-atom &self (= ...)` (spaces.pl:26-73, compiles to a clause, bumps dependents). The `add-atom` payload is NOT evaluated (translator.pl:873-879), so it can carry the call's bound variables; that is how `examples/pln_direct.metta:4-8` works. `quote`d equations stay data (translator.pl:968-971). |
| `case` (Deriver:42,45,70; Translator:12-36) | supported | `translator.pl:727-740`, `1482-1495`: nested if-then-else, first match by unification; no matching branch makes the whole `case` fail; a nondeterministic key is matched per solution. Mis-shaped `case` compiles as data with a warning (722-726). |
| `superpose` (Utils:19,72,79; Deriver:3-4,39,42,70) | supported | metta.pl:139; literal list becomes a disjunction (translator.pl:637-640). **Trap not used by the fork:** a literal empty `(superpose ())` fails to compile (translator.pl:1533-1534 has no clause for `[]`). |
| `collapse` | supported | `findall` (translator.pl:641-650). |
| `let`, `let*`, destructuring `(let (Sentence $x $Ev1) ...)` | supported | translator.pl:769-778, 1428-1429; a failing pattern fails the branch. |
| `(Sentence ...)` pattern matching in heads (Deriver:19-23, 64-65) | supported | `constrain_args` translator.pl:5-17, 76-78; `Sentence`/`stv` are not functions, so they are structural. |
| Result tuples with a variable head `($B tv)`, `($Tasks $Beliefs)` (Rules, Deriver:35) | supported with a trap | Unknown head, so `reduce([F|Args])` at run time (translator.pl:1008-1027, 294-309): data unless `F` is an atom naming a registered function (then it is CALLED). A conclusion term that is a bare atom such as `id`, `first`, `exp`, `test` would be called and, on error, silently dropped (`catch(call(Goal),_,fail)`, line 300). KB terms such as `(Concept x)`, `(Effect ...)` are lists and safe. |
| `==` on tuples (Deriver:3,35; Utils) | supported | structural identity: metta.pl:61, fused to native `==` in `if` conditions (translator.pl:1372). `1 == 1.0` is false; variables compare by identity. |
| `car-atom`, `cdr-atom` | supported | metta.pl:180-189; `(car-atom ())` raises, the fork only calls it after an `== ()` test. |
| `import!` of a file by name (`PLN`, examples line 1) | **partial** | metta.pl:344-386, 414-426: a path relative to the importing file's directory, `.metta` appended, `.py` imported as a Python module. No module-name resolution, so `PLN` means `<dir of file>/PLN.metta` (missing: `existence_error(source_sink,'PLN')`, metta.pl:368-369). |
| `empty` | supported | metta.pl:140 (`fail`). |
| `min`, `max`, `+ - * /`, `< > <= >=` | supported, binary only | metta.pl:52-58, 77-82; inlined to native arithmetic (translator.pl:1337-1354). `<` on a non-number raises. |
| unary `-` (`PriorityRankNeg`, Deriver:23) | **unsupported** | `(- $c)` is an under-applied binary call: `partial('-',[C])` (translator.pl:284-286, 1383-1393; `arity('-',3)` only, metta.pl:53, 441-447). A later `(> partial partial)` raises a type error. Fires on the first queue overflow in `LimitSize`. |
| `and`, `or`, `not` | supported, eager, Bool only | metta.pl:129-133; short-circuit forms are `and-then`/`or-else` (translator.pl:742-767). Both operands of `and` are evaluated, so a division inside the second operand still runs. |
| Number-typed signatures `(: f (-> Number Number Number))` (Constraints:3-12; Formulas:58,62) | supported and enforced | compile-time check plus runtime guards where types are unknown: translator.pl:1109-1166, README.md:43-60. A call whose argument is definitely the wrong shape is rejected at compile time (`examples/fail_inferred_literal_mismatch.metta`). |
| `(: ...)` declarations | supported | declaration prepass, filereader.pl:77-82; arrow annotations `-[det]->` are PeTTa-only (README.md:347-373). |
| Definition order across files (`STV` used in `\|-` bodies) | **partial** | Within one file or string the function set is registered up front (filereader.pl:127-130), across loads a head that is not yet a function is compiled as data (translator.pl:993-997) and warned about later (filereader.pl:114-118). |
| `Predicate` as a data constructor (Rules:56-74, FlyingRaven, Smokes, 2 ruletests) | **unsupported without a workaround** | `Predicate` is a registered builtin function: metta.pl:336 (`'Predicate'([F|Args], Term) :- Term =.. [F|Args]`), 462 (registration), builtin_registry.pl:198. `(Predicate friend)` therefore CALLS it and fails (`friend` is not a list), so any list containing it vanishes. In rule heads it is turned into a call (translator.pl:12-25). Workaround used by the newer trueagi-io tests: `!(translatePredicate (retractall (Predicate (fun Predicate))))` before anything else (`PLN-trueagi/ruletests/evaluationImplicationRuleA.metta:1`); translator.pl:899-904. Not exercised by PeTTa's own tests. |
| `halt` inside `test` / `assert` | note | metta.pl:256-260 (test) and 262-266 (assert): a mismatch calls `halt(1)`; under janus this can end the Python process. |

### 3.1 What the Translator would do in PeTTa (item "asserted, printed or ignored")

`PLN.Init` in the fork is `examples/PLN.py:25-54`: it runs `sh build.sh`, copies `Translator.metta`, appends `!(superpose <static KB>)`,
runs the **hyperon `metta` CLI** and captures **stderr** (`2>> TRANSLATED.metta`), because hyperon's `trace!` prints its
unevaluated first argument there; then it strips `#` from variable names with a regex and calls metta-morph `compile!`.
In PeTTa none of those steps exists. If `Translator.metta` is loaded as is, `(sentence S ids)` calls reach `buildlink`, whose
`trace!` evaluates the equation instead of printing it (table above): the equations are **ignored** (and may raise from the
unguarded arithmetic when a rule happens to match the unbound premises). Two working routes:
**R1** keep the Translator under hyperon (hyperon stays in the loop, as the fork does), save the stderr text, load it into PeTTa as a file; PeTTa's parser accepts hyperon's `$TV1#123` variable names as ordinary names, so the regex is not needed (and it could merge `$A#12` with `$A1#2`);
**R2** port it: replace the three `trace!` by `(progn (add-atom &self (= ...)) msg)` and rename `sentence` to `PLN.Translate` so it does not clash with the `sentence`->`Sentence` alias (`artifacts/Translator_petta_R2.metta`, 3 call sites + 1 entry point; **untested**, written statically). `pettamorph.py` (PeTTa/python/HE, commit 97b85a0, 2025-10-07, "a better (PeTTa) mettamorph compiler for Hyperon-Experimental") is PeTTa's own replacement for metta-morph if the hyperon-hosted flow is wanted.

### 3.2 Per-step trace

Both fork and bundled print per Derive step through `println!` on stdout (the fork also prints the whole `DERIVED` list). With the Python interface this is process stdout, not part of the returned result list. Remove it (patch P3 / B1).

### 3.3 Other traps that matter for this project

* PeTTa function names are global to the process. The fork/bundled names (51 distinct function names across the fork build and the bundled lib) do not collide with any `(= (name ...))` in this repo's `.metta` files [computed]; they do collide with PeTTa's own `lib_nars` (same `Unique`, `Without`, `PriorityRank`, ...), so never import both.
* Loading the same text twice in one process duplicates clauses (and answers); `import!` of one canonical path is idempotent (metta.pl:393-396, 410-413).
* A mismatch of the *file-level analysis* with a later `retractall(fun(...))` is possible if the retract and the code that depends on it sit in the same string (analysis runs before the forms, filereader.pl:75-100, unified_checker_bridge.pl:38-50, 961). Put the retract in an earlier file or earlier `process_metta_string` call.

## 4. Do the fork's own examples run on PeTTa? Edits, comparison, what to vendor

### 4.1 Prediction per file [predicted; pins are the `!(Test ... expected)` lines of the fork]

Unmodified: **0 of 11 run**. Every file starts with `!(import! &self PLN)` and stops there with `existence_error(source_sink, 'PLN')`.
Past that line, each file would hit, in this order: `PLN.Init` is not a function in PeTTa, so `!(PLN.Init (...))` quietly evaluates to a data list
and installs nothing; `STV` and the static implications do not exist; `sentence` is not aliased to `Sentence`; for the 4 files that use
`(Predicate ...)` the KB-building function fails and `!(Test ...)` prints nothing at all (an empty result, not an error);
and the pins would read `(Passed: false)` anyway (15 digits). With the edits of 4.2 (`artifacts/fork_examples_petta/*.metta`, generic route):

| File | Predicate | Static KB | Pin | Prediction with E1-E8 |
|---|---|---|---|---|
| examples/FlyingRaven | yes | concepts only | `((stv 0.97052375 0.81) (1 4 6))`, `((stv 0.332225678571428 0.83690036900369) (2 3 5 6))` | NUMERIC pass if the `Predicate` retract works; the trailing `PLN.Derive` print has no pin |
| examples/Robot | no | 1 implication | `((detection frisbee someCoords1) (detection orange someCoords4))` | pass (a filter; stamps irrelevant) |
| examples/Smokes | yes | 2 implications | `((stv 0.519920454545454 0.829078220412911) (4 9 10))` | stv NUMERIC pass, ids become `(1 2 4 9 10)` (same as PeTTa's `pln_tuffy` pin); may need a larger budget than the fork default 20 steps |
| examples/Toothbrush | no | 6 nested implications | `((stv 0.905100467696027 0.895338254258716) (10 11 12 13 14))` | **fails by design on strength** (generic 0.9013534742103302); passes only on the Translator route R2 |
| ruletests/RuleTester | no | concepts | `((stv 0.01 0.9) (5))`, `((stv 0.99 0.54) (1))`, `((stv 0.994974874371859 0.9) (2))` | NUMERIC pass (needs the full rule set: not-elimination, inversion, equivalence) |
| ruletests/equivalenceToImplication | no | concepts | `((stv 0.98989898989899 0.87) (1))` | NUMERIC pass |
| ruletests/evaluationImplicationRuleA | yes | concepts | `((stv 1.0 0.729) (1 2 3))` | pass if the retract works |
| ruletests/evaluationWithNegationAndInheritanceInversion | yes | concepts | `((stv 0.8 0.6) (1 2))` | pass if the retract works |
| ruletests/inversion | no | concepts | `((stv 0.87 0.486) (1))` | pass |
| ruletests/memberDeductionA | no | concepts | `((stv 0.75 0.5) (1 2))` | pass |
| ruletests/transitiveSimilarity | no | concepts | `((stv 1.0 0.5) (1 2))` | pass |

The three files with a static implication (Smokes, Toothbrush, Robot) are also converted for the Translator route:
`artifacts/fork_examples_petta_r2/`.

### 4.2 Minimal edits (all in `artifacts/`)

| # | Edit | Where | Why | Artefact |
|---|---|---|---|---|
| E1 | `PriorityRankNeg`: `(- $c)` -> `(- 0.0 $c)` | lib (Deriver:23) | no unary minus (3, table) | `fork_A0_to_A1.patch` hunk 3 |
| E2 | add `(= (STV $stv) (empty))` before the rules | lib | `(STV $A)` in rule bodies must compile as a call whatever the load order | hunk 1 |
| E3 | drop `(trace! (SELECTED ...) ...)` and one `)` | lib (Deriver:46) | stdout noise, evaluated argument | hunk 2 |
| E4 | `!(import! &self PLN)` -> `!(import! &self "<abs path to lib>")` | each example | module-name import does not exist in PeTTa | `to_petta.py` |
| E5 | `!(PLN.Init X)` -> `(= (STV t) tv)` per concept sentence; implication sentences become `(= (kbrules) (Sentence ...))` and every `(kbdynamic)` becomes `(TupleConcat (kbrules) (kbdynamic))` | each example | replaces the hyperon/metta-morph compile step (generic route) | `to_petta.py` |
| E6 | `(= (sentence $A $B) (Sentence $A $B))` | each example | alias that `PLN.py` installed after `PLN.Init` | `to_petta.py` |
| E7 | `!(translatePredicate (retractall (Predicate (fun Predicate))))` as the first form, before the lib import | the 4 files that use `(Predicate ...)` | builtin collision (3, table) | `to_petta.py` |
| E8 | compare with tolerance; for the generic route compare only `(stv s c)` | checker | 15-digit pins; ids differ by design | `check_pins.py [--stv-only]` |
| E9 | optional: replace the Utils helpers by the builtins used in the bundled lib | lib | speed (quadratic `Unique`/`Without` on `collapse`), changes tie order | `bundled_lib_pln_B1.metta` is the result |

Order matters in E7: the retract must run before the file that holds `Predicate` heads is compiled and before the lib is imported.

### 4.3 Using the bundled lib unmodified

Its own examples are in PeTTa's CI (`.github/workflows/ci.yml` runs `sh test.sh`, which runs every `examples/*.metta`, including `pln_roman`, `pln_tuffy`, `pln_direct`);
commit `703b7e2` says "suite 912/0" [read, upstream's claim, not reproduced]. Needs from the caller: `(= (STV t) (stv s c))` clauses for every node that takes part in
deduction/induction/abduction/inversion/equivalence/transitive-similarity/member-deduction (revision, modus ponens, symmetric modus ponens, not-elimination and the predicate-usage rules do not read STV), the KB as a tuple of
`(Sentence (term (stv s c)) (id ...))`, no `Predicate` terms (or the retract first, otherwise the 3 predicate-usage clauses compile with function calls in their heads and can bind garbage for any `Evaluation` premise),
and acceptance of one stdout line per step. It has no Translator, so no `Not` inside implications beyond what generic rules give (a derived `(Not X)` is eliminated by the `Not` rule). It lacks `Truth_Identity`, which the Translator needs.

### 4.4 Recommendation and provenance

Vendor, in this repo (suggested `vendor/pln_fork/`):

* `lib_pln_fork.metta` = `PeTTa/lib/lib_pln.metta` @ fea83178 (sha256 `d864c3cf...`) + header comment (provenance, below) + the B1 patch (drop the per-step trace; `artifacts/bundled_B0_to_B1.patch`);
* `LICENSE.NARTECH` = the fork's `LICENSE` (MIT, "Copyright (c) 2025 NARTECH", sha256 `922e8a4a2003a4325dd44fa8e8f16cd7277c513de447ea0a3a6b0908c81bee51`; identical to trueagi-io/PLN's);
* `LICENSE.PeTTa` = PeTTa's `LICENSE` (MIT, "Copyright (c) 2025 Patrick Hammer", sha256 `451e08f082f06f100facd2ef99f3c661155a7a433af091889d4c04e264838ce5`);
* `PROVENANCE.md`: patham9/PLN `55f1751d993f71b8a24da03e3aec94ab40789a59`, trueagi-io/PeTTa `fea83178af3722666d2948d074591d10a3134f56`, the file hashes in 0 and 4.4, the patch list, and "not vendored: `Translator.metta`, `examples/PLN.py`, the fork's tests".

Why this and not the other two:
(a) fork `src/` as built by `build.sh` (`artifacts/fork_PLN_A0.metta`) fails on the three defects of section 1, item 3 (and its `Translator`/`PLN.Init` half needs hyperon plus metta-morph).
(c) the fork plus a patch set converges on (b): fork + E1-E3 + E9 + Config + `Test2` + arrows == the bundled file, and (b) is the variant upstream exercises in CI. Use (c) as `fork_PLN_A1.metta` in the spike (Stage A) to prove the port is faithful and to reproduce the fork's own pins; it keeps the fork's slow helpers and tie order.
(b) matches Ben's phrase "the PLN version from patham9 github on the PeTTa Atomspace" literally: this file is patham9's PeTTa port of that code. Caveat to tell the owner: PeTTa's wiki (Libraries-and-extensions.md) says `lib_pln` is trueagi-io/PLN, but `(import! &self (library lib_pln))` loads the bundled, fork-derived copy (metta.pl:4-8, 21-24); trueagi-io's newer file is loaded by `(library PLN lib_pln)` after a `git-import!`. There are three different things called lib_pln.

Licences: both MIT, so copying is permitted if each copyright and permission notice is kept in copies or substantial portions. The bundled file carries no header of its own, so it sits under PeTTa's MIT but derives from NARTECH's MIT code; keep both. The formulas cite `trueagi-io/hyperon-pln` (`DeductionDTL.metta`, Constraints.metta:2, Formulas.metta:4) and opencog/pln `.scm` files only in comments; I did not check those projects' licences (unknown) and saw no copied code.

Restricted rule set hint (the design belongs to other agents). In the bundled file the `|-` clauses are: revision 255-257, modus ponens 260-262, symmetric modus ponens 270-273, deduction 280-286, induction 290-295, abduction 298-303, predicate usage 307-325, not-elimination 328, inheritance and implication inversion 331-336, equivalence to implication 339-345, transitive similarity 348-352, evaluation implication 355-359, member deduction 363-367. Keep revision, modus ponens, deduction and not-elimination; delete or guard the rest. The smallest switch is a guard function that fails in restricted mode, used in place of `SyllogisticRuleGuard` in the induction and abduction clauses and added to the two inversion clauses.

## 5. How a program or a Python caller makes PeTTa load a library

[read] unless marked.

| Mechanism | Behaviour | Evidence |
|---|---|---|
| `!(import! &self NAME)` / `"path"` | a path relative to the **importing file's directory** (a stack of working dirs pushed per loaded file), or to the process cwd when no file is being loaded (the Python `process_metta_string` case); `.metta` appended when missing; absolute paths work; `.py` names import a Python module; missing file throws `existence_error(source_sink, NAME)`; a nested import does not fall back to cwd | metta.pl:344-386, 414-426; filereader.pl:9-16, 35, 41; python/tests/test_imports.py (`test_nested_import_does_not_fall_back_to_cwd`) |
| idempotence | a second import of the same canonical path in the same space is a no-op; into another space only atoms are replayed, functions stay as compiled once | metta.pl:393-396, 410-413; filereader.pl:102-108 |
| `!(import! &self (library NAME))` | rewritten to `library-import!`; file `<PeTTa>/lib/NAME(.metta)`, "library-trusted" declarations | translator.pl:436-437; metta.pl:4-24, 350-354 |
| `(library X Y)` (two arguments) | a directory registered in `library_path/1` whose path ends in `X`, plus `/Y` | metta.pl:5, 9-11 |
| `git-import!` (needs `(import! &self (library lib_import))` first) | `!(git-import! URL)` clones with `--depth 1` into `./repos/<name>` **relative to the process cwd**, skips the clone when the directory exists, registers the directory in `library_path/1`; `!(git-import! URL "build.sh")` runs `sh build.sh` there; `!(git-import! URL BUILD BASE SHA40)` is the pinned, locked, clean-checkout-only form | lib_import.pl:81-108, 264-282; tests/test_git_import.sh |
| layout | `<PeTTa>/repos/` is git-ignored; `repos/PLN` here is a plain clone of the fork at the pin (origin `https://github.com/patham9/PLN.git`); PLN's wiki expects exactly this | .gitignore, `.petta/PeTTa/repos/PLN` |
| `cached-import!`, `load_metta_file_cached` | snapshot replay of a load keyed by file, state and runtime digest, stored in `$PETTA_LOAD_CACHE` or the user cache dir; output printed during the load is not replayed | metta.pl:429-431; load_cache.pl:1-46, 288-295 |

Fork-specific consequences:

* The fork's repo root has **no** `PLN.metta` or `lib_pln.metta`; `PLN.metta` exists only after `sh build.sh` (`cd src; cat Utils Constraints Formulas Rules Config Deriver > ../PLN.metta`, build.sh:1-2). `(library PLN PLN)` would need that build first, which is why vendoring (a plain file, absolute path) is simpler than `git-import!` (network, build step, clone into cwd).
* A library file **can** be loaded from an absolute path: `(import! &self "/abs/dir/lib_pln_fork")` (extension optional).
* The library **can** be concatenated into a single string given to `process_metta_string`: the string is parsed into top-level forms, all `(= (f ...) ...)` heads are registered first (filereader.pl:75-100, 127-130), so internal order does not matter. Constraints: do it once per process (a second load duplicates every clause); keep the `Predicate` retract in an earlier unit; `import!` lines inside the string resolve against the cwd.
* Python: `PeTTa(verbose=False, petta_path=...)` consults `src/main.pl` and `python/helper.pl` into an in-process janus engine once, then `load_metta_file(path)`, `load_metta_file_cached(path)`, `process_metta_string(code)` each return a flat `list[str]`, one entry per solution of every `!` form (python/petta/__init__.py:29-87, python/helper.pl). State persists between calls. `verbose=False` means silent compile output. Package `petta` is not pip-installed here: put `<PeTTa>/python` on `sys.path` (`artifacts/py_run.py`). `janus_swi 1.5.3` and `sexpdata 1.0.2` are present in `/Users/kolya/PycharmProjects/pln_longevity/.venv`; Homebrew `swipl` is 10.0.2 (PeTTa documents >= 9.3).
* `test`, `assert` call `halt(1)` on mismatch: run one example per Python process.

## 6. Unknowns the spike must settle

1. Does janus-backed PeTTa start on this Mac (macOS dylib caveat, SWI 10.0.2 versus the 9.3.x PeTTa documents)? (runbook G0)
2. Does the `Predicate` retract trick work at this pin (it is exercised only by the older trueagi-io tests), including when `(STV (Predicate IsFat))` heads sit in the same file as the retract? (probe_predicate, ruletests)
3. Does unmodified A0 fail to load (compile-time type mismatch on `(STV $A)` data) or load with dead rules? (runbook step F1)
4. Does the unary-minus crash appear exactly at the first queue overflow? (probe_queue)
5. Is `add-atom` of translator-generated `|-` clauses (R2) correct and fast enough at hundreds/thousands of clauses? (each add triggers dependency recompilation; spaces.pl:60-73)
6. Cost: `consult(main.pl)`, load time of the lib, per-query latency, RSS (fork helpers versus builtin helpers).
7. Which results does the project's float/int/bool printing normalisation need (`true` lowercase, `1.0e-5` style, ints stay ints)?

## 7. Index of generated files (all under `/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/forkdesign/compat/`)

| File | Content |
|---|---|
| `petta_compat.md`, `spike_runbook.md` | this note and the runbook |
| `artifacts/fork_PLN_A0.metta` | exactly what the fork's `build.sh` produces (461 lines, equals `cat` of the six files; checked with `cmp`) |
| `artifacts/fork_PLN_A0S.metta` | A0 + only the STV forward declaration (isolates the unary-minus defect) |
| `artifacts/fork_PLN_A1.metta`, `fork_A0_to_A1.patch` | A0 + E1, E2, E3 |
| `artifacts/fork_PLN_A1B.metta` | A1 with default budget 100 steps / 50 tasks / 400 beliefs |
| `artifacts/bundled_lib_pln_B0.metta`, `bundled_lib_pln_B1.metta`, `bundled_B0_to_B1.patch` | the bundled lib (sha256 verified equal) and the recommended 1-hunk patch |
| `artifacts/Translator_petta_R2.metta`, `Translator_to_R2.patch` | the Translator with `add-atom` (untested) |
| `artifacts/to_petta.py`, `fork_examples_petta/`, `fork_examples_petta_r2/` | text converter and its outputs for the 11 files (`@LIB@`, `@TRANS@` are placeholders) |
| `artifacts/hyperon_translate.py` | R1 helper: the fork's Translator under hyperon, equations on stderr (untested, executes fork code) |
| `artifacts/check_pins.py` | numeric pin checker (`--stv-only`) |
| `artifacts/py_run.py` | one-file-per-process Python runner |
| `artifacts/probes/` | construct probes with predicted outputs |
| `work/` | my scratch: `cmp_forms.py`, `numcheck*.py`, `mk_artifacts.py`, historical copies of the lib, `fork_built_PLN.metta` |
