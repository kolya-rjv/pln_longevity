# Notes on the PeTTa spike kit

Written while building the kit, for whoever runs it and reads `out/*.md`. Nothing here was measured on PeTTa except the one
incident under "Disclosure". Cites are `file:line` of the work tree (branch `petta-migration`, content = linage2 @ 47130aa;
HEAD `8a2a6a0` adds only the docs and the setup script) or of the pinned PeTTa @ `fea83178`.

## 1. What the baseline recorded (hyperon 0.2.10, `baseline_hyperon.json`)

145 shapes, 144 runnable, 145 KB loads, 177 `!` expressions, recorded in 171 s, work tree unchanged during the run
(`meta.worktree_status_before/after`).

| source | n | note |
|---|---|---|
| smoke | 5 | `pln_deduction.metta` section 5; load order and expected answers are parsed from the comment block |
| few_shot | 46 | entry 7 is a clarification with no program (skipped); 22-24 are the DrugAge route (fixture rows, `score-candidate`) |
| family | 16 (13 families) | the 13 families of `10_metta_constructs.md` section 6, pinned to the patient stack except F10 (DrugAge) and F13 (needs human_evidence.metta) |
| rule_form | 25 | forms named by system_prompt rules 11-17 that no few-shot covers (human-evidence, TargetsHallmark, LinAge2 incl. a mixed two-load program, ...) |
| route | 3 | DrugAge linear, DrugAge metta_sort, CellAge per-row effects |
| canary | 8 | one per stack |
| probe_print | 27 | numbers, booleans, `1e-05`, `(superpose ())`, `(car-atom ())`, `(log-math 10 0)` ... |
| probe_order | 15 | seven primitives + the seven first-answer sites, with atoms injected so each has two candidates |

Status on hyperon: 131 ok, 13 empty, 1 abort. The abort is `canary:full`: `(match &self (TargetsHallmark $i $h) (pair $i $h))` in the full
shared stack kills the process (rc -6, trie.rs:179, the 1,024-key-atom bug); no other shape aborts in its routed stack.
The 13 empties are honest empties: smoke:5 (no false chain), fs:02/03 and fam:F06 (no such atoms exist), fs:08-12 (DrugAge row predicates are not
in the generic space, system_prompt rule 15), fs:37 (supplement irrelevant to the patient), rule:R16-risk (no baseline loaded), the two `(superpose ())` /
`(empty)` probes. Two probes return an `(Error …)` atom (car-atom / cdr-atom of `()`).

The three Phase 2 fixes applied to the KB text change **no hyperon answer** (95 parts compared byte for byte: identical; 49 parts have no fix site).
Counts match LAUNCH.md: 20 `(superpose ())` code lines in 8 files, 25 stored `(evidence-confidence T)` facts (19 + 3 + 2 + 1), 1 nested conjunction.

Stack loads on hyperon take 1.3-7.9 ms (`MeTTa()` 9 ms, RSS 44-50 MB): hyperon does no compilation. Queries dominate: 0.1 ms to 2.7 s
(slowest: `fs:42` linage-scenarios 2.7 s, `fs:34` recommend-supplements 2.5 s, `fs:38` 1.7 s).

### Hyperon behaviours the PeTTa run will be compared against (all measured, `probe:*`)

| probe | hyperon prints | what PeTTa is expected to do (from the source, to be confirmed) |
|---|---|---|
| `(let $x 1e-05 $x)` | `1e-5` (a Grounded number, `get-metatype` = Grounded) | SWI `write/1` prints `1.0e-5`; whether the reader accepts `1e-05` without a dot is **unknown** (parser.pl `number//1` from dcg/basics) |
| `(+ 2.0e-75 0)` | `2e-75` | `2.0e-75` |
| `(/ 6 2)`, `(/ 1 3)` | `3`, **`0`** (integer division) | `3`, `0.3333333333333333` (metta.pl:55 `R is A / B`): a real divergence, not formatting |
| `(== 1 1.0)` | **`True`** (numbers compare by value) | `false` (structural `==`, metta.pl:61) |
| `(< 1 2)`, `(and True False)` | `True`, `False` | `true`, `false` (parser.pl `atom_symbol`: `True`/`False` are read as `true`/`false`) |
| `(log-math 10 0)` | `-inf` | `evaluation_error` raised by `log(0)` unless the SWI float flags say otherwise (metta.pl:104) |
| `(car-atom ())` | `(Error (car-atom ()) "…")` | raises `car_atom_empty` (metta.pl:186) |
| `(superpose ())` | `[]` (no answer) | does not compile (translator.pl:1533-1538, `build_superpose_branches([], _, [])` at :1537) |
| `(known-f 3)` with only `(known-f 1)`, `(known-f 2)` defined | `(known-f 3)` unreduced | fails: no answer (README.md:670 says a callee yields zero results when no clause head matches) |
| `(car-atom (collapse (match …)))` | **`collapse`** (car-atom does not evaluate its argument) | `a` (arguments are evaluated): harmless for the KB, which let-forces every collapse |
| `(collapse (let $x (superpose (1 2 3)) (* $x 10)))` | **`(30 20 10)`** (let fan-out comes back reversed) | `(10 20 30)` (Prolog backtracking order) |
| `(collapse (match &self (Q $x) $x))`, `(collapse (g))` over three equations | insertion order `(a b c d)`, `(1 2 3)` | insertion order |

The seven first-answer sites, with two candidates injected, on hyperon: `pos-transmission` first = the longer path `(stv 0.578… 0.2224…)`;
`resolve-lever` (Neg and incoming) first = InsulinResistance / ChronicInflammation (insertion order); `cf-component`, `patient-z`
(1.7), `species-tier-conf` (0.5), `human-evidence-absent` (MultipleHumanTrials) = insertion order; but **`patient-baseline` returns 0.3, the
SECOND of `(BaseRisk NB_ORD 0.2) (BaseRisk NB_ORD 0.3)`**: the `let*` fan-out inside `baseline-risk-data` reverses the order, so hyperon's
first answer there is the last asserted. If PeTTa returns 0.2 that is an order difference in a pinned number. `measure.md` "the seven
first-answer sites" tabulates raw order and first answer for both runtimes.

Number rendering matters in exactly one repo parser: `linage2_router._contribution` compares `reads[2][1] == "True"`, so a lower-case `true` silently
turns `witnessed` into False (`selftest.py` shows it). Every other parser (`_STV_RE`, `parse_scored`, `parse_cellage_effects`, `parse_sexpr`)
accepts both renderings. The adapter normalises anyway and keeps `raw` next to `norm`; `selftest` checks that `normalise(petta_style(x)) == x` for
every one of the 150+ recorded answers and that the repo's parsers return identical results on `x` and on the normalised PeTTa-style text.

## 2. What the kit settles on its first PeTTa run (the unknowns of 01_verdict.md section 2)

1. PeTTa on this Mac: `import janus_swi` + `PeTTa()`; the first error per stack in `loads.md`. janus is linked to Homebrew `libswipl.10.dylib` (arm64).
2. Per-process cost: import / consult / load seconds and RSS per stack, string vs cached; `measure.md` "consult" separates `main.pl` from `helper.pl` and
   times the first three helper calls (lazy autoloads such as clpfd, pcre, yall).
3. Failure mode of a form that does not compile: `load_error` (exception) vs `load_failed` (goal failed, wrapper would say `[]`) vs a silent partial
   load (`verify.n_missing / n_short`). `limits` shows what a stack overflow, a runaway, `halt(1)` via `assert` / `test`, `remove-atom` on duplicates and
   a second load do. `run_loads.py --variants asis` names the first error of each stack that the Phase 2 fixes must remove.
4. Type checker verdict on the KB: it surfaces as a load exception (filereader.pl:98-100, 136-149); the text is recorded in full (`exception.message`, 4,000 chars).
5. Result strings vs the pins: every shape's raw and normalised strings are in `shapes_report.json`; digit pins at 1e-12 are `within_tolerance` at worst.
6. Answer order at the seven sites; (7) deadline and abort story: SIGKILL works on a child inside a janus call (`limits`), `halt(1)` surfaces as `crash` with exit code 1,
   `fork` after a child-only import and the GIL probe (`gil`: canary ticks during a long Prolog call and two Python threads running Prolog at once; mod_swipl.c
   releases the GIL around some calls, so a result of "released" is plausible, but it was not run).

## 3. Things found while reading PeTTa that the kit is built around

* `PeTTa._run_helper` (python/petta/__init__.py:62-72) maps a failed `run_metta_helper` goal to `[]`. A failed load and a good load of a KB without `!` forms look the same. The adapter
  calls `janus.query_once("run_metta_helper(...)")` itself and keeps the `None`. `use_wrapper=True` reproduces the wrapper's behaviour for comparison.
* `process_metta_string` returns **one flat list per call** (filereader.pl:100 `append(ResultsList, Results)`), so per-expression grouping needs one call per `!`.
* `top_forms` accepts only `(` or `!(` at top level (filereader.pl:206-209): a bare top-level symbol, which `pln_runner._normalize_query` would prefix to `!sym`, is a syntax error.
* Each helper call asserts one `silent(true)` fact (helper.pl:1-2 `maybe_enable_silent(false)`), and `load_metta_file_cached`'s snapshot is keyed by the whole dynamic database
  (load_cache.pl:110 `state_fingerprint`). So in one process a cached load is a hit only if it is the **first** helper call; the adapter's cached mode is therefore always the first call of its child, and `PETTA_LOAD_CACHE` is a kit-local directory.
* `load_metta_file_cached` replays the *file* load; the snapshot also records the digests of every file read and of `src/`, `lib/`, `python/`, so editing PeTTa invalidates it. Output printed while loading is not replayed.
* A function's registration happens in the *first pass* (`parse_form` -> `register_function_signature`), before its clause compiles, so `fun(F)` is true even for a head whose clause failed. The verification therefore asks for **clauses** (`predicate_property(H, number_of_clauses(C))`), with arity = arguments + 1.
* `main.pl:123` is `:- initialization(main, main)`. The wrapper consults it from Python exactly as the kit does, and PeTTa's own python tests do the same.
* `Verbose` is passed as the string `"false"`; the wrapper's own test (`python/tests/test_petta.py`) pins that exact call, so the kit mirrors it.

## 4. Design decisions and why

* **Default deny** (`KIT_ALLOW_PETTA=1`). The kit was built under "do not run PeTTa"; the gate makes running it an explicit act and lets `selftest.py` prove (gate tests) that nothing starts without it.
* **Hyperon engine in the same plumbing** (`engine="hyperon"`) so `run_shapes`, `run_loads` and `measure` can be exercised end to end now: every class comes back `identical`, the pipeline is tested, only PeTTa-specific code
  (the Prolog verification goal, the fork and GIL children) is untested against the real thing; they are tested against a fake `janus_swi` / `petta` written by `selftest.py` (75 offline checks in all, 82 with `--hyperon`; the adapter ones include SIGKILL on a sleeping expression, an `os._exit(3)` crash, a load goal that fails without raising, stdout written below Python, cached miss then hit).
* **Batch by KB** in `run_shapes` (one fresh child per distinct KB, all its expressions in it): 24 loads instead of 145. The one-load-per-process invariant holds; a batch that times out or crashes is re-run one shape per process (`--mode auto`). `--mode isolate` prices the fresh-process-per-request design.
* **Class is judged on PeTTa's raw strings**; `norm_equal` says whether the normaliser hides the difference. A difference the normaliser hides is `number_format_only`; one it cannot is `within_tolerance` / `different`.
* **Routing is the repo's own**: `shapes.py` calls `api._generic_kb`, `parse_drugage_query`, `split_linage2_program`, `patient_atoms_for`; the KB text is `Part.kb_text()` = `pln_runner._hyperon_run`'s join, and `kb_sha256` in the baseline detects a changed work tree (`kb_drift` in the report).
* **Importing the repo**: `stacks.repo_import()` switches python-dotenv off (`PYTHON_DOTENV_DISABLED`, and a patched `load_dotenv`) before `config` is imported, so `pln_chat/.env` is never read; `config.LOGS_DIR` is redirected so the runner's `last_query.metta` is not written into `pln_chat/logs`; bytecode writing is off.
* **DrugAge / CellAge slices come from the committed fixtures** (`tests/fixtures/*_real_rows.metta`, 4 and 9 rows) because `build/` is gitignored and absent. `Ethanol` (fs:24) is taken from `drugage_etl_short.metta`. Latency of the real 3,423-row build is **not** measured.
* **Mixed LinAge2 + generic program** (`rule:R16-mixed`) is two parts, two KB loads, as in production; it is the case the "fresh process per load" rule exists for.

## 5. Known gaps and risks of the kit itself

* `verify_heads` builds a Prolog goal that was only run against the fake. If SWI rejects it the run still proceeds and `verify.error` carries the message (never a failure). If PeTTa names clauses differently than assumed, every head shows as missing and `gate.silent_partial_loads` lists every stack: read `verify.error` and one `missing` entry before believing it; `--strict-verify` is off by default for that reason.
* `fork` probe: `os.fork()` after SWI-Prolog is initialised is the thing being asked; a hang is reported (SIGKILL after 20 s), a crash as `died`. The template's health afterwards is checked with one more query.
* The `gil` loop sizes (`--gil-loop`, `--gil-count`) are guesses for 1-2 s; if a loop finishes in under 0.3 s the tick ratio is noise.
* `limits` runs with `--stack-limit 256 MB`; the deep-recursion probe uses 128 MB. Memory is bounded by the Prolog stack limit only (macOS ignores `RLIMIT_AS`).
* Not covered: lib_pln / the fork's own examples (the fork's `examples/PLN.py` is a hyperon extension that shells out to `metta`, `build.sh`, `~/PLN`, `~/metta-morph`; not PeTTa-ready), `import!`, the validator's builtin lists, the HTTP surface, caller-supplied atoms containing `(= …)`.
* The stored-call fix reads the tier number from `epistemic_calibration.metta`, so a stack without that file keeps its stored calls (only `fork_pln` and nothing else is such a stack).

## 6. For the owner

* Decisions this kit cannot make: whether the Phase 2 fixes are applied as text edits (as here) or by changing the three lifter rules; whether `(== 1 1.0)` and integer `/` differences matter for any pinned number (the KB never `==`-compares numbers; `grep` for `/` on integer literals before relying on that).
* If `loads.md` shows load seconds in the tens, the next experiment is already in the kit: `measure.md` "fork" prices a warm template that forks per query; the executor already forks per query (executor.py:`_mp_context`) and the Phase 1 gate is written for fresh processes.
* After the first run, copy `petta_adapter.py` (without the gate and the hyperon engine), `kit_common.normalise_answer` and `kb_fixes.py` into `pln_chat/`; the rest stays in `scripts/`.

## 7. Disclosure: PeTTa was run once, by mistake

During self-testing, `measure.py --engine hyperon --quick` ran its `loads` and `latency` sections on the **real PeTTa**: those two calls to `run_program_report`
lacked `engine=` and defaulted to `petta`. That is 17 fresh child processes (3 small stacks x cold / cached miss / cached hit, then 8 KB groups), each importing `janus_swi`, constructing `PeTTa()` from
`.petta/PeTTa/python` and loading one of smoke10, human_evidence or cellage_slice (39-66 KB, fixes applied) plus a few queries. It was a violation of "do not run PeTTa" and I noticed it from a
2.5 s `consult` in a run that should have taken milliseconds. What I checked: no file inside `.petta/` outside `.git` is newer than the session start; the three `.git` directories were last modified at 00:24:41, twenty minutes before the incident (00:44-00:46), and I ran nothing in `.petta/` then;
`~/.cache/petta` and `~/.local/share/swi-prolog` do not exist; no swipl process remained; `git status` of the work tree is unchanged; PeTTa's snapshots went to the kit's `out/` (deleted). Two earlier
`selftest.py` runs also started the embedded SWI-Prolog through the real `janus_swi` (an import-error test pointed at a home without a wrapper) without loading any PeTTa code; that test now uses a fake `janus_swi`.

Fixes made: `engine=` passed everywhere; the default-deny gate (`KIT_ALLOW_PETTA`); `selftest.py` unsets it and has gate tests; no kit script is run any more without `--engine hyperon` unless the gate is open.
Incidental observations from that run, **not used anywhere in the kit and not checked against hyperon's answers**: import 0.02-0.03 s; `consult` 2.5 s the first time on this boot, 0.12 s afterwards; string load of
39-66 KB 0.025-0.045 s; cached miss 0.07-0.10 s and hit 0.035-0.04 s (`cache_hit_guess` False then True); RSS 40-50 MB; no load or expression error was reported for those three stacks. Treat them as a hint to run
the real thing, not as a result.
