# Python <-> hyperon coupling map — pln_longevity, branch `linage2` @ 47130aa

Scope: read-only survey of `/Users/kolya/PycharmProjects/pln_longevity` (branch linage2, HEAD 47130aa) for a
runtime port from hyperon-experimental 0.2.10 to PeTTa. Nothing was executed; no test suite or app was run.
PeTTa was read (cat/grep/sed only) from a clone that already existed at
`scratchpad/feasibility/upstream/PeTTa/` (HEAD fea83178, 2026-10-15; the directory was not empty when this
task started, so no new clone was made). "unknown" below means: not determinable without running something.

Line numbers are for the repo files as of 47130aa.

---

## 0. The one-paragraph picture

The application talks to hyperon through exactly ONE execution site, `core/pln_runner.py:_hyperon_run`
(734-831), which does `from hyperon import MeTTa`, builds a FRESH `MeTTa()` per query, runs the concatenated
text of the stack's .metta files plus caller atoms into `&self`, runs the `!`-prefixed program, and converts
every result atom with `str(atom)`. Nothing else in `pln_chat/` imports hyperon; two places only probe
`importlib.util.find_spec("hyperon")` (config.py:64, api.py:1879). Everything downstream is string/regex based.
The REST of the coupling is behavioural, and it is large: a forked worker process with a SIGKILL deadline
(core/executor.py) exists because hyperon holds the GIL and aborts the interpreter with a non-unwinding Rust
panic; five "scoped stacks" and a program router (pln_runner.py 21-335, api.py `_generic_kb`, app.py
`_generic_kb`, patient_context.py, patient_builder.shared_atoms, linage2_router split/merge) exist because
hyperon 0.2.10 decodes a space's trie keys wrongly past ~1,024 distinct key atoms (upstream #1076/#1095, fix
PR #1081 unreleased); three row/size caps (PLN_MAX_KB_FILE_BYTES=60000, DrugAge MAX_ROWS=150, CellAge
MAX_ROWS=100) and several Python re-implementations of inference (drugage_scoring/drugage_top, gene_index,
hallmarks, human_evidence, lever_gate) exist because the engine could not hold the data; and 16 test files
drive `hyperon.MeTTa` directly while 8 run abort probes in subprocesses and assert on exit codes.

---

## 1. Every site that imports hyperon or depends on its behaviour

### 1a. Direct imports and detection

| file:lines | what | adapter must provide |
|---|---|---|
| pln_chat/core/pln_runner.py:767 | `from hyperon import MeTTa` (lazy, inside `_hyperon_run`, so the parent process never loads hyperon and a forked child imports it) | the only import to replace; keep it lazy and inside the worker (see §3: PeTTa/janus must not be initialised in the parent before forking) |
| pln_chat/config.py:53-68 | `_detect_hyperon()`: `PLN_RUNTIME_AVAILABLE` env override else `find_spec("hyperon")` -> `PLN_RUNTIME_AVAILABLE` (read by pln_runner.py:20,855; drugage_router.py:40,150,300,315,338; api.py:74,1880,1889) | `runtime.available()`; for PeTTa this must also check that `swipl` is on PATH (petta imports `janus_swi`, which needs libswipl) — unknown whether `find_spec("petta")` alone is enough |
| pln_chat/api.py:1859, 1879-1880 | `/health.runtime_importable` ("Is `hyperon` importable here?"), `runtime_ready` | same probe via the adapter; keep the KEY NAMES (tests/test_api.py:166-167 and test_api_operations.py:425 pin them); add e.g. `runtime_name` |
| pln_chat/requirements.txt:9-11 | `hyperon==0.2.10` ("calibration + deduction layers are verified against this exact version") | `petta`, `janus-swi`, `sexpdata` (if used), plus a SYSTEM dependency (SWI-Prolog >= 9.3.x) that pip cannot express |

### 1b. The execution site, `_hyperon_run` (pln_runner.py:734-831), piece by piece

| lines | what it does today | adapter need |
|---|---|---|
| 740-765 docstring | loads the KB by CONCATENATING file text into one space, deliberately not `!(import! &self …)` (per-file imports put each module in its own space; `match &self` inside an imported module then reads only that module — docs/calibration_layer.md §10.1, tests/test_nhanes_integration.py:137-139) | keep concatenation; PeTTa compiles a string the same way (`process_metta_string`), so the order-dependence is identical |
| 769-771 | `_normalize_query` (726-731): every top-level expression gets a `!` | unchanged |
| 773-783 | reads `kb_files` (missing/unreadable file silently skipped), appends `extra_atoms` (caller slice/patient) AFTER the files | unchanged |
| 785-795 | writes `pln_chat/logs/last_query.metta` (`;; loaded:` names + program) | unchanged (runtime-agnostic) |
| 797-801 | `metta = MeTTa(); metta.run(kb_text); raw = metta.run(normalized)` — a FRESH space per call; `raw` is `list[list[Atom]]`, ONE GROUP PER `!` EXPRESSION, an expression yielding nothing gives `[]` | PeTTa: `PeTTa().process_metta_string(code)` returns a FLAT `list[str]` (filereader.pl:75-100 `append(ResultsList, Results)`; python/tests/test_load_cache.py: `"!(f 0) !(f 999) …"` -> `['0','999','1000','2499']`; a no-result `!` contributes nothing). The adapter must call `process_metta_string` ONCE PER `!` EXPRESSION after loading the KB to recover the per-expression grouping that `expr_index` (342-345), `merge_run_results` (674-706) and `run_cellage_effects` (1093-1106) rely on. Fresh space per query: see §3 (PeTTa's space and compiled clauses are process-global) |
| 802-806 | `except Exception as run_exc` -> `PLNRunResult(status="error", error_code="runtime_error")` (hyperon raises `RuntimeError` for interpreter errors; nothing narrower is caught anywhere) -> api.py:939-957 maps `runtime_error` to HTTP 502 | PeTTa raises `janus.PrologError` (a Python exception; upstream python/tests/test_imports.py:88-96 `pytest.raises(Exception, match=…)`), so the same `except Exception` works; the message text will be Prolog's, not Rust's |
| 809-813 | `str(atom)` on every result atom -> `PLNAtomResult(atom=…, stv=_stv_from_atom(...), expr_index=index)` — the ONLY contact with hyperon atom objects; no `.get_object()`, `repr()`, `GroundedAtom`/`ExpressionAtom`/`SymbolAtom`/`ValueAtom` checks exist anywhere in pln_chat/ or tests/ (grep confirmed) | PeTTa already hands back strings (helper.pl `maplist(swrite, …)`, parser.pl:9-53); drop the `str()` |
| 815-821 | `_apply_threshold` then `status = "ok" if results else "empty"` | unchanged |
| 822-825 | outer `except Exception` -> same `runtime_error` (covers an ImportError of hyperon) | unchanged |

`run_query` (834-858) is the public wrapper: `if PLN_RUNTIME_AVAILABLE: return _hyperon_run(...)` else `_stub_run`.
The `mode` field ("runtime" | "stub") is an API contract (`pln_mode`, api.py:1239,1358,1889; tests pin
`mode="runtime"`), so keep it.

### 1c. Result conversion and parsing (all string based — this is the GOOD news)

| file:lines | what | adapter need / PeTTa difference |
|---|---|---|
| pln_runner.py:583-598 `_STV_RE` / `_stv_from_atom` | regex `\(stv s c\)`; comment 583-587: hyperon prints 0.00001 as `1e-05`, the old `[\d.]+` missed it and such atoms were "no STV" and escaped the confidence filter | regex already accepts `[eE][-+]?\d+`; SWI-Prolog `write/1` prints floats shortest-round-trip and always with a dot (e.g. `1.0e-5`, unknown exact form) — still matches; `float()` parses both |
| pln_runner.py:415-439 `parse_scored` | regex over `(scored C score (signed Pos|Neg (stv s c)))` | same |
| pln_runner.py:469-538 `_split_top_level` / `_filter_tuple_entries` / `_apply_threshold` | a collapsed ranking arrives as ONE atom holding a tuple; threshold descends one level | same |
| pln_runner.py:1021-1041 `_CELLAGE_EFFECT_RE` | regex over `(Effect G CellularSenescence Pos|Neg (stv s c))` | same |
| pln_runner.py:1093-1106 | aligns effects to injected rows by ORDER (one `!` per row, one link each); guard `len(effects)==len(rows)` | order preserved by per-expression calls; flat list would also preserve order |
| pln_runner.py:926-975 `run_drugage_ranking` linear strategy | one `!(score-candidate …)` per compound, "hyperon returns one result group each" (928-929); parsed by content, sorted in Python | per-expression calls keep it; content-parsed, so grouping is not load-bearing here |
| linage2_router.py:211-251 `parse_sexpr` | hand-written S-expression reader ("`_hyperon_run` hands back every result atom as a STRING", 212); numbers -> float, `"…"` -> str, symbols -> str; feeds `/linage2/analyze` JSON (396-428) | same input shape; `sexpdata` is not needed — this reader already exists. PeTTa prints variables as `$_N` (parser.pl:25-31) where hyperon prints `$x`; the parsers treat both as symbols |
| drugage_router.py:348-357, 385-427 | `parse_scored(result.results[0].atom)` — relies on the FIRST atom being the ranked tuple (contract built by pln_runner.py:966-975) | unchanged (Python builds that atom) |
| utils/formatting.py:46-61 | renders `r.atom` and `stv(...:.2f, ...:.2f)` | unchanged |
| utils/logging.py:46-58 | logs status/mode only | unchanged |
| tests (see §4) | pin hyperon's FLOAT and SPECIAL-VALUE RENDERING: `(point 0.12605177716424967)` (tests/test_patient_stack.py:115), `0.2306046747621094 abs=1e-12` (:124), coverage ` 2.0 ` (:289), `str(...) == "-inf"` / `"NaN"` (tests/test_nhanes_common.py:766-767), `["0.9"]` / `["0.35"]` (tests/test_executor.py:106,138), `[[]]` empty-group shape (`_is_empty` helpers in 7 files) | SWI prints ints as ints (`3` where hyperon's `+` promoted to `3.0`, docs/abductive_diagnosis.md:92); `log(0)`/`log(-5)` raise evaluation errors in SWI unless float flags are set — PeTTa's `log-math` behaviour is unknown; the patient-stack digit pins may or may not survive (same IEEE arithmetic, different printing) |

Empty results: hyperon returns `[[]]` for one expression with no answers; `_hyperon_run` turns that into
`status="empty"`, `results=[]` (815-821). PeTTa returns `[]`. Only the test helpers notice.
Unreduced expressions: hyperon returns the unreduced call when no rule head matches
(tests/test_nhanes_common.py:733-754, docs/nhanes_integration.md:247-252; pln_risk_prediction.metta:232-234
and nhanes_baseline.metta:83-85 DESIGN around it). PeTTa also returns an unreduced call for an UNKNOWN
function (upstream python/tests/test_imports.py:115-131) but for a known function with no matching clause
Prolog fails -> zero results — a dialect difference that reaches Python as `status="empty"` instead of an
atom; flagged, not resolved here.

### 1d. Error model

| file:lines | what | adapter need |
|---|---|---|
| pln_runner.py:802-806, 822-825 | interpreter exceptions -> `error_code="runtime_error"` | keep; PeTTa's `janus.PrologError` is an `Exception` |
| pln_runner.py:356-358 | the two error codes: `runtime_error`, `drugage_build_missing` | unchanged |
| api.py:939-957 `_PLN_ERROR_STATUS` / `_raise_pln_failure` | `runtime_error` -> 502, `drugage_build_missing` -> 503 | unchanged |
| api.py:980-1013 | `PLNExecutionTimeout` -> 504 `pln_timeout`, `PLNOverloaded` -> 503, `PLNWorkerCrashed` -> 500 `pln_worker_crashed` | unchanged |
| app.py:439-452 | the Gradio chat renders those three + `DrugAgePoolTooLarge` as `status="error"` | unchanged |
| drugage_router.py:325-343 | `except Exception` around `run_drugage_ranking` -> `DrugAge ranking failed: …` | unchanged |
| api.py:528-596 `_guard_extra_atoms` | refuses `(= (…) …)` in caller atoms: "hyperon keeps both and every call becomes non-deterministic" (536-541; measured 256 RiskPrediction atoms, one 5.4e11) | KEEP: PeTTa also keeps every clause (spaces.pl `add-atom` asserts; filereader.pl:160-170 `assertz(Clause)`), so the hazard is the same |
| patient_builder.py:1-33 | id collision: "hyperon unions the facts and every accessor goes non-deterministic" (19-23) | keep (same under Prolog clause semantics) |

### 1e. The non-catchable abort, the GIL, and the worker-process-with-deadline

| file:lines | what | why it exists (hyperon-specific) |
|---|---|---|
| core/executor.py:1-49 | module rationale: measured GIL hold (2.46 s / 4.83 s / ratio 0.998; canary 3 of ~386 ticks); abort = non-unwinding panic once a space holds too many distinct head symbols; `except` cannot catch it | both measurements are hyperon's |
| executor.py:80-99 `PLNWorkerCrashed` | message text narrates trie.rs:179, ~1,024 key atoms, #1076/#1095, and the stack routing | text must change |
| executor.py:117-159 | task registry: child-side names (`run_query`, `run_query_parts`, `rank_drugage`, `route_drugage_ranking`, `drugage_top`, `cellage_effects`) resolved INSIDE the child; kwargs must be picklable | unchanged |
| executor.py:162-167, 197-204 | `fork` preferred ("copy-on-write, single-digit ms"; avoids re-importing `__main__`/Gradio); `spawn` last resort with `_child_init` | see §3: fork is only safe because the PARENT never imports hyperon; the same discipline is required for petta/janus |
| executor.py:233-243 `_child_main` | one task, one Pipe message, exit | unchanged |
| executor.py:246-258 `_terminate` | SIGKILL (`proc.kill()`), not SIGTERM: "inside a Rust call that does not check signals"; runaway grows to 3.7 GB in two minutes | still the right tool for a child; a janus child inside Prolog C code is equally un-interruptible by SIGTERM (unknown) |
| executor.py:289-389 `run_offloaded` | admission (`_inflight`, `_effective_inflight_limit`), BoundedSemaphore slots, `parent_conn.poll(remaining)` deadline -> kill -> `PLNExecutionTimeout`; `EOFError` on recv = "child died without sending anything: a hyperon abort" -> `PLNWorkerCrashed(exit code)`; task exception -> `PLNWorkerCrashed("the query raised in the worker: …")` | mechanism is runtime-agnostic; only the EOF comment and messages name hyperon |
| executor.py:272-286 `executor_stats` | `/health.pln_execution` says inline mode cannot enforce a deadline ("you cannot preempt a GIL-holding Rust call", api.py:1824-1828) | wording |
| config.py:125-149 | `PLN_WORKER_POOL_SIZE` (default 2; 0 = inline), `PLN_QUERY_TIMEOUT_SECONDS` (60), `PLN_MAX_INFLIGHT_QUERIES`; comment 126-135 measures the GIL hold | values stay; comments change |
| app.py:333-345, api.py:3465-3468 | every UI/API MeTTa call goes through `run_offloaded` ("hyperon holds the GIL for the whole of MeTTa.run()", app.py:341) | keep; tests/test_executor.py:276-303 and tests/test_ui_api_parity.py enforce it structurally |
| core/rate_limit.py:31 | "single-process demo (hyperon's GIL behaviour is why MeTTa work is offloaded…)" | comment |

### 1f. The scoped stacks and the key-atom budget (hyperon #1076) — every Python site

| file:lines | what | status under a runtime without the bug |
|---|---|---|
| pln_runner.py:21-40 `DRUGAGE_STACK` | minimal layers for lifting+ranking DrugAge rows in a QUERY-SCOPED space; excludes grim_age/hallmarks/mechanistic_bridges ("CHD axis, add atoms, and would let compound names collide with curated Effect bridges") | MUST STAY: the exclusion is also semantic (name collisions), and the slice is Python-chosen |
| pln_runner.py:43-62 `CELLAGE_STACK` | same shape for CellAge; boundary "between 125 and 150 data rows (measured)" | stays (per-row lifting with provenance); cap tunable |
| pln_runner.py:64-115 `NHANES_PATIENT_STACK`, `nhanes_patient_kb` | NHANES layers kept out of the shared space ("~9 head symbols the NHANES layers introduce aborts it") | could merge into the shared stack |
| pln_runner.py:117-152 `PATIENT_STACK_EXCLUDED`, `patient_stack()` | shared stack minus 7 files; measured byte-identical answers; 64 head symbols of margin | could disappear |
| pln_runner.py:155-218 `GENERIC_STACK_EXCLUDED`, `generic_stack()`, `defined_heads()`, `patient_layer_symbols()`, `needs_patient_layers()` | shared stack minus 4 patient layers (~804 vs ~1,064 key atoms); the mechanism paragraph (160-180) is the authoritative description of #1076 | could disappear |
| pln_runner.py:220-245 `HUMAN_EVIDENCE_FILES`, `human_evidence_stack()`, `is_human_evidence_program()` | 4-file stack because `(human-evidence &self Metformin)` aborts the full space (rc -6) | could disappear |
| pln_runner.py:247-335 `LINAGE2_PATIENT_STACK`, `LINAGE2_LAYER_FILES`, `LINAGE2_GENERATED_BASELINE`, `linage2_patient_kb()` | LinAge2 layer "never in _INFERENCE_STACK — the shared space cannot take their head symbols"; minimal underneath to leave room for the generated baseline (margin test +16 heads) | the SCOPING could disappear (LinAge2 files join the shared stack); `linage2_patient_kb`'s optional generated baseline is a data-presence concern and stays in some form |
| pln_runner.py:338-345 `expr_index`, 674-706 `merge_run_results`, 709-723 `run_query_parts` | re-join a program split across two spaces in program order; one worker/one deadline for both halves | could disappear with the split |
| api.py:184-231 `_INFERENCE_STACK` (214-219 note) and app.py:117-159 (mirrors; "keep in sync") | the curated execution stack; NHANES/LinAge2 layers deliberately absent | stays as the KB manifest; exclusions could lift |
| api.py:241-282 `_runtime_kb_paths`, app.py:170-222 | stack minus files over `PLN_MAX_KB_FILE_BYTES`; 248-268 comment measures "full root (26 files) aborts with NO padding / curated stack (25) still answering at +4,096 definitions" | byte filter could go; the print at app.py:213-218 names hyperon |
| api.py:3189-3200 `_generic_kb`, app.py:244-251 | the ROUTER: human-evidence-only program -> HE stack; names a patient / reads patient facts / uses a patient-layer form -> patient stack; else generic stack; "Never the full shared space" | could collapse to one stack |
| api.py:3215-3287 `_run_linage2_program`, `_SPACE_LABELS`, `_validate_linage2_query` (3160-3167); app.py:365-417 | LinAge2 form(s) run in the scoped stack; mixed programs are split per top-level expression and both parts run in ONE offloaded task; validated per space; `routed` = "linage2" / "linage2+generic" (API contract, api.py:1247-1256, 1365-1373) | could disappear, but `routed` values are documented API — a version note |
| api.py:3471-3490 (/query), 3660-3690 (/metta/run); app.py:421-437 | the shared space gets the patient WITHOUT its LinAge2 atoms, and only when the program reads patients (`patient_atoms_for`) | could disappear |
| core/patient_context.py:62-90 `names_a_patient`, `reads_patients`, `patient_atoms_for` | regexes that decide the patient stack: "holding a caller patient, the full space aborts as soon as a program enumerates patient facts" | could disappear (`with_injected` 92-107 stays: validation needs injected symbols) |
| core/patient_builder.py:495-509, 1040 `BuiltPatient.shared_atoms` | the patient minus its 61 LinAge2 atoms, because with them "hyperon 0.2.10 aborts on diagnose-patient, predict-risk-patient and recommend-supplements-patient" | could disappear |
| core/linage2_router.py:1-27, 75-162 (`parse_linage2_query`, `SplitProgram`, `_STRUCTURAL_HEADS`, `split_linage2_program`, `nesting_warnings`), 196-208 `route_linage2_query` | detects/splits/routes LinAge2 forms to the scoped stack | split/route could disappear; `_STRUCTURAL_HEADS` (122-130) is a hyperon stdlib name list used to tell "another layer's form" from a builtin |
| api.py:1075, 2125-2130, 2776-2778, 3191-3192, 3519 | docstrings/comments naming hyperon spaces and the abort | text |
| ontology/scoped_forms.py:1-25, 109-138 | warns that a form's data lives in a scoped space ("a variable-slot match over ~900 rows aborts hyperon 0.2.10", 131-134) | the warning TEXT changes; the data-less-form detection is still true as long as ETL rows are kept out of the generic stack |
| ontology/lever_gate.py:13-26 | the smoking-lever precondition is enforced in PYTHON because "the curated runtime KB tolerates exactly FOUR more trivial rule definitions before hyperon 0.2.10 aborts"; pln_counterfactual.metta §3b records the gap | could move back into MeTTa (a redesign, not a deletion) |
| ontology/human_evidence.py:10-13 | record shape is one atom per study because of the budget (`human_evidence.metta` §1) | KB-side; Python reader unaffected |
| ontology/cellage_selector.py:43-61 `MAX_ROWS = 100` | measured bisection (100 ok … 150 ABORT, trie.rs:179), "enforced BEFORE the hyperon call because the abort is uncatchable"; `select_rows` clamps `min(limit, MAX_ROWS)` (183-224) | the clamp stays as a cost bound; the number is re-measurable |
| ontology/drugage_selector.py:5-12, 38-41 `MAX_ROWS = 150` | "~420 DrugAge rows loaded alone already panics"; 150 rows keeps ~1.8k atoms | same |
| config.py:70-78 `PLN_MAX_KB_FILE_BYTES = 60000` and users: api.py:277-282, 1378-1383, app.py:206-218, ontology/write_gate.py:135-146 (`file_would_exceed_kb_cap`), ontology/expander.py:386-404, ontology/gene_index.py:28-32 | a .metta over 60 KB is skipped at execution ("hyperon 0.2.10 panics … once a space grows past a few thousand atoms") | could go, with the write_gate refusal and `/ontology/files.excluded_from_runtime` semantics revisited |
| config.py:80-100 `PLN_MAX_RANK_COMPOUNDS=60`, `PLN_MAX_METTA_SORT_COMPOUNDS=10`, `PLN_MAX_RANK_ROWS` | cost caps (70 ms per compound; O(n^2) MeTTa insertion sort, 115 s freeze) | stay; numbers depend on PeTTa's speed (unknown) |
| nhanes_common.py:921-972, 1036-1046 (root ETL) | `DEFAULT_ATOM_BUDGET=3000`, `DEFAULT_BYTE_BUDGET=60000` (tracks config), `AtomBudgetExceeded` ("hyperon 0.2.10 hard-aborts (not raises) past roughly 4,000 extra atoms"); `num()` (985-998) writes fixed notation because "hyperon's reader does not accept nan/inf, and exponent forms like 1e-05 are read as symbols by some versions" | budgets could relax; the fixed-notation emitter is harmless to keep (tests/test_nhanes_common.py:683-689, 712-720 pin the budgets) |
| scripts/run_etl.sh:19-24 | refuses OUT_DIR=repo root; comment narrates the 2026-09-28 abort | comment |

### 1g. Python re-implementations that exist because the engine could not hold the data

| file:lines | what | status |
|---|---|---|
| core/drugage_router.py:507-531 `drugage_top` + ontology/drugage_scoring.py:1-30 | whole-build ranking in Python: "1,043 compounds would be 1,043 MeTTa calls, and loading the rows to rank them in one space aborts the interpreter"; constants PARSED from the .metta knobs; parity test vs the engine (tests/test_drugage_discovery.py:64-86) | stays unless PeTTa is both fast enough and roomy enough — unknown; the parity test keeps it honest either way |
| ontology/gene_index.py:1-45 | CellAge/GenAge lookups from the CSVs under data/: files 8-14x over the byte cap, "loading a bulk ETL file into a hyperon 0.2.10 space ABORTS the interpreter" (29-32) | stays (also a prompt/discoverability design) |
| ontology/hallmarks.py:40-41, ontology/human_evidence.py:38-40 | text scans "keeping it out of hyperon keeps it off the worker pool" | stay (cheap, LLM-free endpoints) |
| ontology/inventory.py:37-38, ontology/expansion_schema.py:21, 53-54 | text scans that "cannot panic hyperon" | stay |
| core/metta_validator.py:1-12, 23-55 | no evaluation ("would need the Hyperon runtime"); `_BUILTINS` is a hand list of hyperon stdlib names "checked against `!(get-type …)` on hyperon 0.2.10" (32-33; `flip` deliberately absent) | the list must be re-checked against PeTTa's builtins (lib_he.metta provides hyperon-compat names; `min-atom`/`max-atom`/`pow-math` etc. need confirming); `_NUMBER_RE` (64) already handles exponents |
| core/linage2_router.py:122-130 `_STRUCTURAL_HEADS` | same kind of stdlib-name list | same re-check |
| ontology/registry.py:43-54 `BUILTIN_REGISTRY` | PLN/MeTTa builtin entries for the symbol registry | re-check |

### 1h. Empty-result and confidence-threshold handling (runtime-agnostic, listed for completeness)

- `_apply_threshold` (pln_runner.py:522-538): keeps atoms with no STV; descends into collapsed tuples.
- `status="empty"` when no atom survives (815-821); `/query` and `/metta/run` then attach `ungrounded_predicates`
  (metta_validator.py:212-227), `scoped_form_warnings`, `linage2_form_warnings`, `patient_form_warnings`
  (api.py:3504-3539, 3693-3710) so an empty answer is not read as "no".
- `run_drugage_ranking` filters per compound AFTER scoring with threshold 0 (pln_runner.py:926-975; drugage_router.py:325-351).
- tests/test_drugage_ranking_contract.py:230-262 drives the real `run_query` with a 4-file KB and two
  `infer` chains (c=0.648 / 0.054) — a numeric pin of hyperon's arithmetic that should hold under IEEE doubles.

### 1i. Peripheral mentions

- pln_chat/prompts/system_prompt.txt:2 and prompts/few_shot_examples.json:72 say "OpenCog Hyperon" to the LLM (cosmetic).
- pln_chat/logs/last_query.metta is written per query (pln_runner.py:785-795); logs/ is gitignored (.gitignore:218).
- scripts/counterfactual_chat_demo.py:27,37,151; scripts/risk_prediction_chat_demo.py:27,37,147;
  scripts/supplement_chat_demo.py:27,37,151; scripts/drive_drugage_chat_queries.py:28;
  docs/kb_quick_wins/probes/run1.py:4-17 — call `run_query` in-process with `PLN_RUNTIME_AVAILABLE=true`
  (several calls per process: see §3 on PeTTa's process-global state).
- Untracked (not in `git ls-files`): docs/nl_query_examples/run_examples.py:8-13,28,41,124 and make_pdf.py:96;
  HANDOVER_PROMPT.md (1 mention), REPORT.md (14), report_2.md (14), RESEARCH_BRIEF.md (1).

---

## 2. Stack routing in one paragraph, and what could go

Execution never uses "every .metta in the root": `_runtime_kb_paths()` (api.py:269-282, app.py:198-220) is the
curated `_INFERENCE_STACK` (26 names, api.py:185-231) minus any file over 60 KB. From that list
`_generic_kb(program)` (api.py:3189-3200, app.py:244-251) picks ONE of three sub-stacks: a program made only of
`human-evidence*` forms runs in the 4-file human-evidence stack (pln_runner.py:231-245); a program that names a
patient, reads a patient fact, or uses a form only the four patient layers define runs in the patient stack
(runtime stack minus 7 files, 139-152; detection in patient_context.py:62-81 and pln_runner.py:198-218); every
other program runs in the generic stack (runtime stack minus the 4 patient layers, 181-195). Before that choice,
two routers divert by FORM: `(rank-drugage-lifespan (…))` goes to `route_drugage_ranking` ->
`run_drugage_ranking` in `DRUGAGE_STACK` + a Python-selected slice of at most 150 DrugAge rows
(drugage_router.py:73-91, 430-462; pln_runner.py:861-975); any `linage-*` form goes to `LINAGE2_PATIENT_STACK`
(+ the generated baseline if present) with the caller's full atoms, and a MIXED program is cut per top-level
expression, the LinAge2 half runs there, the rest in `_generic_kb(rest)` with the patient's `shared_atoms`,
both in one offloaded task and re-joined in program order (linage2_router.py:134-154; api.py:3215-3287;
pln_runner.py:674-723). `GET /genes/{g}?infer=true` builds its own `CELLAGE_STACK` + <=100 rows
(api.py:2824-2905; pln_runner.py:1044-1106), and NHANES tests/scripts use `NHANES_PATIENT_STACK`.
A caller patient's atoms enter the shared space only when the program reads patients (`patient_atoms_for`),
and always without its LinAge2 atoms (`BuiltPatient.shared_atoms`).

If the new runtime has no key-atom limit, these exist ONLY because of hyperon #1076 and could disappear:
the generic/patient/human-evidence split and its detectors (`generic_stack`, `patient_stack`,
`human_evidence_stack`, `needs_patient_layers`, `defined_heads`, `patient_layer_symbols`,
`is_human_evidence_program`, `names_a_patient`, `reads_patients`, `patient_atoms_for`), `_generic_kb` on both
surfaces, `BuiltPatient.shared_atoms`, the LinAge2 scoping (`LINAGE2_LAYER_FILES` out of `_INFERENCE_STACK`,
`split_linage2_program`, `nesting_warnings`, `_run_linage2_program`, `run_query_parts`, `merge_run_results`,
`expr_index`, the `routed: "linage2+generic"` value), `NHANES_PATIENT_STACK`'s separateness, the
`PLN_MAX_KB_FILE_BYTES` byte filter and its write-gate refusal, the Python-side smoking-lever gate, and the
eight subprocess abort-probe test modules (§4). Must stay for other reasons: `DRUGAGE_STACK` + the selector
slice (the stack excludes the CHD-axis files to avoid compound/Effect name collisions, and the slice carries
row provenance and the resolver's synonym notes), `CELLAGE_STACK` + per-row lifting (provenance alignment),
`route_drugage_ranking`'s distinct symbol (semantic routing, API `routed: "drugage_ranking"`),
`drugage_top`/`drugage_scoring` in Python (N engine calls per ranking regardless of space size — unless PeTTa
is measured fast), the row/compound caps as COST bounds (re-measured), `linage2_patient_kb`'s optional
generated baseline (data presence), `/linage2/analyze`'s program + `parse_sexpr` (JSON contract),
`_guard_extra_atoms` (duplicate definitions are non-deterministic in PeTTa too), the validator and its
builtin lists (re-checked, not removed), and `scoped_forms` (ETL rows are still not in the generic stack).

---

## 3. Deadline and abort: today, and what PeTTa needs

Today (core/executor.py): every engine call from the API and the UI goes through `run_offloaded(task, kwargs,
inline)` (289-389). With `PLN_WORKER_POOL_SIZE>0` (default 2) it FORKS one child per query (`_mp_context`
197-204 prefers fork; the parent never imports hyperon because pln_runner.py:767 imports lazily inside the
function that only the child runs), sends the result back over a Pipe, and waits with
`parent_conn.poll(remaining)` where `remaining` = `PLN_QUERY_TIMEOUT_SECONDS` (60) minus slot-wait time.
Timeout -> `proc.kill()` (SIGKILL, 246-258) -> `PLNExecutionTimeout` -> HTTP 504 `pln_timeout`. A child that
dies without a message (hyperon's non-unwinding `abort()`, exit -6) -> `EOFError` -> `PLNWorkerCrashed(exit
code)` -> HTTP 500 `pln_worker_crashed`, service stays up. A Python exception inside the task ->
`PLNWorkerCrashed("the query raised in the worker: …")`. Admission: `_inflight` counter vs
`PLN_MAX_INFLIGHT_QUERIES` (default 4x pool) -> `PLNOverloaded` -> 503; a `BoundedSemaphore` bounds concurrent
children. `PLN_WORKER_POOL_SIZE=0` runs the `inline` thunk in the request thread with NO deadline, NO
admission and NO abort containment (`executor_stats` reports it; API.md:229-236), and is what the in-process
contract tests use because a forked child cannot see a monkeypatched `api.run_query`.

What the PeTTa Python interface is (upstream python/petta/__init__.py, read at fea83178):
- `PeTTa(verbose=False, petta_path=None)` is a process-wide SINGLETON: a module global `CONSULTED` + lock;
  the first construction imports `janus_swi`, sets `stack_limit` to `DEFAULT_STACK_LIMIT = 8_000_000_000`
  (8 GB, line 8), consults `src/main.pl` and `python/helper.pl`. All later instances share that one Prolog.
- `process_metta_string(code) -> list[str]` and `load_metta_file(path) -> list[str]` call
  `run_metta_helper(Verbose, HelperName, Argument, Results)` via `janus.query_once` and return
  `maplist(swrite, …)` strings (python/helper.pl:6-10). Non-`!` forms are `add-atom`ed into the Prolog-global
  space `'&self'` as asserted facts (spaces.pl:2-10) and `(= …)` definitions are compiled to GLOBAL Prolog
  clauses (filereader.pl:160-170 `assertz(Clause)`). Nothing in the Python API creates, selects or clears a
  space; `load_metta_file(path, Results, Space)` takes a space only on the Prolog side (filereader.pl:19-20).
  So two `run_query` calls in one process would ACCUMULATE both KBs and duplicate every rule definition —
  the non-determinism `_guard_extra_atoms` exists to prevent. A fresh space per query therefore means a
  fresh PROCESS per query (or a Prolog-side reset that PeTTa does not expose — unknown).
- Results are a FLAT list across all `!` forms (filereader.pl:100); an expression with no results adds
  nothing; variables print `$_N`; strings keep their quotes; numbers via SWI `write/1`.
- Errors are Prolog exceptions and reach Python as `janus.PrologError` (upstream python/tests/test_imports.py
  `pytest.raises(Exception, match="python import failed")`); a stack overflow is a resource error with the
  text "Stack limit (…) exceeded" (python/tests/test_load_errors.py) — catchable, unlike hyperon's abort.
  Two paths call `halt(1)` and would terminate the HOST process: `test` and `assert` (src/metta.pl:255-265);
  the curated KB uses neither (grep: the three `(assert` hits are comments), but a hand-written `/metta/run`
  program could — the child-process model contains that exactly as it contains hyperon's abort.
- There is no time limit anywhere in PeTTa (grep `time_limit`/`alarm`: none); the CLI runs
  `swipl --stack_limit=8g` (python/petta/cli.py, run.sh).
- Verbose off asserts `silent(true)` (helper.pl:1-2) so the "--> metta sexpr" ANSI banners stay off stdout.
- `load_metta_file_cached` + `PETTA_LOAD_CACHE` replay a load snapshot (python/tests/test_load_cache.py)
  — a possible amortisation of per-process KB compile cost; applicability to concatenated-text loading unknown.

What PeTTa therefore needs from this codebase:
1. Keep process-per-query. The parent must NEVER construct `PeTTa()`: importing petta initialises SWI-Prolog
   (threads, signal handlers) in the parent and forking afterwards is of unknown safety; importing only inside
   the child (as pln_runner.py:767 does now for hyperon) keeps `fork` viable. If the import must move to the
   parent, `_mp_context` has to prefer `spawn`/`forkserver` (executor.py:197-204; `_child_init` 162-167
   already exists for that) at the cost of re-importing `__main__` (app.py builds the Gradio UI at import,
   executor.py:186-188).
2. Per-query cost: each child pays `janus.consult(main.pl)` + compiling ~9,400 lines of .metta
   (`wc -l *.metta` = 9,364; the generic stack is a subset) — unknown, must be measured against hyperon's
   10-600 ms queries. If large, a long-lived PeTTa worker per STACK (KB loaded once, one `!` per request)
   becomes attractive, which reintroduces the shared-pool kill problem executor.py:170-188 measured.
3. Deadline: SIGKILL of the child keeps working. In-process (inline mode) a deadline would need SWI
   `call_with_time_limit/2` wrapped around `run_metta_helper` through a custom `janus.query_once` — PeTTa does
   not expose it, and whether janus releases the GIL while Prolog runs is unknown, so inline mode stays
   "no deadline, no admission" (executor_stats already says so).
4. Crash containment: `janus.PrologError` arrives as `status="error"/runtime_error` (502) through
   pln_runner's existing `except Exception`, which is BETTER than today's 500; the `EOFError` branch
   (executor.py:361-369) stays for `halt/1`, a Prolog C-level crash, or the OOM killer (8 GB stack limit per
   Prolog instance means a runaway can take far more than hyperon's measured 3.7 GB before failing on its
   own — lower `stack_limit` via `set_prolog_flag` after construction, or keep the deadline short).
5. Memory/concurrency bound: `PLN_WORKER_POOL_SIZE` semantics unchanged; each child is a full SWI-Prolog.
6. Per-expression evaluation: after loading `kb_text` (+ `extra_atoms`) with one `process_metta_string`
   (its results are discarded exactly as `metta.run(kb_text)`'s are today, pln_runner.py:800), run each
   normalised `!` expression with its own `process_metta_string` call to get `list[list[str]]` back.

---

## 4. Tests

Files that import hyperon directly (16) and what they need:

| file | hyperon-specific content | change |
|---|---|---|
| tests/test_counterfactual.py:31-32, 67-80 | `importorskip("hyperon")`, `MeTTa()` fixture over `KB_FILES`, `_is_empty(res)` = `res == [[]]`, `_one` = `str(res[0][0])` | shared runtime fixture |
| tests/test_drugage_calibration.py:28-36, 143-170 | same + `os.environ["PLN_RUNTIME_AVAILABLE"]="true"`; `_ranked` runs `rank-interventions` (metta_sort) | fixture |
| tests/test_drugage_chat_routing.py:27,33 | importorskip only; routes through `route_drugage_ranking` | importorskip |
| tests/test_hallmark_targeting.py:250-269 | `MeTTa()` loaded file-by-file in `_INFERENCE_STACK` order; `_run` = `str(atom)` loop | fixture |
| tests/test_human_evidence.py:314-326, 379-430 | fixture; `_BUDGET = 1_100` top-level expressions (391-400, "hyperon 0.2.10 ABORTS … past ~1050"); subprocess canary `predict-risk-patient` on the full runtime KB (404-430) | fixture; budget test obsolete; canary becomes a plain test |
| tests/test_kb_head_symbol_budget.py:61-183 | `importorskip`; `_CHILD` script imports `hyperon.MeTTa` directly (69-86); asserts exit -6 semantics (126), 400 atoms under one head ANSWER (154-167), 256 new heads ABORT (170-183, "makes this file obsolete" if it stops) | whole mechanism section is hyperon-only; 186-236 (run_etl.sh root guard) is runtime-independent and stays |
| tests/test_linage2.py:420-440, 603-690 | fixture; margin probe (+16 heads, subprocess rc); control asserting `returncode == -6 and "panic" in stderr` for a LinAge2 patient's full atoms in the shared space (680-690) | fixture; both probes obsolete or inverted |
| tests/test_nhanes_common.py:683-689, 712-720, 733-775 | atom/byte budget pins (track config); hyperon SEMANTICS probes: unreduced-expression propagation (733-754), `log-math` -> `"-inf"`/`"NaN"` strings and `Low` (757-774) | the two semantics probes are hyperon pins; PeTTa behaviour unknown |
| tests/test_nhanes_integration.py:46-47, 134-166, 311-312, 397-408 | fixture (ETL output + full stack in one `MeTTa()`; note 137-139 on `import!`); `_one_number` = `float(str(res[0]))`; exclusion-of-NHANES-layers guard (397-408, "putting them back aborts the process") | fixture; exclusion guard obsolete |
| tests/test_ontology_expansion.py:520-521, 545-556, 613-618 | fixture; `_empty(res)` | fixture |
| tests/test_patient_grounding.py:29-30, 59-78 | fixture; `_is_empty` | fixture |
| tests/test_pln_deduction.py:24-25, 50-75 | fixture; `_stv`/`_signed` regex on `str(res[0][0])` | fixture |
| tests/test_risk_prediction.py:31-32, 61-76 | fixture | fixture |
| tests/test_smoking_lever.py:247-270, 372-418 | fixture; subprocess probe "+4 definitions answers / +5 SIGABRT" on the full KB (372-418) | fixture; probe obsolete |
| tests/test_supplement_recommendation.py:31-32, 71-86 | fixture | fixture |
| tests/test_gene_queries.py:337-460 | `importorskip` x8; direct `MeTTa()` retune test (378-402, `str(metta.run(...))` of the whole `list[list]`) | importorskip; one fixture |

Files that do NOT import hyperon but depend on its behaviour:

| file | content | change |
|---|---|---|
| tests/test_generic_stack.py (182 lines) | subprocess `_run` over full/generic/routed stacks; `ABORTED` programs must answer (149-151); CONTROL: full stack must still abort (155-158); margin +48 rows (174-182) | control and margin tests are hyperon-only; routing-agreement tests (94-142) survive while the routing exists |
| tests/test_patient_stack.py (534 lines) | same probe; digit pins `(point 0.12605177716424967)` (115), `0.2306046747621094 abs=1e-12` (124), ` 2.0 ` coverage (289, 363); controls `_run("full", q)["rc"] != 0` (136-138, 198); margin +32 heads (150-156); the rest are answer-content assertions that stay | controls/margins obsolete; float-format pins to re-check |
| tests/test_human_evidence_route.py:60-91 | subprocess `_run` via `run_query`; control `control["rc"] != 0` (86-91) | control obsolete |
| tests/test_executor.py:94-218 | real queries in a worker; pins `["0.9"]`/`["0.35"]` (106,138); runaway `(= (spin $n) (spin (+ $n 1)))` and `(slow 14)` shapes; `importorskip("hyperon")` x4 | importorskip; `spin` is a tail call that loops forever in Prolog too (deadline test still meaningful — unknown until run) |
| tests/test_patient_tab.py:650-672 | reuses `test_patient_stack._run` subprocess probe | works if `run_query` works |
| tests/test_caller_patients.py:204, 232 | `importorskip`; `run_query` in-process twice in one test (232-249: baseline vs with-caller on `_runtime_kb_paths()`) | importorskip; TWO KB loads in one process is exactly what PeTTa's global state forbids (§3) |
| tests/test_compound_names.py:188,203; tests/test_drugage_discovery.py:67; tests/test_drugage_ranking_contract.py:30,238; tests/test_prompt_size.py:140; tests/test_runtime_grounding.py:339 | `importorskip("hyperon")` only | importorskip |
| tests/test_error_surfacing.py:143 | `error="hyperon exploded"` string | cosmetic |
| tests/test_api.py:165-167, 487 | `/health` keys; docstring | keys must stay |
| tests/test_ui_api_parity.py:4 | docstring | none |
| tests/conftest.py:20-25 | `slow` marker "runs a real MeTTa query" | none |

Unaffected (mock `run_query`/`run_offloaded`, inline pool): test_api.py (rest), test_api_operations.py,
test_error_surfacing.py, test_combined_app.py, test_linage2_model.py, test_patient_*.py (except the probe
reuse), test_ui_api_parity.py.

In-process multi-load hazard for tests: every `importorskip("hyperon")` module that calls `run_query` more
than once in one process (test_caller_patients.py:232-249, test_drugage_ranking_contract.py:230-262,
test_gene_queries.py, test_drugage_chat_routing.py, test_compound_names.py) assumes a fresh space per call,
which PeTTa's singleton does not give (§3). The module-scoped `MeTTa()` fixtures (one KB per module) are the
PeTTa-compatible shape only if each module runs in its own process (pytest does not do that by default).

---

## 5. Deployment and documentation mentions of hyperon

- pln_chat/requirements.txt:9-11 — `hyperon==0.2.10` with the "verified against this exact version" note; requirements-dev.txt:1 includes it.
- pln_chat/.env.example — `PLN_RUNTIME_AVAILABLE=false` (line 6, "force stub mode"); worker block ("MeTTa runs in a forked process per query because hyperon holds the GIL", ~32-38); `PLN_MAX_KB_FILE_BYTES` note ("hyperon aborts on large ones", ~63-66).
- pln_chat/config.py comments: 53-55, 70-78, 126-132, 179.
- pln_chat/API.md (23 mentions): 20-40 install/run; 129; 183-236 "PLN execution runs in a worker process" (GIL measurements, SIGKILL, inline-mode caveat); 322-323 error table (`runtime_error` = "the hyperon interpreter raised", `pln_worker_crashed`); 397, 511 (patient/LinAge2 head-symbol notes); 606-611 (/drugage/top scored in Python); 796-800 (lever gate gap); 857-859; 904-922, 962 (CellAge scoped space, 100-row cap); 1009-1060 ("A note on KB size" + the key-atom table); 1382; 1454 ("You need `hyperon==0.2.10`, `fastapi`, `httpx`, `pytest` and `pandas`").
- docs/*.md (tracked): abductive_diagnosis.md:3,84-92; calibration_layer.md:180-215 (§10, §10.1 cross-module `&self`); counterfactual_analysis.md:3,212-233; counterfactual_chat_test_queries.md:23-26,54,289; deduction_layer.md:3; drugage_chat_test_queries.md:5,26 (`pip install hyperon==0.2.10 pandas python-dotenv`),207; etl_inference_wiring.md:3,42-46,201-222 (§6 panic table),224-253 (§7 "Hyperon gotchas"),301; intervention_ranking.md:3,80-92,124-135 (§6 full-KB panic); linage2_integration.md:3,170-230 (§5 budget table); nhanes_integration.md:178-190 (`log-math` -inf/NaN),247-252 (unreduced expression),386-400 (§8 two budgets); patient_grounding.md:3; risk_prediction.md:3,121-131; risk_prediction_chat_test_queries.md:4-6,146; supplement_chat_test_queries.md:6,16; supplement_recommendations.md:3,136-141.
- scripts/run_etl.sh:19-24; scripts/*_chat_demo.py and drive_drugage_chat_queries.py docstrings (§1i).
- pln_chat/prompts/system_prompt.txt:2, prompts/few_shot_examples.json:72 ("OpenCog Hyperon").
- No README.md exists at the repo root or under pln_chat/ (`ls README*` matched nothing); no Dockerfile, CI workflow, pyproject or setup file is tracked (`git ls-files` shows only the two requirements files).
- Untracked: HANDOVER_PROMPT.md, REPORT.md, report_2.md, RESEARCH_BRIEF.md, docs/nl_query_examples/.

---

## 6. Minimal adapter interface and LOC estimate

```python
# pln_chat/core/metta_runtime.py (proposed)
from typing import Protocol, Sequence

class MettaRuntimeError(Exception):
    """An interpreter error the runtime could raise and we could catch (hyperon RuntimeError,
    janus.PrologError). Mapped to PLNRunResult(error_code="runtime_error") -> HTTP 502."""

class MettaRuntime(Protocol):
    name: str                                   # "hyperon" | "petta" — for /health, not for `mode`

    @classmethod
    def available(cls) -> bool: ...             # importable AND its system runtime present (swipl)

    def run_program(self, kb_text: str, expressions: Sequence[str]) -> list[list[str]]:
        """Load `kb_text` (concatenated .metta files + caller atoms) into a FRESH space, then evaluate
        each already-`!`-prefixed top-level expression in order and return, per expression, the list of
        result atoms AS STRINGS (an expression with no result -> []). Must be called at most once per
        process for PeTTa (process-global space + compiled clauses), which the executor guarantees.
        Raises MettaRuntimeError; never lets a runtime-specific exception type escape; never aborts the
        process by design (a crash is the executor's EOFError path)."""
```

- hyperon implementation: `m = MeTTa(); m.run(kb_text); return [[str(a) for a in g] for g in m.run("\n".join(expressions))]`
  (grouping is native).
- PeTTa implementation: `p = PeTTa(verbose=False); p.process_metta_string(kb_text); return [p.process_metta_string(e) for e in expressions]`
  wrapped in `try/except Exception -> MettaRuntimeError`; optionally lower `stack_limit` via `janus.query_once`.
- `_hyperon_run` becomes `_runtime_run(runtime, …)` keeping its KB concatenation, logging, threshold and
  `PLNRunResult` assembly; `run_query` chooses the runtime from config (`PLN_RUNTIME=hyperon|petta|auto`).
  `expr_index` keeps working because the adapter returns per-expression groups.
- `PLNRunResult.mode` stays "runtime"/"stub"; `/health` gains `runtime_name`; `runtime_importable` keeps its key.

LOC estimate (lines, from the file reads above):

must_change (breaks on day one, ~600): pln_runner.py `_hyperon_run` + `run_query` dispatch (~95);
config.py detection (~15); api.py health probe + field text (~4); executor.py messages/comments and, if
fork proves unsafe, the context choice (~30); app.py skip message (~3); requirements (~3) = ~150 of app
code; plus ~450 of tests: the 16 direct-`MeTTa` fixtures and `importorskip("hyperon")` lines (~390 incl.
the `[[]]`/`res[0][0]` helpers), the abort-CONTROL assertions that would now fail
(test_generic_stack.py:155-158, test_patient_stack.py:136-138,198, test_kb_head_symbol_budget.py:170-183 and
its hyperon-importing `_CHILD` 69-86, test_linage2.py:680-690, test_human_evidence_route.py:86-91,
test_nhanes_common.py:733-775), and the float/special-value string pins (~10).

may_simplify_away (exists only for hyperon's bug/GIL and could be deleted if PeTTa has no space-size bug,
~1,500): pln_runner.py stack machinery and rationale 64-245, 247-335 (scoping), 338-345, 674-723 (~430 incl.
comments); api.py `_generic_kb`, `_run_linage2_program` and friends, routing branches, stack comments
(~190); app.py `_generic_kb`, `_linage2_context`, LinAge2 branch, comments (~110); linage2_router.py
split/route (~90); patient_context.py detectors (~30); patient_builder.py `shared_atoms` (~25);
PLN_MAX_KB_FILE_BYTES and users (~50); selector cap rationale (~30); lever_gate.py (~100, a move into
MeTTa rather than a deletion) = ~1,050 app lines (upper bound: ~800 if the routing comments are not
counted) and ~700 test lines (test_generic_stack.py 182, test_kb_head_symbol_budget.py 1-183,
test_human_evidence_route.py 91, the probes in test_patient_stack.py, test_linage2.py 603-690,
test_smoking_lever.py 372-418, test_human_evidence.py 379-430, test_nhanes_integration.py 397-408).
The ~450 must-change test lines overlap this set by ~100 (the controls and `_CHILD`).

Not counted either way (stay, re-checked): metta_validator `_BUILTINS`, linage2_router `_STRUCTURAL_HEADS`,
registry builtins; `drugage_top`/scoring, gene_index, hallmarks, human_evidence; the DrugAge/CellAge slices
and caps; `_guard_extra_atoms`; API.md and docs (~120 lines of prose naming hyperon).

---

## 7. Unknowns that only running something can settle

1. Per-process cost of `janus.consult(main.pl)` + compiling a ~25-file stack (vs hyperon's 10-600 ms queries).
2. Fork safety after `import petta` (SWI-Prolog initialised in the parent) — avoided entirely if the import stays child-only.
3. Whether janus releases the GIL during a Prolog call (decides whether inline mode could ever get a deadline).
4. PeTTa's rendering of floats (`1.0e-5` vs `1e-05`), integer promotion (`3` vs `3.0`), `log-math 10 0`
   (`-inf` string vs evaluation error), `nan`; the digit-level pins in tests/test_patient_stack.py.
5. Whether a known function with no matching clause returns an unreduced expression (hyperon) or nothing
   (Prolog failure) — pln_risk_prediction.metta:232-234 / nhanes_baseline.metta:83-85 depend on the former.
6. Whether PeTTa's dialect accepts the KB at all (54 `collapse`, 46 `superpose`, 37 `car-atom`, 26 `cdr-atom`,
   15 `cons-atom`, 7 `log-math`, 3 `pow-math`, 4 `foldl-atom`, 3 `unique-atom`, 5 `chain`, 14 `(empty)`,
   72 `let*`, 93 `match`, `(Error …)` once) — a dialect question outside this map, listed because the
   validator's builtin lists mirror it.
7. Whether `sexpdata` is needed at all: `linage2_router.parse_sexpr` already parses result strings.
