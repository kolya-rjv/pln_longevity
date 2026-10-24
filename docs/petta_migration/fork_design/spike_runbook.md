# Spike runbook: fork PLN, bundled lib_pln and trueagi-io PLN on PeTTa fea83178

Companion of `petta_compat.md` (read its sections 1-4 first). Written 2026-10-24 for a later session; **nothing here has been run**.

**Authorisation.** Every step below except section 0's `cmp`/`shasum`/`git status` executes upstream code (`swipl` with PeTTa, `run.sh`, the Python
`PeTTa` class, `test.sh`, and in R1 the fork's Translator under hyperon). Run them only after the owner has granted that. Nothing is installed; nothing
is written under `.petta/` (all runs use a scratch copy of PeTTa). Steps are independent after G0 unless noted: a failure in one is recorded and the next
step still runs. Commands are for `bash` or `zsh`; no word starts with an equals sign; globs are quoted.

**Order of execution:** section 0, G0, section 1 (probes), then (ii) in section 2 (CI-tested upstream code: it validates the runtime and the harness before anything of the fork is blamed), then (i) in section 3, then (iii) in section 4. Every example is run twice: with `run.sh` and through the Python `PeTTa` class (one process per file).

## 0. Setup (static, safe)

```sh
SCR=/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/forkdesign
S=$SCR/spike_run
mkdir -p "$S/out" "$S/probes" "$S/lib" "$S/cache"
cat > "$S/env.sh" <<'EOF'
REPO=/Users/kolya/PycharmProjects/pln_longevity_petta
PINNED=$REPO/.petta/PeTTa                  # read only
FORK=$PINNED/repos/PLN                     # read only: patham9/PLN 55f1751
NEWPLN=$REPO/.petta/PLN-trueagi            # read only: trueagi-io/PLN 4405956
PY=/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python
SCR=/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/forkdesign
ART=$SCR/compat/artifacts
S=$SCR/spike_run
P=$S/petta                                 # scratch copy of PeTTa: every run uses this one
export PETTA=$P                            # read by artifacts/py_run.py
export PETTA_STACK_LIMIT=2g                # run.sh default is 8g
export PETTA_LOAD_CACHE=$S/cache
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }          # macOS has no coreutils timeout
pt()  { tmo 600 sh "$P/run.sh" "$@" silent; }              # pt a.metta [b.metta ...]  ('silent' must come after the files)
ALL="FlyingRaven Robot Smokes Toothbrush RuleTester equivalenceToImplication evaluationImplicationRuleA evaluationWithNegationAndInheritanceInversion inversion memberDeductionA transitiveSimilarity"
EOF
. "$S/env.sh"
cp -R "$PINNED" "$P" && rm -rf "$P/.git" "$P/repos"        # 6 MB
diff -r -x .git -x repos "$PINNED" "$P" && echo "scratch PeTTa identical to the pin"
for d in "$PINNED" "$FORK" "$NEWPLN"; do echo "$d $(git -C "$d" rev-parse HEAD) dirty=$(git -C "$d" status --porcelain | wc -l)"; done
cp "$ART"/probes/* "$S/probes/"
cp "$ART/fork_PLN_A0.metta" "$S/lib/fork_A0.metta";   cp "$ART/fork_PLN_A0S.metta" "$S/lib/fork_A0S.metta"
cp "$ART/fork_PLN_A1.metta" "$S/lib/fork_A1.metta";   cp "$ART/fork_PLN_A1B.metta" "$S/lib/fork_A1B.metta"
cp "$ART/bundled_lib_pln_B1.metta" "$S/lib/bundled_B1.metta"; cp "$ART/Translator_petta_R2.metta" "$S/lib/Translator_R2.metta"
cmp "$ART/bundled_lib_pln_B0.metta" "$PINNED/lib/lib_pln.metta" && echo "B0 == pinned lib_pln.metta"
cmp "$ART/fork_PLN_A0.metta" <(cd "$FORK/src" && cat Utils.metta Constraints.metta Formulas.metta Rules.metta Config.metta Deriver.metta) && echo "A0 == fork build.sh output"
```

Expected: the three SHAs are `fea83178af37...`, `55f1751d993f...`, `4405956947c4...` with `dirty=0`; both `cmp`/`diff` lines print their OK text.
Every later block starts with `. "$S/env.sh"`.

## G0. Runtime gate (stop here if it fails)

```sh
. "$S/env.sh"; cd "$S"
command -v swipl; swipl --version                            # PeTTa documents >= 9.3; Homebrew has 10.0.2
"$PY" -I -c "import janus_swi; print('janus_swi import ok')"
pt "$S/probes/hello.metta"; echo "exit=$?"                   # expect: 3 , exit=0
"$PY" -I "$ART/py_run.py" "$S/probes/hello.metta"; echo "exit=$?"   # expect: 3 , exit=0
for i in 1 2 3; do /usr/bin/time -l sh "$P/run.sh" "$S/probes/hello.metta" silent 2>&1 | grep -E 'real|maximum resident'; done   # startup cost and RSS
```

| Observed | Meaning |
|---|---|
| `swipl: command not found` | SWI-Prolog missing; the owner's authorised `brew install swi-prolog` did not happen |
| `janus_swi` import error mentioning a dylib | macOS loader path (`DYLD_LIBRARY_PATH` / libswipl); the CLI route may still work, the Python route does not (record, continue with `run.sh` only) |
| `run.sh` prints errors while loading `src/*.pl` | SWI 10.0.2 versus the 9.3.x PeTTa targets, or a missing package (pcre, clpfd): record the first error line; everything after is moot |
| `3` printed but exit != 0 | record the exit code and stderr; retry with `PETTA_STACK_LIMIT=8g` (the default of `run.sh`) to separate a stack-limit problem from an exit-status problem |

## 1. Construct probes (confirm or refute `petta_compat.md` section 3)

```sh
. "$S/env.sh"; cd "$S"
for p in probe_core probe_runtime_eq; do
  pt "$S/probes/$p.metta" > "out/$p.out" 2> "out/$p.err"; echo "$p exit=$?"
  diff "out/$p.out" "probes/$p.predicted" && echo "$p: matches the static prediction"
done
# if a probe aborts (exit != 0 and no output), run its forms one by one:
n=0; grep '^!' "$S/probes/probe_core.metta" | while IFS= read -r line; do n=$((n+1)); printf '%s\n' "$line" > "out/one_$n.metta"; printf '[%s] %s  =>  ' "$n" "$line"; pt "out/one_$n.metta" 2>&1 | tr '\n' ' '; echo; done
pt "$S/probes/probe_predicate.metta"; echo "exit=$?"
pt "$S/probes/probe_forward_a.metta"  "$S/probes/probe_forward_b.metta"  2> out/fwd_ab.err;  cat out/fwd_ab.err | head -3    # expect (STV k) and a stderr warning 'STV is defined or imported after already being used'
pt "$S/probes/probe_forward_a2.metta" "$S/probes/probe_forward_b.metta"                                                 # expect (stv 1 1)
pt "$S/probes/probe_div0.metta"; echo "exit=$?"                                                                         # expect non-zero exit and 'zero_divisor' in the error text
```

Predictions are in `artifacts/probes/*.predicted` (static reading). What a mismatch means:

| Probe | Prediction | If it differs |
|---|---|---|
| `(- 5)` | `(partial - (5))` | unary minus works: drop edit E1 from the patch set |
| `(trace! (+ 1 2) done)` | prints `3` on stdout, result `done` | trace! still prints its argument unevaluated: the Translator may work as is, redo section 3.1 of the notes |
| `(trace! (= (probe-x) 1) ok)` | prints `false` | same |
| `add-atom` then call | `42`; the quoted `(= ...)` is not installed (`(probe-quoted)` stays data) | runtime installation behaves differently: re-read spaces.pl:26-73 |
| `(Predicate friend)` before the retract | no output | `Predicate` is not a builtin at this pin: edit E7 unnecessary |
| `(Predicate friend)` after the retract | `(Predicate friend)` | the retract trick does not work: rename the functor in KB and rules instead (design agents) |
| forward-declaration pair | `(STV k)` without, `(stv 1 1)` with | definition order is not an issue in this PeTTa: edit E2 optional |
| `(/ 1 0)` | exception text with `zero_divisor`, exit != 0 | division is guarded: lower risk for node strengths of 0 |

## 2. (ii) The bundled lib's own PeTTa examples and `test.sh`

Pins (from `.petta/PeTTa/examples`), each printed by PeTTa's `test` as `is X, should Y.` followed by a tick mark, then the result lines `true`:

| File | Pin |
|---|---|
| `pln_roman.metta:15-16` | `((stv 0.5 0.9473684210526316) (1 2 3 4))` |
| `pln_tuffy.metta:47-50` | `((stv 0.5199204545454544 0.8290782204129108) (1 2 4 9 10))` |
| `pln_direct.metta:26-31` | `(((grandfather a c) (stv 0.51 0.8432620011567381)))` twice (the second query uses `$who`), and `(((old a) (stv 0.5198 0.7923434575206812)))` |

The bundled lib also prints one `(SELECTED n (Sentence ...))` line per Derive step on stdout (`check_pins.py` ignores them).

```sh
. "$S/env.sh"; cd "$S"
for f in pln_roman pln_tuffy pln_direct; do
  echo "== $f"; /usr/bin/time -l sh "$P/run.sh" "$P/examples/$f.metta" silent > "out/B_$f.out" 2> "out/B_$f.err"; echo "exit=$?"
  grep -E 'real|maximum resident' "out/B_$f.err"
  python3 -I "$ART/check_pins.py" < "out/B_$f.out"
done
# PeTTa's own pass rule: source the run_test function of test.sh and apply it to these three (cwd must be the PeTTa root)
( cd "$P" && sed -n '/^mode_arg_for_test()/,/^}/p;/^run_test()/,/^}/p' test.sh > "$S/out/run_test_fns.sh" && . "$S/out/run_test_fns.sh" \
  && for f in pln_roman pln_tuffy pln_direct; do run_test "./examples/$f.metta" > "$S/out/B_${f}_runtest.out" 2>&1; echo "$f run_test rc=$?"; done )
# same files through the Python interface, one process each (test calls halt(1) on a mismatch)
for f in pln_roman pln_tuffy pln_direct; do
  "$PY" -I "$ART/py_run.py" "$P/examples/$f.metta" > "out/BP_$f.out" 2> "out/BP_$f.err"; echo "$f python exit=$?"; python3 -I "$ART/check_pins.py" < "out/BP_$f.out" | tail -1
done
```

Expected: exit 0, every pin line EXACT, `pin lines seen: 1, failed: 0` (pln_direct: 3 lines). The Python route also returns the list `['true', ...]`, printed one per line.

Optional, heavy (several hundred examples, 8 background jobs by default, writes to `/tmp` through `mktemp -d /tmp/petta_test_XXXX`, and `examples/git_import.metta` needs the network):

```sh
. "$S/env.sh"; cd "$P"
PETTA_TEST_JOBS=2 PETTA_STACK_LIMIT=3g SKIP_SOUNDNESS=1 sh test.sh > "$S/out/petta_test_sh.out" 2>&1; echo "exit=$?"
grep -c '^OK' "$S/out/petta_test_sh.out"; grep -E 'FAILURE|Failed tests' "$S/out/petta_test_sh.out" | head
```
Expected offline: only the git-import examples fail. Any `pln_*` failure here would contradict upstream's CI.

| Observed | Meaning |
|---|---|
| a pin line with a cross mark and exit 1 | the bundled lib's numbers differ on this PeTTa/SWI build: compare the two numbers digit by digit before blaming the lib (SWI 10 versus 9.3 float printing) |
| `Type error at .../lib_pln.metta:LINE` | PeTTa's checker rejects the typed lib on this build: record LINE, the declaration is the cause |
| exit 0 but no pin line | `test` was never reached: a form failed silently (an `(empty)` in the KB build) |

## 3. (i) The fork's own examples and ruletests

The fork's own runners (`test.sh`, `examples/test.sh`, `ruletests/test.sh`) call the hyperon `metta` CLI and `examples/PLN.py`
(needs `~/PLN`, `~/metta-morph`); they cannot run here and are replaced by the converter and checker. Pins are the `!(Test ... expected)` lines:

| File | Pin(s) |
|---|---|
| examples/FlyingRaven | `((stv 0.97052375 0.81) (1 4 6))` ; `((stv 0.332225678571428 0.83690036900369) (2 3 5 6))` |
| examples/Robot | `((detection frisbee someCoords1) (detection orange someCoords4))` |
| examples/Smokes | `((stv 0.519920454545454 0.829078220412911) (4 9 10))` |
| examples/Toothbrush | `((stv 0.905100467696027 0.895338254258716) (10 11 12 13 14))` |
| ruletests/RuleTester | `((stv 0.01 0.9) (5))` ; `((stv 0.99 0.54) (1))` ; `((stv 0.994974874371859 0.9) (2))` |
| ruletests/equivalenceToImplication | `((stv 0.98989898989899 0.87) (1))` |
| ruletests/evaluationImplicationRuleA | `((stv 1.0 0.729) (1 2 3))` |
| ruletests/evaluationWithNegationAndInheritanceInversion | `((stv 0.8 0.6) (1 2))` |
| ruletests/inversion | `((stv 0.87 0.486) (1))` |
| ruletests/memberDeductionA | `((stv 0.75 0.5) (1 2))` |
| ruletests/transitiveSimilarity | `((stv 1.0 0.5) (1 2))` |

The fork's `Test` prints `((Is: X) (Should: Y) (Passed: true|false))`; the 15-digit pins make `Passed: false` on a correct run, so `check_pins.py` (relative tolerance 1e-12) is the judge: EXACT, NUMERIC (pass), FAIL.

### F0. The 11 files unmodified (expect all to fail at line 1)

```sh
. "$S/env.sh"; cd "$S"
mkdir -p fork_orig && cp -R "$FORK/examples" "$FORK/ruletests" fork_orig/
for f in examples/FlyingRaven examples/Robot examples/Smokes examples/Toothbrush ruletests/RuleTester ruletests/equivalenceToImplication ruletests/evaluationImplicationRuleA ruletests/evaluationWithNegationAndInheritanceInversion ruletests/inversion ruletests/memberDeductionA ruletests/transitiveSimilarity; do
  b=$(basename "$f"); pt "$S/fork_orig/$f.metta" > "out/F0_$b.out" 2> "out/F0_$b.err"; echo "$b exit=$? source_sink_lines=$(grep -c source_sink "out/F0_$b.err")"
done
```
Expected: exit != 0 and `source_sink_lines=1` for all 11 (`existence_error(source_sink, 'PLN')`).
If one runs further: the import resolved to something unexpected; read its `.err` for the first error and stop that file.

### F1. The library files alone

```sh
. "$S/env.sh"; cd "$S"
for L in fork_A0 fork_A0S fork_A1 fork_A1B bundled_B1; do
  printf '!(import! &self "%s/lib/%s")\n!(+ 1 2)\n' "$S" "$L" > "out/load_$L.metta"
  echo "== $L"; ( time pt "$S/out/load_$L.metta" ) 2>&1 | tail -6
done
```
Expected: A1, A1B and bundled_B1 print `true` and `3` (import result, then the sum). **A0 is the open question**: either it loads (then rules reading `(STV $A)` are dead) or it stops with
`Type error at ...Rules...: Type mismatch: got (STV $A) but expected (stv Number Number)` (or similar). Record the first error line verbatim.
A0S (A0 plus only the STV declaration) should print `true`, `3`.

### F2. The queue-overflow defect (unary minus) and the helper semantics

```sh
. "$S/env.sh"; cd "$S"
for L in fork_A0 fork_A0S fork_A1 bundled_B1; do
  sed "s#@LIB@#$S/lib/$L#" probes/probe_queue.metta > "out/q_$L.metta"
  pt "$S/out/q_$L.metta" > "out/q_$L.out" 2>&1; echo "== $L exit=$?"; tail -4 "out/q_$L.out"
done
for L in fork_A1 bundled_B1; do
  sed "s#@LIB@#$S/lib/$L#" probes/probe_unique.metta > "out/u_$L.metta"
  case $L in fork_A1) pred=probes/probe_unique.predicted_A1;; *) pred=probes/probe_unique.predicted_B1;; esac
  pt "$S/out/u_$L.metta" | tail -6 > "out/u_$L.out"                     # 7 result lines: the import's true, then the 6 probes
  diff "out/u_$L.out" "$pred" && echo "$L: helper semantics as predicted"
done
```
Expected: A0S exits non-zero with an arithmetic type error naming `partial` (`evaluable`/`partial/2`); A1 and bundled_B1 exit 0 without an exception (the value returned may be empty because the queue is tiny: only the absence of an exception matters).
`Unique (3 1 2 1 3) ()` is `(2 1 3)` with the fork's helpers and `(3 1 2)` with the bundled ones.

### F3. Generic route (no Translator): the converted files with the patched fork lib

```sh
. "$S/env.sh"; cd "$S"
mkdir -p fork_gen
for b in $ALL; do for L in fork_A1 fork_A1B; do sed "s#@LIB@#$S/lib/$L#" "$ART/fork_examples_petta/$b.metta" > "fork_gen/${b}__$L.metta"; done; done
for b in $ALL; do
  case $b in Smokes|Toothbrush) opts=--stv-only;; *) opts=;; esac           # ids differ by design when implications are beliefs
  ( time pt "$S/fork_gen/${b}__fork_A1.metta" > "out/G_$b.out" 2> "out/G_$b.err" ) 2> "out/G_$b.time"; rc=$?
  echo "== $b exit=$rc $(grep real "out/G_$b.time")"; python3 -I "$ART/check_pins.py" $opts < "out/G_$b.out" | tail -3
done
# bigger budget for the two files with static implications (fork default is 20 steps / 20 tasks / 200 beliefs)
for b in Smokes Toothbrush; do pt "$S/fork_gen/${b}__fork_A1B.metta" 2>/dev/null | python3 -I "$ART/check_pins.py" --stv-only | tail -3; done
```
Expected (predicted): 10 of the 11 files pass numerically: the 7 ruletests and FlyingRaven (FlyingRaven and the two `Predicate` ruletests only if the retract works), Robot, and Smokes with `--stv-only` (maybe only with the larger budget); **Toothbrush FAILs** on strength `0.9013534742103302` against the pin `0.905100467696027` (generic route; this is the computed difference, not a bug).

| Observed | Meaning |
|---|---|
| a `Predicate` file prints no pin line at all | the KB-building function failed: the retract did not take effect (probe 1) or came too late (same unit as the code that needs it) |
| `Passed: false` but NUMERIC | the 15-digit pin; accept |
| FAIL with `is=()` | the query term was not derived: budget too small (use `fork_A1B`), a rule is dead (STV missing), or an atom-headed conclusion was called as a function |
| FAIL with a close but different number | formula or order difference: compare with `work/numcheck*.py` before suspecting PeTTa |
| Python and `run.sh` outputs differ for the same file | a stdout/stderr or state issue in the Python embedding: diff `out/G_*.out` against `out/GP_*.out` |

### F4. Translator route R2 (the port with `add-atom`; untested design)

```sh
. "$S/env.sh"; cd "$S"
mkdir -p fork_r2
for b in Smokes Toothbrush Robot; do
  sed "s#@LIB@#$S/lib/fork_A1#; s#@TRANS@#$S/lib/Translator_R2#" "$ART/fork_examples_petta_r2/$b.metta" > "fork_r2/$b.metta"
  ( time pt "$S/fork_r2/$b.metta" > "out/R2_$b.out" 2> "out/R2_$b.err" ) 2>&1 | grep real
  head -3 "out/R2_$b.out"; python3 -I "$ART/check_pins.py" < "out/R2_$b.out" | tail -3
done
```
Expected if R2 works: the `PLN.Init` result lists the messages `PLN.Init: Translated ...` (Smokes: 6 concept and 2 implication lines; Toothbrush: 12 and 6; Robot: 7 and 1), and the pins pass with the fork's own stamps:
Smokes `(4 9 10)`, Toothbrush strength `0.905100467696027` with stamps `(10 11 12 13 14)`.

| Observed | Meaning |
|---|---|
| `PLN.Init` prints `()` or nothing, queries return nothing | the `add-atom` payload was evaluated or the `case` branches failed: probe the equation count with `!(collapse (match &self (= (\|- $a $b) $c) $c))` |
| an exception from `add-atom` / `translate_clause` | runtime installation of translator output is not viable as written; take R1 |
| Toothbrush differs from the pin but equals `0.9013534742103302` | the nested-implication rule was not installed and generic modus ponens ran instead |
| load time grows super-linearly with the number of implications | each `add-atom` recompiles dependents (spaces.pl:60-73): measure with 10, 100, 1000 generated implications before choosing R2 |

### F5. Translator route R1 (the fork's own Translator under hyperon, optional; executes fork code in hyperon)

```sh
. "$S/env.sh"; cd "$S"
for b in Smokes Toothbrush Robot; do
  "$PY" -I "$ART/hyperon_translate.py" "$FORK/src/Translator.metta" "$FORK/examples/$b.metta" > "out/R1_$b.hyp" 2> "out/R1_$b.raw"
  grep '^(= ' "out/R1_$b.raw" > "lib/translated_$b.metta"; echo "$b equations: $(wc -l < "lib/translated_$b.metta")"   # expect 8 / 18 / 8
  python3 -I "$ART/to_petta.py" "$FORK/examples/$b.metta" "fork_r2/${b}_R1.metta" "$S/lib/fork_A1" --r2 "$S/lib/translated_$b"
  pt "$S/fork_r2/${b}_R1.metta" | python3 -I "$ART/check_pins.py" | tail -3
done
```
Expected: the same pins as F4. The raw file holds hyperon's printed equations with variables like `$TV1#123`. If hyperon prints nothing on stderr, `trace!` goes elsewhere in 0.2.10: capture stdout instead.
If R1 passes and R2 fails, R2's port is wrong; if both fail, the pin comes from something other than the Translator.

### F6. The same generated files through the Python interface

```sh
. "$S/env.sh"; cd "$S"
for b in $ALL; do
  "$PY" -I "$ART/py_run.py" "$S/fork_gen/${b}__fork_A1.metta" > "out/GP_$b.out" 2> "out/GP_$b.err"; echo "$b python exit=$?"
  case $b in Smokes|Toothbrush) opts=--stv-only;; *) opts=;; esac
  python3 -I "$ART/check_pins.py" $opts < "out/GP_$b.out" | tail -1
done
```
Same verdicts as F3 expected.

### F7. Stage B: the recommended artefact on the same files

```sh
. "$S/env.sh"; cd "$S"
for b in $ALL; do
  sed "s#@LIB@#$S/lib/bundled_B1#; s#!(Test #!(Test2 #" "$ART/fork_examples_petta/$b.metta" > "fork_gen/${b}__bundled_B1.metta"
  case $b in Smokes|Toothbrush) opts=--stv-only;; *) opts=;; esac
  ( time pt "$S/fork_gen/${b}__bundled_B1.metta" > "out/H_$b.out" 2> "out/H_$b.err" ) 2>&1 | grep real
  python3 -I "$ART/check_pins.py" $opts < "out/H_$b.out" | tail -1
done
```
(The bundled lib names the harness function `Test2`, hence the second substitution.) Expected: the same verdicts as F3 (bundled defaults are 100/10/100, the fork's 20/20/200; a file whose result depends on the budget may flip). Compare the `real` times with F3: the bundled helpers are builtins, the fork's are `collapse`-based. A different numeric result between F3 and F7 for the same query is the effect of `Unique` order or of the budget; list those cases for the owner.

## 4. (iii) The newer trueagi-io/PLN examples and ruletests

`git-import!` clones into `./repos` of the **current directory** and skips the clone when `repos/PLN` exists, and `.petta/PeTTa/repos/PLN` is the *fork*, so the newer repo is placed in its own scratch root:

```sh
. "$S/env.sh"
N=$S/new; rm -rf "$N"; mkdir -p "$N/repos"
cp -R "$NEWPLN" "$N/repos/PLN" && rm -rf "$N/repos/PLN/.git"
printf '#!/bin/sh\nexec sh "%s/run.sh" "$@"\n' "$P" > "$N/run.sh"     # the repo's test.sh does: cd ../../ ; sh run.sh FILE
cd "$N"
for f in repos/PLN/examples/*.metta repos/PLN/ruletests/*.metta; do
  b=$(basename "$f" .metta)
  ( time sh "$N/run.sh" "./$f" silent > "$S/out/N_$b.out" 2> "$S/out/N_$b.err" ) 2> "$S/out/N_$b.time"; rc=$?
  echo "$b exit=$rc $(grep real "$S/out/N_$b.time")"
  python3 -I "$ART/check_pins.py" < "$S/out/N_$b.out" | tail -2
done
```
The files run `git-import!` (no network because the directory exists), then `(import! &self (library PLN lib_pln))`. Their `test` pins (printed as `is X, should Y.`):

| File | Pin |
|---|---|
| examples/DeductionRevision | `((stv 0.125 0.30429942952553224) (1 2 3 4))` |
| examples/Direct | `(((grandfather a c) (stv 1.0 0.7290000000000001)))` twice, `(((old a) (stv 1.0 0.6561000000000001)))` |
| examples/FlyingRaven | `((stv 0.9703557256235827 0.6932852196717858) (1 4 6))` ; `((stv 0.009800000000000031 0.8019000000000001) (2 3))` |
| examples/RavenInduction | `((stv 0.38000000000000006 0.6541049798115747) (1 2 3 4))` (imports `(library lib_pln)`, the *bundled* lib, not trueagi's) |
| examples/Robot | `((detection frisbee someCoords1) (detection orange someCoords4))` |
| examples/Smokes | `((stv 0.6 0.486) (2 10))` |
| examples/Toothbrush | `((stv 0.9013534742103302 0.4776362988903337) (2 3 4 5 6 7 10 11 12 13 14))` |
| ruletests/RuleTester | `((stv 0.010000000000000009 0.9) (5))` ; `((stv 0.99 0.54) (1))` ; `((stv 0.9949748743718593 0.9) (2))` |
| ruletests/equivalenceToImplication | `((stv 0.98989898989899 0.87) (1))` |
| ruletests/evaluationImplicationRuleA | `((stv 1.0 0.7290000000000001) (1 2 3))` (starts with the `Predicate` retract) |
| ruletests/evaluationWithNegationAndInheritanceInversion | `((stv 0.8 0.38880000000000003) (1 2))` (same retract) |
| ruletests/inversion | `((stv 0.87 0.486) (1))` |
| ruletests/memberDeductionA | `((stv 0.75 0.375) (1 2))` |
| ruletests/transitiveSimilarity | `((stv 1.0 0.445) (1 2))` |

Their own script, exactly as designed (it starts all 14 files at once, so keep the stack limit low; it exits at the first failure):

```sh
. "$S/env.sh"; cd "$S/new/repos/PLN" && PETTA_STACK_LIMIT=1g sh test.sh > "$S/out/N_test_sh.out" 2>&1; echo "exit=$?"; tail -15 "$S/out/N_test_sh.out"
```

Through the Python interface (cwd matters: `./repos` is relative to the process cwd):

```sh
. "$S/env.sh"; cd "$S/new"
for b in DeductionRevision Direct Smokes Toothbrush; do
  "$PY" -I "$ART/py_run.py" "$S/new/repos/PLN/examples/$b.metta" > "$S/out/NP_$b.out" 2> "$S/out/NP_$b.err"; echo "$b python exit=$?"
  python3 -I "$ART/check_pins.py" < "$S/out/NP_$b.out" | tail -1
done
```

| Observed | Meaning |
|---|---|
| `existence_error(source_sink, ...lib_pln)` | `git-import!` resolved to the wrong `repos/PLN` (the fork): cwd was not `$S/new` |
| network messages from `git` | the clone was attempted: the directory test failed; stop and fix the layout |
| a pin differs in the last digits only | float printing; look at whether check_pins says NUMERIC |
| `Predicate` ruletests fail but others pass | the retract trick does not survive this PeTTa pin (the lib dates from March 2026, the PeTTa pin from October): relevant to every design that keeps `Predicate` terms |
| Smokes/Toothbrush pass with strengths 0.6 and 0.9013534742103302 | this is the generic route of the newer formulas; compare with F3's Toothbrush strength (equal) and with F4/F5 (the fork's translator route differs) |

## 5. Closing checks

```sh
. "$S/env.sh"
for d in "$PINNED" "$FORK" "$NEWPLN"; do echo "$d dirty=$(git -C "$d" status --porcelain | wc -l)"; done     # must stay 0
shasum -a 256 "$PINNED/lib/lib_pln.metta"      # d864c3cf376ad1f9252d153306d92b7aae72a184701c2eb4805c9661eed27ece
```
Record in a table, per step: command, exit code, first error line, verdict (EXACT / NUMERIC / FAIL / ERROR), `real` seconds, RSS. The decisions the results feed: (1) does the `Predicate` retract work (keep or rename the functor), (2) is A0 viable (it should not be), (3) Translator route R1/R2/none, (4) fork helpers versus builtin helpers, (5) whether any bundled or newer example fails at this PeTTa pin.
