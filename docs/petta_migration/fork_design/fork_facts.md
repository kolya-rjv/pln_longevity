# patham9/PLN @ 55f1751 ("the fork"): how it works, from the source

Written 2026-10-24. Everything here comes from reading the files and git history of the pinned checkouts, plus a pure-Python
port that was validated against every pinned expectation (section H). Nothing from `.petta/` was executed, imported or copied.

Paths: `FORK/` = `/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa/repos/PLN` (detached at 55f1751 = `main` = origin/main,
2025-08-15, "Merge pull request #54 from trueagi-io/Negation"; 144 commits; remote `https://github.com/patham9/PLN.git`).
`BUNDLED/` = `.petta/PeTTa/lib/lib_pln.metta` (PeTTa fea83178). `TRUEAGI/` = `.petta/PLN-trueagi/lib_pln.metta` (4405956, 2026-03-15).
`SIM/` = `/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity/536046cf-1647-43f4-b3c4-093bc03cc2c3/scratchpad/forkdesign/sim`.

Evidence tags: **[S]** source read (file:line), **[P]** probe I wrote and ran on hyperon 0.2.10 (own snippets only, scripts in `facts/probes/`),
**[W]** read-only web fetch of trueagi-io/metta-morph (summarised by the fetch tool, so indicative, not verbatim), **[I]** inferred,
**[X]** exploration with the simulator. "unknown" means I could not establish it.

Files read completely: `FORK/{README.md,build.sh,test.sh,LICENSE}`, `src/{Config,Constraints,Deriver,Formulas,Rules,Translator,Utils}.metta`,
`examples/{Smokes,FlyingRaven,Robot,Toothbrush}.metta`, `examples/{PLN.py,test.sh}`, `ruletests/*` (7 `.metta`, `test.sh`), the whole `git log --stat`
(144 commits on main, 165 on all refs), `git log -S` for `PLN.Init`, `STV`, `call_plninit`, `2>> TRANSLATED.metta`, `(- $c)`, `PriorityRankNeg`;
`BUNDLED`, `TRUEAGI`, PeTTa `examples/{pln_direct,pln_roman,pln_tuffy,plntest,plntestdirect}.metta`, the trueagi wiki page in the earlier scratchpad.

---------------------------------------------------------------------------------------------------------------------------------------

## A. Sentence formats, PLN.Init, and the data flow of Smokes.metta

### A1. The two spellings

| Spelling | Where | Role |
|---|---|---|
| `(sentence ((term) (stv s c)) (id ...))` lowercase | user-facing KB format: `kbstatic`/`kbdynamic` in every example (`Smokes.metta:4-21,27-56`, `FlyingRaven.metta:4-8,14-36`, `Robot.metta:6-18,25-32,37-38`, `Toothbrush.metta:5-63,69-80`, `ruletests/*` Init lists and most query lists); also the head of the two **Translator** equations (`Translator.metta:11,38`) | input format. At translate time it is a *function* (the Translator) that emits equations; at query time it is an *alias* for `Sentence` |
| `(Sentence ((term) (stv s c)) (id ...))` capitalised | the only form the compiled reasoner knows: `Deriver.metta:19,23` (PriorityRank/Neg patterns), `:37,39,44,45,46,47` (PLN.Derive), `:71` (PLN.Query); written directly in `ruletests/{evaluationWithNegationAndInheritanceInversion:10,12; inversion:13; memberDeductionA:16,20,23}.metta` | internal runtime format |
| bridge | `examples/PLN.py:51`: `globalmetta.run("(= (sentence $A $B) (Sentence $A $B))")`, executed at the end of `PLN.Init` **[S]** | after `PLN.Init`, evaluating a list of lowercase `sentence` atoms in the host MeTTa turns them into `Sentence` atoms before they reach `PLN.Query` |

`PriorityRank`, `Derive` and `Query` pattern-match on the literal head `Sentence` and on `(stv $f $c)`, so a sentence with any other TV shape is invisible to them.
The id tuple `(...)` is the *stamp* (evidence ids). Static sentences carry `(0)` or an arbitrary id; **those ids never reach a derived stamp** (see A4).

### A2. Where PLN.Init is defined

Not in `src/`, not in PeTTa, not in the trueagi rewrite: `grep -rIn "PLN.Init\|plninit"` over `.petta/PeTTa` and `.petta/PLN-trueagi` finds nothing **[S]**.
It is a **Python grounded operation** in `FORK/examples/PLN.py`: `call_plninit` (`PLN.py:25-54`), registered at `PLN.py:57-63`
(`@register_atoms(pass_metta=True)` returning `{r"PLN.Init": PatternOperation('PLN.Init', wrapnpop(call_plninit), unwrap=False)}`).
`git log -S"call_plninit"` -> only `3d55f3f` (2025-07-20 "working Mettamorph for PLN"). `!(import! &self PLN)` (`Smokes.metta:1`) loads that file from the module path;
`ruletests/test.sh:2` copies `../examples/PLN.py` into the cwd for exactly this reason. Hard-coded paths: `PLN.py:27` `~/PLN`, `:28` `~/metta-morph`;
`PLN.py:34` uses the relative `../PLN.metta`, so the cwd must be a sub-directory of `~/PLN` (examples/ or ruletests/).

What `call_plninit(kb)` does, in order **[S]** (`PLN.py` lines):
1. `:30` `cd ~/PLN && sh build.sh`; `build.sh:1-2` = `cd src; cat Utils Constraints Formulas Rules Config Deriver > ../PLN.metta`. **Translator.metta is not in PLN.metta.**
2. `:31` copy `src/Translator.metta` to `./TRANSLATE.metta`; `:32-33` append `!(superpose <kb>)` where `<kb>` = `str(a[0]).replace("#","")`.
3. `:34` `cat ../PLN.metta > TRANSLATED.metta`; **`:35` `metta TRANSLATE.metta 2>> TRANSLATED.metta`**: run the hyperon-experimental CLI on the translator and **append its stderr to TRANSLATED.metta**.
4. `:36-40` regex `\$([A-Za-z0-9]+)#([0-9]+)` -> `$\1\2` (removes hyperon's variable-renaming `#id` suffixes).
5. `:41-44` copy to `~/metta-morph/extend/plnblob.metta` (name overridable by argv[1]) unless identical; `:46-50` chdir there, `!(import! &self mettamorph)`, **`!(compile! plnblob.metta)`** (metta-morph = "macro-based MeTTa to (Chicken) Scheme translator" **[W]**; the comment in every example says "takes 30 sec to compile").
6. `:51` add the `sentence -> Sentence` alias to the host space; `:53-54` return `(PLN.Init.Success)`.

**What installs the translated equations (the trace! question).** `trace!` prints its FIRST argument to **stderr** and returns its SECOND argument. **[P]** `probe1.py`: `!(trace! (= (foo $x) (bar $x)) "second")`
printed `(= (foo $x) (bar $x))` on stderr and returned `"second"`. The Translator wraps every `(= ...)` it wants in `(trace! (= ...) "PLN.Init: Translated ... into function")`
(`Translator.metta:2-4,7-9,40-41`), so the equation is only *printed*; the returned string goes to stdout and is discarded. Nothing is asserted into any space. The line `2>>` at `PLN.py:35`
turns stderr into source text, so the engraved equations become **text appended after the library inside `plnblob.metta`, which metta-morph then compiles**. The same trick was used
before the Translator existed: `git show 3d55f3f -- src/Translator.metta` has `(= (question $x $q $d) (trace! (= (RUN) (Question $x $q $d)) (empty)))`.
Corollaries: (i) any other stderr output of that `metta` run also lands in the blob; (ii) the equations are in the order the translator produced them, which in hyperon 0.2.10 is **last-first** for
`!(superpose (...))` **[P]** `probe3.py` (`(superpose ((f 1) (f 2) (f 3)))` with a `trace!` in `f` printed 3, 2, 1) — the author's hyperon version is unknown, so the engraved-rule order is unknown;
(iii) free variables are printed as `$TV1#91`-style names **[P]** `probe2.py`, hence step 4 of the pipeline; variables in the KB itself print as `$1#926` after the host evaluated `(kbstatic)`, hence the `.replace("#","")` at `:32`.
(iv) a second `PLN.Init` in the same process would receive an argument already rewritten to `Sentence` by the alias of step 6 (arguments are evaluated first, **[P]** `probe2.py` second block shows `(kb)` evaluating),
and the Translator only matches lowercase `sentence`, so it would engrave nothing **[I]**; treat Init as once per process.

### A3. Data flow of examples/Smokes.metta

1. `Smokes.metta:1` import PLN.py. `:3-21` `(= (kbstatic) (...))`: six concept sentences `(Concept Anna..Helen) (stv 0.1667 0.9)`, one nested implication
   `(Implication (Evaluation (Predicate friend) (List $1 $2)) (Implication (Evaluation (Predicate smokes) (List $1)) (Evaluation (Predicate smokes) (List $2)))) (stv 0.4 0.9)`, one plain implication `smokes($1) -> cancerous($1) (stv 0.6 0.9)`.
2. `:24` `!(PLN.Init (kbstatic))`: the argument is evaluated, stringified, translated by the *hyperon CLI* as in A2, compiled by metta-morph.
3. The translator output (reconstructed by hand from `Translator.metta`, not executed; ids shown as printed, the regex then glues them on) is:
   ```
   (= (STV (Concept Anna)) (stv 0.1667 0.9))            ... x6, one per concept sentence
   (= (|- ((Evaluation (Predicate friend) (List $1 $2)) $TV1#n) ((Evaluation (Predicate smokes) (List $1)) $TV3#n))
      ((Evaluation (Predicate smokes) (List $2)) (Truth_Identity (Truth_ModusPonens (Truth_ModusPonens (Truth_Identity $TV1#n) (Truth_Identity $TV3#n)) (stv 0.4 0.9)))))
   (= (|- ((Evaluation (Predicate smokes) (List $1)) $TV1#m))
      ((Evaluation (Predicate cancerous) (List $1)) (Truth_Identity (Truth_ModusPonens (Truth_Identity $TV1#m) (stv 0.6 0.9)))))
   ```
4. **At query time the compiled blob contains**: the library (Utils, Constraints, Formulas, Rules, Config, Deriver, incl. `Test`), 6 `STV` facts, one 2-premise and one 1-premise engraved rule.
   **The static sentences are not beliefs and not tasks.** Their stamps (0),(1),(2) appear nowhere; the Smokes pin has stamp `(4 9 10)` (dynamic facts only), the engraved implications' evidence is *not tracked* in stamps.
5. `:26-56` `(kbdynamic)` = 8 `sentence`s with ids 3..10. `:58-62` `!(Test (PLN.Query (kbdynamic) (Evaluation (Predicate cancerous) (List (Concept Edward)))) ((stv 0.519920454545454 0.829078220412911) (4 9 10)))`:
   the KB list is evaluated in the host (alias makes them `Sentence`), `PLN.Query kb term` -> `PLN.Query kb kb term 20 20 200` (A/E below). The answer arises as: engraved binary rule on `friend(Anna,Edward)`[4] + `smokes(Anna)`[9] gives
   `smokes(Edward) (0.4, 0.729)` stamp (4 9); revision with the given `smokes(Edward) (1.0,0.9)`[10] gives `(0.8619, 0.9212)` stamp (4 9 10); engraved unary rule gives `cancerous(Edward) = MP((0.862,0.921),(0.6,0.9)) = (0.51992, 0.82908)`. [X] reproduced by the simulator.

Other facts about the flow: `Test` is `Utils.metta:48-49` `(= (Test $Is $Should) ((Is: $Is) (Should: $Should) (Passed: (== $Is $Should))))`; `examples/test.sh` and `ruletests/test.sh` run `metta X.metta | grep "Passed:"`.
The pins carry 15 significant digits (e.g. `0.519920454545454` for the double `0.5199204545454544`), consistent with Chicken Scheme printing flonums with 15 digits **[I]** (PeTTa prints the 17-digit shortest repr: `pln_tuffy.metta:50`).

---------------------------------------------------------------------------------------------------------------------------------------

## B. What `(STV X)` evaluates to for a node without a concept sentence

There is no catch-all equation in the fork (`grep STV src/` finds only the call sites in `Rules.metta` and the emitter at `Translator.metta:40`). A concept sentence = **any non-Implication static sentence**, not only `(Concept ..)`: `Translator.metta:38-42`
engraves `(= (STV $C) $TV)` for every static sentence whose term does not match `(Implication $A $B)` (`RuleTester.metta:5-11` engraves `Thunder`, `A`, `(Concept Sam)`, ...; `evaluationImplicationRuleA.metta:5-8` engraves `(Predicate IsFat)`, `(List (Concept Cat))`).

| Runtime | `(STV X)` with no matching equation | Evidence |
|---|---|---|
| hyperon-experimental 0.2.10 (runs the Translator only) | stays the inert data atom `(STV X)`; `(Truth_Deduction (STV A) ...)` then does not match `(stv $Ps $Pc)` and also stays inert; `(if <unreduced> a b)` stays inert; `(case ...)` with no matching branch returns nothing | **[P]** `probe1.py` |
| metta-morph (the runtime that actually executes `PLN.Query`) | the call **fails** (backtracks like `(empty)`): functions are wrapped in `(handle-exceptions exn ((amb-failure-continuation)) (apply (amb1 clauses) args))`, so a call without a matching clause yields no result and kills the enclosing alternative | **[W]** `mettamorph.scm` |
| PeTTa bundled lib | `(= (STV $stv) (empty))` "Forward declaration" (`BUNDLED:116`), added by `ccf834f` 2025-10-29 "forward declaration of STV function" | **[S]** |
| trueagi rewrite | the same line (`TRUEAGI:68`) | **[S]** |

So for the fork the operative semantics is: **a rule that calls STV on a node without an equation derives nothing**. Rules that call STV: Deduction (`Rules.metta:33-35`), Induction (`:42-44`), Abduction (`:50-52`),
Inheritance inversion (`:81`), Implication inversion (`:85`), Equivalence->Implication x2 (`:89-90,93-94`), Transitive similarity (`:99-101`), Evaluation implication (`:106-108`), Member deduction (`:114-116`).
Rules that never touch STV: Revision (`:4-6`), Modus Ponens (`:9-11`), Symmetric MP (`:19-22`), the three Evaluation-with-Inheritance rules (`:56-74`), Not elimination (`:77`), and **every engraved rule**.
Hence a fork run restricted to Revision + ModusPonens + engraved rules needs no node probabilities at all (only MP's constant 0.02). [X] `behaviours.py`: Deduction without STV equations -> no answer, with them -> `(0.8125, 0.9)`.
The rule examples show the consequence: `evaluationImplicationRuleA` has no `STV (Concept Cat)`, and its Inheritance inversion of `Tom->Cat` silently produces nothing.
The hyperon-experimental reading cannot be what the author ran: `min`/`max` are undefined in hyperon 0.2.10 (**[P]** `!(min 1 2)` is inert; metta-morph's own `mettamorph.metta` defines them **[W]**, which is why `Utils.metta:2-10` has them commented out) and `(- $c)` (`Deriver.metta:23`) is `(Error (- 0.9) IncorrectNumberOfArguments)` in hyperon 0.2.10 **[P]** while it is plain negation in Scheme.

---------------------------------------------------------------------------------------------------------------------------------------

## C. The Translator's exact behaviour (src/Translator.metta, 42 lines)

All engraving is `(sentence (<term> <tv>) <ids>)` -> zero or one `(= ...)`. Let `MP(a,b)=Truth_ModusPonens(a,b)`, `Neg=Truth_Negation` (`Formulas.metta:77-78`: `(stv (- 1.0 s) c)`), `Id=Truth_Identity` (`:138`).
Only **one** level of `(Not X)` is stripped per slot (the `(Not $Link1c)` pattern); the stripped inner term is what the rule matches and what it emits.

| Static sentence | Engraved equation | Lines |
|---|---|---|
| `(Implication L1 L2)`, L2 not an Implication, tv `T` | `(= (\|- (L1c $TV1)) (L2c (r (MP (p1 $TV1) T))))` with `p1=Neg` iff L1=`(Not L1c)` else `Id`; `r=Neg` iff L2=`(Not L2c)` else `Id` | emitter `:6-9`; plain case `:36` |
| `(Implication (Not A) B)` | fires on a belief whose term is **A** (positive), `B (MP (Neg tvA) T)` | `:33` |
| `(Implication A (Not B))` | fires on **A**, derives term **B** (not `(Not B)`) with **`Neg(MP(tvA,T))`** | `:35` |
| `(Implication (Not A) (Not B))` | fires on A, derives B, `Neg(MP(Neg tvA, T))` | `:32` |
| `(Implication L1 (Implication L3 L4))`, tv `T` | **2-premise** equation `(= (\|- (L1c $TV1) (L3c $TV3)) (L4c (r (MP (MP (p1 $TV1) (p3 $TV3)) T))))`; Not stripped independently on L1, L3, L4 (8 combinations: Not-L1 `:15-21`, plain L1 `:22-28`) | emitter `:1-4`, selection `:13-28` |
| `(Implication L1 (Implication L3 (Implication L5 L6)))` | L4 = `(Implication L5 L6)` is not a `Not`, so it stays a **term**: the engraved rule derives the *sentence* `((Implication L5 L6) MP(MP(..),T))`, which later composes through the generic Modus Ponens rule | `:13-28` (no deeper recursion) |
| any other static sentence | `(= (STV <term>) <tv>)`; guard `(== () (collapse (superpose ((let (Implication $A $B) $C 42)))))` excludes only 3-element Implication terms | `:38-42` |

Answers to the specific questions:
* **Which truth function each Not triggers**: a `Not` on a premise slot wraps that premise's TV in `Truth_Negation` before `Truth_ModusPonens`; a `Not` on the consequent wraps the *result* in `Truth_Negation` (`Translator.metta:17-28,32-36`).
* **The derived TERM of a negated consequent is `B`, with `Neg(MP(..))` as its truth value.** There is no `(Not B)` sentence anywhere. The generic negation *introduction* rule that used to create `(Not x)` for every task was removed in `7135ece` (rule comment `Rules.metta:76`: "introduction now handled by Translator"); only elimination remains (`Rules.metta:77`: `(Not A) T -> A (Neg T)`). Polarity is therefore carried by strength alone.
* A negated *antecedent* matches the positive term and flips its strength; it does **not** wait for a `(Not A)` belief. A `(Not A)` belief is first turned into `A (1-s, c)` by the unary elimination rule (one step, stamp unchanged), after which the engraved rule applies. [X] `behaviours.py`: with `Implication (Not A) B (0.9,0.9)`, belief `A(0.1,0.9)` and belief `(Not A)(0.9,0.9)` both give `B = (0.812, 0.81)`; belief `A(0.9,0.9)` gives `B = (0.108, 0.81)`.
* MP is `(f1*f2 + 0.02*(1-f1), c1*c2)` (`Formulas.metta:46-48`): a *weak* antecedent yields a conclusion strength near **0.02**, i.e. "B is probably false", with confidence `c1*c2` — not "no information". Consequence for `A -> not B` chains: [X] A(1.0,0.9), `A -> Not B (0.9,0.9)`, `B -> C (0.9,0.9)` gives `B=(0.1,0.81)`, `C=(0.108,0.729)`.
* Engraved implications keep their **own evidence out of the stamp** (A3.4), while an implication supplied as a *dynamic* belief is stamped like any fact and is used by the generic MP/Deduction/Inversion rules. Dynamic `(Implication A (Not B))` works in two steps: MP derives `(Not B)` then elimination derives `B` with the same numbers but stamp `(1 2)` instead of `(1)` [X] `dyn_impl.py`. Dynamic `(Implication (Not A) B)` only fires on a belief whose term is literally `(Not A)`; a belief `A` does nothing (`B = None`).
* Double counting [X]: two independent facts `P(0.8,0.9)[1]`, `P[2]` and an engraved `P->Q (0.6,0.9)` give `Q = (0.484, 0.895)` stamp (1 2) (versus `(0.484,0.81)` from one fact) because each derivation multiplies the implication's confidence but the implication is not in the stamp, so the two derivations are revised as independent.

---------------------------------------------------------------------------------------------------------------------------------------

## D. Complete rule list (the order is file order; `\|-` with one or two `(term tv)` arguments)

Truth formulas are quoted from `Formulas.metta`; `conditional-probability-consistency` (cpc) is `Constraints.metta:11-15` (`0<As`, `clamp((As+Bs-1)/As,0,1) <= ABs <= clamp(Bs/As,0,1)`);
`/safe` (`Utils.metta:34-37`) yields **no result** when its divisor is not > 0.0; `c2w(c)=c/(1-c)`, `w2c(w)=w/(w+1)` (`Formulas.metta:58-64`).
Worked examples were produced by the simulator and recomputed by hand-typed arithmetic (`SIM/rule_examples.py`, 26/26 equal). `stv a b` below is (strength, confidence).

| # | Rule (Rules.metta) | Premises -> conclusion | Guard | Truth |
|---|---|---|---|---|
| R1 | Revision `:4-6` | `(T t1)`, `(T t2)` -> `(T Rev(t1,t2))` | same term (unification) | `Formulas.metta:66-73`: `w1=c2w c1, w2=c2w c2, w=w1+w2, f=(w1 f1+w2 f2)/w (via /safe), c=w2c w; (stv (min 1.0 f) (min 1.0 (max (max c c1) c2)))`. **c=1.0 gives no result** (c2w divides by 0). Ex: `(0.6,0.5)`+`(0.9,0.5)` -> `(0.75, 0.6667)` |
| R2 | Modus Ponens `:9-11` | `(A t1)`, `((Implication A B) t2)` -> `(B MP(t1,t2))` | — | `:46-48`: `(stv (+ (* f1 f2) (* 0.02 (- 1 f1))) (* c1 c2))`. Ex: `(0.9,0.9)`,`(0.8,0.9)` -> `(0.722, 0.81)` |
| R3 | Symmetric MP `:19-22` | `(A tA)`, `((L A B) tAB)` -> `(B SMP)` | `L` in {Similarity, IntentionalSimilarity, ExtensionalSimilarity} (`:14-16`), else `(empty)` | `:51-55`: `(stv (+ (* sA sAB) (* (* 0.2 (- 1.0 sA)) (+ 1.0 sAB))) (min (min cAB 1.0) cA))`. Ex: `(0.8,0.9)`,`(0.7,0.6)` -> `(0.628, 0.6)` |
| R4 | Deduction `:29-35` | `((L A B) t1)`, `((L B C) t2)` -> `((L A C) Ded(STV A,STV B,STV C,t1,t2))` | `L` in {Inheritance, Implication} (`:25-26`); needs STV of A,B,C | `:5-20`: if cpc(Ps,Qs,PQs) and cpc(Qs,Rs,QRs): `stv (if (< 0.9999 Qs) Rs (+ (* PQs QRs) (/safe (* (- 1 PQs) (- Rs (* Qs QRs))) (- 1 Qs)))) (min Pc (min Qc (min Rc (min PQc QRc))))`, else the **integer** pair `(stv 1 0)`. Ex (STV=0.2): `(0.99,0.9)`,`(0.99,0.9)` -> `(0.980125, 0.9)`; inconsistent premises -> `(1 0)` |
| R5 | Induction `:39-44` | `((L C A) t1)`, `((L C B) t2)` -> `((L A B) Ind(STV A,STV B,STV C,t1,t2))` | `L` syllogistic | `:23-31`: `s = /safe(sBA*sBC*sB, sA) + (1 - /safe(sBA*sB, sA)) * /safe(sC - sB*sBC, 1-sB)`, `c = w2c(min cBA cBC)` ("confidence TODO check") |
| R6 | Abduction `:47-52` | `((L A C) t1)`, `((L B C) t2)` -> `((L A B) Abd(STV A,STV B,STV C,t1,t2))` | `L` syllogistic | `:34-43`: `s = /safe(sAB*sCB*sC, sB) + /safe(sC*((1-sAB)*(1-sCB)), 1-sB)`, `c = w2c(min cAB cCB)` |
| R7 | Evaluation-with-Inheritance, unary `:56-60` | `((Evaluation (Predicate x) (List (Concept C))) t1)`, `((Inheritance (Concept S) (Concept C)) t2)` -> `((Evaluation (Predicate x) (List (Concept S))) MP(t1,t2))` | exact shapes | MP. Ex: `(0.9,0.8)`,`(0.7,0.9)` -> `(0.632, 0.72)` |
| R8 | binary arg1 `:63-67` | `Evaluation x (List C1 C2)` + `Inheritance S C1` -> `Evaluation x (List S C2)` | shapes | MP |
| R9 | binary arg2 `:70-74` | `... Inheritance S C2` -> `Evaluation x (List C1 S)` | shapes | MP |
| R10 | Not elimination `:77` (unary) | `((Not A) t)` -> `(A Neg(t))` | — | `(stv (- 1.0 s) c)`. Ex: `(0.99,0.9)` -> `(0.01, 0.9)` |
| R11 | Inheritance inversion `:80-81` (unary) | `((Inheritance A B) t)` -> `((Inheritance B A) Inv(STV B, t))` | needs STV B | `:80-83`: `(stv ABs (* Bc (* ABc 0.6)))` (strength **copied**, only confidence penalised). Ex: `(0.87,0.81)`, STV B conf 1.0 -> `(0.87, 0.486)` |
| R12 | Implication inversion `:84-85` | same for `Implication` | needs STV B | same |
| R13 | Equivalence -> Implication `:88-90` | `((Equivalence A B) t)` -> `((Implication A B) E2I(STV A,STV B,t))` | needs STV A, B | `:86-92`: `s = (if (< 0.99 (* ABs ABc)) ABs (/safe (* (+ 1.0 (/safe Bs As)) ABs) (+ 1.0 ABs)))`, `c = ABc`. Ex: `(0.8,0.7)`, As=.4,Bs=.2 -> `(0.6667, 0.7)` |
| R14 | `:92-94` | same premise -> `((Implication B A) E2I(STV A,STV B,t))` (STV order unchanged) | | same |
| R15 | Transitive similarity `:97-101` | `((Similarity A B) t1)`, `((Similarity B C) t2)` -> `((Similarity A C) ...)` | needs STV A,B,C | `:95-111`: `TransitiveSimilarityStrength` (T1..T4 with plain `/`), `c = min ABc BCc`. Ex: uniform .333, `(1.0,.89)`,`(1.0,.5)` -> `(1.0, 0.5)` |
| R16 | Evaluation implication `:104-108` | `((Evaluation A B) tAB)`, `((Implication A C) tAC)` -> `((Evaluation C B) ...)` | needs STV A,B,C | `:114-135`: `s = simpleDeductionStrength(Bs,As,Cs,ABs,ACs)` (no result if cpc fails), `c = (0.9*0.9) * min5(Bc,Ac,Cc,ACc,0.9*ABc)`. Ex: all 1.0, STV .25 -> `(1.0, 0.729)` |
| R17 | Member deduction `:112-116` | `((Member A B) t1)`, `((Inheritance B C) t2)` -> `((Member A C) Ded(STV A,STV B,STV C,t1,t2))` | needs STV A,B,C | R4's formula |
| T* | engraved rules (section C) | 1 or 2 premises | pattern from the static sentence | MP chains with Neg/Id |

Every rule is tried in both argument orders by the Deriver (`(|- x y)` and `(|- y x)`), so the table lists each pattern once. All 2-premise rules require stamp-disjoint premises (E3).
**Latent argument-order mismatch in R5 and R6 [S][X]**: both call `(Truth_X (STV $A) (STV $B) (STV $C) $T1 $T2)` with the *shared* premise node in `$C`, but the formula's second slot is the shared node (book formula: premises B->A, B->C). The shared node's probability thus lands in the third slot.
Not visible when all node strengths are equal (every pinned example uses a uniform prior). [X] `rule_examples.py`: nodes A=.3,B=.4,C=.5, `T1=(0.8,0.9)`, `T2=(0.6,0.8)`: Induction gives s=0.6111 as called vs 0.7333 with the shared node in the formula's slot; Abduction 0.6667 vs 0.4480. The same call text is in `BUNDLED` and `TRUEAGI`. Deduction, Member deduction and the other STV-using rules are aligned.
Other things that matter for a design: MP's literal 0.02; inversion copies strength (right only for equal node probabilities); `Truth_Deduction` can emit the integer pair `(stv 1 0)`: a sentence with confidence 0 is selected only when no task has higher confidence, occupies a queue slot, and `PLN.Query` never returns it (rank 0 does not beat `()`).

---------------------------------------------------------------------------------------------------------------------------------------

## E. The control loop (src/Deriver.metta, Config.metta)

* **Defaults** `Config.metta:4,7,10`: `MaxSteps 20`, `TaskQueueSize 20`, `BeliefQueueSize 200`. Entry points `PLN.Derive` (6/5/3/2 args, `Deriver.metta:34-61`) returning `($Tasks $Beliefs)`, `PLN.Query` (6/5/3/2 args, `:68-81`). `PLN.Query kb term` = `PLN.Query kb kb term 20 20 200`; the 6-arg form takes separate tasks and beliefs.
  (BUNDLED has 100/10/100, `BUNDLED:369-376`; TRUEAGI also 100/10/100.)
* **Initial state**: `Tasks = Beliefs = kb`, as given: not de-duplicated, not truncated. The first iteration therefore pairs the selected task with the whole KB, however large.
* **Selection** (`:37`, `BestCandidate :8-16`, `PriorityRank :19`): the task with the highest **confidence**; strict `>` so the **first** of equal confidences wins. Strength plays no role.
* **Derivation** (`:38-45`): for every belief y in belief order: `stamp = sorted(Ev1 ++ Ev2)` (`InsertionSort`, `Utils.metta:52-68`); if `StampDisjoint(Ev1,Ev2)` (`:2-5`: no common id) take all results of `(|- x y)` then all of `(|- y x)` as `Sentence (t tv) stamp`; after all beliefs, the results of the **unary** `(|- x)` with stamp `Ev1`. Only results shaped `(term tv)` survive (`case`).
  Stamps hold dynamic-fact ids only (engraved implications add none). A conclusion never combines with a belief that shares any id, so two derivations of the same term from overlapping evidence both stay as separate beliefs; `PLN.Query` returns the more confident, it does not revise them.
* **Queue update** (`:47-50`): `Tasks' = LimitSize(Without(Unique(Tasks ++ derivations), selected), TaskQueueSize)`; `Beliefs' = LimitSize(Unique(Beliefs ++ derivations), BeliefQueueSize)`. The selected task stays in Beliefs. Derived sentences that equal an existing *belief* but not an existing *task* are re-queued as tasks (Unique is per list), so a sentence can be selected again.
  `Unique` (`Utils.metta:85-92`) **prepends** each first occurrence (`(Unique rest (TupleConcat ($x) $Ret))`), so it returns the de-duplicated list in **reverse** order, on every iteration (so the order of both queues flips each step). Equality is whole-sentence equality including the exact floats (a path-dependent `0.7` vs `0.7000000000000001` is not a duplicate).
  `LimitSize` (`:27-31`) `if count < size then L else remove the lowest-confidence item (BestCandidate over -c: first of equals) and recurse`, so a queue of size N **keeps at most N-1 items** (19 tasks / 199 beliefs by default).
  *Consequence [X]*: a KB larger than the belief bound is cut to the 199 most confident sentences after step 1 and the others vanish from the final belief set: a 301-fact KB containing one `(0.9,0.3)` fact returns `None` when that fact is queried, and the fact is found with `belief_queue_size=400` (`behaviours.py`).
* **Termination** (`:35`): `steps > maxsteps` or empty task queue; `steps` starts at 1, so exactly `maxsteps` selections unless the queue empties. There is no goal test: the loop never stops early when the query term appears. No cycle can hang it (every iteration consumes one step), but cycles waste steps (inversion of an inversion, A->B->A deduction).
* **PLN.Query** (`:68-72`): run `Derive`, collect every final belief whose term `==` the query term, return **the single `($TV $Ev)` pair, i.e. `((stv s c) (ids))`, of the highest-confidence one (first of equals)**; `()` if none. `ConfidenceRank () = 0`, so a candidate with confidence 0 never beats `()`. No revision is done at query time and the answer is not guaranteed to be the revision of everything known about the term.
* **Cost** [X] `SIM/cost_stats.py` (instrumented counts, default 20/20/200): Smokes 331 belief pairs / 4514 rule-equation attempts / 14.8 ms in Python; FlyingRaven 734 / 5932 / 18 ms; Toothbrush 303 / 2124 / 4.9 ms. Per step the cost is `|Beliefs| x 2 orders x (#library pair rules + #engraved pair rules)` plus the unary rules, plus O(n^2) for `Unique`/`LimitSize` (each is a MeTTa tuple walk with an O(n) membership test); with the defaults `<= 20 x 199 x 2 x (12 + K)` rule attempts. Every engraved implication is one more equation tried per call, i.e. cost grows **linearly in the number of engraved rules** (K=1000 irrelevant unary implications add 1000 attempts per selected task), and compile time (the "30 sec" comment) grows with the blob. `PLN.Query` re-runs the whole derivation every call.

---------------------------------------------------------------------------------------------------------------------------------------

## F. What every example and ruletest pins

Pins are the arguments of `Test` (`Utils.metta:48-49`); the `;a b c)` fragments are comments, not tests.

| # | File:lines | Query (steps tq bq) | Pinned `((stv s c) (ids))` |
|---|---|---|---|
| 1 | `examples/Smokes.metta:58-62` | `cancerous(Edward)` default 20 20 200 | `((stv 0.519920454545454 0.829078220412911) (4 9 10))` |
| 2 | `examples/FlyingRaven.metta:38-42` | `flies(Sam)` default | `((stv 0.97052375 0.81) (1 4 6))` |
| 3 | `examples/FlyingRaven.metta:44-48` | `flies(Pingu)` default | `((stv 0.332225678571428 0.83690036900369) (2 3 5 6))` |
| 4 | `examples/Robot.metta:56-61` | `PLNobjectsOfCategory` over 4 detections, category `bring` (each is a `PLN.Query`, default) | `((detection frisbee someCoords1) (detection orange someCoords4))` |
| 5 | `examples/Toothbrush.metta:82-86` | `Inheritance X removed` default | `((stv 0.905100467696027 0.895338254258716) (10 11 12 13 14))` |
| 6 | `ruletests/RuleTester.metta:30-33` | `Inheritance Penguin Car`, 2 steps | `((stv 0.01 0.9) (5))` |
| 7 | `ruletests/RuleTester.metta:36-39` | `Inheritance Raven Sam`, 2 | `((stv 0.99 0.54) (1))` |
| 8 | `ruletests/RuleTester.metta:42-45` | `Implication Thunder Donner`, 5 | `((stv 0.994974874371859 0.9) (2))` |
| 9 | `ruletests/equivalenceToImplication.metta:9-13` | `Implication Anna Frank`, 1 | `((stv 0.98989898989899 0.87) (1))` |
| 10 | `ruletests/evaluationImplicationRuleA.metta:11-24` | `Evaluation IsFat (List Tom)`, 10 | `((stv 1.0 0.729) (1 2 3))` |
| 11 | `ruletests/evaluationWithNegationAndInheritanceInversion.metta:9-19` | `Evaluation IsFat (List D)`, 4 | `((stv 0.8 0.6) (1 2))` |
| 12 | `ruletests/inversion.metta:13-17` | `Inheritance A B`, 1 | `((stv 0.87 0.486) (1))` |
| 13 | `ruletests/memberDeductionA.metta:14-28` | `Member Tom SpeciesCats`, 1 | `((stv 0.75 0.5) (1 2))` |
| 14 | `ruletests/transitiveSimilarity.metta:12-17` | `Similarity A C` default | `((stv 1.0 0.5) (1 2))` |
| — | `examples/FlyingRaven.metta:50-55` | `PLN.Derive` 1 step from `Inheritance Pingu Penguin`; prints a set difference, no pin | [X] new beliefs: `Penguin->Pingu (0.99,0.486)(2)`, `Pingu->Bird (0.980125,0.9)(2 5)`, `flies(Pingu) (0.0297,0.81)(2 3)` |

PeTTa-side pins (all use BUNDLED, not the fork's files): `pln_roman.metta:15-16` `((stv 0.5 0.9473684210526316) (1 2 3 4))`; `pln_tuffy.metta:47-50` `((stv 0.5199204545454544 0.8290782204129108) (1 2 4 9 10))`;
`plntest.metta:50-52` `(stv 0.7333333333333334 0.9)`; `plntestdirect.metta:46-47` `(stv 0.8166666666666668 0.9)`; `pln_direct.metta:26-31` `(stv 0.51 0.8432620011567381)`, `(stv 0.5198 0.7923434575206812)` (formula level: MP chains + Revision fold from `(stv 0.5 0.0)`). `nars_direct.metta` uses `lib_nars` and is out of scope.

**Step counts are tight [X]**: for `RuleTester` tests 6 and 8 and test 11, steps-1 loses the pin (`explore_ruletests_tight.py`); test 6 additionally needs the `Unique` reversal (E): with an order-preserving Unique the second step selects the Equivalence and the answer is `None`.

**The "example-specific minimal resource" comments (`;11 2 11`, `;2 6 9`, `;6 5 9`, `;13 7 17`, `;10 2 12`) are stale and are not pins.** They were last edited in `39b771e` (2025-08-04), when `Rules.metta` still had a generic negation-introduction rule (removed `7135ece`, 2025-08-15) and Robot queried `sportsball`.
[X] With the HEAD rules the simulator reproduces `2 6 9` exactly (every decrement breaks it) and the `tq`/`bq` tightness of `11 2 11` and `10 2 12` (steps may drop by one), but `6 5 9` (Pingu) gives no answer (smallest found: steps 7 with tq 5, bq 11; or steps 6 with tq 6, bq 11) and `13 7 17` (Robot) loses `orange`. Re-adding either historical negation rule (`explore_hist.py`) or varying tie-breaks/pair order/equation order (64 combinations, `explore_conventions.py`) does not reproduce all four. **Unreconciled**; not used for any decision. Keeping the `LimitSize` `<` (N-1 items) is what makes the tq/bq tightness appear (`<=` loosens it).

---------------------------------------------------------------------------------------------------------------------------------------

## G. Evaluation-order choices, runtime unknowns, and the three code lines compared

Choices made in `fork_sim.py` (switchable): Unique reverses (`unique_mode='fork'`; `'keep'` is BUNDLED's `list_to_set`); equations tried in file order (library first, then engraved in KB order; `eq_order='reverse'` = metta-morph's newest-first cons order **[W]**, unverified);
`BestCandidate`/`LimitSize` first-of-equals; per belief `(|- x y)` results before `(|- y x)`; unary results last; `Without` removes all equal elements; `InsertionSort` ascending; int/float distinction kept (repr-based equality).
Sensitivity [X]: all 14 fork pins and all PeTTa pins are insensitive to `eq_order`, tie-breaks, pair order and unary-first (`explore_conventions.py`, 64 combinations all keep the 4 example pins); only `unique_mode` is discriminated (RuleTester #6 fails with `'keep'`).
Not established: metta-morph's exact clause order and how compiled functions are exposed to the host (`compile!` source not read); the hyperon CLI version used for translation; the `Passed:` outcome of the real runs (only the pins as written are known).

What changed, and why, between the lines:
| Item | fork 55f1751 | BUNDLED (PeTTa lib, 2025-10-30 from the fork lineage) | TRUEAGI 4405956 (2026-03-15) |
|---|---|---|---|
| Static/engraved KB, `PLN.Init`, Translator | yes (Python + metta-morph) | **removed**: all knowledge is `Sentence` beliefs plus hand-written `(= (STV X) tv)` (`pln_roman.metta:3-13`), rule file is one library; no `Truth_Identity` | removed; single file |
| Defaults steps/tasks/beliefs | 20/20/200 | 100/10/100 | 100/10/100 |
| `Unique` | reversing prepend | `list_to_set` (keeps order) | `ConcatUnique/PushUnique` incremental, order kept (`c17c750`, "ordering fixes") |
| `StampDisjoint` | `collapse`/`case` | same | recursive `is-member` |
| STV fallback | none (fails in metta-morph) | `(= (STV $stv) (empty))` | same |
| Deduction conf | `min` of 5 confs (`Formulas.metta:18`) | same | `(* (* PQs QRs) (* PQc QRc))` (`TRUEAGI:97`) |
| Modus Ponens conf | `c1*c2` (`:48`) | same | `(* (* Ps PQs) (* Pc PQc))` (`:126`) |
| Induction / Abduction conf | `w2c(min c c)` (`:31,:43`) | same | `w2c((sBC*cBC)*cBA)`, `w2c((sAB*cAB)*cCB)` (`:110,:121`) |
| Symmetric MP / transitive similarity / evaluation-implication conf | `min`s / `min` / `.81*min5` | same | `(cA*cAB)*Truth_or`, `(ABc*BCc)*Truth_or`, `(ABs*ACs)*(ABc*ACc)` |
| Strengths, Revision, Negation, inversion, E2I | — | identical to the fork | identical |
The trueagi confidence changes came in `3529c70` "Truth Functions and Confidences are better fit" (2026-03-12, Peter Isaev). A separate branch `origin/NARS` (c7232a9) of the fork repo changes the engraving to carry implication stamps (`|-s`, `7d634fe`) and adds backward abduction equations; it is not the target.

---------------------------------------------------------------------------------------------------------------------------------------

## H. The simulator and its validation

`SIM/fork_sim.py` (stdlib only). API: `Sim(rules, maxsteps, task_queue_size, belief_queue_size, static, variant, unique_mode, eq_order)`, `.init(static)` (the Translator's effect: non-Implication static sentences -> STV table, Implication sentences -> engraved rules incl. the Not handling of section C),
`.derive(kb)` -> `(tasks, beliefs)`, `.query(kb, term)` -> `((s, c), stamp)` or `None`, `.infer1/.infer2` (one `(|- ...)`), `.stats`. Rule sets by name (`RULE_NAMES`; presets `full`, `restricted` = Revision+ModusPonens+NotElimination+Translated, `restricted+deduction`); `variant='petta_lib'` = BUNDLED defaults. Sentences are `(term, (s, c), (ids))` (`S(term,s,c,ids)`); `parse_sexprs`/`kb_from_metta` read MeTTa text as data.
`SIM/validate_fork_sim.py` reads the upstream example files as **data** (s-expression parser; no execution) and runs all 14 Test lines plus PeTTa's five example files: **20/20 match**. Fork pins match after rounding the simulated numbers to 15 significant digits (the pin format; max abs difference 5e-16);
PeTTa pins match bit-exactly. Results in `SIM/validation_results.json`. `pln_tuffy` (variable-containing implication *beliefs*, stamps (1 2 4 9 10)) needs the full two-way unifier and passes with BUNDLED defaults; with the fork defaults 20/20/200 and reversing Unique the same KB answers `(0.6, 0.81) (2 10)` [X] — the answer depends on the resource settings.
Run: `python3 -I validate_fork_sim.py [--json out.json] [--eq-order file|reverse] [--unique fork|keep]` (cwd-independent, ~0.2 s). Files: `explore_*.py` (sensitivity/minimal-resource/historical-rule exploration), `rule_examples.py`, `cost_stats.py`, `behaviours.py`, `dyn_impl.py`, `show_traces.py`, `ravens_derive.py`, `robot_minimal.py`. Probes of hyperon builtins: `facts/probes/probe{1,2,3}.py`.

## I. What this means for the migration design (short)

1. The runtime unit of reasoning is one Sentence list per query; the cost of a query is `maxsteps x |beliefs| x rules`, with no indexing. A KB bigger than ~199 sentences is silently truncated by confidence after step 1.
2. A restricted set `Revision + ModusPonens + NotElimination + engraved rules` needs no node probabilities and is exactly the part of the fork whose numbers are fully pinned and unambiguous; Deduction adds the STV table and the consistency check (`(stv 1 0)` on failure). Induction/Abduction have the slot mismatch above and inversion copies strengths: keep them behind the comparison switch.
3. Sign handling: the fork stores no `(Not B)` conclusions; a negative edge is `B` with `1 - MP`. A node's single strength therefore mixes positive and negative evidence; whether that is acceptable for Pos/Neg effect edges is a design decision, not something the fork decides for us.
4. Stamps record dynamic facts only. If the longevity KB's implications/edges must count as evidence for revision, they have to be supplied as dynamic stamped beliefs (generic rules) rather than engraved rules.
5. Port `PLN.Init` as an explicit Python function (the `Sim.init` of this port): the real one is a shell pipeline over stderr that needs hyperon CLI, metta-morph, hard-coded home paths and runs once per process.
