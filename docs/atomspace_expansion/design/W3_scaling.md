# W3: PeTTa and hyperon against a synthetic Layer A of 100, 300 and 1,000 abstracts

Measured 2026-11-03 on this work tree (then on branch `claude/gpt6-temperature`, `1a5fc78`; repo line citations
below are to `d21ae14`, the base of both that branch and `claude/atomspace-expansion-design`; PeTTa `fea8317` at
`/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa`, SWI-Prolog 10.0.2, hyperon 0.2.10, macOS arm64,
48 GiB). No LLM was called, nothing was installed, both API keys were unset in every child, and a write guard over
this tree, the PeTTa tree and the venv found no change after any run. Numbers are in `W3_scaling_summary.json`
beside this note; the raw JSON and the harness live in the session scratchpad (`$SP/w3/`) and are throwaway.

**What the data are.** No real Layer A exists yet: the dry run parsed 3 abstracts (22 sentences). The harness
(`gen_layer_a.py`) builds synthetic abstracts by resampling parsed sentences and renaming their concepts, then plants
33 chains of 1 to 4 hops with known answers, 1,000 unrelated hops per 1,000 abstracts and fan-out around the chains.
It ran twice, from two seeds of sentences:

| Run | Sentences resampled | Statements per abstract | Distinct concepts at N=1,000 |
|---|---|---|---|
| hand-made | W1's 7 exemplar sentences, 37 statements | 39 to 41 | 13,668 |
| real density | the 22 sentences gpt-6.1-terra parsed on 2026-11-02, 209 statements | 70 to 74 | 25,193 |

Each run is the runner's A, then B replayed from A's recorded commands, then C for any step whose metrics differed
by more than 10%. Every answer was identical across A, B and C (0 nondeterministic metrics of 7,788 and 8,142); the
flagged gaps are wall-clock noise on steps under a second. One harness bug cost time, not data quality, and is
fixed: at N=1,000 a layout that overflowed the 2 GB stack was reloaded before each of its 119 queries (about 48 of
the 55 minutes of that step, in all three passes, and those queries had no results). The match and derive phases now
reopen such a layout once at the 8 GB stack, as the load phase already did; that step was re-run twice on the same
data (A 428 s, B 441 s, identical answers), and the N=1,000 real-density figures below are from the re-run.

## PeTTa holds 1,000 abstracts in one space

The parser layout, one fresh child per load (`load_s` and peak RSS as the child reports them):

| N | hand-made: file, load, peak RSS | real density: file, load, peak RSS |
|---|---|---|
| 100 | 0.9 MB, 0.62 s, 174 MB | 2.0 MB, 1.41 s, 592 MB |
| 300 | 2.6 MB, 1.96 s, 599 MB | 5.7 MB, 4.32 s, 1,170 MB |
| 1,000 | 8.8 MB, 6.93 s, 2,292 MB | 18.9 MB, 15.5 to 15.8 s, 4,093 MB |

- Every planted chain is found by `match` at every size: 683 queries per size, 0 errors, 0 misses. At N=1,000 real
  density the median query is 1.8 to 2.3 ms; the slowest is an open 4-hop pattern at 129 to 167 ms.
- The curated stack is undisturbed: with Layer A loaded beside the generic stack (`generic+parser`, 15.5 to 16.2 s,
  4,324 MB at N=1,000) its control query still returns its 16 rows.
- Two heavier layouts overflow the 2,048 MB Prolog stack (the repo default, `config.py:154`) at N=1,000 real
  density (on the hand-made seed only the second does) and load at 8,192 MB: the parse plus its type definitions (29 to 30 s, 5,074 MB) and the parse plus the sentence text as
  SentenceLinks (26 to 27 s, 9,050 MB). Keeping provenance text in the same space more than doubles memory.
- Memory grows faster than linearly: about 2.9 MB per real-density abstract from 100 to 300 and 4.1 MB from 300 to
  1,000. By extrapolation (not measured), Ben's 600-paper seed on abstracts is about 2.5 GB; one full text at W2's
  expected 250 sentences and terra's 9.5 statements per sentence is about 33 abstracts' worth, so the 356-paper
  stage 1 on full text would be at least 48 GB, this machine's whole memory, in one space.

**The repo cannot run this as it stands.** The inference path forks a fresh PeTTa child per program and loads the
whole KB into it (`pln_runner.py:871`, `metta_runtime.py:18-30`), under a 512 MB growth cap by default
(`PETTA_MEMORY_LIMIT_MB`, `metta_runtime.py:38`). At real density that cap is passed below 100 abstracts (592 MB at N=100), and at
N=1,000 each question would pay 16 s and 4 GB before its first step. Layer A therefore needs either a warm
long-lived session (the runtime has one, `metta_runtime.py:107, 250`; the harness's matches used it) or a per-query
slice of Layer A pulled into the KB. That choice is W4's.

## hyperon cannot hold Layer A

| Probe | hand-made | real density |
|---|---|---|
| Layer A alone, largest prefix that answers / smallest that aborts | 6 abstracts, 914 keys / 7, 1,085 | 2, 732 / 3, 1,032 |
| beside the generic stack (812 keys) | 1 abstract, 999 / 2, 1,187 | none / 1 abstract, 1,259 |
| curated pair query with N = 100, 300, 1,000 loaded (up to 196,387 keys) | answers | answers |
| lookups of Layer A sentences at N = 100, 300, 1,000 (3 per N) | 4 of 9 answer | 2 of 9 answer |

The abort (`trie.rs:179`, hyperon #1076/#1081) is deterministic, identical in all passes, and depends on the query,
not only on the space: a query that never enters Layer A's part of the index answers with 196,000 keys loaded, while
lookups into Layer A abort once a few abstracts are in. That is consistent with the limit applying to the keys a
query has to walk, not to the whole space; it was not checked against hyperon's source. For the design it settles
the same thing either way: Layer A lives on PeTTa only, which matches draft row 7 ("hyperon never sees them"). With
the quiet-abort handler in the probe child, the 60 deliberate aborts of the full runs filed no crash report.

## What patham9's PLN can read, and what it costs

The restricted library reads only `(Implication (State a) (State b))` sentences, so each Layer A statement is either
rendered to that shape or dropped with a reason. At N=1,000 real density, 70,494 statements went in:

| Outcome | Count |
|---|---|
| not an Implication (facts, events, attitudes): no rule renders them | 33,213 statements (47%) |
| dropped inside Implications: multi-premise event bridges | 15,279 |
| dropped inside Implications: conclusion conjuncts naming an instance or an existential variable | 26,167 conjuncts |
| dropped inside Implications: premise not `(Pred $var concept)`; no renderable conclusion | 1,704; 2,247 |
| rendered | 22,705 links over 13,239 nodes (1,665 duplicate source and target), including the planted ones |

The hand-made seed renders far less (76% of statements are not Implications; 3,271 links from 39,115 statements).
Every rendered link carries the parser's LLM-assigned STV, which the honesty contract keeps out of calibrated
answers; this is row 4's problem, not a runtime one.

`PLN.Derive` on the rendered links derived all 116 planted conclusions per size, in both runs and every pass, with
the expected STV exactly (absolute error 0). Its cost follows the beliefs reachable from the query's source, not the
size of Layer A:

| Rendering | Reachable links handed to Derive | Median | Where |
|---|---|---|---|
| plain links | up to 61 | at most 0.055 s | both seeds, every N |
| with Equivalence links instantiated | 46 | 1.6 s | real density, N=1,000 |
| with Equivalence links instantiated | 52 to 53 | 2.8 s | hand-made, N=300 |
| with Equivalence links instantiated | 107 | 9.0 s | hand-made, N=1,000, step budget raised to 2,048 |
| with Equivalence links instantiated | 125 | 43 s | hand-made, N=1,000, step budget 2,048 |

The hand-made seed reuses fewer concept names (a Zipf over a smaller vocabulary), so its graph has bigger hubs and
bigger neighbourhoods; the real vocabulary spread is not measured, and Derive's cost depends on it more than on N.
Unreachable beliefs also cost: adding 256, 1,024 and 4,096 of them to a 3-hop Derive call (N=300) raised it to 0.06,
0.24 and 0.94 s, about 0.23 ms each, so handing all 22,705 links at N=1,000 to one call would be about 5 s before
any search (linear extrapolation). Derive needs a neighbourhood, not Layer A.

## Parser-shape hazards on PeTTa

All 23 hazard KBs load and all 28 hazard queries match, including the 4-element `(: proof body (STV s c))`, uppercase
`STV` destructuring, ISO timestamps, `PointMass` TVs, SentenceLink punctuation and an unescaped quote,
`Premises`/`Conclusions` conjunctions, open-predicate patterns, and `PLN.Derive` beside parser atoms and the parser's
own type definitions. Two name hazards matter for the bridge:

- 20 lowercase names collide with PeTTa builtins in head position (`length max min log sort exp sqrt member append
  reverse first last msort not and or let empty eval superpose`): an atom headed by one does not read back as data
  (`max` comes back as `(partial max ..)`, `length` as nothing). They still match as arguments and constants.
- The KB scan refuses 12 names (`argv assert call foldl halt include library maplist repra shell sread test`), and a
  KB holding one is refused whole, so one parsed concept named `test` or `library` would take Layer A down with it.
  The bridge has to rename or quote these before anything reaches PeTTa.

## What W3 could not measure

- Content: whether real Layer A is right, useful or redundant needs a real stage 1 (the paid run, about $55 on
  abstracts, behind the owner's yes after sign-off).
- Real vocabulary spread and hub structure (the generator's Zipf is a guess), and with it the real Derive cost.
- Real rates of noise hops, fan-out and duplicate proofs; how many statements a full text yields beyond the one
  luna parse (167 statements for 55 sentences).
- Anything inside the server: concurrent queries, a warm session's lifetime and memory under load, incremental
  adds without a reload (PeTTa has no reset: one load per process, `metta_runtime.py:22-26`).
- Beyond 1,000 abstracts; the 16 GB growth cap the harness set was never reached.

## What this changes in the draft rows

| Row | What W3 settles or changes |
|---|---|
| 4 (W3's row) | Runtime side measured: PeTTa holds Layer A at 1,000 abstracts and patham9's PLN derives exactly on what renders, but only part of Layer A renders (47% of statements cannot), every rendered STV is the LLM's, and Derive must be handed a neighbourhood. The three stances (PLN reads Layer A; Layer A is mined into Layer B only; a hybrid) are weighed in DESIGN.md against these numbers and the honesty contract. |
| 5 (W4's row) | The bridge must sanitise the 12 refused and 20 colliding names; Layer A cannot ride the per-program child under the 512 MB cap, so W4 picks a warm session or a per-query slice; provenance text belongs outside the query space (9 GB with it, 4 GB without). |
| 7 staging | Confirmed: hyperon never sees Layer A. Staged Layer B entries in the PeTTa-only scoped stack are unaffected by these limits (Layer B is small). |
| 8 cost | Unchanged in dollars; full-text stage 1 also needs a different storage shape (about 48 GB in one space, extrapolated). |

Commands (from this work tree; `$SP` is the session scratchpad):

```
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/w3/harness/runner.py all --sp $SP --runs $SP/w3/runs_w1
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/w3/harness/runner.py all --sp $SP --runs $SP/w3/runs_terra \
  --exemplar $SP/w3/real/exemplar_terra.metta
# the N=1,000 real-density step after the harness fix, twice (A and B), on runs_terra/A/data:
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 TMPDIR=$SP/w3/runs_terra_fix/A/tmp \
  PETTA_HOME=/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa PETTA_MEMORY_LIMIT_MB=16384 PETTA_STACK_LIMIT_MB=2048 \
  PLN_RUNTIME=hyperon PLN_CHAINER=legacy /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B \
  $SP/w3/harness/measure_petta.py --mode layer --data $SP/w3/runs_terra/A/data --n 1000 --seed 20261026 --out $SP/w3/runs_terra_fix/A/petta
```
