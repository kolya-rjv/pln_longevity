# Atomspace expansion: handoff

Snapshot taken 2026-11-03, 21:30 local time (+04). A paid parse was still running in the background at that time.
Everything here was checked against the files listed. Anything not checked is marked **unverified**.

## 1. What this is

Ben Goertzel asked on 2026-10-25 for an agentic loop that grows the app's Atomspace from papers. Stage 1 seeds it:
an LLM recommends papers per query, and leungmanhin/semantic-parser parses them. Stage 2 learns from use: run a
query, show an LLM the PLN reasoning trace, ask what knowledge is missing, find papers per gap, parse them in.
A design phase (W1 to W5) runs from a launch brief and stops at an owner sign-off gate. Nothing is built yet.
The design keeps two layers apart. **Layer A** is the semantic parser's open-vocabulary parse of every sentence.
Its truth values are invented by the LLM, so it is background only and never feeds a calibrated answer. **Layer B**
is the repo's own expander output (`pln_chat/ontology/expander.py`): only facts in the app's fixed vocabulary,
checked by the schema gate, with confidence taken from the evidence tier, never from the LLM.

## 2. Where everything lives

`$SP` below means a scratchpad root: the durable copy or the live one (see the first two rows).

| Location | What is there | Safe? |
|---|---|---|
| `/Users/kolya/pln_atomspace_work/scratchpad/` | **Durable copy of the scratchpad. Use this one.** Same layout as the live copy, minus `dryrun/venv/`, `__pycache__/` and W3's regenerable synthetic data. Its parse results are a snapshot from about 21:26 (about 557 sentences); refresh it when the parse ends (§4.5). | durable |
| `/private/tmp/claude-501/-Users-kolya-PycharmProjects-pln-longevity--claude-worktrees-atomspace-expansion-launch-edebbf/99db15ee-a283-4eee-b551-0f75f0c2d357/scratchpad/` | Live scratchpad. The running parse writes here. Holds the parser venv (`dryrun/venv/`). | **at risk**: macOS cleans `/tmp`, and it already deleted the W1 fact sheets and W2 scratch results once. Only `w1/exemplar.metta` survives from those. |
| This work tree: `/Users/kolya/PycharmProjects/pln_longevity/.claude/worktrees/atomspace-expansion-launch-edebbf` | Branch `claude/atomspace-expansion-design`, from `petta-migration` (d21ae14). Commit `d390b80` holds `docs/atomspace_expansion/LAUNCH.md`, `LAUNCH_ULTRACODE.md` and the five design notes. The home of the design docs. | committed, **not pushed** |
| Branch `claude/gpt6-temperature`, commit `1a5fc78` | Temperature 1 by default; gpt-6.1-terra, gpt-6.1-sol, gpt-6-luna in both UI model lists (`api.py`, `app.py`, `config.py`). Not on the design branch. | committed, **not pushed** |
| `/Users/kolya/PycharmProjects/pln_longevity/.claude/worktrees/atomspace-expansion-workflow-2cc380/docs/atomspace_expansion/` | Uncommitted originals of the two launch briefs (identical to the committed copies). A hook forbids writing in that work tree. | read only |
| `/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa` | The PeTTa checkout every PeTTa run uses (`PETTA_HOME`). | outside this work |
| `/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python` | Project venv (hyperon 0.2.10, janus-swi). Runs the expander, trace, gap and Layer A scripts. | do not install into it |
| `/Users/kolya/PycharmProjects/pln_longevity/pln_chat/.env` | The OpenAI key. `$SP/dryrun/ledger.py` (`ENV_FILE`) reads it into the process only. Never open or print it. | owner's file |
| Session transcript (10 MB JSONL, grep only): `/Users/kolya/.claude/projects/-Users-kolya-PycharmProjects-pln-longevity--claude-worktrees-atomspace-expansion-launch-edebbf/99db15ee-a283-4eee-b551-0f75f0c2d357.jsonl` | Owner messages are `"type":"user"` lines with plain-text content. The showcase answers are in an AskUserQuestion result at 2026-11-03T16:39Z. | durable |
| Memory: `/Users/kolya/.claude/projects/-Users-kolya-PycharmProjects-pln-longevity/memory/` | `stakeholder-ben-atomspace-expansion.md` (this work), the `feedback-*.md` rules. That note's branch facts are stale (it says the temperature fix is uncommitted). | durable |

Inside `$SP`: `showcase/bin/` (scripts), `showcase/data/` (PubMed and PMC XML, untrusted), `showcase/out/` (all
outputs, below), `dryrun/` (dry-run scripts, `ledger.py`, outputs), `w3/harness/` (W3 harness; `layer_a.py` imports
its renderer), `semparser_src/repo` (the parser at `d00331d`, unmodified, `git status` clean).
In `showcase/out/`: `papers.json` (19 fetched papers; 10 used), `parse/<pmid>/results.json` + `meta.json`,
`ledgers/*.jsonl`, `expand/results_gpt-6.1-terra.json`, `layer_b/` (`staged.metta`, `summary.json`,
`whatif_bridge.metta`), `trace/trace_*.json`, `gap/result_after_petta.json`, `layer_a/` (a test on partial data).

## 3. What is done

| Phase | Date | Headline results | Note |
|---|---|---|---|
| W1 ground truth | 2026-10-25/26 | 36 agents checked the brief against the code. 16 contradictions. The biggest: a custom file named like a stack file runs in place of the curated one; the parser's number format fails its own check; no shipped script keeps `SentenceLink`s. | [design/W1_ground_truth.md](design/W1_ground_truth.md) |
| W2 checkpoint | 2026-10-26 | Draft slate of 10 queries (not signed off), cost model, draft rows 2, 3, 6, 7. | [design/W2_checkpoint.md](design/W2_checkpoint.md) |
| Paid dry run | 2026-11-02 | $1.83 of a $5 cap. Model picks: recall gpt-6.1-sol medium; parser and expander gpt-6.1-terra; gap call gpt-6.1-sol high. gpt-5.4-mini invented 8 of 14 references. Stage 1 projected at about $55 on abstracts, about $961 on full text. Every gpt-6 model refuses a temperature other than 1. | [design/dryrun_2026-11-02.md](design/dryrun_2026-11-02.md) |
| W3 scaling | 2026-11-03 | No API calls. PeTTa holds 1,000 synthetic real-density abstracts (19 MB, 16 s load, 4.1 GB). hyperon aborts after 1 to 3 abstracts, so Layer A is PeTTa-only. 47% of statements never render to patham9's Implication shape. 12 names make PeTTa refuse a whole KB. | [design/W3_scaling.md](design/W3_scaling.md), `W3_scaling_summary.json` |
| Showcase | 2026-11-03, running | §4 | not written yet |

Not started: row 4 stances, W4, W5 (`DESIGN.md`), the gate. The draft rows live in the W2 and dry-run notes.

## 4. The showcase

**Owner decisions (2026-11-03):** "Yes, cap $50"; topic rapamycin → GrimAge (slate query 1); "Patch the check in
our script" (the parser's number check), without editing the parser's files. Models: gpt-6.1-terra at medium for the
parser; gpt-6.1-terra for the expander (the script sends no effort, so the model default applies); gpt-6.1-sol at high
for the gap call. Temperature 1: the expander sends 1.0; the parser and gap calls send none (API default 1).

### 4.1 Status at 21:28

- Done: paper choice and fetch, Layer B, before and after traces, the gap call, the what-if.
- Running: the parser on 10 papers in parallel, started 20:53 under `caffeinate`, PIDs 89853 and up.
  589 of 1,940 sentences done, all `ok`. ETA about 00:00 to 00:30 on 2026-11-04 (my estimate from about 0.55 min
  per sentence on the 348-sentence paper).
- Spend so far: **$10.94** of $50 (parse $10.00, expander $0.92, gap $0.02), from `showcase/out/ledgers/`.
  Projected parse total about $33 at $0.017 per sentence (estimate). Dry run plus showcase: about $12.77 so far.

### 4.2 The papers

Ten PMC full texts, all open for text mining (CC BY, CC BY-NC, CC BY-NC-SA, or NIH author manuscripts).
Nine came from the dry run's stage-1 search. Correia-Melo 2016 (PMID 26848154) came from a PubMed search for the
missing mTOR → mitochondria link. 1,969 sentences in all: 1,940 parsed plus 29 in skipped admin sections.
PMIDs: 21258367, 21205641, 26848154, 24231806, 26463265, 26867126, 27986797, 21536998, 30669119, 31136303.

### 4.3 Results so far

| Step | Result |
|---|---|
| Parser patches (at import, `parse_paper.py`) | `stmt_format_check` accepts PointMass, ParticleFromNormal, ParticleFromPairs, NatDist, FloatDist; the predicate checks treat them as built-ins; "Current time" = publication date, noon UTC. |
| Numbers kept (`layer_a.py` `number_survival`, partial data: 81 sentences) | 80 of 97 (82.5%). Same measure on the dry run's unpatched parses: luna full text 25 of 56 (44.6%), terra abstracts 9 of 16 (56%). |
| Layer B (`expand_papers.py`, 54 parts of up to 9,000 chars) | 215 staged entries (158 Effect, 33 publication, 20 type, 4 fact). 2 duplicates of the KB. 4 refused, all `missing_identifier` (no PMID or DOI), all from 31136303. $0.92. |
| Before (`trace.py`, 25-file runtime stack) | `explain Rapamycin DNAmGDF15` empty on hyperon and on PeTTa. Of the chain's links, only `MitochondrialDysfunction → DNAmGDF15` answers (Pos, stv 0.75 0.65, curated). |
| After (PeTTa, stack + `staged.metta`) | 5 chains. Best: Rapamycin → SASP → ChronicInflammation → DNAmGDF15, Neg, stv 0.39 0.17. Its first link is Correia-Melo's result (rapamycin-fed mice, less SASP in liver). `Rapamycin → MTORC1` now answers (Neg, stv 0.5 0.35). `Rapamycin → MitochondrialDysfunction` and `explain Rapamycin GrimAge` stay empty. |
| After on hyperon | aborts at `trie.rs:179` with the staged entries; no trace file written. |
| Gap call (`gap2.py`, after_petta) | 3 `missing_edge` gaps: MTORC1 → Mitophagy (58 PubMed hits), Mitophagy → MitochondrialDysfunction (208), Rapamycin → GrimAge (0; no human trial). $0.018. |
| What-if (`whatif_bridge.metta`) | One edge, ULK1Activity → MitochondrialDysfunction (21205641), restated on `ULK1` as a name bridge would. It adds a sixth chain: Rapamycin → MTORC1 → ULK1 → MitochondrialDysfunction → DNAmGDF15 (Neg, stv 0.21 0.02). |
| Naming | Layer B names one thing several ways: MTORC1, Mtorc1, MTORC1Activity, MTORC1Inhibition, MTOR, MTORSignaling; ULK1, Ulk1, ULK1Activity. GDF15 (protein) and DNAmGDF15 are separate nodes. |
| Layer A test (partial, 20:58) | 777 statements, 351 predicates, 164 links render to the library shape. Loads in PeTTa in 0.31 s. A concept named `test` appeared; PeTTa's KB scan refuses a KB holding it, so `layer_a.py` renames such tokens to `<name>_c` (test 5, length 1). |

All traces used the default legacy chainer (`PLN_CHAINER` unset) on the PeTTa runtime.

### 4.4 How the scripts find things

Each script has a usage line in its docstring. Most paths are relative to the script's own folder, so the scripts
run from either copy. The exceptions:
- `showcase/bin/run_all.sh` line 3 hard-codes `SP=` to the `/tmp` scratchpad. Edit it to run from the durable copy.
- `REPO` in `expand_papers.py`, `trace.py`, `gap2.py`, `layer_a.py` points at this work tree (imports `pln_chat`).
  The tree is now on `d390b80` (d21ae14 code), not `1a5fc78`. That is harmless: the expander gets
  `temperature=1.0` explicitly, and nothing else uses the changed lines.
- `dryrun/ledger.py` `ENV_FILE` points at the main checkout's `pln_chat/.env`. If `OPENAI_API_KEY` is already
  set in the environment, that is used instead.
- `run_all.sh` and `parse_paper.py` need the parser venv `$SP/dryrun/venv`, which the durable copy lacks. Rebuild:
  `/Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -m venv $SP/dryrun/venv && $SP/dryrun/venv/bin/pip install -q faiss-cpu==1.13.2 networkx==3.6.1 numpy==2.4.2 pydantic==2.12.5 sexpdata==1.0.2 hyperon==0.2.10 openai==3.42.1` (the versions in the live venv, checked
  with `pip freeze`)

### 4.5 How to resume each step

Set `SP` first. Run every command from `$SP/showcase/out`. Keep the env prefixes: they unset keys where no LLM is
needed and stop `load_dotenv`.

1. **Check the parse.** `ps -axo pid,etime,command | grep parse_paper | grep -v grep` and
   `tail -qn1 $SP/showcase/out/logs/parse_*.err` (a finished paper prints `done paper_spent=...`). Never start a
   second copy while one runs. If it died, rerun `run_all.sh` (paid; the shared ledger caps all showcase spend at
   $44 for the parse). `parse_paper.py` resumes: it skips sentences already `ok` and retries `skipped_budget` and
   `fatal_api` ones. Original launch: `caffeinate -i $SP/showcase/bin/run_all.sh` (background), which runs per paper
   `env -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 $SP/dryrun/venv/bin/python -I -B $SP/showcase/bin/parse_paper.py <pmid> gpt-6.1-terra medium 44 20000`.
2. **Refresh the durable copy** (free; run again after every later step):
   `rsync -a --exclude 'dryrun/venv/' --exclude '__pycache__/' --exclude 'w3/runs_*/*/data/' --exclude 'w3/runs_*/*/data_sweep/' --exclude 'w3/runs_*/*/tmp/' --exclude 'w3/smoke/*/data*/' --exclude 'w3/smoke/*/tmp/' --exclude 'w3/gencheck/' --exclude 'w3/*/*/hyperon/hyperon_home/' --exclude 'w3/*/*/hyperon/prefix/' --exclude 'w3/*/*/hyperon/children/' <live SP>/ /Users/kolya/pln_atomspace_work/scratchpad/`
3. **Layer A on the full parses** (free, PeTTa only, overwrites `out/layer_a/`):
   `env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 PLN_RUNTIME=petta PETTA_HOME=/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/showcase/bin/layer_a.py`
   It sets `PETTA_MEMORY_LIMIT_MB=8192` itself. If the load overflows the 2 GB Prolog stack, add
   `PETTA_STACK_LIMIT_MB=8192` (W3 needed that only at 1,000 abstracts; unverified for this size).
4. **Reruns, only if needed.** Layer B staging (free): `/usr/bin/python3 -I -B $SP/showcase/bin/stage_b.py`.
   Traces (free): `env -u OPENAI_API_KEY -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 PLN_RUNTIME=petta PETTA_HOME=/Users/kolya/PycharmProjects/pln_longevity_petta/.petta/PeTTa /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/showcase/bin/trace.py <label> [$SP/showcase/out/layer_b/staged.metta ...]`.
   Use `PLN_RUNTIME=petta` whenever staged files are passed. Gap call:
   `env -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 PLN_RUNTIME=petta /Users/kolya/PycharmProjects/pln_longevity/.venv/bin/python -I -B $SP/showcase/bin/gap2.py after_petta gpt-6.1-sol high 48`.
   It reuses its cached answer (`out/gap/raw_S2_<label>_<model>_<effort>.json`), so a rerun of the same label is free.
   A new label is a paid call. Expander (paid, already done): `expand_papers.py gpt-6.1-terra 46`, same prefix as
   the gap call; it resumes per part.
5. **Write the showcase evidence note** in `docs/atomspace_expansion/design/` (suggested name
   `showcase_2026-11-03.md`), in the style of the dry-run note: owner decisions, papers, patches, Layer A numbers
   from step 3, Layer B, before and after traces, gaps, what-if, naming, spend, what it changes in rows 4, 5 and 7.
6. **Build the shareable showcase page** as a plain HTML Artifact (load the `artifact-design` skill first).
   Audience: the owner and Ben. Show the before and after chains, the what-if, the gaps, the two layers, the cost.

## 5. Open owner decisions

| Decision | Options and context |
|---|---|
| The six repo tests that abort hyperon on purpose (`test_human_evidence.py`, `test_human_evidence_route.py`, `test_linage2.py`, `test_nhanes_integration.py`, `test_patient_stack.py`, `test_smoking_lever.py`) | They cause macOS "Python quit unexpectedly" pop-ups. Either patch them on their own branch (quiet exit code 6, tests expect 6), or the owner silences the dialog themselves (`defaults write com.apple.CrashReporter DialogType none`). We never change that setting. |
| Cherry-pick `1a5fc78` onto `linage2` and/or `petta-migration` | Touches the main checkout or its branch; needs the owner's OK. |
| Contact Man Hin Leung with two upstream parser fixes | (1) the prompt asks for PointMass numbers that `stmt_format_check` rejects; (2) parses are anchored to parse time. Proposal: no fork, keep the parser pinned and wrapped. |
| Duplicated key name in `pln_chat/.env` (main checkout) | The key line repeats `OPENAI_API_KEY=`. The owner fixes it themselves. `ledger.py` tolerates it; the app may not. |
| A second stage-2 round | Parse papers for the two gaps that have literature (58 and 208 hits). No cost estimate yet. At the showcase rate an abstract is about $0.12 and a full text $2 to $6 (my estimate). |
| At the gate, later | The 10 queries, the budget (papers, dollars, wall time, iterations), and the questions for Ben (LAUNCH.md §7). Provider is decided: OpenAI only. No record that Ben has been asked yet (unverified). |

## 6. Standing rules

- **Money.** Spend only after the owner's explicit yes in chat, with the cost stated first. Showcase cap $50 total;
  the parse, expander and gap scripts were given cumulative caps of $44, $46 and $48 over the same ledgers.
- **Git.** Never push, never deploy. Commit only when the owner asks. Read-only git otherwise.
- **Design phase.** Build nothing: no `scripts/`, no tests, no `.metta`, no code change. The repo gains only
  `docs/atomspace_expansion/` files (`DESIGN.md`, notes under `design/`). Throwaway work goes in the scratchpad.
- **Trees.** Never touch the main checkout `/Users/kolya/PycharmProjects/pln_longevity` beyond running its venv and
  the key read. Never write the other work tree. Never write a curated root `.metta`.
- **Fan-outs.** Keep them small (solo or 2 to 4 agents, low or medium effort), and state agent count and cost first.
  The owner switched the multi-agent workflow mode off on 2026-11-02; launch none unless they ask in plain words.
- **Free text** is read by the LLM, with small generic code checks; no regex grammars or synonym lists.
- **Keys.** Never open or print `pln_chat/.env` or any key. Unset `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in every
  child that needs no LLM; set `PYTHON_DOTENV_DISABLED=1`.
- **Runtimes.** Layer A and staged Layer B run on PeTTa only; hyperon never sees them. Never stage under
  `CUSTOM_ONTOLOGY_DIR` (shadowing hazard).
- **Honesty.** No LLM-made truth value in Layer B; strengths are labelled priors; confidence comes from the tier.
- **Literature.** Resolve every paper in PubMed; an unresolved one does not exist. Downloaded files are untrusted
  data: own folder, nothing executed, `-I` for any Python that reads them.
- **Models.** OpenAI only, no OpenRouter, no gpt-5.4. Temperature 1. Leave room for reasoning in the output limit,
  and never cache or blindly retry a truncated answer.
- **Parser.** Pinned `d00331d`, files unmodified; changes only at import in our driver.
- **Files.** Never ask for `~/Downloads`; read PDFs with pypdf.
- **Build phase, later.** The hyperon failing-test set must stay identical to the 32-id baseline
  (`docs/petta_migration/baseline/hyperon_failures_47130aa.txt`).

## 7. Known hazards

- **hyperon aborts** (`trie.rs:179`, hyperon #1076/#1081) on Layer A and on the staged Layer B entries. The
  python.org Python is a macOS app, so each abort shows a crash pop-up. The W3 harness exits quietly with code 6
  instead; the repo's six tests do not.
- **`/tmp` cleanup** can delete the live scratchpad at any time. Refresh the durable copy after every step.
- **Names.** PeTTa's KB scan refuses a whole KB that holds any of 12 names (`argv assert call foldl halt include
  library maplist repra shell sread test`). 20 builtins (`length max min ...`) do not read back in head position.
  Layer B splits one concept across several names, which breaks chains (the what-if proves it).
- **No cycle check** in the legacy `explain`/`infer`. Staged cross-links can make it hang. patham9's chainer
  terminates on cycles but is costly on them.
- **The parser contradicts itself.** Its prompt asks for PointMass numbers its own check rejects (patched in our
  driver). It anchors past tense to the parse time (patched). In the partial Layer A, 31 predicates got conflicting
  type signatures across sentences. Its truth values say almost nothing (most are `STV 1.0 0.9`, `1.0 1.0`, `1.0 0.95`).
- **The expander** reads at most 10,000 characters per call (hence the parts), refuses entries with no PMID or DOI,
  and mints nodes no rule reads.
- **Memory.** The repo's per-program PeTTa child under the 512 MB cap cannot carry a large Layer A (W3).

## 8. Next steps, in order

1. Wait for the parse to finish (§4.5 step 1). Then refresh the durable copy.
2. Run `layer_a.py` on the full parses. Refresh the durable copy again.
3. Write the showcase evidence note in `docs/atomspace_expansion/design/`.
4. Build the showcase page (HTML Artifact) and give the owner the link.
5. Report to the owner: final spend, results, and the open decisions in §5.
6. Update the memory note `stakeholder-ben-atomspace-expansion.md` (branch facts, showcase results).
7. Row 4: weigh the three stances (patham9's PLN reads Layer A directly / Layer B only / hybrid) with the real
   Layer A numbers. Then W4 (the bridge from Layer A to Layer B, names and tiers, trace and record formats).
8. W5: write `DESIGN.md` and verify it. Agent panels need the owner's OK first (§6).
9. Stop at the gate and present (LAUNCH_ULTRACODE.md, "The gate"). The build starts only after sign-off.

## 9. Not verified

- The parse's end time and final cost (estimates in §4.1).
- gpt-6.1-terra's default effort, which the expander used.
- Whether `layer_a.py` loads the full Layer A within the default 2 GB stack.
- Whether the §7 questions were ever sent to Ben.
