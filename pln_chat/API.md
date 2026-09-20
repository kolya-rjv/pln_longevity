# PLN Query API

A plain HTTP/JSON API for `pln_chat`, meant for scripts and agents to call
directly instead of driving the Gradio UI through a browser. The normal launch
mounts it and Gradio on the same server and public origin. It wraps the
exact same pipeline as the "PLN Query" and "Ontology Expander" tabs in
`app.py` — same `translate -> validate -> run_query -> format` logic, same
`.metta` files — just returned as structured JSON instead of chat HTML.

The KB is now a curated inference stack — calibration, deduction, abductive
diagnosis, intervention ranking, patient grounding, counterfactual analysis,
risk prediction, supplement recommendations — plus a scoped DrugAge
lifespan-ranking engine. See "Demo query forms" below for what's reachable
and how.

## Run it

```bash
cd pln_chat
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY

python app.py
# UI:               http://localhost:7860/
# interactive docs: http://localhost:7860/docs
# raw OpenAPI schema (useful for pointing an agent at the API's shape):
#   http://localhost:7860/openapi.json
```

Equivalent importable combined app:
`uvicorn server:app --host 127.0.0.1 --port 7860`

Point ngrok at that one listener:

```bash
ngrok http 7860
```

The resulting origin serves Gradio at `/` and the JSON API at `/query`,
`/metta/run`, `/patients`, etc. For an API-only process, `python api.py` still
listens on port 8000 by default.

There is no authentication layer. The service is intended to run only in the
private environment where the Gradio UI and invited agents can already reach
it.

## PLN execution runs in a worker process

hyperon 0.2.10 holds the GIL for the whole of `MeTTa.run()` — measured, two
MeTTa runs in two Python threads take exactly as long as two runs in sequence
(ratio 0.998), and a canary thread gets 3 of ~386 expected ticks during one
2.3-second query. So while a query runs, the event loop and every other caller
is starved: a `GET /health` issued 0.4 s into a 3.9 s request took 3.5 s, and
one 35-compound ranking blocked the whole service for 115 s. No threadpool
size, `async def` conversion or asyncio timeout can fix that — you cannot
preempt a Rust call that holds the GIL.

MeTTa therefore runs in a `ProcessPoolExecutor` (`pln_chat/core/executor.py`).
Three things follow:

* **The API stays responsive.** `/health` during a 3.7 s hyperon call: 0.007 s.
* **Deadlines are enforceable.** A query past `PLN_QUERY_TIMEOUT_SECONDS`
  (default 60) returns **504** `pln_timeout`, and the worker is *killed* — an
  abandoned divergent recursion pins a core and grows without bound (measured
  3.7 GB within two minutes), so a signal is the only way to stop it.
* **A hyperon abort no longer takes the API down.** hyperon aborts the
  interpreter with a non-unwinding Rust panic on some query shapes past a few
  hundred rows; `except Exception` cannot catch it. Out of process it is a
  **500** `pln_worker_crashed` and the service keeps serving.

When more than `PLN_MAX_INFLIGHT_QUERIES` PLN tasks are queued or running, new
ones get **503** `pln_overloaded` with `Retry-After` rather than piling up
behind a deadline they cannot meet. `GET /health` reports the live picture
under `pln_execution`.

| env var                     | default | meaning                                        |
|-----------------------------|---------|------------------------------------------------|
| `PLN_WORKER_POOL_SIZE`      | 2       | worker processes; **0 runs inline**, as before  |
| `PLN_QUERY_TIMEOUT_SECONDS` | 60      | per-request PLN budget; 0 disables              |
| `PLN_MAX_INFLIGHT_QUERIES`  | 0       | admission limit; 0 derives 4x the worker count  |
| `PLN_WORKER_MAX_TASKS`      | 50      | recycle workers after N tasks; 0 never          |
| `PLN_MAX_RANK_COMPOUNDS`    | 60      | cap on `/drugage/rank`'s compound list          |
| `PLN_MAX_ONTOLOGY_FILES`    | 64      | cap on an `ontology_files` selection (deduped)  |

Because the Gradio UI is mounted on the same ASGI app and drives the same
pipeline, this also stops a UI query from freezing the REST API and vice versa.

OpenAI calls have a 60-second timeout and one SDK retry by default; configure
`OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` as needed. Every HTTP request,
its raw body, each raw chat prompt, and each translated MeTTa query are written
to `pln_chat/logs/session_*.jsonl`. For browser clients, set
`PLN_CORS_ORIGINS` to a comma-separated origin allowlist.

## Endpoints

| Method | Path               | Purpose                                                    |
|--------|--------------------|--------------------------------------------------------------|
| GET    | `/health`          | Liveness + readiness check (PLN, OpenAI, KB, and DrugAge build) |
| GET    | `/ontology/files`  | List discovered `.metta` files (+ default selection, + which are excluded from execution) |
| GET    | `/kb/schema`       | What the KB actually holds: predicates, arities, fact counts, and which predicates are declared but empty |
| GET    | `/patients`        | List known patient profiles (for the `<Patient>` query forms below) |
| POST   | `/query`           | Ask a natural-language question of the KB (goes through the LLM translator) |
| POST   | `/metta/run`       | Validate + execute a raw MeTTa query directly (no LLM call)  |
| POST   | `/drugage/rank`    | Rank real DrugAge compounds by lifespan/mortality effect, no MeTTa needed |
| POST   | `/ontology/expand` | Extract new ontology entries from pasted paper text          |
| POST   | `/ontology/apply`  | Write a previously-previewed MeTTa block to disk             |

Full request/response schemas are in `/docs` and `/openapi.json` once the
server is running.

Invalid model names, ontology selections, MeTTa, and unsafe ontology target
filenames return HTTP 422 before paid inference, PLN execution, or disk writes.

### Failures are HTTP failures

**This is a behaviour change.** A translator or runtime failure used to come
back as HTTP 200 with `intent: "clarification"`, `validation_valid: true`,
`pln_status: "empty"` and the upstream error text buried in `answer` — the
exact shape of a successful-but-empty answer, so a client filtering on status
never noticed. Every failure now carries its own status and a structured
`detail` with a machine-readable `code`:

| code                      | status | meaning                                           |
|---------------------------|--------|---------------------------------------------------|
| `prompt_too_large`        | 413    | the assembled prompt exceeds `PLN_MAX_PROMPT_TOKENS` — **checked before the call is made** |
| `context_length_exceeded` | 413    | upstream rejected the prompt anyway                |
| `rate_limit`              | 429    | upstream throttling (`Retry-After` is set)         |
| `timeout`                 | 504    | upstream did not answer in time                    |
| `pln_timeout`             | 504    | PLN execution passed its per-request budget        |
| `connection`              | 502    | could not reach upstream                           |
| `upstream_error`          | 502    | any other OpenAI-side failure                      |
| `bad_json`                | 502    | the model answered with something unparseable      |
| `runtime_error`           | 502    | the hyperon interpreter raised                     |
| `pln_worker_crashed`      | 500    | the MeTTa worker process died (hyperon abort)      |
| `missing_api_key` / `auth`| 503    | the service has no usable OpenAI credential        |
| `drugage_build_missing`   | 503    | `build/drugage_etl.metta` has not been generated   |
| `pln_overloaded`          | 503    | too many PLN tasks in flight (`Retry-After` is set)|

Nothing is lost: `detail` carries the original message, the `stage` it failed
at, and — for a translation failure — the token `usage` that was already
spent.

The **413 `prompt_too_large`** guard is the one that removes a whole class of
wasted calls. The default `/query` prompt pastes the curated `.metta` files
verbatim and already measures ~62,000 tokens; selecting a gene ETL file pushed
it to 417,000 and came back as a billed upstream 400. The size is now estimated
first (characters / `PLN_CHARS_PER_TOKEN`), the request is refused with the
estimate, the limit and the largest selected files named, and every successful
response reports its own `prompt_tokens_estimate`.

A failed translation also sets `intent: "error"` rather than borrowing
`clarification`, which is a real answer the engine gives when a question needs
narrowing.

### Valid, and guaranteed to return nothing

The ontology DECLARES a much larger vocabulary than it populates.
`logical_predicates.metta` declares `TargetsHallmark`, `Causes`, `Predicts`,
`Extends`, the gene predicates and the DrugAge row predicates; the generic
runtime space holds **zero** facts for any of them. A query over one of those
is perfectly well-formed MeTTa and returns nothing — which reads exactly like
"the answer is no". The translator emitted `TargetsHallmark` three times in the
2026-09-18 evaluation, and `UsesIntervention` / `AvgLifespanChangePercent`
three more.

Two changes remove that failure mode:

* `GET /kb/schema` reports the truth — every grounded predicate with its arity,
  fact count and source files, plus `declared_but_empty_predicates`. It is also
  the answer to "what data do you have and how much of it".
* every `/query` and `/metta/run` response carries `ungrounded_predicates` and
  `validation_warnings`. An empty `pln_results` next to a non-empty
  `ungrounded_predicates` means *the KB cannot express this relation*, not *no*.

The same inventory now backs the validator and the LLM's system prompt. That
also fixes the **opposite** bug, which was quietly worse: the symbol registry
only ever harvested type declarations, `Inheritance` and `InstanceOf`, so 71
symbols that genuinely exist in the KB — `MTORC1`, `AMPK`, `SIRT1`, `Mouse`,
`Human`, `CDKN2A_P16` — were reported as unknown, and `/metta/run` answered 422
to queries the engine would have served. Validation is now measured against the
actual ground atoms. (A p-value written as `2.0e-75` also no longer contributes
two imaginary "unknown symbols".)

**A note on KB size:** hyperon 0.2.10 panics once a space gets too large, so
any `.metta` file over `PLN_MAX_KB_FILE_BYTES` (default 60 KB — currently
just `drugage_etl_short.metta`) is excluded from execution (`run_query`,
`/metta/run`'s default validation) but still listed by `/ontology/files`
under `excluded_from_runtime`. It's still queryable in stub mode (no
`hyperon` installed / `PLN_RUNTIME_AVAILABLE=false`).

The same size question applies to the **prompt**, and used to be fatal there.
The LLM context pasted every selected file verbatim; selecting a CellAge or
GenAge ETL file took the prompt to 417,000 tokens and the call came back as a
billed upstream 400. A selected file over `PLN_PROMPT_FILE_MAX_BYTES`
(default 25 KB) is now replaced by its **schema card** — `cellage_genes.metta`
goes from ~131,000 tokens to ~215, and says which predicates it holds and how
many facts each has, which is more useful to a translator than the rows are.
Every hand-written layer in this repo is under that limit and is still pasted
verbatim, because its prose is what the translator reasons from. The default
prompt also drops from ~62,400 to ~57,400 tokens, because the grounded schema
card replaces a flat ~7,000-token symbol index in which a declaration and 400
facts looked identical.

## Examples

```bash
curl localhost:7860/health

curl localhost:7860/ontology/files

curl localhost:7860/patients

curl -X POST localhost:7860/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "What interventions might help reduce GrimAge acceleration?"}'
```

Multi-turn: pass the `history` array from a response back into the next
request's `history` field to keep the conversation going.

Ask about a specific patient (see `GET /patients` for valid IDs) — this
routes through the same dedicated MeTTa forms listed below:

```bash
curl -X POST localhost:7860/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is Patient001'"'"'s 10-year CHD risk, and what drives it?"}'
```

If you already know the MeTTa you want to run — e.g. an agent iterating on
queries directly — skip the LLM translator with `/metta/run`:

```bash
curl -X POST localhost:7860/metta/run \
  -H 'Content-Type: application/json' \
  -d '{"metta_query": "!(predict-risk-patient &self Patient001)"}'
```

This only runs `validate` + `run_query` (no OpenAI call), so it's free and
instant. `ontology_files` (optional) scopes symbol validation; execution
always runs against the same runtime-safe file set either way.

Rank real compounds by lifespan/mortality effect — either ask in natural
language (`/query` detects a "rank X, Y by lifespan" question and routes it
automatically — see `routed` in the response) or call the dedicated endpoint
directly:

```bash
curl -X POST localhost:7860/drugage/rank \
  -H 'Content-Type: application/json' \
  -d '{"compounds": ["Rapamycin", "Metformin", "Resveratrol"]}'
```

This requires `build/drugage_etl.metta` to exist (`bash scripts/run_etl.sh`
generates it) — check `GET /health`'s `drugage_build_available` first. As of
this writing the engine does **not** fall back to the smaller committed
sample (`drugage_etl_short.metta`) when the build is missing; it returns a
clear `error` instead.

**Compound names are resolved, not just normalised.** DrugAge stores one
spelling per compound, so `sirolimus`, `NMN`, `NAD+`, `EGCG`, `NAC` and every
spelling of `17-alpha-estradiol` used to match nothing and vanish from the
ranking. Each requested name now goes through a shared resolver
(`pln_chat/ontology/compound_names.py`) and the response carries a
`resolutions` array saying what happened to it:

| `method`       | meaning                                                          |
|----------------|------------------------------------------------------------------|
| `exact`        | the string is a DrugAge symbol verbatim                           |
| `normalized`   | case / separator / unicode / Greek-letter difference only         |
| `synonym`      | a curated chemical identity (`sirolimus` is rapamycin)            |
| `etl_artifact` | the ETL prefixes a symbol starting with a digit with `N_`         |
| `fuzzy`        | an unambiguous near-miss accepted as a typo, with a warning       |
| `ambiguous`    | several candidates — **nothing is ranked**, candidates returned   |
| `unmatched`    | no match — **nothing is ranked**, nearest names returned          |

```json
{"query": "sirolimus", "matched": "Rapamycin", "method": "synonym",
 "score": 1.0, "note": "INN for rapamycin (Rapamune)", "suggestions": []}
```

`ambiguous` and `unmatched` never resolve to a compound: ranking the wrong
substance is worse than omitting one. The same table is injected into the LLM
translator's system prompt, so `/query` and `/drugage/rank` agree on what a
name means instead of each guessing separately.

### The ranking response, as data

`results` (the MeTTa atoms) is unchanged, but nothing needs to parse it any
more:

| field           | what it is                                                            |
|-----------------|------------------------------------------------------------------------|
| `ranked`        | most protective first: compound, score, sign, direction, strength, confidence |
| `rows`          | **every** matching DrugAge row — species, **sex**, significance, change %, PMID |
| `unscorable`    | compounds whose row reports no average lifespan change (no score exists) |
| `filtered_out`  | compounds that scored but fell below `confidence_threshold`             |
| `semantics`     | what the numbers mean (below)                                           |
| `strategy`      | `linear` (default) or `metta_sort`                                      |
| `source`        | which DrugAge file the rows came from                                   |

**Sign convention: `Neg` is the good direction.** The lift is on the lifespan
axis (extending lifespan is `Pos`), then chained through the curated
`(Effect Lifespan Mortality Neg)` adapter, so a life-extender reads as `Neg`
— protective — on the mortality axis every other ranking in this KB uses.
`score` is `strength x confidence`, signed so that **higher is better**.

**Strength** is `|change%| / (|change%| + 20)` — saturating, so +20 % reads
0.50 and +80 % reads 0.80. **Confidence** is `min(evidence tier, significance
gate) x 0.9`, where the 0.9 is the per-hop chain discount for the
lifespan → mortality step. So the confidence you see is always 0.9 x the row's
tier: **0.81** ITP, **0.45** non-ITP vertebrate, **0.315** invertebrate,
**0.18** yeast. A score of exactly 0.0 is a reported null (metformin and
resveratrol are ITP negatives at confidence 0.81), never a missing value.

**One row per compound, and which one.** A compound usually has several rows —
rapamycin has 37, astaxanthin 6. The score uses one representative: the
highest evidence tier; within a tier a reported-significant result over an
unreported one over a reported null; remaining ties broken by the *median*
change, never the maximum. `rows` returns all of them so that choice is
auditable — astaxanthin's ITP study reports +12 % in males (significant) and
+3 % in females (not), and the response now shows both.

**`confidence_threshold` works.** A collapsed MeTTa result arrives as one atom
holding a tuple of entries, each with its own truth value; the filter used to
keep or drop the whole tuple on the leading entry's confidence. It now filters
entry by entry, on `/drugage/rank`, `/query` and `/metta/run` alike, and
`/query` reports the threshold it applied as `confidence_threshold_applied`
(`confidence_filter` is the LLM translator's suggestion, which is
informational only).

**Ranking is linear in the pool size.** `strategy: "linear"` (the default)
scores each compound separately and sorts in Python: ~70 ms per compound, and
each compound carries its own truth value. `strategy: "metta_sort"` is the
original single `rank-interventions` call, whose MeTTa insertion sort is
O(n^2) with a large constant — measured 1.1 s at n=5, 6.1 s at n=10, and the
115 s the evaluation saw at n=35. Both produce identical scores (asserted in
`tests/test_drugage_ranking_contract.py`); the compound list is capped at
`PLN_MAX_RANK_COMPOUNDS` (default 60).

```bash
curl -X POST localhost:7860/ontology/expand \
  -H 'Content-Type: application/json' \
  -d '{
        "paper_text": "<abstract or full text here>",
        "new_filename": "my_paper_extract",
        "apply": false
      }'
# review the returned metta_block, then either re-call with "apply": true,
# or POST it separately:
curl -X POST localhost:7860/ontology/apply \
  -H 'Content-Type: application/json' \
  -d '{"metta_block": "...", "target_file": "my_paper_extract.metta"}'
```

## Demo query forms

These map to dedicated MeTTa functions rather than hand-built patterns — the
LLM translator already knows to emit them for matching natural-language
questions (`/query`), or write them directly for `/metta/run`. `<Patient>` is
a known ID from `GET /patients` (currently `Patient001` / `Patient002`);
`<Lever>` is a cause (`ChronicInflammation`, `CellularSenescence`,
`InsulinResistance`), an intervention (`DasatinibPlusQuercetin`,
`Metformin`), or a marker (`CRP`).

| Ask (natural language)                                    | Dedicated form                                          |
|-------------------------------------------------------------|----------------------------------------------------------|
| "decompose/break down `<Patient>`'s GrimAge into components" | `(decompose-grimage &self <Patient>)`                    |
| "`<Component>`'s share of `<Patient>`'s GrimAge"             | `(grimage-share &self <Patient> <Component>)`             |
| "if `<Lever>` were normalized, expected change in GrimAge"   | `(counterfactual-patient &self <Patient> <Lever>)`         |
| "`<Patient>`'s 10-year CHD risk"                              | `(predict-risk-patient &self <Patient>)`                   |
| "what drives `<Patient>`'s CHD risk"                          | `(risk-decomposition-patient &self <Patient>)`             |
| "how much would `<Lever>` lower `<Patient>`'s CHD risk"       | `(project-risk-patient &self <Patient> <Lever>)`            |
| "what supplements should `<Patient>` take"                    | `(recommend-supplements-patient &self <Patient>)`           |
| "should `<Patient>` take `<Supplement>`"                       | `(supplement-for-patient &self <Patient> <Supplement>)`      |
| "rank omega3, fisetin and nmn for `<Patient>`"                | `(recommend-supplements &self <Patient> (<Supplement> …))`  |
| "rank rapamycin, metformin by lifespan benefit"               | `(rank-drugage-lifespan (<Compound1> <Compound2> …))` — or just call `POST /drugage/rank` |

A finding with no mechanistic path is omitted rather than invented; a
compound with a negative gold-standard trial (e.g. `Resveratrol`, ITP
negative) is still surfaced but flagged `NotRecommended`, never silently
dropped.

## Pointing an agent at it

Give the agent the base URL plus `/openapi.json` (or the `/docs` page) —
that's enough for most HTTP-capable agents to discover the endpoints and
call `/query` on their own. Good first calls: `/health` (confirms the server
is ready before it starts spending OpenAI calls) and `/patients` (valid
`<Patient>` IDs for the forms above).

This satisfies a local or otherwise network-reachable agent integration. It
does not itself provision a public URL, TLS, process supervision, rate limits,
or a reverse proxy; add those deployment controls before giving a remote agent
access. Keep the listener within the intended private network because endpoints
are intentionally unauthenticated.

## Testing

The complete test matrix and commands are in
[`docs/api_testing.md`](../docs/api_testing.md). The fast HTTP + combined-mount
suite is `pytest tests/test_api.py tests/test_combined_app.py -q`;
`scripts/test_pln_api.py` is the black-box runner for a live deployment.
