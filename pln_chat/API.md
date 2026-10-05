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

## Operational controls: keys, rate limit, version

**Both controls below are off by default.** Unset, the service behaves exactly
as it always has: no credential, no metering, every endpoint open. That is
still the intended shape for a private environment where the Gradio UI and
invited agents can already reach the listener. They exist so a deployment that
leaves that environment does not have to be rewritten first.

### Optional API key

```bash
# one shared secret...
PLN_API_KEY=some-shared-secret
# ...or one per consumer, so a single agent can be revoked on its own
PLN_API_KEYS=key-for-agent-a,key-for-agent-b
```

With either set, every request must carry the secret in `X-API-Key`:

```bash
curl -s localhost:7860/patients -H 'X-API-Key: some-shared-secret'
```

| behaviour                     | key unset (default)      | key set                                  |
|-------------------------------|--------------------------|------------------------------------------|
| an ordinary request           | 200                      | 401 `api_key_required` without the header |
| a wrong header value          | —                        | 401 `api_key_invalid`                     |
| `/openapi.json` `components`  | no `securitySchemes`     | `securitySchemes.PLNApiKey` (`apiKey`, header `X-API-Key`) |
| `/openapi.json` per-operation | no `security`            | `security: [{PLNApiKey: []}]` on everything but the open paths |
| `GET /health`                 | `api_key_required: false`| `api_key_required: true`                  |

The schema tracks the deployment rather than describing a fixed ideal, so an
agent pointed at `/openapi.json` discovers the header it needs — or discovers
it needs none.

**Always open, and never metered:** `/health`, `/docs`, `/redoc`,
`/openapi.json` and CORS preflights (`OPTIONS`). A readiness probe that needs a
secret reports the wrong thing when the secret is wrong; one that can be
rate-limited reports the service as *down* when it is merely busy (observed
against a live listener with the limit at 3/min — a six-call smoke test left
`GET /health` answering 429). An agent that cannot read the schema cannot find
out what it is missing, either. None of those routes touch the LLM, the
knowledge base or the disk, so exempting them costs nothing a limit would have
protected.

**Setting a key also locks the browser UI.** Gradio is mounted at `/` on the
same application and cannot send the header, so a combined deployment is either
open or API-only. That is deliberate: exempting the UI's own routes would leave
the whole pipeline reachable without a key and make "authentication" a word
rather than a control.

### Optional rate limit

```bash
PLN_RATE_LIMIT_PER_MINUTE=60   # per client address; 0 = off (the default)
```

A stdlib token bucket keyed on `request.client.host`, capacity one minute's
allowance, refilling continuously — so a burst of the full budget is allowed
once and the caller is metered after that. Over the limit:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
{"detail": {"code": "rate_limited", "limit_per_minute": 2,
            "retry_after_seconds": 30, "message": "…"}}
```

Only this API's own routes are metered, and not the always-open ones above;
Gradio's static assets and queue polls are not metered either (its mount is not
a FastAPI route). `slowapi` is deliberately not a dependency.

**This is a courtesy limit, not a security control** — the honesty contract is
in [`core/rate_limit.py`](core/rate_limit.py). It is per process, so two
uvicorn workers mean twice the configured rate. Behind ngrok or any reverse
proxy, `request.client.host` is the proxy, so every caller shares one bucket;
nothing here reads `X-Forwarded-For`, because trusting a client-settable header
would make the limit both evadable and abusable. It stops one enthusiastic
agent or a runaway retry loop, which is the failure mode the evaluation hit. It
does not stop an adversary.

### Versioned responses

Every response carries `X-API-Version`; `GET /health` reports the same string
as `version`, and so does `info.version` in `/openapi.json`. It is **2.0.0**,
and the major bump is not cosmetic: in 1.x an upstream outage, a hyperon
exception or an oversized prompt came back as HTTP 200 with
`intent: "clarification"`. They are 4xx/5xx with a machine-readable `code` now
(see "Failures are HTTP failures"), which breaks any client that branched on
the status code. The same release added the discovery endpoints, caller-
supplied patients and the two controls above.

`X-API-Version` and `Retry-After` are in the CORS `expose_headers` list, so a
browser client can actually read them.

### Prompt caching — what was already true

The evaluation asked for "prompt caching of the fixed system prompt". The
honest answer is that there is nothing to build here and a hand-rolled cache
would be the wrong thing: OpenAI applies automatic prefix caching to a stable
prompt prefix, and the system prompt **is** static for a given ontology
selection — the ontology snapshot, the schema card, the alias table and the
few-shot examples are all assembled the same way on every request.

The ordering was checked rather than assumed, and it is already cache-friendly:
the static prompt is the `system` message, the conversation history and the
question follow as separate messages, and the one genuinely per-request piece —
a caller-supplied `patient` — is **appended after** the static block, never
inserted before it. `tests/test_api_operations.py` pins both properties so a
future edit cannot quietly move dynamic text in front of the cacheable prefix.

The lever that actually mattered was prompt **size**, and it is already pulled:
an oversized `.metta` selection is replaced by a schema card rather than pasted
verbatim (`PLN_PROMPT_FILE_MAX_BYTES`, default 25 KB — measured on this
checkout, `drugage_etl_short.metta` is ~26,900 estimated tokens verbatim and
~340 as a card), and anything still too large is refused with **413
`prompt_too_large`** before a call is billed.

The default selection measures **308,717 characters, about 77,179 estimated
tokens** (25 files; `tests/test_prompt_size.py` pins that figure against the
real `build_system_prompt`, so this sentence cannot drift from the code again).
It came down from 301,035 (to 297,671) when execution was scoped to the curated stack:
`cellage_calibration.metta` left the runtime inventory the prompt reports, since
the CellAge feature builds its own query-scoped space and the translator was
never shown that file anyway. It has grown again since, to the figure above, with
the patient layers' rules and few-shots (rules 12-17: the default diagnosis, the
heart-risk and first-event rules, the single-supplement flag, the no-lever rule, the reported-CHD observation).
Every `/query` response also reports its own `prompt_tokens_estimate`, which is
the number to size `PLN_MAX_PROMPT_TOKENS` against — it must be comfortably
ABOVE this figure or every default query is refused with a 413 before the LLM
is called. The default limit is 200,000.

**Streaming and batching for `/query` are not implemented.** They were part of
the same recommendation and are deliberately left out: they need their own
response contract (an SSE or chunked shape that the current
`QueryResponse` model cannot express), and the LLM call is only part of a
request's wall time, so streaming would improve *perceived* latency and nothing
else. It belongs in its own change.

## PLN execution runs in a worker process

hyperon 0.2.10 holds the GIL for the whole of `MeTTa.run()` — measured, two
MeTTa runs in two Python threads take exactly as long as two runs in sequence
(ratio 0.998), and a canary thread gets 3 of ~386 expected ticks during one
2.3-second query. So while a query runs, the event loop and every other caller
is starved: a `GET /health` issued 0.4 s into a 3.9 s request took 3.5 s, and
one 35-compound ranking blocked the whole service for 115 s. No threadpool
size, `async def` conversion or asyncio timeout can fix that — you cannot
preempt a Rust call that holds the GIL.

MeTTa therefore runs in a separate PROCESS — one forked per query
(`pln_chat/core/executor.py`). Not a shared pool: a pool cannot aim a kill. A
timed-out query there forced a pool-wide shutdown, and a healthy 3-second
ranking well inside its own budget was SIGKILLed by an unrelated caller's
1-second deadline, then handed a 500 blaming a hyperon abort that never
happened to it. One process per query makes the kill precise; a
`BoundedSemaphore` caps how many run at once (`PLN_WORKER_POOL_SIZE`).

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

| env var                          | default | meaning                                              |
|----------------------------------|---------|------------------------------------------------------|
| `PLN_WORKER_POOL_SIZE`           | 2       | concurrent query processes; **0 runs inline** (below) |
| `PLN_QUERY_TIMEOUT_SECONDS`      | 60      | per-request PLN budget; 0 disables                    |
| `PLN_MAX_INFLIGHT_QUERIES`       | 0       | admission limit; 0 derives 4x `PLN_WORKER_POOL_SIZE`  |
| `PLN_MAX_RANK_COMPOUNDS`         | 60      | cap on a ranking's compound pool, at every entrance    |
| `PLN_MAX_METTA_SORT_COMPOUNDS`   | 10      | the much smaller cap for `strategy="metta_sort"`       |
| `PLN_MAX_ONTOLOGY_FILES`         | 64      | cap on an `ontology_files` selection (deduped)        |
| `PLN_ALLOW_CURATED_WRITES`       | 0       | let `/ontology/apply` append to a CURATED `.metta`     |
| `PLN_MAX_APPLY_BYTES`            | 32000   | largest block one ontology write may append            |

**`PLN_WORKER_POOL_SIZE=0` turns off the deadline and the admission limit**, not
just the subprocess. Neither is enforceable in-process — you cannot preempt a
GIL-holding Rust call — so inline mode gets no 504 and no 503, and a hyperon
abort takes the whole service down again. `GET /health` says so rather than
advertising protections it is not applying: `pln_execution.timeout_enforced` and
`admission_control` read `false`, and `timeout_seconds` is `null`. Inline mode
exists for the in-process contract tests (a forked child cannot see a
monkeypatched module global) and for debugging; it is not a deployment setting.

`PLN_MAX_RANK_COMPOUNDS` is now applied at every entrance to the ranking engine,
not just on `DrugAgeRankRequest`. The pool can arrive as a JSON field
(`POST /drugage/rank`) or inside a `(rank-drugage-lifespan (C1 … Cn))` form that
pydantic never sees — from `/query`, from a hand-written `/metta/run` body, or
from the chat tab. A hand-written 400-compound form went straight past the cap
into ~70 ms of GIL-held work per compound; it is now a 422 `too_many_compounds`
wherever it comes from.

`strategy="metta_sort"` gets its own, much smaller cap. Its MeTTa insertion sort
is O(n²): 6.1 s at n=10, 52.6 s at n=20, past the 60-second deadline by n=40 —
so requests between 40 and the published 60 were advertised as legal and
answered with a 504. Over `PLN_MAX_METTA_SORT_COMPOUNDS` it is a 422 naming
`linear`, which is the default and agrees with it bit for bit.

Because the Gradio UI is mounted on the same ASGI app and drives the same
pipeline, its chat handler routes through the same worker pool — so a query
typed into the UI no longer freezes every REST caller, and a REST ranking no
longer freezes the UI. The UI also builds its prompt with the same grounded
schema card, validates against the same runtime inventory, and applies the same
oversized-file guard, so the two surfaces cannot answer the same question
differently.

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
| GET    | `/drugage/top`     | Rank the **whole** DrugAge build by calibrated lifespan effect — no LLM, no compound list |
| GET    | `/interventions`   | Which interventions target a hallmark of aging (and the reverse), with provenance — review records and bare `TargetsHallmark` facts kept apart |
| GET    | `/hallmarks`       | Every hallmark in the KB, its anchor components and its interventions |
| GET    | `/evidence/human`  | What HUMAN evidence the KB holds for an intervention — design, n, what was measured, what was found, PMID, tier; a null result and a missing record kept apart |
| GET    | `/genes/sources`   | Which gene tables this instance can answer from — CellAge curated + expression, GenAge human + models — with row counts and availability |
| GET    | `/genes`           | Filtered listing across all four gene sources: "which genes drive cellular senescence", "GenAge human genes", cross-source membership |
| GET    | `/genes/{symbol_or_entrez}` | Everything known about one gene, with provenance; `?infer=true` also lifts its CellAge rows into calibrated senescence `Effect` links |
| GET    | `/genes/intersection` | Genes two sources share — the CellAge ∩ GenAge answer — joined on entrez, with a warning when a symbol join is asked for across species |
| GET    | `/patients`        | List the built-in patient profiles                            |
| GET    | `/patients/markers`| Which biomarkers a caller-supplied patient may carry, and in what units |
| POST   | `/patients/preview`| Validate your own patient and see the atoms it becomes — no inference |
| POST   | `/patients/from-text`| A few lines of plain text (age, sex, smoking, labs with units, diagnoses) -> what was read, and a patient with LinAge2 scored in-process — the UI's "My Patient" tab; fixed rules, no LLM, unless `reader: "model"` |
| GET    | `/linage2/features`| The LinAge2 clinical clock's 59 inputs (NHANES codes, KB symbols, which read out a KB biomarker), the forms, the scoped stack, whether an all-cause baseline is loaded |
| POST   | `/linage2/analyze` | A patient's LinAge2 result, analysed without an LLM: per-lab years with causes credited under evidence, mortality hazard, absolute risk when a baseline exists, counterfactuals in years |
| POST   | `/query`           | Ask a natural-language question of the KB (goes through the LLM translator) |
| POST   | `/metta/run`       | Validate + execute a raw MeTTa query directly (no LLM call)  |
| POST   | `/drugage/rank`    | Rank real DrugAge compounds by lifespan/mortality effect, no MeTTa needed |
| POST   | `/ontology/expand` | Extract new ontology entries from pasted paper text, gated against the KB's schema |
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
| `api_key_required`        | 401    | the deployment sets `PLN_API_KEY` and none was sent |
| `api_key_invalid`         | 401    | the `X-API-Key` header was not recognised          |
| `rate_limited`            | 429    | this address' own budget (`Retry-After` is set)    |
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

## Bring your own patient

The personalized stack — risk, decomposition, counterfactuals, intervention
ranking, tiered supplements — used to work for exactly two people, so an app
user's biomarkers could not be scored at all. The inference never needed
changing: it reads `PatientAge`, `PatientSex` and `MeasuredZ` and nothing else.
What was missing was a typed surface and sanitisation.

`/query` and `/metta/run` now take a `patient` object. It is loaded into that
request's space(s) only and never written to disk (a `linage2` block's atoms go
only to the LinAge2 scoped space — see "…and your LinAge2 result"):

```bash
curl -X POST localhost:7860/query -H 'Content-Type: application/json' -d '{
  "message": "what is my 10-year CHD risk, and what should I do about it?",
  "patient": {
    "id": "W45", "age": 45, "sex": "Female", "smoking": "NeverSmoker",
    "markers": {
      "AgeAccelGrim": {"value": 2.1, "unit": "years"},
      "CRP":          {"value": 4.0, "unit": "mg/L"},
      "DNAmGDF15":    1.3
    }
  }
}'
```

A bare number is a **z-score** — standard deviations from the age- and
sex-adjusted mean, which is what the KB reasons in. A `value` is standardised
server-side and the response says exactly how (`derived: true` plus the
formula). `GET /patients/markers` lists what is supported; `POST
/patients/preview` shows the atoms and each marker's Elevated/Normal/Low status
without running anything.

**Where a patient question runs.** A program that names a patient — a built-in
`Patient00N` or your `Caller_<id>` — runs in the *patient stack*: the shared
runtime stack minus seven files no patient form reads (the DrugAge species and
short entries, the DrugAge calibration, the human-evidence and hallmark-targeting
layers, the López-Otín anchors, the measurement-type vocabulary). The full shared
space sits at hyperon 0.2.10's head-symbol limit, and there the patient forms
abort the interpreter depending on details as small as one float — built-in
patients included (`rank-interventions-for-patient` for Patient001). In the
patient stack they answer, with at least 64 head symbols of margin, and wherever
the full stack answers at all the answer is byte-identical
(`tests/test_patient_stack.py`). A program that names no patient runs in the full
stack as before.

**A z you send and a z we derive are not the same thing, and the atom cannot
say which is which.** `Reference.to_z` standardises against a single POOLED
mean and sd — there is no age/sex-stratified reference table in this
repository — so a derived z is *not* adjusted, while the convention published
at `GET /patients/markers` says z means adjusted. Both end up as the same
`(MeasuredZ <patient> <marker> <z>)`, and writing the difference into the space
would mean inventing an adjustment that was never made. So it is reported
instead: `derived: true` and the formula on the marker, a warning on the
patient naming every marker it applies to, and the caveat in the convention
text itself. Send `z` when you have a properly standardised measurement. The
one exception is age acceleration in years, which is divided by the KB's own
`grimaccel-sd-to-years` knob and is exact.

**`/patients/preview` takes the patient object at the TOP LEVEL**, not wrapped
in `{"patient": …}` the way `/query` and `/metta/run` take it. Send the wrapped
shape and you now get a **422** naming the stray key; it used to be accepted
with a 200 and a confident, well-formed description of an EMPTY patient — the
whole payload discarded, and every documented refusal (`unknown_marker`,
`invalid_sex`) bypassed because nothing was ever read. An endpoint whose job is
to check a payload must not answer for one it did not look at.

A malformed field is a 422 that names it, too: `{"markers": {"AgeAccelGrim":
{"z": "high"}}}` returns `invalid_marker_value` with the value it received,
where it used to escape as an unhandled `ValueError` and a 500 with no field
name in it.

**Honesty about the conversions.** There is no calibrated age/sex-stratified
reference table anywhere in this repository. The raw-value conversions use
documented coarse priors, in the same "curated prior" tradition as
`mechanistic_bridges.metta`, and are flagged `provisional: true` with their
source text. Send `z` when you have a properly standardised measurement.
Age acceleration in years is the exception — it is divided by the KB's own
`grimaccel-sd-to-years` knob, so the two can never drift apart.

**What it refuses, and why.** These are not pedantry; each one was reproduced
against the raw `extra_atoms` path:

* an id is namespaced `Caller_…` and must be alphanumeric. An id of
  `Evil) (= (grimage-weight $m) 9.9) (PatientAge Zzz 10` redefined a
  calibration knob and made `decompose-grimage` report weights of 9.9 for a
  *different, pre-existing* patient.
* a colliding id is refused. A second `Patient001` does not replace the first —
  the engine unions both, and `predict-risk-patient` then took 63 seconds and
  returned 512 answers, some pairing a point estimate from one age/sex branch
  with a confidence interval from another.
* `extra_atoms` may no longer carry `(= (…) …)` definitions unless
  `allow_definitions: true`. A definition does not shadow the KB's own.
* an unsupported marker (`LDL`) is rejected **with the supported list**, rather
  than carried as an atom nothing reads.
* a sex outside Male/Female is refused: the baseline CHD table is stratified by
  exactly those two, and a third branch would be an invented number.
* a query naming a patient the KB does not hold is flagged
  `unpersonalized` — `rank-interventions-for-patient` otherwise returns a
  confident population-level ranking for a typo'd id.
* `medications` (`["Metformin"]`, from the drugs the KB holds an interaction fact for; anything else is a
  422 naming them) becomes `(CurrentMedication <id> <drug>)` in the **shared** space only — not in `atoms`, not
  in the LinAge2 space — so a supplement plan, and `(supplement-for-patient …)`, flag a supplement that
  interacts with it. It changes no ranking and no LinAge2 number ("already taking" is not modelled).
* a patient with no **elevated** value the KB has a curated edge into (CRP, DNAmGDF15,
  DNAmPACKYRS, DNAmPAI1, FastingGlucose, HbA1c, LowSerumAlbumin, RDW, Triglycerides) is told so: `patient.warnings` says its
  diagnosis returns `()`, every supplement tier is empty and its ranking is the population
  one, and a `diagnose-patient`, `recommend-supplements[-patient]`, `supplement-for-patient`
  or `rank-interventions-for-patient` form naming it carries the same note in `warnings`.
  That is "nothing to work from", not "no cause": a typed diagnosis, a lab with no edge
  (creatinine, blood pressure, cholesterol), a low value, and a glucose or a triglyceride not typed as fasting count for
  none of them. One exception: a reported heart disease (`prevalent_chd`) is an observation the diagnosis alone explains
  (a prevalence item, never a measured value), so for such a patient the diagnosis answers and the plan and ranking
  still have nothing to work from. `PatientCondition` is in the shared space only and is not in `atoms`.

### …and your LinAge2 result

GrimAge is a DNA-methylation clock almost no app user has. **LinAge2** (Fong et
al. 2025, npj Aging) is a mortality clock built from routine labs, and the
`Rejuve/LinAge2-Python` service already returns, per person, a biological age,
the BA−CA delta in years and one years-contribution per model input. Forward
that response under `patient.linage2` and it becomes part of the same
request-scoped patient:

```bash
curl -X POST localhost:7860/query -H 'Content-Type: application/json' -d '{
  "message": "which of my labs make me biologically older, and what could I do about it?",
  "patient": {
    "id": "W58", "age": 58, "sex": "Male", "smoking": "CurrentSmoker",
    "markers": {"HbA1c": 1.6, "CRP": {"value": 1.2, "unit": "mg/L"}},
    "linage2": { … the LinAge2 /predict response body, as received … }
  }
}'
```

What it becomes: the `LinAgeAccel` clock marker (`z = delta / linage-sd-to-years`,
8.66 years per SD, the spread of the delta across the model's training cohort) plus
one `(LinAgeContribution <Patient> <Input> <years> Measured|Imputed)` atom per
input. What it unlocks — the dedicated forms, which the translator emits for
matching questions and which run in their own **query-scoped space**
(`routed: "linage2"`; see "Demo query forms"). A program may mix them with other
forms — "my LinAge2 drivers and my supplement plan" — and is then split per
top-level expression: the `linage-*` forms run in the scoped space, the rest in the
generic one, in one request (one deadline), and the answers come back in program
order (`routed: "linage2+generic"`). Put each form on its own line: an expression
that nests a LinAge2 form inside another form cannot be split, and the response
warns about it. The generic
space receives the patient **without** the LinAge2 atoms (`LinAgeDelta`,
`LinAgeContribution`, the `LinAgeAccel` z): with them, hyperon 0.2.10 aborts on
`diagnose-patient`, `predict-risk-patient` and `recommend-supplements-patient` for
that patient, and nothing there reads them anyway.

| form | answers |
|---|---|
| `(linage-decomposition-patient &self <P>)` | every input's years, measured and imputed apart, the totals, the explicit age-term residual, and for the six inputs that read out a KB biomarker the hallmark causes — credited **only under a witness** |
| `(linage-hazard-patient &self <P>)` | the relative all-cause-mortality hazard, `1.093^delta`, always |
| `(linage-risk-patient &self <P>)` | an absolute ten-year risk — only when the generated NHANES all-cause baseline is loaded |
| `(linage-counterfactual-patient &self <P> <Lever>)` | how many LinAge2 years normalizing the lever's driver would remove, through the causal graph |
| `(linage-scenarios-patient &self <P>)` | the four standing levers at once |

**Send the witnesses.** A LinAge2 contribution's *sign is not the lab's
direction*: the per-input weights are sex-specific projections that flip sign
between the male and female models (CRP is +4.4 months/SD in women and −0.2 in
men). So the engine credits a cause to an input only when the patient's own z for
the biomarker it reads out is Elevated — or, for the cotinine input, when the
patient is a `CurrentSmoker`. Send `CRP`, `HbA1c`, `FastingGlucose` (z or value), `RDW` or `LowSerumAlbumin` (z
only) and `smoking` alongside the block; without any of them the years come back with no causes
and every counterfactual is 0, and the builder warns you so.

**Or just type it.** `POST /patients/from-text` takes `{"text": "58 year old male,
current smoker\nalbumin 4.1 g/dL\nHbA1c 6.4 %\nCRP 3.1 mg/L"}` and scores LinAge2
in-process from parameters extracted out of `Rejuve/LinAge2-Python` (matches the
service to 2e-14 years; `docs/linage2_integration.md §9`), so no service and no
pasted JSON are needed. It returns `read` — every value as typed, converted to
LinAge2's unit, with a status — and, when everything is usable, `patient`: send it
as `patient` anywhere above. A value whose unit cannot be pinned down (`CRP 3.1`:
mg/L or mg/dL?) is not guessed; `ok` is false and `problems` says which. The same
value doubles as the knowledge base's witness (CRP in mg/L, HbA1c, a *fasting*
glucose, RDW and a low albumin against LinAge2's reference, a *fasting* triglyceride, the stated smoking
status), so causes can be credited without typing
anything twice. This is what the Gradio UI's **My Patient** tab does; the patient
then lives in that browser session only.

`reader` (since API 2.1.0) is `"rules"` by default — fixed rules, no LLM, exactly as
before. `"model"` also sends the text to OpenAI (`PLN_EXTRACT_MODEL`): the model may
**rewrite** statements the rules did not understand into the rules' own wording, which
the rules then read; it never supplies a value (numbers and units are copied from your
text), never overrides a refusal, and never sets a smoking status on its own. The
response says what happened: `reader_used` (`"rules"` or `"rules+model"`),
`read_as_text` (what the rules read — send it back with `reader: "rules"` to get the
same patient without the model), `model_error` (`{code, message, configuration}` when
the model could not be used and the rules read alone: `timeout`, `refused`,
`truncated`, `bad_output`, ...), `suggestions` (for a refused statement, or one the
rules and the model read differently: `{line, start, end, original, wordings, reason,
blocking}` — put a wording in place of `original` and resend) and `statements` (each
with `source: "rules" | "model"` and what was `typed` there). `reader: "model"` answers
503 `model_reader_not_configured` without the server's `OPENAI_API_KEY` (or with a
`PLN_EXTRACT_MODEL` that lacks strict structured outputs) and 413
`text_too_long_for_model_reader` over `PLN_EXTRACT_MAX_CHARS` (4,000).

**No LLM needed.** `POST /linage2/analyze` takes the same patient at the top level
(the `linage2` block required, plus optional `levers`) and returns the whole
picture as JSON: decomposition, hazard, risk (or the reason there is none),
counterfactuals. `GET /linage2/features` publishes the 59 input codes and which
six can be explained. The refusals, each a 422 with a code:
`unknown_linage2_feature`, `duplicate_linage2_feature`, `linage2_inconsistent`
(delta ≠ BA − CA), `linage2_age_mismatch` (a result computed for another age),
`implausible_linage2_delta`, `implausible_linage2_contribution`,
`duplicate_clock` (a block and a bare `markers.LinAgeAccel`), `unknown_lever`,
`linage2_required`. A `linage-*` form naming a patient with no LinAge2 result is
valid, empty, and **warned** — the built-in patients have none.

Design, measurements and limits: `docs/linage2_integration.md`.

## Discovery without an LLM

The questions that used to be unanswerable are now plain GETs. None of them
spends an OpenAI call, and none of them needs the caller to already know the
answer.

**"Which drugs extend lifespan in mice with the strongest evidence?"** —
`POST /drugage/rank` only ever scored a pool the caller supplied, so the
translator invented a four-compound pool and ranked that. `GET /drugage/top`
ranks every scorable compound in the build — **1,035** of the 1,043 DrugAge
lists, the other 8 reporting no lifespan change on any row, and both numbers
are in the response (`total_compounds`, `total_compounds_in_source`,
`unscorable_compounds`):

```bash
curl 'localhost:7860/drugage/top?n=10&species=Mus_musculus&itp_only=true'
# -> Rapamycin +0.319, Astaxanthin +0.304, 17alpha-estradiol +0.304,
#    Canagliflozin +0.251, NDGA +0.251, Aspirin +0.231, Acarbose +0.187 ...
#    each with its species, sex, significance, change %, PMID and evidence tier
```

Filters compose: `species=`, `clade=` (Vertebrate / Invertebrate / Fungi /
Protozoa), `itp_only=`, `significant_only=`, `min_confidence=`, `n=`, and
`direction=harmful` to rank the compounds that SHORTENED lifespan, most harmful
first.

This one is scored in Python rather than in MeTTa, and the reason is not
performance: ranking 1,043 compounds would be 1,043 engine calls, and loading
the rows to rank them in one space **aborts** hyperon (a non-unwinding panic
past a few hundred rows in a variable-slot match). The scorer reads its
constants out of `drugage_calibration.metta` / `epistemic_calibration.metta` /
`species_taxonomy.metta` at load time — tuning a knob retunes both paths — and
`tests/test_drugage_discovery.py` asserts it reproduces the engine bit for bit.

**"Which interventions target mitochondrial dysfunction?"** and **"which
hallmarks does rapamycin target?"** — both directions of a relation that kept
returning empty, for two separate reasons: the translator reached for
`TargetsHallmark`, which the ontology declared and nothing populated, and
`Rapamycin` did not exist as an atom anywhere in the runtime, so any query
naming it was rejected with a 422.

```bash
curl 'localhost:7860/interventions?hallmark=MitochondrialDysfunction'
curl 'localhost:7860/interventions?intervention=Rapamycin'
curl 'localhost:7860/interventions?intervention=Fisetin'
curl localhost:7860/hallmarks
```

`hallmark_targeting.metta` populates `TargetsHallmark` — one line per curated
review record, plus Rapamycin (Harrison 2009, PMID 19587680; Saxton & Sabatini
2017, PMID 28283069) and Metformin (Bannister 2014, PMID 25041462). So the
predicate the translator kept reaching for is now the RIGHT form, in MeTTa too:

```
!(match &self (TargetsHallmark Rapamycin $h) $h)
!(match &self (TargetsHallmark $i MitochondrialDysfunction) $i)
!(hallmarks-of &self Rapamycin)
!(interventions-for &self CellularSenescence)
```

The response keeps the two shapes apart, and this is deliberate:

* `evidence` — a López-Otín 2023 review record, with the species model, the
  reported outcome text, the review reference number and the publication;
* `targeting` — a bare `(TargetsHallmark …)` fact, with its publications and
  `provenance: "targeting_fact"`. It says WHAT the intervention acts on and
  nothing else. Padding it out into an evidence record would mean inventing a
  species model and an outcome sentence no source states.

A link that a record already covers is not repeated under `targeting`. An
intervention with no link of either kind (say `?intervention=Berberine`) still
gets an explicit `note` saying the curated layer is a review table rather than
a census — not silence.

**`TargetsHallmark` is a targeting claim, not a causal edge.** It carries no
truth value, and it does NOT put the intervention into a chain: `!(infer &self
Rapamycin CoronaryHeartDisease)` still returns nothing, and rapamycin still
does not appear in `rank-interventions`. That is correct, not a gap. The
obvious shortcut — hanging rapamycin off the existing
`DeregulatedNutrientSensing -> InsulinResistance -> FastingGlucose -> CHD` axis
— would get the SIGN wrong: chronic rapamycin *causes* glucose intolerance in
mice while extending lifespan (Weiss 2018, PMID 29579736), and the KB records
that as a `Limitation` fact rather than a comment. An LLM will make that chain
on request; the point of the deduction layer is that it will not.

**"What does the evidence say about metformin in HUMANS?"** — this used to be
answered in prose. The answer was true (the bulk data loaded here — DrugAge,
GenAge, CellAge — is model-organism and cell evidence, which cannot speak to
people) but it was un-grounded: there was nothing in the KB to point at, so the
honest answer had to be *written* by a language model. `human_evidence.metta`
is the record set, and `GET /evidence/human` reads it:

```bash
curl 'localhost:7860/evidence/human?intervention=Metformin'
curl 'localhost:7860/evidence/human?intervention=DasatinibPlusQuercetin'
curl localhost:7860/evidence/human           # the whole table
```

```
Metformin
  Bannister2014_Metformin_Survival  ObservationalCohort  n=78241
      result ReportedBenefit   tier Epidemiological   PMID 25041462
  Barzilai2016_TAME_Metformin       PlannedTrial       n=null
      result NotYetReported    tier null              PMID 27304507
```

Each record carries its design (`RandomizedControlledTrial` /
`ObservationalCohort` / `OpenLabelPilot` / `PlannedTrial`), its n, what was
measured, what was found, the PMID and an `evidence_tier` drawn from
`epistemic_calibration.metta`. The point of the shape is the three states prose
destroys:

* `result: "ReportedNull"` — somebody measured it in people and nothing
  changed. Dasatinib + quercetin returns three records from two trials, and one
  of them is the null: pulmonary function in 14 people with IPF was
  **unchanged** (Justice 2019, PMID 30616998), from the same n=14 pilot whose
  physical-function measures improved. A summary sentence keeps the first and
  loses the second.
* `evidence_tier: null` with `result: "NotYetReported"` — **TAME has not
  reported.** Barzilai 2016 (PMID 27304507) is the trial's design rationale, not
  a result, so the record deliberately carries no tier and no truth value. This
  is the most consequential fabrication available in this field, and the record
  shape is built so it cannot be committed by accident.
* no record at all — an explicit `note` ("an absence of a RECORD, not evidence
  of absence"), never an unexplained empty list. Omega-3 is a fourth case: its
  human tier already lives in `supplement_evidence.metta`, so it comes back
  under `cross_references` rather than duplicated here.

Nothing in this layer carries a truth value or adds an edge to the graph
`infer` traverses — same stance as `TargetsHallmark`. A study record says what
was measured in how many people and what was found; turning that into a signed,
weighted causal edge is a separate act of calibration a curator performs
deliberately.

In MeTTa, the same thing:

```
!(human-evidence &self Metformin)
!(human-evidence &self DasatinibPlusQuercetin)
!(human-evidence-interventions &self)
```

**"What would his risk look like if he had never smoked?"** — recorded as an
honest Gap, and it was one. `PatientSmoking` reached **zero** inference paths:
before `lifestyle_evidence.metta` it appeared in its type declaration, two
patient facts, one doc line and a regex in `api.py`, and nothing consumed it.

The fix is not a `smoking -> CHD` edge. The risk model reads exactly one
predictor — the composite clock `AgeAccelGrim` — and a parallel smoking term
would double-count an exposure the clock already carries. GrimAge carries it
explicitly: `DNAmPACKYRS` is a GrimAge component built from 172 CpGs and is the
DNAm surrogate *for* pack-years. So the wiring is the one the clock's own
construction implies:

```
SmokingCessation --Neg--> SmokingPackYears --Pos--> DNAmPACKYRS  (PartOf GrimAge)
```

and the existing machinery does the rest — `resolve-lever` routes the lever to
the driver it reduces, `cf-parts` finds `DNAmPACKYRS` among the clock's
components, and `project-risk` turns the clock reduction into absolute risk.
Not one line of the counterfactual or risk layer changed.

```
!(counterfactual-patient &self Patient003 SmokingCessation)
;; => (Counterfactual SmokingCessation AgeAccelGrim (expected-delta -0.21)
;;       (signed Neg (stv 0.21 0.85)) (Via (DNAmPACKYRS)))

!(project-risk-patient &self Patient003 SmokingCessation)
;; => (ProjectedRisk SmokingCessation CoronaryHeartDisease
;;       (point 0.2226…) (reduction 0.01369…) (delta-clock -0.21)
;;       (confidence 0.85) (Via (DNAmPACKYRS)))
;;    23.63% -> 22.27% absolute over ten years.
```

`Patient003` is a new, **synthetic** smoking-dominant profile in
`lifestyle_evidence.metta` — a 64-year-old male CURRENT smoker with a strongly
elevated `DNAmPACKYRS` and normal senescence, inflammation and metabolic
markers, the third axis alongside senescence-dominant `Patient001` and
metabolic-dominant `Patient002`. `Patient002` was NOT edited to demo this,
although he is a `FormerSmoker`: his numbers are quoted in the evaluation and
pinned by a test, and rewriting a published patient to make a new feature look
good is how a knowledge base stops being trustworthy. (`Patient003` was written
a former smoker too, at first — which made the file's own flagship demo a case
the lever does not apply to, and it returned the full benefit only because the
precondition did not exist yet.)
`Patient001`'s and `Patient002`'s answers are byte-identical before and after.

Two consequences worth reading literally:

* The lever acts on `DNAmPACKYRS`, not on the status string. A caller-supplied
  smoker who sends no `DNAmPACKYRS` gets an expected delta of 0 and an empty
  `(Via ())`, and `POST /patients/preview` warns about exactly that — the zero
  means "no measured pack-years signal to act on", not "quitting would not
  help".
* **And the symmetric case, which is the one that mattered.** `DNAmPACKYRS` is
  an elastic-net estimate with real error that also responds to second-hand
  exposure, so an elevated value in a NEVER SMOKER is an ordinary data state —
  and it used to buy them a quantified benefit from quitting, byte-identical to
  a smoker's, silently. The knowledge base now declares
  `(LeverRequiresSmoking SmokingCessation CurrentSmoker)`, and `/query`,
  `/metta/run`, `/patients/preview` and the chat tab all warn when a query
  names a lever the patient cannot pull. The ENGINE does not yet enforce it:
  the curated KB is at hyperon 0.2.10's ceiling (four more trivial rule
  definitions before it aborts — measured, and asserted by
  `tests/test_smoking_lever.py`), and the composition below spends that budget.
  A caller talking to hyperon directly still gets the ungated number. That gap
  is recorded in `pln_counterfactual.metta` §3b rather than papered over.
* Quitting does not restore a never-smoker, and **the engine now applies that
  discount instead of only documenting it.** The cessation edge's strength is
  0.39, anchored on Duncan 2019's HR of 0.61 (95% CI 0.49–0.76) for quitting
  within 5 years versus continuing (PMID 31429895). `resolve-lever` routed the
  lever to the driver it reduces and DROPPED that edge, so the counterfactual
  answered "if the exposure burden vanished" for a question that asked "if he
  quits" — the exact claim the file's own `Limitation` fact says it does not
  make. Measured on Patient003: **−0.21 at confidence 0.85 before, −0.0819 at
  0.324 now**, and a projected 10-year CHD reduction of 0.54 percentage points
  rather than 1.37. It agrees exactly with
  `!(infer &self SmokingCessation DNAmPACKYRS)`, which it used to contradict.
  The same silent discard affected every intervention lever:
  `DasatinibPlusQuercetin` returned the senescence driver's number byte for
  byte, and now carries its own `(stv 0.70 …)`.
* Joehanes 2016 is tiered **Epidemiological (0.60)**, not `MultipleHumanTrials`
  (0.85). It is an observational 16-cohort EWAS meta-analysis, and this ladder
  orders DESIGNS, not sample sizes — `Epidemiological` sits below
  `SingleHumanTrial` precisely because pooling observational cohorts does not
  make them a trial. Duncan 2019, in the same file, was already tiered that way.
  Two observational studies cannot sit 0.25 apart on a design ladder.

**"Which genes drive cellular senescence?"**, **"GenAge human genes"** and
**"CellAge ∩ GenAge"** — the first returned four hallmark *components*
(`TelomereAttrition`, `DNADamage`, …) and not one of CellAge's 927 curated
genes; the second came back empty; the third could not be expressed at all. And
the one workaround failed too: selecting `cellage_genes.metta` or
`genage_models_etl.metta` as `ontology_files` took the prompt to 417k / 386k
tokens, over OpenAI's limit.

The prompt half is fixed (an oversized file becomes a schema card — see the KB
size note below). `GET /genes` is the other half:

```bash
curl 'localhost:7860/genes?source=cellage_curated&effect=Induces'      # 417 rows
curl 'localhost:7860/genes?source=cellage_curated&effect=Inhibits'     # 510 rows
curl 'localhost:7860/genes?source=genage_human'                        # 307 rows
curl 'localhost:7860/genes?source=genage_models&organism=Mus%20musculus'
curl 'localhost:7860/genes/TP53'                                       # or /genes/7157
curl 'localhost:7860/genes/intersection'                               # 113 genes
curl localhost:7860/genes/sources
```

```
GET /genes?source=cellage_curated&effect=Induces
  AAK1    22848  Induces  UnspecifiedSenescenceType  NonCancerCellContext  PMID_26583757
  ABCB1    5243  Induces  StressInducedSenescence    CancerCellContext     PMID_10825123
  ...  total 417, truncated true
```

These are **curated experimental annotations, not inference.** A CellAge effect
label means a curator read a paper reporting that gene inducing or inhibiting
senescence in a cell line. Nothing in a plain `/genes` response is ranked,
weighted or derived.

Four things about the shape are deliberate:

* **It is read from the CSVs under `data/`, not from the generated MeTTa.**
  `build/` is gitignored and is not on `_discover_metta_files`'s path, the files
  are 8-14x over `PLN_MAX_KB_FILE_BYTES`, and loading one into a hyperon space
  aborts the interpreter. There is also a data reason:
  `genage_models_parser.py:57` writes the sanitised atom back out as the gene
  name, so `aak-2` is emitted as `(GeneSymbol aak_2 "aak_2")` and the real
  symbol is gone. The index keeps `aak-2`. When the unpacked tables are missing
  (they are gitignored) the committed `.zip` archives are read in memory
  instead, and a source that cannot be read at all comes back
  `available: false` with a note — never a silently short answer.
* **One record per (source, row), not per gene.** CellAge annotates TP53 three
  times, one per senescence type; collapsing those would pick a winner no source
  picked. Records are *sparse* — a CellAge row has no `organism` key at all
  rather than `organism: null`, because a null there would read as "looked and
  found nothing". `GET /genes/sources` lists which keys each source contributes.
* **Entrez is the join key.** CellAge ∩ GenAge human is 113 genes by entrez and
  112 by symbol — close enough to look interchangeable, and it is not. Against
  `genage_models` the gap stops being cosmetic: 1 gene by entrez, 67 by symbol,
  and those 67 are worm/yeast/fly ORTHOLOGUES that happen to share a name
  (`ATM`, `AKT1`, `BRCA1`), not the same gene measured twice. A symbol join
  across that boundary always returns a `warnings` entry saying so.
* **Every listing is capped** (200 max) and carries `total` with an explicit
  `truncated` flag, so a page is never mistaken for the whole answer.

Each record also carries `metta_row_id` — the atom the ETL gives that row
(`CellAgeRow_869`, `GenAgeHumanRow_5`) — so a Python answer can be traced back
to the MeTTa the engine would see. Rows the ETL filters out (an `Unclear`
CellAge effect, an organism the models parser does not map) carry `null` there,
which is how you can tell.

**A real ETL defect, fixed in passing.** `cellage_etl.py` built its provenance
token as `f"PMID_{atom(ref, 'PMID_')}"`, and `atom()` already prepends `PMID_`
to an all-digit reference — which every CellAge reference is. So every CellAge
provenance atom read `(ReportedIn CellAgeRow_0 PMID_PMID_26583757)`: a symbol
matching no real PMID and joining to nothing. The ETL now emits
`PMID_26583757`, **so regenerating it changes the generated symbol**; readers
normalise both spellings, because a `build/` made before the fix is still on
disk and still has to be readable.

**Gene inference, opt-in:** `GET /genes/{gene}?infer=true`

```bash
curl 'localhost:7860/genes/TP53?infer=true'
# -> (Effect Gene_TP53 CellularSenescence Pos (stv 0.7 0.35))  x3, one per row
curl 'localhost:7860/genes/SIRT1?infer=true'
# -> (Effect Gene_SIRT1 CellularSenescence Neg (stv 0.7 0.35))
```

This is the one place in the gene stack a number is produced. It selects the
gene's CellAge rows, injects them into a query-scoped hyperon space with
`cellage_calibration.metta`, and lifts each into a signed, calibrated link — the
same architecture as the DrugAge lift, with a smaller stack
(`pln_runner.CELLAGE_STACK`). In MeTTa:

```
!(cellage-effect &self CellAgeRow_869)
!(cellage-gene-effects &self Gene_TP53)
!(genes-affecting-senescence &self Increases)
```

**These three run inside the query-scoped space that `GET /genes/{symbol}
?infer=true` builds — not through `POST /metta/run`.** The rules are loaded in
the generic space, but the CellAge rows they match are not (a variable-slot
match over ~900 of them aborts hyperon 0.2.10, which is why the slice is capped
at 100 and built per query). Sent to `/metta/run`, the first two are a 422
naming `CellAgeRow_869` / `Gene_TP53` — symbols the generic space does not hold
— and the third used to be the worse case: a 200 with `pln_status: "empty"`,
`pln_results: []` and an EMPTY `ungrounded_predicates`, which by the reading
rule two sections below says "no genes increase senescence". It now carries a
warning saying the space holds no rows for that rule and naming the endpoint
that does. The same warning covers the `drugage-*` accessors, and is derived
from the runtime inventory rather than a hand-kept list, so a new scoped layer
is covered the day it lands.

The returned `semantics` block says where both numbers come from, every time:

* **strength is a curated prior, identical for every curated row.** CellAge
  records a direction and no effect size, so there is no magnitude to report and
  none is invented. A per-row spread here would rank genes by a number no source
  states.
* **confidence is `(evidence-confidence InVitro)`** from
  `epistemic_calibration.metta`, written as that lookup rather than as a
  literal — retuning the tier retunes this layer, and a test proves it by
  retuning it.
* **the ETL's own numbers are not used.** `build/cellage_genes.metta` also
  carries `(Causes Gene_TP53 (Increases CellularSenescence) (stv 0.82 0.70))`.
  Those are computed by `cellage_etl.calibrated_stv` from the senescence type
  and the cancer-cell flag; they do not come from `epistemic_calibration.metta`,
  they are not calibrated, and this layer ignores them.

Three absences are reported rather than papered over. An `Unclear` CellAge
curation yields **no** link — the source declined to state a direction, so
neither does the engine. A gene with no CellAge row says so instead of returning
an empty list. And the expression signatures (1,259 genes differentially
expressed in senescent cells) are deliberately **not** lifted: a gene going up
in senescent cells is a correlation, and lifting it into `(Effect …)` would
convert association into causation at scale.

**This layer does not chain senescence to mortality.** `CellularSenescence` is a
hallmark, not an outcome, and the obvious bridge would get the sign wrong in
half the biology — senescence is tumour-suppressive in a cancer cell and
damaging in an ageing tissue. The KB holds no evidence that fixes that sign, so
no gene falls out of a mortality ranking on the strength of a CellAge row. Same
stance as `TargetsHallmark`.

The slice is capped at **100 rows**, enforced before hyperon is called, because
the failure it prevents is an abort rather than an exception. Measured on this
machine against `!(genes-affecting-senescence &self Increases)`: 125 rows fine,
150 rows abort. `build/` being gitignored, the selector falls back to a
committed 25-row fixture and the response says which one it used — a sample is
never presented as the corpus.

**"List your data sources and counts"** — `GET /kb/schema`, below.

### Valid, and guaranteed to return nothing

The ontology DECLARES a much larger vocabulary than it populates.
`logical_predicates.metta` declares `Predicts`, `Extends`, `HazardRatio`, the
gene predicates and the DrugAge row predicates; the generic runtime space holds
**zero** facts for any of them. A query over one of those is perfectly
well-formed MeTTa and returns nothing — which reads exactly like "the answer is
no". The translator emitted `TargetsHallmark` three times in the 2026-09-18
evaluation, and `UsesIntervention` / `AvgLifespanChangePercent` three more.
(`TargetsHallmark` and `Causes` have since been populated by
`hallmark_targeting.metta` and are no longer on that list — `GET /kb/schema` is
the live answer, not this paragraph.)

Two changes remove that failure mode:

* `GET /kb/schema` reports the truth — every grounded predicate with its arity,
  fact count and source files, plus `declared_but_empty_predicates`. It is also
  the answer to "what data do you have and how much of it".
* every `/query` and `/metta/run` response carries `ungrounded_predicates` and
  `validation_warnings`. An empty `pln_results` next to a non-empty
  `ungrounded_predicates` means *the KB cannot express this relation*, not *no*.
* and the third case, which this rule used to misread: a rule that IS loaded
  whose DATA lives in a scoped space. `ungrounded_predicates` is empty there,
  because the predicate is declared and the rule is real — so those responses
  now carry an explicit `warnings` entry naming the predicates that hold no
  facts here and the endpoint that does serve them. **An empty `pln_results` is
  only a real "no" when `ungrounded_predicates` AND `warnings` are both
  empty.**

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

There is a second, sharper limit on the same space, and it is not about file
size: **the number of top-level expressions the runtime KB loads in total.**
Adding `lifestyle_evidence.metta` and a first draft of `human_evidence.metta`
took it from ~940 to ~1050 expressions, and `POST /query` for a caller-supplied
patient started aborting the interpreter outright — the non-unwinding panic in
`hyperon-space/src/index/trie.rs`, which no `except` can catch. In the API that
surfaces as a 500 and a replaced worker (`core/executor.py` exists for exactly
this); in the test suite it killed the run with "Fatal Python error: Aborted"
and no failing test to point at.

The human-evidence records were reshaped to one atom per study rather than one
atom per field, which cost ~49 expressions instead of ~113, and
`tests/test_human_evidence.py` now carries two guards: a cheap budget assertion
on the expression count, and the query that died, re-run in a **subprocess**, so
the next regression is a red test rather than a dead process.

The same size question applies to the **prompt**, and used to be fatal there.
The LLM context pasted every selected file verbatim; selecting a CellAge or
GenAge ETL file took the prompt to 417,000 tokens and the call came back as a
billed upstream 400. A selected file over `PLN_PROMPT_FILE_MAX_BYTES`
(default 25 KB) is now replaced by its **schema card** — `cellage_genes.metta`
goes from ~131,000 tokens to ~215, and says which predicates it holds and how
many facts each has, which is more useful to a translator than the rows are.
Every hand-written layer in this repo is under that limit and is still pasted
verbatim, because its prose is what the translator reasons from. The grounded
schema card also replaces a flat symbol index in which a declaration and 400
facts looked identical.

Earlier drafts of this section carried "~62,400 to ~57,400 tokens" for the
default prompt. Those figures were wrong — the measured value is the ~75,250
above — and an operator sizing `PLN_MAX_PROMPT_TOKENS` off them (say, 60,000 as
generous headroom) gets a **413 on every default query**. The number is now
asserted by a test rather than written down twice.

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
0.50 and +80 % reads 0.80.

**Confidence** is `evidence tier x significance gate x 0.9`, where the 0.9 is
the per-hop chain discount for the lifespan → mortality step. For a row whose
result was reported **Significant** the gate is 1.0, so the familiar numbers
hold: **0.81** ITP, **0.45** non-ITP vertebrate, **0.315** invertebrate,
**0.18** yeast. An **Unreported** row reads 0.6x those and a **NotSignificant**
row 0.4x. An ITP row folds significance into its tier, so its gate stays 1.0 —
a well-run null is high-confidence evidence of ~no effect, and it is the
near-zero STRENGTH that sinks it, not low confidence.

That gate used to be a CAP — `min(tier, gate)` — and a cap did almost nothing,
because every non-ITP tier already sits at or below every gate. Measured over
the full 3,423-row build: the cap changed the confidence of **101 rows out of
3,208** non-ITP ones, all of them vertebrate NotSignificant. The other 995
NotSignificant and 187 Unreported rows scored at *exactly* the confidence of a
significant result on the same species, so a compound whose only evidence found
no significant change sat in the whole-build ranking between two that did, at
the same number. Every Significant row keeps its old value under the
multiplier; only the weak rows moved, downward.

A score of exactly 0.0 is a reported null (metformin and resveratrol are ITP
negatives at confidence 0.81), never a missing value — and it now reads
`direction: "no_effect"`. `sign` is the MeTTa Effect convention and has no
zero, so a 0.0 % row is `Neg` because it is not negative; rendering that as
`"protective"` put `direction: "protective"` next to `score: 0.0` and a
`semantics.zero_score` note calling it a measured null, in one response.
`GET /drugage/top?direction=none` lists them.

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
`PLN_MAX_RANK_COMPOUNDS` (default 60) at every entrance to the engine, and
`metta_sort` additionally at `PLN_MAX_METTA_SORT_COMPOUNDS` (default 10),
because beyond that its O(n²) sort cannot finish inside the query deadline.

## Expanding the KB from a paper

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

### What `/ontology/apply` will not do

The block arrives as text and nothing required it to be the one
`/ontology/expand` produced, so it faces that endpoint's own schema gate here:
predicates the rules consume, no hand-typed `(stv x y)`, no redefinition of a
calibration constant, a tier the table scores, and a PMID or DOI **in an atom**
— a `;;` comment is not provenance, because comments are never loaded into the
space, so nothing in the knowledge base would carry the citation. (The gate
searched the raw text, comments included, which is the exact failure its own
docstring lists as one of the four it exists to catch.) A refused
block comes back **422 `block_failed_schema_gate`** with a `refusals` list
naming what each entry broke — reported, never silently dropped.

Three further refusals, each with a measured reason:

* **403 `curated_file_is_read_only`.** The path was already confined to the two
  ontology roots; the CONTENT was not. One unauthenticated request appending
  `(= (evidence-confidence RCT_Human) 0.05)` to `epistemic_calibration.metta`
  re-tiers every human trial in the knowledge base, permanently, for every
  later caller — and the response says `applied: true` and nothing else.
  Caller-generated entries go to `pln_chat/ontology/metta_files/`, which is
  loaded the same way. Set `PLN_ALLOW_CURATED_WRITES=1` to override, knowing
  what it means.
* **422 `block_too_large`** over `PLN_MAX_APPLY_BYTES` (32 KB; the canonical
  taurine block is ~2 KB).
* **422 `file_would_exceed_kb_cap`** when the append would push the file past
  `PLN_MAX_KB_FILE_BYTES`. Over that size the runtime stops loading the file —
  no error, the layer just leaves the knowledge base and answers get quietly
  thinner. A refusal is the better failure.

The Gradio "Apply to Ontology" button goes through the identical gate: it is
mounted on this same ASGI app, so "it is only the local UI" was never true.

### Extracted knowledge now lands in a schema the rules read

**This is a behaviour change.** Fed the taurine abstract, this endpoint used to
return valid MeTTa that no rule could use: it minted `increases-life-span`,
`declines-with-aging` and `reduces` instead of the KB's own predicates, wrote
its own truth values (`(stv 0.93 0.9)`, plus a hand-rolled
`(= (study-confidence …) 0.92)`) straight past the calibration tables, and put
the PMID in a comment and nowhere else. Applied as-is, every atom in it was
inert — `infer`, `explain`, `rank-interventions`, `recommend-supplements`,
`patient-relevance` and `drugage-effect` all returned nothing for all of it.

Three things changed.

**The model is shown the schema, and only the schema.** The prompt used to
paste in `existing_raw_content[:6000]` — an *alphabetical* 6 KB slice of ~290 KB
of ontology, 2.1 % of the KB and not one example of the target form. It now
carries ~4 KB of verbatim canonical forms (a `Publication` record, a typed
intervention node, a raw `Experiment` row, a `HallmarkInterventionEvidence`
audit record, an `Effect` link, `TargetsHallmark`, `EvidenceLevel` /
`SafetyProfile`), a closed predicate list, and the eleven-value
`EvidenceCategory` enum **read out of `epistemic_calibration.metta`** rather
than restated. The two prompt lines that asked for invented confidences are
gone.

**Confidence is never the model's to propose.** It may name a *study type*; the
emitted atom carries `(evidence-confidence <Tier>)` unevaluated, exactly as
`mechanistic_bridges.metta` writes it, so the calibration table stays the single
authority. Strength follows the same rule as everywhere else in this KB: when
the paper reports a percent lifespan change, it is *derived* with
`drugage_calibration.metta`'s own `|pct| / (|pct| + 20)` (the knob is read from
that file, so retuning it retunes generated blocks); when it does not, the
strength is a **curated prior** and says so, in the response and in a `;;`
comment above the atom.

**Nothing is refused quietly.** Every entry passes a gate before it can reach
`metta_block`, and a refusal comes back in `rejected_entries`:

| code                           | what tripped it                                     |
|--------------------------------|------------------------------------------------------|
| `unknown_predicate`            | a head no rule reads — including one nested inside `(Evaluation (pred …) …)`, or declared as a signature |
| `invented_truth_value`         | a two-float `(stv x y)`; the confidence slot must be the lookup |
| `invented_confidence_constant` | `(= (<name> …) <float>)` minting a new confidence knob |
| `redefines_calibration`        | a redefinition of `evidence-confidence`, `sig-gate`, `calibrate-tv`, … |
| `unknown_evidence_category`    | a tier `epistemic_calibration.metta` does not declare |
| `missing_identifier`           | no PMID and no DOI — the entry cannot be traced to a paper |

New response fields: `rejected_entries` (`kind`, `name`, `metta`, `codes`,
`reasons`), `unconsumed_predicates` (heads in the block the runtime KB grounds
nowhere else — not an error, but the difference between joining existing data
and starting a table of one), and, on each accepted entry, `identifier`,
`evidence_tier`, `effect_size_pct`, `provisional`, `provisional_fields` and
`notes`. `provisional` is true whenever a value was *proposed* rather than
derived; the same lines are written into the block as `PROVISIONAL` comments,
because the reviewer who needs them may only ever see the file.

The identifier reaches the atoms, not just the header comment: a measurement row
that arrives without one is given `(ReportedIn <row> PMID_<digits>)`, the same
provenance shape the DrugAge ETL emits, so
`!(match &self (ReportedIn $r PMID_37289866) $r)` answers.

Duplicate detection was fixed in the same pass, because the constrained output
walks into both of its holes: expressions are now compared whole (the canonical
`Effect` form is written over **two** lines, and the old line-anchored
STV-stripper matched neither half, so re-extracting an existing bridge looked
net-new), and the name fallback asks the runtime inventory instead of
regex-searching 290 KB of raw text — which included `;;` comments and the 107 KB
ETL dump the runtime excludes, so a symbol mentioned once in prose was reported
as an existing duplicate and discarded.

`tests/test_ontology_expansion.py` builds the canonical taurine block by hand
(Singh 2023, *Science*, PMID 37289866, doi 10.1126/science.abn9257), loads it
into hyperon beside the runtime KB and asserts that `infer`, `explain`,
`rank-interventions`, `hallmarks-of` and `drugage-effect` all consume it — and
that the evaluation's own block, in the same engine, still answers nothing.

## Demo query forms

These map to dedicated MeTTa functions rather than hand-built patterns — the
LLM translator already knows to emit them for matching natural-language
questions (`/query`), or write them directly for `/metta/run`. `<Patient>` is
a known ID from `GET /patients` (currently `Patient001` / `Patient002` /
`Patient003`); `<Lever>` is a cause (`ChronicInflammation`,
`CellularSenescence`, `InsulinResistance`, `SmokingPackYears`), an intervention
(`DasatinibPlusQuercetin`, `Metformin`, `SmokingCessation`), or a marker
(`CRP`).

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
| "which hallmarks does `<Intervention>` target"                | `(hallmarks-of &self <Intervention>)`                        |
| "which interventions target `<Hallmark>`"                     | `(interventions-for &self <Hallmark>)`                        |
| "what if `<Patient>` had never smoked / had quit"              | `(counterfactual-patient &self <Patient> SmokingCessation)`   |
| "what does the evidence say about `<Intervention>` in humans" | `(human-evidence &self <Intervention>)` — or just call `GET /evidence/human` |
| "rank rapamycin, metformin by lifespan benefit"               | `(rank-drugage-lifespan (<Compound1> <Compound2> …))` — or just call `POST /drugage/rank` |
| "which labs make me biologically older (LinAge2)"              | `(linage-decomposition-patient &self <Patient>)` — needs a `patient.linage2` block |
| "how much does my LinAge2 raise my mortality hazard"           | `(linage-hazard-patient &self <Patient>)` — absolute risk via `(linage-risk-patient …)` once a NHANES baseline is generated |
| "how many LinAge2 years would quitting smoking remove"          | `(linage-counterfactual-patient &self <Patient> SmokingCessation)` — 0 with an empty `Via` for a never smoker |
| "what could I do about my LinAge2 biological age"               | `(linage-scenarios-patient &self <Patient>)` — or just call `POST /linage2/analyze` |

A finding with no mechanistic path is omitted rather than invented; a
compound with a negative gold-standard trial (e.g. `Resveratrol`, ITP
negative) is still surfaced but flagged `NotRecommended`, never silently
dropped. The last two forms are lookups, not inference: they answer what an
intervention targets, and they deliberately do not make it rankable — see
"Discovery without an LLM" above for why rapamycin has hallmarks but no chain.

## Pointing an agent at it

Give the agent the base URL plus `/openapi.json` (or the `/docs` page) —
that's enough for most HTTP-capable agents to discover the endpoints and
call `/query` on their own. Good first calls: `/health` (confirms the server
is ready before it starts spending OpenAI calls) and `/patients` (valid
`<Patient>` IDs for the forms above).

This satisfies a local or otherwise network-reachable agent integration. A
shared-secret key check and a per-address rate limit ship with it, both off by
default — see "Operational controls" above for how to turn them on and what
they are worth. Neither replaces the rest of a deployment: a public URL, TLS,
process supervision and a reverse proxy are still yours to provide, and the
rate limit in particular is a per-process courtesy limit, not a defence. With
`PLN_API_KEY` unset the endpoints are unauthenticated by design, so keep the
listener within the intended private network.

## Testing

```bash
python3 -m pytest tests/ -q          # everything: ~470 tests, ~90 s
python3 -m pytest tests/ -q -m slow  # only the ones that drive a real engine
```

`tests/` is the matrix; each module's docstring says which finding it guards
and what was measured. The fast HTTP + combined-mount subset is
`pytest tests/test_api.py tests/test_combined_app.py -q`, and
`scripts/test_pln_api.py` is the black-box runner for a live deployment.

You need `hyperon==0.2.10`, `fastapi`, `httpx`, `pytest` and `pandas`. `gradio`
is optional — its tests skip without it — and `bash scripts/run_etl.sh`
generates `build/`, without which the DrugAge-ranking tests skip and the
endpoints answer 503 rather than guessing.

(This section used to link `docs/api_testing.md`, which has never existed in
this repository.)
