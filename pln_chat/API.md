# PLN Query API

A plain HTTP/JSON API for `pln_chat`, meant for scripts and agents to call
directly instead of driving the Gradio UI through a browser. It wraps the
exact same pipeline as the "PLN Query" and "Ontology Expander" tabs in
`app.py` — same `translate -> validate -> run_query -> format` logic, same
`.metta` files — just returned as structured JSON instead of chat HTML.

It's a separate process from the Gradio UI (`app.py`); run one or both,
on different ports.

## Run it

```bash
cd pln_chat
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY

python api.py
# -> http://0.0.0.0:8000
# interactive docs (try requests from the browser): http://localhost:8000/docs
# raw OpenAPI schema (useful for pointing an agent at the API's shape):
#   http://localhost:8000/openapi.json
```

Equivalent: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`

By default there's no auth, for fast local experimentation. To require a
shared secret (e.g. before exposing this past localhost), set `PLN_API_KEY`
in `.env` and send it back as an `X-API-Key` header on every request.

## Endpoints

| Method | Path               | Purpose                                                    |
|--------|--------------------|--------------------------------------------------------------|
| GET    | `/health`          | Liveness + config check (PLN mode, whether a key is required) |
| GET    | `/ontology/files`  | List discovered `.metta` files                              |
| POST   | `/query`           | Ask a natural-language question of the KB                   |
| POST   | `/ontology/expand` | Extract new ontology entries from pasted paper text          |
| POST   | `/ontology/apply`  | Write a previously-previewed MeTTa block to disk             |

Full request/response schemas are in `/docs` and `/openapi.json` once the
server is running.

## Examples

```bash
curl localhost:8000/health

curl localhost:8000/ontology/files

curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "What interventions might help reduce GrimAge acceleration?"}'
```

Multi-turn: pass the `history` array from a response back into the next
request's `history` field to keep the conversation going.

```bash
curl -X POST localhost:8000/ontology/expand \
  -H 'Content-Type: application/json' \
  -d '{
        "paper_text": "<abstract or full text here>",
        "new_filename": "my_paper_extract",
        "apply": false
      }'
# review the returned metta_block, then either re-call with "apply": true,
# or POST it separately:
curl -X POST localhost:8000/ontology/apply \
  -H 'Content-Type: application/json' \
  -d '{"metta_block": "...", "target_file": "my_paper_extract.metta"}'
```

## Pointing an agent at it

Give the agent the base URL plus `/openapi.json` (or the `/docs` page) —
that's enough for most HTTP-capable agents to discover the endpoints and
call `/query` on their own. `/health` is a good first call to confirm the
server is up and whether `X-API-Key` is required before it starts spending
OpenAI calls on `/query`.
