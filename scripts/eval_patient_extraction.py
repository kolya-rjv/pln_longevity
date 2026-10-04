"""Live evaluation of the My Patient model reader (core/patient_read.py + core/patient_extract.py).

    python scripts/eval_patient_extraction.py                    # PLN_EXTRACT_MODEL, everything
    python scripts/eval_patient_extraction.py --model gpt-5.4-mini --limit 40
    python scripts/eval_patient_extraction.py --record           # also write the replay fixture

Needs OPENAI_API_KEY (pln_chat/.env). It is a script, not a test: it calls OpenAI once per
text (about 950 texts; the ~3k-token instructions are a cached prefix).

1. One schema-acceptance call. A 400 (the model refuses the strict schema or a
   parameter) is a configuration error: exit 2 before anything else runs.
2. Every corpus entry (tests/test_patient_text_corpus.py: smoking and diagnoses, each
   with its expected outcome) and every reproduction quoted in
   docs/patient_extraction/review_round4.json, read by the rules alone and by
   read_patient with the model.
3. Reported: agreement with the corpus's expected outcomes (rules alone and with the
   model), the rewrite rate, suggestions and blocks, what the model changed on the
   round-4 texts, why items were discarded, p50/p95 latency and tokens.
4. Exit 1 if any expected-refused text became usable without a click, or any text
   came back read by the rules only (a model error).

Writes docs/patient_extraction/eval_<model>.md and .json; with --record, the raw
extractions to tests/fixtures/patient_extractions.json.gz, which tests replay offline.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import gzip
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pln_chat"))
sys.path.insert(0, str(REPO / "tests"))
OUT = REPO / "docs" / "patient_extraction"
FIXTURE = REPO / "tests" / "fixtures" / "patient_extractions.json.gz"

from core import patient_extract as px  # noqa: E402
from core.patient_read import read_patient  # noqa: E402
from core.patient_text import read_patient_text  # noqa: E402


def corpus() -> list[dict]:
    from test_patient_text_corpus import DIAGNOSES, SMOKING, _STATUS, _text

    out = []
    for text, expect in SMOKING:
        out.append({"text": _text(text), "source": "corpus:smoking", "refused": expect == "X",
                    "expect": None if expect == "X" else list(_STATUS[expect])})
    for text, ok, items in DIAGNOSES:
        out.append({"text": _text(text), "source": "corpus:diagnoses", "refused": not ok,
                    "expect": None if not ok else items})
    return out


#: the first line of a round-4 reproduction that is itself an age/sex header
HEADER = r"""(?i)^(?:\d{2}\s*(?:year|yr|y|,|m\b|f\b)|age\b|aged\b|(?:male|female|man|woman|sex|gender)\b|i'?m \d|[mf]\s*,?\s*\d)"""


def round4() -> list[dict]:
    d = json.loads((OUT / "review_round4.json").read_text(encoding="utf-8"))
    seen, out = set(), []
    for section in ("confirmed", "critic_unverified"):
        for n, f in enumerate(d[section]):
            for a, b in re.findall(r"'([^'\n]{3,200})'|\"([^\"\n]{3,200})\"", f["reproduction"]):
                t = (a or b).replace("\\n", "\n")
                header = re.match(HEADER, t)            # a header test brings its own age and sex
                t = t if header else "58 year old male\n" + t
                if t not in seen:
                    seen.add(t)
                    out.append({"text": t, "source": f"round4:{section[:4]}{n}", "refused": None,
                                "expect": None, "finding": f["title"]})
    return out


def _matches(entry: dict, p) -> bool:
    """The corpus's own check: the expected values, and usable."""
    if not p.ok:
        return False
    if entry["source"] == "corpus:smoking":
        return [p.smoking, p.cotinine_level] == entry["expect"]
    return all(p.questionnaire.get(k) == v for k, v in entry["expect"].items())


def _schema_acceptance(model: str) -> None:
    ex = px.OpenAIExtractor(model=model)
    if ex.error is not None:
        print(f"configuration error: {ex.error.message}")
        sys.exit(2)
    try:
        content, finish, refusal, usage, seconds = ex._request("58 year old male, never smoked\nalbumin 4.1 g/dL")
        out = ex._parse(content, finish, refusal, usage, seconds)
    except px.ExtractError as exc:
        print(f"schema acceptance failed ({exc.code}): {exc.message}")
        sys.exit(2 if exc.is_config else 1)
    print(f"schema accepted by {model}: {len(out.items)} items in {seconds:.1f} s, usage {usage}")


def _read(entry: dict, ex) -> dict:
    text = entry["text"]
    for attempt in range(4):
        t0 = time.monotonic()
        r = read_patient(text, ex)
        if r.model_error is None or r.model_error.code not in px.TRANSIENT_ERRORS:
            break
        time.sleep(2 * (attempt + 1))
    rules = read_patient_text(text)
    raw = px._cache_get(px.cache_key(ex.model, text))
    return {
        **entry,
        "seconds": time.monotonic() - t0 if not r.cached else r.latency_s,
        "model_seconds": r.latency_s, "usage": r.usage,
        "model_error": None if r.model_error is None else [r.model_error.code, r.model_error.message],
        "rules_ok": rules.ok, "ok": r.parsed.ok,
        "rules_match": _matches(entry, rules) if entry["expect"] is not None else None,
        "match": _matches(entry, r.parsed) if entry["expect"] is not None else None,
        "read_as": r.read_as if r.read_as != text else None,
        "rewrites": len(r.substitutions),
        "unread_statements": sum(st.outcome in ("not_understood", "partly_read") for st in rules.statements),
        "suggestions": [s.as_dict() for s in r.suggestions],
        "model_problems": [[x.kind, x.topic, str(x)] for x in r.parsed.all_problems()
                           if str(x) not in {str(y) for y in rules.all_problems()}],
        "notes": r.notes, "discarded": r.discarded,
        "values": {"smoking": [r.parsed.smoking, r.parsed.cotinine_level], "age": r.parsed.age,
                   "sex": r.parsed.sex},
        "rules_values": {"smoking": [rules.smoking, rules.cotinine_level], "age": rules.age, "sex": rules.sex},
        "items": raw.items if isinstance(raw, px.Extraction) else None,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def report(model: str, rows: list[dict], wall: float) -> tuple[str, list[str]]:
    failures = []
    errors = [r for r in rows if r["model_error"]]
    if errors:
        failures.append(f"{len(errors)} texts came back read by the rules only: "
                        + "; ".join(f"{r['model_error'][0]}" for r in errors[:5]))
    leaked = [r for r in rows if r["refused"] and r["ok"]]
    if leaked:
        failures.append(f"{len(leaked)} expected-refused texts became usable: "
                        + "; ".join(repr(r["text"]) for r in leaked[:5]))
    c = [r for r in rows if r["source"].startswith("corpus")]
    c_ok = [r for r in c if r["expect"] is not None]
    c_x = [r for r in c if r["refused"]]
    r4 = [r for r in rows if r["source"].startswith("round4")]
    secs = sorted(r["model_seconds"] for r in rows if r["model_seconds"])
    tok = collections.Counter()
    for r in rows:
        for k, v in (r["usage"] or {}).items():
            tok[k] += v or 0
    unread = [r for r in rows if r["unread_statements"]]
    disc = collections.Counter(d[2].split(" (")[0] for r in rows for d in r["discarded"])
    disc_kind = collections.Counter(d[1] for r in rows for d in r["discarded"])
    blocked_ok = [r for r in c_ok if r["rules_match"] and not r["ok"]]
    wrong = [r for r in c_ok if r["ok"] and not r["match"]]
    r4_blocked = [r for r in r4 if r["rules_ok"] and not r["ok"]]
    r4_changed = [r for r in r4 if r["rules_ok"] and r["ok"] and r["values"] != r["rules_values"]]

    md = [f"# Model reader — live evaluation ({model})", "",
          f"Run {date.today().isoformat()}: {len(rows)} texts ({len(c)} corpus entries with expected "
          f"outcomes, {len(r4)} round-4 reproductions), wall {wall:.0f} s.", "",
          "## Verdict", "",
          ("**FAIL**: " + " | ".join(failures)) if failures else
          "**PASS**: no expected-refused text became usable without a click, and every text was "
          "read by the model (no rules-only fall-back).", "",
          "## Corpus (expected outcomes)", "",
          "| | rules alone | rules + model |", "|---|---|---|",
          f"| expected refused, still refused | {_pct(sum(not r['rules_ok'] for r in c_x), len(c_x))} | "
          f"{_pct(sum(not r['ok'] for r in c_x), len(c_x))} |",
          f"| expected usable, read as expected | {_pct(sum(bool(r['rules_match']) for r in c_ok), len(c_ok))} | "
          f"{_pct(sum(bool(r['match']) for r in c_ok), len(c_ok))} |",
          f"| expected usable, blocked by the model | — | {_pct(len(blocked_ok), len(c_ok))} |",
          f"| expected usable, usable but different | — | {len(wrong)} |", ""]
    if blocked_ok:
        md += ["Blocked by the model although the corpus expects a usable reading (each needs a click "
               "or a canonical wording):", ""]
        for r in blocked_ok[:40]:
            why = "; ".join(p[2] for p in r["model_problems"])[:300]
            md.append(f"- `{r['text'].splitlines()[-1]}` — {why}")
        md.append("")
    if wrong:
        md += ["Usable but not the corpus's values:", ""]
        md += [f"- `{r['text']!r}` → {r['values']}" for r in wrong[:20]] + [""]
    md += ["## What the model did", "",
           f"- Texts with a statement the rules did not (fully) understand: {len(unread)}; "
           f"rewritten: {_pct(sum(1 for r in unread if r['rewrites']), len(unread))} "
           f"({sum(r['rewrites'] for r in rows)} rewrites in all).",
           f"- Texts with a suggestion: {sum(bool(r['suggestions']) for r in rows)}; with a blocking "
           f"one: {sum(any(s['blocking'] for s in r['suggestions']) for r in rows)}.",
           f"- Round-4 texts the rules read as usable that the model now blocks: "
           f"{_pct(len(r4_blocked), sum(r['rules_ok'] for r in r4))}; usable with a different "
           f"smoking, age or sex: {len(r4_changed)}.",
           f"- Items discarded by the checks: {sum(disc.values())} of "
           f"{sum(len(r['items'] or []) for r in rows)} ({', '.join(f'{k} {v}' for k, v in disc_kind.most_common(8))}).",
           "", "Why items were discarded (top 15):", ""]
    md += [f"- {n} × {why}" for why, n in disc.most_common(15)] + [""]
    md += ["## Latency and tokens", "",
           f"- Model call: p50 {statistics.median(secs):.1f} s, p95 {secs[int(0.95 * (len(secs) - 1))]:.1f} s, "
           f"max {secs[-1]:.1f} s ({len(secs)} calls)." if secs else "- no calls",
           f"- Tokens: prompt {tok['prompt_tokens']:,} (cached {tok['cached_tokens']:,}), completion "
           f"{tok['completion_tokens']:,} (reasoning {tok['reasoning_tokens']:,}); per call "
           f"{tok['prompt_tokens'] // max(1, len(secs)):,} in, {tok['completion_tokens'] // max(1, len(secs)):,} out.",
           ""]
    md += ["## Round 4: texts the model now blocks (sample)", ""]
    for r in r4_blocked[:40]:
        why = "; ".join(p[2] for p in r["model_problems"])[:260]
        md.append(f"- `{r['text'].splitlines()[-1]}` — {why}")
    md.append("")
    if r4_changed:
        md += ["## Round 4: usable, with a different smoking, age or sex than the rules alone", ""]
        md += [f"- `{r['text']!r}`: rules {r['rules_values']} → {r['values']}" for r in r4_changed[:30]] + [""]
    return "\n".join(md), failures


def main() -> None:
    from config import PLN_EXTRACT_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PLN_EXTRACT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="only the first N texts of each set")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=("corpus", "round4"), default=None)
    ap.add_argument("--record", action="store_true", help="write the replay fixture")
    args = ap.parse_args()

    px._CACHE_SIZE = 8192                       # keep every reading for the report and --record
    _schema_acceptance(args.model)
    entries = [] if args.only == "round4" else corpus()
    r4 = [] if args.only == "corpus" else round4()
    if args.limit:
        entries, r4 = entries[:args.limit], r4[:args.limit]
    entries += r4
    ex = px.OpenAIExtractor(model=args.model)
    t0 = time.monotonic()
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(lambda e: _read(e, ex), entries), 1):
            rows.append(row)
            if n % 50 == 0:
                print(f"  {n}/{len(entries)}", flush=True)
    wall = time.monotonic() - t0
    md, failures = report(args.model, rows, wall)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.model}"
    (OUT / f"{stem}.md").write_text(md + "\n", encoding="utf-8")
    (OUT / f"{stem}.json").write_text(json.dumps(
        {"model": args.model, "date": date.today().isoformat(), "prompt_hash": px.prompt_hash(),
         "schema_hash": px.schema_hash(), "rows": [{k: v for k, v in r.items() if k != "items"} for r in rows]},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.record:
        rec = {hashlib.sha256(r["text"].encode()).hexdigest()[:16]: {"text": r["text"], "items": r["items"]}
               for r in rows if r["items"] is not None}
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(FIXTURE, "wt", encoding="utf-8") as fh:
            json.dump({"model": args.model, "prompt_hash": px.prompt_hash(), "schema_hash": px.schema_hash(),
                       "extractions": rec}, fh, ensure_ascii=False, sort_keys=True)
        print(f"recorded {len(rec)} extractions to {FIXTURE.relative_to(REPO)}")
    print(md.split("## What the model did")[0])
    print(f"wrote {(OUT / f'{stem}.md').relative_to(REPO)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
