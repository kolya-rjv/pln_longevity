"""Live evaluation of the My Patient model reader (core/patient_read.py + core/patient_extract.py).

    python scripts/eval_patient_extraction.py                    # PLN_EXTRACT_MODEL, everything
    python scripts/eval_patient_extraction.py --model gpt-5.4-mini --limit 40
    python scripts/eval_patient_extraction.py --record           # also write the replay fixture

Needs OPENAI_API_KEY (pln_chat/.env). It is a script, not a test: it calls OpenAI once per
text (the instructions are a cached prefix).

1. One schema-acceptance call. A 400 (the model refuses the strict schema or a
   parameter) is a configuration error: exit 2 before anything else runs.
2. Every text in docs/patient_extraction/eval_corpus.json — smoking and diagnoses
   phrasings, and the body measurements people type — and every reproduction quoted in
   docs/patient_extraction/review_round4.json, read with the model.
3. Reported: agreement with each entry's expected reading; the texts the old rules
   refused and how the model reads them (a judgement to review, not a failure); why
   items were discarded; p50/p95 latency and tokens.
4. Exit 1 if any text with an expected reading is read as USABLE with a different value
   (a confident, wrong patient), or any text came back without the model (a model error).
   A value that is missing where one was expected (the model skipped a statement: it is
   listed as not used, and LinAge2 imputes and flags the input) is reported, not failed.

Writes docs/patient_extraction/eval_<model>.md and .json; with --record, the raw
extractions to tests/fixtures/patient_extractions.json.gz.
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
OUT = REPO / "docs" / "patient_extraction"
FIXTURE = REPO / "tests" / "fixtures" / "patient_extractions.json.gz"

from core import patient_extract as px  # noqa: E402
from core.patient_read import read_patient  # noqa: E402


def corpus() -> list[dict]:
    d = json.loads((OUT / "eval_corpus.json").read_text(encoding="utf-8"))
    return [{**e, "source": f"corpus:{section}"} for section in ("smoking", "diagnoses", "body")
            for e in d[section]]


#: the first line of a round-4 reproduction that is itself an age/sex header
HEADER = r"""(?i)^(?:\d{2}\s*(?:year|yr|y|,|m\b|f\b)|age\b|aged\b|(?:male|female|man|woman|sex|gender)\b|i'?m \d|[mf]\s*,?\s*\d)"""


def round4() -> list[dict]:
    d = json.loads((OUT / "review_round4.json").read_text(encoding="utf-8"))
    seen, out = set(), []
    for section in ("confirmed", "critic_unverified"):
        for n, f in enumerate(d[section]):
            for a, b in re.findall(r"'([^'\n]{3,200})'|\"([^\"\n]{3,200})\"", f["reproduction"]):
                t = (a or b).replace("\\n", "\n")
                t = t if re.match(HEADER, t) else "58 year old male\n" + t
                if t not in seen:
                    seen.add(t)
                    out.append({"text": t, "source": f"round4:{section[:4]}{n}", "refused": None,
                                "expect": None, "finding": f["title"]})
    return out


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(0.05, 0.002 * abs(b))
    return a == b


def _mismatches(entry: dict, p) -> list[str]:
    """How the reading differs from the entry's expected one ([] = as expected)."""
    e = entry["expect"]
    if entry["source"] == "corpus:smoking":
        got = {"smoking": p.smoking, "cotinine_level": p.cotinine_level}
        return [f"{k} {got[k]!r} (expected {v!r})" for k, v in e.items() if got[k] != v]
    if entry["source"] == "corpus:diagnoses":
        return [f"{k} {p.questionnaire.get(k)!r} (expected {v!r})" for k, v in e.items()
                if p.questionnaire.get(k) != v]
    out = []
    labs = p.labs()
    for k, v in e.items():
        if k == "labs":
            out += [f"{c} {labs.get(c)!r} (expected {x!r})" for c, x in v.items() if not _close(labs.get(c), x)]
        elif k == "questionnaire":
            out += [f"{c} {p.questionnaire.get(c)!r} (expected {x!r})" for c, x in v.items()
                    if p.questionnaire.get(c) != x]
        elif not _close(getattr(p, k), v):
            out.append(f"{k} {getattr(p, k)!r} (expected {v!r})")
    return out


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
        r = read_patient(text, ex)
        if r.model_error is None or r.model_error.code not in px.TRANSIENT_ERRORS:
            break
        time.sleep(2 * (attempt + 1))
    raw = px._cache_get(px.cache_key(ex.model, text))
    p = r.parsed
    return {
        **entry,
        "model_seconds": r.latency_s, "cached": r.cached, "usage": r.usage,
        "model_error": None if r.model_error is None else [r.model_error.code, r.model_error.message],
        "ok": p.ok,
        "mismatches": _mismatches(entry, p) if entry["expect"] is not None else None,
        "problems": [[x.kind, x.topic, str(x)] for x in p.all_problems()],
        "not_used": list(p.not_understood), "discarded": r.discarded,
        "values": {"age": p.age, "sex": p.sex, "smoking": p.smoking, "cotinine_level": p.cotinine_level,
                   "height_cm": p.height_cm, "weight_kg": p.weight_kg, "medications": list(p.medications)},
        "items": raw.items if isinstance(raw, px.Extraction) else None,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def report(model: str, rows: list[dict], wall: float) -> tuple[str, list[str]]:
    failures = []
    errors = [r for r in rows if r["model_error"]]
    if errors:
        failures.append(f"{len(errors)} texts were read without the model: "
                        + "; ".join(f"{r['model_error'][0]}" for r in errors[:5]))
    expected = [r for r in rows if r["expect"] is not None]
    right = [r for r in expected if r["ok"] and not r["mismatches"]]
    wrong = [r for r in expected if r["ok"] and any(" None (" not in m for m in r["mismatches"])]
    missing = [r for r in expected if r["ok"] and r["mismatches"] and r not in wrong]
    asked = [r for r in expected if not r["ok"]]
    if wrong:
        failures.append(f"{len(wrong)} texts read as a usable patient with a different value")
    refused = [r for r in rows if r.get("refused")]
    refused_read = [r for r in refused if r["ok"]]
    r4 = [r for r in rows if r["source"].startswith("round4")]
    disc = collections.Counter(why for r in rows for _, _, why in r["discarded"])
    disc_kind = collections.Counter(kind for r in rows for _, kind, _ in r["discarded"])
    secs = sorted(r["model_seconds"] for r in rows if not r["cached"] and r["model_seconds"])
    tok = collections.Counter()
    for r in rows:
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
            tok[k] += r["usage"].get(k, 0)

    md = [f"# Model reader — live evaluation ({model})", "",
          f"Run {date.today().isoformat()}: {len(rows)} texts ({len(expected)} with an expected reading, "
          f"{len(refused)} the old rules refused, {len(r4)} round-4 reproductions), wall {wall:.0f} s.", "",
          "## Verdict", "",
          ("**PASS**: no text with an expected reading was read as a usable patient with a different value, "
           "and every text was read by the model." if not failures else
           "**FAIL**: " + "; ".join(failures)), "",
          "## Texts with an expected reading", "",
          "| | count |", "|---|---|",
          f"| read as expected | {_pct(len(right), len(expected))} |",
          f"| asked about (not usable until reworded) | {_pct(len(asked), len(expected))} |",
          f"| a value missing (the statement listed as not used) | {_pct(len(missing), len(expected))} |",
          f"| **usable but different** | {_pct(len(wrong), len(expected))} |", ""]
    by_source = collections.defaultdict(lambda: [0, 0])
    for r in expected:
        by_source[r["source"]][1] += 1
        by_source[r["source"]][0] += bool(r["ok"] and not r["mismatches"])
    md += [f"- {s}: {_pct(*v)} as expected" for s, v in sorted(by_source.items())] + [""]
    if wrong:
        md += ["Usable but different (each is a wrong patient):", ""]
        md += [f"- `{r['text']!r}`: {'; '.join(r['mismatches'])}" for r in wrong] + [""]
    if missing:
        md += ["A value missing (the model skipped the statement; it is listed as not used):", ""]
        md += [f"- `{r['text']!r}`: {'; '.join(r['mismatches'])}" for r in missing] + [""]
    if asked:
        md += ["Asked about (the person rewords; nothing wrong is built):", ""]
        md += [f"- `{r['text']!r}`: {'; '.join(p[2] for p in r['problems'])[:240]}" for r in asked[:40]] + [""]
    md += ["## Texts the old rules refused", "",
           f"The model asked about {_pct(len(refused) - len(refused_read), len(refused))} of them and read "
           f"{len(refused_read)} as usable. A usable reading here is a judgement to review, not a failure: the "
           f"rules refused what they could not parse, not only what is ambiguous.", ""]
    md += [f"- `{r['text']!r}` → smoking {r['values']['smoking']} (level {r['values']['cotinine_level']})"
           for r in refused_read[:60]] + [""]
    r4_asked = [r for r in r4 if not r["ok"]]
    md += ["## Round-4 reproductions", "",
           f"The texts the rules reader's fourth review round broke (no expected reading): read as usable "
           f"{_pct(len(r4) - len(r4_asked), len(r4))}, asked about {len(r4_asked)}.", ""]
    md += [f"- `{r['text'].splitlines()[-1]!r}` — {'; '.join(p[2] for p in r['problems'])[:200]}"
           for r in r4_asked[:25]] + [""]
    md += ["## What the checks discarded", "",
           f"- Items discarded: {sum(disc.values())} of {sum(len(r['items'] or []) for r in rows)} "
           f"({', '.join(f'{k} {v}' for k, v in disc_kind.most_common(8))}).", ""]
    md += [f"- {n} × {why}" for why, n in disc.most_common(15)] + [""]
    md += ["## Latency and tokens", "",
           f"- Model call: p50 {statistics.median(secs):.1f} s, p95 {secs[int(0.95 * (len(secs) - 1))]:.1f} s, "
           f"max {secs[-1]:.1f} s ({len(secs)} calls)." if secs else "- no calls",
           f"- Tokens: prompt {tok['prompt_tokens']:,} (cached {tok['cached_tokens']:,}), completion "
           f"{tok['completion_tokens']:,} (reasoning {tok['reasoning_tokens']:,}); per call "
           f"{tok['prompt_tokens'] // max(1, len(secs)):,} in, {tok['completion_tokens'] // max(1, len(secs)):,} out.",
           ""]
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
        by: dict = collections.defaultdict(list)
        for e in entries:
            by[e["source"]].append(e)
        entries = [e for v in by.values() for e in v[:args.limit]]
        r4 = r4[:args.limit]
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
    print(md.split("## Texts the old rules refused")[0])
    print(f"wrote {(OUT / f'{stem}.md').relative_to(REPO)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
