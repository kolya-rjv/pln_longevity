"""The documented prompt size must be the measured prompt size.

API.md carried THREE different figures for the default system prompt —
"~73,100", "~62,400" and "~57,400" — in one document. They are not rounding
differences; they disagree by 28%, and the smallest is the one an operator
reads when sizing `PLN_MAX_PROMPT_TOKENS`. Reproduced: with
`PLN_MAX_PROMPT_TOKENS=60000`, chosen as generous headroom over "~57,400",
API.md's own example query comes back **413 `prompt_too_large`** — every
default `/query` on that deployment refused before the LLM is ever called.

So the number lives in exactly one place that a test can read, and this module
reads it out of the prose and compares it with what `build_system_prompt`
actually produces. A KB edit that moves the prompt moves this test, and the
person who moves it updates the sentence.

Run from the repository root:
    pytest tests/test_prompt_size.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLN_CHAT = REPO / "pln_chat"
if str(PLN_CHAT) not in sys.path:
    sys.path.insert(0, str(PLN_CHAT))

pytest.importorskip("fastapi")

import api as api_module  # noqa: E402

API_MD = PLN_CHAT / "API.md"

#: "301,035 characters, about 75,250 estimated tokens"
_DOCUMENTED_RE = re.compile(
    r"([\d,]+)\s+characters,\s+about\s+([\d,]+)\s+estimated\s+tokens"
)

#: How far the real value may drift from the documented one before the
#: sentence is wrong enough to matter. 2% absorbs a comment edit; it does not
#: absorb adding or dropping a layer, which is exactly when the prose should
#: be rewritten.
TOLERANCE = 0.02


def _measured() -> tuple[int, int]:
    selection = api_module._default_selection(
        list(api_module._discover_metta_files().keys())
    )
    registry, raw = api_module._build_context(selection)
    prompt = api_module.build_system_prompt(
        registry, raw, api_module._runtime_inventory()
    )
    return len(prompt), api_module._estimate_tokens(prompt)


def _documented() -> tuple[int, int]:
    match = _DOCUMENTED_RE.search(API_MD.read_text(encoding="utf-8"))
    assert match, "API.md no longer states the default prompt size"
    return (
        int(match.group(1).replace(",", "")),
        int(match.group(2).replace(",", "")),
    )


def test_the_documented_default_prompt_size_is_the_real_one():
    doc_chars, doc_tokens = _documented()
    real_chars, real_tokens = _measured()

    for label, documented, real in (
        ("characters", doc_chars, real_chars),
        ("tokens", doc_tokens, real_tokens),
    ):
        drift = abs(real - documented) / max(documented, 1)
        assert drift <= TOLERANCE, (
            f"API.md says {documented:,} {label}; the prompt is {real:,} "
            f"({drift:.1%} off). Update the sentence in API.md — an operator "
            f"sizes PLN_MAX_PROMPT_TOKENS off it."
        )


def test_api_md_states_only_one_default_prompt_size():
    """Three figures in one document is how the wrong one gets believed."""
    text = API_MD.read_text(encoding="utf-8")
    quoted = set(_DOCUMENTED_RE.findall(text))
    assert len(quoted) == 1, f"API.md quotes several default prompt sizes: {quoted}"


def test_the_default_limit_leaves_room_for_the_default_prompt():
    """The guard must not refuse the service's own default configuration."""
    _, real_tokens = _measured()
    assert api_module.PLN_MAX_PROMPT_TOKENS > real_tokens, (
        f"PLN_MAX_PROMPT_TOKENS={api_module.PLN_MAX_PROMPT_TOKENS:,} is at or "
        f"below the {real_tokens:,}-token default prompt: every /query would "
        f"return 413 before the LLM is called."
    )


# ── every MeTTa example API.md prints must do what the page says ─────────────

def _metta_examples() -> list[str]:
    import re

    return sorted(set(re.findall(
        r"^\s*(!\([^\n]+\))\s*$", API_MD.read_text(encoding="utf-8"), re.M
    )))


def _run(query: str):
    import asyncio

    httpx = pytest.importorskip("httpx")

    async def send():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=180
        ) as client:
            return await client.post("/metta/run", json={"metta_query": query})

    return asyncio.run(send())


@pytest.mark.slow
def test_every_metta_example_in_api_md_behaves_as_documented():
    """Three of them did not, and the worst one looked like an answer.

    `!(cellage-effect &self CellAgeRow_869)` and
    `!(cellage-gene-effects &self Gene_TP53)` are 422s — the rows they name are
    not in the generic space — and `!(genes-affecting-senescence &self
    Increases)` returned a 200 with `pln_status: "empty"` and NOTHING to say it
    was structurally empty, which by API.md's own reading rule means "no genes
    increase senescence". All three are now documented as query-scoped forms,
    and the third carries a warning. Every OTHER example must simply run.
    """
    pytest.importorskip("hyperon")
    examples = _metta_examples()
    assert len(examples) >= 10, "API.md's MeTTa examples went missing"

    # The three the page explicitly documents as reachable only through the
    # query-scoped space GET /genes/{symbol}?infer=true builds.
    scoped = {
        "!(cellage-effect &self CellAgeRow_869)",
        "!(cellage-gene-effects &self Gene_TP53)",
        "!(genes-affecting-senescence &self Increases)",
    }
    page = API_MD.read_text(encoding="utf-8")
    assert "not through `POST /metta/run`" in page, (
        "API.md must say the scoped forms are not /metta/run callable"
    )

    broken: list[str] = []
    for query in examples:
        response = _run(query)
        if query in scoped:
            # Either refused outright, or answered with an explicit warning —
            # never a bare, confident empty.
            if response.status_code == 200:
                body = response.json()
                if body["pln_status"] == "empty" and not body["warnings"]:
                    broken.append(f"{query}: silent empty")
            continue
        if response.status_code != 200:
            broken.append(f"{query}: HTTP {response.status_code}")
        elif response.json()["pln_status"] != "ok":
            broken.append(f"{query}: pln_status={response.json()['pln_status']}")
    assert not broken, "API.md examples that do not work as printed: " + "; ".join(broken)


def test_no_new_default_prompt_file_goes_over_the_per_file_limit_and_is_replaced_by_a_schema_card():
    """A file over PLN_PROMPT_FILE_MAX_BYTES is not pasted into the prompt: the translator gets a schema card of it.
    patient_profile.metta crossed 25,000 bytes with a 1.4 KB addition once, and the prompt silently lost 23,000
    characters of it; only the +-2 % size check noticed. Today exactly one default-selected file is over (it was
    before): a second one fails here, with the file named. Shorten its comments rather than raise the limit."""
    from config import PLN_PROMPT_FILE_MAX_BYTES
    import api as api_module
    files = api_module._discover_metta_files()
    selected = api_module._default_selection(list(files))
    over = {name for name in selected if name in files and files[name].stat().st_size > PLN_PROMPT_FILE_MAX_BYTES}
    assert over == {"pln_risk_prediction.metta"}, (
        f"files over {PLN_PROMPT_FILE_MAX_BYTES} bytes are shown to the translator as a card, not as written: {sorted(over)}")
