#!/usr/bin/env python3
"""Black-box acceptance runner for the PLN Longevity HTTP API.

The service must already be running. The script uses only Python's standard
library and never writes to the ontology. It exits non-zero when a required
check fails.

Examples:
    python scripts/test_pln_api.py
    python scripts/test_pln_api.py --base-url https://example.ngrok-free.app
    python scripts/test_pln_api.py --require-drugage --require-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


JsonObject = dict[str, Any]
Predicate = Callable[[int, JsonObject], tuple[bool, str]]


@dataclass
class Result:
    name: str
    outcome: str
    status: Optional[int] = None
    elapsed_ms: Optional[int] = None
    note: str = ""


@dataclass
class Runner:
    client: "APIClient"
    results: list[Result] = field(default_factory=list)

    def check(
        self,
        name: str,
        method: str,
        path: str,
        *,
        payload: Optional[JsonObject] = None,
        predicate: Predicate,
    ) -> Optional[JsonObject]:
        status, body, elapsed_ms = self.client.request(method, path, payload=payload)
        passed, note = predicate(status, body)
        outcome = "PASS" if passed else "FAIL"
        self.results.append(Result(name, outcome, status, elapsed_ms, note))
        print(f"[{outcome}] {name} ({status}, {elapsed_ms} ms) {note}")
        return body if passed else None

    def skip(self, name: str, note: str) -> None:
        self.results.append(Result(name, "SKIP", note=note))
        print(f"[SKIP] {name} {note}")


class APIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[JsonObject] = None,
    ) -> tuple[int, JsonObject, int]:
        url = urllib.parse.urljoin(f"{self.base_url}/", path.lstrip("/"))
        headers = {
            "Accept": "application/json",
            "User-Agent": "pln-api-acceptance/1.0",
            "ngrok-skip-browser-warning": "1",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except urllib.error.URLError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            return 0, {"detail": {"message": str(exc.reason)}}, elapsed

        elapsed = int((time.monotonic() - started) * 1000)
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"raw": raw.decode("utf-8", errors="replace")}
        return status, body, elapsed


def status_is(expected: int) -> Predicate:
    def predicate(status: int, body: JsonObject) -> tuple[bool, str]:
        if status == expected:
            return True, ""
        return False, f"expected HTTP {expected}; body={short(body)}"

    return predicate


def json_check(
    expected_status: int,
    check: Callable[[JsonObject], bool],
    description: str,
) -> Predicate:
    def predicate(status: int, body: JsonObject) -> tuple[bool, str]:
        if status != expected_status:
            return False, f"expected HTTP {expected_status}; body={short(body)}"
        if not check(body):
            return False, f"expected {description}; body={short(body)}"
        return True, ""

    return predicate


def atoms(body: JsonObject) -> str:
    return " ".join(
        str(item.get("atom", ""))
        for item in body.get("pln_results", body.get("results", []))
        if isinstance(item, dict)
    )


def short(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else f"{text[:limit]}…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PLN_API_URL", "http://127.0.0.1:7860"),
        help="API root URL; PLN_API_URL is used when set.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Do not fail when the service is in canned stub mode.",
    )
    parser.add_argument(
        "--require-drugage",
        action="store_true",
        help="Fail instead of skipping when build/drugage_etl.metta is absent.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip live /query checks that spend OpenAI tokens.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if a live OpenAI translator check cannot run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Use only with synthetic/de-identified data.",
    )
    args = parser.parse_args()
    if args.skip_llm and args.require_llm:
        parser.error("--skip-llm and --require-llm cannot be combined")
    return args


def main() -> int:
    args = parse_args()
    runner = Runner(APIClient(args.base_url, args.timeout))

    health = runner.check(
        "health",
        "GET",
        "/health",
        predicate=json_check(
            200,
            lambda body: body.get("status") == "ok",
            "status=ok",
        ),
    )
    if health is None:
        return finish(runner, args.output)

    runner.check(
        "ontology discovery",
        "GET",
        "/ontology/files",
        predicate=json_check(
            200,
            lambda body: "patient_profile.metta" in body.get("files", []),
            "patient_profile.metta in files",
        ),
    )
    runner.check(
        "patient discovery",
        "GET",
        "/patients",
        predicate=json_check(
            200,
            lambda body: {"Patient001", "Patient002"}.issubset(
                {item.get("id") for item in body.get("patients", [])}
            ),
            "Patient001 and Patient002",
        ),
    )

    runner.check(
        "empty query rejected",
        "POST",
        "/query",
        payload={"message": "   "},
        predicate=status_is(422),
    )
    runner.check(
        "unsupported model rejected before OpenAI call",
        "POST",
        "/query",
        payload={"message": "test", "model": "not-a-real-model"},
        predicate=status_is(422),
    )
    runner.check(
        "unknown MeTTa symbol rejected before execution",
        "POST",
        "/metta/run",
        payload={"metta_query": "!(DefinitelyNotInTheOntology Foo)"},
        predicate=json_check(
            422,
            lambda body: body.get("detail", {}).get("code") == "invalid_metta_query",
            "detail.code=invalid_metta_query",
        ),
    )
    runner.check(
        "ontology path traversal rejected",
        "POST",
        "/ontology/apply",
        payload={
            "metta_block": "(InstanceOf ApiTraversalProbe Type)",
            "target_file": "../api_escape_test",
        },
        predicate=status_is(422),
    )

    runtime_ready = bool(health.get("runtime_ready"))
    if not runtime_ready:
        note = (
            f"mode={health.get('pln_mode')} "
            f"runtime_importable={health.get('runtime_importable')}"
        )
        if args.allow_stub:
            runner.skip("runtime semantic checks", note)
        else:
            runner.results.append(Result("runtime ready", "FAIL", note=note))
            print(f"[FAIL] runtime ready {note}")
    else:
        raw_cases: list[tuple[str, JsonObject, Predicate]] = [
            (
                "calibration retrieval",
                {"metta_query": "!(evidence-confidence InVitro)"},
                json_check(
                    200,
                    lambda body: body.get("pln_status") == "ok"
                    and "0.35" in atoms(body),
                    "runtime result containing 0.35",
                ),
            ),
            (
                "honest empty causal result",
                {
                    "metta_query": (
                        "!(infer &self TelomereAttrition AgeAccelGrim)"
                    )
                },
                json_check(
                    200,
                    lambda body: body.get("pln_status") == "empty",
                    "pln_status=empty",
                ),
            ),
            (
                "static patient risk",
                {
                    "metta_query": (
                        "!(predict-risk-patient &self Patient002)"
                    )
                },
                json_check(
                    200,
                    lambda body: body.get("pln_status") == "ok"
                    and "RiskPrediction" in atoms(body)
                    and "Patient002" in atoms(body),
                    "Patient002 RiskPrediction",
                ),
            ),
            (
                "ephemeral patient data",
                {
                    "metta_query": (
                        "!(predict-risk-patient &self ApiPatient)"
                    ),
                    "extra_atoms": "\n".join([
                        "(InstanceOf ApiPatient PatientProfile)",
                        "(PatientAge ApiPatient 58)",
                        "(PatientSex ApiPatient Male)",
                        "(PatientSmoking ApiPatient NeverSmoker)",
                        "(MeasuredZ ApiPatient AgeAccelGrim 1.2)",
                    ]),
                },
                json_check(
                    200,
                    lambda body: body.get("pln_status") == "ok"
                    and "ApiPatient" in atoms(body),
                    "risk result for ephemeral ApiPatient",
                ),
            ),
        ]
        for name, payload, predicate in raw_cases:
            runner.check(
                name,
                "POST",
                "/metta/run",
                payload=payload,
                predicate=predicate,
            )

    drugage_ready = bool(health.get("drugage_build_available"))
    if drugage_ready and runtime_ready:
        runner.check(
            "DrugAge scoped ranking",
            "POST",
            "/drugage/rank",
            payload={"compounds": ["Resveratrol", "Acarbose"]},
            predicate=json_check(
                200,
                lambda body: body.get("status") == "ok"
                and "Resveratrol" in atoms(body)
                and "Acarbose" in atoms(body),
                "ranking containing Resveratrol and Acarbose",
            ),
        )
    elif args.require_drugage:
        runner.results.append(Result(
            "DrugAge scoped ranking",
            "FAIL",
            note="health reports drugage_build_available=false",
        ))
        print("[FAIL] DrugAge scoped ranking build is unavailable")
    else:
        runner.skip(
            "DrugAge scoped ranking",
            "build/drugage_etl.metta is unavailable",
        )

    llm_ready = bool(health.get("openai_key_configured"))
    if not args.skip_llm and llm_ready and runtime_ready:
        runner.check(
            "live natural-language translation and execution",
            "POST",
            "/query",
            payload={
                "message": (
                    "What is Patient002's 10-year cardiovascular risk?"
                ),
                "temperature": 0.0,
                "show_debug": False,
            },
            predicate=json_check(
                200,
                lambda body: (
                    body.get("intent") == "inference"
                    and "predict-risk-patient" in body.get("metta_query", "")
                    and "Patient002" in body.get("metta_query", "")
                    and body.get("pln_status") == "ok"
                ),
                "risk translation for Patient002 with successful PLN execution",
            ),
        )
    elif args.require_llm:
        runner.results.append(Result(
            "live natural-language translation and execution",
            "FAIL",
            note=(
                "OpenAI key/runtime unavailable"
                if not args.skip_llm
                else "disabled by --skip-llm"
            ),
        ))
        print("[FAIL] live natural-language translation and execution unavailable")
    else:
        runner.skip(
            "live natural-language translation and execution",
            "disabled or OpenAI/runtime unavailable",
        )

    return finish(runner, args.output)


def finish(runner: Runner, output: Optional[Path]) -> int:
    counts = {
        outcome: sum(result.outcome == outcome for result in runner.results)
        for outcome in ("PASS", "FAIL", "SKIP")
    }
    report = {
        "base_url": runner.client.base_url,
        "summary": counts,
        "results": [result.__dict__ for result in runner.results],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {output}")
    print(
        f"Summary: {counts['PASS']} passed, "
        f"{counts['FAIL']} failed, {counts['SKIP']} skipped"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
