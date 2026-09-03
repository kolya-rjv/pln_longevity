"""Session-level JSONL logging for PLN chat turns."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_session_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
_log_file = LOGS_DIR / f"session_{_session_id}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pln_chat")


def _append_record(record: dict) -> None:
    """Append one JSON record without allowing logging failures to break a call."""
    try:
        with _log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("Could not write to session log file: %s", _log_file)


def log_query(user_message: str) -> None:
    """Append the raw user input query as soon as it's received.

    Logged independently of `log_turn` so the input is captured even if
    translation or PLN execution later fails or raises.
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event":     "input_query",
        "user":      user_message,
    }
    _append_record(record)


def log_turn(user_message: str, translation, pln_result) -> None:
    """Append a single conversation turn as a JSON line to the session log."""
    record = {
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "user":        user_message,
        "metta_query": getattr(translation, "metta_query", ""),
        "intent":      getattr(translation, "intent", ""),
        "pln_status":  getattr(pln_result, "status", ""),
        "pln_mode":    getattr(pln_result, "mode", ""),
        "error":       getattr(translation, "error", None),
    }
    _append_record(record)


def log_http_request(
    *,
    method: str,
    path: str,
    query: str,
    body: str,
    status_code: int,
    duration_ms: int,
    client: str | None,
    content_type: str | None,
    user_agent: str | None,
    error: str | None = None,
) -> None:
    """Log every HTTP API request, including its raw request body."""
    _append_record({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event": "http_request",
        "method": method,
        "path": path,
        "query": query,
        "body": body,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "client": client,
        "content_type": content_type,
        "user_agent": user_agent,
        "error": error,
    })
