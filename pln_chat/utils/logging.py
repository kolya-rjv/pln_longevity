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
    try:
        with _log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("Could not write to session log file: %s", _log_file)
