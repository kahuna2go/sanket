"""Append a structured trade record to trades.jsonl at the project root."""
import json
import pathlib
from datetime import datetime, timezone

_LOG_PATH = pathlib.Path(__file__).parent.parent.parent / "trades.jsonl"


def append(record: dict) -> None:
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
