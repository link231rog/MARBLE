from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class TraceLogger:
    """Append-only JSONL logger for memory decisions and reads."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_record(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log(self, event: str, **payload: Any) -> None:
        self.log_record({"event": event, **payload})

    def log_proposal(self, proposal: Any) -> None:
        self.log("memory_proposal", proposal=proposal.__dict__)

    def log_decision(self, proposal: Any, target: Any,
                     memory_id: Optional[str] = None,
                     **metadata: Any) -> None:
        record: Dict[str, Any] = {
            "event": "memory_decision",
            "proposal": proposal.__dict__,
            "target": target.__dict__,
        }
        if memory_id is not None:
            record["memory_id"] = memory_id
        record.update(metadata)
        self.log_record(record)

    def log_read(self, memory_id: str, reader_id: str, task_id: str) -> None:
        self.log(
            "memory_read",
            memory_id=memory_id,
            reader_id=reader_id,
            task_id=task_id,
        )
