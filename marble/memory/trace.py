from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict


class TraceLogger:
    """Append-only JSONL logger for memory decisions and reads."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        record: Dict[str, Any] = {"event": event, **payload}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_proposal(self, proposal: Any) -> None:
        self.log("memory_proposal", proposal=proposal.__dict__)

    def log_decision(self, proposal: Any, target: Any) -> None:
        self.log(
            "memory_decision",
            proposal=proposal.__dict__,
            target=target.__dict__,
        )

    def log_read(self, memory_id: str, reader_id: str, task_id: str) -> None:
        self.log(
            "memory_read",
            memory_id=memory_id,
            reader_id=reader_id,
            task_id=task_id,
        )
