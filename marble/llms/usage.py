"""Lightweight, per-context accounting for successful model API calls."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class UsageSnapshot:
    """Serializable counters captured by an API usage meter."""

    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


_active_meters: ContextVar[Tuple["ApiUsageMeter", ...]] = ContextVar(
    "marble_active_api_usage_meters",
    default=(),
)


class ApiUsageMeter:
    """Collect API usage for the current context and any nested work."""

    def __init__(self) -> None:
        self._snapshot = UsageSnapshot()
        self._context_token: Token[Tuple["ApiUsageMeter", ...]] | None = None

    def __enter__(self) -> "ApiUsageMeter":
        if self._context_token is not None:
            raise RuntimeError("An ApiUsageMeter cannot be entered twice concurrently.")
        self._context_token = _active_meters.set((*_active_meters.get(), self))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._context_token is not None:
            _active_meters.reset(self._context_token)
            self._context_token = None

    def snapshot(self) -> Dict[str, int]:
        return self._snapshot.to_dict()

    def _record(self, input_tokens: int, output_tokens: int) -> None:
        self._snapshot.api_calls += 1
        self._snapshot.input_tokens += input_tokens
        self._snapshot.output_tokens += output_tokens


def record_successful_completion(completion: Any) -> None:
    """Add a successful completion's usage to every active meter."""

    usage = _value(completion, "usage")
    input_tokens = _token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens = _token_count(usage, "completion_tokens", "output_tokens")
    for meter in _active_meters.get():
        meter._record(input_tokens=input_tokens, output_tokens=output_tokens)


def _token_count(usage: Any, *field_names: str) -> int:
    for field_name in field_names:
        value = _value(usage, field_name)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _value(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)
