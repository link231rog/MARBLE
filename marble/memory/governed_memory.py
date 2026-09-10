from __future__ import annotations

from typing import List, Optional, Sequence

from .bank import MemoryBank
from .retriever import KeyRetriever
from .schema import MemoryCard, MemoryItem, MemoryProposal
from .trace import TraceLogger


class _NullTrace:
    def log_proposal(self, proposal: object) -> None:
        pass

    def log_decision(self, proposal: object, target: object,
                     memory_id: Optional[str] = None, **metadata: object) -> None:
        pass

    def log_read(self, memory_id: str, reader_id: str, task_id: str) -> None:
        pass


class GovernedMemory:
    """Facade wiring proposal -> controller -> bank -> trace."""

    def __init__(
        self,
        bank: MemoryBank,
        controller,
        retriever: Optional[KeyRetriever] = None,
        trace: Optional[TraceLogger] = None,
    ) -> None:
        self.bank = bank
        self.controller = controller
        self.retriever = retriever or KeyRetriever()
        self.trace = trace if trace is not None else _NullTrace()

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[MemoryItem]:
        self.trace.log_proposal(proposal)
        current_state: Sequence[MemoryItem] = [
            item
            for item in self.bank.all_items()
            if item.task_id == proposal.task_id and item.active
        ]
        metadata.pop("controller_prompt", None)
        metadata.pop("controller_output", None)
        metadata["active_memory_index"] = [
            item.__dict__.copy() for item in current_state
        ]
        target = self.controller.decide(proposal, current_state)
        prompt = getattr(self.controller, "last_prompt", None)
        raw = getattr(self.controller, "last_raw", None)
        parse_status = getattr(self.controller, "last_parse_status", None)
        log_prob = getattr(self.controller, "last_log_prob", None)
        if prompt is not None:
            metadata["controller_prompt"] = prompt
        if raw is not None:
            metadata["controller_output"] = raw
        if log_prob is not None:
            metadata["old_log_prob"] = float(log_prob)
            metadata["controller_log_prob"] = float(log_prob)
        if parse_status is not None:
            metadata["parse_status"] = parse_status
        elif raw is not None or prompt is not None:
            metadata["parse_status"] = "valid_json"
        else:
            metadata["parse_status"] = "not_applicable"
        item = self.bank.apply(proposal, target)
        self.trace.log_decision(proposal, target,
                                memory_id=item.memory_id if item else None,
                                **metadata)
        return item

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryCard]:
        cards = self.bank.visible_keys(reader_id, task_id)
        if hasattr(self.retriever, "rank"):
            try:
                return self.retriever.rank(cards, query=query, top_k=top_k, reader_id=reader_id)
            except TypeError:
                return self.retriever.rank(cards, query=query, top_k=top_k)
        return cards[:top_k]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryItem:
        item = self.bank.read(memory_id, reader_id, task_id)
        self.trace.log_read(memory_id, reader_id, task_id)
        return item
