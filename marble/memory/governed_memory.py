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
                     memory_id: Optional[str] = None) -> None:
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

    def submit(self, proposal: MemoryProposal) -> Optional[MemoryItem]:
        self.trace.log_proposal(proposal)
        current_state: Sequence[MemoryItem] = [
            item
            for item in self.bank.all_items()
            if item.task_id == proposal.task_id and item.active
        ]
        target = self.controller.decide(proposal, current_state)
        item = self.bank.apply(proposal, target)
        self.trace.log_decision(proposal, target,
                                memory_id=item.memory_id if item else None)
        return item

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        cards = self.bank.visible_keys(reader_id, task_id)
        return self.retriever.rank(cards, query=query, top_k=top_k)

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryItem:
        item = self.bank.read(memory_id, reader_id, task_id)
        self.trace.log_read(memory_id, reader_id, task_id)
        return item
