from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .schema import MemoryCard


def rank_cards(
    cards: Sequence[MemoryCard],
    query: Optional[str] = None,
    top_k: int = 6,
) -> List[MemoryCard]:
    """Rank cards by token overlap between query and title."""
    if top_k <= 0 or not cards:
        return []
    if not query:
        return list(cards[:top_k])
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_tokens:
        return list(cards[:top_k])
    return sorted(
        cards,
        key=lambda c: -len(q_tokens & set(re.findall(r"[a-z0-9]+", c.title.lower()))),
    )[:top_k]


class KeyRetriever:
    """Rank visible key cards by token overlap between query and title."""

    def rank(
        self,
        cards: List[MemoryCard],
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        return rank_cards(cards, query=query, top_k=top_k)
