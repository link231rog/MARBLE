from __future__ import annotations

import re
from typing import List, Optional

from .schema import MemoryCard


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


class KeyRetriever:
    """Rank visible key cards by token overlap between query and title.

    Fixed scorer, not trained. No external dependencies.
    """

    def rank(
        self,
        cards: List[MemoryCard],
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        if top_k <= 0 or not cards:
            return []
        if not query:
            return list(cards[:top_k])
        query_tokens = _tokens(query)
        if not query_tokens:
            return list(cards[:top_k])

        scored = []
        for index, card in enumerate(cards):
            overlap = len(query_tokens & _tokens(card.title))
            scored.append((-overlap, index, card))
        scored.sort(key=lambda entry: (entry[0], entry[1]))
        return [card for _, _, card in scored[:top_k]]
