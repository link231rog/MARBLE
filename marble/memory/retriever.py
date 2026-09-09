from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import MemoryCard

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
})

# Feature weights for domain-agnostic key-first ranking
WEIGHT_TARGETED: float = 0.20
WEIGHT_LINEAGE: float = 0.10


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    filtered = tokens - _STOPWORDS
    return filtered if filtered else tokens


def compute_card_relevance(
    card: MemoryCard,
    query: Optional[str] = None,
    reader_id: Optional[str] = None,
    q_tokens: Optional[set[str]] = None,
    weight_targeted: float = WEIGHT_TARGETED,
    weight_lineage: float = WEIGHT_LINEAGE,
) -> Tuple[float, Dict[str, float]]:
    """Compute composite relevance score for a visible memory card.

    Domain-Agnostic Key-First Ranking Formulation:
    Score(Q, C, reader_id) = S_lexical + B_targeted + B_lineage

    1. S_lexical (Normalized Jaccard token overlap in [0.0, 1.0]):
       |Q_tokens & T_tokens| / |Q_tokens | T_tokens|
    2. B_targeted (Targeted Communication Prior, weight=0.20):
       Prioritizes memories explicitly routed by Controller to reader_id
       over ambient global pool broadcast noise.
    3. B_lineage (Version Evolution / Refinement Bonus, weight=0.10):
       Memories updated/superseded via Controller update represent higher-confidence,
       consolidated facts with higher information density than initial raw proposals.
    """
    # 1. Normalized Lexical overlap
    sim = 0.0
    if q_tokens:
        t_tokens = _tokenize(card.title)
        if t_tokens:
            inter = len(q_tokens & t_tokens)
            union = len(q_tokens | t_tokens)
            sim = (inter / union) if union > 0 else 0.0

    # 2. Targeted / Private communication prior
    is_targeted_to_reader = (
        (reader_id is not None and card.visibility == reader_id)
        or (
            card.visibility == "targeted"
            and reader_id is not None
            and reader_id in getattr(card, "target_recipients", ())
        )
    )
    b_targeted = weight_targeted if is_targeted_to_reader else 0.0

    # 3. Lineage / version evolution bonus (controller update/supersedes)
    has_lineage = bool(getattr(card, "supersedes", None) or getattr(card, "update", None))
    b_lineage = weight_lineage if has_lineage else 0.0

    total = sim + b_targeted + b_lineage
    breakdown = {
        "lexical": sim,
        "targeted": b_targeted,
        "lineage": b_lineage,
        "total": total,
    }
    return total, breakdown


def rank_cards(
    cards: Sequence[MemoryCard],
    query: Optional[str] = None,
    top_k: int = 5,
    reader_id: Optional[str] = None,
    weight_targeted: float = WEIGHT_TARGETED,
    weight_lineage: float = WEIGHT_LINEAGE,
) -> List[MemoryCard]:
    """Rank visible key cards by lexical overlap + targeted prior + lineage bonus.

    Maintains deterministic stable tie-breaking on original insertion order.
    """
    if top_k <= 0 or not cards:
        return []

    q_tokens = _tokenize(query) if query else set()

    scored: List[Tuple[float, MemoryCard]] = []
    for c in cards:
        score, _ = compute_card_relevance(
            card=c,
            query=query,
            reader_id=reader_id,
            q_tokens=q_tokens,
            weight_targeted=weight_targeted,
            weight_lineage=weight_lineage,
        )
        scored.append((score, c))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


class KeyRetriever:
    """Deterministic lexical key-first retriever with targeted affinity and lineage evolution."""

    def __init__(
        self,
        weight_targeted: float = WEIGHT_TARGETED,
        weight_lineage: float = WEIGHT_LINEAGE,
    ) -> None:
        self.weight_targeted = weight_targeted
        self.weight_lineage = weight_lineage

    def rank(
        self,
        cards: List[MemoryCard],
        query: Optional[str] = None,
        top_k: int = 5,
        reader_id: Optional[str] = None,
    ) -> List[MemoryCard]:
        return rank_cards(
            cards,
            query=query,
            top_k=top_k,
            reader_id=reader_id,
            weight_targeted=self.weight_targeted,
            weight_lineage=self.weight_lineage,
        )
