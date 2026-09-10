"""CommunicationGovernor: Independent Communication Optimization & Deduplication Module.

Decoupled from MemoryController to keep the memory governance policy clean and focused
on discrete action space {visibility, supersedes}. This module operates at the MAS
communication layer (AgentGraph / Message Channel) to suppress redundant restatement of
facts already present in active global memory cards.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


class CommunicationGovernor:
    """Zero-inference, keyword/fingerprint-based communication deduplicator."""

    def __init__(self, echo_similarity_threshold: float = 0.60):
        self.echo_similarity_threshold = echo_similarity_threshold
        self.stats = {
            "total_messages_processed": 0,
            "messages_compressed": 0,
            "raw_characters": 0,
            "compressed_characters": 0,
        }

    def _extract_keywords(self, text: str) -> Set[str]:
        stopwords = {"for", "the", "and", "that", "was", "with", "from", "this", "have", "been", "all", "are", "just"}
        words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    def filter_message(
        self,
        sender_id: str,
        receiver_id: str,
        message: str,
        active_memory_cards: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, bool]:
        """Inspect and compress message if it mechanically echoes public memory cards.

        Args:
            sender_id: ID of the sender agent.
            receiver_id: ID of the receiver agent.
            message: The raw natural language message text.
            active_memory_cards: List of dicts with 'id', 'key' / 'title', 'summary'.

        Returns:
            (filtered_message, was_compressed)
        """
        if not message or not active_memory_cards:
            return message, False

        self.stats["total_messages_processed"] += 1
        raw_len = len(message)
        self.stats["raw_characters"] += raw_len

        msg_keywords = self._extract_keywords(message)
        if not msg_keywords:
            self.stats["compressed_characters"] += raw_len
            return message, False

        echoed_cards: List[str] = []
        for card in active_memory_cards:
            card_id = card.get("id", "")
            card_key = card.get("key", "") or card.get("title", "")
            if not card_key:
                continue

            card_keywords = self._extract_keywords(card_key)
            if not card_keywords:
                continue

            intersection = msg_keywords.intersection(card_keywords)
            overlap_ratio = len(intersection) / float(len(card_keywords))
            if overlap_ratio >= self.echo_similarity_threshold:
                echoed_cards.append(f"{card_id} ({card_key})")

        words_count = len(message.split())
        if echoed_cards and words_count > 15:
            refs = ", ".join(echoed_cards[:2])
            conclusion_match = re.search(
                r"(?:recommend|suggest|next step|conclude|therefore|focus on|investigate)\s+.*",
                message,
                re.IGNORECASE,
            )
            tail = conclusion_match.group(0) if conclusion_match else "Proceed with recommended investigation."
            compressed_msg = f"[Governor Ref: {refs}] {tail}"

            self.stats["messages_compressed"] += 1
            self.stats["compressed_characters"] += len(compressed_msg)
            return compressed_msg, True

        self.stats["compressed_characters"] += raw_len
        return message, False

    def get_efficiency_metrics(self) -> Dict[str, float]:
        if self.stats["total_messages_processed"] == 0:
            return {
                "compression_rate": 0.0,
                "messages_compressed_count": 0,
                "total_messages": 0,
            }
        raw = max(1, self.stats["raw_characters"])
        comp = self.stats["compressed_characters"]
        return {
            "compression_rate": round(1.0 - (comp / raw), 4),
            "messages_compressed_count": self.stats["messages_compressed"],
            "total_messages": self.stats["total_messages_processed"],
        }
