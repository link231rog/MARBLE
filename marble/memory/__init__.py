from .base_memory import BaseMemory
from .bank import MemoryBank
from .governed_memory import GovernedMemory
from .retriever import KeyRetriever
from .schema import MemoryCard, MemoryItem, MemoryProposal, MemoryTargetState
from .trace import TraceLogger

__all__ = [
    "BaseMemory",
    "GovernedMemory",
    "KeyRetriever",
    "MemoryBank",
    "MemoryCard",
    "MemoryItem",
    "MemoryProposal",
    "MemoryTargetState",
    "TraceLogger",
]

# Keep legacy memory classes available when the optional runtime dependencies
# are installed. The governed core should remain importable on its own.
try:
    from .long_term_memory import LongTermMemory
    from .shared_memory import SharedMemory
    from .short_term_memory import ShortTermMemory
except ImportError:
    pass
else:
    __all__ += ["SharedMemory", "LongTermMemory", "ShortTermMemory"]
