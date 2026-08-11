"""Runtime-exact prompt/KV building blocks for streaming UniSS v2.

This package is intentionally independent from the historical demos.  It is a
small integration seam that can be used by the repaired runtime without
changing any previous experiment or checkpoint loader.
"""

from .session import (
    CommittedTick,
    KVAppendResult,
    PersistentPromptSession,
    SessionPhase,
    TickObservation,
)

__all__ = [
    "CommittedTick",
    "KVAppendResult",
    "PersistentPromptSession",
    "SessionPhase",
    "TickObservation",
]
