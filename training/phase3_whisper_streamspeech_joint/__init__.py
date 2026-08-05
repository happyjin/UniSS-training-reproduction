"""Single-stage Phase3 + StreamSpeech joint-training components.

This package is intentionally isolated from the historical UniSS Phase1/2/3
and multi-stage streaming implementations.  Nothing in the old entrypoints
imports this package.
"""

from .config import JointLossWeights, MultiChunkConfig

__all__ = ["JointLossWeights", "MultiChunkConfig"]
