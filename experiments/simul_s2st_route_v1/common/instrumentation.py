"""Call-tree wall-clock profiler installed by monkey patching, then removed.

The Stage09/10/11 runtime is treated as read-only. Timing is obtained by temporarily
replacing attributes on classes, module globals or live instances; :meth:`Patcher.close`
restores the original binding, including deleting instance attributes that did not exist
before.

Measurements are keyed by their full call path (``a/b/c``) so the same callee reached from
two different callers stays separable, which is what makes ``qwen_forward`` under prefill
distinguishable from ``qwen_forward`` under autoregressive decoding.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

try:  # torch is optional so the profiler stays unit-testable without CUDA
    import torch
except ImportError:  # pragma: no cover - torch is always present in the train env
    torch = None  # type: ignore[assignment]


_MISSING = object()


@dataclass
class PathStat:
    """Inclusive and exclusive wall time accumulated for one call path."""

    path: str
    calls: int = 0
    inclusive_seconds: float = 0.0
    exclusive_seconds: float = 0.0

    @property
    def label(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def depth(self) -> int:
        return self.path.count("/")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "label": self.label,
            "depth": self.depth,
            "calls": self.calls,
            "inclusive_seconds": self.inclusive_seconds,
            "exclusive_seconds": self.exclusive_seconds,
        }


class CallTreeTimer:
    """Re-entrant wall-clock timer that records a call tree.

    ``synchronize`` drains the CUDA queue on both span boundaries so asynchronous kernel
    time is charged to the span that launched it rather than to whichever span later
    happens to touch the result.
    """

    def __init__(self, *, device: Any = None, synchronize: bool = True) -> None:
        self._device = device
        self._synchronize = bool(synchronize) and torch is not None
        if self._synchronize:
            is_cuda = getattr(device, "type", None) == "cuda"
            self._synchronize = bool(is_cuda and torch.cuda.is_available())
        self._stats: dict[str, PathStat] = {}
        self._labels: list[str] = []
        self._child_seconds: list[float] = []

    @property
    def synchronizing(self) -> bool:
        return self._synchronize

    def reset(self) -> None:
        if self._labels:
            raise RuntimeError("cannot reset the timer while a span is open")
        self._stats.clear()

    def _sync(self) -> None:
        if self._synchronize:
            torch.cuda.synchronize(self._device)

    @contextmanager
    def span(self, label: str) -> Iterator[None]:
        if "/" in label:
            raise ValueError(f"span labels must not contain '/': {label}")
        self._labels.append(label)
        self._child_seconds.append(0.0)
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - started
            path = "/".join(self._labels)
            children = self._child_seconds.pop()
            self._labels.pop()
            stat = self._stats.get(path)
            if stat is None:
                stat = PathStat(path=path)
                self._stats[path] = stat
            stat.calls += 1
            stat.inclusive_seconds += elapsed
            stat.exclusive_seconds += elapsed - children
            if self._child_seconds:
                self._child_seconds[-1] += elapsed

    def stats(self) -> list[PathStat]:
        return [self._stats[path] for path in sorted(self._stats)]

    def to_dict(self) -> dict[str, object]:
        return {
            "synchronized": self._synchronize,
            "paths": [stat.to_dict() for stat in self.stats()],
        }

    def roots(self) -> list[PathStat]:
        return [stat for stat in self.stats() if "/" not in stat.path]

    def total_seconds(self) -> float:
        return sum(stat.inclusive_seconds for stat in self.roots())

    def merge(self, other: "CallTreeTimer") -> None:
        """Fold another timer's tree into this one (used to aggregate over samples)."""

        for stat in other.stats():
            current = self._stats.get(stat.path)
            if current is None:
                current = PathStat(path=stat.path)
                self._stats[stat.path] = current
            current.calls += stat.calls
            current.inclusive_seconds += stat.inclusive_seconds
            current.exclusive_seconds += stat.exclusive_seconds


@dataclass
class Patcher:
    """Installs timing wrappers and guarantees exact restoration.

    ``timer`` is read at call time, not at patch time, so a long-lived set of class-level
    patches can be redirected to a fresh per-sample timer by assigning ``patcher.timer``.
    """

    timer: CallTreeTimer
    _undo: list[Callable[[], None]] = field(default_factory=list, init=False, repr=False)

    def wrap(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)
        previous = vars(owner).get(name, _MISSING) if hasattr(owner, "__dict__") else _MISSING
        patcher = self

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with patcher.timer.span(label):
                return original(*args, **kwargs)

        setattr(owner, name, wrapper)

        def undo() -> None:
            if previous is _MISSING:
                try:
                    delattr(owner, name)
                except AttributeError:
                    setattr(owner, name, original)
            else:
                setattr(owner, name, previous)

        self._undo.append(undo)

    def wrap_optional(self, owner: Any, name: str, label: str) -> bool:
        """Wrap ``owner.name`` when it exists; report whether it was installed."""

        if owner is None or not hasattr(owner, name):
            return False
        self.wrap(owner, name, label)
        return True

    def close(self) -> None:
        while self._undo:
            self._undo.pop()()

    def __enter__(self) -> "Patcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
