from __future__ import annotations

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    """Decides which key to evict next.

    A policy stores ordering and frequency metadata only; values live in
    ``Cache``. A policy that can reach a backend is a bug.
    """

    @abstractmethod
    def on_insert(self, key: str) -> None:
        """Record that ``key`` was inserted."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def on_access(self, key: str) -> None:
        """Record that ``key`` was accessed."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def on_remove(self, key: str) -> None:
        """Forget any bookkeeping for ``key``."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def victim(self) -> str:
        """Return the next key to evict, or raise ``KeyError`` if empty."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def clear(self) -> None:
        """Remove all policy bookkeeping."""
        raise NotImplementedError("see issue #1.1")
