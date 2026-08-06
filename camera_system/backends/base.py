from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CaptureBackend(Protocol):
    @property
    def is_open(self) -> bool:
        ...

    def open(self) -> None:
        ...

    def read(self, timeout: float | None = None) -> tuple[bool, Any]:
        ...

    def release(self) -> None:
        ...
