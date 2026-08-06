from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Mapping


class CameraEventType(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class CameraEvent:
    event_type: CameraEventType
    unique_id: str
    device: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.unique_id.strip():
            raise ValueError("unique_id darf nicht leer sein.")

        if (
            self.event_type is CameraEventType.DISCONNECTED
            and self.device is not None
        ):
            raise ValueError(
                "Ein disconnect-Ereignis darf kein device-Objekt enthalten."
            )

    @classmethod
    def connected(
        cls,
        unique_id: str,
        *,
        device: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CameraEvent":
        return cls(
            event_type=CameraEventType.CONNECTED,
            unique_id=unique_id,
            device=device,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def disconnected(
        cls,
        unique_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CameraEvent":
        return cls(
            event_type=CameraEventType.DISCONNECTED,
            unique_id=unique_id,
            metadata={} if metadata is None else dict(metadata),
        )


class CameraEventQueue:
    """Kleine threadsichere Fassade um queue.Queue."""

    def __init__(self) -> None:
        self._queue: Queue[CameraEvent] = Queue()

    def publish(self, event: CameraEvent) -> None:
        if not isinstance(event, CameraEvent):
            raise TypeError("Es dürfen nur CameraEvent-Objekte publiziert werden.")
        self._queue.put(event)

    def get(self, timeout: float | None = None) -> CameraEvent:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> CameraEvent:
        return self._queue.get_nowait()

    def drain(self, limit: int | None = None) -> list[CameraEvent]:
        if limit is not None and limit < 0:
            raise ValueError("limit darf nicht negativ sein.")

        events: list[CameraEvent] = []

        while limit is None or len(events) < limit:
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break

        return events

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
