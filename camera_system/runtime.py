from __future__ import annotations

from threading import Event
from typing import Protocol

from .events import CameraEvent
from .manager import CameraManager


class MainThreadHotplugObserver(Protocol):
    @property
    def is_started(self) -> bool:
        ...

    def start(self, *, seed_existing: bool = True) -> None:
        ...

    def run_once(self, timeout: float = 0.25) -> None:
        ...

    def stop(self) -> None:
        ...


class MainThreadCameraRuntime:
    """Verbindet Hauptthread-Hot-Plug und CameraManager ohne Capture-Details."""

    def __init__(
        self,
        manager: CameraManager,
        observer: MainThreadHotplugObserver,
    ) -> None:
        self._manager = manager
        self._observer = observer
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self, *, seed_existing: bool = True) -> list[CameraEvent]:
        if self._started:
            return []

        self._manager.start()

        try:
            self._observer.start(seed_existing=seed_existing)
            self._started = True
            return self._manager.drain_events()
        except Exception:
            self._manager.stop()
            raise

    def pump_once(self, timeout: float = 0.25) -> list[CameraEvent]:
        if not self._started:
            raise RuntimeError("CameraRuntime wurde noch nicht gestartet.")

        self._observer.run_once(timeout=timeout)
        return self._manager.drain_events()

    def run_forever(
        self,
        *,
        stop_event: Event | None = None,
        timeout: float = 0.25,
    ) -> None:
        if not self._started:
            raise RuntimeError("CameraRuntime wurde noch nicht gestartet.")

        while stop_event is None or not stop_event.is_set():
            self.pump_once(timeout=timeout)

    def stop(self) -> None:
        if not self._started:
            return

        try:
            self._observer.stop()
        finally:
            self._manager.stop()
            self._started = False
