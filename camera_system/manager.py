from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from queue import Empty
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from .backends.base import CaptureBackend
from .events import CameraEvent, CameraEventQueue, CameraEventType


class CameraManagerState(str, Enum):
    STOPPED = "STOPPED"
    WAITING_FOR_CAMERAS = "WAITING_FOR_CAMERAS"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass(frozen=True, slots=True)
class CameraSpec:
    camera_id: str
    unique_id: str
    profile: str | None = None
    enabled: bool = True
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id darf nicht leer sein.")
        if not self.unique_id.strip():
            raise ValueError("unique_id darf nicht leer sein.")


@dataclass(slots=True)
class CameraRuntime:
    spec: CameraSpec
    present: bool = False
    device: Any | None = None
    backend: CaptureBackend | None = None
    last_error: str | None = None


BackendFactory = Callable[[CameraSpec, Any | None], CaptureBackend]


class CameraManager:
    """Generischer, backendunabhängiger Zustands- und Capture-Manager."""

    def __init__(
        self,
        specs: Iterable[CameraSpec],
        *,
        event_queue: CameraEventQueue | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        enabled_specs = [spec for spec in specs if spec.enabled]

        if not enabled_specs:
            raise ValueError("Mindestens eine aktivierte Kamera ist erforderlich.")

        camera_ids = [spec.camera_id for spec in enabled_specs]
        unique_ids = [spec.unique_id for spec in enabled_specs]

        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera_id muss eindeutig sein.")

        if len(unique_ids) != len(set(unique_ids)):
            raise ValueError("unique_id muss eindeutig sein.")

        self._runtimes_by_id = {
            spec.camera_id: CameraRuntime(spec=spec)
            for spec in enabled_specs
        }
        self._camera_id_by_unique_id = {
            spec.unique_id: spec.camera_id
            for spec in enabled_specs
        }
        self._event_queue = event_queue or CameraEventQueue()
        self._backend_factory = backend_factory
        self._lock = RLock()
        self._started = False
        self._capture_requested = False
        self._paused = False
        self._state = CameraManagerState.STOPPED

    @property
    def event_queue(self) -> CameraEventQueue:
        return self._event_queue

    @property
    def state(self) -> CameraManagerState:
        with self._lock:
            return self._state

    def start(self) -> None:
        with self._lock:
            self._started = True
            self._paused = False
            self._refresh_state_locked()

    def stop(self) -> None:
        with self._lock:
            self._capture_requested = False
            self._paused = False
            self._release_all_locked()
            self._started = False
            self._refresh_state_locked()

    def start_captures(self) -> None:
        with self._lock:
            self._require_started_locked()

            missing = [
                runtime.spec.camera_id
                for runtime in self._runtimes_by_id.values()
                if runtime.spec.required and not runtime.present
            ]

            if missing:
                raise RuntimeError(
                    "Erforderliche Kameras fehlen: " + ", ".join(missing)
                )

            if self._backend_factory is None:
                raise RuntimeError("Es wurde keine backend_factory konfiguriert.")

            self._capture_requested = True
            self._paused = False

            for runtime in self._runtimes_by_id.values():
                if runtime.present:
                    self._ensure_backend_open_locked(runtime)

            self._refresh_state_locked()

    def stop_captures(self) -> None:
        with self._lock:
            self._capture_requested = False
            self._paused = False
            self._release_all_locked()
            self._refresh_state_locked()

    def pause(self) -> None:
        with self._lock:
            self._require_started_locked()

            if not self._capture_requested:
                raise RuntimeError("Capture wurde noch nicht gestartet.")

            self._paused = True
            self._refresh_state_locked()

    def resume(self) -> None:
        with self._lock:
            self._require_started_locked()

            if not self._capture_requested:
                raise RuntimeError("Capture wurde noch nicht gestartet.")

            self._paused = False

            for runtime in self._runtimes_by_id.values():
                if runtime.present:
                    self._ensure_backend_open_locked(runtime)

            self._refresh_state_locked()

    def publish(self, event: CameraEvent) -> None:
        self._event_queue.publish(event)

    def process_next_event(self, timeout: float | None = None) -> CameraEvent:
        try:
            event = self._event_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError("Kein Kameraereignis innerhalb des Timeouts.") from None

        self.handle_event(event)
        return event

    def drain_events(self, limit: int | None = None) -> list[CameraEvent]:
        events = self._event_queue.drain(limit=limit)

        for event in events:
            self.handle_event(event)

        return events

    def handle_event(self, event: CameraEvent) -> bool:
        with self._lock:
            camera_id = self._camera_id_by_unique_id.get(event.unique_id)

            if camera_id is None:
                return False

            runtime = self._runtimes_by_id[camera_id]

            if event.event_type is CameraEventType.CONNECTED:
                runtime.present = True
                runtime.device = event.device
                runtime.last_error = None

                if self._capture_requested and not self._paused:
                    self._ensure_backend_open_locked(runtime)

            elif event.event_type is CameraEventType.DISCONNECTED:
                runtime.present = False
                runtime.device = None
                self._release_backend_locked(runtime)

            else:
                raise ValueError(f"Unbekannter Ereignistyp: {event.event_type!r}")

            self._refresh_state_locked()
            return True

    def read(
        self,
        camera_id: str,
        timeout: float | None = None,
    ) -> tuple[bool, Any]:
        with self._lock:
            runtime = self._runtime_for_id_locked(camera_id)

            if self._paused:
                return False, None

            backend = runtime.backend

            if backend is None or not backend.is_open:
                return False, None

        return backend.read(timeout=timeout)

    def backend_for(self, camera_id: str) -> CaptureBackend | None:
        with self._lock:
            return self._runtime_for_id_locked(camera_id).backend

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cameras = []

            for runtime in self._runtimes_by_id.values():
                backend_open = (
                    runtime.backend is not None
                    and runtime.backend.is_open
                )
                cameras.append(
                    {
                        "camera_id": runtime.spec.camera_id,
                        "unique_id": runtime.spec.unique_id,
                        "profile": runtime.spec.profile,
                        "required": runtime.spec.required,
                        "present": runtime.present,
                        "backend_open": backend_open,
                        "last_error": runtime.last_error,
                    }
                )

            return {
                "state": self._state.value,
                "capture_requested": self._capture_requested,
                "paused": self._paused,
                "cameras": cameras,
            }

    def _runtime_for_id_locked(self, camera_id: str) -> CameraRuntime:
        try:
            return self._runtimes_by_id[camera_id]
        except KeyError:
            raise KeyError(f"Unbekannte camera_id: {camera_id}") from None

    def _require_started_locked(self) -> None:
        if not self._started:
            raise RuntimeError("CameraManager wurde noch nicht gestartet.")

    def _ensure_backend_open_locked(self, runtime: CameraRuntime) -> None:
        if runtime.backend is not None and runtime.backend.is_open:
            return

        if self._backend_factory is None:
            runtime.last_error = "Keine backend_factory konfiguriert."
            return

        try:
            backend = self._backend_factory(runtime.spec, runtime.device)
            backend.open()
            runtime.backend = backend
            runtime.last_error = None
        except Exception as exc:
            runtime.backend = None
            runtime.last_error = f"{type(exc).__name__}: {exc}"

    def _release_backend_locked(self, runtime: CameraRuntime) -> None:
        backend = runtime.backend
        runtime.backend = None

        if backend is None:
            return

        try:
            backend.release()
        except Exception as exc:
            runtime.last_error = f"{type(exc).__name__}: {exc}"

    def _release_all_locked(self) -> None:
        for runtime in self._runtimes_by_id.values():
            self._release_backend_locked(runtime)

    def _refresh_state_locked(self) -> None:
        if not self._started:
            self._state = CameraManagerState.STOPPED
            return

        if self._paused:
            self._state = CameraManagerState.PAUSED
            return

        required = [
            runtime
            for runtime in self._runtimes_by_id.values()
            if runtime.spec.required
        ]
        missing_required = [
            runtime for runtime in required if not runtime.present
        ]
        present_count = sum(
            runtime.present
            for runtime in self._runtimes_by_id.values()
        )

        if missing_required:
            self._state = (
                CameraManagerState.WAITING_FOR_CAMERAS
                if present_count == 0
                else CameraManagerState.DEGRADED
            )
            return

        if self._capture_requested:
            present_runtimes = [
                runtime
                for runtime in self._runtimes_by_id.values()
                if runtime.present
            ]
            all_open = all(
                runtime.backend is not None
                and runtime.backend.is_open
                for runtime in present_runtimes
            )

            self._state = (
                CameraManagerState.RUNNING
                if all_open
                else CameraManagerState.DEGRADED
            )
            return

        self._state = CameraManagerState.READY
