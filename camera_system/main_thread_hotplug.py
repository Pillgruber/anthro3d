from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from .events import CameraEvent, CameraEventQueue


@dataclass(frozen=True, slots=True)
class AVFoundationBindings:
    AVFoundation: Any
    Foundation: Any


def load_avfoundation_bindings() -> AVFoundationBindings:
    """Importiert macOS-Bindings erst beim tatsächlichen Hardwareeinsatz."""

    try:
        import AVFoundation  # type: ignore[import-not-found]
        import Foundation  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "AVFoundation/PyObjC ist in dieser Python-Umgebung nicht verfügbar."
        ) from exc

    return AVFoundationBindings(
        AVFoundation=AVFoundation,
        Foundation=Foundation,
    )


def _safe_device_value(device: Any, method_name: str) -> str | None:
    try:
        value = getattr(device, method_name)()
    except Exception:
        return None

    if value is None:
        return None

    return str(value)


def device_metadata(device: Any) -> dict[str, str]:
    """Extrahiert nur robuste, serialisierbare Geräteinformationen."""

    fields = {
        "name": "localizedName",
        "model_id": "modelID",
        "manufacturer": "manufacturer",
        "device_type": "deviceType",
    }

    metadata: dict[str, str] = {}

    for key, method_name in fields.items():
        value = _safe_device_value(device, method_name)
        if value:
            metadata[key] = value

    return metadata


class AVFoundationMainThreadHotplugObserver:
    """Empfängt AVFoundation-Geräteereignisse zwingend im Hauptthread.

    Der Observer führt keinerlei Capture- oder Scanlogik aus. Er übersetzt
    Verbindungsereignisse lediglich in kleine ``CameraEvent``-Objekte und legt
    sie in einer threadsicheren Queue ab.
    """

    def __init__(
        self,
        event_queue: CameraEventQueue,
        *,
        bindings_loader: Callable[[], AVFoundationBindings] = (
            load_avfoundation_bindings
        ),
    ) -> None:
        self._event_queue = event_queue
        self._bindings_loader = bindings_loader
        self._bindings: AVFoundationBindings | None = None
        self._center: Any | None = None
        self._observer_tokens: list[Any] = []
        self._callbacks: list[Callable[[Any], None]] = []
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Der AVFoundation-Hot-Plug-Observer muss im Hauptthread laufen."
            )

    @staticmethod
    def _unique_id(device: Any) -> str:
        unique_id = _safe_device_value(device, "uniqueID")

        if not unique_id:
            raise RuntimeError(
                "AVFoundation-Gerät besitzt keine verwendbare uniqueID."
            )

        return unique_id

    def start(self, *, seed_existing: bool = True) -> None:
        self._require_main_thread()

        if self._started:
            return

        bindings = self._bindings_loader()
        avf = bindings.AVFoundation
        foundation = bindings.Foundation
        center = foundation.NSNotificationCenter.defaultCenter()

        def connected_callback(notification: Any) -> None:
            device = notification.object()
            unique_id = self._unique_id(device)
            self._event_queue.publish(
                CameraEvent.connected(
                    unique_id,
                    device=device,
                    metadata=device_metadata(device),
                )
            )

        def disconnected_callback(notification: Any) -> None:
            device = notification.object()
            unique_id = self._unique_id(device)
            self._event_queue.publish(
                CameraEvent.disconnected(
                    unique_id,
                    metadata=device_metadata(device),
                )
            )

        main_queue = foundation.NSOperationQueue.mainQueue()
        connected_token = center.addObserverForName_object_queue_usingBlock_(
            avf.AVCaptureDeviceWasConnectedNotification,
            None,
            main_queue,
            connected_callback,
        )
        disconnected_token = center.addObserverForName_object_queue_usingBlock_(
            avf.AVCaptureDeviceWasDisconnectedNotification,
            None,
            main_queue,
            disconnected_callback,
        )

        self._bindings = bindings
        self._center = center
        self._observer_tokens = [connected_token, disconnected_token]
        # PyObjC-Blöcke explizit halten, solange die Observer aktiv sind.
        self._callbacks = [connected_callback, disconnected_callback]
        self._started = True

        if seed_existing:
            self.seed_existing_devices()

    def seed_existing_devices(self) -> int:
        self._require_main_thread()

        if not self._started or self._bindings is None:
            raise RuntimeError("Hot-Plug-Observer wurde noch nicht gestartet.")

        avf = self._bindings.AVFoundation
        devices = list(
            avf.AVCaptureDevice.devicesWithMediaType_(avf.AVMediaTypeVideo)
        )

        for device in devices:
            unique_id = self._unique_id(device)
            self._event_queue.publish(
                CameraEvent.connected(
                    unique_id,
                    device=device,
                    metadata=device_metadata(device),
                )
            )

        return len(devices)

    def run_once(self, timeout: float = 0.25) -> None:
        self._require_main_thread()

        if timeout < 0:
            raise ValueError("timeout darf nicht negativ sein.")

        if not self._started or self._bindings is None:
            raise RuntimeError("Hot-Plug-Observer wurde noch nicht gestartet.")

        foundation = self._bindings.Foundation
        until = foundation.NSDate.dateWithTimeIntervalSinceNow_(timeout)
        foundation.NSRunLoop.currentRunLoop().runMode_beforeDate_(
            foundation.NSDefaultRunLoopMode,
            until,
        )

    def stop(self) -> None:
        self._require_main_thread()

        if not self._started:
            return

        if self._center is not None:
            for token in self._observer_tokens:
                self._center.removeObserver_(token)

        self._observer_tokens.clear()
        self._callbacks.clear()
        self._center = None
        self._bindings = None
        self._started = False
