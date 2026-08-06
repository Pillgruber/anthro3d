from __future__ import annotations

import threading

from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

import AVFoundation
import Foundation
import objc


ConnectedCallback = Callable[[dict[str, Any], Any], None]
DisconnectedCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CameraPresence:
    camera_id: str
    unique_id: str
    required: bool
    present: bool
    device_name: str | None


# DIRECT_NOTIFICATION_OBJECT_HANDLING
class _DeviceNotificationObserver(Foundation.NSObject):
    """
    Übergibt das von AVFoundation gemeldete Gerät unmittelbar
    an den CameraHotplugMonitor.
    """

    def initWithHandler_(self, handler):
        self = objc.super(
            _DeviceNotificationObserver,
            self,
        ).init()

        if self is None:
            return None

        self._handler = handler
        return self

    def cameraConnected_(self, notification):
        self._handler(
            True,
            notification.object(),
        )

    def cameraDisconnected_(self, notification):
        self._handler(
            False,
            notification.object(),
        )


class CameraHotplugMonitor:
    """
    Überwacht beliebig viele Kameras anhand ihrer uniqueID.

    Erforderliche Kameras:
        Das System kann warten, bis alle vorhanden sind.

    Optionale Kameras:
        Werden automatisch erkannt, blockieren den Start aber nicht.

    Bereits laufende Kameras:
        Bei Trennung wird on_disconnected aufgerufen.
        Bei Wiederverbindung wird on_connected erneut aufgerufen.
    """

    def __init__(
        self,
        devices: Sequence[Mapping[str, Any]],
        *,
        poll_interval_seconds: float = 1.0,
        on_connected: ConnectedCallback | None = None,
        on_disconnected: DisconnectedCallback | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds muss größer als 0 sein."
            )

        self._devices: dict[str, dict[str, Any]] = {}
        self._required_ids: set[str] = set()

        camera_ids: set[str] = set()

        for position, original in enumerate(devices):
            if not isinstance(original, Mapping):
                raise ValueError(
                    f"Registry-Eintrag {position} ist ungültig."
                )

            if original.get("enabled", True) is False:
                continue

            device = dict(original)

            camera_id = str(
                device.get("camera_id") or f"Kamera_{position}"
            )
            unique_id_raw = device.get("unique_id")

            if not unique_id_raw:
                raise ValueError(
                    f"{camera_id}: unique_id fehlt."
                )

            unique_id = str(unique_id_raw)

            if camera_id in camera_ids:
                raise ValueError(
                    f"camera_id ist doppelt: {camera_id}"
                )

            if unique_id in self._devices:
                raise ValueError(
                    "uniqueID ist mehreren Kameras zugeordnet: "
                    f"{unique_id}"
                )

            camera_ids.add(camera_id)

            device["camera_id"] = camera_id
            device["unique_id"] = unique_id
            device["required"] = bool(
                device.get("required", True)
            )

            self._devices[unique_id] = device

            if device["required"]:
                self._required_ids.add(unique_id)

        self._poll_interval = float(
            poll_interval_seconds
        )

        self._on_connected = on_connected
        self._on_disconnected = on_disconnected

        self._present: dict[str, Any] = {}

        self._lock = RLock()
        self._condition = Condition(self._lock)

        self._stop_event = Event()
        self._wake_event = Event()
        self._initialized_event = Event()

        self._thread: Thread | None = None
        self._observer: Any = None
        self._notification_center: Any = None

    @staticmethod
    def _discover_connected_devices() -> dict[str, Any]:
        """
        Erstellt bei jeder Prüfung eine neue Discovery-Session.

        Dadurch werden keine möglicherweise veralteten
        AVCaptureDevice-Objekte weiterverwendet.
        """
        device_types = []

        for constant_name in (
            "AVCaptureDeviceTypeExternal",
            "AVCaptureDeviceTypeExternalUnknown",
            "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        ):
            value = getattr(
                AVFoundation,
                constant_name,
                None,
            )

            if (
                value is not None
                and value not in device_types
            ):
                device_types.append(value)

        discovery = (
            AVFoundation.AVCaptureDeviceDiscoverySession
            .discoverySessionWithDeviceTypes_mediaType_position_(
                device_types,
                AVFoundation.AVMediaTypeVideo,
                AVFoundation.AVCaptureDevicePositionUnspecified,
            )
        )

        return {
            str(device.uniqueID()): device
            for device in list(discovery.devices())
        }

    @staticmethod
    def _device_name(device: Any) -> str:
        try:
            return str(device.localizedName())
        except Exception:
            return "Unbekannte Kamera"

    def _handle_device_notification(
        self,
        connected: bool,
        device: Any,
    ) -> None:
        """
        Verarbeitet ein angeschlossenes oder getrenntes Gerät direkt
        aus notification.object().

        Die periodische Discovery bleibt zusätzlich als Sicherheitsnetz.
        """
        if device is None:
            self._wake_event.set()
            return

        try:
            unique_id = str(device.uniqueID())
        except Exception:
            self._wake_event.set()
            return

        config = self._devices.get(unique_id)

        # Nicht registrierte Geräte beeinflussen diesen Monitor nicht.
        if config is None:
            self._wake_event.set()
            return

        callback = None

        with self._condition:
            if connected:
                if unique_id in self._present:
                    self._wake_event.set()
                    return

                self._present[unique_id] = device
                callback = self._on_connected

            else:
                if unique_id not in self._present:
                    self._wake_event.set()
                    return

                self._present.pop(unique_id, None)
                callback = self._on_disconnected

            self._condition.notify_all()

        # Benutzer-Callbacks bewusst außerhalb des Locks ausführen.
        if connected:
            print(
                "\nKAMERA VERBUNDEN:"
                f"\n  Rolle:    {config['camera_id']}"
                f"\n  Name:     {self._device_name(device)}"
                f"\n  uniqueID: {unique_id}"
            )

            if callback is not None:
                callback(config, device)

        else:
            print(
                "\nKAMERA GETRENNT:"
                f"\n  Rolle:    {config['camera_id']}"
                f"\n  uniqueID: {unique_id}"
            )

            if callback is not None:
                callback(config)

        # Discovery anschließend zur Kontrolle des Gesamtzustands wecken.
        self._wake_event.set()

    def _register_notifications(self) -> None:
        """
        Registriert AVFoundation-Hot-Plug-Notifications.

        Falls die Registrierung auf einem System nicht funktioniert,
        bleibt die periodische Prüfung als Fallback aktiv.
        """
        try:
            observer = (
                _DeviceNotificationObserver
                .alloc()
                .initWithHandler_(
                    self._handle_device_notification
                )
            )

            center = (
                Foundation.NSNotificationCenter
                .defaultCenter()
            )

            connected_notification = getattr(
                AVFoundation,
                "AVCaptureDeviceWasConnectedNotification",
            )

            disconnected_notification = getattr(
                AVFoundation,
                "AVCaptureDeviceWasDisconnectedNotification",
            )

            center.addObserver_selector_name_object_(
                observer,
                b"cameraConnected:",
                connected_notification,
                None,
            )

            center.addObserver_selector_name_object_(
                observer,
                b"cameraDisconnected:",
                disconnected_notification,
                None,
            )

            self._observer = observer
            self._notification_center = center

            print(
                "AVFoundation-Hot-Plug-Benachrichtigungen aktiv."
            )

        except Exception as exc:
            self._observer = None
            self._notification_center = None

            print(
                "Hinweis: AVFoundation-Notifications konnten "
                "nicht registriert werden. "
                "Die periodische Geräteprüfung bleibt aktiv."
            )
            print(f"  Ursache: {exc}")

    def _unregister_notifications(self) -> None:
        if (
            self._notification_center is not None
            and self._observer is not None
        ):
            try:
                self._notification_center.removeObserver_(
                    self._observer
                )
            except Exception:
                pass

        self._observer = None
        self._notification_center = None

    def refresh(self) -> None:
        discovered = self._discover_connected_devices()

        currently_present = {
            unique_id: discovered[unique_id]
            for unique_id in self._devices
            if unique_id in discovered
        }

        with self._condition:
            previous_ids = set(self._present)
            current_ids = set(currently_present)

            connected_ids = current_ids - previous_ids
            disconnected_ids = previous_ids - current_ids

            self._present = currently_present
            self._condition.notify_all()

        # Callbacks niemals unter dem Lock ausführen.
        for unique_id in sorted(disconnected_ids):
            config = self._devices[unique_id]

            print(
                "\nKAMERA GETRENNT:"
                f"\n  Rolle:    {config['camera_id']}"
                f"\n  uniqueID: {unique_id}"
            )

            if self._on_disconnected is not None:
                self._on_disconnected(config)

        for unique_id in sorted(connected_ids):
            config = self._devices[unique_id]
            device = currently_present[unique_id]

            print(
                "\nKAMERA VERBUNDEN:"
                f"\n  Rolle:    {config['camera_id']}"
                f"\n  Name:     {self._device_name(device)}"
                f"\n  uniqueID: {unique_id}"
            )

            if self._on_connected is not None:
                self._on_connected(config, device)

    def _monitor_loop(self) -> None:
        """
        Eigener Foundation-Runloop für Gerätebenachrichtigungen.

        Zusätzlich erfolgt weiterhin eine periodische Discovery
        als Sicherheitsnetz.
        """
        self._register_notifications()

        try:
            self.refresh()
            self._initialized_event.set()

            next_poll = monotonic() + self._poll_interval

            while not self._stop_event.is_set():
                run_until = (
                    Foundation.NSDate
                    .dateWithTimeIntervalSinceNow_(0.20)
                )

                Foundation.NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    Foundation.NSDefaultRunLoopMode,
                    run_until,
                )

                if self._stop_event.is_set():
                    break

                now = monotonic()

                if (
                    self._wake_event.is_set()
                    or now >= next_poll
                ):
                    self._wake_event.clear()

                    try:
                        self.refresh()
                    except Exception as exc:
                        print(
                            "Fehler bei der Hot-Plug-Prüfung: "
                            f"{exc}"
                        )

                    next_poll = now + self._poll_interval

        finally:
            self._initialized_event.set()
            self._unregister_notifications()

    def start(self) -> None:
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            return

        self._stop_event.clear()
        self._wake_event.clear()
        self._initialized_event.clear()

        self._thread = Thread(
            target=self._monitor_loop,
            name="anthro3d-camera-hotplug",
            daemon=True,
        )
        self._thread.start()

        if not self._initialized_event.wait(timeout=5.0):
            self.stop()
            raise RuntimeError(
                "Hot-Plug-Monitor konnte nicht initialisiert werden."
            )

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)

        self._thread = None
        self._unregister_notifications()

    def present_unique_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._present))

    def missing_required(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            missing_ids = (
                self._required_ids - set(self._present)
            )

            return tuple(
                self._devices[unique_id]
                for unique_id in sorted(missing_ids)
            )

    def all_required_present(self) -> bool:
        return not self.missing_required()

    def snapshot(self) -> tuple[CameraPresence, ...]:
        with self._lock:
            result = []

            for unique_id, config in self._devices.items():
                device = self._present.get(unique_id)

                result.append(
                    CameraPresence(
                        camera_id=config["camera_id"],
                        unique_id=unique_id,
                        required=bool(
                            config.get("required", True)
                        ),
                        present=device is not None,
                        device_name=(
                            self._device_name(device)
                            if device is not None
                            else None
                        ),
                    )
                )

            return tuple(result)

    def wait_for_required(
        self,
        timeout_seconds: float | None = None,
    ) -> bool:
        """
        Wartet auf alle erforderlichen Kameras.

        timeout_seconds=None:
            Wartet unbegrenzt.

        Rückgabe:
            True  = alle erforderlichen Kameras vorhanden
            False = Zeitlimit abgelaufen
        """
        deadline = (
            None
            if timeout_seconds is None
            else monotonic() + timeout_seconds
        )

        with self._condition:
            while True:
                missing_ids = (
                    self._required_ids
                    - set(self._present)
                )

                if not missing_ids:
                    return True

                if deadline is None:
                    self._condition.wait()
                    continue

                remaining = deadline - monotonic()

                if remaining <= 0:
                    return False

                self._condition.wait(timeout=remaining)

    def print_status(self) -> None:
        snapshot = self.snapshot()

        present_count = sum(
            item.present for item in snapshot
        )

        required = [
            item for item in snapshot
            if item.required
        ]

        present_required = sum(
            item.present for item in required
        )

        print()
        print("=" * 72)
        print(
            "Kamerastatus: "
            f"{present_count}/{len(snapshot)} aktiv | "
            f"{present_required}/{len(required)} "
            "erforderliche Kameras vorhanden"
        )

        for item in snapshot:
            status = (
                "VERBUNDEN"
                if item.present
                else "FEHLT"
            )

            requirement = (
                "Pflicht"
                if item.required
                else "optional"
            )

            print(
                f"  {item.camera_id}: "
                f"{status} | {requirement} | "
                f"{item.unique_id}"
            )

        print("=" * 72)
