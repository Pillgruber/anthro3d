from __future__ import annotations

import threading
import unittest

from camera_system.events import CameraEvent, CameraEventQueue
from camera_system.main_thread_hotplug import (
    AVFoundationBindings,
    AVFoundationMainThreadHotplugObserver,
    device_metadata,
)
from camera_system.manager import CameraManager, CameraManagerState, CameraSpec
from camera_system.runtime import MainThreadCameraRuntime


class FakeDevice:
    def __init__(self, unique_id: str, name: str = "Testkamera") -> None:
        self._unique_id = unique_id
        self._name = name

    def uniqueID(self) -> str:
        return self._unique_id

    def localizedName(self) -> str:
        return self._name

    def modelID(self) -> str:
        return "FakeModel"

    def manufacturer(self) -> str:
        return "Anthro3D"

    def deviceType(self) -> str:
        return "FakeExternalCamera"


class FakeObserver:
    def __init__(self, queue: CameraEventQueue) -> None:
        self.queue = queue
        self.is_started = False
        self.pending: list[CameraEvent] = []

    def start(self, *, seed_existing: bool = True) -> None:
        self.is_started = True
        if seed_existing:
            self.queue.publish(
                CameraEvent.connected("camera-1", device=FakeDevice("camera-1"))
            )

    def run_once(self, timeout: float = 0.25) -> None:
        if not self.is_started:
            raise RuntimeError("Observer nicht gestartet.")
        for event in self.pending:
            self.queue.publish(event)
        self.pending.clear()

    def stop(self) -> None:
        self.is_started = False


class MainThreadRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = CameraEventQueue()
        self.manager = CameraManager(
            [
                CameraSpec(
                    camera_id="CAM1",
                    unique_id="camera-1",
                    required=True,
                )
            ],
            event_queue=self.queue,
        )
        self.observer = FakeObserver(self.queue)
        self.runtime = MainThreadCameraRuntime(self.manager, self.observer)

    def test_seed_disconnect_and_reconnect(self) -> None:
        seeded = self.runtime.start()
        self.assertEqual(len(seeded), 1)
        self.assertEqual(self.manager.state, CameraManagerState.READY)

        self.observer.pending.append(CameraEvent.disconnected("camera-1"))
        events = self.runtime.pump_once(timeout=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            self.manager.state,
            CameraManagerState.WAITING_FOR_CAMERAS,
        )

        self.observer.pending.append(
            CameraEvent.connected("camera-1", device=FakeDevice("camera-1"))
        )
        self.runtime.pump_once(timeout=0)
        self.assertEqual(self.manager.state, CameraManagerState.READY)

        self.runtime.stop()
        self.assertEqual(self.manager.state, CameraManagerState.STOPPED)
        self.assertFalse(self.observer.is_started)

    def test_metadata_is_serializable(self) -> None:
        metadata = device_metadata(FakeDevice("camera-1", "OV9281 links"))
        self.assertEqual(metadata["name"], "OV9281 links")
        self.assertEqual(metadata["model_id"], "FakeModel")
        self.assertEqual(metadata["manufacturer"], "Anthro3D")

    def test_real_observer_rejects_background_thread(self) -> None:
        observer = AVFoundationMainThreadHotplugObserver(
            self.queue,
            bindings_loader=lambda: AVFoundationBindings(None, None),
        )
        errors: list[Exception] = []

        def worker() -> None:
            try:
                observer.start()
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("Hauptthread", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
