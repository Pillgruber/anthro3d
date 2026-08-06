from __future__ import annotations

import unittest

from camera_system.backends.mock import MockCaptureBackend
from camera_system.events import CameraEvent, CameraEventQueue
from camera_system.manager import (
    CameraManager,
    CameraManagerState,
    CameraSpec,
)


class CameraManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.created_backends: dict[str, MockCaptureBackend] = {}

        def factory(spec: CameraSpec, device: object | None) -> MockCaptureBackend:
            backend = MockCaptureBackend(spec.unique_id)
            self.created_backends[spec.camera_id] = backend
            return backend

        self.queue = CameraEventQueue()
        self.manager = CameraManager(
            [
                CameraSpec(
                    camera_id="ELP1",
                    unique_id="elp-1",
                    profile="stereo_side_by_side_3200x1200",
                    required=True,
                ),
                CameraSpec(
                    camera_id="OV9281_L",
                    unique_id="ov-left",
                    profile="mono_1280x800",
                    required=True,
                ),
            ],
            event_queue=self.queue,
            backend_factory=factory,
        )

    def test_state_transitions_and_reconnect(self) -> None:
        self.assertEqual(self.manager.state, CameraManagerState.STOPPED)

        self.manager.start()
        self.assertEqual(
            self.manager.state,
            CameraManagerState.WAITING_FOR_CAMERAS,
        )

        self.manager.publish(CameraEvent.connected("elp-1", device="ELP"))
        self.manager.process_next_event(timeout=0.1)
        self.assertEqual(self.manager.state, CameraManagerState.DEGRADED)

        self.manager.publish(
            CameraEvent.connected("ov-left", device="OV9281")
        )
        self.manager.process_next_event(timeout=0.1)
        self.assertEqual(self.manager.state, CameraManagerState.READY)

        self.manager.start_captures()
        self.assertEqual(self.manager.state, CameraManagerState.RUNNING)

        elp_backend = self.created_backends["ELP1"]
        elp_backend.push_frame({"frame": 1})
        ok, frame = self.manager.read("ELP1", timeout=0.1)
        self.assertTrue(ok)
        self.assertEqual(frame, {"frame": 1})

        self.manager.handle_event(CameraEvent.disconnected("ov-left"))
        self.assertEqual(self.manager.state, CameraManagerState.DEGRADED)
        self.assertFalse(self.created_backends["OV9281_L"].is_open)

        self.manager.handle_event(
            CameraEvent.connected("ov-left", device="OV9281-new")
        )
        self.assertEqual(self.manager.state, CameraManagerState.RUNNING)
        self.assertTrue(self.created_backends["OV9281_L"].is_open)

    def test_unknown_camera_is_ignored(self) -> None:
        self.manager.start()
        handled = self.manager.handle_event(
            CameraEvent.connected("unknown-camera")
        )
        self.assertFalse(handled)
        self.assertEqual(
            self.manager.state,
            CameraManagerState.WAITING_FOR_CAMERAS,
        )

    def test_pause_resume_and_stop(self) -> None:
        self.manager.start()
        self.manager.handle_event(CameraEvent.connected("elp-1"))
        self.manager.handle_event(CameraEvent.connected("ov-left"))
        self.manager.start_captures()

        self.manager.pause()
        self.assertEqual(self.manager.state, CameraManagerState.PAUSED)
        self.assertEqual(self.manager.read("ELP1", timeout=0), (False, None))

        self.manager.resume()
        self.assertEqual(self.manager.state, CameraManagerState.RUNNING)

        self.manager.stop()
        self.assertEqual(self.manager.state, CameraManagerState.STOPPED)
        self.assertFalse(self.created_backends["ELP1"].is_open)
        self.assertFalse(self.created_backends["OV9281_L"].is_open)


if __name__ == "__main__":
    unittest.main()
