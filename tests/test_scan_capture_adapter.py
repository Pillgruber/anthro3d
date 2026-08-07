from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_system.scan_capture import (
    AnthroCameraCapture,
    get_camera_profile,
    load_camera_profiles,
)


REGISTRY_TEXT = """\
version: 1
cameras:
  ELP1:
    unique_id: "elp1-id"
    width: 3200
    height: 1200
    fps: 30.0
    role: "ELP1"
    confirmed: true
  OV9281_L:
    unique_id: "ov-left-id"
    width: 1280
    height: 800
    fps: 30.0
    role: "OV9281_L"
    confirmed: true
"""


class FakeCapture:
    def __init__(self, unique_id, width, height, fps):
        self.unique_id = unique_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_open = False

    def open(self):
        self.is_open = True

    def read(self, timeout=None):
        if not self.is_open:
            return False, None
        return (
            True,
            np.zeros(
                (self.height, self.width, 3),
                dtype=np.uint8,
            ),
        )

    def release(self):
        self.is_open = False


class ScanCaptureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = (
            Path(self.tempdir.name)
            / "camera_registry.scan.local.yaml"
        )
        self.path.write_text(REGISTRY_TEXT, encoding="utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_registry_loads_unique_profiles(self):
        profiles = load_camera_profiles(self.path)
        self.assertEqual(
            profiles["elp1"].unique_id,
            "elp1-id",
        )
        self.assertEqual(
            get_camera_profile(
                "OV9281 L",
                self.path,
            ).unique_id,
            "ov-left-id",
        )

    def test_adapter_exposes_cv2_like_contract(self):
        cap = AnthroCameraCapture(
            "ELP1",
            self.path,
            capture_factory=FakeCapture,
        )

        self.assertTrue(cap.isOpened())
        self.assertEqual(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            3200.0,
        )
        self.assertEqual(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            1200.0,
        )
        self.assertTrue(
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
        )
        self.assertFalse(
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        )

        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (1200, 3200, 3))

        cap.release()
        self.assertFalse(cap.isOpened())

    def test_duplicate_unique_id_is_rejected(self):
        duplicate = self.path.read_text(
            encoding="utf-8"
        ).replace(
            'unique_id: "ov-left-id"',
            'unique_id: "elp1-id"',
        )
        self.path.write_text(duplicate, encoding="utf-8")

        with self.assertRaises(RuntimeError):
            load_camera_profiles(self.path)


if __name__ == "__main__":
    unittest.main()
