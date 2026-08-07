from __future__ import annotations

import unittest

from camera_system.backends.avfoundation import (
    AVFoundationCapture,
)


class AVFoundationCaptureContractTests(unittest.TestCase):
    def test_constructor_rejects_empty_unique_id(self):
        with self.assertRaises(ValueError):
            AVFoundationCapture("", 1280, 800, 30)

    def test_constructor_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            AVFoundationCapture("test", 0, 800, 30)

        with self.assertRaises(ValueError):
            AVFoundationCapture("test", 1280, -1, 30)

    def test_constructor_rejects_invalid_fps(self):
        with self.assertRaises(ValueError):
            AVFoundationCapture("test", 1280, 800, 0)

    def test_closed_capture_read_and_release_are_safe(self):
        capture = AVFoundationCapture(
            "not-opened",
            1280,
            800,
            30,
        )

        self.assertFalse(capture.is_open)
        ok, frame = capture.read(timeout=0)
        self.assertFalse(ok)
        self.assertIsNone(frame)

        capture.release()
        self.assertFalse(capture.is_open)


if __name__ == "__main__":
    unittest.main()
