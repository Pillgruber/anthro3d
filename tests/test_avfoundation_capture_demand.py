from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import camera_system.backends.avfoundation as avf


class AVFoundationDemandDrivenTests(unittest.TestCase):
    def make_capture(self):
        capture = avf.AVFoundationCapture(
            unique_id="unit-test-camera",
            width=4,
            height=3,
            fps=30.0,
        )
        with capture._condition:
            capture._is_open = True
            capture._session = object()
        return capture

    def test_unrequested_callback_is_dropped_before_conversion(self):
        capture = self.make_capture()

        with mock.patch.object(
            avf,
            "_pixel_buffer_to_bgr",
        ) as converter:
            capture._handle_sample_buffer(object())
            converter.assert_not_called()

        self.assertEqual(capture._latest_sequence, 0)

    def test_requested_callback_converts_exactly_once(self):
        capture = self.make_capture()
        frame = np.zeros((3, 4, 3), dtype=np.uint8)

        with capture._condition:
            capture._frame_requested = True

        with (
            mock.patch.object(
                avf.CoreMedia,
                "CMSampleBufferGetImageBuffer",
                return_value=object(),
            ),
            mock.patch.object(
                avf,
                "_pixel_buffer_to_bgr",
                return_value=frame,
            ) as converter,
        ):
            capture._handle_sample_buffer(object())
            capture._handle_sample_buffer(object())

        self.assertEqual(converter.call_count, 1)
        self.assertEqual(capture._latest_sequence, 1)
        self.assertIs(capture._latest_frame, frame)
        self.assertFalse(capture._frame_requested)
        self.assertFalse(capture._conversion_in_progress)

    def test_callback_error_clears_request_state(self):
        capture = self.make_capture()

        with capture._condition:
            capture._frame_requested = True
            capture._conversion_in_progress = True

        capture._publish_callback_error(
            RuntimeError("synthetic")
        )

        self.assertFalse(capture._frame_requested)
        self.assertFalse(capture._conversion_in_progress)
        self.assertIn(
            "RuntimeError: synthetic",
            capture.last_error,
        )


if __name__ == "__main__":
    unittest.main()


class AVFoundationAutoreleasePoolTests(unittest.TestCase):
    def test_requested_callback_uses_autorelease_pool(self):
        capture = avf.AVFoundationCapture(
            unique_id="unit-test-camera-pool",
            width=4,
            height=3,
            fps=30.0,
        )
        with capture._condition:
            capture._is_open = True
            capture._session = object()
            capture._frame_requested = True

        frame = np.zeros((3, 4, 3), dtype=np.uint8)

        class FakePool:
            def __init__(self):
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc, tb):
                self.exited = True
                return False

        pool = FakePool()

        with (
            mock.patch.object(
                avf.objc,
                "autorelease_pool",
                return_value=pool,
            ) as factory,
            mock.patch.object(
                avf.CoreMedia,
                "CMSampleBufferGetImageBuffer",
                return_value=object(),
            ),
            mock.patch.object(
                avf,
                "_pixel_buffer_to_bgr",
                return_value=frame,
            ),
        ):
            capture._handle_sample_buffer(object())

        factory.assert_called_once_with()
        self.assertTrue(pool.entered)
        self.assertTrue(pool.exited)
        self.assertEqual(capture._latest_sequence, 1)
