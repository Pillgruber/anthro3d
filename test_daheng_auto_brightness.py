"""Offline tests for the AnthroPrecis auto-brightness controller (no camera required)."""

import time
import unittest

import numpy as np

from daheng_auto_brightness import (AutoBrightness, brightness_metric,
                                    evaluate_pair, evaluate_capture_quality)


class FakeFloatFeature:
    def __init__(self, value, minimum, maximum):
        self.value = value
        self.bounds = {'min': minimum, 'max': maximum}

    def get(self):
        return self.value

    def get_range(self):
        return self.bounds

    def set(self, value):
        if not self.bounds['min'] <= value <= self.bounds['max']:
            raise ValueError('out of camera range')
        self.value = value


class FakeEnumFeature:
    def __init__(self, value):
        self.value = value

    def get(self):
        return (0, self.value)

    def set(self, value):
        self.value = value


class FakeControl:
    def __init__(self):
        self.floats = {
            'ExposureTime': FakeFloatFeature(2000.0, 100.0, 25000.0),
            'Gain': FakeFloatFeature(0.0, 0.0, 16.0),
        }
        self.enums = {
            'ExposureAuto': FakeEnumFeature('Continuous'),
            'GainAuto': FakeEnumFeature('Continuous'),
        }

    def get_float_feature(self, name):
        return self.floats[name]

    def get_enum_feature(self, name):
        return self.enums[name]


class AutoBrightnessTests(unittest.TestCase):
    def test_center_region_ignores_bright_edges(self):
        frame = np.full((1200, 2048), 105, dtype=np.uint8)
        frame[:, :200] = 255
        metric = brightness_metric(frame, 'Mono8', np)
        self.assertEqual(metric['gray'], 105.0)
        self.assertEqual(metric['saturated_percent'], 0.0)

    def test_mono12_normalization(self):
        frame = np.full((1200, 2048), 105 << 4, dtype=np.uint16)
        self.assertEqual(brightness_metric(frame, 'Mono12', np)['gray'], 105.0)

    def test_pair_gate(self):
        self.assertTrue(evaluate_pair({
            '3': {'gray': 104, 'saturated_percent': 0},
            '30': {'gray': 106, 'saturated_percent': 1},
        })[0])
        self.assertFalse(evaluate_pair({
            '3': {'gray': 90, 'saturated_percent': 0},
            '30': {'gray': 106, 'saturated_percent': 1},
        })[0])
        self.assertFalse(evaluate_pair({
            '3': {'gray': 104, 'saturated_percent': 48},
            '30': {'gray': 106, 'saturated_percent': 1},
        })[0])

    def test_backlit_scene_can_capture_without_target_match(self):
        # Strongly backlit test: different image brightness cannot always be
        # equalized, yet both images contain measurable grayscale detail.
        first = {
            'gray': 42, 'saturated_percent': 33, 'dark_percent': 58,
            'midtone_percent': 9, 'p10': 2, 'p90': 255,
        }
        second = {
            'gray': 78, 'saturated_percent': 12, 'dark_percent': 42,
            'midtone_percent': 23, 'p10': 4, 'p90': 247,
        }
        pair = {'FHK26060051': first, 'FHK26080099': second}
        self.assertFalse(evaluate_pair(pair)[0])
        self.assertTrue(evaluate_capture_quality(pair)[0])

    def test_unusable_black_or_white_frame_blocks_capture(self):
        dark = {
            'gray': 1, 'saturated_percent': 0, 'dark_percent': 100,
            'midtone_percent': 0, 'p10': 0, 'p90': 2,
        }
        clipped = {
            'gray': 245, 'saturated_percent': 100, 'dark_percent': 0,
            'midtone_percent': 0, 'p10': 255, 'p90': 255,
        }
        usable = {
            'gray': 104, 'saturated_percent': 0, 'dark_percent': 0,
            'midtone_percent': 100, 'p10': 100, 'p90': 120,
        }
        self.assertFalse(evaluate_capture_quality({'a': dark, 'b': usable})[0])
        self.assertFalse(evaluate_capture_quality({'a': clipped, 'b': usable})[0])

    def test_brightness_is_adjusted_during_countdown(self):
        control = FakeControl()
        controller = AutoBrightness(control)
        start = time.perf_counter() + 1.0
        exp_before = control.floats['ExposureTime'].get()
        for step in range(10):
            controller.update({'gray': 38.0, 'saturated_percent': 1},
                              start + step * .2)
        self.assertGreater(control.floats['ExposureTime'].get(), exp_before)
        self.assertLessEqual(control.floats['ExposureTime'].get(), 8000)
        self.assertGreater(controller.changes, 1)
        controller.restore_settings()

    def test_adjust_and_restore_camera_values(self):
        control = FakeControl()
        controller = AutoBrightness(control)
        self.assertEqual(control.enums['ExposureAuto'].value, 'Off')
        now = time.perf_counter() + 1.0
        controller.update({'gray': 40.0, 'saturated_percent': 0}, now)
        self.assertGreater(control.floats['ExposureTime'].value, 2000)
        self.assertEqual(controller.restore_settings(), [])
        self.assertEqual(control.floats['ExposureTime'].value, 2000)
        self.assertEqual(control.enums['ExposureAuto'].value, 'Continuous')


if __name__ == '__main__':
    unittest.main()
