"""Offline tests for the AnthroPrecis auto-brightness controller (no camera required)."""

import time
import unittest

import numpy as np

from daheng_auto_brightness import AutoBrightness, brightness_metric, evaluate_pair


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
