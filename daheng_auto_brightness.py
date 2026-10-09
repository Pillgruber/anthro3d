"""Shared, conservative software auto-brightness for monochrome Daheng cameras.

A single grayscale target is used across all camera workers. Each camera
adjusts physical exposure independently and uses limited gain only when
needed. The central ROI and highlight rejection mitigate side/back lighting;
they cannot restore clipped detail or guarantee identical appearance.
"""

import math
import re


TARGET_GRAY_8BIT = 105.0
EXPOSURE_LIMIT_US = 8000.0
GAIN_LIMIT_DB = 8.0
UPDATE_INTERVAL_S = 0.25


def brightness_metric(frame, pixel_format, np):
    """Measure center-region subject brightness, ignoring saturated highlights.

    Sample sparsely to limit CPU use at high resolution. This is a temporary
    fixed ROI; once human segmentation exists use a body-only mask.
    """
    match = re.fullmatch(r"Mono(8|10|12|14|16)", str(pixel_format), re.I)
    if not match:
        raise ValueError(f"Unsupported monochrome pixel format: {pixel_format}")
    h, w = frame.shape
    # Central 60% of width and 70% of height; exclude image edges/windows.
    region = frame[int(h * .15):int(h * .85):8,
                   int(w * .20):int(w * .80):8]
    if region.size == 0:
        raise ValueError("No pixels within brightness measurement ROI")
    bits = int(match.group(1))
    if bits > 8:
        # The raw 10/12/14/16-bit values stay untouched; only metric is 8-bit.
        samples = np.right_shift(region, bits - 8)
    else:
        samples = region
    samples = np.asarray(samples, dtype=np.uint8).ravel()
    # Avoid bright windows and sensor-saturated pixels driving subject exposure.
    unsaturated = samples[samples < 245]
    if unsaturated.size < max(30, int(samples.size * .15)):
        # Mostly saturated ROI: prioritize highlight recovery by reducing exposure.
        value = 245.0
    else:
        value = float(np.percentile(unsaturated, 55))
    return {"gray": value,
            "saturated_percent": round(100.0 * float(np.mean(samples >= 245)), 2)}


def _range(feature, low, high):
    try:
        bounds = feature.get_range()
        if isinstance(bounds, dict):
            low = max(low, float(bounds.get("min", low)))
            high = min(high, float(bounds.get("max", high)))
    except (AttributeError, ValueError, TypeError, RuntimeError):
        pass
    if high < low:
        raise RuntimeError(f"Camera feature has no usable range: {low}..{high}")
    return low, high


class AutoBrightness:
    def __init__(self, control):
        self.control = control
        self.restore = []
        self.exposure = None
        self.gain = None
        self.last_update = 0.0
        self.changes = 0
        self.last_metric = None
        self.exposure_bounds = None
        self.gain_bounds = None
        try:
            self.exposure = control.get_float_feature("ExposureTime")
            self.gain = control.get_float_feature("Gain")
            # Disable device auto control so it cannot race software controller.
            for name in ("ExposureAuto", "GainAuto"):
                try:
                    feature = control.get_enum_feature(name)
                    old = feature.get()[1]
                    if old != "Off":
                        self.restore.append((feature, old))
                        feature.set("Off")
                except (AttributeError, KeyError):
                    pass  # Some SDK devices expose no auto feature.
            self.exposure_bounds = _range(self.exposure, 100.0, EXPOSURE_LIMIT_US)
            self.gain_bounds = _range(self.gain, 0.0, GAIN_LIMIT_DB)
            original_exposure = float(self.exposure.get())
            original_gain = float(self.gain.get())
            self.restore.extend([(self.exposure, original_exposure), (self.gain, original_gain)])
            # Start within exposure/FPS budget and low-gain budget.
            initial_exposure = min(max(original_exposure, self.exposure_bounds[0]),
                                   self.exposure_bounds[1])
            initial_gain = min(max(original_gain, self.gain_bounds[0]),
                               self.gain_bounds[1])
            if abs(initial_exposure - original_exposure) > 1:
                self.exposure.set(initial_exposure)
            if abs(initial_gain - original_gain) > .01:
                self.gain.set(initial_gain)
        except Exception:
            self.restore_settings()
            raise

    def should_sample(self, now):
        return now - self.last_update >= UPDATE_INTERVAL_S

    def update(self, metric, now):
        if not self.should_sample(now):
            return
        self.last_update = now
        self.last_metric = metric
        gray = max(1.0, float(metric["gray"]))
        error = TARGET_GRAY_8BIT / gray
        if .93 <= error <= 1.07:
            return
        # Limit each adjustment to minimize flicker and runaway oscillations.
        scale = min(1.35, max(.75, error))
        exposure = float(self.exposure.get())
        gain = float(self.gain.get())
        exp_min, exp_max = self.exposure_bounds
        gain_min, gain_max = self.gain_bounds
        if scale > 1.0:
            if exposure < exp_max * .98:
                new_exposure = min(exp_max, max(exp_min, exposure * scale))
                if abs(new_exposure - exposure) >= 1:
                    self.exposure.set(new_exposure)
                    self.changes += 1
            elif gain < gain_max - .05:
                new_gain = min(gain_max, gain + 20 * math.log10(scale))
                self.gain.set(new_gain)
                self.changes += 1
        else:
            if gain > gain_min + .05:
                new_gain = max(gain_min, gain + 20 * math.log10(scale))
                self.gain.set(new_gain)
                self.changes += 1
            elif exposure > exp_min * 1.02:
                new_exposure = max(exp_min, min(exp_max, exposure * scale))
                if abs(new_exposure - exposure) >= 1:
                    self.exposure.set(new_exposure)
                    self.changes += 1

    def state(self):
        return {
            "enabled": True,
            "target_gray_8bit": TARGET_GRAY_8BIT,
            "measurement_roi": "center 60% width x 70% height, ignore saturated highlights",
            "exposure_time_us": float(self.exposure.get()),
            "gain_db": float(self.gain.get()),
            "exposure_limit_us": self.exposure_bounds[1],
            "gain_limit_db": self.gain_bounds[1],
            "adjustment_count": self.changes,
            "last_metric": self.last_metric,
        }

    def restore_settings(self):
        errors = []
        for feature, value in reversed(self.restore):
            try:
                feature.set(value)
            except Exception as exc:
                errors.append(str(exc))
        self.restore.clear()
        return errors
