from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


class MarkerIntegrityMonitor:
    """
    Überwacht ausschließlich die interne Geometrie der festen
    ArUco-Paare auf den drei Kamerastativen.

    Keine R/T-Dateien werden verändert.
    Keine Extrinsics werden gespeichert.
    """

    STANDS = {
        "OV9281_STAND": (2, 20),
        "ELP1_STAND": (3, 30),
        "ELP2_STAND": (4, 40),
    }

    OBSERVERS = {
        "OV9281_STAND": ("ELP2", "ELP1"),
        "ELP1_STAND": ("ELP2", "OV9281"),
        "ELP2_STAND": ("ELP1", "OV9281"),
    }

    BUFFER_SECONDS = 12.0
    MIN_SAMPLES = 20

    # Nicht geraten als absolute Präzision:
    # MAD der echten Referenzmessung wird zusätzlich berücksichtigt.
    DIST_WARN_FLOOR_CM = 0.75
    DIST_BLOCK_FLOOR_CM = 1.50

    ROT_WARN_FLOOR_DEG = 2.0
    ROT_BLOCK_FLOOR_DEG = 4.0

    DIST_WARN_MAD_MULT = 4.0
    DIST_BLOCK_MAD_MULT = 7.0

    ROT_WARN_MAD_MULT = 4.0
    ROT_BLOCK_MAD_MULT = 7.0

    def __init__(self, base="~/anthro3d"):
        self.base = Path(base).expanduser()

        ref_path = (
            self.base
            / "aruco_state"
            / "marker_integrity_reference.yaml"
        )

        if not ref_path.exists():
            raise FileNotFoundError(
                f"Integritätsreferenz fehlt: {ref_path}"
            )

        with ref_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.reference = yaml.safe_load(f)

        observers = self.reference.get(
            "observers",
            {},
        )

        if not observers:
            raise RuntimeError(
                "Integritätsreferenz enthält keine Observer."
            )

        self.buffers = defaultdict(deque)
        self.last_status = {}

    @staticmethod
    def _rotation_angle_deg(R):
        value = np.clip(
            (np.trace(R) - 1.0) / 2.0,
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(value)
            )
        )

    @staticmethod
    def _choose_pair(top_obs, bottom_obs):
        candidates = []

        for top in top_obs:
            for bottom in bottom_obs:
                same_side = (
                    top.get("side")
                    == bottom.get("side")
                )

                score = (
                    float(top.get("err", 999.0))
                    + float(bottom.get("err", 999.0))
                )

                candidates.append(
                    (
                        0 if same_side else 1,
                        score,
                        top,
                        bottom,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        return (
            candidates[0][2],
            candidates[0][3],
        )

    def _measure_pair(
        self,
        detections,
        top_id,
        bottom_id,
    ):
        top_obs = detections.get(
            top_id,
            [],
        )

        bottom_obs = detections.get(
            bottom_id,
            [],
        )

        if not top_obs or not bottom_obs:
            return None

        selected = self._choose_pair(
            top_obs,
            bottom_obs,
        )

        if selected is None:
            return None

        top, bottom = selected

        R_top, _ = cv2.Rodrigues(
            np.asarray(
                top["rvec"],
                dtype=np.float64,
            ).reshape(3, 1)
        )

        R_bottom, _ = cv2.Rodrigues(
            np.asarray(
                bottom["rvec"],
                dtype=np.float64,
            ).reshape(3, 1)
        )

        T_top = np.asarray(
            top["tvec"],
            dtype=np.float64,
        ).reshape(3)

        T_bottom = np.asarray(
            bottom["tvec"],
            dtype=np.float64,
        ).reshape(3)

        delta = T_bottom - T_top

        distance_cm = float(
            np.linalg.norm(delta)
            * 100.0
        )

        R_relative = (
            R_top.T
            @ R_bottom
        )

        rotation_deg = (
            self._rotation_angle_deg(
                R_relative
            )
        )

        return {
            "distance_cm": distance_cm,
            "rotation_deg": rotation_deg,
        }

    def _prune(self, now):
        minimum = (
            now
            - self.BUFFER_SECONDS
        )

        for buf in self.buffers.values():
            while (
                buf
                and buf[0]["time"] < minimum
            ):
                buf.popleft()

    def _observer_reference(
        self,
        observer,
        stand,
    ):
        return (
            self.reference
            ["observers"]
            [observer]
            [stand]
        )

    def _observer_status(
        self,
        observer,
        stand,
    ):
        key = (
            observer,
            stand,
        )

        samples = list(
            self.buffers[key]
        )

        if len(samples) < self.MIN_SAMPLES:
            return {
                "level": "unknown",
                "ready": False,
                "samples": len(samples),
                "reason": "not_enough_samples",
            }

        distances = np.asarray(
            [
                s["distance_cm"]
                for s in samples
            ],
            dtype=np.float64,
        )

        rotations = np.asarray(
            [
                s["rotation_deg"]
                for s in samples
            ],
            dtype=np.float64,
        )

        dist_med = float(
            np.median(distances)
        )

        rot_med = float(
            np.median(rotations)
        )

        ref = self._observer_reference(
            observer,
            stand,
        )

        ref_dist = float(
            ref["distance_cm_median"]
        )

        ref_rot = float(
            ref["rotation_deg_median"]
        )

        ref_dist_mad = max(
            float(
                ref.get(
                    "distance_cm_mad",
                    0.0,
                )
            ),
            1e-6,
        )

        ref_rot_mad = max(
            float(
                ref.get(
                    "rotation_deg_mad",
                    0.0,
                )
            ),
            1e-6,
        )

        dist_delta = abs(
            dist_med - ref_dist
        )

        rot_delta = abs(
            rot_med - ref_rot
        )

        dist_warn = max(
            self.DIST_WARN_FLOOR_CM,
            self.DIST_WARN_MAD_MULT
            * ref_dist_mad,
        )

        dist_block = max(
            self.DIST_BLOCK_FLOOR_CM,
            self.DIST_BLOCK_MAD_MULT
            * ref_dist_mad,
        )

        rot_warn = max(
            self.ROT_WARN_FLOOR_DEG,
            self.ROT_WARN_MAD_MULT
            * ref_rot_mad,
        )

        rot_block = max(
            self.ROT_BLOCK_FLOOR_DEG,
            self.ROT_BLOCK_MAD_MULT
            * ref_rot_mad,
        )

        if (
            dist_delta >= dist_block
            or rot_delta >= rot_block
        ):
            level = "block"

        elif (
            dist_delta >= dist_warn
            or rot_delta >= rot_warn
        ):
            level = "warn"

        else:
            level = "ok"

        return {
            "level": level,
            "ready": True,
            "samples": len(samples),

            "distance_cm": dist_med,
            "distance_reference_cm": ref_dist,
            "distance_delta_cm": dist_delta,
            "distance_warn_cm": dist_warn,
            "distance_block_cm": dist_block,

            "rotation_deg": rot_med,
            "rotation_reference_deg": ref_rot,
            "rotation_delta_deg": rot_delta,
            "rotation_warn_deg": rot_warn,
            "rotation_block_deg": rot_block,
        }

    def update(
        self,
        detections,
        now=None,
    ):
        if now is None:
            now = time.time()

        for stand, marker_ids in self.STANDS.items():
            top_id, bottom_id = marker_ids

            for observer in self.OBSERVERS[stand]:
                rig_detections = detections.get(
                    observer,
                    {},
                )

                measurement = self._measure_pair(
                    rig_detections,
                    top_id,
                    bottom_id,
                )

                if measurement is None:
                    continue

                self.buffers[
                    (observer, stand)
                ].append(
                    {
                        "time": float(now),
                        **measurement,
                    }
                )

        self._prune(now)

        observer_status = {}

        for stand in self.STANDS:
            observer_status[stand] = {}

            for observer in self.OBSERVERS[stand]:
                observer_status[
                    stand
                ][observer] = (
                    self._observer_status(
                        observer,
                        stand,
                    )
                )

        stand_status = {}

        for stand in self.STANDS:
            items = list(
                observer_status[
                    stand
                ].values()
            )

            ready = [
                item
                for item in items
                if item.get(
                    "ready",
                    False,
                )
            ]

            levels = [
                item["level"]
                for item in ready
            ]

            # Redundanzprinzip:
            # Ein einzelner abweichender Beobachter kann
            # die verschobene Kamera selbst sein.
            if len(ready) < 2:
                level = "unknown"

            elif "ok" in levels:
                level = "ok"

            elif "warn" in levels:
                level = "warn"

            else:
                # Erst wenn BEIDE unabhängigen Beobachter
                # die interne Markergeometrie als stark
                # verändert sehen, gilt das Stativ als BLOCK.
                level = "block"

            stand_status[stand] = {
                "level": level,
                "observers": tuple(
                    self.OBSERVERS[stand]
                ),
            }

        all_stands_ok = (
            len(stand_status)
            == len(self.STANDS)
            and all(
                item["level"] == "ok"
                for item in stand_status.values()
            )
        )

        any_block = any(
            item["level"] == "block"
            for item in stand_status.values()
        )

        self.last_status = {
            "observer_status": observer_status,
            "stand_status": stand_status,

            "all_stands_ok": all_stands_ok,

            # Nur bei vollständig bestätigter interner
            # Markergeometrie darf später eine automatische
            # Kamerarekalibrierung stattfinden.
            "safe_for_auto_recalibration": (
                all_stands_ok
            ),

            "marker_fault_detected": (
                any_block
            ),
        }

        return self.last_status

    def get_status(self):
        return self.last_status
