from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Optional
import argparse
import time

import cv2
import numpy as np
import yaml


@dataclass
class PairEstimate:
    ok: bool
    R: Optional[np.ndarray]
    T: Optional[np.ndarray]
    samples: int
    std_max_cm: float
    reproj_median: float
    reproj_max: float
    source: str


class ArucoMonitor:
    """
    Read-only ArUco-Kontrolle für ANTHRO3D.

    calibrate.py bleibt die einzige Stelle für echte Referenz-Kalibrierungen.
    Dieses Modul prüft nur, ob die Kamerapositionen noch plausibel sind.
    Bestehende R_rel und T_rel Dateien werden niemals automatisch überschrieben.

    Dieses Modul liest selbst keine Kameras aus.
    scan3d.py muss fertige Frames an update(...) übergeben.
    """

    MARKER_SIZE_M = 0.19

    KNOWN_MARKER_IDS = {2, 3, 10}

    EXPECTED_VISIBLE_MARKERS = {
        "ELP2": {2, 3},
        "ELP1": {2, 10},
        "OV9281": {3, 10},
    }

    EXPECTED_BASELINE_M = {
        "ELP2": 0.0588,
        "ELP1": 0.0588,
        "OV9281": 0.0819,
    }

    BASELINE_WARN_REL_DIFF = 0.25

    MIN_MARKER_AREA_PX = 40 * 40
    MAX_MARKER_DISTANCE_M = 6.0
    MIN_MARKER_RATIO = 0.60
    MAX_MARKER_RATIO = 1.40
    MAX_REPROJECTION_ERROR_PX = 6.0

    BUFFER_SECONDS = 45.0
    MIN_SAMPLES_FOR_LIVE = 12

    MAX_STABLE_STD_CM = 8.0

    WARN_TRANSLATION_SHIFT_CM = 8.0
    WARN_ROTATION_SHIFT_DEG = 5.0

    BLOCK_TRANSLATION_SHIFT_CM = 18.0
    BLOCK_ROTATION_SHIFT_DEG = 10.0

    def __init__(
        self,
        base,
        cfg2,
        cfg1,
        cfgov=None,
        require_ov=False,
        stereo_T_units=None,
        pre_warnings=None,
    ):
        self.base = Path(base).expanduser()
        self.lock = RLock()
        self.require_ov = bool(require_ov)
        self.warnings = list(pre_warnings or [])
        self.stereo_T_units = self._normalize_stereo_T_units(stereo_T_units)

        self.rigs = {}
        self.rigs["ELP2"] = self._make_rig(cfg2, "ELP2", self.stereo_T_units["ELP2"])
        self.rigs["ELP1"] = self._make_rig(cfg1, "ELP1", self.stereo_T_units["ELP1"])

        if cfgov is not None:
            try:
                self.rigs["OV9281"] = self._make_rig(cfgov, "OV9281", self.stereo_T_units["OV9281"])
            except Exception as exc:
                if self.require_ov:
                    raise
                self.warnings.append(
                    "OV9281 wurde deaktiviert, weil das Stereo-Schema oder die Einheit nicht passt: "
                    + str(exc)
                )
        else:
            if not any("OV9281" in msg for msg in self.warnings):
                self.warnings.append(
                    "OV9281 wurde nicht geladen. Für OV9281 wird ein klassisches Stereo-YAML mit "
                    "camera_matrix_l, camera_matrix_r, dist_l, dist_r, R und T erwartet."
                )

        self.pair_rules = self._make_pair_rules()

        self.obj_pts = np.array(
            [
                [-self.MARKER_SIZE_M / 2, self.MARKER_SIZE_M / 2, 0],
                [self.MARKER_SIZE_M / 2, self.MARKER_SIZE_M / 2, 0],
                [self.MARKER_SIZE_M / 2, -self.MARKER_SIZE_M / 2, 0],
                [-self.MARKER_SIZE_M / 2, -self.MARKER_SIZE_M / 2, 0],
            ],
            dtype=np.float32,
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.params = cv2.aruco.DetectorParameters()
        self.params.minMarkerPerimeterRate = 0.05
        self.params.maxMarkerPerimeterRate = 0.80

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        else:
            self.detector = None

        self.buffers = {pair_name: deque() for pair_name in self.pair_rules}
        self.saved = self._load_saved_transforms()

        self.live_estimates = {
            pair_name: PairEstimate(False, None, None, 0, 999.0, 999.0, 999.0, "not_started")
            for pair_name in self.pair_rules
        }

        self.last_detections = {}
        self.last_gate = {}
        self.last_update_time = None

    @classmethod
    def from_files(cls, base="~/anthro3d", require_ov=False, stereo_T_units=None):
        base = Path(base).expanduser()
        warnings = []

        def load_yaml(name):
            path = base / name
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        cfg2 = load_yaml("stereo_config.yaml")
        cfg1 = load_yaml("stereo_config_elp1.yaml")

        cfgov = None
        ov_path = base / "stereo_config_ov9281.yaml"

        if ov_path.exists():
            raw_ov = load_yaml("stereo_config_ov9281.yaml")
            needed = {"camera_matrix_l", "camera_matrix_r", "dist_l", "dist_r", "R", "T"}
            missing = sorted(needed - set(raw_ov.keys()))

            if missing:
                msg = "stereo_config_ov9281.yaml passt nicht zum klassischen Stereo-Schema. Fehlende Schlüssel: "
                msg += ", ".join(missing)
                if require_ov:
                    raise KeyError(msg)
                warnings.append(msg)
            else:
                cfgov = raw_ov
        else:
            msg = "stereo_config_ov9281.yaml wurde nicht gefunden."
            if require_ov:
                raise FileNotFoundError(str(ov_path))
            warnings.append(msg)

        return cls(
            base=base,
            cfg2=cfg2,
            cfg1=cfg1,
            cfgov=cfgov,
            require_ov=require_ov,
            stereo_T_units=stereo_T_units,
            pre_warnings=warnings,
        )

    def _normalize_stereo_T_units(self, stereo_T_units):
        default_units = {
            "ELP2": "auto",
            "ELP1": "auto",
            "OV9281": "auto",
        }

        allowed = {"auto", "m", "cm", "mm"}

        if stereo_T_units is None:
            return default_units

        if isinstance(stereo_T_units, str):
            unit = stereo_T_units.lower().strip()

            if unit not in allowed:
                raise ValueError("T-Einheit muss auto, m, cm oder mm sein.")

            return {
                "ELP2": unit,
                "ELP1": unit,
                "OV9281": unit,
            }

        merged = dict(default_units)
        merged.update(stereo_T_units)

        for rig_name, unit in merged.items():
            unit = str(unit).lower().strip()

            if unit not in allowed:
                raise ValueError(f"{rig_name}: T-Einheit muss auto, m, cm oder mm sein.")

            merged[rig_name] = unit

        return merged

    def _make_pair_rules(self):
        rules = {
            "ELP1_to_ELP2": {
                "target": "ELP2",
                "source": "ELP1",
                "marker": 2,
                "required": True,
                "R_file": "R_rel_elp1_to_elp2.npy",
                "T_file": "T_rel_elp1_to_elp2.npy",
            }
        }

        if "OV9281" in self.rigs:
            rules["OV9281_to_ELP2"] = {
                "target": "ELP2",
                "source": "OV9281",
                "marker": 3,
                "required": self.require_ov,
                "R_file": "R_rel_ov9281_to_elp2.npy",
                "T_file": "T_rel_ov9281_to_elp2.npy",
            }

            rules["OV9281_to_ELP1"] = {
                "target": "ELP1",
                "source": "OV9281",
                "marker": 10,
                "required": False,
                "R_file": None,
                "T_file": None,
            }

        return rules

    def _make_rig(self, cfg, rig_name, stereo_T_unit):
        needed = ["camera_matrix_l", "camera_matrix_r", "dist_l", "dist_r", "R", "T"]
        missing = [key for key in needed if key not in cfg]

        if missing:
            raise KeyError(f"{rig_name}: fehlende YAML-Schlüssel: {', '.join(missing)}")

        T_lr_raw = np.array(cfg["T"], dtype=np.float64).reshape(3)
        requested_unit = str(stereo_T_unit).lower().strip()

        if requested_unit == "auto":
            chosen_unit = self._auto_detect_T_unit(T_lr_raw, rig_name)
            self.warnings.append(f"{rig_name}: Stereo-T-Einheit automatisch erkannt als '{chosen_unit}'.")
        else:
            chosen_unit = requested_unit

        T_lr = self._convert_T_to_meters(T_lr_raw, chosen_unit, rig_name)
        baseline_m = float(np.linalg.norm(T_lr))

        if not np.isfinite(baseline_m) or baseline_m <= 0:
            raise ValueError(f"{rig_name}: Stereo-T ist ungültig.")

        if baseline_m < 0.01 or baseline_m > 0.50:
            raise ValueError(
                f"{rig_name}: Stereo-T wirkt unplausibel: {baseline_m:.4f} m. "
                "Die Baseline muss nach Umrechnung zwischen 1 cm und 50 cm liegen."
            )

        baseline_sanity = self._baseline_sanity_check(rig_name, baseline_m)

        return {
            "K_l": np.array(cfg["camera_matrix_l"], dtype=np.float64),
            "d_l": np.array(cfg["dist_l"], dtype=np.float64),
            "K_r": np.array(cfg["camera_matrix_r"], dtype=np.float64),
            "d_r": np.array(cfg["dist_r"], dtype=np.float64),
            "R_lr": np.array(cfg["R"], dtype=np.float64),
            "T_lr": T_lr,
            "T_lr_raw": T_lr_raw,
            "baseline_m": baseline_m,
            "T_unit": chosen_unit,
            "baseline_sanity": baseline_sanity,
        }

    def _auto_detect_T_unit(self, T_raw, rig_name):
        norm_raw = float(np.linalg.norm(T_raw))

        candidates = []

        unit_to_baseline_m = {
            "m": norm_raw,
            "cm": norm_raw * 0.01,
            "mm": norm_raw * 0.001,
        }

        for unit, baseline_m in unit_to_baseline_m.items():
            baseline_cm = baseline_m * 100.0

            if 1.0 <= baseline_cm <= 50.0:
                candidates.append((unit, baseline_m, baseline_cm))

        if not candidates:
            raise ValueError(
                f"{rig_name}: T-Einheit konnte nicht automatisch erkannt werden. "
                f"Norm roh: {norm_raw:.6f}."
            )

        expected = self.EXPECTED_BASELINE_M.get(rig_name)

        if expected is not None:
            best = min(candidates, key=lambda item: abs(item[1] - expected))
        else:
            best = min(candidates, key=lambda item: abs(item[2] - 8.0))

        if len(candidates) > 1:
            self.warnings.append(
                f"{rig_name}: T-Einheit war mehrdeutig. "
                f"Automatisch gewählt wurde '{best[0]}'. "
                f"Kandidaten: {[(u, round(cm, 2)) for u, _m, cm in candidates]} cm."
            )

        return best[0]

    def _baseline_sanity_check(self, rig_name, baseline_m):
        expected = self.EXPECTED_BASELINE_M.get(rig_name)

        out = {
            "expected_baseline_m": expected,
            "baseline_diff_m": None,
            "baseline_diff_percent": None,
            "level": "unknown",
        }

        if expected is None:
            out["level"] = "no_expected_baseline"
            return out

        diff_m = abs(baseline_m - expected)
        rel_diff = diff_m / expected if expected > 0 else None

        out["baseline_diff_m"] = diff_m
        out["baseline_diff_percent"] = None if rel_diff is None else rel_diff * 100.0

        if rel_diff is not None and rel_diff > self.BASELINE_WARN_REL_DIFF:
            out["level"] = "warn"
            self.warnings.append(
                f"{rig_name}: Baseline weicht deutlich vom erwarteten Wert ab. "
                f"Ist {baseline_m * 100.0:.2f} cm, erwartet ca. {expected * 100.0:.2f} cm, "
                f"Abweichung {rel_diff * 100.0:.1f} Prozent."
            )
        else:
            out["level"] = "ok"

        return out

    def _convert_T_to_meters(self, T_raw, unit, rig_name):
        unit = str(unit).lower().strip()

        if unit == "m":
            return T_raw

        if unit == "cm":
            self.warnings.append(f"{rig_name}: Stereo-T wurde von Zentimeter in Meter umgerechnet.")
            return T_raw * 0.01

        if unit == "mm":
            self.warnings.append(f"{rig_name}: Stereo-T wurde von Millimeter in Meter umgerechnet.")
            return T_raw * 0.001

        raise ValueError(f"{rig_name}: T-Einheit muss auto, m, cm oder mm sein.")

    def _load_saved_transforms(self):
        out = {}

        for pair_name, rule in self.pair_rules.items():
            R_file = rule["R_file"]
            T_file = rule["T_file"]

            if R_file is None or T_file is None:
                out[pair_name] = {
                    "ok": False,
                    "R": None,
                    "T": None,
                    "source": "not_saved_pair",
                }
                continue

            R_path = self.base / R_file
            T_path = self.base / T_file

            if R_path.exists() and T_path.exists():
                out[pair_name] = {
                    "ok": True,
                    "R": np.load(R_path),
                    "T": np.load(T_path).reshape(3),
                    "source": "saved_reference",
                    "R_file": str(R_path),
                    "T_file": str(T_path),
                }
            else:
                out[pair_name] = {
                    "ok": False,
                    "R": None,
                    "T": None,
                    "source": "missing_reference",
                    "R_file": str(R_path),
                    "T_file": str(T_path),
                }

        return out

    def update(
        self,
        frame_elp2_bgr=None,
        frame_elp1_bgr=None,
        frame_ov_l_bgr=None,
        frame_ov_r_bgr=None,
        now=None,
    ):
        if now is None:
            now = time.time()

        detections = self._detect_all_available_rigs(
            frame_elp2_bgr=frame_elp2_bgr,
            frame_elp1_bgr=frame_elp1_bgr,
            frame_ov_l_bgr=frame_ov_l_bgr,
            frame_ov_r_bgr=frame_ov_r_bgr,
        )

        with self.lock:
            self.last_update_time = now
            self.last_detections = detections

            for pair_name, rule in self.pair_rules.items():
                target = rule["target"]
                source = rule["source"]
                marker = rule["marker"]

                if target not in detections or source not in detections:
                    continue

                target_obs_list = detections.get(target, {}).get(marker, [])
                source_obs_list = detections.get(source, {}).get(marker, [])

                if not target_obs_list or not source_obs_list:
                    continue

                best = None

                for target_obs in target_obs_list:
                    for source_obs in source_obs_list:
                        R_rel, T_rel, score = self._relative_from_common_marker(target_obs, source_obs)
                        candidate = {
                            "time": now,
                            "R": R_rel,
                            "T": T_rel,
                            "score": float(score),
                            "marker": marker,
                            "target_side": target_obs["side"],
                            "source_side": source_obs["side"],
                        }

                        if best is None or candidate["score"] < best["score"]:
                            best = candidate

                if best is not None:
                    self.buffers[pair_name].append(best)

            self._prune_buffers(now)

            for pair_name in self.pair_rules:
                self.live_estimates[pair_name] = self._estimate_pair_from_samples(pair_name)

            self.last_gate = self._build_scan_gate_unlocked()

            return self.get_status()

    def get_scan_transforms(self):
        """
        Gibt nur gespeicherte Referenz-Transformationen zurück.
        Live-ArUco-Schätzungen werden niemals automatisch als Scan-Transformation verwendet.
        """

        with self.lock:
            out = {}

            for pair_name, saved in self.saved.items():
                out[pair_name] = {
                    "ok": bool(saved.get("ok", False)),
                    "R": saved.get("R"),
                    "T": saved.get("T"),
                    "source": saved.get("source", "unknown"),
                }

            return out

    def get_scan_gate(self):
        with self.lock:
            self.last_gate = self._build_scan_gate_unlocked()
            return self.last_gate

    def get_status(self):
        with self.lock:
            return {
                "last_update_time": self.last_update_time,
                "warnings": list(self.warnings),
                "detections": self._detections_to_status(self.last_detections),
                "references": self._references_to_status(),
                "live_estimates": self._live_estimates_to_status(),
                "scan_gate": self.last_gate,
                "buffers": {name: len(buf) for name, buf in self.buffers.items()},
                "rigs": self._rigs_to_status(),
            }

    def export_candidate(self, pair_name, note=""):
        """
        Speichert eine stabile Live-Schätzung manuell als Kandidat.
        Es werden keine Referenzdateien überschrieben.
        """

        with self.lock:
            if pair_name not in self.live_estimates:
                raise ValueError(f"Unbekanntes Paar: {pair_name}")

            estimate = self.live_estimates[pair_name]

            if not estimate.ok or estimate.R is None or estimate.T is None:
                raise RuntimeError(f"Keine stabile Live-Schätzung für {pair_name} vorhanden.")

            candidate_dir = self.base / "aruco_candidates"
            candidate_dir.mkdir(exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")

            R_path = candidate_dir / f"{pair_name}_{stamp}_R_candidate.npy"
            T_path = candidate_dir / f"{pair_name}_{stamp}_T_candidate.npy"
            report_path = candidate_dir / f"{pair_name}_{stamp}_report.txt"

            np.save(R_path, estimate.R)
            np.save(T_path, estimate.T)

            saved = self.saved.get(pair_name, {})
            t_diff_cm = None
            r_diff_deg = None

            if saved.get("ok") and saved.get("R") is not None and saved.get("T") is not None:
                t_diff_cm = float(np.linalg.norm(estimate.T - saved["T"]) * 100.0)
                r_diff_deg = self._rotation_diff_deg(estimate.R, saved["R"])

            lines = [
                "ArUco Kandidat wurde manuell exportiert.",
                "Referenzdateien wurden nicht überschrieben.",
                f"Pair: {pair_name}",
                f"Samples: {estimate.samples}",
                f"Std max cm: {estimate.std_max_cm:.2f}",
                f"Reprojection median: {estimate.reproj_median:.2f}",
                f"Reprojection max: {estimate.reproj_max:.2f}",
                f"Kandidat T cm: {(estimate.T * 100.0).tolist()}",
                f"Abweichung Translation cm: {t_diff_cm}",
                f"Abweichung Rotation Grad: {r_diff_deg}",
                f"Notiz: {note}",
            ]

            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            return candidate_dir

    def print_short_status(self):
        status = self.get_status()

        print("ArUco Monitor Status")

        if status["warnings"]:
            print("Warnungen")
            for msg in status["warnings"]:
                print("  " + msg)

        print("Kamera-Rigs")
        for name, item in status["rigs"].items():
            diff = item["baseline_diff_percent"]
            diff_text = "n/a" if diff is None else f"{diff:.1f} Prozent"
            expected = item["expected_baseline_cm"]
            expected_text = "n/a" if expected is None else f"{expected:.2f} cm"

            print(
                f"  {name}: baseline={item['baseline_cm']:.2f} cm | "
                f"erwartet={expected_text} | Abweichung={diff_text} | "
                f"T-Einheit={item['T_unit']} | T roh={item['T_raw']}"
            )

        print("Referenz-Kalibrierung")
        for pair_name, item in status["references"].items():
            if item["ok"]:
                print(
                    f"  {pair_name}: OK | Quelle={item['source']} | "
                    f"T={item['T_cm']} cm | Distanz={item['distance_cm']:.1f} cm"
                )
            else:
                print(f"  {pair_name}: fehlt oder nicht gespeichert | Quelle={item['source']}")

        print("Live-Schätzung")
        for pair_name, item in status["live_estimates"].items():
            if item["ok"]:
                print(
                    f"  {pair_name}: OK | Samples={item['samples']} | "
                    f"T={item['T_cm']} cm | Std max={item['std_max_cm']:.1f} cm"
                )
            else:
                print(f"  {pair_name}: nicht stabil | Grund={item['source']} | Samples={item['samples']}")

        gate = status["scan_gate"]

        print("Scan-Gate")
        if gate:
            print(f"  allow_scan={gate.get('allow_scan')}")
            for msg in gate.get("messages", []):
                print("  " + msg)
        else:
            print("  Noch kein Live-Update vorhanden.")

    def _detect_all_available_rigs(self, frame_elp2_bgr, frame_elp1_bgr, frame_ov_l_bgr, frame_ov_r_bgr):
        detections = {}

        if frame_elp2_bgr is not None and "ELP2" in self.rigs:
            left, right = self._split_stereo_frame(frame_elp2_bgr)
            detections["ELP2"] = self._detect_valid_markers_rig(left, right, "ELP2")

        if frame_elp1_bgr is not None and "ELP1" in self.rigs:
            left, right = self._split_stereo_frame(frame_elp1_bgr)
            detections["ELP1"] = self._detect_valid_markers_rig(left, right, "ELP1")

        if frame_ov_l_bgr is not None and frame_ov_r_bgr is not None and "OV9281" in self.rigs:
            detections["OV9281"] = self._detect_valid_markers_rig(frame_ov_l_bgr, frame_ov_r_bgr, "OV9281")

        return detections

    def _split_stereo_frame(self, frame):
        width = frame.shape[1]
        mid = width // 2
        return frame[:, :mid], frame[:, mid:]

    def _to_gray_equalized(self, frame):
        if frame is None:
            return None

        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return cv2.equalizeHist(gray)

    def _detect_valid_markers_rig(self, frame_l, frame_r, cam_name):
        rig = self.rigs[cam_name]

        gray_l = self._to_gray_equalized(frame_l)
        gray_r = self._to_gray_equalized(frame_r)

        left = self._detect_valid_markers_single(gray_l, rig["K_l"], rig["d_l"], cam_name, "L", rig)
        right = self._detect_valid_markers_single(gray_r, rig["K_r"], rig["d_r"], cam_name, "R", rig)

        merged = {}

        for marker_id, obs_list in left.items():
            merged.setdefault(marker_id, []).extend(obs_list)

        for marker_id, obs_list in right.items():
            merged.setdefault(marker_id, []).extend(obs_list)

        for marker_id in list(merged.keys()):
            merged[marker_id] = sorted(merged[marker_id], key=lambda obs: obs["err"])

        return merged

    def _detect_valid_markers_single(self, gray, K, dist, cam_name, side_name, rig):
        if gray is None:
            return {}

        corners, ids = self._detect_markers(gray)

        if ids is None:
            return {}

        result = {}

        for idx, marker_raw in enumerate(ids.flatten()):
            marker_id = int(marker_raw)

            if marker_id not in self.KNOWN_MARKER_IDS:
                continue

            if marker_id not in self.EXPECTED_VISIBLE_MARKERS.get(cam_name, set()):
                continue

            corner = np.asarray(corners[idx], dtype=np.float32).reshape(4, 2)

            flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
            ok, rvec, tvec = cv2.solvePnP(self.obj_pts, corner, K, dist, flags=flags)

            if not ok:
                continue

            if not self._marker_geometry_ok(corner, tvec):
                continue

            err = self._marker_reprojection_error(corner, rvec, tvec, K, dist)

            if err > self.MAX_REPROJECTION_ERROR_PX:
                continue

            if side_name == "R":
                rvec_l, tvec_l = self._right_pose_to_left_pose(rvec, tvec, rig)
            else:
                rvec_l, tvec_l = rvec, tvec

            obs = {
                "rvec": rvec_l,
                "tvec": tvec_l,
                "err": float(err),
                "side": side_name,
            }

            result.setdefault(marker_id, []).append(obs)

        return result

    def _detect_markers(self, gray):
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
            return corners, ids

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.params,
        )
        return corners, ids

    def _marker_geometry_ok(self, corner, tvec):
        dist_m = float(np.linalg.norm(tvec))

        if dist_m > self.MAX_MARKER_DISTANCE_M:
            return False

        area = float(cv2.contourArea(corner.astype(np.float32)))

        if area < self.MIN_MARKER_AREA_PX:
            return False

        width = float(np.linalg.norm(corner[0] - corner[1]))
        height = float(np.linalg.norm(corner[1] - corner[2]))

        if height <= 0:
            return False

        ratio = width / height

        if ratio < self.MIN_MARKER_RATIO or ratio > self.MAX_MARKER_RATIO:
            return False

        return True

    def _marker_reprojection_error(self, corner, rvec, tvec, K, dist):
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(4, 2)
        err = np.sqrt(((proj - corner) ** 2).sum(axis=1))
        return float(np.mean(err))

    def _right_pose_to_left_pose(self, rvec_r, tvec_r, rig):
        R_lr = rig["R_lr"]
        T_lr = rig["T_lr"]

        R_marker_r, _ = cv2.Rodrigues(rvec_r)

        R_marker_l = R_lr.T @ R_marker_r
        T_marker_l = R_lr.T @ (tvec_r.reshape(3) - T_lr)

        rvec_l, _ = cv2.Rodrigues(R_marker_l)

        return rvec_l, T_marker_l.reshape(3, 1)

    def _relative_from_common_marker(self, target_obs, source_obs):
        R_t, _ = cv2.Rodrigues(target_obs["rvec"])
        R_s, _ = cv2.Rodrigues(source_obs["rvec"])

        T_t = target_obs["tvec"].reshape(3)
        T_s = source_obs["tvec"].reshape(3)

        R_rel = R_t @ R_s.T
        T_rel = T_t - R_rel @ T_s

        score = float(target_obs["err"] + source_obs["err"])

        return R_rel, T_rel, score

    def _prune_buffers(self, now):
        min_time = now - self.BUFFER_SECONDS

        for buf in self.buffers.values():
            while buf and buf[0]["time"] < min_time:
                buf.popleft()

    def _estimate_pair_from_samples(self, pair_name):
        samples = list(self.buffers[pair_name])

        if len(samples) < self.MIN_SAMPLES_FOR_LIVE:
            return PairEstimate(False, None, None, len(samples), 999.0, 999.0, 999.0, "not_enough_samples")

        R_arr = np.array([s["R"] for s in samples])
        T_arr = np.array([s["T"] for s in samples])
        score_arr = np.array([s["score"] for s in samples])

        T_med_raw = np.median(T_arr, axis=0)
        dist_to_med = np.linalg.norm(T_arr - T_med_raw, axis=1)

        if len(T_arr) >= 15:
            limit = np.percentile(dist_to_med, 75)
            limit = max(float(limit), 0.03)
            mask_translation = dist_to_med <= limit * 2.5

            R_arr = R_arr[mask_translation]
            T_arr = T_arr[mask_translation]
            score_arr = score_arr[mask_translation]

        if len(T_arr) < self.MIN_SAMPLES_FOR_LIVE:
            return PairEstimate(False, None, None, len(T_arr), 999.0, 999.0, 999.0, "translation_filter_failed")

        if len(T_arr) >= 20:
            score_limit = np.percentile(score_arr, 70)
            mask_score = score_arr <= score_limit

            R_arr = R_arr[mask_score]
            T_arr = T_arr[mask_score]
            score_arr = score_arr[mask_score]

        if len(T_arr) < self.MIN_SAMPLES_FOR_LIVE:
            return PairEstimate(False, None, None, len(T_arr), 999.0, 999.0, 999.0, "score_filter_failed")

        T_med = np.median(T_arr, axis=0)
        R_avg = self._average_rotations(R_arr)

        std_cm = T_arr.std(axis=0) * 100.0
        std_max_cm = float(np.max(std_cm))

        reproj_median = float(np.median(score_arr))
        reproj_max = float(np.max(score_arr))

        if std_max_cm > self.MAX_STABLE_STD_CM:
            return PairEstimate(False, R_avg, T_med, len(T_arr), std_max_cm, reproj_median, reproj_max, "too_unstable")

        return PairEstimate(True, R_avg, T_med, len(T_arr), std_max_cm, reproj_median, reproj_max, "live_readonly")

    def _average_rotations(self, R_arr):
        R_mean = np.mean(R_arr, axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R_avg = U @ Vt

        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt

        return R_avg

    def _build_scan_gate_unlocked(self):
        allow_scan = True
        messages = []
        pair_status = {}

        for pair_name, rule in self.pair_rules.items():
            if not rule.get("required", False):
                continue

            saved = self.saved.get(pair_name, {})
            live = self.live_estimates.get(pair_name)

            item = {
                "saved_ok": bool(saved.get("ok", False)),
                "live_ok": bool(live.ok) if live is not None else False,
                "translation_diff_cm": None,
                "rotation_diff_deg": None,
                "level": "unknown",
            }

            if not saved.get("ok"):
                allow_scan = False
                item["level"] = "block"
                messages.append(f"{pair_name}: gespeicherte Referenz-Kalibrierung fehlt.")
                pair_status[pair_name] = item
                continue

            if live is None or not live.ok or live.R is None or live.T is None:
                allow_scan = False
                item["level"] = "block"
                reason = live.source if live is not None else "no_live_estimate"
                messages.append(f"{pair_name}: ArUco-Position konnte nicht stabil geprüft werden. Grund: {reason}.")
                pair_status[pair_name] = item
                continue

            t_diff_cm = float(np.linalg.norm(live.T - saved["T"]) * 100.0)
            r_diff_deg = self._rotation_diff_deg(live.R, saved["R"])

            item["translation_diff_cm"] = t_diff_cm
            item["rotation_diff_deg"] = r_diff_deg

            if t_diff_cm >= self.BLOCK_TRANSLATION_SHIFT_CM or r_diff_deg >= self.BLOCK_ROTATION_SHIFT_DEG:
                allow_scan = False
                item["level"] = "block"
                messages.append(
                    f"{pair_name}: starke Kameraverschiebung erkannt. "
                    f"Abweichung {t_diff_cm:.1f} cm und {r_diff_deg:.1f} Grad."
                )
            elif t_diff_cm >= self.WARN_TRANSLATION_SHIFT_CM or r_diff_deg >= self.WARN_ROTATION_SHIFT_DEG:
                item["level"] = "warn"
                messages.append(
                    f"{pair_name}: mögliche Kameraverschiebung. "
                    f"Abweichung {t_diff_cm:.1f} cm und {r_diff_deg:.1f} Grad."
                )
            else:
                item["level"] = "ok"
                messages.append(
                    f"{pair_name}: ArUco-Kontrolle plausibel. "
                    f"Abweichung {t_diff_cm:.1f} cm und {r_diff_deg:.1f} Grad."
                )

            pair_status[pair_name] = item

        return {
            "allow_scan": allow_scan,
            "messages": messages,
            "pair_status": pair_status,
        }

    def _rotation_diff_deg(self, R_a, R_b):
        R_delta = R_a @ R_b.T
        trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace_val)))

    def _detections_to_status(self, detections):
        out = {}

        for cam, det in detections.items():
            out[cam] = {}
            for marker_id, obs_list in det.items():
                out[cam][int(marker_id)] = {
                    "sides": [obs["side"] for obs in obs_list],
                    "errors": [float(obs["err"]) for obs in obs_list],
                }

        return out

    def _references_to_status(self):
        out = {}

        for pair_name, item in self.saved.items():
            T = item.get("T")
            ok = bool(item.get("ok", False))

            out[pair_name] = {
                "ok": ok,
                "source": item.get("source", "unknown"),
                "T_cm": None if T is None else np.round(T * 100.0, 2).tolist(),
                "distance_cm": None if T is None else float(np.linalg.norm(T) * 100.0),
                "R_file": item.get("R_file"),
                "T_file": item.get("T_file"),
            }

        return out

    def _live_estimates_to_status(self):
        out = {}

        for pair_name, estimate in self.live_estimates.items():
            T = estimate.T

            out[pair_name] = {
                "ok": bool(estimate.ok),
                "source": estimate.source,
                "samples": int(estimate.samples),
                "T_cm": None if T is None else np.round(T * 100.0, 2).tolist(),
                "distance_cm": None if T is None else float(np.linalg.norm(T) * 100.0),
                "std_max_cm": float(estimate.std_max_cm),
                "reproj_median": float(estimate.reproj_median),
                "reproj_max": float(estimate.reproj_max),
            }

        return out

    def _rigs_to_status(self):
        out = {}

        for name, rig in self.rigs.items():
            sanity = rig.get("baseline_sanity", {})

            expected = sanity.get("expected_baseline_m")
            diff_percent = sanity.get("baseline_diff_percent")

            out[name] = {
                "baseline_cm": float(rig["baseline_m"] * 100.0),
                "expected_baseline_cm": None if expected is None else float(expected * 100.0),
                "baseline_diff_percent": diff_percent,
                "baseline_sanity_level": sanity.get("level"),
                "T_unit": rig["T_unit"],
                "T_raw": np.round(rig["T_lr_raw"], 6).tolist(),
            }

        return out


def print_yaml_T_unit_report(base):
    base = Path(base).expanduser()

    files = [
        ("ELP2", "stereo_config.yaml"),
        ("ELP1", "stereo_config_elp1.yaml"),
        ("OV9281", "stereo_config_ov9281.yaml"),
    ]

    print("YAML T-Einheiten Prüfung")
    print("Diese Ausgabe hilft zu erkennen, ob T wahrscheinlich in m, cm oder mm gespeichert ist.")

    for rig_name, filename in files:
        path = base / filename

        print()
        print(f"{rig_name}: {filename}")

        if not path.exists():
            print("  Datei nicht gefunden.")
            continue

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            print("  YAML konnte nicht als Dictionary gelesen werden.")
            continue

        if "T" not in cfg:
            print("  Kein T-Wert vorhanden.")
            continue

        try:
            T = np.array(cfg["T"], dtype=np.float64).reshape(3)
        except Exception as exc:
            print("  T konnte nicht als 3D-Vektor gelesen werden:", exc)
            continue

        norm_raw = float(np.linalg.norm(T))
        expected = ArucoMonitor.EXPECTED_BASELINE_M.get(rig_name)

        print(f"  T roh: {T.tolist()}")
        print(f"  Norm roh: {norm_raw:.6f}")
        print(f"  Falls T Meter ist:      {norm_raw * 100.0:.2f} cm")
        print(f"  Falls T Zentimeter ist: {norm_raw:.2f} cm")
        print(f"  Falls T Millimeter ist: {norm_raw / 10.0:.2f} cm")

        if expected is not None:
            print(f"  Erwartete Baseline:     {expected * 100.0:.2f} cm")

        candidates = []

        unit_to_baseline_m = {
            "m": norm_raw,
            "cm": norm_raw * 0.01,
            "mm": norm_raw * 0.001,
        }

        for unit, baseline_m in unit_to_baseline_m.items():
            baseline_cm = baseline_m * 100.0

            if 1.0 <= baseline_cm <= 50.0:
                diff_percent = None

                if expected is not None:
                    diff_percent = abs(baseline_m - expected) / expected * 100.0

                candidates.append((unit, baseline_cm, diff_percent))

        if candidates:
            print("  Plausible Einheiten:")
            for unit, baseline_cm, diff_percent in candidates:
                if diff_percent is None:
                    print(f"    {unit}: {baseline_cm:.2f} cm")
                else:
                    print(f"    {unit}: {baseline_cm:.2f} cm, Abweichung {diff_percent:.1f} Prozent")
        else:
            print("  Keine plausible Einheit im Bereich 1 bis 50 cm erkannt.")


def main():
    parser = argparse.ArgumentParser(description="Read-only ArUco Monitor für ANTHRO3D")
    parser.add_argument("--base", default="~/anthro3d")
    parser.add_argument("--require-ov", action="store_true")
    parser.add_argument("--unit-elp2", default="auto", choices=["auto", "m", "cm", "mm"])
    parser.add_argument("--unit-elp1", default="auto", choices=["auto", "m", "cm", "mm"])
    parser.add_argument("--unit-ov9281", default="auto", choices=["auto", "m", "cm", "mm"])
    parser.add_argument("--inspect-yaml", action="store_true")
    parser.add_argument("--only-inspect", action="store_true")

    args = parser.parse_args()

    if args.inspect_yaml:
        print_yaml_T_unit_report(args.base)

        if args.only_inspect:
            return

        print()

    monitor = ArucoMonitor.from_files(
        args.base,
        require_ov=args.require_ov,
        stereo_T_units={
            "ELP2": args.unit_elp2,
            "ELP1": args.unit_elp1,
            "OV9281": args.unit_ov9281,
        },
    )

    monitor.print_short_status()

    print()
    print("Hinweis:")
    print("Dieser Direktaufruf liest keine Kamera aus.")
    print("Für eine Live-Prüfung muss scan3d.py fertige Frames an monitor.update(...) übergeben.")
    print("Referenz-Kalibrierungen werden von diesem Modul niemals automatisch überschrieben.")


if __name__ == "__main__":
    main()
