from __future__ import annotations

from collections import defaultdict
import time

import cv2
import numpy as np

from aruco_monitor import ArucoMonitor
from camera_system.scan_capture import AnthroCameraCapture


RUN_SECONDS = 30.0

ROLES = (
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
)


def rotation_diff_deg(rvec_a, rvec_b):
    R_a, _ = cv2.Rodrigues(
        np.asarray(rvec_a, dtype=np.float64).reshape(3, 1)
    )

    R_b, _ = cv2.Rodrigues(
        np.asarray(rvec_b, dtype=np.float64).reshape(3, 1)
    )

    R_delta = R_a @ R_b.T

    value = np.clip(
        (np.trace(R_delta) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


def robust_stats(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    median = np.median(
        arr,
        axis=0,
    )

    mad = np.median(
        np.abs(arr - median),
        axis=0,
    )

    return median, mad


def best_side(obs_list, side):
    items = [
        obs
        for obs in obs_list
        if obs.get("side") == side
    ]

    if not items:
        return None

    return min(
        items,
        key=lambda obs: float(
            obs.get("err", 999.0)
        ),
    )


def main():
    print()
    print("=" * 78)
    print("ANTHRO3D – STEREO POSE CONSISTENCY")
    print("Linke Pose gegen transformierte rechte Pose")
    print("=" * 78)

    monitor = ArucoMonitor.from_files(
        base="~/anthro3d",
        require_ov=True,
        stereo_T_units="m",
    )

    captures = {}

    data = defaultdict(
        lambda: {
            "t_delta": [],
            "xyz_delta": [],
            "r_delta": [],
            "err_l": [],
            "err_r": [],
        }
    )

    try:
        for role in ROLES:
            cap = AnthroCameraCapture(
                role,
                read_timeout=5.0,
            )

            if not cap.isOpened():
                cap.release()

                raise RuntimeError(
                    f"{role}: Kamera konnte nicht geöffnet werden."
                )

            captures[role] = cap

            print(
                f"{role:<10} offen"
            )

        print()
        print(
            "30 Sekunden messen – nichts bewegen."
        )
        print()

        start = time.monotonic()

        while (
            time.monotonic() - start
            < RUN_SECONDS
        ):
            frames = {}
            good = True

            for role in ROLES:
                ok, frame = captures[
                    role
                ].read()

                if not ok or frame is None:
                    good = False
                    break

                frames[role] = frame

            if not good:
                continue

            detections = (
                monitor._detect_all_available_rigs(
                    frame_elp2_bgr=frames["ELP2"],
                    frame_elp1_bgr=frames["ELP1"],
                    frame_ov_l_bgr=frames["OV9281_L"],
                    frame_ov_r_bgr=frames["OV9281_R"],
                )
            )

            for rig_name, marker_map in detections.items():

                for marker_id, obs_list in marker_map.items():

                    left = best_side(
                        obs_list,
                        "L",
                    )

                    right = best_side(
                        obs_list,
                        "R",
                    )

                    # Wir vergleichen nur Frames, in denen
                    # derselbe Marker auf BEIDEN Seiten erkannt wurde.
                    if left is None or right is None:
                        continue

                    T_l = np.asarray(
                        left["tvec"],
                        dtype=np.float64,
                    ).reshape(3)

                    # Wichtig:
                    # right["tvec"] wurde im ArucoMonitor
                    # bereits R -> L transformiert.
                    T_r_as_l = np.asarray(
                        right["tvec"],
                        dtype=np.float64,
                    ).reshape(3)

                    xyz_delta_cm = (
                        T_r_as_l - T_l
                    ) * 100.0

                    t_delta_cm = float(
                        np.linalg.norm(
                            xyz_delta_cm
                        )
                    )

                    r_delta_deg = (
                        rotation_diff_deg(
                            left["rvec"],
                            right["rvec"],
                        )
                    )

                    key = (
                        rig_name,
                        int(marker_id),
                    )

                    data[key]["t_delta"].append(
                        t_delta_cm
                    )

                    data[key]["xyz_delta"].append(
                        xyz_delta_cm
                    )

                    data[key]["r_delta"].append(
                        r_delta_deg
                    )

                    data[key]["err_l"].append(
                        float(left["err"])
                    )

                    data[key]["err_r"].append(
                        float(right["err"])
                    )

        print()
        print("=" * 78)
        print("ERGEBNIS")
        print("=" * 78)

        rig_summary = defaultdict(list)

        for key in sorted(data):
            rig_name, marker_id = key
            d = data[key]

            n = len(
                d["t_delta"]
            )

            if n < 10:
                print(
                    f"{rig_name:<8} ID{marker_id:<2}: "
                    f"zu wenige gemeinsame L/R Samples ({n})"
                )
                continue

            t_med, t_mad = robust_stats(
                d["t_delta"]
            )

            xyz_med, xyz_mad = robust_stats(
                d["xyz_delta"]
            )

            r_med, r_mad = robust_stats(
                d["r_delta"]
            )

            err_l_med = float(
                np.median(
                    d["err_l"]
                )
            )

            err_r_med = float(
                np.median(
                    d["err_r"]
                )
            )

            print()
            print(
                f"{rig_name} / ID{marker_id}"
            )

            print(
                f"  gemeinsame Samples: {n}"
            )

            print(
                f"  L↔R Translation Δ: "
                f"{float(t_med):.3f} cm "
                f"| MAD {float(t_mad):.3f}"
            )

            print(
                "  XYZ Δ R→L minus L: "
                f"{np.round(xyz_med, 3).tolist()} cm"
            )

            print(
                "  XYZ MAD:            "
                f"{np.round(xyz_mad, 3).tolist()} cm"
            )

            print(
                f"  Rotation Δ: "
                f"{float(r_med):.3f}° "
                f"| MAD {float(r_mad):.3f}°"
            )

            print(
                f"  Reproj median: "
                f"L={err_l_med:.3f}px "
                f"R={err_r_med:.3f}px"
            )

            rig_summary[
                rig_name
            ].append(
                {
                    "marker": marker_id,
                    "t": float(t_med),
                    "r": float(r_med),
                }
            )

        print()
        print("=" * 78)
        print("RIG-ZUSAMMENFASSUNG")
        print("=" * 78)

        for rig_name in (
            "ELP2",
            "ELP1",
            "OV9281",
        ):
            rows = rig_summary.get(
                rig_name,
                [],
            )

            if not rows:
                print(
                    f"{rig_name:<8}: "
                    "keine ausreichenden L/R-Vergleiche"
                )
                continue

            t_values = [
                row["t"]
                for row in rows
            ]

            r_values = [
                row["r"]
                for row in rows
            ]

            print(
                f"{rig_name:<8}: "
                f"Median Marker-Translationsfehler "
                f"{np.median(t_values):.2f} cm | "
                f"Median Rotationsfehler "
                f"{np.median(r_values):.2f}°"
            )

        print()
        print(
            "Nichts wurde gespeichert oder verändert."
        )

    finally:
        print()
        print(
            "Kameras freigeben ..."
        )

        for role in reversed(
            ROLES
        ):
            cap = captures.get(
                role
            )

            if cap is None:
                continue

            try:
                cap.release()
                print(
                    f"  {role}: frei"
                )
            except Exception as exc:
                print(
                    f"  WARNUNG {role}: {exc}"
                )


if __name__ == "__main__":
    main()
