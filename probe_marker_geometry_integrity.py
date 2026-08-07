from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from aruco_monitor import ArucoMonitor
from camera_system.scan_capture import AnthroCameraCapture


BASE = Path.home() / "anthro3d"
RUN_SECONDS = 30.0
PRINT_INTERVAL_S = 2.0

ROLES = [
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
]

STANDS = {
    "OV9281_STAND": (2, 20),
    "ELP1_STAND": (3, 30),
    "ELP2_STAND": (4, 40),
}

# Welche Markerpaare eine Kamerastation sehen können.
VISIBLE_STANDS = {
    "ELP2": (
        "OV9281_STAND",
        "ELP1_STAND",
    ),
    "ELP1": (
        "OV9281_STAND",
        "ELP2_STAND",
    ),
    "OV9281": (
        "ELP1_STAND",
        "ELP2_STAND",
    ),
}


def rotation_angle_deg(R):
    value = np.clip(
        (np.trace(R) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(value)))


def choose_pair(top_obs, bottom_obs):
    """
    Bevorzugt Beobachtungen aus derselben Stereo-Seite.
    Falls das nicht möglich ist, wird die Kombination mit
    dem kleinsten Reprojektionsfehler genommen.
    """
    candidates = []

    for top in top_obs:
        for bottom in bottom_obs:
            same_side = (
                top.get("side") == bottom.get("side")
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

    _, _, top, bottom = candidates[0]

    return top, bottom


def measure_pair(detections, top_id, bottom_id):
    top_list = detections.get(top_id, [])
    bottom_list = detections.get(bottom_id, [])

    if not top_list or not bottom_list:
        return None

    chosen = choose_pair(
        top_list,
        bottom_list,
    )

    if chosen is None:
        return None

    top, bottom = chosen

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

    delta_camera_m = T_bottom - T_top

    # Position des unteren Markers im lokalen
    # Koordinatensystem des oberen Markers.
    offset_local_cm = (
        R_top.T @ delta_camera_m
    ) * 100.0

    distance_cm = float(
        np.linalg.norm(delta_camera_m)
        * 100.0
    )

    R_relative = R_top.T @ R_bottom

    rot_angle_deg = rotation_angle_deg(
        R_relative
    )

    return {
        "offset_local_cm": offset_local_cm,
        "distance_cm": distance_cm,
        "rotation_angle_deg": rot_angle_deg,
        "top_side": top.get("side"),
        "bottom_side": bottom.get("side"),
        "reproj_sum": (
            float(top.get("err", 0.0))
            + float(bottom.get("err", 0.0))
        ),
    }


def median_and_mad(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    med = np.median(
        arr,
        axis=0,
    )

    mad = np.median(
        np.abs(arr - med),
        axis=0,
    )

    return med, mad


def main():
    ref_path = (
        BASE
        / "aruco_state"
        / "marker_vertical_offsets.yaml"
    )

    with ref_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        reference = yaml.safe_load(f)

    print()
    print("=" * 78)
    print("ANTHRO3D – MARKER-GEOMETRIE-INTEGRITÄT")
    print("=" * 78)

    print()
    print("Gespeicherte Hardware-Referenz:")

    for stand, (top_id, bottom_id) in STANDS.items():
        pooled = reference[stand]["pooled"]

        print(
            f"  {stand:<14} "
            f"ID {top_id}->{bottom_id} | "
            f"Dist={pooled['distance_cm_median']:.3f} cm | "
            f"Rot={pooled['rotation_angle_deg_median']:.3f}°"
        )

    monitor = ArucoMonitor.from_files(
        base=BASE,
        require_ov=True,
        stereo_T_units="m",
    )

    captures = {}

    samples = defaultdict(
        lambda: {
            "offset": [],
            "distance": [],
            "rotation": [],
        }
    )

    try:
        print()
        print("Öffne Kameras ...")

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
            print(f"  {role}: offen")

        print()
        print(
            "30 Sekunden messen – "
            "Kameras, Stative und Marker NICHT bewegen."
        )
        print()

        start = time.monotonic()
        last_print = 0.0

        while True:
            elapsed = (
                time.monotonic()
                - start
            )

            if elapsed >= RUN_SECONDS:
                break

            frames = {}

            good = True

            for role in ROLES:
                ok, frame = captures[role].read()

                if not ok or frame is None:
                    good = False
                    break

                frames[role] = frame

            if not good:
                continue

            monitor.update(
                frame_elp2_bgr=frames["ELP2"],
                frame_elp1_bgr=frames["ELP1"],
                frame_ov_l_bgr=frames["OV9281_L"],
                frame_ov_r_bgr=frames["OV9281_R"],
            )

            detections_all = monitor.last_detections

            latest = {}

            for rig_name, stand_names in VISIBLE_STANDS.items():
                rig_detections = detections_all.get(
                    rig_name,
                    {},
                )

                for stand_name in stand_names:
                    top_id, bottom_id = STANDS[
                        stand_name
                    ]

                    measurement = measure_pair(
                        rig_detections,
                        top_id,
                        bottom_id,
                    )

                    if measurement is None:
                        continue

                    key = (
                        rig_name,
                        stand_name,
                    )

                    samples[key]["offset"].append(
                        measurement[
                            "offset_local_cm"
                        ]
                    )

                    samples[key]["distance"].append(
                        measurement[
                            "distance_cm"
                        ]
                    )

                    samples[key]["rotation"].append(
                        measurement[
                            "rotation_angle_deg"
                        ]
                    )

                    latest[key] = measurement

            now = time.monotonic()

            if (
                now - last_print
                < PRINT_INTERVAL_S
            ):
                continue

            last_print = now

            print(
                f"--- {elapsed:5.1f}s ---"
            )

            for key in sorted(latest):
                rig_name, stand_name = key

                measurement = latest[key]

                ref = reference[
                    stand_name
                ]["pooled"]

                d_ref = float(
                    ref[
                        "distance_cm_median"
                    ]
                )

                r_ref = float(
                    ref[
                        "rotation_angle_deg_median"
                    ]
                )

                print(
                    f"{rig_name:<8} "
                    f"{stand_name:<14} "
                    f"Dist="
                    f"{measurement['distance_cm']:6.2f} "
                    f"(Δ "
                    f"{measurement['distance_cm'] - d_ref:+5.2f}) cm | "
                    f"Rot="
                    f"{measurement['rotation_angle_deg']:5.2f} "
                    f"(Δ "
                    f"{measurement['rotation_angle_deg'] - r_ref:+5.2f})°"
                )

        print()
        print("=" * 78)
        print("ERGEBNIS – 30 SEKUNDEN")
        print("=" * 78)

        for key in sorted(samples):
            rig_name, stand_name = key
            data = samples[key]

            if not data["distance"]:
                continue

            ref = reference[
                stand_name
            ]["pooled"]

            offset_med, offset_mad = median_and_mad(
                data["offset"]
            )

            dist_med, dist_mad = median_and_mad(
                data["distance"]
            )

            rot_med, rot_mad = median_and_mad(
                data["rotation"]
            )

            ref_offset = np.asarray(
                ref[
                    "offset_local_cm_median"
                ],
                dtype=np.float64,
            )

            ref_dist = float(
                ref[
                    "distance_cm_median"
                ]
            )

            ref_rot = float(
                ref[
                    "rotation_angle_deg_median"
                ]
            )

            offset_delta = (
                offset_med
                - ref_offset
            )

            print()
            print(
                f"{rig_name} sieht "
                f"{stand_name}"
            )

            print(
                f"  Samples: "
                f"{len(data['distance'])}"
            )

            print(
                "  Offset median: "
                f"{np.round(offset_med, 3).tolist()} cm"
            )

            print(
                "  Referenz:      "
                f"{np.round(ref_offset, 3).tolist()} cm"
            )

            print(
                "  Offset Δ:      "
                f"{np.round(offset_delta, 3).tolist()} cm"
            )

            print(
                "  Offset MAD:    "
                f"{np.round(offset_mad, 3).tolist()} cm"
            )

            print(
                f"  Distanz: "
                f"{float(dist_med):.3f} cm | "
                f"Referenz {ref_dist:.3f} | "
                f"Δ {float(dist_med-ref_dist):+.3f} | "
                f"MAD {float(dist_mad):.3f}"
            )

            print(
                f"  Rotation: "
                f"{float(rot_med):.3f}° | "
                f"Referenz {ref_rot:.3f}° | "
                f"Δ {float(rot_med-ref_rot):+.3f}° | "
                f"MAD {float(rot_mad):.3f}°"
            )

    finally:
        print()
        print("Kameras freigeben ...")

        for role in reversed(ROLES):
            cap = captures.get(role)

            if cap is None:
                continue

            try:
                cap.release()
                print(
                    f"  {role}: freigegeben"
                )
            except Exception as exc:
                print(
                    f"  WARNUNG {role}: {exc}"
                )


if __name__ == "__main__":
    main()
