from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from aruco_monitor import ArucoMonitor
from camera_system.scan_capture import AnthroCameraCapture


BASE = Path.home() / "anthro3d"

OUT = (
    BASE
    / "aruco_state"
    / "marker_integrity_reference.yaml"
)

RUN_SECONDS = 30.0

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


def choose_pair(top_obs, bottom_obs):
    candidates = []

    for top in top_obs:
        for bottom in bottom_obs:
            same_side = (
                top.get("side")
                == bottom.get("side")
            )

            score = (
                float(
                    top.get("err", 999.0)
                )
                + float(
                    bottom.get("err", 999.0)
                )
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
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return (
        candidates[0][2],
        candidates[0][3],
    )


def measure_pair(
    detections,
    top_id,
    bottom_id,
):
    top_list = detections.get(
        top_id,
        [],
    )

    bottom_list = detections.get(
        bottom_id,
        [],
    )

    if not top_list or not bottom_list:
        return None

    selected = choose_pair(
        top_list,
        bottom_list,
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

    delta = (
        T_bottom
        - T_top
    )

    offset_cm = (
        R_top.T
        @ delta
    ) * 100.0

    distance_cm = float(
        np.linalg.norm(delta)
        * 100.0
    )

    R_rel = (
        R_top.T
        @ R_bottom
    )

    rotation_deg = (
        rotation_angle_deg(
            R_rel
        )
    )

    return {
        "offset_cm": offset_cm,
        "distance_cm": distance_cm,
        "rotation_deg": rotation_deg,
    }


def main():
    print()
    print("=" * 76)
    print(
        "ANTHRO3D – "
        "MARKER-INTEGRITÄTSREFERENZ"
    )
    print("=" * 76)

    monitor = ArucoMonitor.from_files(
        base=BASE,
        require_ov=True,
        stereo_T_units="m",
    )

    captures = {}

    data = defaultdict(
        lambda: {
            "offset": [],
            "distance": [],
            "rotation": [],
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
                    f"{role}: Kamera "
                    "konnte nicht geöffnet werden."
                )

            captures[role] = cap

            print(
                f"{role:<10} offen"
            )

        print()
        print(
            "30 Sekunden Referenzmessung."
        )
        print(
            "NICHTS bewegen."
        )

        start = time.monotonic()

        while (
            time.monotonic()
            - start
            < RUN_SECONDS
        ):
            frames = {}

            valid = True

            for role in ROLES:
                ok, frame = (
                    captures[role].read()
                )

                if (
                    not ok
                    or frame is None
                ):
                    valid = False
                    break

                frames[role] = frame

            if not valid:
                continue

            monitor.update(
                frame_elp2_bgr=(
                    frames["ELP2"]
                ),
                frame_elp1_bgr=(
                    frames["ELP1"]
                ),
                frame_ov_l_bgr=(
                    frames["OV9281_L"]
                ),
                frame_ov_r_bgr=(
                    frames["OV9281_R"]
                ),
            )

            all_det = (
                monitor.last_detections
            )

            for (
                rig_name,
                stand_names,
            ) in VISIBLE_STANDS.items():

                detections = (
                    all_det.get(
                        rig_name,
                        {},
                    )
                )

                for stand_name in stand_names:
                    top_id, bottom_id = (
                        STANDS[stand_name]
                    )

                    m = measure_pair(
                        detections,
                        top_id,
                        bottom_id,
                    )

                    if m is None:
                        continue

                    key = (
                        rig_name,
                        stand_name,
                    )

                    data[key][
                        "offset"
                    ].append(
                        m["offset_cm"]
                    )

                    data[key][
                        "distance"
                    ].append(
                        m["distance_cm"]
                    )

                    data[key][
                        "rotation"
                    ].append(
                        m["rotation_deg"]
                    )

        output = {
            "schema_version": 1,
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "duration_seconds": (
                RUN_SECONDS
            ),
            "observers": {},
        }

        print()
        print(
            "Ermittelte Referenzen:"
        )

        for key in sorted(data):
            rig_name, stand_name = key

            values = data[key]

            n = len(
                values["distance"]
            )

            if n < 30:
                raise RuntimeError(
                    f"{rig_name}/"
                    f"{stand_name}: "
                    f"nur {n} Samples."
                )

            offset_med, offset_mad = (
                robust_stats(
                    values["offset"]
                )
            )

            dist_med, dist_mad = (
                robust_stats(
                    values["distance"]
                )
            )

            rot_med, rot_mad = (
                robust_stats(
                    values["rotation"]
                )
            )

            top_id, bottom_id = (
                STANDS[stand_name]
            )

            observer = (
                output[
                    "observers"
                ].setdefault(
                    rig_name,
                    {},
                )
            )

            observer[
                stand_name
            ] = {
                "marker_ids": [
                    int(top_id),
                    int(bottom_id),
                ],
                "samples": int(n),

                "offset_local_cm_median": [
                    float(v)
                    for v in offset_med
                ],

                "offset_local_cm_mad": [
                    float(v)
                    for v in offset_mad
                ],

                "distance_cm_median": (
                    float(dist_med)
                ),

                "distance_cm_mad": (
                    float(dist_mad)
                ),

                "rotation_deg_median": (
                    float(rot_med)
                ),

                "rotation_deg_mad": (
                    float(rot_mad)
                ),
            }

            print(
                f"  {rig_name:<8} "
                f"{stand_name:<14} "
                f"n={n:<4} "
                f"Dist="
                f"{float(dist_med):.3f} "
                f"±MAD "
                f"{float(dist_mad):.3f} cm | "
                f"Rot="
                f"{float(rot_med):.3f} "
                f"±MAD "
                f"{float(rot_mad):.3f}°"
            )

        if len(output["observers"]) != 3:
            raise RuntimeError(
                "Nicht alle drei Kamerastationen "
                "haben gültige Referenzen."
            )

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = OUT.with_suffix(
            ".yaml.tmp"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                output,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        tmp.replace(OUT)

        print()
        print(
            f"Gespeichert: {OUT}"
        )

    finally:
        print()
        print(
            "Kameras freigeben ..."
        )

        for role in reversed(ROLES):
            cap = captures.get(role)

            if cap is None:
                continue

            try:
                cap.release()
                print(
                    f"  {role}: frei"
                )
            except Exception as exc:
                print(
                    f"  WARNUNG "
                    f"{role}: {exc}"
                )


if __name__ == "__main__":
    main()
