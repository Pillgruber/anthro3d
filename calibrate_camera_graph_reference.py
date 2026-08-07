from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import yaml

from aruco_monitor import ArucoMonitor
from marker_integrity_monitor import MarkerIntegrityMonitor
from camera_system.scan_capture import AnthroCameraCapture


BASE = Path.home() / "anthro3d"

OUT = (
    BASE
    / "aruco_state"
    / "camera_graph_reference.yaml"
)

ROLES = [
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
]

PAIR_NAMES = [
    "ELP1_to_ELP2",
    "OV9281_to_ELP2",
    "OV9281_to_ELP1",
]

RUN_SECONDS = 30.0


def rotation_diff_deg(R_a, R_b):
    R_delta = (
        np.asarray(R_a)
        @ np.asarray(R_b).T
    )

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


def triangle_closure(estimates):
    """
    Beziehungen:

    ELP1_to_ELP2:
        ELP1 -> ELP2

    OV9281_to_ELP1:
        OV -> ELP1

    OV9281_to_ELP2:
        OV -> ELP2

    Vergleich:
        OV -> ELP2 direkt

    gegen:
        OV -> ELP1 -> ELP2
    """

    e12 = estimates[
        "ELP1_to_ELP2"
    ]

    e02 = estimates[
        "OV9281_to_ELP2"
    ]

    e01 = estimates[
        "OV9281_to_ELP1"
    ]

    R12 = np.asarray(
        e12.R,
        dtype=np.float64,
    )

    T12 = np.asarray(
        e12.T,
        dtype=np.float64,
    ).reshape(3)

    R02 = np.asarray(
        e02.R,
        dtype=np.float64,
    )

    T02 = np.asarray(
        e02.T,
        dtype=np.float64,
    ).reshape(3)

    R01 = np.asarray(
        e01.R,
        dtype=np.float64,
    )

    T01 = np.asarray(
        e01.T,
        dtype=np.float64,
    ).reshape(3)

    # OV -> ELP1 -> ELP2
    R_indirect = (
        R12
        @ R01
    )

    T_indirect = (
        R12
        @ T01
        + T12
    )

    translation_error_cm = float(
        np.linalg.norm(
            T_indirect - T02
        )
        * 100.0
    )

    rotation_error_deg = (
        rotation_diff_deg(
            R_indirect,
            R02,
        )
    )

    return {
        "translation_error_cm": (
            translation_error_cm
        ),
        "rotation_error_deg": (
            rotation_error_deg
        ),
        "direct_T_m": [
            float(x)
            for x in T02
        ],
        "indirect_T_m": [
            float(x)
            for x in T_indirect
        ],
    }


def main():
    print()
    print("=" * 78)
    print("ANTHRO3D – KAMERA-GRAPH-REFERENZ")
    print("=" * 78)

    aruco = ArucoMonitor.from_files(
        base=BASE,
        require_ov=True,
        stereo_T_units="m",
    )

    integrity = MarkerIntegrityMonitor(
        base=BASE,
    )

    captures = {}

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
                    f"{role}: Kamera konnte "
                    "nicht geöffnet werden."
                )

            captures[role] = cap
            print(
                f"  {role}: offen"
            )

        print()
        print(
            "30 Sekunden Kamera-Graph messen."
        )
        print(
            "Kameras, Stative und Marker NICHT bewegen."
        )
        print()

        start = time.monotonic()

        integrity_status = None
        aruco_status = None

        while (
            time.monotonic()
            - start
            < RUN_SECONDS
        ):
            frames = {}
            good = True

            for role in ROLES:
                ok, frame = captures[
                    role
                ].read()

                if (
                    not ok
                    or frame is None
                ):
                    good = False
                    break

                frames[role] = frame

            if not good:
                continue

            now = time.time()

            aruco_status = aruco.update(
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
                now=now,
            )

            integrity_status = (
                integrity.update(
                    aruco.last_detections,
                    now=now,
                )
            )

        print()
        print("Prüfe Referenz ...")

        if integrity_status is None:
            raise RuntimeError(
                "Keine Integritätsdaten vorhanden."
            )

        if not integrity_status.get(
            "safe_for_auto_recalibration",
            False,
        ):
            raise RuntimeError(
                "Markerintegrität nicht bestätigt. "
                "Graph-Referenz wird NICHT gespeichert."
            )

        estimates = aruco.live_estimates

        for pair_name in PAIR_NAMES:
            estimate = estimates.get(
                pair_name
            )

            if (
                estimate is None
                or not estimate.ok
                or estimate.R is None
                or estimate.T is None
            ):
                raise RuntimeError(
                    f"{pair_name}: "
                    "keine stabile Graph-Kante."
                )

        closure = triangle_closure(
            estimates
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

            "marker_integrity_confirmed": (
                True
            ),

            "pairs": {},

            "triangle_closure": closure,
        }

        print()
        print("Graph-Kanten:")

        for pair_name in PAIR_NAMES:
            estimate = estimates[
                pair_name
            ]

            R = np.asarray(
                estimate.R,
                dtype=np.float64,
            )

            T = np.asarray(
                estimate.T,
                dtype=np.float64,
            ).reshape(3)

            distance_cm = float(
                np.linalg.norm(T)
                * 100.0
            )

            output["pairs"][
                pair_name
            ] = {
                "R": [
                    [
                        float(x)
                        for x in row
                    ]
                    for row in R
                ],

                "T_m": [
                    float(x)
                    for x in T
                ],

                "distance_cm": (
                    distance_cm
                ),

                "samples": int(
                    estimate.samples
                ),

                "std_max_cm": float(
                    estimate.std_max_cm
                ),

                "reproj_median": float(
                    estimate.reproj_median
                ),

                "reproj_max": float(
                    estimate.reproj_max
                ),

                "source": str(
                    estimate.source
                ),
            }

            print(
                f"  {pair_name:<20} "
                f"Dist="
                f"{distance_cm:7.2f} cm | "
                f"Samples="
                f"{estimate.samples:<4} | "
                f"Std="
                f"{estimate.std_max_cm:.2f} cm"
            )

        print()
        print("Dreiecksschluss:")

        print(
            "  Translation error: "
            f"{closure['translation_error_cm']:.2f} cm"
        )

        print(
            "  Rotation error:    "
            f"{closure['rotation_error_deg']:.2f}°"
        )

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = OUT.with_suffix(
            ".yaml.tmp"
        )

        with temp.open(
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                output,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        temp.replace(OUT)

        print()
        print(
            f"Gespeichert: {OUT}"
        )

        print()
        print(
            "WICHTIG:"
        )

        print(
            "Bestehende R_rel/T_rel-Dateien "
            "wurden NICHT verändert."
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
                    f"  {role}: frei"
                )
            except Exception as exc:
                print(
                    f"  WARNUNG "
                    f"{role}: {exc}"
                )


if __name__ == "__main__":
    main()
