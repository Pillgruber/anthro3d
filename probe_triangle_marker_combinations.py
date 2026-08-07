from __future__ import annotations

from collections import defaultdict
import itertools
import time

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

# pair_name: target, source, mögliche gemeinsame Marker
PAIR_RULES = {
    "ELP1_to_ELP2": (
        "ELP2",
        "ELP1",
        (2, 20),
    ),
    "OV9281_to_ELP2": (
        "ELP2",
        "OV9281",
        (3, 30),
    ),
    "OV9281_to_ELP1": (
        "ELP1",
        "OV9281",
        (4, 40),
    ),
}

MIN_SAMPLES = 20


def average_rotations(R_arr):
    R_mean = np.mean(
        np.asarray(R_arr),
        axis=0,
    )

    U, _, Vt = np.linalg.svd(R_mean)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


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


def robust_estimate(samples):
    if len(samples) < MIN_SAMPLES:
        return None

    R_arr = np.asarray(
        [s["R"] for s in samples],
        dtype=np.float64,
    )

    T_arr = np.asarray(
        [s["T"] for s in samples],
        dtype=np.float64,
    )

    score_arr = np.asarray(
        [s["score"] for s in samples],
        dtype=np.float64,
    )

    # 1. Translation-Ausreißer entfernen.
    T0 = np.median(
        T_arr,
        axis=0,
    )

    dist = np.linalg.norm(
        T_arr - T0,
        axis=1,
    )

    if len(T_arr) >= 30:
        q75 = float(
            np.percentile(
                dist,
                75,
            )
        )

        limit = max(
            q75 * 2.5,
            0.02,
        )

        mask = dist <= limit

        R_arr = R_arr[mask]
        T_arr = T_arr[mask]
        score_arr = score_arr[mask]

    if len(T_arr) < MIN_SAMPLES:
        return None

    # 2. Schlechteste Reprojektionen entfernen.
    if len(T_arr) >= 30:
        score_limit = float(
            np.percentile(
                score_arr,
                75,
            )
        )

        mask = (
            score_arr
            <= score_limit
        )

        R_arr = R_arr[mask]
        T_arr = T_arr[mask]
        score_arr = score_arr[mask]

    if len(T_arr) < MIN_SAMPLES:
        return None

    T = np.median(
        T_arr,
        axis=0,
    )

    R = average_rotations(
        R_arr
    )

    std_cm = (
        T_arr.std(axis=0)
        * 100.0
    )

    return {
        "R": R,
        "T": T,
        "samples": int(
            len(T_arr)
        ),
        "std_max_cm": float(
            np.max(std_cm)
        ),
        "reproj_median": float(
            np.median(score_arr)
        ),
    }


def best_marker_candidate(
    monitor,
    target_observations,
    source_observations,
):
    best = None

    for target_obs in target_observations:
        for source_obs in source_observations:

            R, T, score = (
                monitor
                ._relative_from_common_marker(
                    target_obs,
                    source_obs,
                )
            )

            candidate = {
                "R": R,
                "T": T,
                "score": float(score),
                "target_side": (
                    target_obs.get(
                        "side",
                        "?",
                    )
                ),
                "source_side": (
                    source_obs.get(
                        "side",
                        "?",
                    )
                ),
            }

            if (
                best is None
                or candidate["score"]
                < best["score"]
            ):
                best = candidate

    return best


def compose(first, second):
    # second: A -> B
    # first:  B -> C
    # Resultat: A -> C

    R1 = np.asarray(
        first["R"],
        dtype=np.float64,
    )

    T1 = np.asarray(
        first["T"],
        dtype=np.float64,
    ).reshape(3)

    R2 = np.asarray(
        second["R"],
        dtype=np.float64,
    )

    T2 = np.asarray(
        second["T"],
        dtype=np.float64,
    ).reshape(3)

    return {
        "R": R1 @ R2,
        "T": (
            R1 @ T2
            + T1
        ),
    }


def main():
    print()
    print("=" * 78)
    print(
        "ANTHRO3D – "
        "EINZELMARKER-DREIECKSDIAGNOSE"
    )
    print("=" * 78)

    monitor = (
        ArucoMonitor.from_files(
            base="~/anthro3d",
            require_ov=True,
            stereo_T_units="m",
        )
    )

    captures = {}

    samples = defaultdict(
        list
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
                    f"{role}: "
                    "Kamera konnte nicht "
                    "geöffnet werden."
                )

            captures[role] = cap

            print(
                f"{role:<10} offen"
            )

        print()
        print(
            "30 Sekunden Einzelmarker sammeln."
        )
        print(
            "Kameras und Stative NICHT bewegen."
        )
        print()

        start = time.monotonic()

        while (
            time.monotonic()
            - start
            < RUN_SECONDS
        ):
            frames = {}

            good = True

            for role in ROLES:
                ok, frame = (
                    captures[
                        role
                    ].read()
                )

                if (
                    not ok
                    or frame is None
                ):
                    good = False
                    break

                frames[
                    role
                ] = frame

            if not good:
                continue

            detections = (
                monitor
                ._detect_all_available_rigs(
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
            )

            for (
                pair_name,
                (
                    target,
                    source,
                    marker_ids,
                ),
            ) in PAIR_RULES.items():

                if (
                    target not in detections
                    or source not in detections
                ):
                    continue

                for marker in marker_ids:

                    target_list = (
                        detections
                        .get(
                            target,
                            {},
                        )
                        .get(
                            marker,
                            [],
                        )
                    )

                    source_list = (
                        detections
                        .get(
                            source,
                            {},
                        )
                        .get(
                            marker,
                            [],
                        )
                    )

                    if (
                        not target_list
                        or not source_list
                    ):
                        continue

                    best = (
                        best_marker_candidate(
                            monitor,
                            target_list,
                            source_list,
                        )
                    )

                    if best is not None:
                        samples[
                            (
                                pair_name,
                                marker,
                            )
                        ].append(
                            best
                        )

        estimates = {}

        print()
        print("=" * 78)
        print(
            "EINZELMARKER-SCHÄTZUNGEN"
        )
        print("=" * 78)

        for (
            pair_name,
            (
                _target,
                _source,
                marker_ids,
            ),
        ) in PAIR_RULES.items():

            estimates[
                pair_name
            ] = {}

            print()
            print(pair_name)

            for marker in marker_ids:

                raw = samples.get(
                    (
                        pair_name,
                        marker,
                    ),
                    [],
                )

                est = robust_estimate(
                    raw
                )

                if est is None:
                    print(
                        f"  ID{marker}: "
                        f"NICHT STABIL "
                        f"(raw={len(raw)})"
                    )
                    continue

                estimates[
                    pair_name
                ][marker] = est

                distance_cm = float(
                    np.linalg.norm(
                        est["T"]
                    )
                    * 100.0
                )

                print(
                    f"  ID{marker:<2} | "
                    f"Dist="
                    f"{distance_cm:7.2f} cm | "
                    f"n="
                    f"{est['samples']:<4} | "
                    f"Std="
                    f"{est['std_max_cm']:.2f} cm | "
                    f"Reproj="
                    f"{est['reproj_median']:.2f}"
                )

        required = (
            "ELP1_to_ELP2",
            "OV9281_to_ELP2",
            "OV9281_to_ELP1",
        )

        if any(
            not estimates.get(name)
            for name in required
        ):
            raise RuntimeError(
                "Mindestens eine Graph-Kante "
                "hat keine stabile "
                "Einzelmarker-Schätzung."
            )

        print()
        print("=" * 78)
        print(
            "ALLE DREIECKS-KOMBINATIONEN"
        )
        print("=" * 78)

        rows = []

        for (
            marker12,
            marker02,
            marker01,
        ) in itertools.product(
            sorted(
                estimates[
                    "ELP1_to_ELP2"
                ]
            ),
            sorted(
                estimates[
                    "OV9281_to_ELP2"
                ]
            ),
            sorted(
                estimates[
                    "OV9281_to_ELP1"
                ]
            ),
        ):

            e12 = estimates[
                "ELP1_to_ELP2"
            ][marker12]

            direct = estimates[
                "OV9281_to_ELP2"
            ][marker02]

            e01 = estimates[
                "OV9281_to_ELP1"
            ][marker01]

            via = compose(
                e12,
                e01,
            )

            t_error_cm = float(
                np.linalg.norm(
                    np.asarray(
                        direct["T"]
                    )
                    - np.asarray(
                        via["T"]
                    )
                )
                * 100.0
            )

            r_error_deg = (
                rotation_diff_deg(
                    direct["R"],
                    via["R"],
                )
            )

            # Nur für Sortierung.
            score = (
                t_error_cm
                + 2.0
                * r_error_deg
            )

            rows.append(
                {
                    "m12": marker12,
                    "m02": marker02,
                    "m01": marker01,
                    "t": t_error_cm,
                    "r": r_error_deg,
                    "score": score,
                }
            )

        rows.sort(
            key=lambda row: (
                row["score"]
            )
        )

        for i, row in enumerate(
            rows,
            start=1,
        ):
            prefix = (
                "BESTE"
                if i == 1
                else f"{i}."
            )

            print(
                f"{prefix:<6} "
                f"ELP1→ELP2 ID{row['m12']:<2} | "
                f"OV→ELP2 ID{row['m02']:<2} | "
                f"OV→ELP1 ID{row['m01']:<2} | "
                f"dT={row['t']:6.2f} cm | "
                f"dR={row['r']:5.2f}° | "
                f"Score={row['score']:6.2f}"
            )

        best = rows[0]

        print()
        print("=" * 78)
        print("BESTE KOMBINATION")
        print("=" * 78)

        print(
            f"ELP1→ELP2 : ID{best['m12']}"
        )

        print(
            f"OV→ELP2   : ID{best['m02']}"
        )

        print(
            f"OV→ELP1   : ID{best['m01']}"
        )

        print(
            f"Translation closure: "
            f"{best['t']:.2f} cm"
        )

        print(
            f"Rotation closure:    "
            f"{best['r']:.2f}°"
        )

        print()
        print(
            "Nichts wurde gespeichert."
        )

    finally:
        print()
        print("Kameras freigeben ...")

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
                    f"  WARNUNG "
                    f"{role}: {exc}"
                )


if __name__ == "__main__":
    main()
