from __future__ import annotations

from collections import defaultdict
import itertools
import time

import numpy as np

from aruco_monitor import ArucoMonitor
from camera_system.scan_capture import AnthroCameraCapture


RUN_SECONDS = 30.0
MIN_SAMPLES = 20

ROLES = (
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
)

PAIR_RULES = {
    "ELP1_to_ELP2": {
        "target": "ELP2",
        "source": "ELP1",
        "markers": (2, 20),
    },

    "OV9281_to_ELP2": {
        "target": "ELP2",
        "source": "OV9281",
        "markers": (3, 30),
    },

    "OV9281_to_ELP1": {
        "target": "ELP1",
        "source": "OV9281",
        "markers": (4, 40),
    },
}


def average_rotations(rotations):
    M = np.mean(
        np.asarray(rotations, dtype=np.float64),
        axis=0,
    )

    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotation_diff_deg(R_a, R_b):
    Rd = (
        np.asarray(R_a, dtype=np.float64)
        @ np.asarray(R_b, dtype=np.float64).T
    )

    x = np.clip(
        (np.trace(Rd) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(np.arccos(x))
    )


def robust_estimate(samples):
    if len(samples) < MIN_SAMPLES:
        return None

    R_arr = np.asarray(
        [x["R"] for x in samples],
        dtype=np.float64,
    )

    T_arr = np.asarray(
        [x["T"] for x in samples],
        dtype=np.float64,
    )

    score_arr = np.asarray(
        [x["score"] for x in samples],
        dtype=np.float64,
    )

    T0 = np.median(
        T_arr,
        axis=0,
    )

    d = np.linalg.norm(
        T_arr - T0,
        axis=1,
    )

    if len(T_arr) >= 30:
        limit = max(
            float(np.percentile(d, 75)) * 2.5,
            0.02,
        )

        mask = d <= limit

        R_arr = R_arr[mask]
        T_arr = T_arr[mask]
        score_arr = score_arr[mask]

    if len(T_arr) < MIN_SAMPLES:
        return None

    if len(T_arr) >= 30:
        limit = float(
            np.percentile(score_arr, 75)
        )

        mask = score_arr <= limit

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
        "samples": len(T_arr),
        "std_max_cm": float(
            np.max(std_cm)
        ),
        "reproj": float(
            np.median(score_arr)
        ),
    }


def compose(first, second):
    # second = A -> B
    # first  = B -> C
    # result = A -> C

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
        "T": R1 @ T2 + T1,
    }


def best_obs(obs_list, side):
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
    print("=" * 80)
    print("ANTHRO3D – DREIECK NACH FESTER RIG-SEITE")
    print("=" * 80)

    monitor = ArucoMonitor.from_files(
        base="~/anthro3d",
        require_ov=True,
        stereo_T_units="m",
    )

    captures = {}

    # key:
    # pair, marker, target_side, source_side
    samples = defaultdict(list)

    try:
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
                f"{role:<10} offen"
            )

        print()
        print(
            "30 Sekunden sammeln – "
            "Kameras/Stative NICHT bewegen."
        )
        print()

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

                if not ok or frame is None:
                    valid = False
                    break

                frames[role] = frame

            if not valid:
                continue

            detections = (
                monitor
                ._detect_all_available_rigs(
                    frame_elp2_bgr=frames["ELP2"],
                    frame_elp1_bgr=frames["ELP1"],
                    frame_ov_l_bgr=frames["OV9281_L"],
                    frame_ov_r_bgr=frames["OV9281_R"],
                )
            )

            for pair_name, rule in PAIR_RULES.items():

                target = rule["target"]
                source = rule["source"]

                if (
                    target not in detections
                    or source not in detections
                ):
                    continue

                for marker in rule["markers"]:

                    t_list = (
                        detections[target]
                        .get(marker, [])
                    )

                    s_list = (
                        detections[source]
                        .get(marker, [])
                    )

                    for target_side in ("L", "R"):
                        target_obs = best_obs(
                            t_list,
                            target_side,
                        )

                        if target_obs is None:
                            continue

                        for source_side in ("L", "R"):
                            source_obs = best_obs(
                                s_list,
                                source_side,
                            )

                            if source_obs is None:
                                continue

                            R, T, score = (
                                monitor
                                ._relative_from_common_marker(
                                    target_obs,
                                    source_obs,
                                )
                            )

                            samples[
                                (
                                    pair_name,
                                    marker,
                                    target_side,
                                    source_side,
                                )
                            ].append(
                                {
                                    "R": R,
                                    "T": T,
                                    "score": score,
                                }
                            )

        estimates = {}

        for key, raw in samples.items():
            est = robust_estimate(raw)

            if est is not None:
                estimates[key] = est

        print()
        print("=" * 80)
        print("GLOBALE RIG-SIDE-KOMBINATIONEN")
        print("=" * 80)
        print()
        print(
            "Jede Zeile benutzt für ein Rig "
            "über ALLE Kanten dieselbe optische Seite."
        )
        print()

        results = []

        for (
            elp2_side,
            elp1_side,
            ov_side,
        ) in itertools.product(
            ("L", "R"),
            repeat=3,
        ):

            rig_side = {
                "ELP2": elp2_side,
                "ELP1": elp1_side,
                "OV9281": ov_side,
            }

            best_for_assignment = None

            for (
                m12,
                m02,
                m01,
            ) in itertools.product(
                (2, 20),
                (3, 30),
                (4, 40),
            ):

                keys = {
                    "ELP1_to_ELP2": (
                        "ELP1_to_ELP2",
                        m12,
                        rig_side["ELP2"],
                        rig_side["ELP1"],
                    ),

                    "OV9281_to_ELP2": (
                        "OV9281_to_ELP2",
                        m02,
                        rig_side["ELP2"],
                        rig_side["OV9281"],
                    ),

                    "OV9281_to_ELP1": (
                        "OV9281_to_ELP1",
                        m01,
                        rig_side["ELP1"],
                        rig_side["OV9281"],
                    ),
                }

                if any(
                    key not in estimates
                    for key in keys.values()
                ):
                    continue

                e12 = estimates[
                    keys["ELP1_to_ELP2"]
                ]

                direct = estimates[
                    keys["OV9281_to_ELP2"]
                ]

                e01 = estimates[
                    keys["OV9281_to_ELP1"]
                ]

                via = compose(
                    e12,
                    e01,
                )

                t_error = float(
                    np.linalg.norm(
                        direct["T"]
                        - via["T"]
                    )
                    * 100.0
                )

                r_error = (
                    rotation_diff_deg(
                        direct["R"],
                        via["R"],
                    )
                )

                score = (
                    t_error
                    + 2.0 * r_error
                )

                row = {
                    "elp2": elp2_side,
                    "elp1": elp1_side,
                    "ov": ov_side,
                    "m12": m12,
                    "m02": m02,
                    "m01": m01,
                    "t": t_error,
                    "r": r_error,
                    "score": score,
                }

                if (
                    best_for_assignment is None
                    or row["score"]
                    < best_for_assignment["score"]
                ):
                    best_for_assignment = row

            if best_for_assignment is not None:
                results.append(
                    best_for_assignment
                )

        results.sort(
            key=lambda row: row["score"]
        )

        for index, row in enumerate(
            results,
            start=1,
        ):
            prefix = (
                "BESTE"
                if index == 1
                else f"{index}."
            )

            print(
                f"{prefix:<6} "
                f"ELP2={row['elp2']} "
                f"ELP1={row['elp1']} "
                f"OV={row['ov']} | "
                f"Marker "
                f"{row['m12']}/"
                f"{row['m02']}/"
                f"{row['m01']} | "
                f"dT={row['t']:6.2f} cm | "
                f"dR={row['r']:5.2f}° | "
                f"Score={row['score']:6.2f}"
            )

        print()
        print("=" * 80)
        print("WICHTIGE SONDERFÄLLE")
        print("=" * 80)

        for desired in (
            ("L", "L", "L"),
            ("R", "R", "R"),
        ):
            row = next(
                (
                    r
                    for r in results
                    if (
                        r["elp2"],
                        r["elp1"],
                        r["ov"],
                    ) == desired
                ),
                None,
            )

            if row is None:
                print(
                    f"{desired}: keine gültige Messung"
                )
                continue

            print(
                f"ELP2={desired[0]} "
                f"ELP1={desired[1]} "
                f"OV={desired[2]} → "
                f"dT={row['t']:.2f} cm | "
                f"dR={row['r']:.2f}° | "
                f"Marker "
                f"{row['m12']}/"
                f"{row['m02']}/"
                f"{row['m01']}"
            )

        print()
        print(
            "Nichts gespeichert oder verändert."
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
                    f"  WARNUNG {role}: {exc}"
                )


if __name__ == "__main__":
    main()
