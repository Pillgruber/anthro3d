from __future__ import annotations

import time

from aruco_monitor import ArucoMonitor
from marker_integrity_monitor import MarkerIntegrityMonitor
from camera_system.scan_capture import AnthroCameraCapture


ROLES = [
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
]

RUN_SECONDS = 30.0
PRINT_INTERVAL = 2.0


def main():
    print()
    print("=" * 78)
    print("ANTHRO3D – REDUNDANTE MARKER-INTEGRITÄT")
    print("=" * 78)

    aruco = ArucoMonitor.from_files(
        base="~/anthro3d",
        require_ov=True,
        stereo_T_units="m",
    )

    integrity = MarkerIntegrityMonitor(
        base="~/anthro3d",
    )

    captures = {}

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
            print(f"{role:<10} offen")

        print()
        print("30 Sekunden Test.")
        print("Noch NICHTS bewegen.")
        print()

        start = time.monotonic()
        last_print = 0.0

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

            now_wall = time.time()

            aruco.update(
                frame_elp2_bgr=frames["ELP2"],
                frame_elp1_bgr=frames["ELP1"],
                frame_ov_l_bgr=frames["OV9281_L"],
                frame_ov_r_bgr=frames["OV9281_R"],
                now=now_wall,
            )

            status = integrity.update(
                aruco.last_detections,
                now=now_wall,
            )

            now = time.monotonic()

            if (
                now - last_print
                < PRINT_INTERVAL
            ):
                continue

            last_print = now

            print("-" * 78)

            for stand, item in (
                status
                .get(
                    "stand_status",
                    {},
                )
                .items()
            ):
                print(
                    f"{stand:<14} "
                    f"{item['level'].upper()}"
                )

                observers = (
                    status[
                        "observer_status"
                    ][stand]
                )

                for observer, obs in observers.items():
                    if not obs.get(
                        "ready",
                        False,
                    ):
                        print(
                            f"  {observer:<8} "
                            f"warming up "
                            f"({obs.get('samples', 0)} Samples)"
                        )
                        continue

                    print(
                        f"  {observer:<8} "
                        f"{obs['level'].upper():<5} | "
                        f"ΔDist="
                        f"{obs['distance_delta_cm']:.3f} cm | "
                        f"ΔRot="
                        f"{obs['rotation_delta_deg']:.3f}° | "
                        f"n={obs['samples']}"
                    )

            print(
                "AUTO-RECALIBRATION SAFE:",
                status.get(
                    "safe_for_auto_recalibration"
                ),
            )

            print(
                "MARKER FAULT:",
                status.get(
                    "marker_fault_detected"
                ),
            )

        print()
        print("=" * 78)
        print("FINAL")
        print("=" * 78)

        status = integrity.get_status()

        for stand, item in status[
            "stand_status"
        ].items():
            print(
                f"{stand:<14} "
                f"{item['level'].upper()}"
            )

        print()
        print(
            "safe_for_auto_recalibration =",
            status[
                "safe_for_auto_recalibration"
            ],
        )

        print(
            "marker_fault_detected =",
            status[
                "marker_fault_detected"
            ],
        )

    finally:
        print()
        print("Kameras freigeben ...")

        for role in reversed(ROLES):
            cap = captures.get(role)

            if cap is None:
                continue

            cap.release()
            print(
                f"  {role}: frei"
            )


if __name__ == "__main__":
    main()
