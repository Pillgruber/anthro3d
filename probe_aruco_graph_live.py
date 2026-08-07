from __future__ import annotations

import time

from aruco_monitor import ArucoMonitor
from camera_system.scan_capture import AnthroCameraCapture


ROLES = [
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
]

PRINT_INTERVAL_S = 1.0
RUN_SECONDS = 30.0


def marker_ids(status, rig_name):
    detections = status.get("detections", {}).get(rig_name, {})

    if isinstance(detections, dict):
        ids = []
        for key, value in detections.items():
            try:
                marker_id = int(key)
            except Exception:
                continue

            # Ein Marker gilt hier als erkannt, wenn der Status
            # mindestens eine Beobachtung enthält.
            if value:
                ids.append(marker_id)

        return sorted(ids)

    return []


def fmt_estimate(item):
    if not item:
        return "FEHLT"

    if not item.get("ok", False):
        return (
            f"nicht stabil | "
            f"Samples={item.get('samples', 0)} | "
            f"Grund={item.get('source', '?')}"
        )

    t = item.get("T_cm")
    distance = item.get("distance_cm")
    std = item.get("std_max_cm")

    return (
        f"OK | "
        f"Samples={item.get('samples', 0)} | "
        f"T={t} cm | "
        f"Dist={distance:.1f} cm | "
        f"Std={std:.2f} cm"
    )


def main():
    print()
    print("=" * 78)
    print("ANTHRO3D – LIVE ARUCO CAMERA GRAPH")
    print("uniqueID Capture -> ArucoMonitor")
    print("=" * 78)

    monitor = ArucoMonitor.from_files(
        base="~/anthro3d",
        require_ov=True,
        stereo_T_units="m",
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
            print(f"{role:<10} geöffnet")

        print()
        print("Sammle 30 Sekunden Live-ArUco-Daten ...")
        print("Kameras und Stative währenddessen NICHT bewegen.")
        print()

        start = time.monotonic()
        last_print = 0.0
        frame_count = 0

        while True:
            elapsed = time.monotonic() - start

            if elapsed >= RUN_SECONDS:
                break

            frames = {}

            all_ok = True

            for role in ROLES:
                ok, frame = captures[role].read()

                if not ok or frame is None:
                    all_ok = False
                    print(f"WARNUNG: kein Frame von {role}")
                    break

                frames[role] = frame

            if not all_ok:
                continue

            status = monitor.update(
                frame_elp2_bgr=frames["ELP2"],
                frame_elp1_bgr=frames["ELP1"],
                frame_ov_l_bgr=frames["OV9281_L"],
                frame_ov_r_bgr=frames["OV9281_R"],
            )

            frame_count += 1

            now = time.monotonic()

            if now - last_print < PRINT_INTERVAL_S:
                continue

            last_print = now

            print("-" * 78)
            print(
                f"Zeit {elapsed:5.1f}s | "
                f"Monitor-Updates: {frame_count}"
            )

            print(
                "Marker: "
                f"ELP2={marker_ids(status, 'ELP2')} | "
                f"ELP1={marker_ids(status, 'ELP1')} | "
                f"OV9281={marker_ids(status, 'OV9281')}"
            )

            estimates = status.get(
                "live_estimates",
                {},
            )

            for pair_name in (
                "ELP1_to_ELP2",
                "OV9281_to_ELP2",
                "OV9281_to_ELP1",
            ):
                print(
                    f"{pair_name:<20} "
                    f"{fmt_estimate(estimates.get(pair_name))}"
                )

            gate = status.get("scan_gate", {})

            if gate:
                print(
                    "Scan-Gate: "
                    f"allow_scan={gate.get('allow_scan')}"
                )

                for message in gate.get("messages", []):
                    print(f"  {message}")

        print()
        print("=" * 78)
        print("ENDE LIVE-TEST")
        print("=" * 78)

        final = monitor.get_status()

        print()
        print("Erkannte Marker am Ende:")
        print(
            f"  ELP2:    {marker_ids(final, 'ELP2')}"
        )
        print(
            f"  ELP1:    {marker_ids(final, 'ELP1')}"
        )
        print(
            f"  OV9281:  {marker_ids(final, 'OV9281')}"
        )

        print()
        print("Live-Kanten:")

        estimates = final.get("live_estimates", {})

        for pair_name in (
            "ELP1_to_ELP2",
            "OV9281_to_ELP2",
            "OV9281_to_ELP1",
        ):
            print(
                f"  {pair_name:<20} "
                f"{fmt_estimate(estimates.get(pair_name))}"
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
                print(f"  {role}: freigegeben")
            except Exception as exc:
                print(
                    f"  WARNUNG {role}: {exc}"
                )


if __name__ == "__main__":
    main()
