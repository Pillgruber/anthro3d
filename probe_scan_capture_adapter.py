from __future__ import annotations

import os
from pathlib import Path

import cv2

from camera_system.scan_capture import AnthroCameraCapture


OUTDIR = Path(os.environ["ANTHRO3D_PHASE6_OUTDIR"])
CAMERAS = ["ELP1", "ELP2", "OV9281_L", "OV9281_R"]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    captures = []

    print("=" * 88)
    print("PHASE 6 – scan3d-kompatibler Capture-Adapter")
    print("=" * 88)

    try:
        for name in CAMERAS:
            print()
            print(f"Öffne {name} ...")

            cap = AnthroCameraCapture(
                name,
                read_timeout=10.0,
            )
            captures.append((name, cap))

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"  uniqueID={cap.unique_id}")
            print(
                f"  Profil={width}x{height} @ {fps:g} fps"
            )
            print(f"  isOpened={cap.isOpened()}")

            if not cap.isOpened():
                raise RuntimeError(f"{name}: nicht geöffnet.")

        print()
        print("=== Alle vier Adapter gleichzeitig offen ===")

        for name, cap in captures:
            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError(
                    f"{name}: read() fehlgeschlagen."
                )

            expected_width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            expected_height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            expected_shape = (
                expected_height,
                expected_width,
                3,
            )

            if frame.shape != expected_shape:
                raise RuntimeError(
                    f"{name}: {frame.shape} != {expected_shape}"
                )

            path = OUTDIR / (
                f"{name}_{cap.unique_id}_"
                f"{expected_width}x{expected_height}.png"
            )

            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(
                    f"{name}: Speichern fehlgeschlagen."
                )

            print(
                f"  {name:<10} "
                f"{frame.shape} | {cap.unique_id}"
            )

        print()
        print("=" * 88)
        print("PHASE-6-ADAPTERTEST ERFOLGREICH")
        print(
            "read(), release(), isOpened() und get() "
            "sind für scan3d.py verfügbar."
        )
        print("=" * 88)

    finally:
        for name, cap in reversed(captures):
            try:
                cap.release()
                print(f"freigegeben: {name}")
            except Exception as exc:
                print(f"WARNUNG release {name}: {exc}")


if __name__ == "__main__":
    main()
