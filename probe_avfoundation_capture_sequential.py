from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from camera_system.backends.avfoundation import (
    AVFoundationCapture,
)


OUTDIR = Path(os.environ["ANTHRO3D_PHASE4_OUTDIR"])

CAMERAS = [
    {
        "camera_id": "ELP1",
        "unique_id": "0x110000032e42b10",
        "width": 3200,
        "height": 1200,
        "fps": 30.0,
        "confirmed": True,
    },
    {
        "camera_id": "ELP2",
        "unique_id": "0x210000032e42b10",
        "width": 2560,
        "height": 720,
        "fps": 30.0,
        "confirmed": True,
    },
    {
        "camera_id": "OV9281_R_vorlaeufig",
        "unique_id": "0x1300000c45636d",
        "width": 1280,
        "height": 800,
        "fps": 30.0,
        "confirmed": False,
    },
    {
        "camera_id": "OV9281_L_vorlaeufig",
        "unique_id": "0x1410000c45636d",
        "width": 1280,
        "height": 800,
        "fps": 30.0,
        "confirmed": False,
    },
]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("AVFoundationCapture – sequenzieller Einzelkameratest")
    print("=" * 88)

    for index, item in enumerate(CAMERAS, start=1):
        print()
        print("-" * 88)
        print(
            f"{index}/4  {item['camera_id']} | "
            f"{item['unique_id']} | "
            f"{item['width']}x{item['height']} @ "
            f"{item['fps']:g} fps"
        )
        print("-" * 88)

        capture = AVFoundationCapture(
            unique_id=item["unique_id"],
            width=item["width"],
            height=item["height"],
            fps=item["fps"],
        )

        try:
            capture.open()

            print(f"Gerät: {capture.device_name}")
            print(
                "Gewähltes Format: "
                f"{capture.selected_dimensions}"
            )

            ok, frame = capture.read(timeout=10.0)

            if not ok or frame is None:
                raise RuntimeError(
                    "Kein NumPy/OpenCV-Frame innerhalb von "
                    f"10 s. last_error={capture.last_error}"
                )

            expected_shape = (
                item["height"],
                item["width"],
                3,
            )

            if frame.shape != expected_shape:
                raise RuntimeError(
                    f"Falsche ndarray-Form: "
                    f"{frame.shape} != {expected_shape}"
                )

            if frame.dtype.name != "uint8":
                raise RuntimeError(
                    f"Falscher dtype: {frame.dtype}"
                )

            output_path = OUTDIR / (
                f"{index:02d}_"
                f"{item['camera_id']}_"
                f"{item['unique_id']}_"
                f"{item['width']}x{item['height']}.png"
            )

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(
                    f"cv2.imwrite fehlgeschlagen: "
                    f"{output_path}"
                )

            print(
                "ERFOLG: "
                f"shape={frame.shape} | "
                f"dtype={frame.dtype}"
            )
            print(f"Gespeichert: {output_path}")

        finally:
            capture.release()

        time.sleep(0.5)

    print()
    print("=" * 88)
    print("PHASE-4-HARDWARETEST ERFOLGREICH")
    print("Alle vier Kameras lieferten über uniqueID ein BGR-NumPy-Frame.")
    print("=" * 88)


if __name__ == "__main__":
    main()
