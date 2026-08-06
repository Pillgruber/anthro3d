import cv2
import time


TESTS = [
    {
        "name": "ELP2",
        "index": 0,
        "modes": [
            (2560, 720),
            (1280, 480),
        ],
    },
    {
        "name": "ELP1",
        "index": 3,
        "modes": [
            (3200, 1200),
            (1280, 480),
        ],
    },
]


def test_mode(name, index, width, height):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"{name}: Kamera Index {index} konnte nicht geöffnet werden.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    time.sleep(0.5)

    actual_property_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_property_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = None

    for _ in range(10):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate

    cap.release()

    print()
    print(f"{name} – angefordert: {width} × {height}")
    print(
        f"OpenCV-Eigenschaften: "
        f"{actual_property_width} × {actual_property_height}"
    )

    if frame is None:
        print("Kein Bild empfangen.")
        return

    frame_height, frame_width = frame.shape[:2]
    print(f"Tatsächliches Frame: {frame_width} × {frame_height}")

    if frame_width % 2 == 0:
        print(
            f"Pro Stereo-Hälfte: "
            f"{frame_width // 2} × {frame_height}"
        )


for device in TESTS:
    for mode_width, mode_height in device["modes"]:
        test_mode(
            device["name"],
            device["index"],
            mode_width,
            mode_height,
        )
