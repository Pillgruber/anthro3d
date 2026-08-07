from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from camera_system.scan_capture import (
    AnthroCameraCapture,
    get_camera_profile,
)


WINDOW = "ANTHRO3D - Kameras & Ausrichtung"

ROLES = [
    "ELP2",
    "ELP1",
    "OV9281_L",
    "OV9281_R",
]

DISPLAY_NAMES = {
    "ELP2": "ELP2",
    "ELP1": "ELP1",
    "OV9281_L": "OV9281 LINKS",
    "OV9281_R": "OV9281 RECHTS",
}

# Erwartete Stativmarker entsprechend ANTHRO3D-Multimarker-System.
# Bei den beiden OV-Kameras ist nicht zwingend jeder Marker gleichzeitig
# in jeder Einzelkamera sichtbar.
EXPECTED_MARKERS = {
    "ELP2": {2, 3, 20, 30},
    "ELP1": {2, 4, 20, 40},
    "OV9281_L": {3, 4, 30, 40},
    "OV9281_R": {3, 4, 30, 40},
}

CELL_W = 760
CELL_H = 350
HEADER_H = 74

selected_role = None
window_fullscreen = False


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_ARUCO_ORIGINAL
    )

    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_APRILTAG
        )
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(
            dictionary,
            params,
        ), dictionary, params

    return None, dictionary, params


ARUCO_DETECTOR, ARUCO_DICT, ARUCO_PARAMS = make_aruco_detector()


def detect_and_draw_aruco(image):
    display = image.copy()

    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    if ARUCO_DETECTOR is not None:
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            ARUCO_DICT,
            parameters=ARUCO_PARAMS,
        )

    detected = []

    if ids is not None and len(ids):
        ids_flat = ids.reshape(-1)

        for marker_id in ids_flat:
            detected.append(int(marker_id))

        cv2.aruco.drawDetectedMarkers(
            display,
            corners,
            ids,
            borderColor=(0, 255, 255),
        )

    return display, sorted(set(detected))


def letterbox(frame, width, height):
    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = min(width / w, height / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def status_color(ok):
    return (80, 220, 80) if ok else (50, 70, 255)


def make_panel(role, frame, profile, fps_value):
    panel = np.zeros(
        (HEADER_H + CELL_H, CELL_W, 3),
        dtype=np.uint8,
    )

    panel[:] = (18, 18, 18)

    if frame is None:
        cv2.putText(
            panel,
            "KEIN BILD",
            (35, HEADER_H + CELL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (40, 40, 255),
            3,
            cv2.LINE_AA,
        )
        return panel, []

    preview = letterbox(
        frame,
        CELL_W,
        CELL_H,
    )

    preview, ids = detect_and_draw_aruco(preview)

    panel[
        HEADER_H:HEADER_H + CELL_H,
        0:CELL_W,
    ] = preview

    name = DISPLAY_NAMES[role]

    cv2.putText(
        panel,
        name,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.73,
        (80, 230, 150),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"{profile.width}x{profile.height} @ {profile.fps:g} fps"
        f" | live {fps_value:.1f}",
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    short_uid = profile.unique_id

    cv2.putText(
        panel,
        f"uniqueID: {short_uid}",
        (260, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (145, 145, 145),
        1,
        cv2.LINE_AA,
    )

    ids_text = ",".join(str(x) for x in ids) if ids else "-"

    expected = EXPECTED_MARKERS[role]
    expected_visible = sorted(expected.intersection(ids))

    cv2.putText(
        panel,
        f"ArUco erkannt: {ids_text}",
        (260, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color(bool(expected_visible)),
        1,
        cv2.LINE_AA,
    )

    return panel, ids


def make_grid(panels):
    top = np.hstack([
        panels["ELP2"],
        panels["ELP1"],
    ])

    bottom = np.hstack([
        panels["OV9281_L"],
        panels["OV9281_R"],
    ])

    body = np.vstack([top, bottom])

    footer = np.zeros(
        (52, body.shape[1], 3),
        dtype=np.uint8,
    )
    footer[:] = (12, 12, 12)

    cv2.putText(
        footer,
        "Doppelklick = Kamera gross | "
        "1=ELP2  2=ELP1  3=OV-L  4=OV-R  "
        "0=Uebersicht  F=Vollbild  S=Snapshot  Q=Ende",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([body, footer])


def make_single_panel(role, frame, profile, ids, fps_value):
    screen_w = 1500
    screen_h = 850
    header_h = 90

    canvas = np.zeros(
        (screen_h, screen_w, 3),
        dtype=np.uint8,
    )
    canvas[:] = (15, 15, 15)

    if frame is not None:
        view = letterbox(
            frame,
            screen_w,
            screen_h - header_h,
        )

        view, ids = detect_and_draw_aruco(view)

        canvas[
            header_h:screen_h,
            0:screen_w,
        ] = view

    cv2.putText(
        canvas,
        f"{DISPLAY_NAMES[role]} - AUSRICHTUNG",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 230, 150),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"{profile.width}x{profile.height} @ "
        f"{profile.fps:g} fps | live {fps_value:.1f} fps",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    ids_text = ",".join(str(x) for x in ids) if ids else "-"

    cv2.putText(
        canvas,
        f"ArUco IDs: {ids_text}",
        (650, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "0=Uebersicht | F=Vollbild | S=Snapshot | Q=Ende",
        (650, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    return canvas


def mouse_callback(event, x, y, flags, param):
    global selected_role

    if event != cv2.EVENT_LBUTTONDBLCLK:
        return

    if selected_role is not None:
        selected_role = None
        return

    if y >= (HEADER_H + CELL_H) * 2:
        return

    col = 0 if x < CELL_W else 1
    row = 0 if y < HEADER_H + CELL_H else 1

    lookup = {
        (0, 0): "ELP2",
        (0, 1): "ELP1",
        (1, 0): "OV9281_L",
        (1, 1): "OV9281_R",
    }

    selected_role = lookup.get((row, col))


def save_snapshot(image):
    outdir = (
        Path.home()
        / "anthro3d"
        / "alignment_snapshots"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    path = outdir / f"camera_alignment_{timestamp}.png"

    if cv2.imwrite(str(path), image):
        print(f"Snapshot gespeichert: {path}")
    else:
        print("WARNUNG: Snapshot konnte nicht gespeichert werden.")


def main():
    global selected_role
    global window_fullscreen

    print()
    print("=" * 72)
    print("ANTHRO3D - Kameras & Ausrichtung")
    print("Automatische Kamerarollen ueber AVFoundation uniqueID")
    print("=" * 72)

    profiles = {}
    captures = {}

    for role in ROLES:
        profile = get_camera_profile(role)

        if not profile.confirmed:
            raise RuntimeError(
                f"{role}: Kamera-Rolle ist in der Registry "
                "nicht bestaetigt."
            )

        profiles[role] = profile

        print(
            f"{role:<10} "
            f"uniqueID={profile.unique_id} | "
            f"{profile.width}x{profile.height} "
            f"@ {profile.fps:g} fps"
        )

    unique_ids = [
        profiles[role].unique_id
        for role in ROLES
    ]

    if len(unique_ids) != len(set(unique_ids)):
        raise RuntimeError(
            "Mindestens zwei Rollen verwenden dieselbe uniqueID."
        )

    last_time = {
        role: time.monotonic()
        for role in ROLES
    }

    fps_smooth = {
        role: 0.0
        for role in ROLES
    }

    last_frames = {
        role: None
        for role in ROLES
    }

    last_ids = {
        role: []
        for role in ROLES
    }

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW,
        CELL_W * 2,
        (HEADER_H + CELL_H) * 2 + 52,
    )

    cv2.setMouseCallback(
        WINDOW,
        mouse_callback,
    )

    try:
        print()
        print("Oeffne Kameras ...")

        for role in ROLES:
            cap = AnthroCameraCapture(
                role,
                read_timeout=5.0,
            )

            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"{role}: Kamera konnte nicht geoeffnet werden."
                )

            captures[role] = cap
            print(f"  {role}: offen")

        print()
        print("Alle vier Kameras offen.")
        print("Q beendet.")
        print()

        while True:
            panels = {}

            for role in ROLES:
                cap = captures[role]

                ok, frame = cap.read()

                if ok and frame is not None:
                    expected_shape = (
                        profiles[role].height,
                        profiles[role].width,
                    )

                    actual_shape = frame.shape[:2]

                    if actual_shape != expected_shape:
                        print(
                            f"WARNUNG {role}: Frame "
                            f"{actual_shape[::-1]} statt "
                            f"{expected_shape[::-1]}"
                        )
                    else:
                        last_frames[role] = frame

                        now = time.monotonic()
                        dt = now - last_time[role]
                        last_time[role] = now

                        if dt > 0:
                            instant = 1.0 / dt

                            if fps_smooth[role] == 0:
                                fps_smooth[role] = instant
                            else:
                                fps_smooth[role] = (
                                    0.90 * fps_smooth[role]
                                    + 0.10 * instant
                                )

                panel, ids = make_panel(
                    role,
                    last_frames[role],
                    profiles[role],
                    fps_smooth[role],
                )

                panels[role] = panel
                last_ids[role] = ids

            if selected_role is None:
                output = make_grid(panels)
            else:
                output = make_single_panel(
                    selected_role,
                    last_frames[selected_role],
                    profiles[selected_role],
                    last_ids[selected_role],
                    fps_smooth[selected_role],
                )

            cv2.imshow(
                WINDOW,
                output,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("0"):
                selected_role = None

            elif key == ord("1"):
                selected_role = "ELP2"

            elif key == ord("2"):
                selected_role = "ELP1"

            elif key == ord("3"):
                selected_role = "OV9281_L"

            elif key == ord("4"):
                selected_role = "OV9281_R"

            elif key in (ord("f"), ord("F")):
                window_fullscreen = not window_fullscreen

                cv2.setWindowProperty(
                    WINDOW,
                    cv2.WND_PROP_FULLSCREEN,
                    (
                        cv2.WINDOW_FULLSCREEN
                        if window_fullscreen
                        else cv2.WINDOW_NORMAL
                    ),
                )

            elif key in (ord("s"), ord("S")):
                save_snapshot(output)

    finally:
        print()
        print("Kameras freigeben ...")

        for role in reversed(ROLES):
            cap = captures.get(role)

            if cap is not None:
                try:
                    cap.release()
                    print(f"  {role}: freigegeben")
                except Exception as exc:
                    print(
                        f"  WARNUNG {role}: {exc}"
                    )

        cv2.destroyAllWindows()

    print("Ausrichtungsansicht beendet.")


if __name__ == "__main__":
    main()
