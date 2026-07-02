import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"

LEFT_INDEX = 1
RIGHT_INDEX = 2
WIDTH = 1280
HEIGHT = 800

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015

MIN_CORNERS_CAPTURE = 8
MIN_CORNERS_STEREO = 6
MIN_PAIRS = 50
MIN_STEREO_PAIRS = 18
AUTO_SAVE_INTERVAL_SEC = 0.5
MIN_CENTER_MOVE_PX = 120.0
MIN_AREA_CHANGE_RATIO = 0.25

OUT_DIR = BASE / "calib_ov9281_charuco_auto_gated"
OUT_DIR.mkdir(exist_ok=True)

def make_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE_M,
        MARKER_SIZE_M,
        aruco_dict,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, aruco_dict

def detector_params():
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params

def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap

def detect_charuco(gray, board, aruco_dict, params):
    marker_corners = None
    marker_ids = None
    try:
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        if ch_corners is not None and ch_ids is not None:
            return ch_corners, ch_ids, int(len(ch_ids)), marker_corners, marker_ids
    except Exception:
        pass
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, 0, marker_corners, marker_ids
    try:
        ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
    except Exception:
        return None, None, 0, marker_corners, marker_ids
    if ch_corners is None or ch_ids is None:
        return None, None, 0, marker_corners, marker_ids
    return ch_corners, ch_ids, int(len(ch_ids)), marker_corners, marker_ids

def draw_board_frame(vis, ch_corners, marker_corners, ok_color=(0, 255, 0)):
    pts = []
    if ch_corners is not None and len(ch_corners) > 0:
        pts.extend(ch_corners.reshape(-1, 2))
    if marker_corners is not None:
        for c in marker_corners:
            pts.extend(c.reshape(-1, 2))
    if len(pts) < 4:
        return
    pts = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(pts).astype(np.int32)
    cv2.polylines(vis, [hull], True, ok_color, 4)

def board_corners_3d(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return np.asarray(board.chessboardCorners, dtype=np.float32)

def calibrate_one(all_corners, all_ids, image_size, board):
    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    return rms, K, dist

def stereo_points(left_data, right_data, board):
    obj_all = []
    img_l_all = []
    img_r_all = []
    obj_board = board_corners_3d(board)
    for (corn_l, ids_l), (corn_r, ids_r) in zip(left_data, right_data):
        ids_l = ids_l.flatten().astype(int)
        ids_r = ids_r.flatten().astype(int)
        common = sorted(set(ids_l).intersection(set(ids_r)))
        if len(common) < MIN_CORNERS_STEREO:
            continue
        map_l = {int(i): corn_l[j, 0].astype(np.float32) for j, i in enumerate(ids_l)}
        map_r = {int(i): corn_r[j, 0].astype(np.float32) for j, i in enumerate(ids_r)}
        obj = []
        img_l = []
        img_r = []
        for cid in common:
            obj.append(obj_board[cid])
            img_l.append(map_l[cid])
            img_r.append(map_r[cid])
        obj_all.append(np.asarray(obj, dtype=np.float32))
        img_l_all.append(np.asarray(img_l, dtype=np.float32))
        img_r_all.append(np.asarray(img_r, dtype=np.float32))
    return obj_all, img_l_all, img_r_all

def view_signature(ch_l, ch_r):
    pts = np.vstack((ch_l.reshape(-1, 2), ch_r.reshape(-1, 2)))
    center = pts.mean(axis=0)
    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    area = float(w * h)
    return center, area

def is_new_view(signature, last_signature):
    if last_signature is None:
        return True
    center, area = signature
    last_center, last_area = last_signature
    center_move = float(np.linalg.norm(center - last_center))
    if center_move >= MIN_CENTER_MOVE_PX:
        return True
    if last_area > 1.0:
        area_change = abs(area - last_area) / last_area
        if area_change >= MIN_AREA_CHANGE_RATIO:
            return True
    return False

def main():
    board, aruco_dict = make_board()
    params = detector_params()

    cap_l = open_cam(LEFT_INDEX)
    cap_r = open_cam(RIGHT_INDEX)

    if not cap_l.isOpened():
        raise RuntimeError("OV9281 L konnte nicht geoeffnet werden")
    if not cap_r.isOpened():
        raise RuntimeError("OV9281 R konnte nicht geoeffnet werden")

    left_data = []
    right_data = []
    image_size = None
    saved = 0
    live_ok = 0
    skipped_same = 0
    last_auto_save = 0.0
    last_signature = None

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            time.sleep(0.05)
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        image_size = (gray_l.shape[1], gray_l.shape[0])

        ch_l, ids_l, n_l, marker_corners_l, marker_ids_l = detect_charuco(gray_l, board, aruco_dict, params)
        ch_r, ids_r, n_r, marker_corners_r, marker_ids_r = detect_charuco(gray_r, board, aruco_dict, params)

        markers_l = 0 if marker_ids_l is None else len(marker_ids_l)
        markers_r = 0 if marker_ids_r is None else len(marker_ids_r)

        ok_pair = n_l >= MIN_CORNERS_CAPTURE and n_r >= MIN_CORNERS_CAPTURE
        signature = view_signature(ch_l, ch_r) if ok_pair else None
        new_view = is_new_view(signature, last_signature) if ok_pair else False

        if ok_pair:
            live_ok += 1

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()

        if marker_corners_l is not None and marker_ids_l is not None:
            cv2.aruco.drawDetectedMarkers(vis_l, marker_corners_l, marker_ids_l)
        if marker_corners_r is not None and marker_ids_r is not None:
            cv2.aruco.drawDetectedMarkers(vis_r, marker_corners_r, marker_ids_r)
        if ch_l is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_l, ch_l, ids_l)
        if ch_r is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_r, ch_r, ids_r)

        color = (0, 255, 0) if ok_pair and new_view else (0, 180, 255) if ok_pair else (0, 0, 255)

        draw_board_frame(vis_l, ch_l, marker_corners_l, color)
        draw_board_frame(vis_r, ch_r, marker_corners_r, color)

        cv2.rectangle(vis_l, (10, 10), (760, 185), (0, 0, 0), -1)
        cv2.rectangle(vis_r, (10, 10), (760, 185), (0, 0, 0), -1)

        status = "NEUE POSITION" if ok_pair and new_view else "BEWEGEN" if ok_pair else "BOARD SUCHEN"

        cv2.putText(vis_l, f"L Charuco: {n_l}  ArUco: {markers_l}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_l, f"saved: {saved}/{MIN_PAIRS}  ok: {live_ok}  same: {skipped_same}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_l, status, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_l, "s=manuell speichern  q=abbrechen", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(vis_r, f"R Charuco: {n_r}  ArUco: {markers_r}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_r, f"saved: {saved}/{MIN_PAIRS}  ok: {live_ok}  same: {skipped_same}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_r, status, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis_r, "s=manuell speichern  q=abbrechen", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        combo = np.hstack((cv2.resize(vis_l, (640, 400)), cv2.resize(vis_r, (640, 400))))
        cv2.imshow("OV9281 Charuco Auto Gated", combo)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        manual_save = key == ord("s")

        if ok_pair and saved < MIN_PAIRS and (manual_save or now - last_auto_save >= AUTO_SAVE_INTERVAL_SEC):
            if new_view or manual_save:
                left_data.append((ch_l.copy(), ids_l.copy()))
                right_data.append((ch_r.copy(), ids_r.copy()))
                cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_L.png"), frame_l)
                cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_R.png"), frame_r)
                saved += 1
                last_auto_save = now
                last_signature = signature
                mode = "manual" if manual_save else "auto"
                print(f"{mode} saved {saved}/{MIN_PAIRS} L_charuco={n_l} R_charuco={n_r} L_aruco={markers_l} R_aruco={markers_r}")
            else:
                skipped_same += 1
                if skipped_same % 20 == 0:
                    print(f"blocked_same_view saved={saved}/{MIN_PAIRS} L_charuco={n_l} R_charuco={n_r} L_aruco={markers_l} R_aruco={markers_r}")

        if saved >= MIN_PAIRS:
            break

        if key == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    if len(left_data) < MIN_STEREO_PAIRS:
        raise RuntimeError(f"zu wenige Paare: {len(left_data)} (mindestens {MIN_STEREO_PAIRS})")

    rms_l, K_l, dist_l = calibrate_one(
        [x[0] for x in left_data],
        [x[1] for x in left_data],
        image_size,
        board,
    )

    rms_r, K_r, dist_r = calibrate_one(
        [x[0] for x in right_data],
        [x[1] for x in right_data],
        image_size,
        board,
    )

    obj, img_l, img_r = stereo_points(left_data, right_data, board)

    if len(obj) < MIN_STEREO_PAIRS:
        raise RuntimeError(f"zu wenige Stereo-Paare: {len(obj)} (mindestens {MIN_STEREO_PAIRS})")

    rms_stereo, K_l2, dist_l2, K_r2, dist_r2, R, T, E, F = cv2.stereoCalibrate(
        obj,
        img_l,
        img_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    baseline_cm = float(np.linalg.norm(T) * 100.0)

    out = {
        "calibrated_ov9281": True,
        "camera_matrix_l": K_l.tolist(),
        "dist_l": dist_l.tolist(),
        "camera_matrix_r": K_r.tolist(),
        "dist_r": dist_r.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "baseline_cm": baseline_cm,
        "charuco_board": {
            "dictionary": "DICT_4X4_50",
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_size_m": SQUARE_SIZE_M,
            "marker_size_m": MARKER_SIZE_M,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "pairs": len(left_data),
            "stereo_pairs": len(obj),
            "rms_left": float(rms_l),
            "rms_right": float(rms_r),
            "rms_stereo": float(rms_stereo),
        },
    }

    out_path = BASE / "stereo_config_ov9281.yaml"
    backup_dir = BASE / "local_uncommitted_backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"stereo_config_ov9281_before_charuco_auto_gated_save_{int(time.time())}.yaml"

    if out_path.exists():
        backup.write_text(out_path.read_text())

    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"rms_l {rms_l:.4f}")
    print(f"rms_r {rms_r:.4f}")
    print(f"rms_stereo {rms_stereo:.4f}")
    print(f"baseline_cm {baseline_cm:.2f}")
    print(f"saved {out_path}")
    print(f"backup {backup}")

if __name__ == "__main__":
    main()
