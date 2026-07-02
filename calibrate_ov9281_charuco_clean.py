import cv2
import yaml
import time
import math
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
OUT_DIR = BASE / "calib_ov9281_charuco_clean"
OUT_DIR.mkdir(exist_ok=True)

LEFT_INDEX = 1
RIGHT_INDEX = 2

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015
DICT_ID = cv2.aruco.DICT_4X4_50

TARGET_PAIRS = 60
MIN_GOOD_PAIRS = 35
MIN_CORNERS_CAPTURE = 18
MIN_COMMON_STEREO = 16
AUTO_SAVE_INTERVAL_SEC = 0.6
MIN_CENTER_MOVE_PX = 90.0
MIN_SCALE_CHANGE = 0.10
MIN_ANGLE_CHANGE_DEG = 7.0
STABLE_MIN_FRAMES = 8
STABLE_CENTER_MOVE_PX = 4.0
STABLE_SCALE_CHANGE = 0.015
STABLE_ANGLE_CHANGE_DEG = 1.5

def make_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, aruco_dict)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, aruco_dict

def make_detector(board, aruco_dict):
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board), params
    return None, params

def detect_charuco(frame, board, aruco_dict, detector, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)
    else:
        mk_corners, mk_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if mk_ids is None or len(mk_ids) == 0:
            return None, None, 0
        ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(mk_corners, mk_ids, gray, board)
        if not ok:
            return None, None, int(len(mk_ids))
    if ch_corners is None or ch_ids is None:
        marker_count = 0 if mk_ids is None else int(len(mk_ids))
        return None, None, marker_count
    return ch_corners.astype(np.float32), ch_ids.astype(np.int32).reshape(-1), 0 if mk_ids is None else int(len(mk_ids))

def signature(corners):
    pts = corners.reshape(-1, 2)
    center = pts.mean(axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    scale = float(np.linalg.norm(span))
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    angle = float(math.degrees(math.atan2(v[1], v[0])))
    return center, scale, angle

def angle_delta(a, b):
    d = abs(a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d

def is_new_view(sig, last_sig):
    if last_sig is None:
        return True
    c, sc, ang = sig
    lc, lsc, lang = last_sig
    center_move = float(np.linalg.norm(c - lc))
    scale_change = abs(sc - lsc) / max(lsc, 1.0)
    angle_change = angle_delta(ang, lang)
    return center_move >= MIN_CENTER_MOVE_PX or scale_change >= MIN_SCALE_CHANGE or angle_change >= MIN_ANGLE_CHANGE_DEG


def is_stable_view(sig, prev_sig):
    if prev_sig is None:
        return False
    c, sc, ang = sig
    pc, psc, pang = prev_sig
    center_move = float(np.linalg.norm(c - pc))
    scale_change = abs(sc - psc) / max(psc, 1.0)
    angle_change = angle_delta(ang, pang)
    return center_move <= STABLE_CENTER_MOVE_PX and scale_change <= STABLE_SCALE_CHANGE and angle_change <= STABLE_ANGLE_CHANGE_DEG

def common_count(ids_l, ids_r):
    if ids_l is None or ids_r is None:
        return 0
    return len(set(int(x) for x in ids_l) & set(int(x) for x in ids_r))

def charuco_to_calib_points(records, board):
    chess = board.getChessboardCorners()
    objpoints = []
    imgpoints = []
    kept = []
    for idx, corners, ids in records:
        if corners is None or ids is None or len(ids) < MIN_CORNERS_CAPTURE:
            continue
        ids = ids.reshape(-1).astype(np.int32)
        pts3 = np.array([chess[int(i)] for i in ids], dtype=np.float32).reshape(-1, 1, 3)
        pts2 = corners.reshape(-1, 1, 2).astype(np.float32)
        objpoints.append(pts3)
        imgpoints.append(pts2)
        kept.append(idx)
    return objpoints, imgpoints, kept

def stereo_points(left_records, right_records, board):
    chess = board.getChessboardCorners()
    right_by_idx = {idx: (corners, ids) for idx, corners, ids in right_records}
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    kept = []
    for idx, corners_l, ids_l in left_records:
        if idx not in right_by_idx:
            continue
        corners_r, ids_r = right_by_idx[idx]
        map_l = {int(i): corners_l.reshape(-1, 2)[k] for k, i in enumerate(ids_l.reshape(-1))}
        map_r = {int(i): corners_r.reshape(-1, 2)[k] for k, i in enumerate(ids_r.reshape(-1))}
        common = sorted(set(map_l.keys()) & set(map_r.keys()))
        if len(common) < MIN_COMMON_STEREO:
            continue
        pts3 = np.array([chess[i] for i in common], dtype=np.float32).reshape(-1, 1, 3)
        pts_l = np.array([map_l[i] for i in common], dtype=np.float32).reshape(-1, 1, 2)
        pts_r = np.array([map_r[i] for i in common], dtype=np.float32).reshape(-1, 1, 2)
        objpoints.append(pts3)
        imgpoints_l.append(pts_l)
        imgpoints_r.append(pts_r)
        kept.append(idx)
    return objpoints, imgpoints_l, imgpoints_r, kept

def calibrate_camera(objpoints, imgpoints, image_size):
    K_init = cv2.initCameraMatrix2D(objpoints, imgpoints, image_size, 0)
    dist_init = np.zeros((8, 1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, K_init, dist_init, flags=flags)
    return rms, K, dist

def main():
    board, aruco_dict = make_board()
    detector, params = make_detector(board, aruco_dict)

    cap_l = cv2.VideoCapture(LEFT_INDEX)
    cap_r = cv2.VideoCapture(RIGHT_INDEX)

    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    if not cap_l.isOpened():
        raise RuntimeError("OV9281 links konnte nicht geoeffnet werden")
    if not cap_r.isOpened():
        raise RuntimeError("OV9281 rechts konnte nicht geoeffnet werden")

    left_records = []
    right_records = []
    saved = 0
    frame_i = 0
    last_sig = None
    prev_sig = None
    stable_frames = 0
    last_save = 0.0
    image_size = None

    print("OV9281 ChArUco Clean Calibration")
    print("s = manuell speichern | q = beenden und kalibrieren")
    print(f"Ziel: {TARGET_PAIRS} gute, unterschiedliche Paare")

    while True:
        frame_i += 1
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()

        if not ok_l or frame_l is None or not ok_r or frame_r is None:
            continue

        image_size = (frame_l.shape[1], frame_l.shape[0])

        ch_l, ids_l, mk_l = detect_charuco(frame_l, board, aruco_dict, detector, params)
        ch_r, ids_r, mk_r = detect_charuco(frame_r, board, aruco_dict, detector, params)

        n_l = 0 if ids_l is None else len(ids_l)
        n_r = 0 if ids_r is None else len(ids_r)
        n_common = common_count(ids_l, ids_r)

        ok_pair = n_l >= MIN_CORNERS_CAPTURE and n_r >= MIN_CORNERS_CAPTURE and n_common >= MIN_COMMON_STEREO

        sig = None
        new_view = False
        if ok_pair:
            sig_l = signature(ch_l)
            sig_r = signature(ch_r)
            sig = ((sig_l[0] + sig_r[0]) / 2.0, (sig_l[1] + sig_r[1]) / 2.0, (sig_l[2] + sig_r[2]) / 2.0)
            new_view = is_new_view(sig, last_sig)
            if is_stable_view(sig, prev_sig):
                stable_frames += 1
            else:
                stable_frames = 0
            prev_sig = sig
        else:
            stable_frames = 0
            prev_sig = None

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()

        if ch_l is not None and ids_l is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_l, ch_l, ids_l)
        if ch_r is not None and ids_r is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_r, ch_r, ids_r)

        manual = (cv2.waitKey(1) & 0xFF) == ord("s")
        now = time.time()

        if ok_pair and saved < TARGET_PAIRS and (manual or (new_view and stable_frames >= STABLE_MIN_FRAMES and now - last_save >= AUTO_SAVE_INTERVAL_SEC)):
            left_records.append((saved, ch_l.copy(), ids_l.copy()))
            right_records.append((saved, ch_r.copy(), ids_r.copy()))
            cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_L.png"), frame_l)
            cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_R.png"), frame_r)
            saved += 1
            last_sig = sig
            last_save = now
            print(f"saved {saved}/{TARGET_PAIRS} L={n_l} R={n_r} common={n_common}")

        for vis, name, n, mk in [(vis_l, "OV9281 L", n_l, mk_l), (vis_r, "OV9281 R", n_r, mk_r)]:
            cv2.rectangle(vis, (0, 0), (1280, 95), (0, 0, 0), -1)
            color = (0, 255, 0) if ok_pair else (0, 0, 255)
            cv2.putText(vis, f"{name} charuco={n} markers={mk}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(vis, f"saved={saved}/{TARGET_PAIRS} common={n_common} new={new_view} stable={stable_frames}/{STABLE_MIN_FRAMES}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        combo = np.vstack((cv2.resize(vis_l, (640, 400)), cv2.resize(vis_r, (640, 400))))
        cv2.imshow("OV9281 ChArUco Clean | q beendet und kalibriert", combo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if saved >= TARGET_PAIRS:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("keine Bilder aufgenommen")

    obj_l, img_l, kept_l = charuco_to_calib_points(left_records, board)
    obj_r, img_r, kept_r = charuco_to_calib_points(right_records, board)
    obj_s, simg_l, simg_r, kept_s = stereo_points(left_records, right_records, board)

    print(f"saved_pairs={saved}")
    print(f"left_calib_pairs={len(obj_l)}")
    print(f"right_calib_pairs={len(obj_r)}")
    print(f"stereo_pairs={len(obj_s)}")

    if len(obj_l) < MIN_GOOD_PAIRS or len(obj_r) < MIN_GOOD_PAIRS or len(obj_s) < MIN_GOOD_PAIRS:
        raise RuntimeError("zu wenige gute Paare fuer saubere Kalibrierung")

    rms_l, K_l, dist_l = calibrate_camera(obj_l, img_l, image_size)
    rms_r, K_r, dist_r = calibrate_camera(obj_r, img_r, image_size)

    flags = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, K_l2, dist_l2, K_r2, dist_r2, R, T, E, F = cv2.stereoCalibrate(
        obj_s,
        simg_l,
        simg_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )

    baseline_cm = float(np.linalg.norm(T) * 100.0)

    print(f"rms_l={rms_l:.4f}")
    print(f"rms_r={rms_r:.4f}")
    print(f"rms_stereo={rms_stereo:.4f}")
    print(f"baseline_cm={baseline_cm:.2f}")
    print(f"fx_l={K_l[0,0]:.2f} fy_l={K_l[1,1]:.2f}")
    print(f"fx_r={K_r[0,0]:.2f} fy_r={K_r[1,1]:.2f}")

    if not (6.0 <= baseline_cm <= 10.0):
        bad_path = BASE / "local_uncommitted_backups" / f"stereo_config_ov9281_REJECTED_baseline_{baseline_cm:.1f}cm_{int(time.time())}.yaml"
        out_bad = {
            "rejected": True,
            "reason": "baseline outside plausible range",
            "baseline_cm": baseline_cm,
            "rms_left": float(rms_l),
            "rms_right": float(rms_r),
            "rms_stereo": float(rms_stereo),
            "camera_matrix_l": K_l.tolist(),
            "dist_l": dist_l.tolist(),
            "camera_matrix_r": K_r.tolist(),
            "dist_r": dist_r.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
        }
        bad_path.write_text(yaml.safe_dump(out_bad, sort_keys=False))
        raise RuntimeError(f"Kalibrierung verworfen: baseline_cm={baseline_cm:.2f}, gespeichert als {bad_path}")

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
            "saved_pairs": int(saved),
            "left_calib_pairs": int(len(obj_l)),
            "right_calib_pairs": int(len(obj_r)),
            "stereo_pairs": int(len(obj_s)),
            "rms_left": float(rms_l),
            "rms_right": float(rms_r),
            "rms_stereo": float(rms_stereo),
        },
    }

    out_path = BASE / "stereo_config_ov9281.yaml"
    backup_dir = BASE / "local_uncommitted_backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"stereo_config_ov9281_before_clean_charuco_{int(time.time())}.yaml"

    if out_path.exists():
        backup.write_text(out_path.read_text())

    out_path.write_text(yaml.safe_dump(out, sort_keys=False))

    print(f"saved={out_path}")
    print(f"backup={backup}")

if __name__ == "__main__":
    main()
