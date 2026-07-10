import cv2
import yaml
import time
import math
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
OUT_DIR = BASE / "calib_elp2_charuco_clean"
OUT_DIR.mkdir(exist_ok=True)

CAM_INDEX = 0
OUT_PATH = BASE / "stereo_config.yaml"

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

BASELINE_MIN_CM = 5.0
BASELINE_MAX_CM = 7.0

def make_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, aruco_dict)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, aruco_dict

def make_detector(board):
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None
    return detector, params

def split_side_by_side(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()

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
    marker_count = 0 if mk_ids is None else int(len(mk_ids))
    return ch_corners.astype(np.float32), ch_ids.astype(np.int32).reshape(-1), marker_count

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

def calib_points(records, side, board):
    chess = board.getChessboardCorners()
    objpoints = []
    imgpoints = []
    for rec in records:
        corners = rec[side + "_corners"]
        ids = rec[side + "_ids"]
        if corners is None or ids is None or len(ids) < MIN_CORNERS_CAPTURE:
            continue
        obj = np.array([chess[int(i)] for i in ids], dtype=np.float32).reshape(-1, 1, 3)
        img = corners.reshape(-1, 1, 2).astype(np.float32)
        objpoints.append(obj)
        imgpoints.append(img)
    return objpoints, imgpoints

def calibrate_camera(objpoints, imgpoints, image_size):
    K_init = cv2.initCameraMatrix2D(objpoints, imgpoints, image_size, 0)
    dist_init = np.zeros((5, 1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, K_init, dist_init, flags=flags)
    return rms, K, dist

def solve_pose(corners, ids, K, dist, board):
    chess = board.getChessboardCorners()
    if corners is None or ids is None:
        return None, None, 0
    pts = {int(mid): corners[k] for k, mid in enumerate(ids)}
    common = sorted(pts.keys())
    if len(common) < MIN_COMMON_STEREO:
        return None, None, len(common)
    obj = np.array([chess[i] for i in common], dtype=np.float32).reshape(-1, 1, 3)
    img = np.array([pts[i] for i in common], dtype=np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, len(common)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), len(common)

def project_rotation_mean(rotations):
    M = np.mean(np.array(rotations, dtype=np.float64), axis=0)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def main():
    board, aruco_dict = make_board()
    detector, params = make_detector(board)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("ELP2 Index 0 konnte nicht geoeffnet werden")
    records = []
    saved = 0
    last_sig = None
    prev_sig = None
    stable_frames = 0
    last_save = 0.0
    image_size = None
    print("ELP2 ChArUco Clean Calibration")
    print("Board bewegen, kurz ruhig halten, speichern abwarten.")
    print("s = manuell speichern | q = beenden und kalibrieren")
    print(f"Ziel: {TARGET_PAIRS} gute, unterschiedliche Paare")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = split_side_by_side(frame)
        image_size = (left.shape[1], left.shape[0])
        ch_l, ids_l, mk_l = detect_charuco(left, board, aruco_dict, detector, params)
        ch_r, ids_r, mk_r = detect_charuco(right, board, aruco_dict, detector, params)
        n_l = 0 if ids_l is None else len(ids_l)
        n_r = 0 if ids_r is None else len(ids_r)
        n_common = common_count(ids_l, ids_r)
        ok_pair = n_l >= MIN_CORNERS_CAPTURE and n_r >= MIN_CORNERS_CAPTURE and n_common >= MIN_COMMON_STEREO
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
            sig = None
        vis_l = left.copy()
        vis_r = right.copy()
        if ch_l is not None and ids_l is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_l, ch_l, ids_l)
        if ch_r is not None and ids_r is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_r, ch_r, ids_r)
        key = cv2.waitKey(1) & 0xFF
        manual = key == ord("s")
        now = time.time()
        if ok_pair and saved < TARGET_PAIRS and (manual or (new_view and stable_frames >= STABLE_MIN_FRAMES and now - last_save >= AUTO_SAVE_INTERVAL_SEC)):
            records.append({"idx": saved, "left_corners": ch_l.copy(), "left_ids": ids_l.copy(), "right_corners": ch_r.copy(), "right_ids": ids_r.copy(), "common": int(n_common)})
            cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_L.png"), left)
            cv2.imwrite(str(OUT_DIR / f"pair_{saved:03d}_R.png"), right)
            saved += 1
            last_sig = sig
            last_save = now
            print(f"saved {saved}/{TARGET_PAIRS} L={n_l} R={n_r} common={n_common}")
        for vis, name, n, mk in [(vis_l, "ELP2 L", n_l, mk_l), (vis_r, "ELP2 R", n_r, mk_r)]:
            cv2.rectangle(vis, (0, 0), (1280, 95), (0, 0, 0), -1)
            color = (0, 255, 0) if ok_pair else (0, 0, 255)
            cv2.putText(vis, f"{name} charuco={n} markers={mk}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(vis, f"saved={saved}/{TARGET_PAIRS} common={n_common} stable={stable_frames}/{STABLE_MIN_FRAMES}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        combo = np.vstack((cv2.resize(vis_l, (640, 360)), cv2.resize(vis_r, (640, 360))))
        cv2.imshow("ELP2 ChArUco Clean | q beendet und kalibriert", combo)
        if key == ord("q"):
            break
        if saved >= TARGET_PAIRS:
            break
    cap.release()
    cv2.destroyAllWindows()
    if image_size is None:
        raise RuntimeError("keine Bilder aufgenommen")
    if len(records) < MIN_GOOD_PAIRS:
        raise RuntimeError(f"zu wenige gute Paare: {len(records)}")
    obj_l, img_l = calib_points(records, "left", board)
    obj_r, img_r = calib_points(records, "right", board)
    rms_l, K_l, dist_l = calibrate_camera(obj_l, img_l, image_size)
    rms_r, K_r, dist_r = calibrate_camera(obj_r, img_r, image_size)
    pose_rows = []
    for rec in records:
        map_l = {int(i): rec["left_corners"][k] for k, i in enumerate(rec["left_ids"])}
        map_r = {int(i): rec["right_corners"][k] for k, i in enumerate(rec["right_ids"])}
        common = sorted(set(map_l.keys()) & set(map_r.keys()))
        if len(common) < MIN_COMMON_STEREO:
            continue
        c_l = np.array([map_l[i] for i in common], dtype=np.float32)
        ids_l = np.array(common, dtype=np.int32)
        c_r = np.array([map_r[i] for i in common], dtype=np.float32)
        ids_r = np.array(common, dtype=np.int32)
        R_l, t_l, n_l = solve_pose(c_l, ids_l, K_l, dist_l, board)
        R_r, t_r, n_r = solve_pose(c_r, ids_r, K_r, dist_r, board)
        if R_l is None or R_r is None:
            continue
        R_lr = R_r @ R_l.T
        T_lr = t_r - R_lr @ t_l
        baseline_cm = float(np.linalg.norm(T_lr) * 100.0)
        pose_rows.append({"R": R_lr, "T": T_lr, "baseline_cm": baseline_cm, "common": len(common)})
    if len(pose_rows) < MIN_GOOD_PAIRS:
        raise RuntimeError(f"zu wenige Pose-Paare: {len(pose_rows)}")
    baselines = np.array([x["baseline_cm"] for x in pose_rows], dtype=np.float64)
    baseline_median = float(np.median(baselines))
    baseline_mad = float(np.median(np.abs(baselines - baseline_median)))
    limit = max(0.50, baseline_mad * 4.0)
    inliers = [x for x in pose_rows if abs(x["baseline_cm"] - baseline_median) <= limit]
    if len(inliers) < 25:
        raise RuntimeError(f"zu wenige Stereo-Inlier: {len(inliers)}")
    R_final = project_rotation_mean([x["R"] for x in inliers])
    T_final = np.median(np.array([x["T"] for x in inliers], dtype=np.float64), axis=0).reshape(3)
    baseline_cm = float(np.linalg.norm(T_final) * 100.0)
    inlier_baselines = np.array([x["baseline_cm"] for x in inliers], dtype=np.float64)
    print("ELP2 Pose-Median-Kalibrierung")
    print(f"records={len(records)}")
    print(f"pose_pairs={len(pose_rows)}")
    print(f"inliers={len(inliers)}")
    print(f"rms_l={rms_l:.4f}")
    print(f"rms_r={rms_r:.4f}")
    print(f"fx_l={K_l[0,0]:.2f} fy_l={K_l[1,1]:.2f}")
    print(f"fx_r={K_r[0,0]:.2f} fy_r={K_r[1,1]:.2f}")
    print(f"baseline_all_median_cm={baseline_median:.3f}")
    print(f"baseline_all_mad_cm={baseline_mad:.3f}")
    print(f"baseline_final_cm={baseline_cm:.3f}")
    print(f"T_cm=({T_final[0]*100.0:.3f}, {T_final[1]*100.0:.3f}, {T_final[2]*100.0:.3f})")
    backup_dir = BASE / "local_uncommitted_backups"
    backup_dir.mkdir(exist_ok=True)
    if not (BASELINE_MIN_CM <= baseline_cm <= BASELINE_MAX_CM):
        bad_path = backup_dir / f"stereo_config_elp2_REJECTED_baseline_{baseline_cm:.1f}cm_{int(time.time())}.yaml"
        bad_path.write_text(yaml.safe_dump({"rejected": True, "baseline_cm": baseline_cm, "rms_left": float(rms_l), "rms_right": float(rms_r), "camera_matrix_l": K_l.tolist(), "dist_l": dist_l.tolist(), "camera_matrix_r": K_r.tolist(), "dist_r": dist_r.tolist(), "R": R_final.tolist(), "T": T_final.tolist()}, sort_keys=False))
        raise RuntimeError(f"Kalibrierung verworfen: baseline_cm={baseline_cm:.2f}, gespeichert als {bad_path}")
    backup = backup_dir / f"stereo_config_before_elp2_pose_median_{int(time.time())}.yaml"
    if OUT_PATH.exists():
        backup.write_text(OUT_PATH.read_text())
    out = {"calibrated_elps": True, "calibrated_indices": [CAM_INDEX], "camera_matrix_l": K_l.tolist(), "dist_l": dist_l.tolist(), "camera_matrix_r": K_r.tolist(), "dist_r": dist_r.tolist(), "R": R_final.tolist(), "T": T_final.tolist(), "baseline_cm": baseline_cm, "charuco_board": {"dictionary": "DICT_4X4_50", "squares_x": SQUARES_X, "squares_y": SQUARES_Y, "square_size_m": SQUARE_SIZE_M, "marker_size_m": MARKER_SIZE_M, "image_width": image_size[0], "image_height": image_size[1], "method": "elp2_pose_median_charuco", "records": int(len(records)), "pose_pairs": int(len(pose_rows)), "inliers": int(len(inliers)), "rms_left": float(rms_l), "rms_right": float(rms_r), "baseline_all_median_cm": float(baseline_median), "baseline_all_mad_cm": float(baseline_mad), "baseline_inlier_median_cm": float(np.median(inlier_baselines)), "baseline_inlier_mad_cm": float(np.median(np.abs(inlier_baselines - np.median(inlier_baselines))))}}
    OUT_PATH.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"saved={OUT_PATH}")
    print(f"backup={backup}")

if __name__ == "__main__":
    main()
