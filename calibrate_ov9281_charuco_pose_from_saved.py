import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
IMG_DIR = BASE / "calib_ov9281_charuco_clean"
OUT_PATH = BASE / "stereo_config_ov9281.yaml"
BACKUP_DIR = BASE / "local_uncommitted_backups"
BACKUP_DIR.mkdir(exist_ok=True)

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015
DICT_ID = cv2.aruco.DICT_4X4_50
MIN_CORNERS = 18
MIN_COMMON = 16

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, aruco_dict)

if hasattr(board, "setLegacyPattern"):
    board.setLegacyPattern(True)

params = cv2.aruco.DetectorParameters()

if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None
chess = board.getChessboardCorners()

def detect(path):
    img = cv2.imread(str(path))
    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, mk_corners, mk_ids = detector.detectBoard(gray)
    else:
        mk_corners, mk_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if mk_ids is None or len(mk_ids) == 0:
            return None, None, img.shape[:2]
        ok, corners, ids = cv2.aruco.interpolateCornersCharuco(mk_corners, mk_ids, gray, board)
        if not ok:
            return None, None, img.shape[:2]

    if corners is None or ids is None:
        return None, None, img.shape[:2]

    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32), img.shape[:2]

def calib_points(records, side):
    objpoints = []
    imgpoints = []

    for rec in records:
        corners = rec[side + "_corners"]
        ids = rec[side + "_ids"]

        if corners is None or ids is None or len(ids) < MIN_CORNERS:
            continue

        obj = np.array([chess[int(i)] for i in ids], dtype=np.float32).reshape(-1, 1, 3)
        img = corners.reshape(-1, 1, 2).astype(np.float32)

        objpoints.append(obj)
        imgpoints.append(img)

    return objpoints, imgpoints

def calibrate_camera(objpoints, imgpoints, image_size):
    K_init = cv2.initCameraMatrix2D(objpoints, imgpoints, image_size, 0)
    dist_init = np.zeros((8, 1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, K_init, dist_init, flags=flags)
    return rms, K, dist

def solve_pose(corners, ids, K, dist):
    if corners is None or ids is None:
        return None, None, 0

    pts = {int(mid): corners[k] for k, mid in enumerate(ids)}
    common = sorted(pts.keys())

    if len(common) < MIN_COMMON:
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

records = []
image_size = None

for lp in sorted(IMG_DIR.glob("pair_*_L.png")):
    rp = IMG_DIR / lp.name.replace("_L.png", "_R.png")

    if not rp.exists():
        continue

    c_l, ids_l, shape_l = detect(lp)
    c_r, ids_r, shape_r = detect(rp)

    if shape_l is None or shape_r is None:
        continue

    if image_size is None:
        image_size = (shape_l[1], shape_l[0])

    if c_l is None or c_r is None:
        continue

    common = sorted(set(int(x) for x in ids_l) & set(int(x) for x in ids_r))

    if len(common) < MIN_COMMON:
        continue

    records.append({
        "name": lp.name,
        "left_corners": c_l,
        "left_ids": ids_l,
        "right_corners": c_r,
        "right_ids": ids_r,
        "common": len(common),
    })

if image_size is None:
    raise RuntimeError("Keine Bilder gefunden.")

if len(records) < 35:
    raise RuntimeError(f"Zu wenige gute Bildpaare: {len(records)}")

obj_l, img_l = calib_points(records, "left")
obj_r, img_r = calib_points(records, "right")

if len(obj_l) < 35 or len(obj_r) < 35:
    raise RuntimeError(f"Zu wenige Mono-Paare: L={len(obj_l)} R={len(obj_r)}")

rms_l, K_l, dist_l = calibrate_camera(obj_l, img_l, image_size)
rms_r, K_r, dist_r = calibrate_camera(obj_r, img_r, image_size)

pose_rows = []

for rec in records:
    map_l = {int(i): rec["left_corners"][k] for k, i in enumerate(rec["left_ids"])}
    map_r = {int(i): rec["right_corners"][k] for k, i in enumerate(rec["right_ids"])}
    common = sorted(set(map_l.keys()) & set(map_r.keys()))

    if len(common) < MIN_COMMON:
        continue

    c_l = np.array([map_l[i] for i in common], dtype=np.float32)
    ids_l = np.array(common, dtype=np.int32)
    c_r = np.array([map_r[i] for i in common], dtype=np.float32)
    ids_r = np.array(common, dtype=np.int32)

    R_l, t_l, n_l = solve_pose(c_l, ids_l, K_l, dist_l)
    R_r, t_r, n_r = solve_pose(c_r, ids_r, K_r, dist_r)

    if R_l is None or R_r is None:
        continue

    R_lr = R_r @ R_l.T
    T_lr = t_r - R_lr @ t_l
    baseline_cm = float(np.linalg.norm(T_lr) * 100.0)

    pose_rows.append({
        "name": rec["name"],
        "R": R_lr,
        "T": T_lr,
        "baseline_cm": baseline_cm,
        "common": len(common),
    })

if len(pose_rows) < 35:
    raise RuntimeError(f"Zu wenige Pose-Paare: {len(pose_rows)}")

baselines = np.array([x["baseline_cm"] for x in pose_rows], dtype=np.float64)
baseline_median = float(np.median(baselines))
baseline_mad = float(np.median(np.abs(baselines - baseline_median)))
limit = max(0.50, baseline_mad * 4.0)

inliers = [x for x in pose_rows if abs(x["baseline_cm"] - baseline_median) <= limit]

if len(inliers) < 25:
    raise RuntimeError(f"Zu wenige Stereo-Inlier: {len(inliers)}")

R = project_rotation_mean([x["R"] for x in inliers])
T = np.median(np.array([x["T"] for x in inliers], dtype=np.float64), axis=0).reshape(3)

baseline_cm = float(np.linalg.norm(T) * 100.0)
inlier_baselines = np.array([x["baseline_cm"] for x in inliers], dtype=np.float64)

print("OV9281 Pose-Median-Kalibrierung aus gespeicherten Bildern")
print(f"records={len(records)}")
print(f"pose_pairs={len(pose_rows)}")
print(f"inliers={len(inliers)}")
print(f"rms_l={rms_l:.4f}")
print(f"rms_r={rms_r:.4f}")
print(f"fx_l={K_l[0,0]:.2f} fy_l={K_l[1,1]:.2f}")
print(f"fx_r={K_r[0,0]:.2f} fy_r={K_r[1,1]:.2f}")
print(f"baseline_all_median_cm={baseline_median:.3f}")
print(f"baseline_all_mad_cm={baseline_mad:.3f}")
print(f"baseline_inlier_median_cm={float(np.median(inlier_baselines)):.3f}")
print(f"baseline_inlier_mad_cm={float(np.median(np.abs(inlier_baselines - np.median(inlier_baselines)))):.3f}")
print(f"baseline_final_cm={baseline_cm:.3f}")
print(f"T_cm=({T[0]*100.0:.3f}, {T[1]*100.0:.3f}, {T[2]*100.0:.3f})")

if not (7.0 <= baseline_cm <= 9.0):
    raise RuntimeError(f"Baseline unplausibel: {baseline_cm:.3f} cm")

if baseline_mad > 0.75:
    raise RuntimeError(f"Baseline-Streuung zu hoch: MAD={baseline_mad:.3f} cm")

backup = BACKUP_DIR / f"stereo_config_ov9281_before_pose_median_{int(time.time())}.yaml"

if OUT_PATH.exists():
    backup.write_text(OUT_PATH.read_text())

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
        "method": "pose_median_from_saved_charuco",
        "records": int(len(records)),
        "pose_pairs": int(len(pose_rows)),
        "inliers": int(len(inliers)),
        "rms_left": float(rms_l),
        "rms_right": float(rms_r),
        "baseline_all_median_cm": float(baseline_median),
        "baseline_all_mad_cm": float(baseline_mad),
        "baseline_inlier_median_cm": float(np.median(inlier_baselines)),
        "baseline_inlier_mad_cm": float(np.median(np.abs(inlier_baselines - np.median(inlier_baselines)))),
    },
}

OUT_PATH.write_text(yaml.safe_dump(out, sort_keys=False))

print(f"saved={OUT_PATH}")
print(f"backup={backup}")
