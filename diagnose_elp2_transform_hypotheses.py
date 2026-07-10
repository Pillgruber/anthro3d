import cv2
import yaml
import time
import numpy as np
from pathlib import Path
import calibrate_positions as cp

BASE = Path.home() / "anthro3d"
CFG_PATH = BASE / "stereo_config.yaml"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
d_l = np.array(cfg["dist_l"], dtype=np.float64)
K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
d_r = np.array(cfg["dist_r"], dtype=np.float64)
R = np.array(cfg["R"], dtype=np.float64)
T = np.array(cfg["T"], dtype=np.float64).reshape(3)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params = cv2.aruco.DetectorParameters()

if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def detect_side(frame, K, dist):
    gray = cp.preprocess(frame)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for corner, mid_raw in zip(corners, ids.flatten()):
        mid = int(mid_raw)
        if mid not in cp.KNOWN_MARKER_IDS:
            continue
        ok, rvec, tvec, err = cp.solve_marker_pose(corner[0].astype(np.float32), K, dist)
        if not ok:
            continue
        if err > cp.MAX_REPROJECTION_ERROR_PX:
            continue
        out[mid] = {
            "rvec": rvec,
            "tvec": tvec.reshape(3),
            "err": float(err),
        }
    return out

def rot_diff_deg(rvec_a, rvec_b):
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    Rd = Ra @ Rb.T
    tr = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def convert_pose(rvec_r, tvec_r, mode):
    R_marker_r, _ = cv2.Rodrigues(rvec_r)
    tvec_r = tvec_r.reshape(3)

    if mode == "current_Rt_T":
        R_marker_l = R.T @ R_marker_r
        t_marker_l = R.T @ (tvec_r - T)
    elif mode == "current_Rt_negT":
        R_marker_l = R.T @ R_marker_r
        t_marker_l = R.T @ (tvec_r + T)
    elif mode == "direct_R_T":
        R_marker_l = R @ R_marker_r
        t_marker_l = R @ tvec_r + T
    elif mode == "direct_R_negT":
        R_marker_l = R @ R_marker_r
        t_marker_l = R @ tvec_r - T
    else:
        raise ValueError(mode)

    rvec_l, _ = cv2.Rodrigues(R_marker_l)
    return rvec_l, t_marker_l.reshape(3)

def summarize(vals):
    arr = np.array(vals, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad, float(np.min(arr)), float(np.max(arr))

modes = ["current_Rt_T", "current_Rt_negT", "direct_R_T", "direct_R_negT"]
samples = {m: {} for m in modes}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise SystemExit("ELP2 index 0 nicht offen")

for frame_i in range(80):
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.03)
        continue

    left, right = split_sbs(frame)
    det_l = detect_side(left, K_l, d_l)
    det_r = detect_side(right, K_r, d_r)

    for mid in sorted(set(det_l.keys()) & set(det_r.keys())):
        l = det_l[mid]
        r = det_r[mid]
        for mode in modes:
            rvec_conv, tvec_conv = convert_pose(r["rvec"], r["tvec"], mode)
            dt_cm = float(np.linalg.norm(l["tvec"] - tvec_conv) * 100.0)
            dr_deg = rot_diff_deg(l["rvec"], rvec_conv)
            samples[mode].setdefault(mid, []).append((dt_cm, dr_deg, l["err"], r["err"]))

cap.release()

print("ELP2 Transformations-Hypothesen")
print("T_config_m", T)
print("baseline_cm", float(np.linalg.norm(T) * 100.0))

for mode in modes:
    print("")
    print("===", mode, "===")
    all_dt = []
    all_dr = []
    for mid in sorted(samples[mode]):
        rows = samples[mode][mid]
        dt = [x[0] for x in rows]
        dr = [x[1] for x in rows]
        dt_med, dt_mad, dt_min, dt_max = summarize(dt)
        dr_med, dr_mad, dr_min, dr_max = summarize(dr)
        all_dt.extend(dt)
        all_dr.extend(dr)
        print(f"ID{mid}: n={len(rows)} dT_med={dt_med:.2f}cm dT_mad={dt_mad:.2f}cm dT_minmax={dt_min:.2f}/{dt_max:.2f}cm dR_med={dr_med:.2f}deg")
    if all_dt:
        dt_med, dt_mad, dt_min, dt_max = summarize(all_dt)
        dr_med, dr_mad, dr_min, dr_max = summarize(all_dr)
        print(f"GESAMT: dT_med={dt_med:.2f}cm dT_mad={dt_mad:.2f}cm dR_med={dr_med:.2f}deg")
    else:
        print("keine gemeinsamen Marker")

print("")
print("Fertig.")
