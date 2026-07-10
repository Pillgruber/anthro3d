import cv2
import yaml
import time
import numpy as np
from pathlib import Path
import calibrate_positions as cp

BASE = Path.home() / "anthro3d"
MARKER_SIZE_M = 0.1865

RIGS = [
    {"name": "ELP2", "kind": "sbs", "index": 0, "config": "stereo_config.yaml", "width": 2560, "height": 720},
    {"name": "ELP1", "kind": "sbs", "index": 3, "config": "stereo_config_elp1.yaml", "width": 2560, "height": 720},
    {"name": "OV9281", "kind": "dual", "left_index": 1, "right_index": 2, "config": "stereo_config_ov9281.yaml", "width": 1280, "height": 800},
]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params = cv2.aruco.DetectorParameters()

if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def detect(frame):
    gray = cp.preprocess(frame)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid in cp.KNOWN_MARKER_IDS:
            out[mid] = c[0].astype(np.float32)
    return out

def triangulate(c_l, c_r, K_l, d_l, K_r, d_r, R, T):
    pl = cv2.undistortPoints(c_l.reshape(-1, 1, 2), K_l, d_l).reshape(-1, 2)
    pr = cv2.undistortPoints(c_r.reshape(-1, 1, 2), K_r, d_r).reshape(-1, 2)
    P_l = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    P_r = np.hstack([R, T.reshape(3, 1)]).astype(np.float64)
    X_h = cv2.triangulatePoints(P_l, P_r, pl.T, pr.T)
    X = (X_h[:3] / X_h[3]).T
    return X

def marker_stats(X):
    edges = [
        np.linalg.norm(X[1] - X[0]),
        np.linalg.norm(X[2] - X[1]),
        np.linalg.norm(X[3] - X[2]),
        np.linalg.norm(X[0] - X[3]),
    ]
    center = X.mean(axis=0)
    size = float(np.median(edges))
    return center, size

def summarize(vals):
    a = np.array(vals, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, mad, float(np.min(a)), float(np.max(a))

def open_cap(idx, w, h):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def process_rig(rig):
    cfg = load_cfg(BASE / rig["config"])
    K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
    d_l = np.array(cfg["dist_l"], dtype=np.float64)
    K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
    d_r = np.array(cfg["dist_r"], dtype=np.float64)
    R = np.array(cfg["R"], dtype=np.float64)
    T = np.array(cfg["T"], dtype=np.float64).reshape(3, 1)

    samples = {}
    centers_by_frame = []

    if rig["kind"] == "sbs":
        cap = open_cap(rig["index"], rig["width"], rig["height"])
        if not cap.isOpened():
            print(f"{rig['name']}: Kamera nicht offen")
            return
        for _ in range(100):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            left, right = split_sbs(frame)
            handle_frame(left, right, K_l, d_l, K_r, d_r, R, T, samples, centers_by_frame)
        cap.release()
    else:
        cap_l = open_cap(rig["left_index"], rig["width"], rig["height"])
        cap_r = open_cap(rig["right_index"], rig["width"], rig["height"])
        if not cap_l.isOpened() or not cap_r.isOpened():
            print(f"{rig['name']}: Kamera nicht offen")
            cap_l.release()
            cap_r.release()
            return
        for _ in range(100):
            ok_l, left = cap_l.read()
            ok_r, right = cap_r.read()
            if not (ok_l and ok_r) or left is None or right is None:
                time.sleep(0.03)
                continue
            handle_frame(left, right, K_l, d_l, K_r, d_r, R, T, samples, centers_by_frame)
        cap_l.release()
        cap_r.release()

    print("")
    print(f"=== {rig['name']} triangulierte Marker-Geometrie ===")
    print("baseline_cm", float(np.linalg.norm(T) * 100.0))

    if not samples:
        print("keine gemeinsamen Marker")
        return

    for mid in sorted(samples):
        sizes = [s["size"] * 100.0 for s in samples[mid]]
        zs = [s["z"] for s in samples[mid]]
        smed, smad, smin, smax = summarize(sizes)
        zmed, zmad, zmin, zmax = summarize(zs)
        scale = smed / (MARKER_SIZE_M * 100.0)
        status = "OK" if abs(smed - MARKER_SIZE_M * 100.0) <= 2.5 and smad <= 1.0 else "AUFFAELLIG"
        print(f"ID{mid}: {status} n={len(sizes)} marker_size={smed:.2f}cm MAD={smad:.2f}cm scale={scale:.3f} z={zmed:.2f}m")

    print("")
    print("Vertikale Paarabstaende")
    for a, b in [(2, 20), (3, 30), (4, 40)]:
        vals = []
        for fc in centers_by_frame:
            if a in fc and b in fc:
                vals.append(float(np.linalg.norm(fc[b] - fc[a]) * 100.0))
        if vals:
            med, mad, vmin, vmax = summarize(vals)
            print(f"ID{a}->ID{b}: n={len(vals)} dist={med:.2f}cm MAD={mad:.2f}cm minmax={vmin:.2f}/{vmax:.2f}cm")
        else:
            print(f"ID{a}->ID{b}: keine gemeinsamen Frames")

def handle_frame(left, right, K_l, d_l, K_r, d_r, R, T, samples, centers_by_frame):
    dl = detect(left)
    dr = detect(right)
    frame_centers = {}
    for mid in sorted(set(dl.keys()) & set(dr.keys())):
        X = triangulate(dl[mid], dr[mid], K_l, d_l, K_r, d_r, R, T)
        if not np.isfinite(X).all():
            continue
        center, size = marker_stats(X)
        if center[2] <= 0:
            continue
        samples.setdefault(mid, []).append({
            "center": center,
            "size": size,
            "z": float(center[2]),
        })
        frame_centers[mid] = center
    centers_by_frame.append(frame_centers)

def main():
    print("ANTHRO3D triangulierte Marker-Geometrie")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    for rig in RIGS:
        process_rig(rig)
    print("")
    print("Fertig.")

if __name__ == "__main__":
    main()
