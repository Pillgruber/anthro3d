import cv2
import yaml
import time
import random
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"

RIGS = [
    {"name": "ELP2", "kind": "sbs", "index": 0, "config": "stereo_config.yaml"},
    {"name": "ELP1", "kind": "sbs", "index": 3, "config": "stereo_config_elp1.yaml"},
    {"name": "OV9281", "kind": "dual", "left_index": 1, "right_index": 2, "config": "stereo_config_ov9281.yaml"},
]

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def mat(x):
    return np.array(x, dtype=np.float64)

def open_cap(idx, width=None, height=None):
    cap = cv2.VideoCapture(idx)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def read_good(cap, count=10):
    frame = None
    ok_any = False
    for _ in range(count):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            ok_any = True
        time.sleep(0.03)
    return ok_any, frame

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()

def build_rectifier(cfg, size):
    K1 = mat(cfg["camera_matrix_l"])
    D1 = mat(cfg["dist_l"])
    K2 = mat(cfg["camera_matrix_r"])
    D2 = mat(cfg["dist_r"])
    R = mat(cfg["R"])
    T = mat(cfg["T"]).reshape(3, 1)
    image_size = (int(size[0]), int(size[1]))
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q

def disparity_points(left, right, cfg):
    h, w = left.shape[:2]
    map1x, map1y, map2x, map2y, Q = build_rectifier(cfg, (w, h))
    rl = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rr = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
    block = 5
    num_disp = max(64, min(256, ((w // 8) // 16 + 1) * 16))
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        disp12MaxDiff=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(gl, gr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    yy, xx = np.indices(disp.shape)
    mask = np.isfinite(pts).all(axis=2)
    mask &= disp > 2.0
    mask &= yy > int(h * 0.48)
    mask &= pts[:, :, 2] > 0.35
    mask &= pts[:, :, 2] < 8.0
    mask &= np.abs(pts[:, :, 0]) < 5.0
    mask &= np.abs(pts[:, :, 1]) < 5.0
    cloud = pts[mask].reshape(-1, 3)
    if len(cloud) > 30000:
        sel = np.random.choice(len(cloud), 30000, replace=False)
        cloud = cloud[sel]
    return cloud, disp, rl, rr

def fit_plane(points, iterations=900, threshold=0.035):
    if len(points) < 800:
        return None
    best_idx = None
    best_count = 0
    npts = len(points)
    for _ in range(iterations):
        ids = random.sample(range(npts), 3)
        p1, p2, p3 = points[ids]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        idx = np.where(dist < threshold)[0]
        c = len(idx)
        if c > best_count:
            best_count = c
            best_idx = idx
    if best_idx is None or best_count < 500:
        return None
    inliers = points[best_idx]
    center = inliers.mean(axis=0)
    A = inliers - center
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    d = -float(np.dot(n, center))
    dist = np.abs(points @ n + d)
    mad = float(np.median(np.abs(dist - np.median(dist))))
    ratio = float(len(inliers) / max(len(points), 1))
    angle_y = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(n, np.array([0.0, 1.0, 0.0])))), -1.0, 1.0))))
    return {
        "normal": n,
        "d": d,
        "center": center,
        "inliers": len(inliers),
        "total": len(points),
        "ratio": ratio,
        "mad": mad,
        "angle_to_camera_y_deg": angle_y,
    }

def process_rig(rig):
    cfg_path = BASE / rig["config"]
    if not cfg_path.exists():
        print(f"{rig['name']}: FEHLT config {cfg_path}")
        return
    cfg = load_yaml(cfg_path)
    if rig["kind"] == "sbs":
        cap = open_cap(rig["index"], 2560, 720)
        if not cap.isOpened():
            print(f"{rig['name']}: Kamera index {rig['index']} nicht offen")
            return
        ok, frame = read_good(cap)
        cap.release()
        if not ok:
            print(f"{rig['name']}: kein Bild")
            return
        left, right = split_sbs(frame)
    else:
        cap_l = open_cap(rig["left_index"], 1280, 800)
        cap_r = open_cap(rig["right_index"], 1280, 800)
        if not cap_l.isOpened() or not cap_r.isOpened():
            print(f"{rig['name']}: OV9281 Kameras nicht offen")
            cap_l.release()
            cap_r.release()
            return
        ok_l, left = read_good(cap_l)
        ok_r, right = read_good(cap_r)
        cap_l.release()
        cap_r.release()
        if not (ok_l and ok_r):
            print(f"{rig['name']}: kein Bild")
            return
    try:
        cloud, disp, rl, rr = disparity_points(left, right, cfg)
    except Exception as e:
        print(f"{rig['name']}: Fehler bei Disparity/Boden: {e}")
        return
    plane = fit_plane(cloud)
    print("")
    print(f"=== {rig['name']} ===")
    print(f"points={len(cloud)}")
    if plane is None:
        print("boden=NICHT STABIL ERKANNT")
        return
    n = plane["normal"]
    c = plane["center"]
    status = "OK" if plane["inliers"] >= 1500 and plane["ratio"] >= 0.12 and plane["mad"] <= 0.025 else "UNSICHER"
    print(f"boden={status}")
    print(f"inliers={plane['inliers']} total={plane['total']} ratio={plane['ratio']:.3f}")
    print(f"normal=({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
    print(f"center_m=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    print(f"mad_m={plane['mad']:.4f}")
    print(f"angle_to_camera_y_deg={plane['angle_to_camera_y_deg']:.2f}")

def main():
    print("ANTHRO3D Boden-Diagnose")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    for rig in RIGS:
        process_rig(rig)
    print("")
    print("Fertig.")

if __name__ == "__main__":
    main()
