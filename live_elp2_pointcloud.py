import cv2
import yaml
import time
import numpy as np
import open3d as o3d
from pathlib import Path

BASE = Path.home() / "anthro3d"
CONFIG = BASE / "config.yaml"
STEREO_CONFIG = BASE / "stereo_config.yaml"

FRAME_W = 2560
FRAME_H = 720
PIXEL_STRIDE = 4
VOXEL_SIZE_M = 0.01
MAX_POINTS = 120000
UPDATE_DELAY = 0.03

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def camera_roles():
    cfg = load_yaml(CONFIG)
    roles = {}
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled"):
            roles[cam.get("name")] = int(cam.get("device_index"))
    return roles

def split_sbs(frame):
    h, w = frame.shape[:2]
    m = w // 2
    return frame[:, :m].copy(), frame[:, m:].copy()

def make_maps(size):
    cfg = load_yaml(STEREO_CONFIG)
    K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
    d_l = np.array(cfg["dist_l"], dtype=np.float64)
    K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
    d_r = np.array(cfg["dist_r"], dtype=np.float64)
    R = np.array(cfg["R"], dtype=np.float64)
    T = np.array(cfg["T"], dtype=np.float64).reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, d_l, K_r, d_r, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ml1, ml2 = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
    mr1, mr2 = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
    return ml1, ml2, mr1, mr2, Q

def compute_cloud(frame, maps, stereo):
    left, right = split_sbs(frame)
    ml1, ml2, mr1, mr2, Q = maps
    rect_l = cv2.remap(left, ml1, ml2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gl, gr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    valid = np.isfinite(pts).all(axis=2) & (disp > 1.0)
    z = pts[:, :, 2]
    valid = valid & (z > 0.35) & (z < 4.00)
    yy, xx = np.indices(valid.shape)
    valid = valid & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    xyz = pts[valid].astype(np.float64)
    rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    if len(xyz) > MAX_POINTS:
        idx = np.random.choice(len(xyz), MAX_POINTS, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    return xyz, rgb

def main():
    print("")
    print("ANTHRO3D Live-Punktewolke ELP2")
    roles = camera_roles()
    if "ELP2" not in roles:
        raise RuntimeError("ELP2 fehlt in config.yaml")
    idx = roles["ELP2"]
    print(f"ELP2 Index {idx}")
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"ELP2 Index {idx} konnte nicht geoeffnet werden")
    for _ in range(20):
        cap.read()
        time.sleep(0.03)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Kein ELP2 Frame")
    left, right = split_sbs(frame)
    maps = make_maps((left.shape[1], left.shape[0]))
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ANTHRO3D ELP2 Live Pointcloud", width=1400, height=900)
    vis.add_geometry(pcd)
    first = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            xyz, rgb = compute_cloud(frame, maps, stereo)
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            if len(pcd.points) > 0:
                pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
            if first:
                vis.clear_geometries()
                vis.add_geometry(pcd)
                first = False
            else:
                vis.update_geometry(pcd)
            print(f"ELP2 live_points={len(pcd.points)}", end="\r")
            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
            time.sleep(UPDATE_DELAY)
    finally:
        cap.release()
        vis.destroy_window()
        print("")
        print("Fertig.")

if __name__ == "__main__":
    main()
