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
PIXEL_STRIDE = 5
VOXEL_SIZE_M = 0.012
MAX_POINTS = 90000

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def get_elp2_index():
    cfg = load_yaml(CONFIG)
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled") and cam.get("name") == "ELP2":
            return int(cam.get("device_index"))
    raise RuntimeError("ELP2 nicht in config.yaml gefunden")

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

def make_stereo():
    return cv2.StereoSGBM_create(
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

def compute_cloud(left, right, maps, stereo):
    ml1, ml2, mr1, mr2, Q = maps
    rect_l = cv2.remap(left, ml1, ml2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gl, gr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    valid = np.isfinite(pts).all(axis=2) & (disp > 1.0)
    z = pts[:, :, 2]
    valid = valid & (z > 0.35) & (z < 4.0)
    yy, xx = np.indices(valid.shape)
    valid = valid & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    xyz = pts[valid].astype(np.float64)
    rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    if len(xyz) > MAX_POINTS:
        idx = np.random.choice(len(xyz), MAX_POINTS, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    return xyz, rgb

def camera_view(left, right):
    l = cv2.resize(left, (640, 360))
    r = cv2.resize(right, (640, 360))
    cv2.rectangle(l, (0, 0), (640, 42), (0, 0, 0), -1)
    cv2.rectangle(r, (0, 0), (640, 42), (0, 0, 0), -1)
    cv2.putText(l, "ELP2 LEFT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(r, "ELP2 RIGHT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return np.hstack([l, r])

def main():
    idx = get_elp2_index()
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
    stereo = make_stereo()

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ANTHRO3D ELP2 Live-Punktewolke", width=900, height=800, left=1320, top=60)
    vis.add_geometry(pcd)

    cv2.namedWindow("ANTHRO3D ELP2 Live-Kamera - q beendet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANTHRO3D ELP2 Live-Kamera - q beendet", 1280, 360)
    cv2.moveWindow("ANTHRO3D ELP2 Live-Kamera - q beendet", 20, 60)

    first = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            left, right = split_sbs(frame)
            view = camera_view(left, right)
            cv2.imshow("ANTHRO3D ELP2 Live-Kamera - q beendet", view)

            xyz, rgb = compute_cloud(left, right, maps, stereo)
            temp = o3d.geometry.PointCloud()
            temp.points = o3d.utility.Vector3dVector(xyz)
            temp.colors = o3d.utility.Vector3dVector(rgb)

            if len(temp.points) > 0:
                temp = temp.voxel_down_sample(VOXEL_SIZE_M)

            pcd.points = temp.points
            pcd.colors = temp.colors

            if first:
                vis.update_geometry(pcd)
                vis.reset_view_point(True)
                first = False
            else:
                vis.update_geometry(pcd)

            alive = vis.poll_events()
            vis.update_renderer()

            print(f"ELP2 live_points={len(pcd.points)}", end="\r")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not alive:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("")
        print("Fertig.")

if __name__ == "__main__":
    main()
