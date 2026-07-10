import cv2
import yaml
import time
import numpy as np
import open3d as o3d
from pathlib import Path

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "ov9281_scanbox"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1280
HEIGHT = 800
WARMUP_FRAMES = 25
CAPTURE_FRAMES = 8
COUNTDOWN_SECONDS = 5
PIXEL_STRIDE = 2
MAX_POINTS = 900000
VOXEL_SIZE_M = 0.006

SCANBOX = {
    "x_min": -0.80,
    "x_max": 0.80,
    "y_min": -1.50,
    "y_max": 0.70,
    "z_min": 0.45,
    "z_max": 1.45,
}

MAP_CACHE = {}

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def camera_roles():
    cfg = load_yaml(BASE / "config.yaml")
    roles = {}
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled"):
            roles[cam.get("name")] = int(cam.get("device_index"))
    return roles

def open_cap(name, index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"{name} Index {index} konnte nicht geoeffnet werden")
    return cap

def maps_for(size):
    key = size
    if key in MAP_CACHE:
        return MAP_CACHE[key]
    cfg = load_yaml(BASE / "stereo_config_ov9281.yaml")
    K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
    d_l = np.array(cfg["dist_l"], dtype=np.float64)
    K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
    d_r = np.array(cfg["dist_r"], dtype=np.float64)
    R = np.array(cfg["R"], dtype=np.float64)
    T = np.array(cfg["T"], dtype=np.float64).reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, d_l, K_r, d_r, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ml1, ml2 = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
    mr1, mr2 = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
    MAP_CACHE[key] = (ml1, ml2, mr1, mr2, Q)
    return MAP_CACHE[key]

def compute_points(left, right):
    size = (left.shape[1], left.shape[0])
    ml1, ml2, mr1, mr2, Q = maps_for(size)
    rect_l = cv2.remap(left, ml1, ml2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
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
    disp = stereo.compute(gl, gr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    valid = np.isfinite(pts).all(axis=2) & (disp > 1.0)
    z = pts[:, :, 2]
    valid = valid & (z > 0.25) & (z < 5.0)
    yy, xx = np.indices(valid.shape)
    valid = valid & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    crop = (
        valid
        & (pts[:, :, 0] >= SCANBOX["x_min"])
        & (pts[:, :, 0] <= SCANBOX["x_max"])
        & (pts[:, :, 1] >= SCANBOX["y_min"])
        & (pts[:, :, 1] <= SCANBOX["y_max"])
        & (pts[:, :, 2] >= SCANBOX["z_min"])
        & (pts[:, :, 2] <= SCANBOX["z_max"])
    )
    xyz = pts[crop].astype(np.float64)
    rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)[crop].astype(np.float64) / 255.0
    return xyz, rgb

def pcd_from_arrays(points, colors):
    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def countdown():
    print("OV9281 Scanbox-Aufnahme in 5 Sekunden. Person in den Messbereich stellen.")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme startet.")
    print("")

def main():
    print("")
    print("ANTHRO3D OV9281 Scanbox-Punktwolke ohne Personenmaske mit vorderstem 3D-Cluster")
    print("Ausgabe:", OUT)
    print("")
    roles = camera_roles()
    if "OV9281 L" not in roles or "OV9281 R" not in roles:
        raise RuntimeError("OV9281 L/R fehlen in config.yaml")
    print(f"OV9281 L Index {roles['OV9281 L']}")
    print(f"OV9281 R Index {roles['OV9281 R']}")
    cap_l = open_cap("OV9281 L", roles["OV9281 L"])
    cap_r = open_cap("OV9281 R", roles["OV9281 R"])
    try:
        for _ in range(WARMUP_FRAMES):
            cap_l.read()
            cap_r.read()
        countdown()
        all_points = []
        all_colors = []
        for i in range(CAPTURE_FRAMES):
            ok_l, left = cap_l.read()
            ok_r, right = cap_r.read()
            if not ok_l or not ok_r or left is None or right is None:
                print(f"Frame {i + 1}: FEHLER")
                continue
            pts, cols = compute_points(left, right)
            print(f"Frame {i + 1}: scanbox_points={len(pts)}")
            if len(pts) > 0:
                all_points.append(pts)
                all_colors.append(cols)
            time.sleep(0.08)
    finally:
        cap_l.release()
        cap_r.release()
    if not all_points:
        raise RuntimeError("Keine Punkte in der Scanbox")
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), MAX_POINTS, replace=False)
        points = points[idx]
        colors = colors[idx]
    pcd = pcd_from_arrays(points, colors)
    raw_path = OUT / "ov9281_scanbox_raw.ply"
    o3d.io.write_point_cloud(str(raw_path), pcd, write_ascii=False, compressed=False)
    print("")
    print(f"Raw gespeichert: {raw_path}")
    print(f"Raw Punkte: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
    if len(pcd.points) > 200:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    labels = np.array(pcd.cluster_dbscan(eps=0.035, min_points=30, print_progress=True))
    if labels.size > 0 and labels.max() >= 0:
        pts_arr = np.asarray(pcd.points)
        candidates = []
        for label in range(int(labels.max()) + 1):
            idx = np.where(labels == label)[0]
            if len(idx) < 2500:
                continue
            z_med = float(np.median(pts_arr[idx, 2]))
            x_med = float(np.median(pts_arr[idx, 0]))
            y_med = float(np.median(pts_arr[idx, 1]))
            candidates.append((z_med, -len(idx), label, len(idx), x_med, y_med))
        if candidates:
            candidates.sort()
            z_med, neg_count, best_label, count, x_med, y_med = candidates[0]
            keep = np.where(labels == best_label)[0]
            pcd = pcd.select_by_index(keep)
            print(f"Vorderster 3D-Cluster: label={best_label} punkte={len(pcd.points)} z_median_cm={z_med*100:.1f} x_median_cm={x_med*100:.1f} y_median_cm={y_med*100:.1f}")
        else:
            print("WARNUNG: Keine ausreichend grossen Cluster gefunden, verwende bereinigte Punktwolke")
    else:
        print("WARNUNG: Kein stabiler 3D-Cluster gefunden, verwende bereinigte Punktwolke")
    clean_path = OUT / "ov9281_scanbox_front_cluster_clean.ply"
    o3d.io.write_point_cloud(str(clean_path), pcd, write_ascii=False, compressed=False)
    pts_np = np.asarray(pcd.points)
    print(f"Clean gespeichert: {clean_path}")
    print(f"Clean Punkte: {len(pcd.points)}")
    if len(pts_np) > 0:
        print(f"x_cm median={np.median(pts_np[:,0])*100:.1f} min={np.percentile(pts_np[:,0],5)*100:.1f} max={np.percentile(pts_np[:,0],95)*100:.1f}")
        print(f"y_cm median={np.median(pts_np[:,1])*100:.1f} min={np.percentile(pts_np[:,1],5)*100:.1f} max={np.percentile(pts_np[:,1],95)*100:.1f}")
        print(f"z_cm median={np.median(pts_np[:,2])*100:.1f} min={np.percentile(pts_np[:,2],5)*100:.1f} max={np.percentile(pts_np[:,2],95)*100:.1f}")
    print("")
    print("Open3D-Viewer startet. Fenster drehen/pruefen, dann schliessen.")
    o3d.visualization.draw_geometries([pcd], window_name="ANTHRO3D OV9281 Scanbox Clean Pointcloud")

if __name__ == "__main__":
    main()
