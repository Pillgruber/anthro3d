import cv2
import yaml
import time
import numpy as np
import open3d as o3d
from pathlib import Path
from datetime import datetime

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "all_rig_pointcloud_snapshot"
OUT.mkdir(parents=True, exist_ok=True)

COUNTDOWN_SECONDS = 5
WARMUP_FRAMES = 25
PIXEL_STRIDE = 3
VOXEL_SIZE_M = 0.008
SCREENSHOT_DELAY_SECONDS = 5

ROLE_DIMS = {
    "OV9281 L": (1280, 800),
    "OV9281 R": (1280, 800),
    "ELP2": (2560, 720),
    "ELP1": (2560, 720),
}

RIGS = [
    ("OV9281", "dual", ("OV9281 L", "OV9281 R"), "stereo_config_ov9281.yaml", (0.0, 0.0, 0.0)),
    ("ELP2", "sbs", ("ELP2",), "stereo_config.yaml", (1.6, 0.0, 0.0)),
    ("ELP1", "sbs", ("ELP1",), "stereo_config_elp1.yaml", (3.2, 0.0, 0.0)),
]

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

def open_caps(roles):
    caps = {}
    for name, idx in roles.items():
        w, h = ROLE_DIMS.get(name, (1280, 720))
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if cap.isOpened():
            caps[name] = cap
            print(f"{name}: offen index={idx}")
        else:
            print(f"{name}: FEHLER index={idx}")
    return caps

def release_caps(caps):
    for cap in caps.values():
        cap.release()

def read_frames(caps):
    frames = {}
    for name, cap in caps.items():
        ok, frame = cap.read()
        if ok and frame is not None:
            frames[name] = frame
    return frames

def split_sbs(frame):
    h, w = frame.shape[:2]
    m = w // 2
    return frame[:, :m].copy(), frame[:, m:].copy()

def maps_for(cfg_file, size):
    key = (cfg_file, size)
    if key in MAP_CACHE:
        return MAP_CACHE[key]
    cfg = load_yaml(BASE / cfg_file)
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

def compute_cloud(left, right, cfg_file):
    size = (left.shape[1], left.shape[0])
    ml1, ml2, mr1, mr2, Q = maps_for(cfg_file, size)
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
    valid = valid & (z > 0.35) & (z < 4.00)
    yy, xx = np.indices(valid.shape)
    valid = valid & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
    xyz = pts[valid].astype(np.float64)
    col = rgb[valid].astype(np.float64) / 255.0
    return xyz, col

def pcd_from_arrays(points, colors, offset):
    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        points = points.copy()
        points[:, 0] += offset[0]
        points[:, 1] += offset[1]
        points[:, 2] += offset[2]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def countdown():
    print("")
    print("Punktewolken-Aufnahme aller Kameras in 5 Sekunden.")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme startet.")
    print("")

def main():
    print("")
    print("ANTHRO3D Punktewolke aller Kamera-Rigs mit Screenshot")
    print("Ausgabe:", OUT)
    print("")
    roles = camera_roles()
    caps = open_caps(roles)
    if not caps:
        raise RuntimeError("Keine Kameras geoeffnet")
    try:
        for _ in range(WARMUP_FRAMES):
            read_frames(caps)
            time.sleep(0.03)
        countdown()
        frames = read_frames(caps)
    finally:
        release_caps(caps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    geometries = []

    for rig_name, mode, rig_roles, cfg_file, offset in RIGS:
        if mode == "dual":
            l_name, r_name = rig_roles
            if l_name not in frames or r_name not in frames:
                print(f"{rig_name}: Frames fehlen")
                continue
            left = frames[l_name]
            right = frames[r_name]
        else:
            role = rig_roles[0]
            if role not in frames:
                print(f"{rig_name}: Frame fehlt")
                continue
            left, right = split_sbs(frames[role])

        points, colors = compute_cloud(left, right, cfg_file)
        pcd = pcd_from_arrays(points, colors, offset)
        raw_path = OUT / f"{timestamp}_{rig_name.lower()}_raw.ply"
        o3d.io.write_point_cloud(str(raw_path), pcd, write_ascii=False, compressed=False)
        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)
            if len(pcd.points) > 200:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.2)
        clean_path = OUT / f"{timestamp}_{rig_name.lower()}_clean.ply"
        o3d.io.write_point_cloud(str(clean_path), pcd, write_ascii=False, compressed=False)
        print(f"{rig_name}: raw={len(points)} clean={len(pcd.points)}")
        print(f"{rig_name}: gespeichert {clean_path}")
        geometries.append(pcd)

    if not geometries:
        raise RuntimeError("Keine Punktwolken erzeugt")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ANTHRO3D alle Kamera-Punktwolken - Screenshot nach 5 Sekunden", width=1600, height=1000)
    for g in geometries:
        vis.add_geometry(g)

    start = time.time()
    screenshot_done = False
    screenshot_path = OUT / f"{timestamp}_all_rigs_view.png"

    while True:
        vis.poll_events()
        vis.update_renderer()
        if not screenshot_done and time.time() - start >= SCREENSHOT_DELAY_SECONDS:
            vis.capture_screen_image(str(screenshot_path), do_render=True)
            print(f"Screenshot gespeichert: {screenshot_path}")
            screenshot_done = True
            print("Viewer bleibt offen. Fenster schliessen zum Beenden.")
        if not vis.poll_events():
            break
        time.sleep(0.03)

    vis.destroy_window()
    print("Fertig.")

if __name__ == "__main__":
    main()
