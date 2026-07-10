import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "all_cameras_original_size"
OUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DELAY_SECONDS = 3.0
PIXEL_STRIDE = 3
HEADER_H = 44

ROLE_DIMS = {
    "OV9281 L": (1280, 800),
    "OV9281 R": (1280, 800),
    "ELP2": (2560, 720),
    "ELP1": (2560, 720),
}

RIGS = [
    ("OV9281", "dual", ("OV9281 L", "OV9281 R"), "stereo_config_ov9281.yaml"),
    ("ELP2", "sbs", ("ELP2",), "stereo_config.yaml"),
    ("ELP1", "sbs", ("ELP1",), "stereo_config_elp1.yaml"),
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

def depth_and_mask(left, right, cfg_file, stereo):
    size = (left.shape[1], left.shape[0])
    ml1, ml2, mr1, mr2, Q = maps_for(cfg_file, size)
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
    sampled = valid & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    d = disp.copy()
    d[~valid] = 0
    if valid.any():
        mx = np.percentile(d[valid], 95)
        if mx <= 0:
            mx = 1
    else:
        mx = 1
    depth_u8 = np.clip(d / mx * 255, 0, 255).astype(np.uint8)
    depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    mask_u8 = np.zeros_like(depth_u8)
    mask_u8[sampled] = 255
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    count = int(sampled.sum())
    if valid.any():
        z_med = float(np.median(z[valid]) * 100)
    else:
        z_med = 0.0
    return rect_l, rect_r, depth_col, mask_bgr, count, z_med

def tile_with_header(img, text):
    header = np.zeros((HEADER_H, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return np.vstack([header, img])

def make_rig_row(rig_name, left, right, depth, mask, count, z_med):
    h = left.shape[0]
    w = left.shape[1]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    t1 = tile_with_header(left, f"{rig_name} LEFT ORIGINAL {w}x{h}")
    t2 = tile_with_header(right, f"{rig_name} RIGHT ORIGINAL {w}x{h}")
    t3 = tile_with_header(depth, f"{rig_name} DEPTH points={count} z={z_med:.1f}cm")
    t4 = tile_with_header(mask, f"{rig_name} VALID 3D POINTS")
    return np.hstack([t1, t2, t3, t4])

def build_dashboard(rows, pending_snapshot_at):
    if not rows:
        return np.zeros((720, 1440, 3), dtype=np.uint8)
    max_w = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
    body = np.vstack(padded)
    header = np.zeros((HEADER_H, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "ANTHRO3D ORIGINAL SIZE LIVE: OV9281 + ELP2 + ELP1   SPACE=Screenshot in 3s   q=Ende", (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if pending_snapshot_at is not None:
        remaining = max(0.0, pending_snapshot_at - time.time())
        cv2.putText(header, f"SHOT IN {remaining:.1f}s", (body.shape[1] - 420, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    return np.vstack([header, body])

def main():
    roles = camera_roles()
    caps = open_caps(roles)
    if not caps:
        raise RuntimeError("Keine Kamera geoeffnet")
    stereo_by_rig = {rig[0]: make_stereo() for rig in RIGS}
    pending_snapshot_at = None
    for _ in range(20):
        read_frames(caps)
        time.sleep(0.03)
    cv2.namedWindow("ANTHRO3D Originalgroesse Live - q beendet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANTHRO3D Originalgroesse Live - q beendet", 1800, 950)
    try:
        while True:
            frames = read_frames(caps)
            rows = []
            for rig_name, mode, rig_roles, cfg_file in RIGS:
                if mode == "dual":
                    l_name, r_name = rig_roles
                    if l_name not in frames or r_name not in frames:
                        continue
                    left = frames[l_name]
                    right = frames[r_name]
                else:
                    role = rig_roles[0]
                    if role not in frames:
                        continue
                    left, right = split_sbs(frames[role])
                rect_l, rect_r, depth, mask, count, z_med = depth_and_mask(left, right, cfg_file, stereo_by_rig[rig_name])
                rows.append(make_rig_row(rig_name, rect_l, rect_r, depth, mask, count, z_med))
                print(f"{rig_name}: points={count} z={z_med:.1f}cm", end=" | ")
            dash = build_dashboard(rows, pending_snapshot_at)
            if pending_snapshot_at is not None and time.time() >= pending_snapshot_at:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUT / f"all_cameras_original_size_{ts}.png"
                cv2.imwrite(str(path), dash)
                print("")
                print(f"Originalgroesse-Snapshot gespeichert: {path}")
                pending_snapshot_at = None
            preview_w = min(1800, dash.shape[1])
            preview_h = int(dash.shape[0] * preview_w / dash.shape[1])
            preview = cv2.resize(dash, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.imshow("ANTHRO3D Originalgroesse Live - q beendet", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                pending_snapshot_at = time.time() + SNAPSHOT_DELAY_SECONDS
                print("")
                print("Screenshot in 3 Sekunden...")
            if key == ord("q") or key == 27:
                break
            print("", end="\r")
    finally:
        release_caps(caps)
        cv2.destroyAllWindows()
        print("")
        print("Fertig.")

if __name__ == "__main__":
    main()
