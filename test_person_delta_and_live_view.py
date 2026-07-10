import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
COUNTDOWN_SECONDS = 5
FOREGROUND_Z_DELTA_M = 0.25

ROLE_DIMS = {
    "OV9281 L": (1280, 800),
    "OV9281 R": (1280, 800),
    "ELP2": (2560, 720),
    "ELP1": (2560, 720),
}

RIGS = [
    ("OV9281", "dual", ("OV9281 L", "OV9281 R"), "stereo_config_ov9281.yaml", None),
    ("ELP2", "sbs", ("ELP2",), "stereo_config.yaml", "elp2"),
    ("ELP1", "sbs", ("ELP1",), "stereo_config_elp1.yaml", None),
]

MAP_CACHE = {}

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def camera_roles():
    cfg = load_yaml(BASE / "config.yaml")
    out = {}
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled"):
            out[cam.get("name")] = int(cam.get("device_index"))
    return out

def print_roles(roles):
    print("Kamera-Rollen aus config.yaml")
    print("=============================")
    for name, idx in roles.items():
        print(f"{name}: Index {idx}")
    print("")

def load_elp2_scale():
    p = BASE / "aruco_state" / "elp2_id3_id30_board_calibration.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text()) or {}
    v = d.get("scale_median")
    if v is None:
        return None
    return float(v)

def open_caps(roles):
    caps = {}
    for name, idx in roles.items():
        w, h = ROLE_DIMS.get(name, (1280, 720))
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if cap.isOpened():
            caps[name] = cap
        else:
            print(f"WARNUNG: {name} Index {idx} konnte nicht geoeffnet werden")
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

def compute_depth(left, right, cfg_file):
    size = (left.shape[1], left.shape[0])
    ml1, ml2, mr1, mr2, Q = maps_for(cfg_file, size)
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    rl = cv2.remap(gl, ml1, ml2, cv2.INTER_LINEAR)
    rr = cv2.remap(gr, mr1, mr2, cv2.INTER_LINEAR)
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
    disp = stereo.compute(rl, rr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    valid = np.isfinite(pts).all(axis=2) & (disp > 1.0)
    z = pts[:, :, 2]
    valid = valid & (z > 0.25) & (z < 5.0)
    return disp, pts, valid

def rig_depth(name, mode, roles, cfg_file, frames):
    if mode == "dual":
        l_name, r_name = roles
        if l_name not in frames or r_name not in frames:
            return None
        left = frames[l_name]
        right = frames[r_name]
    else:
        role = roles[0]
        if role not in frames:
            return None
        left, right = split_sbs(frames[role])
    disp, pts, valid = compute_depth(left, right, cfg_file)
    return disp, pts, valid, left, right

def depth_vis(disp, valid):
    if disp is None or valid is None or not valid.any():
        return np.zeros((540, 960, 3), dtype=np.uint8)
    d = disp.copy()
    d[~valid] = 0
    mx = np.percentile(d[valid], 95)
    if mx <= 0:
        mx = 1
    v = np.clip(d / mx * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)

def raw_grid(frames, roles):
    tiles = []
    for name in roles.keys():
        img = frames.get(name)
        if img is None:
            img = np.zeros((270, 480, 3), dtype=np.uint8)
        tile = cv2.resize(img, (480, 270))
        cv2.rectangle(tile, (0, 0), (480, 42), (0, 0, 0), -1)
        cv2.putText(tile, name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        tiles.append(tile)
    while len(tiles) % 2 != 0:
        tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    rows = []
    for i in range(0, len(tiles), 2):
        rows.append(np.hstack([tiles[i], tiles[i + 1]]))
    return np.vstack(rows)

def countdown(text):
    print(text)
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme.")
    print("")

def summarize_points(name, pts, valid, scale=None):
    count = int(valid.sum())
    if count < 100:
        print(f"{name}: points={count} STATUS=ZU_WENIG")
        return
    p = pts[valid]
    if scale is not None:
        p = p * scale
    x = p[:, 0] * 100
    y = p[:, 1] * 100
    z = p[:, 2] * 100
    print(f"{name}: points={count}")
    print(f"  x_cm median={np.median(x):.1f} min={np.percentile(x,5):.1f} max={np.percentile(x,95):.1f}")
    print(f"  y_cm median={np.median(y):.1f} min={np.percentile(y,5):.1f} max={np.percentile(y,95):.1f}")
    print(f"  z_cm median={np.median(z):.1f} min={np.percentile(z,5):.1f} max={np.percentile(z,95):.1f}")

def capture_all_depths(caps, roles, elp2_scale):
    frames = read_frames(caps)
    out = {}
    for rig_name, mode, rig_roles, cfg_file, scale_key in RIGS:
        r = rig_depth(rig_name, mode, rig_roles, cfg_file, frames)
        if r is None:
            print(f"{rig_name}: keine Frames")
            continue
        disp, pts, valid, left, right = r
        scale = elp2_scale if scale_key == "elp2" else None
        summarize_points(rig_name, pts, valid, scale=scale)
        out[rig_name] = {
            "disp": disp,
            "pts": pts,
            "valid": valid,
            "left": left,
            "right": right,
        }
    return frames, out

def show_before_after(before, after):
    for name in after.keys():
        if name not in before:
            continue
        b = before[name]
        a = after[name]
        za = a["pts"][:, :, 2]
        zb = b["pts"][:, :, 2]
        va = a["valid"]
        vb = b["valid"]
        both = va & vb & np.isfinite(za) & np.isfinite(zb)
        dz = np.zeros_like(za, dtype=np.float32)
        dz[both] = zb[both] - za[both]
        fg = both & (dz > FOREGROUND_Z_DELTA_M)
        print(f"{name}: foreground_points={int(fg.sum())}")
        after_vis = depth_vis(a["disp"], va)
        fg_vis = after_vis.copy()
        mask = cv2.resize(fg.astype(np.uint8) * 255, (after_vis.shape[1], after_vis.shape[0]))
        fg_vis[mask > 0] = (0, 255, 255)
        cv2.imshow(name + " depth after", cv2.resize(after_vis, (960, 540)))
        cv2.imshow(name + " foreground delta", cv2.resize(fg_vis, (960, 540)))

def live_loop(caps, roles, elp2_scale):
    print("")
    print("Live-Ansicht gestartet. q im Fenster beendet.")
    while True:
        frames = read_frames(caps)
        if frames:
            cv2.imshow("ACTIVE CAMERA RAW VIEW", raw_grid(frames, roles))
        for rig_name, mode, rig_roles, cfg_file, scale_key in RIGS:
            r = rig_depth(rig_name, mode, rig_roles, cfg_file, frames)
            if r is None:
                continue
            disp, pts, valid, left, right = r
            vis = depth_vis(disp, valid)
            cv2.imshow(rig_name + " live depth", cv2.resize(vis, (960, 540)))
        key = cv2.waitKey(80) & 0xFF
        if key == ord("q"):
            break

def main():
    print("")
    print("ANTHRO3D Person-Differenz und Live-Punktewolkenansicht")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    print("")
    roles = camera_roles()
    print_roles(roles)
    elp2_scale = load_elp2_scale()
    if elp2_scale is not None:
        print(f"ELP2 lokale Skalenkorrektur geladen: {elp2_scale:.4f}")
        print("")
    caps = open_caps(roles)
    if not caps:
        raise RuntimeError("Keine Kameras geoeffnet")
    try:
        countdown("Leerraum-Aufnahme in 5 Sekunden. Bitte aus dem Messbereich gehen.")
        print("Leerraum")
        base_frames, base_depths = capture_all_depths(caps, roles, elp2_scale)
        print("")
        countdown("Person-Aufnahme in 5 Sekunden. Jetzt zwischen die Kameras stellen.")
        print("Mit Person")
        person_frames, person_depths = capture_all_depths(caps, roles, elp2_scale)
        print("")
        show_before_after(base_depths, person_depths)
        print("")
        print("Differenzfenster pruefen. q startet danach die Live-Ansicht.")
        while True:
            key = cv2.waitKey(80) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        live_loop(caps, roles, elp2_scale)
    finally:
        release_caps(caps)
        cv2.destroyAllWindows()
    print("Fertig.")

if __name__ == "__main__":
    main()
