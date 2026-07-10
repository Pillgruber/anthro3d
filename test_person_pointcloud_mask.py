import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
COUNTDOWN_SECONDS = 5
Z_DELTA_M = 0.25
MIN_COMPONENT_AREA = 800

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

def print_roles(roles):
    print("Kamera-Rollen aus config.yaml")
    print("=============================")
    for name, idx in roles.items():
        print(f"{name}: Index {idx}")
    print("")

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
    return disp, pts, valid, rl

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
    disp, pts, valid, rect_left_gray = compute_depth(left, right, cfg_file)
    return disp, pts, valid, left, right, rect_left_gray

def capture_depths(caps):
    frames = read_frames(caps)
    out = {}
    for rig_name, mode, rig_roles, cfg_file in RIGS:
        r = rig_depth(rig_name, mode, rig_roles, cfg_file, frames)
        if r is None:
            print(f"{rig_name}: keine Frames")
            continue
        disp, pts, valid, left, right, rect_left_gray = r
        out[rig_name] = {
            "disp": disp,
            "pts": pts,
            "valid": valid,
            "left": left,
            "right": right,
            "rect_left_gray": rect_left_gray,
        }
    return frames, out

def countdown(text):
    print(text)
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme startet.")
    print("")

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

def largest_component(mask):
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
        return m.astype(bool)
    return labels == best

def clean_mask(mask):
    m = mask.astype(np.uint8) * 255
    k1 = np.ones((5, 5), np.uint8)
    k2 = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    m = m > 0
    m = largest_component(m)
    return m

def foreground_mask(base, current):
    zb = base["pts"][:, :, 2]
    zc = current["pts"][:, :, 2]
    vb = base["valid"]
    vc = current["valid"]
    both = vb & vc & np.isfinite(zb) & np.isfinite(zc)
    dz = np.zeros_like(zc, dtype=np.float32)
    dz[both] = zb[both] - zc[both]
    fg = both & (dz > Z_DELTA_M)
    fg = clean_mask(fg)
    return fg

def summarize_mask(name, current, mask):
    valid = current["valid"] & mask
    count = int(valid.sum())
    if count < 100:
        print(f"{name}: mask_points={count} STATUS=ZU_WENIG")
        return
    pts = current["pts"][valid]
    x = pts[:, 0] * 100
    y = pts[:, 1] * 100
    z = pts[:, 2] * 100
    print(f"{name}: mask_points={count}")
    print(f"  x_cm median={np.median(x):.1f} min={np.percentile(x,5):.1f} max={np.percentile(x,95):.1f}")
    print(f"  y_cm median={np.median(y):.1f} min={np.percentile(y,5):.1f} max={np.percentile(y,95):.1f}")
    print(f"  z_cm median={np.median(z):.1f} min={np.percentile(z,5):.1f} max={np.percentile(z,95):.1f}")

def show_mask_result(name, base, current):
    mask = foreground_mask(base, current)
    summarize_mask(name, current, mask)
    dv = depth_vis(current["disp"], current["valid"])
    mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (dv.shape[1], dv.shape[0]))
    overlay = dv.copy()
    overlay[mask_resized > 0] = (0, 255, 255)
    masked = np.zeros_like(dv)
    masked[mask_resized > 0] = dv[mask_resized > 0]
    cv2.imshow(name + " person mask overlay", cv2.resize(overlay, (960, 540)))
    cv2.imshow(name + " person masked depth", cv2.resize(masked, (960, 540)))

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

def live_mask_loop(caps, roles, base_depths):
    print("")
    print("Live-Personenmaske gestartet. q im Fenster beendet.")
    while True:
        frames, depths = capture_depths(caps)
        if frames:
            cv2.imshow("ACTIVE CAMERA RAW VIEW", raw_grid(frames, roles))
        for name in depths.keys():
            if name not in base_depths:
                continue
            mask = foreground_mask(base_depths[name], depths[name])
            dv = depth_vis(depths[name]["disp"], depths[name]["valid"])
            mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (dv.shape[1], dv.shape[0]))
            overlay = dv.copy()
            overlay[mask_resized > 0] = (0, 255, 255)
            cv2.imshow(name + " live person mask", cv2.resize(overlay, (960, 540)))
        key = cv2.waitKey(80) & 0xFF
        if key == ord("q"):
            break

def main():
    print("")
    print("ANTHRO3D Personenmaske auf Punktewolke")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    print("")
    roles = camera_roles()
    print_roles(roles)
    caps = open_caps(roles)
    if not caps:
        raise RuntimeError("Keine Kameras geoeffnet")
    try:
        countdown("Leerraum-Aufnahme in 5 Sekunden. Bitte aus dem Messbereich gehen.")
        base_frames, base_depths = capture_depths(caps)
        print("Leerraum gespeichert.")
        print("")
        countdown("Person-Aufnahme in 5 Sekunden. Jetzt zwischen die Kameras stellen.")
        person_frames, person_depths = capture_depths(caps)
        print("Person aufgenommen.")
        print("")
        for name in person_depths.keys():
            if name in base_depths:
                show_mask_result(name, base_depths[name], person_depths[name])
        print("")
        print("Maskenfenster pruefen. q startet danach Live-Maske.")
        while True:
            key = cv2.waitKey(80) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        live_mask_loop(caps, roles, base_depths)
    finally:
        release_caps(caps)
        cv2.destroyAllWindows()
    print("Fertig.")

if __name__ == "__main__":
    main()
