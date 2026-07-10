import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "all_cameras_background_delta"
OUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DELAY_SECONDS = 3.0
CAPTURE_FRAMES = 7
WARMUP_FRAMES = 25
PIXEL_STRIDE = 2
HEADER_H = 44
MIN_DELTA_M = 0.20
MIN_Z_M = 0.35
MAX_Z_M = 4.0

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

def compute_depth(left, right, cfg_file, stereo):
    size = (left.shape[1], left.shape[0])
    ml1, ml2, mr1, mr2, Q = maps_for(cfg_file, size)
    rect_l = cv2.remap(left, ml1, ml2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gl, gr).astype(np.float32) / 16.0
    pts = cv2.reprojectImageTo3D(disp, Q)
    z = pts[:, :, 2].astype(np.float32)
    valid = np.isfinite(pts).all(axis=2) & np.isfinite(z) & (disp > 1.0) & (z > MIN_Z_M) & (z < MAX_Z_M)
    return rect_l, rect_r, z, disp, valid

def rig_frame_from_frames(rig_name, mode, rig_roles, frames):
    if mode == "dual":
        l_name, r_name = rig_roles
        if l_name not in frames or r_name not in frames:
            return None, None
        return frames[l_name], frames[r_name]
    role = rig_roles[0]
    if role not in frames:
        return None, None
    return split_sbs(frames[role])

def capture_depth_set(caps, stereo_by_rig):
    result = {}
    for _ in range(3):
        read_frames(caps)
        time.sleep(0.03)
    z_stack = {rig[0]: [] for rig in RIGS}
    valid_stack = {rig[0]: [] for rig in RIGS}
    last_data = {}
    for i in range(CAPTURE_FRAMES):
        frames = read_frames(caps)
        for rig_name, mode, rig_roles, cfg_file in RIGS:
            left, right = rig_frame_from_frames(rig_name, mode, rig_roles, frames)
            if left is None or right is None:
                continue
            rect_l, rect_r, z, disp, valid = compute_depth(left, right, cfg_file, stereo_by_rig[rig_name])
            z_stack[rig_name].append(z)
            valid_stack[rig_name].append(valid)
            last_data[rig_name] = {
                "left": rect_l,
                "right": rect_r,
                "disp": disp,
                "valid": valid,
                "z": z,
            }
        print(f"Frame {i + 1}/{CAPTURE_FRAMES}")
        time.sleep(0.08)
    for rig_name in z_stack:
        if not z_stack[rig_name]:
            continue
        zs = np.stack(z_stack[rig_name], axis=0)
        vs = np.stack(valid_stack[rig_name], axis=0)
        z_masked = np.where(vs, zs, np.nan)
        z_med = np.nanmedian(z_masked, axis=0).astype(np.float32)
        valid_med = np.isfinite(z_med)
        data = last_data[rig_name]
        data["z_med"] = z_med
        data["valid_med"] = valid_med
        result[rig_name] = data
    return result

def z_to_depth_color(z, valid):
    z_show = np.zeros_like(z, dtype=np.float32)
    z_show[valid] = z[valid]
    if valid.any():
        zmin = np.percentile(z[valid], 5)
        zmax = np.percentile(z[valid], 95)
        if zmax <= zmin:
            zmax = zmin + 0.1
    else:
        zmin = 0.5
        zmax = 3.0
    inv = np.zeros_like(z_show, dtype=np.float32)
    inv[valid] = 1.0 - np.clip((z_show[valid] - zmin) / (zmax - zmin), 0, 1)
    u8 = np.clip(inv * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    col[~valid] = 0
    return col

def delta_to_color(delta, mask):
    d = np.zeros_like(delta, dtype=np.float32)
    d[mask] = np.clip(delta[mask], 0, 0.8)
    u8 = np.clip(d / 0.8 * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    col[~mask] = 0
    return col

def clean_mask(mask):
    m = mask.astype(np.uint8) * 255
    k1 = np.ones((3, 3), np.uint8)
    k2 = np.ones((9, 9), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return m > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] < 500:
        return m > 0
    return labels == best

def make_candidate(base, person):
    b_z = base["z_med"]
    p_z = person["z_med"]
    b_valid = base["valid_med"]
    p_valid = person["valid_med"]
    delta = b_z - p_z
    yy, xx = np.indices(p_valid.shape)
    sampled = ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
    raw = b_valid & p_valid & sampled & (delta > MIN_DELTA_M)
    clean = clean_mask(raw)
    return delta, raw, clean

def tile_with_header(img, text):
    header = np.zeros((HEADER_H, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    return np.vstack([header, img])

def make_rig_row(rig_name, base, person):
    left = person["left"]
    right = person["right"]
    depth = z_to_depth_color(person["z_med"], person["valid_med"])
    delta, raw, clean = make_candidate(base, person)
    diff = delta_to_color(delta, raw)
    mask = np.zeros_like(depth)
    mask[clean] = (255, 255, 255)
    h, w = left.shape[:2]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    diff = cv2.resize(diff, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    clean_count = int(clean.sum())
    raw_count = int(raw.sum())
    if clean.any():
        z_med = float(np.median(person["z_med"][clean]) * 100)
        d_med = float(np.median(delta[clean]) * 100)
    else:
        z_med = 0.0
        d_med = 0.0
    t1 = tile_with_header(left, f"{rig_name} PERSON LEFT")
    t2 = tile_with_header(right, f"{rig_name} PERSON RIGHT")
    t3 = tile_with_header(depth, f"{rig_name} CURRENT DEPTH")
    t4 = tile_with_header(diff, f"{rig_name} DELTA raw={raw_count} d={d_med:.1f}cm")
    t5 = tile_with_header(mask, f"{rig_name} CANDIDATE clean={clean_count} z={z_med:.1f}cm")
    return np.hstack([t1, t2, t3, t4, t5]), clean_count, raw_count, z_med, d_med

def build_dashboard(rows, pending_snapshot_at):
    max_w = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
    body = np.vstack(padded)
    header = np.zeros((HEADER_H, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "ANTHRO3D BACKGROUND DELTA: Raw | Depth | Difference | Candidate     SPACE=Snapshot in 3s     q=Ende", (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    if pending_snapshot_at is not None:
        remaining = max(0.0, pending_snapshot_at - time.time())
        cv2.putText(header, f"SHOT IN {remaining:.1f}s", (body.shape[1] - 420, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    return np.vstack([header, body])

def countdown(text, seconds):
    print("")
    print(text)
    for i in range(seconds, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme startet.")
    print("")

def main():
    print("")
    print("ANTHRO3D Leerraum-vs-Person-Differenz alle Kameras")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    print("")
    roles = camera_roles()
    caps = open_caps(roles)
    if not caps:
        raise RuntimeError("Keine Kamera geoeffnet")
    stereo_by_rig = {rig[0]: make_stereo() for rig in RIGS}
    try:
        for _ in range(WARMUP_FRAMES):
            read_frames(caps)
            time.sleep(0.03)
        countdown("Leerraum-Aufnahme in 5 Sekunden. Bitte komplett aus dem Messbereich gehen.", 5)
        base = capture_depth_set(caps, stereo_by_rig)
        print("Leerraum gespeichert.")
        countdown("Person-Aufnahme in 5 Sekunden. Jetzt in den Messbereich stellen.", 5)
        person = capture_depth_set(caps, stereo_by_rig)
        print("Person gespeichert.")
    finally:
        release_caps(caps)
    rows = []
    for rig_name, mode, rig_roles, cfg_file in RIGS:
        if rig_name not in base or rig_name not in person:
            print(f"{rig_name}: fehlt")
            continue
        row, clean_count, raw_count, z_med, d_med = make_rig_row(rig_name, base[rig_name], person[rig_name])
        rows.append(row)
        print(f"{rig_name}: raw={raw_count} clean={clean_count} z={z_med:.1f}cm delta={d_med:.1f}cm")
    if not rows:
        raise RuntimeError("Keine Diagnosebilder erzeugt")
    pending_snapshot_at = None
    cv2.namedWindow("ANTHRO3D Background Delta Diagnose - q beendet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANTHRO3D Background Delta Diagnose - q beendet", 1800, 950)
    try:
        while True:
            dash = build_dashboard(rows, pending_snapshot_at)
            if pending_snapshot_at is not None and time.time() >= pending_snapshot_at:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUT / f"background_delta_{ts}.png"
                cv2.imwrite(str(path), dash)
                print(f"Snapshot gespeichert: {path}")
                pending_snapshot_at = None
            preview_w = min(1800, dash.shape[1])
            preview_h = int(dash.shape[0] * preview_w / dash.shape[1])
            preview = cv2.resize(dash, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.imshow("ANTHRO3D Background Delta Diagnose - q beendet", preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                pending_snapshot_at = time.time() + SNAPSHOT_DELAY_SECONDS
                print("Screenshot in 3 Sekunden...")
            if key == ord("q") or key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        print("Fertig.")

if __name__ == "__main__":
    main()
