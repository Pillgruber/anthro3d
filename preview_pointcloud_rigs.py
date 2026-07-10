import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"
COUNTDOWN_SECONDS = 5

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {path}")
    return yaml.safe_load(p.read_text()) or {}

def camera_index(role_name):
    cfg = load_yaml(BASE / "config.yaml")
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("name") == role_name:
            return int(cam.get("device_index"))
    raise RuntimeError(f"Kamera-Rolle nicht gefunden: {role_name}")

def print_roles():
    cfg = load_yaml(BASE / "config.yaml")
    print("Kamera-Rollen aus config.yaml")
    print("=============================")
    for cam in cfg.get("cameras", {}).get("tracking", []):
        print(f"{cam.get('name')}: Index {cam.get('device_index')} | enabled={cam.get('enabled')}")
    print("")

def countdown():
    print(f"Start in {COUNTDOWN_SECONDS} Sekunden. Jetzt zwischen die Kameras stellen.")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"{i}...")
        time.sleep(1.0)
    print("Aufnahme startet jetzt.")
    print("")

def open_cap(idx, w, h):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def split_sbs(frame):
    h, w = frame.shape[:2]
    m = w // 2
    return frame[:, :m].copy(), frame[:, m:].copy()

def make_maps(cfg_path, size):
    cfg = load_yaml(BASE / cfg_path)
    K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
    d_l = np.array(cfg["dist_l"], dtype=np.float64)
    K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
    d_r = np.array(cfg["dist_r"], dtype=np.float64)
    R = np.array(cfg["R"], dtype=np.float64)
    T = np.array(cfg["T"], dtype=np.float64).reshape(3, 1)
    flags = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, d_l, K_r, d_r, size, R, T, flags=flags, alpha=0)
    ml1, ml2 = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
    mr1, mr2 = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
    return ml1, ml2, mr1, mr2, Q

def disparity_and_cloud(left, right, maps):
    ml1, ml2, mr1, mr2, Q = maps
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

def stats(name, pts, valid, local_scale=None):
    count = int(valid.sum())
    if count < 100:
        print(f"{name}: points={count} STATUS=ZU_WENIG")
        return
    p = pts[valid]
    if local_scale is not None:
        p_scaled = p * local_scale
    else:
        p_scaled = p
    z = p_scaled[:, 2]
    x = p_scaled[:, 0]
    y = p_scaled[:, 1]
    print(f"{name}: points={count}")
    print(f"  x_cm median={np.median(x)*100:.1f} min={np.percentile(x,5)*100:.1f} max={np.percentile(x,95)*100:.1f}")
    print(f"  y_cm median={np.median(y)*100:.1f} min={np.percentile(y,5)*100:.1f} max={np.percentile(y,95)*100:.1f}")
    print(f"  z_cm median={np.median(z)*100:.1f} min={np.percentile(z,5)*100:.1f} max={np.percentile(z,95)*100:.1f}")

def depth_vis(disp, valid):
    d = disp.copy()
    d[~valid] = 0
    mx = np.percentile(d[valid], 95) if valid.any() else 1
    if mx <= 0:
        mx = 1
    v = np.clip(d / mx * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)

def process_sbs(name, role, cfg_path, w, h, local_scale=None):
    idx = camera_index(role)
    cap = open_cap(idx, w, h)
    if not cap.isOpened():
        print(f"{name}: Kamera nicht offen Index {idx}")
        return
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"{name}: kein Bild Index {idx}")
        return
    print(f"{name}: Index {idx} frame={frame.shape[1]}x{frame.shape[0]}")
    left, right = split_sbs(frame)
    size = (left.shape[1], left.shape[0])
    maps = make_maps(cfg_path, size)
    disp, pts, valid = disparity_and_cloud(left, right, maps)
    stats(name, pts, valid, local_scale=local_scale)
    vis = depth_vis(disp, valid)
    cv2.imshow(name + " depth preview", cv2.resize(vis, (960, 540)))

def process_dual(name, role_l, role_r, cfg_path, w, h):
    idx_l = camera_index(role_l)
    idx_r = camera_index(role_r)
    cap_l = open_cap(idx_l, w, h)
    cap_r = open_cap(idx_r, w, h)
    if not cap_l.isOpened() or not cap_r.isOpened():
        print(f"{name}: Kamera nicht offen L={idx_l} R={idx_r}")
        cap_l.release()
        cap_r.release()
        return
    ok_l, left = cap_l.read()
    ok_r, right = cap_r.read()
    cap_l.release()
    cap_r.release()
    if not (ok_l and ok_r) or left is None or right is None:
        print(f"{name}: kein Bild L={idx_l} R={idx_r}")
        return
    print(f"{name}: L Index {idx_l} frame={left.shape[1]}x{left.shape[0]} | R Index {idx_r} frame={right.shape[1]}x{right.shape[0]}")
    size = (left.shape[1], left.shape[0])
    maps = make_maps(cfg_path, size)
    disp, pts, valid = disparity_and_cloud(left, right, maps)
    stats(name, pts, valid)
    vis = depth_vis(disp, valid)
    cv2.imshow(name + " depth preview", cv2.resize(vis, (960, 540)))

def load_elp2_scale():
    p = BASE / "aruco_state" / "elp2_id3_id30_board_calibration.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text()) or {}
    return float(d.get("scale_median"))

def main():
    print("")
    print("ANTHRO3D Punktewolken-Preview pro Rig")
    print("Diese Vorschau veraendert keine Kalibrierdateien.")
    print("")
    print_roles()
    scale = load_elp2_scale()
    if scale is not None:
        print(f"ELP2 lokale Skalenkorrektur geladen: {scale:.4f}")
        print("Hinweis: Nur in der Statistik skaliert, nicht in Dateien gespeichert.")
        print("")
    countdown()
    process_dual("OV9281", "OV9281 L", "OV9281 R", "stereo_config_ov9281.yaml", 1280, 800)
    process_sbs("ELP2", "ELP2", "stereo_config.yaml", 2560, 720, local_scale=scale)
    process_sbs("ELP1", "ELP1", "stereo_config_elp1.yaml", 2560, 720)
    print("")
    print("Fenster pruefen. Taste q schliesst.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    print("Fertig.")

if __name__ == "__main__":
    main()
