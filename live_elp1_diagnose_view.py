import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "elp1_live_diagnose"
OUT.mkdir(parents=True, exist_ok=True)

FRAME_W = 2560
FRAME_H = 720
PIXEL_STRIDE = 2
SNAPSHOT_DELAY_SECONDS = 3.0

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def get_elp1_index():
    cfg = load_yaml(BASE / "config.yaml")
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled") and cam.get("name") == "ELP1":
            return int(cam.get("device_index"))
    raise RuntimeError("ELP1 nicht in config.yaml gefunden")

def split_sbs(frame):
    h, w = frame.shape[:2]
    m = w // 2
    return frame[:, :m].copy(), frame[:, m:].copy()

def make_maps(size):
    cfg = load_yaml(BASE / "stereo_config_elp1.yaml")
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

def depth_and_mask(left, right, maps, stereo):
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

def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    return img

def build_dashboard(left, right, depth, mask, count, z_med, pending_snapshot_at):
    l = cv2.resize(left, (640, 360))
    r = cv2.resize(right, (640, 360))
    l = label(l, "ELP1 LEFT")
    r = label(r, "ELP1 RIGHT")
    raw = np.vstack([l, r])
    depth_big = cv2.resize(depth, (640, 720))
    mask_big = cv2.resize(mask, (640, 720))
    depth_big = label(depth_big, f"DEPTH / POINTS  count={count}  z_med={z_med:.1f}cm")
    mask_big = label(mask_big, "VALID 3D POINT MASK")
    dash = np.hstack([raw, depth_big, mask_big])
    cv2.rectangle(dash, (0, 0), (dash.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(dash, f"ELP1 LIVE  points={count}  z={z_med:.1f}cm   SPACE=Snapshot   q=Ende", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    if pending_snapshot_at is not None:
        remaining = max(0.0, pending_snapshot_at - time.time())
        box_x1 = dash.shape[1] - 430
        box_y1 = 4
        box_x2 = dash.shape[1] - 15
        box_y2 = 42
        cv2.rectangle(dash, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.putText(dash, f"SHOT IN {remaining:.1f}s", (box_x1 + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return dash

def main():
    idx = get_elp1_index()
    print(f"ELP1 Index {idx}")
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"ELP1 Index {idx} konnte nicht geoeffnet werden")
    for _ in range(20):
        cap.read()
        time.sleep(0.03)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Kein ELP1 Frame")
    left, right = split_sbs(frame)
    maps = make_maps((left.shape[1], left.shape[0]))
    stereo = make_stereo()
    pending_snapshot_at = None
    cv2.namedWindow("ANTHRO3D ELP1 Diagnose Live - q beendet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANTHRO3D ELP1 Diagnose Live - q beendet", 1600, 600)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            left, right = split_sbs(frame)
            rect_l, rect_r, depth, mask, count, z_med = depth_and_mask(left, right, maps, stereo)
            dash = build_dashboard(rect_l, rect_r, depth, mask, count, z_med, pending_snapshot_at)
            if pending_snapshot_at is not None and time.time() >= pending_snapshot_at:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUT / f"elp1_diagnose_{ts}.png"
                cv2.imwrite(str(path), dash)
                print("")
                print(f"Screenshot gespeichert: {path}")
                pending_snapshot_at = None
            cv2.imshow("ANTHRO3D ELP1 Diagnose Live - q beendet", dash)
            print(f"ELP1 live_points={count} z_median_cm={z_med:.1f}", end="\r")
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                pending_snapshot_at = time.time() + SNAPSHOT_DELAY_SECONDS
                print("")
                print("Screenshot in 3 Sekunden...")
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("")
        print("Fertig.")

if __name__ == "__main__":
    main()
