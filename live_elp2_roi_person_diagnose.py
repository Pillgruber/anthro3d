import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path.home() / "anthro3d"
OUT = BASE / "output" / "elp2_roi_person_diagnose"
OUT.mkdir(parents=True, exist_ok=True)

FRAME_W = 2560
FRAME_H = 720
PIXEL_STRIDE = 2
SNAPSHOT_DELAY_SECONDS = 3.0

ROI_DEFAULT = [170, 30, 620, 710]

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Datei fehlt: {p}")
    return yaml.safe_load(p.read_text()) or {}

def get_elp2_index():
    cfg = load_yaml(BASE / "config.yaml")
    for cam in cfg.get("cameras", {}).get("tracking", []):
        if cam.get("enabled") and cam.get("name") == "ELP2":
            return int(cam.get("device_index"))
    raise RuntimeError("ELP2 nicht in config.yaml gefunden")

def split_sbs(frame):
    h, w = frame.shape[:2]
    m = w // 2
    return frame[:, :m].copy(), frame[:, m:].copy()

def make_maps(size):
    cfg = load_yaml(BASE / "stereo_config.yaml")
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

def clamp_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 20, x1))
    y1 = max(0, min(h - 20, y1))
    x2 = max(x1 + 20, min(w, x2))
    y2 = max(y1 + 20, min(h, y2))
    return [x1, y1, x2, y2]

def depth_roi(left, right, maps, stereo, roi):
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
    x1, y1, x2, y2 = clamp_roi(roi, rect_l.shape[1], rect_l.shape[0])
    roi_mask = np.zeros(valid.shape, dtype=bool)
    roi_mask[y1:y2, x1:x2] = True
    yy, xx = np.indices(valid.shape)
    sampled = valid & roi_mask & ((xx % PIXEL_STRIDE) == 0) & ((yy % PIXEL_STRIDE) == 0)
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
    roi_depth = np.zeros_like(depth_col)
    roi_depth[sampled] = depth_col[sampled]
    mask_u8 = np.zeros_like(depth_u8)
    mask_u8[sampled] = 255
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    count = int(sampled.sum())
    if sampled.any():
        z_med = float(np.median(z[sampled]) * 100)
    else:
        z_med = 0.0
    return rect_l, rect_r, depth_col, roi_depth, mask_bgr, count, z_med, [x1, y1, x2, y2]

def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    return img

def draw_roi_scaled(img, roi, src_w, src_h):
    x1, y1, x2, y2 = roi
    sx = img.shape[1] / src_w
    sy = img.shape[0] / src_h
    cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 0, 255), 3)
    return img

def dashboard(left, right, depth, roi_depth, mask, count, z_med, roi, pending_snapshot_at):
    l = cv2.resize(left, (640, 360))
    r = cv2.resize(right, (640, 360))
    l = draw_roi_scaled(l, roi, left.shape[1], left.shape[0])
    r = draw_roi_scaled(r, roi, right.shape[1], right.shape[0])
    l = label(l, "ELP2 LEFT ROI")
    r = label(r, "ELP2 RIGHT ROI")
    raw = np.vstack([l, r])
    depth_big = cv2.resize(depth, (640, 360))
    roi_big = cv2.resize(roi_depth, (640, 360))
    depth_stack = np.vstack([label(depth_big, "DEPTH ALL"), label(roi_big, "DEPTH ROI ONLY")])
    mask_big = cv2.resize(mask, (640, 720))
    mask_big = label(mask_big, "ROI VALID 3D POINTS")
    dash = np.hstack([raw, depth_stack, mask_big])
    cv2.rectangle(dash, (0, 0), (dash.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(dash, f"ELP2 ROI LIVE points={count} z={z_med:.1f}cm SPACE=Snapshot q=Ende wasd=Move +/-=Size r=Reset", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
    if pending_snapshot_at is not None:
        remaining = max(0.0, pending_snapshot_at - time.time())
        box_x1 = dash.shape[1] - 350
        cv2.rectangle(dash, (box_x1, 4), (dash.shape[1] - 15, 42), (0, 0, 0), -1)
        cv2.putText(dash, f"SHOT IN {remaining:.1f}s", (box_x1 + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return dash

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
    roi = ROI_DEFAULT.copy()
    pending_snapshot_at = None
    cv2.namedWindow("ANTHRO3D ELP2 ROI Person Diagnose - q beendet", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANTHRO3D ELP2 ROI Person Diagnose - q beendet", 1600, 600)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            left, right = split_sbs(frame)
            rect_l, rect_r, depth, roi_depth, mask, count, z_med, roi = depth_roi(left, right, maps, stereo, roi)
            dash = dashboard(rect_l, rect_r, depth, roi_depth, mask, count, z_med, roi, pending_snapshot_at)
            if pending_snapshot_at is not None and time.time() >= pending_snapshot_at:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUT / f"elp2_roi_person_{ts}.png"
                cv2.imwrite(str(path), dash)
                print("")
                print(f"Screenshot gespeichert: {path}")
                pending_snapshot_at = None
            cv2.imshow("ANTHRO3D ELP2 ROI Person Diagnose - q beendet", dash)
            print(f"ROI points={count} z_median_cm={z_med:.1f} roi={roi}", end="\r")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                pending_snapshot_at = time.time() + SNAPSHOT_DELAY_SECONDS
                print("")
                print("Screenshot in 3 Sekunden...")
            if key == ord("r"):
                roi = ROI_DEFAULT.copy()
            if key == ord("a"):
                roi[0] -= 20
                roi[2] -= 20
            if key == ord("d"):
                roi[0] += 20
                roi[2] += 20
            if key == ord("w"):
                roi[1] -= 20
                roi[3] -= 20
            if key == ord("s"):
                roi[1] += 20
                roi[3] += 20
            if key == ord("+") or key == ord("="):
                roi[0] -= 20
                roi[1] -= 20
                roi[2] += 20
                roi[3] += 20
            if key == ord("-") or key == ord("_"):
                roi[0] += 20
                roi[1] += 20
                roi[2] -= 20
                roi[3] -= 20
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("")
        print("Fertig.")

if __name__ == "__main__":
    main()
