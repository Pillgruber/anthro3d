import cv2
import yaml
import time
import numpy as np
from pathlib import Path
import calibrate_positions as cp

BASE = Path.home() / "anthro3d"
CFG_PATH = BASE / "stereo_config.yaml"
OUT_PATH = BASE / "aruco_state" / "elp2_id3_id30_board_calibration.yaml"

def get_elp2_index():
    try:
        d = yaml.safe_load((BASE / "config.yaml").read_text()) or {}
        for cam in d.get("cameras", {}).get("tracking", []):
            if cam.get("name") == "ELP2":
                return int(cam.get("device_index"))
    except Exception:
        pass
    return 0

CAM_INDEX = get_elp2_index()
MARKER_SIZE_M = 0.1850
TOP_ID = 3
BOTTOM_ID = 30
CENTER_DISTANCE_CM = 34.7

MIN_SAMPLES = 60
MAX_FRAMES = 500
TIME_LIMIT_SEC = 45.0

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

K_l = np.array(cfg["camera_matrix_l"], dtype=np.float64)
d_l = np.array(cfg["dist_l"], dtype=np.float64)
K_r = np.array(cfg["camera_matrix_r"], dtype=np.float64)
d_r = np.array(cfg["dist_r"], dtype=np.float64)
R = np.array(cfg["R"], dtype=np.float64)
T = np.array(cfg["T"], dtype=np.float64).reshape(3, 1)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params = cv2.aruco.DetectorParameters()

if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()

def detect(frame):
    gray = cp.preprocess(frame)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid in {TOP_ID, BOTTOM_ID}:
            out[mid] = c[0].astype(np.float32)
    return out

def triangulate(c_l, c_r):
    pl = cv2.undistortPoints(c_l.reshape(-1, 1, 2), K_l, d_l).reshape(-1, 2)
    pr = cv2.undistortPoints(c_r.reshape(-1, 1, 2), K_r, d_r).reshape(-1, 2)
    P_l = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    P_r = np.hstack([R, T]).astype(np.float64)
    X_h = cv2.triangulatePoints(P_l, P_r, pl.T, pr.T)
    return (X_h[:3] / X_h[3]).T

def marker_center_size(X):
    edges = [
        np.linalg.norm(X[1] - X[0]),
        np.linalg.norm(X[2] - X[1]),
        np.linalg.norm(X[3] - X[2]),
        np.linalg.norm(X[0] - X[3]),
    ]
    return X.mean(axis=0), float(np.median(edges))

def summarize(vals):
    a = np.array(vals, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med, mad, float(np.min(a)), float(np.max(a))

def draw_marker_status(img, name, ids):
    vis = img.copy()
    text = f"{name}: {sorted(ids)}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 65), (0, 0, 0), -1)
    cv2.putText(vis, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return vis

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("ELP2 Index 0 konnte nicht geoeffnet werden")

    samples = []
    start = time.time()
    frame_i = 0

    print("ELP2 ID3/ID30 Board-Kalibrierung")
    print(f"ELP2 Index: {CAM_INDEX}")
    print(f"Ziel: ID{TOP_ID}->ID{BOTTOM_ID}, realer Abstand Mitte-Mitte = {CENTER_DISTANCE_CM:.2f} cm")
    print("Board in den spaeteren ELP2-Messbereich halten. q beendet.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        left, right = split_sbs(frame)
        dl = detect(left)
        dr = detect(right)

        frame_centers = {}
        frame_sizes = {}

        for mid in sorted(set(dl.keys()) & set(dr.keys())):
            X = triangulate(dl[mid], dr[mid])
            if not np.isfinite(X).all():
                continue
            center, size = marker_center_size(X)
            if center[2] <= 0:
                continue
            frame_centers[mid] = center
            frame_sizes[mid] = size

        if TOP_ID in frame_centers and BOTTOM_ID in frame_centers:
            dist_cm = float(np.linalg.norm(frame_centers[BOTTOM_ID] - frame_centers[TOP_ID]) * 100.0)
            top_size_cm = float(frame_sizes[TOP_ID] * 100.0)
            bottom_size_cm = float(frame_sizes[BOTTOM_ID] * 100.0)
            z_cm = float(((frame_centers[TOP_ID][2] + frame_centers[BOTTOM_ID][2]) / 2.0) * 100.0)
            samples.append({
                "distance_cm": dist_cm,
                "top_size_cm": top_size_cm,
                "bottom_size_cm": bottom_size_cm,
                "z_cm": z_cm,
            })

        vis_l = draw_marker_status(left, "ELP2 L", dl.keys())
        vis_r = draw_marker_status(right, "ELP2 R", dr.keys())
        combo = np.vstack((cv2.resize(vis_l, (640, 360)), cv2.resize(vis_r, (640, 360))))

        status = f"samples={len(samples)}/{MIN_SAMPLES}"
        if samples:
            last = samples[-1]
            status += f" dist={last['distance_cm']:.1f}cm top={last['top_size_cm']:.1f}cm bottom={last['bottom_size_cm']:.1f}cm z={last['z_cm']:.0f}cm"

        cv2.rectangle(combo, (0, 0), (640, 45), (0, 0, 0), -1)
        cv2.putText(combo, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
        cv2.imshow("ELP2 ID3/ID30 Board | q beendet", combo)

        key = cv2.waitKey(1) & 0xFF
        frame_i += 1

        if key == ord("q"):
            break
        if len(samples) >= MIN_SAMPLES and frame_i > 80:
            break
        if frame_i >= MAX_FRAMES:
            break
        if time.time() - start >= TIME_LIMIT_SEC:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 10:
        raise RuntimeError(f"zu wenige Samples: {len(samples)}")

    distances = [s["distance_cm"] for s in samples]
    top_sizes = [s["top_size_cm"] for s in samples]
    bottom_sizes = [s["bottom_size_cm"] for s in samples]
    z_vals = [s["z_cm"] for s in samples]

    dist_med, dist_mad, dist_min, dist_max = summarize(distances)
    top_med, top_mad, top_min, top_max = summarize(top_sizes)
    bot_med, bot_mad, bot_min, bot_max = summarize(bottom_sizes)
    z_med, z_mad, z_min, z_max = summarize(z_vals)

    scale_from_distance = CENTER_DISTANCE_CM / dist_med
    scale_from_top_marker = MARKER_SIZE_M * 100.0 / top_med
    scale_from_bottom_marker = MARKER_SIZE_M * 100.0 / bot_med
    scale_median = float(np.median([scale_from_distance, scale_from_top_marker, scale_from_bottom_marker]))

    result = {
        "camera": "ELP2",
        "top_id": TOP_ID,
        "bottom_id": BOTTOM_ID,
        "marker_size_cm_real": MARKER_SIZE_M * 100.0,
        "center_distance_cm_real": CENTER_DISTANCE_CM,
        "samples": int(len(samples)),
        "distance_cm_median": dist_med,
        "distance_cm_mad": dist_mad,
        "distance_cm_min": dist_min,
        "distance_cm_max": dist_max,
        "top_marker_size_cm_median": top_med,
        "top_marker_size_cm_mad": top_mad,
        "bottom_marker_size_cm_median": bot_med,
        "bottom_marker_size_cm_mad": bot_mad,
        "z_cm_median": z_med,
        "z_cm_mad": z_mad,
        "scale_from_distance": float(scale_from_distance),
        "scale_from_top_marker": float(scale_from_top_marker),
        "scale_from_bottom_marker": float(scale_from_bottom_marker),
        "scale_median": scale_median,
        "stereo_config_baseline_cm": float(np.linalg.norm(T) * 100.0),
        "timestamp": time.time(),
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(result, sort_keys=False))

    print("")
    print("ELP2 ID3/ID30 Board Ergebnis")
    print("============================")
    print(f"samples={len(samples)}")
    print(f"real_dist={CENTER_DISTANCE_CM:.2f}cm measured_dist={dist_med:.2f}cm MAD={dist_mad:.2f}cm")
    print(f"top_marker real={MARKER_SIZE_M*100.0:.2f}cm measured={top_med:.2f}cm MAD={top_mad:.2f}cm")
    print(f"bottom_marker real={MARKER_SIZE_M*100.0:.2f}cm measured={bot_med:.2f}cm MAD={bot_mad:.2f}cm")
    print(f"z_median={z_med:.1f}cm")
    print(f"scale_from_distance={scale_from_distance:.4f}")
    print(f"scale_from_top_marker={scale_from_top_marker:.4f}")
    print(f"scale_from_bottom_marker={scale_from_bottom_marker:.4f}")
    print(f"scale_median={scale_median:.4f}")
    print(f"gespeichert={OUT_PATH}")

if __name__ == "__main__":
    main()
