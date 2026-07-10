import cv2
import yaml
import time
import numpy as np
from pathlib import Path
import calibrate_positions as cp

BASE = Path.home() / "anthro3d"

RIGS = [
    {"name": "ELP2", "kind": "sbs", "index": 0, "config": "stereo_config.yaml", "width": 2560, "height": 720},
    {"name": "ELP1", "kind": "sbs", "index": 3, "config": "stereo_config_elp1.yaml", "width": 2560, "height": 720},
    {"name": "OV9281", "kind": "dual", "left_index": 1, "right_index": 2, "config": "stereo_config_ov9281.yaml", "width": 1280, "height": 800},
]

def load_cfg(name):
    with open(BASE / name, "r") as f:
        return yaml.safe_load(f)

def make_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)

def open_cap(idx, w, h):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def read_frame(cap):
    frame = None
    ok_any = False
    for _ in range(5):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            ok_any = True
        time.sleep(0.02)
    return ok_any, frame

def rot_diff_deg(rvec_a, rvec_b):
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    Rd = Ra @ Rb.T
    tr = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def summarize(values):
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad, float(np.min(arr)), float(np.max(arr))

def process_rig(rig, detector):
    cfg = load_cfg(rig["config"])
    rig_cfg = cp.make_rig(cfg)
    samples = {}

    if rig["kind"] == "sbs":
        cap = open_cap(rig["index"], rig["width"], rig["height"])
        if not cap.isOpened():
            print(f"{rig['name']}: Kamera nicht offen")
            return
        for _ in range(80):
            ok, frame = read_frame(cap)
            if not ok:
                continue
            left, right = split_sbs(frame)
            det = cp.detect_valid_markers_rig(left, right, rig["name"], rig_cfg, detector)
            collect(samples, det)
        cap.release()
    else:
        cap_l = open_cap(rig["left_index"], rig["width"], rig["height"])
        cap_r = open_cap(rig["right_index"], rig["width"], rig["height"])
        if not cap_l.isOpened() or not cap_r.isOpened():
            print(f"{rig['name']}: Kamera nicht offen")
            cap_l.release()
            cap_r.release()
            return
        for _ in range(80):
            ok_l, left = read_frame(cap_l)
            ok_r, right = read_frame(cap_r)
            if not (ok_l and ok_r):
                continue
            det = cp.detect_valid_markers_rig(left, right, rig["name"], rig_cfg, detector)
            collect(samples, det)
        cap_l.release()
        cap_r.release()

    print("")
    print(f"=== {rig['name']} Stereo-Marker-Konsistenz ===")

    if not samples:
        print("keine gemeinsamen L/R Marker")
        return

    for mid in sorted(samples):
        dt = [x[0] for x in samples[mid]]
        dr = [x[1] for x in samples[mid]]
        err_l = [x[2] for x in samples[mid]]
        err_r = [x[3] for x in samples[mid]]
        dt_med, dt_mad, dt_min, dt_max = summarize(dt)
        dr_med, dr_mad, dr_min, dr_max = summarize(dr)
        el_med, _, _, _ = summarize(err_l)
        er_med, _, _, _ = summarize(err_r)
        status = "OK" if dt_med <= 5.0 and dt_mad <= 2.0 and dr_med <= 5.0 else "AUFFAELLIG"
        print(f"ID{mid}: {status} n={len(dt)} dT_med={dt_med:.2f}cm dT_mad={dt_mad:.2f}cm dT_minmax={dt_min:.2f}/{dt_max:.2f}cm dR_med={dr_med:.2f}deg errL/R={el_med:.2f}/{er_med:.2f}px")

def collect(samples, det):
    for mid, obs_list in det.items():
        left = [o for o in obs_list if o.get("side") == "L"]
        right = [o for o in obs_list if o.get("side") == "R"]
        if not left or not right:
            continue
        l = sorted(left, key=lambda x: x["err"])[0]
        r = sorted(right, key=lambda x: x["err"])[0]
        dt_cm = float(np.linalg.norm(l["tvec"].reshape(3) - r["tvec"].reshape(3)) * 100.0)
        dr_deg = rot_diff_deg(l["rvec"], r["rvec"])
        samples.setdefault(int(mid), []).append((dt_cm, dr_deg, float(l["err"]), float(r["err"])))

def main():
    print("ANTHRO3D Stereo-Marker-Konsistenz")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    detector = make_detector()
    for rig in RIGS:
        process_rig(rig, detector)
    print("")
    print("Fertig.")

if __name__ == "__main__":
    main()
