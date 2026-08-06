import cv2
import yaml
import time
import numpy as np
from pathlib import Path

BASE = Path.home() / "anthro3d"

MARKER_SIZE_M = 0.1865
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL

PAIRS = {
    "OV9281_STAND": (2, 20),
    "ELP1_STAND": (3, 30),
    "ELP2_STAND": (4, 40),
}

MAX_FRAMES = 400
MIN_SAMPLES_PER_PAIR = 20
TIME_LIMIT_SEC = 35.0

DEVICES = [
    {"name": "ELP2", "index": 0, "width": 2560, "height": 720, "split": True, "config": "stereo_config.yaml", "calib_view_width": 1280, "calib_view_height": 720},
    {"name": "ELP1", "index": 3, "width": 3200, "height": 1200, "split": True, "config": "stereo_config_elp1.yaml", "calib_view_width": 1600, "calib_view_height": 1200},
    {"name": "OV9281_L", "index": 1, "width": 1280, "height": 800, "split": False, "side": "l", "config": "stereo_config_ov9281.yaml", "calib_view_width": 1280, "calib_view_height": 800},
    {"name": "OV9281_R", "index": 2, "width": 1280, "height": 800, "split": False, "side": "r", "config": "stereo_config_ov9281.yaml", "calib_view_width": 1280, "calib_view_height": 800},
]

def load_yaml(path):
    p = BASE / path
    if not p.exists():
        return None
    with open(p, "r") as f:
        return yaml.safe_load(f)

def as_np_matrix(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)

def get_intrinsics(config, side):
    if config is None:
        return None, None
    if side == "l":
        k = config.get("camera_matrix_l") or config.get("K_l") or config.get("mtx_l")
        d = config.get("dist_l") or config.get("dist_coeffs_l") or config.get("D_l")
    else:
        k = config.get("camera_matrix_r") or config.get("K_r") or config.get("mtx_r")
        d = config.get("dist_r") or config.get("dist_coeffs_r") or config.get("D_r")
    K = as_np_matrix(k)
    dist = as_np_matrix(d)
    if K is None or dist is None:
        return None, None
    return K, dist


# SCALED_VIEW_INTRINSICS
_PRINTED_VIEW_INTRINSICS = set()


def scale_camera_matrix(K, calibration_size, image_size):
    """
    Skaliert fx, fy, cx und cy von der Kalibrierauflösung
    auf die tatsächlich gelieferte Bildauflösung.
    """
    if K is None:
        return None

    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size

    if (
        calibration_width <= 0
        or calibration_height <= 0
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            "Ungültige Kalibrier- oder Bildauflösung."
        )

    scale_x = image_width / float(calibration_width)
    scale_y = image_height / float(calibration_height)

    scaled = np.asarray(
        K,
        dtype=np.float64
    ).copy()

    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x

    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y

    return scaled


def print_view_intrinsics_once(
    name,
    K,
    calibration_size,
    image_size
):
    if name in _PRINTED_VIEW_INTRINSICS:
        return

    _PRINTED_VIEW_INTRINSICS.add(name)

    if K is None:
        print(f"{name}: keine Intrinsik vorhanden")
        return

    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size

    print(
        f"{name}: Bild={image_width}x{image_height} | "
        f"Kalibrierung={calibration_width}x{calibration_height} | "
        f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
        f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}"
    )


def marker_object_points():
    s = MARKER_SIZE_M / 2.0
    return np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float32,
    )

def solve_marker_pose(corners, K, dist):
    obj = marker_object_points()
    img = corners.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    reproj = float(np.median(err))
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), reproj

def rotation_angle_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = float(np.clip(v, -1.0, 1.0))
    return float(np.degrees(np.arccos(v)))

def robust_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    return med, mad

def open_devices():
    opened = []
    configs = {}
    for dev in DEVICES:
        cap = cv2.VideoCapture(dev["index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, dev["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dev["height"])
        if not cap.isOpened():
            print(f"{dev['name']}: nicht geoeffnet")
            cap.release()
            continue
        cfg = configs.get(dev["config"])
        if cfg is None:
            cfg = load_yaml(dev["config"])
            configs[dev["config"]] = cfg
        opened.append((dev, cap, cfg))
        print(f"{dev['name']}: offen index={dev['index']} config={dev['config']}")
    return opened

def make_views(dev, frame, cfg):
    views = []

    calibration_size = (
        int(dev["calib_view_width"]),
        int(dev["calib_view_height"]),
    )

    if dev["split"]:
        height, width = frame.shape[:2]
        middle = width // 2

        left = frame[:, :middle].copy()
        right = frame[:, middle:].copy()

        K_l, dist_l = get_intrinsics(cfg, "l")
        K_r, dist_r = get_intrinsics(cfg, "r")

        left_size = (
            int(left.shape[1]),
            int(left.shape[0]),
        )

        right_size = (
            int(right.shape[1]),
            int(right.shape[0]),
        )

        K_l = scale_camera_matrix(
            K_l,
            calibration_size,
            left_size
        )

        K_r = scale_camera_matrix(
            K_r,
            calibration_size,
            right_size
        )

        left_name = dev["name"] + "_L"
        right_name = dev["name"] + "_R"

        print_view_intrinsics_once(
            left_name,
            K_l,
            calibration_size,
            left_size
        )

        print_view_intrinsics_once(
            right_name,
            K_r,
            calibration_size,
            right_size
        )

        views.append(
            (left_name, left, K_l, dist_l)
        )

        views.append(
            (right_name, right, K_r, dist_r)
        )

    else:
        K, dist = get_intrinsics(
            cfg,
            dev["side"]
        )

        image_size = (
            int(frame.shape[1]),
            int(frame.shape[0]),
        )

        K = scale_camera_matrix(
            K,
            calibration_size,
            image_size
        )

        print_view_intrinsics_once(
            dev["name"],
            K,
            calibration_size,
            image_size
        )

        views.append(
            (dev["name"], frame.copy(), K, dist)
        )

    return views

def detect_poses(gray, aruco_dict, params, K, dist):
    result = {}
    if K is None or dist is None:
        return result, None, None
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        return result, corners, ids
    for c, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in {2, 3, 4, 20, 30, 40}:
            continue
        pose = solve_marker_pose(c, K, dist)
        if pose is None:
            continue
        result[marker_id] = pose
    return result, corners, ids

def draw_view(view, name, corners, ids, visible, pair_ok):
    vis = cv2.resize(view, (640, 400))
    scale_x = 640.0 / view.shape[1]
    scale_y = 400.0 / view.shape[0]
    if corners is not None and ids is not None:
        scaled = []
        for c in corners:
            cc = c.copy()
            cc[:, :, 0] *= scale_x
            cc[:, :, 1] *= scale_y
            scaled.append(cc)
        cv2.aruco.drawDetectedMarkers(vis, scaled, ids)
    color = (0, 255, 0) if pair_ok else (0, 0, 255)
    cv2.rectangle(vis, (0, 0), (640, 95), (0, 0, 0), -1)
    cv2.putText(vis, name, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(vis, "IDs: " + " ".join(str(x) for x in sorted(visible)), (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis

def calc_stats(arr):
    offsets = np.array([x["offset_local_cm"] for x in arr], dtype=np.float64)
    distances = np.array([x["distance_cm"] for x in arr], dtype=np.float64)
    angles = np.array([x["rot_angle_deg"] for x in arr], dtype=np.float64)
    reproj_top = np.array([x["reproj_top"] for x in arr], dtype=np.float64)
    reproj_bottom = np.array([x["reproj_bottom"] for x in arr], dtype=np.float64)
    med_offset, mad_offset = robust_stats(offsets)
    med_distance = float(np.median(distances))
    mad_distance = float(np.median(np.abs(distances - med_distance)))
    med_angle = float(np.median(angles))
    mad_angle = float(np.median(np.abs(angles - med_angle)))
    med_reproj_top = float(np.median(reproj_top))
    med_reproj_bottom = float(np.median(reproj_bottom))
    return {
        "samples": len(arr),
        "offset_local_cm_median": med_offset.tolist(),
        "offset_local_cm_mad": mad_offset.tolist(),
        "distance_cm_median": med_distance,
        "distance_cm_mad": mad_distance,
        "rotation_angle_deg_median": med_angle,
        "rotation_angle_deg_mad": mad_angle,
        "reprojection_top_px_median": med_reproj_top,
        "reprojection_bottom_px_median": med_reproj_bottom,
    }

def summarize(samples):
    print("")
    print("Marker-Offset Ergebnis")
    print("======================")
    output = {}
    for label, pair in PAIRS.items():
        arr = samples[label]
        print("")
        print(f"{label} ID{pair[0]} -> ID{pair[1]}")
        if len(arr) == 0:
            print("  keine Samples")
            output[label] = {"top_id": pair[0], "bottom_id": pair[1], "samples": 0, "views": {}}
            continue
        pooled = calc_stats(arr)
        print("  Gesamt gepoolt")
        print(f"    Samples: {pooled['samples']}")
        print(f"    Offset lokal Median cm: x={pooled['offset_local_cm_median'][0]:.2f} y={pooled['offset_local_cm_median'][1]:.2f} z={pooled['offset_local_cm_median'][2]:.2f}")
        print(f"    Offset lokal MAD cm:    x={pooled['offset_local_cm_mad'][0]:.2f} y={pooled['offset_local_cm_mad'][1]:.2f} z={pooled['offset_local_cm_mad'][2]:.2f}")
        print(f"    Abstand Median: {pooled['distance_cm_median']:.2f} cm")
        print(f"    Abstand MAD:    {pooled['distance_cm_mad']:.2f} cm")
        print(f"    Rotation Median: {pooled['rotation_angle_deg_median']:.2f} deg")
        print(f"    Reproj oben/unten: {pooled['reprojection_top_px_median']:.2f}px / {pooled['reprojection_bottom_px_median']:.2f}px")
        grouped = {}
        for sample in arr:
            grouped.setdefault(sample["view"], []).append(sample)
        view_stats = {}
        print("  Pro View")
        for view_name in sorted(grouped):
            stats = calc_stats(grouped[view_name])
            view_stats[view_name] = stats
            status = "OK" if stats["samples"] >= 8 else "WENIG"
            print(f"    {view_name}: {status} n={stats['samples']} Abstand={stats['distance_cm_median']:.2f}cm MAD={stats['distance_cm_mad']:.2f}cm Rot={stats['rotation_angle_deg_median']:.2f}deg Reproj={stats['reprojection_top_px_median']:.2f}/{stats['reprojection_bottom_px_median']:.2f}px Offset=({stats['offset_local_cm_median'][0]:.2f}, {stats['offset_local_cm_median'][1]:.2f}, {stats['offset_local_cm_median'][2]:.2f})cm")
        output[label] = {"top_id": pair[0], "bottom_id": pair[1], "pooled": pooled, "views": view_stats}
    out_path = BASE / "aruco_state" / "marker_vertical_offsets.yaml"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False)
    print("")
    print(f"gespeichert: {out_path}")

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    opened = open_devices()
    if not opened:
        raise RuntimeError("keine Kamera geoeffnet")
    samples = {label: [] for label in PAIRS}
    start = time.time()
    frame_i = 0
    last_print = 0.0
    while True:
        frame_i += 1
        vis_list = []
        current_counts = {label: 0 for label in PAIRS}
        for dev, cap, cfg in opened:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            for view_name, view, K, dist in make_views(dev, frame, cfg):
                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                poses, corners, ids = detect_poses(gray, aruco_dict, params, K, dist)
                visible = set(poses.keys())
                pair_ok_any = False
                for label, (top_id, bottom_id) in PAIRS.items():
                    if top_id not in poses or bottom_id not in poses:
                        continue
                    R_top, t_top, reproj_top = poses[top_id]
                    R_bottom, t_bottom, reproj_bottom = poses[bottom_id]
                    delta_cam = t_bottom - t_top
                    offset_local = R_top.T @ delta_cam
                    R_rel = R_top.T @ R_bottom
                    dist_cm = float(np.linalg.norm(delta_cam) * 100.0)
                    sample = {
                        "view": view_name,
                        "offset_local_cm": (offset_local * 100.0).tolist(),
                        "distance_cm": dist_cm,
                        "rot_angle_deg": rotation_angle_deg(R_rel),
                        "reproj_top": reproj_top,
                        "reproj_bottom": reproj_bottom,
                    }
                    if reproj_top <= 5.0 and reproj_bottom <= 5.0 and dist_cm < 150.0:
                        samples[label].append(sample)
                        current_counts[label] += 1
                        pair_ok_any = True
                vis_list.append(draw_view(view, view_name, corners, ids, visible, pair_ok_any))
        while len(vis_list) < 6:
            vis_list.append(np.zeros((400, 640, 3), dtype=np.uint8))
        combo = np.vstack((np.hstack(vis_list[:3]), np.hstack(vis_list[3:6])))
        cv2.imshow("Marker Offset Messung | q beendet", combo)
        now = time.time()
        if now - last_print > 1.0:
            total = {label: len(samples[label]) for label in PAIRS}
            print(f"Frame {frame_i} Zeit {now - start:.1f}s Samples {total}")
            last_print = now
        all_ok = all(len(samples[label]) >= MIN_SAMPLES_PER_PAIR for label in PAIRS)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if frame_i >= MAX_FRAMES:
            break
        if now - start >= TIME_LIMIT_SEC:
            break
        if all_ok and frame_i >= 80:
            break
    for dev, cap, cfg in opened:
        cap.release()
    cv2.destroyAllWindows()
    summarize(samples)

if __name__ == "__main__":
    main()
