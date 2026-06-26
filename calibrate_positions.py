from pathlib import Path
import time
import math

import cv2
import numpy as np
import yaml


BASE = Path("~/anthro3d").expanduser()

MARKER_SIZE_M = 0.1865

ARUCO_DICT_ID = cv2.aruco.DICT_ARUCO_ORIGINAL
KNOWN_MARKER_IDS = {2, 3, 4, 20, 30, 40}

STAND_MARKERS = {
    "OV9281_STAND": {2, 20},
    "ELP1_STAND": {3, 30},
    "ELP2_STAND": {4, 40},
}

EXPECTED_VISIBLE_MARKERS = {
    "ELP2": STAND_MARKERS["OV9281_STAND"] | STAND_MARKERS["ELP1_STAND"],
    "ELP1": STAND_MARKERS["OV9281_STAND"] | STAND_MARKERS["ELP2_STAND"],
    "OV9281": STAND_MARKERS["ELP1_STAND"] | STAND_MARKERS["ELP2_STAND"],
}

PAIR_RULES = {
    "ELP1_to_ELP2": {
        "target": "ELP2",
        "source": "ELP1",
        "markers": sorted(STAND_MARKERS["OV9281_STAND"]),
        "description": "ELP1 zu ELP2 ueber OV9281-Stativ Marker",
        "legacy_R_file": "R_rel_elp1_to_elp2.npy",
        "legacy_T_file": "T_rel_elp1_to_elp2.npy",
        "required": True,
    },
    "OV9281_to_ELP2": {
        "target": "ELP2",
        "source": "OV9281",
        "markers": sorted(STAND_MARKERS["ELP1_STAND"]),
        "description": "OV9281 zu ELP2 ueber ELP1-Stativ Marker",
        "legacy_R_file": "R_rel_ov9281_to_elp2.npy",
        "legacy_T_file": "T_rel_ov9281_to_elp2.npy",
        "required": True,
    },
    "OV9281_to_ELP1": {
        "target": "ELP1",
        "source": "OV9281",
        "markers": sorted(STAND_MARKERS["ELP2_STAND"]),
        "description": "OV9281 zu ELP1 ueber ELP2-Stativ Marker",
        "legacy_R_file": "R_rel_ov9281_to_elp1.npy",
        "legacy_T_file": "T_rel_ov9281_to_elp1.npy",
        "required": False,
    },
}

MIN_MARKER_AREA_PX = 40 * 40
MAX_MARKER_DISTANCE_M = 8.0
MIN_MARKER_RATIO = 0.55
MAX_MARKER_RATIO = 1.45
MAX_REPROJECTION_ERROR_PX = 8.0

N_MAX_FRAMES = 150
N_MIN_FRAMES = 40
TIME_LIMIT_S = 20.0
MIN_PAIR_SAMPLES = 40


def load_yaml(name):
    with open(BASE / name, "r") as f:
        return yaml.safe_load(f)


def load_camera_indices():
    default = {
        "ELP2": 0,
        "ELP1": 3,
        "OV9281 L": 1,
        "OV9281 R": 2,
    }

    path = BASE / "config.yaml"
    if not path.exists():
        return default

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        out = dict(default)

        for cam in cfg.get("cameras", {}).get("tracking", []):
            name = cam.get("name")
            idx = cam.get("device_index")

            if name in out and idx is not None:
                out[name] = int(idx)

        return out
    except Exception:
        return default


def make_rig(cfg):
    return {
        "K_l": np.array(cfg["camera_matrix_l"], dtype=np.float64),
        "d_l": np.array(cfg["dist_l"], dtype=np.float64),
        "K_r": np.array(cfg["camera_matrix_r"], dtype=np.float64),
        "d_r": np.array(cfg["dist_r"], dtype=np.float64),
        "R_lr": np.array(cfg["R"], dtype=np.float64),
        "T_lr": np.array(cfg["T"], dtype=np.float64).reshape(3),
    }


def split_side_by_side(frame):
    h, w = frame.shape[:2]

    if w >= h * 2:
        mid = w // 2
        return frame[:, :mid], frame[:, mid:]

    return frame, None


def marker_object_points():
    s = MARKER_SIZE_M / 2.0
    return np.array(
        [
            [-s, s, 0],
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, 0],
        ],
        dtype=np.float32,
    )


OBJ_PTS = marker_object_points()


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def reprojection_error(corners_2d, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(OBJ_PTS, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    real = corners_2d.reshape(-1, 2)
    err = np.sqrt(((proj - real) ** 2).sum(axis=1))
    return float(np.mean(err))


def solve_marker_pose(corners_2d, K, dist):
    img_pts = corners_2d.astype(np.float32)

    best = None

    def score_pose(rvec, tvec):
        e = reprojection_error(img_pts, rvec, tvec, K, dist)
        d = float(np.linalg.norm(tvec))
        return e, d

    try:
        result = cv2.solvePnPGeneric(
            OBJ_PTS,
            img_pts,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        ok = bool(result[0])
        rvecs = result[1]
        tvecs = result[2]

        if ok and rvecs is not None and tvecs is not None:
            for rvec, tvec in zip(rvecs, tvecs):
                e, d = score_pose(rvec, tvec)
                if best is None or e < best[0]:
                    best = (e, d, rvec, tvec)
    except Exception:
        pass

    try:
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_PTS,
            img_pts,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if ok:
            e, d = score_pose(rvec, tvec)
            if best is None or e < best[0]:
                best = (e, d, rvec, tvec)
    except Exception:
        pass

    if best is None:
        return False, None, None, 999.0

    return True, best[2], best[3], best[0]


def marker_geometry_ok(corner, tvec):
    pts = corner.reshape(-1, 2)

    area = float(cv2.contourArea(pts.astype(np.float32)))
    if area < MIN_MARKER_AREA_PX:
        return False

    w = float(np.linalg.norm(pts[0] - pts[1]))
    h = float(np.linalg.norm(pts[1] - pts[2]))

    if h <= 0:
        return False

    ratio = w / h

    if ratio < MIN_MARKER_RATIO or ratio > MAX_MARKER_RATIO:
        return False

    dist_m = float(np.linalg.norm(tvec))

    if dist_m > MAX_MARKER_DISTANCE_M:
        return False

    return True


def right_pose_to_left_pose(rvec_r, tvec_r, rig):
    R_lr = rig["R_lr"]
    T_lr = rig["T_lr"]

    R_marker_r, _ = cv2.Rodrigues(rvec_r)
    R_marker_l = R_lr.T @ R_marker_r
    T_marker_l = R_lr.T @ (tvec_r.reshape(3) - T_lr)

    rvec_l, _ = cv2.Rodrigues(R_marker_l)

    return rvec_l, T_marker_l.reshape(3, 1)


def detect_valid_markers_single(gray, K, dist, cam_name, side_name, rig, detector):
    corners, ids, _ = detector.detectMarkers(gray)

    result = {}

    if ids is None:
        return result

    expected = EXPECTED_VISIBLE_MARKERS.get(cam_name, set())

    for corner, mid_raw in zip(corners, ids.flatten()):
        mid = int(mid_raw)

        if mid not in KNOWN_MARKER_IDS:
            continue

        if mid not in expected:
            continue

        ok, rvec, tvec, err = solve_marker_pose(corner[0].astype(np.float32), K, dist)

        if not ok:
            continue

        if err > MAX_REPROJECTION_ERROR_PX:
            continue

        if not marker_geometry_ok(corner, tvec):
            continue

        if side_name == "R":
            rvec_l, tvec_l = right_pose_to_left_pose(rvec, tvec, rig)
        else:
            rvec_l, tvec_l = rvec, tvec

        obs = {
            "mid": mid,
            "rvec": rvec_l,
            "tvec": tvec_l,
            "err": float(err),
            "side": side_name,
        }

        result.setdefault(mid, []).append(obs)

    return result


def detect_valid_markers_rig(left_frame, right_frame, cam_name, rig, detector):
    merged = {}

    if left_frame is not None:
        gray_l = preprocess(left_frame)
        left = detect_valid_markers_single(
            gray_l,
            rig["K_l"],
            rig["d_l"],
            cam_name,
            "L",
            rig,
            detector,
        )

        for mid, obs in left.items():
            merged.setdefault(mid, []).extend(obs)

    if right_frame is not None:
        gray_r = preprocess(right_frame)
        right = detect_valid_markers_single(
            gray_r,
            rig["K_r"],
            rig["d_r"],
            cam_name,
            "R",
            rig,
            detector,
        )

        for mid, obs in right.items():
            merged.setdefault(mid, []).extend(obs)

    for mid in list(merged.keys()):
        merged[mid] = sorted(merged[mid], key=lambda x: x["err"])

    return merged


def relative_from_common_marker(target_obs, source_obs):
    R_t, _ = cv2.Rodrigues(target_obs["rvec"])
    R_s, _ = cv2.Rodrigues(source_obs["rvec"])

    T_t = target_obs["tvec"].reshape(3)
    T_s = source_obs["tvec"].reshape(3)

    R_rel = R_t @ R_s.T
    T_rel = T_t - R_rel @ T_s

    score = float(target_obs["err"] + source_obs["err"])

    return R_rel, T_rel, score


def status_text(detections):
    parts = []

    for mid in sorted(detections.keys()):
        for obs in detections[mid]:
            parts.append(f"{mid}{obs['side']}")

    return parts



PERMARKER_MIN_SAMPLES = 12
PERMARKER_CONSISTENT_T_CM = 8.0
PERMARKER_CONSISTENT_R_DEG = 6.0
PERMARKER_MAX_STD_CM = 30.0
TRIANGLE_COMBO_ROT_WEIGHT = 2.0
TRIANGLE_COMBO_WARN_SCORE = 25.0
TRIANGLE_COMBO_AMBIGUOUS_DELTA = 5.0

def _svd_average_rotations(R_arr):
    R_mean = np.mean(R_arr, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def _estimate_single_marker(samples, label):
    if len(samples) < PERMARKER_MIN_SAMPLES:
        return None
    R_arr = np.array([s["R"] for s in samples], dtype=np.float64)
    T_arr = np.array([s["T"] for s in samples], dtype=np.float64)
    score_arr = np.array([s["score"] for s in samples], dtype=np.float64)
    dist = np.linalg.norm(T_arr, axis=1)
    med_dist = np.median(dist)
    keep = np.abs(dist - med_dist) < max(0.20, med_dist * 0.15)
    if keep.sum() >= PERMARKER_MIN_SAMPLES:
        R_arr, T_arr, score_arr = R_arr[keep], T_arr[keep], score_arr[keep]
    if len(score_arr) >= 20:
        lim = np.percentile(score_arr, 70)
        m = score_arr <= lim
        if m.sum() >= PERMARKER_MIN_SAMPLES:
            R_arr, T_arr, score_arr = R_arr[m], T_arr[m], score_arr[m]
    T_med = np.median(T_arr, axis=0)
    R_avg = _svd_average_rotations(R_arr)
    std_max_cm = float(np.max(T_arr.std(axis=0) * 100.0))
    reproj_med = float(np.median(score_arr))
    return {"R": R_avg, "T": T_med.reshape(3), "n": int(len(T_arr)),
            "std_max_cm": std_max_cm, "reproj_median": reproj_med}

def _rot_diff_deg_local(R_a, R_b):
    Rd = R_a @ R_b.T
    tr = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _sample_side_label(sm):
    if "source_side" in sm and "target_side" in sm:
        return f"{sm['source_side']}->{sm['target_side']}"
    if "side" in sm:
        return str(sm["side"])
    return "unbekannt"


def _print_side_diagnostics(samples, label):
    by_marker_side = {}

    for sm in samples:
        mid = int(sm.get("marker", -1))
        side = _sample_side_label(sm)
        by_marker_side.setdefault((mid, side), []).append(sm)

    if not by_marker_side:
        return

    print("")
    print(f"Side-Diagnose: {label}")
    print("-" * (15 + len(label)))

    for (mid, side), sm_list in sorted(by_marker_side.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        est = _estimate_single_marker(sm_list, f"{label}/ID{mid}/{side}")

        if est is None:
            print(f"    ID{mid} {side}: zu wenige brauchbare Samples ({len(sm_list)})")
            continue

        print(
            f"    ID{mid} {side}: "
            f"T={est['T'] * 100.0} cm | "
            f"n={est['n']} | "
            f"Streuung max={est['std_max_cm']:.1f} cm | "
            f"Reproj median={est['reproj_median']:.2f}px"
        )

    if all(_sample_side_label(sm) == "unbekannt" for sm in samples):
        print("    WARNUNG: Keine Seiteninformation in Samples gefunden.")
        print("    Erwartet werden source_side und target_side.")

def estimate_pair(samples, label):
    raw_count = len(samples)
    _print_side_diagnostics(samples, label)
    if raw_count < MIN_PAIR_SAMPLES:
        print(f"{label}: FEHLER | nur {raw_count} Rohsamples")
        return None
    by_marker = {}
    for sm in samples:
        by_marker.setdefault(int(sm["marker"]), []).append(sm)
    per = {}
    for mid, sm_list in by_marker.items():
        est = _estimate_single_marker(sm_list, f"{label}/ID{mid}")
        if est is not None and est["std_max_cm"] <= PERMARKER_MAX_STD_CM:
            per[mid] = est
    if not per:
        print(f"{label}: FEHLER | keine stabile Einzelmarker-Schaetzung")
        return None
    for mid, est in sorted(per.items()):
        print(f"    {label} ID{mid}: T={est['T']*100.0} cm | n={est['n']} | "
              f"Streuung max={est['std_max_cm']:.1f} cm | Reproj median={est['reproj_median']:.2f}px")
    mids = sorted(per.keys())
    if len(mids) == 1:
        chosen = [mids[0]]
    else:
        ref = min(mids, key=lambda k: per[k]["std_max_cm"])
        cluster = [ref]
        for mid in mids:
            if mid == ref:
                continue
            t_diff = float(np.linalg.norm(per[mid]["T"] - per[ref]["T"]) * 100.0)
            r_diff = _rot_diff_deg_local(per[mid]["R"], per[ref]["R"])
            if t_diff <= PERMARKER_CONSISTENT_T_CM and r_diff <= PERMARKER_CONSISTENT_R_DEG:
                cluster.append(mid)
                print(f"    {label}: ID{mid} konsistent mit ID{ref} (dT={t_diff:.1f}cm, dR={r_diff:.1f}deg) -> kombiniere")
            else:
                print(f"    {label}: ID{mid} INKONSISTENT mit ID{ref} (dT={t_diff:.1f}cm, dR={r_diff:.1f}deg) -> verworfen")
        chosen = cluster
    if len(chosen) == 1:
        est = per[chosen[0]]
        R_final, T_final = est["R"], est["T"]
        std_final = est["std_max_cm"]; reproj_final = est["reproj_median"]; n_final = est["n"]
        used = f"ID{chosen[0]} (einzeln)"
    else:
        weights = np.array([1.0 / max(per[m]["std_max_cm"], 0.5) for m in chosen])
        weights /= weights.sum()
        T_final = np.sum([w * per[m]["T"] for w, m in zip(weights, chosen)], axis=0)
        R_final = _svd_average_rotations(np.array([per[m]["R"] for m in chosen]))
        std_final = float(np.mean([per[m]["std_max_cm"] for m in chosen]))
        reproj_final = float(np.mean([per[m]["reproj_median"] for m in chosen]))
        n_final = int(np.sum([per[m]["n"] for m in chosen]))
        used = "ID" + "+".join(str(m) for m in chosen) + " (kombiniert)"
    marker_counts = {int(m): per[m]["n"] for m in chosen}
    print(f"{label}: OK | T={T_final*100.0} cm | Distanz={np.linalg.norm(T_final)*100.0:.1f} cm | "
          f"Streuung max={std_final:.1f} cm | Reproj median={reproj_final:.2f}px | Quelle={used}")
    if std_final > 8.0:
        print(f"WARNUNG: {label} Eigenstreuung noch hoch ({std_final:.1f} cm).")
    return {"R": R_final, "T": T_final.reshape(3), "samples": int(n_final),
            "raw_samples": int(raw_count), "std_max_cm": std_final,
            "reproj_median": reproj_final, "reproj_max": reproj_final,
            "marker_counts": marker_counts,
            "per_marker": per,
            "selected_markers": [int(m) for m in chosen]}

def rotation_diff_deg(R_a, R_b):
    R_delta = R_a @ R_b.T
    trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


def compose(first, second):
    R1 = first["R"]
    T1 = first["T"].reshape(3)

    R2 = second["R"]
    T2 = second["T"].reshape(3)

    R = R1 @ R2
    T = R1 @ T2 + T1

    return {"R": R, "T": T}



def _single_marker_estimate(pair_est, marker_id):
    marker_id = int(marker_id)
    pm = pair_est.get("per_marker", {})
    if marker_id not in pm:
        return None

    src = pm[marker_id]

    return {
        "R": np.array(src["R"], dtype=np.float64),
        "T": np.array(src["T"], dtype=np.float64).reshape(3),
        "samples": int(src.get("n", 0)),
        "raw_samples": int(pair_est.get("raw_samples", src.get("n", 0))),
        "std_max_cm": float(src.get("std_max_cm", 999.0)),
        "reproj_median": float(src.get("reproj_median", 999.0)),
        "reproj_max": float(src.get("reproj_median", 999.0)),
        "marker_counts": {marker_id: int(src.get("n", 0))},
        "per_marker": pm,
        "selected_markers": [marker_id],
    }


def apply_triangle_combo_selection(estimates):
    required = ["ELP1_to_ELP2", "OV9281_to_ELP2", "OV9281_to_ELP1"]

    if any(estimates.get(name) is None for name in required):
        print("")
        print("Triangle-Combo-Auswahl: nicht moeglich, mindestens eine Beziehung fehlt.")
        return estimates

    per_lists = {}

    for name in required:
        pm = estimates[name].get("per_marker", {})
        mids = sorted(int(m) for m in pm.keys())

        if not mids:
            print("")
            print(f"Triangle-Combo-Auswahl: {name} hat keine Einzelmarker-Schaetzungen.")
            return estimates

        per_lists[name] = mids

    rows = []

    for m12 in per_lists["ELP1_to_ELP2"]:
        e12 = _single_marker_estimate(estimates["ELP1_to_ELP2"], m12)
        if e12 is None:
            continue

        for m02 in per_lists["OV9281_to_ELP2"]:
            e02 = _single_marker_estimate(estimates["OV9281_to_ELP2"], m02)
            if e02 is None:
                continue

            for m01 in per_lists["OV9281_to_ELP1"]:
                e01 = _single_marker_estimate(estimates["OV9281_to_ELP1"], m01)
                if e01 is None:
                    continue

                via = compose(e12, e01)
                direct = e02

                t_diff = float(np.linalg.norm(direct["T"].reshape(3) - via["T"].reshape(3)) * 100.0)
                r_diff = rotation_diff_deg(direct["R"], via["R"])
                score = float(t_diff + TRIANGLE_COMBO_ROT_WEIGHT * r_diff)

                rows.append(
                    {
                        "score": score,
                        "t_diff": t_diff,
                        "r_diff": r_diff,
                        "m12": int(m12),
                        "m02": int(m02),
                        "m01": int(m01),
                        "e12": e12,
                        "e02": e02,
                        "e01": e01,
                    }
                )

    if not rows:
        print("")
        print("Triangle-Combo-Auswahl: keine Kombinationen berechenbar.")
        return estimates

    rows.sort(key=lambda x: x["score"])

    print("")
    print("Triangle-Combo-Auswahl")
    print("======================")
    print(f"Score = Translation_cm + {TRIANGLE_COMBO_ROT_WEIGHT:.1f} * Rotation_deg")

    for idx, row in enumerate(rows, start=1):
        marker_text = (
            f"ELP1_to_ELP2=ID{row['m12']} | "
            f"OV9281_to_ELP2=ID{row['m02']} | "
            f"OV9281_to_ELP1=ID{row['m01']}"
        )

        prefix = "BESTE" if idx == 1 else f"{idx}."

        print(
            f"  {prefix}: {marker_text} | "
            f"dT={row['t_diff']:.1f} cm | "
            f"dR={row['r_diff']:.2f} deg | "
            f"Score={row['score']:.2f}"
        )

    best = rows[0]

    if best["score"] > TRIANGLE_COMBO_WARN_SCORE:
        print(
            f"WARNUNG: Bester Triangle-Combo-Score ist {best['score']:.2f} "
            f"> {TRIANGLE_COMBO_WARN_SCORE:.2f}. "
            "Dreieck schliesst auch optimal nicht. Ursache liegt tiefer."
        )

    if len(rows) >= 2:
        delta = rows[1]["score"] - rows[0]["score"]

        if delta < TRIANGLE_COMBO_AMBIGUOUS_DELTA:
            print(
                f"WARNUNG: Triangle-Combo-Wahl nicht eindeutig. "
                f"Score-Abstand beste zwei Kombinationen: {delta:.2f} "
                f"< {TRIANGLE_COMBO_AMBIGUOUS_DELTA:.2f}."
            )

    estimates["ELP1_to_ELP2"] = best["e12"]
    estimates["OV9281_to_ELP2"] = best["e02"]
    estimates["OV9281_to_ELP1"] = best["e01"]

    print(
        "Triangle-Combo gewaehlt: "
        f"ELP1_to_ELP2=ID{best['m12']}, "
        f"OV9281_to_ELP2=ID{best['m02']}, "
        f"OV9281_to_ELP1=ID{best['m01']}"
    )

    return estimates


def save_legacy(pair_name, estimate):
    rule = PAIR_RULES[pair_name]

    R_file = rule.get("legacy_R_file")
    T_file = rule.get("legacy_T_file")

    if not R_file or not T_file:
        return

    np.save(BASE / R_file, estimate["R"])
    np.save(BASE / T_file, estimate["T"].reshape(3, 1))


def save_state(pair_name, estimate):
    state = BASE / "aruco_state"
    state.mkdir(parents=True, exist_ok=True)

    path = state / f"{pair_name}_active.npz"

    np.savez_compressed(
        path,
        ok=np.array([True]),
        R=np.array(estimate["R"], dtype=np.float64),
        T=np.array(estimate["T"], dtype=np.float64).reshape(3),
        source=np.array(["calibrate_positions_multimarker"]),
        samples=np.array([int(estimate["samples"])]),
        raw_samples=np.array([int(estimate["raw_samples"])]),
        std_max_cm=np.array([float(estimate["std_max_cm"])]),
        reproj_median=np.array([float(estimate["reproj_median"])]),
        reproj_max=np.array([float(estimate["reproj_max"])]),
        timestamp=np.array([time.time()]),
        confirmations=np.array([0]),
        triangle_ok=np.array([False]),
        triangle_t_diff_cm=np.array([-1.0]),
        triangle_r_diff_deg=np.array([-1.0]),
        triangle_status=np.array(["not_checked"]),
    )


def main():
    print("=== ANTHRO3D Positionskalibrierung Multi-Marker ===")
    print("Hintergrundaufnahme wird bewusst nicht gestartet.")
    print(f"Dictionary: DICT_ARUCO_ORIGINAL")
    print(f"Marker size: {MARKER_SIZE_M} m")
    print(f"Bekannte IDs: {sorted(KNOWN_MARKER_IDS)}")
    print(f"OV9281-Stativ: {sorted(STAND_MARKERS['OV9281_STAND'])}")
    print(f"ELP1-Stativ: {sorted(STAND_MARKERS['ELP1_STAND'])}")
    print(f"ELP2-Stativ: {sorted(STAND_MARKERS['ELP2_STAND'])}")

    cfg2 = load_yaml("stereo_config.yaml")
    cfg1 = load_yaml("stereo_config_elp1.yaml")
    cfgov = load_yaml("stereo_config_ov9281.yaml")

    rigs = {
        "ELP2": make_rig(cfg2),
        "ELP1": make_rig(cfg1),
        "OV9281": make_rig(cfgov),
    }

    indices = load_camera_indices()

    print("")
    print("Kamera-Indizes:")
    for k, v in indices.items():
        print(f"  {k}: {v}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.05
    params.maxMarkerPerimeterRate = 0.80

    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap2 = cv2.VideoCapture(indices["ELP2"])
    cap1 = cv2.VideoCapture(indices["ELP1"])
    capovL = cv2.VideoCapture(indices["OV9281 L"])
    capovR = cv2.VideoCapture(indices["OV9281 R"])

    caps = [cap2, cap1, capovL, capovR]

    for cap in caps:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    capovL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capovL.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    capovR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capovR.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    if not cap2.isOpened():
        raise SystemExit("FEHLER: ELP2 Kamera konnte nicht geoeffnet werden.")
    if not cap1.isOpened():
        raise SystemExit("FEHLER: ELP1 Kamera konnte nicht geoeffnet werden.")
    if not capovL.isOpened():
        raise SystemExit("FEHLER: OV9281 L Kamera konnte nicht geoeffnet werden.")
    if not capovR.isOpened():
        raise SystemExit("FEHLER: OV9281 R Kamera konnte nicht geoeffnet werden.")

    pair_samples = {name: [] for name in PAIR_RULES}

    print("")
    print("Erwartete Sicht:")
    for cam_name, mids in EXPECTED_VISIBLE_MARKERS.items():
        print(f"  {cam_name}: {sorted(mids)}")

    print("")
    print(f"ArUco Frames sammeln: maximal {N_MAX_FRAMES}, mindestens {N_MIN_FRAMES}")
    print(f"Zeitlimit: {TIME_LIMIT_S:.1f} Sekunden")

    start_time = time.time()

    last_detections = {}

    for frame_idx in range(N_MAX_FRAMES):
        now = time.time()
        elapsed = now - start_time

        if elapsed > TIME_LIMIT_S:
            print(f"Zeitlimit erreicht nach {elapsed:.1f}s.")
            break

        ret2, f2 = cap2.read()
        ret1, f1 = cap1.read()
        retovL, fovL = capovL.read()
        retovR, fovR = capovR.read()

        if not (ret2 and ret1 and retovL and retovR):
            continue

        f2l, f2r = split_side_by_side(f2)
        f1l, f1r = split_side_by_side(f1)

        detections = {
            "ELP2": detect_valid_markers_rig(f2l, f2r, "ELP2", rigs["ELP2"], detector),
            "ELP1": detect_valid_markers_rig(f1l, f1r, "ELP1", rigs["ELP1"], detector),
            "OV9281": detect_valid_markers_rig(fovL, fovR, "OV9281", rigs["OV9281"], detector),
        }

        last_detections = detections

        for pair_name, rule in PAIR_RULES.items():
            target = rule["target"]
            source = rule["source"]

            for marker in rule["markers"]:
                target_obs_list = detections.get(target, {}).get(marker, [])
                source_obs_list = detections.get(source, {}).get(marker, [])

                if not target_obs_list or not source_obs_list:
                    continue

                for target_obs in target_obs_list:
                    for source_obs in source_obs_list:
                        R_rel, T_rel, score = relative_from_common_marker(target_obs, source_obs)

                        pair_samples[pair_name].append(
                            {
                                "R": R_rel,
                                "T": T_rel,
                                "score": score,
                                "marker": marker,
                                "target_side": target_obs["side"],
                                "source_side": source_obs["side"],
                            }
                        )

        if frame_idx in [0, 10, 30, 60, 90, 120]:
            print(
                f"  Frame {frame_idx} | "
                f"ELP2: {status_text(detections['ELP2'])} | "
                f"ELP1: {status_text(detections['ELP1'])} | "
                f"OV9281: {status_text(detections['OV9281'])} | "
                f"Zeit {elapsed:.1f}s"
            )

        if frame_idx >= N_MIN_FRAMES:
            if all(len(pair_samples[pair]) >= MIN_PAIR_SAMPLES for pair in PAIR_RULES):
                print(
                    f"Adaptive ArUco-Sammlung beendet: alle drei Beziehungen vorhanden "
                    f"nach {frame_idx + 1} Frames und {elapsed:.1f}s."
                )
                break

    for cap in caps:
        cap.release()

    print("")
    print("ArUco Sample-Uebersicht:")

    for pair_name, rule in PAIR_RULES.items():
        marker_text = ",".join(str(x) for x in rule["markers"])
        print(f"  {rule['description']} ID {marker_text}: {len(pair_samples[pair_name])} Rohsamples")

    print("")
    estimates = {}

    for pair_name, rule in PAIR_RULES.items():
        estimates[pair_name] = estimate_pair(pair_samples[pair_name], rule["description"])

    estimates = apply_triangle_combo_selection(estimates)

    print("")
    if estimates["ELP1_to_ELP2"] is not None:
        save_legacy("ELP1_to_ELP2", estimates["ELP1_to_ELP2"])
        save_state("ELP1_to_ELP2", estimates["ELP1_to_ELP2"])

    if estimates["OV9281_to_ELP2"] is not None:
        save_legacy("OV9281_to_ELP2", estimates["OV9281_to_ELP2"])
        save_state("OV9281_to_ELP2", estimates["OV9281_to_ELP2"])

    if estimates["OV9281_to_ELP1"] is not None:
        save_legacy("OV9281_to_ELP1", estimates["OV9281_to_ELP1"])
        save_state("OV9281_to_ELP1", estimates["OV9281_to_ELP1"])

    if all(estimates.get(x) is not None for x in ["ELP1_to_ELP2", "OV9281_to_ELP2", "OV9281_to_ELP1"]):
        via = compose(estimates["ELP1_to_ELP2"], estimates["OV9281_to_ELP1"])
        direct = estimates["OV9281_to_ELP2"]

        t_diff = float(np.linalg.norm(direct["T"].reshape(3) - via["T"].reshape(3)) * 100.0)
        r_diff = rotation_diff_deg(direct["R"], via["R"])

        print(
            f"Dreiecks-Konsistenz Diagnose: "
            f"Translation={t_diff:.1f} cm | Rotation={r_diff:.2f} Grad"
        )

        if t_diff <= 12.0 and r_diff <= 8.0:
            print("Dreiecksschluss: OK")
        else:
            print("WARNUNG: Dreiecksschluss noch nicht ideal. Als Diagnose verwenden, nicht als harte Bedingung.")

    print("")
    print("ArUco Kalibrierung gespeichert.")
    for pair_name, est in estimates.items():
        if est is not None:
            print(f"  {pair_name}: T={est['T'] * 100.0} cm")

    print("")
    print("Positionskalibrierung abgeschlossen.")
    print("Hintergrundaufnahme wurde bewusst uebersprungen.")
    print("Gespeichert wurden ArUco-Positionsdaten und aruco_state active Dateien.")


if __name__ == "__main__":
    main()
