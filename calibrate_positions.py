import cv2, numpy as np, yaml, sys
from pathlib import Path

print("=== ANTHRO3D Kalibrierung ===")
print("Raum muss LEER sein — keine Person, keine beweglichen Objekte")
print()

# Kalibrierungen laden
with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_ov9281.yaml").expanduser()) as f:
    cfgov = yaml.safe_load(f)

R_rel_elp1 = np.load(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser())
T_rel_elp1 = np.load(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser()) * 100

def make_maps(cfg, size=(1600,1200), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5, return_R1=False):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,flags=flags,alpha=alpha)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    out = (ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0]))
    if return_R1:
        return out + (R1,)
    return out

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2 = make_maps(cfg2)
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1 = make_maps(cfg1)
mlov1,mlov2,mrov1,mrov2,fxov,cxov,cyov,blov,R1_ov = make_maps(cfgov,(1280,800), flags=0, alpha=-1, return_R1=True)
print(f"OV9281 Rectify Kalibrierung: fx={fxov:.1f} cx={cxov:.1f} cy={cyov:.1f} baseline={blov*100:.1f}cm flags=0")

lm=cv2.StereoSGBM_create(minDisparity=4,numDisparities=128,blockSize=9,
    P1=8*3*81,P2=32*3*81,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=150,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm=cv2.ximgproc.createRightMatcher(lm)
wls=cv2.ximgproc.createDisparityWLSFilter(lm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

lm_ov=cv2.StereoSGBM_create(minDisparity=2,numDisparities=64,blockSize=7,
    P1=8*3*49,P2=32*3*49,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm_ov=cv2.ximgproc.createRightMatcher(lm_ov)
wls_ov=cv2.ximgproc.createDisparityWLSFilter(lm_ov)
wls_ov.setLambda(8000); wls_ov.setSigmaColor(1.5)

# Kameras
cap2=cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
cap1=cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
capovL=cv2.VideoCapture(1)
capovL.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovL.set(cv2.CAP_PROP_FRAME_HEIGHT,800)
capovR=cv2.VideoCapture(2)
capovR.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovR.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

def get_disp(cap, ml1, ml2, mr1, mr2, lm_, rm_, wls_, split=True):
    ret,frame=cap.read()
    if not ret: return None
    if split:
        w=frame.shape[1]//2
        fl=cv2.remap(frame[:,:w],ml1,ml2,cv2.INTER_LINEAR)
        fr=cv2.remap(frame[:,w:],mr1,mr2,cv2.INTER_LINEAR)
    else:
        fl=cv2.remap(frame,ml1,ml2,cv2.INTER_LINEAR)
        fr=cv2.remap(frame,mr1,mr2,cv2.INTER_LINEAR)
    fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
    fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
    gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    dl=lm_.compute(gl,gr); dr=rm_.compute(gr,gl)
    d=wls_.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4]=0
    return d

# OV9281 Transformation
aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params=cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate=0.05  # Mindestgröße ~80px bei 1600px Bild
params.maxMarkerPerimeterRate=0.5   # Maxgröße
detector=cv2.aruco.ArucoDetector(aruco_dict,params)
MARKER_SIZE=0.19
VALID_IDS={2,3,10}          # Nur bekannte Stativ-Marker
MAX_DIST=5.0                # Max 5m Entfernung
MIN_AREA=80*80              # Mindest-Pixelfläche

def validate_marker(corner, tvec):
    """Prüft ob Marker valide ist (Größe, Distanz, Aspect-Ratio)."""
    # Distanz prüfen
    dist = np.linalg.norm(tvec)
    if dist > MAX_DIST: return False
    # Pixelfläche prüfen
    pts = corner[0]
    area = cv2.contourArea(pts)
    if area < MIN_AREA: return False
    # Aspect-Ratio prüfen (muss quadratisch sein ±20%)
    w = np.linalg.norm(pts[0]-pts[1])
    h = np.linalg.norm(pts[1]-pts[2])
    if h == 0: return False
    ratio = w/h
    if ratio < 0.7 or ratio > 1.3: return False
    return True
obj_pts=np.array([[-MARKER_SIZE/2,MARKER_SIZE/2,0],[MARKER_SIZE/2,MARKER_SIZE/2,0],
                   [MARKER_SIZE/2,-MARKER_SIZE/2,0],[-MARKER_SIZE/2,-MARKER_SIZE/2,0]],dtype=np.float32)
K2=np.array(cfg2['camera_matrix_l']); d2=np.array(cfg2['dist_l'])
Kov=np.array(cfgov['camera_matrix_l']); dov=np.array(cfgov['dist_l'])



print("Schritt 1: Adaptive ArUco Dreiecks-Diagnose mit linker und rechter Kameraseite...")

import time as _time

MARKER_SIZE_M = 0.19
KNOWN_MARKER_IDS = {2, 3, 10}

MIN_MARKER_AREA_PX = 40 * 40
MAX_MARKER_DISTANCE_M = 6.0
MIN_MARKER_RATIO = 0.60
MAX_MARKER_RATIO = 1.40
MAX_REPROJECTION_ERROR_PX = 6.0

N_ARUCO_FRAMES_MAX = 150
N_ARUCO_FRAMES_MIN = 40
MAX_ARUCO_SECONDS = 20.0
OPTIONAL_WAIT_SECONDS = 12.0

MIN_PAIR_SAMPLES = 10
TARGET_REQUIRED_SAMPLES = 30
MAX_PAIR_STD_CM_WARN = 8.0

NO_COMMON_MARKER_WARN_FRAMES = 35
NO_CAMERA_MARKER_WARN_FRAMES = 35

TRIANGLE_WARN_TRANSLATION_CM = 8.0
TRIANGLE_WARN_ROTATION_DEG = 5.0

obj_pts = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
], dtype=np.float32)

EXPECTED_VISIBLE_MARKERS = {
    "ELP2": {2, 3},
    "ELP1": {2, 10},
    "OV9281": {3, 10},
}

PAIR_RULES = {
    "ELP1_to_ELP2": {
        "target": "ELP2",
        "source": "ELP1",
        "marker": 2,
        "description": "ELP1 zu ELP2 ueber Marker ID 2",
        "required": True,
    },
    "OV9281_to_ELP2": {
        "target": "ELP2",
        "source": "OV9281",
        "marker": 3,
        "description": "OV9281 zu ELP2 ueber Marker ID 3",
        "required": True,
    },
    "OV9281_to_ELP1": {
        "target": "ELP1",
        "source": "OV9281",
        "marker": 10,
        "description": "OV9281 zu ELP1 ueber Marker ID 10",
        "required": False,
    },
}

K2_raw = np.array(cfg2["camera_matrix_l"], dtype=np.float64)
d2_raw = np.array(cfg2["dist_l"], dtype=np.float64)
K2r_raw = np.array(cfg2["camera_matrix_r"], dtype=np.float64)
d2r_raw = np.array(cfg2["dist_r"], dtype=np.float64)

K1_raw = np.array(cfg1["camera_matrix_l"], dtype=np.float64)
d1_raw = np.array(cfg1["dist_l"], dtype=np.float64)
K1r_raw = np.array(cfg1["camera_matrix_r"], dtype=np.float64)
d1r_raw = np.array(cfg1["dist_r"], dtype=np.float64)

Kov_raw = np.array(cfgov["camera_matrix_l"], dtype=np.float64)
dov_raw = np.array(cfgov["dist_l"], dtype=np.float64)
Kovr_raw = np.array(cfgov["camera_matrix_r"], dtype=np.float64)
dovr_raw = np.array(cfgov["dist_r"], dtype=np.float64)

K2 = K2_raw
d2 = d2_raw
K1 = K1_raw
d1 = d1_raw
Kov = Kov_raw
dov = dov_raw

CAM_RIGS = {
    "ELP2": {"cfg": cfg2, "K_l": K2_raw, "d_l": d2_raw, "K_r": K2r_raw, "d_r": d2r_raw},
    "ELP1": {"cfg": cfg1, "K_l": K1_raw, "d_l": d1_raw, "K_r": K1r_raw, "d_r": d1r_raw},
    "OV9281": {"cfg": cfgov, "K_l": Kov_raw, "d_l": dov_raw, "K_r": Kovr_raw, "d_r": dovr_raw},
}

def marker_reprojection_error(obj_pts, corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    real = corners.reshape(-1, 2)
    err = np.sqrt(((proj - real) ** 2).sum(axis=1))
    return float(np.mean(err))

def marker_geometry_ok(corner, tvec):
    dist_m = float(np.linalg.norm(tvec))
    if dist_m > MAX_MARKER_DISTANCE_M:
        return False

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

    return True

def rotation_diff_deg(R_a, R_b):
    R_delta = R_a @ R_b.T
    trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))

def right_pose_to_left_pose(rvec_r, tvec_r, rig_cfg):
    R_lr = np.array(rig_cfg["R"], dtype=np.float64)
    T_lr = np.array(rig_cfg["T"], dtype=np.float64).reshape(3)

    R_marker_r, _ = cv2.Rodrigues(rvec_r)

    R_marker_l = R_lr.T @ R_marker_r
    T_marker_l = R_lr.T @ (tvec_r.reshape(3) - T_lr)

    rvec_l, _ = cv2.Rodrigues(R_marker_l)

    return rvec_l, T_marker_l.reshape(3, 1)

def detect_valid_markers_single(gray, K, dist, cam_name, side_name, rig_cfg=None):
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    idsf = ids.flatten()
    result = {}

    for idx, mid_raw in enumerate(idsf):
        mid = int(mid_raw)

        if mid not in KNOWN_MARKER_IDS:
            continue

        if mid not in EXPECTED_VISIBLE_MARKERS.get(cam_name, set()):
            continue

        corner = corners[idx]

        ok, rvec, tvec = cv2.solvePnP(obj_pts, corner, K, dist)
        if not ok:
            continue

        if not marker_geometry_ok(corner, tvec):
            continue

        err = marker_reprojection_error(obj_pts, corner, rvec, tvec, K, dist)

        if err > MAX_REPROJECTION_ERROR_PX:
            continue

        if side_name == "R":
            rvec_l, tvec_l = right_pose_to_left_pose(rvec, tvec, rig_cfg)
        else:
            rvec_l, tvec_l = rvec, tvec

        obs = {
            "rvec": rvec_l,
            "tvec": tvec_l,
            "err": err,
            "side": side_name,
        }

        result.setdefault(mid, []).append(obs)

    return result

def detect_valid_markers_rig(gray_l, gray_r, cam_name):
    rig = CAM_RIGS[cam_name]

    left = detect_valid_markers_single(
        gray_l,
        rig["K_l"],
        rig["d_l"],
        cam_name,
        "L",
        rig["cfg"]
    )

    right = detect_valid_markers_single(
        gray_r,
        rig["K_r"],
        rig["d_r"],
        cam_name,
        "R",
        rig["cfg"]
    )

    merged = {}

    for mid, obs_list in left.items():
        merged.setdefault(mid, []).extend(obs_list)

    for mid, obs_list in right.items():
        merged.setdefault(mid, []).extend(obs_list)

    for mid in list(merged.keys()):
        merged[mid] = sorted(merged[mid], key=lambda obs: obs["err"])

    return merged

def relative_from_common_marker(target_obs, source_obs):
    rvec_t = target_obs["rvec"]
    tvec_t = target_obs["tvec"]
    rvec_s = source_obs["rvec"]
    tvec_s = source_obs["tvec"]

    R_t, _ = cv2.Rodrigues(rvec_t)
    R_s, _ = cv2.Rodrigues(rvec_s)

    R_rel = R_t @ R_s.T
    T_rel = tvec_t.flatten() - R_rel @ tvec_s.flatten()

    score = float(target_obs["err"] + source_obs["err"])

    return R_rel, T_rel, score

def robust_estimate(pair_name, values, required=True):
    if len(values) < MIN_PAIR_SAMPLES:
        if required:
            raise SystemExit(
                f"FEHLER: {pair_name} konnte nicht stabil berechnet werden. "
                f"Nur {len(values)} gueltige Samples. "
                "Bitte ArUco-Sicht ueberpruefen."
            )
        return None, None, None

    R_list = []
    T_list = []
    score_list = []

    for R_rel, T_rel, score in values:
        R_list.append(R_rel)
        T_list.append(T_rel)
        score_list.append(score)

    R_arr = np.array(R_list)
    T_arr = np.array(T_list)
    score_arr = np.array(score_list)

    dists = np.linalg.norm(T_arr, axis=1)
    med_dist = np.median(dists)
    mask_dist = dists < med_dist * 1.3

    R_arr = R_arr[mask_dist]
    T_arr = T_arr[mask_dist]
    score_arr = score_arr[mask_dist]

    if len(T_arr) < MIN_PAIR_SAMPLES:
        if required:
            raise SystemExit(
                f"FEHLER: {pair_name} ist nach Distanzfilter instabil. "
                "Bitte Marker-Sicht und Stative pruefen."
            )
        return None, None, None

    if len(T_arr) >= 20:
        score_limit = np.percentile(score_arr, 70)
        mask_score = score_arr <= score_limit

        R_arr = R_arr[mask_score]
        T_arr = T_arr[mask_score]
        score_arr = score_arr[mask_score]

    if len(T_arr) < MIN_PAIR_SAMPLES:
        if required:
            raise SystemExit(
                f"FEHLER: {pair_name} ist nach Reprojection-Filter instabil. "
                "Bitte Marker-Sicht und Licht pruefen."
            )
        return None, None, None

    T_med = np.median(T_arr, axis=0)

    rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_arr])
    rvec_med = np.median(rvecs, axis=0)
    R_med, _ = cv2.Rodrigues(rvec_med)

    std_cm = T_arr.std(axis=0) * 100
    std_max_cm = float(np.max(std_cm))
    reproj_med = float(np.median(score_arr))
    reproj_max = float(np.max(score_arr))

    print(
        f"{pair_name}: OK | "
        f"T={T_med*100} cm | "
        f"Distanz={np.linalg.norm(T_med)*100:.1f} cm | "
        f"Samples={len(T_arr)} | "
        f"Streuung max={std_max_cm:.1f} cm | "
        f"Reprojection median={reproj_med:.2f}px | "
        f"Reprojection max={reproj_max:.2f}px"
    )

    if std_max_cm > MAX_PAIR_STD_CM_WARN:
        print(
            f"WARNUNG: {pair_name} streut noch relativ stark. "
            f"Streuung max={std_max_cm:.1f} cm. Kalibrierung wird trotzdem fortgesetzt."
        )

    return R_med, T_med, {
        "samples": len(T_arr),
        "std_max_cm": std_max_cm,
        "reproj_med": reproj_med,
        "reproj_max": reproj_max,
    }

pair_samples = {name: [] for name in PAIR_RULES}

pair_missing_counter = {name: 0 for name in PAIR_RULES}
pair_warned = {name: False for name in PAIR_RULES}

camera_missing_counter = {
    "ELP2": 0,
    "ELP1": 0,
    "OV9281": 0,
}

camera_warned = {
    "ELP2": False,
    "ELP1": False,
    "OV9281": False,
}

required_pair_names = [
    name for name, rule in PAIR_RULES.items()
    if rule["required"]
]

t_start = _time.time()
frames_processed = 0

print(f"ArUco Frames sammeln: maximal {N_ARUCO_FRAMES_MAX}, mindestens {N_ARUCO_FRAMES_MIN}")
print(f"Zeitlimit: {MAX_ARUCO_SECONDS:.1f} Sekunden")
print("Bekannte IDs:", sorted(KNOWN_MARKER_IDS))
print("ELP2 erwartet:", sorted(EXPECTED_VISIBLE_MARKERS["ELP2"]))
print("ELP1 erwartet:", sorted(EXPECTED_VISIBLE_MARKERS["ELP1"]))
print("OV9281 erwartet:", sorted(EXPECTED_VISIBLE_MARKERS["OV9281"]))
print("Hinweis: Je Kamerasystem werden linke und rechte Sicht geprueft.")
print("Hinweis: Linke und rechte Beobachtungen werden beide als Samples genutzt.")

for frame_i in range(N_ARUCO_FRAMES_MAX):
    elapsed = _time.time() - t_start

    if elapsed > MAX_ARUCO_SECONDS and frame_i >= N_ARUCO_FRAMES_MIN:
        print(f"ArUco Zeitlimit erreicht nach {elapsed:.1f} Sekunden und {frame_i} Frames.")
        break

    ret2, f2 = cap2.read()
    ret1, f1 = cap1.read()
    retovL, fovL = capovL.read()
    retovR, fovR = capovR.read()

    if not ret2 or not ret1 or not retovL or not retovR:
        continue

    frames_processed += 1

    fl2 = f2[:, :f2.shape[1]//2]
    fr2 = f2[:, f2.shape[1]//2:]

    fl1 = f1[:, :f1.shape[1]//2]
    fr1 = f1[:, f1.shape[1]//2:]

    g2l = cv2.equalizeHist(cv2.cvtColor(fl2, cv2.COLOR_BGR2GRAY))
    g2r = cv2.equalizeHist(cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY))

    g1l = cv2.equalizeHist(cv2.cvtColor(fl1, cv2.COLOR_BGR2GRAY))
    g1r = cv2.equalizeHist(cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY))

    govl = cv2.equalizeHist(cv2.cvtColor(fovL, cv2.COLOR_BGR2GRAY))
    govr = cv2.equalizeHist(cv2.cvtColor(fovR, cv2.COLOR_BGR2GRAY))

    detections = {
        "ELP2": detect_valid_markers_rig(g2l, g2r, "ELP2"),
        "ELP1": detect_valid_markers_rig(g1l, g1r, "ELP1"),
        "OV9281": detect_valid_markers_rig(govl, govr, "OV9281"),
    }

    for cam_name, obs in detections.items():
        if len(obs) == 0:
            camera_missing_counter[cam_name] += 1

            if (
                camera_missing_counter[cam_name] >= NO_CAMERA_MARKER_WARN_FRAMES
                and not camera_warned[cam_name]
            ):
                print(
                    f"WARNUNG: {cam_name} erkennt seit {camera_missing_counter[cam_name]} Frames "
                    "keinen erwarteten ArUco-Marker stabil. "
                    "Bitte Sicht, Licht und Markerposition pruefen."
                )
                camera_warned[cam_name] = True
        else:
            camera_missing_counter[cam_name] = 0
            camera_warned[cam_name] = False

    for pair_name, rule in PAIR_RULES.items():
        target = rule["target"]
        source = rule["source"]
        marker = rule["marker"]

        target_has = marker in detections[target] and len(detections[target][marker]) > 0
        source_has = marker in detections[source] and len(detections[source][marker]) > 0

        if target_has and source_has:
            frame_values = []

            for target_obs in detections[target][marker]:
                for source_obs in detections[source][marker]:
                    R_rel, T_rel, score = relative_from_common_marker(
                        target_obs,
                        source_obs
                    )
                    frame_values.append((score, R_rel, T_rel))

            frame_values = sorted(frame_values, key=lambda x: x[0])

            # Alle guten Kombinationen werden als Samples genutzt.
            # Zur Sicherheit werden pro Frame und Paar maximal vier Kombinationen genommen.
            for score, R_rel, T_rel in frame_values[:4]:
                pair_samples[pair_name].append((R_rel, T_rel, score))

            pair_missing_counter[pair_name] = 0
            pair_warned[pair_name] = False
        else:
            pair_missing_counter[pair_name] += 1

            if (
                pair_missing_counter[pair_name] >= NO_COMMON_MARKER_WARN_FRAMES
                and not pair_warned[pair_name]
            ):
                print(
                    f"WARNUNG: {rule['description']} nicht gemeinsam sichtbar "
                    f"seit {pair_missing_counter[pair_name]} Frames. "
                    f"{target} sieht Marker {marker}: {target_has}, "
                    f"{source} sieht Marker {marker}: {source_has}. "
                    "Kurzzeitiges Flackern ist erlaubt, bei laengerem Ausfall bitte Marker neu ausrichten."
                )
                pair_warned[pair_name] = True

    if frame_i in [0, 10, 30, 60, 90, 120, 149]:
        msg_parts = []

        for cam_name in ["ELP2", "ELP1", "OV9281"]:
            details = []

            for mid, obs_list in detections[cam_name].items():
                for obs in obs_list:
                    details.append(f"{mid}{obs['side']}")

            msg_parts.append(f"{cam_name}: {sorted(details)}")

        elapsed_now = _time.time() - t_start
        print("  Frame", frame_i, "|", " | ".join(msg_parts), f"| Zeit {elapsed_now:.1f}s")

    elapsed_now = _time.time() - t_start

    required_ready = all(
        len(pair_samples[name]) >= TARGET_REQUIRED_SAMPLES
        for name in required_pair_names
    )

    optional_ready = len(pair_samples["OV9281_to_ELP1"]) >= MIN_PAIR_SAMPLES

    if frame_i >= N_ARUCO_FRAMES_MIN and required_ready:
        if optional_ready:
            print(
                f"Adaptive ArUco-Sammlung beendet: Hauptbeziehungen und optionale Dreieckskontrolle vorhanden "
                f"nach {frame_i + 1} Frames und {elapsed_now:.1f} Sekunden."
            )
            break

        if elapsed_now >= OPTIONAL_WAIT_SECONDS:
            print(
                f"Adaptive ArUco-Sammlung beendet: Hauptbeziehungen stabil, optionale ID 10 Verbindung fehlt noch "
                f"nach {frame_i + 1} Frames und {elapsed_now:.1f} Sekunden."
            )
            break

elapsed_total = _time.time() - t_start

print(f"ArUco Sammlung abgeschlossen nach {frames_processed} gelesenen Frames und {elapsed_total:.1f} Sekunden.")
print("ArUco Sample-Uebersicht:")

for pair_name, samples in pair_samples.items():
    print(f"  {PAIR_RULES[pair_name]['description']}: {len(samples)} Rohsamples")

R_rel_elp1, T_rel_elp1, stats_elp1 = robust_estimate(
    "ELP1 zu ELP2 ueber Marker ID 2",
    pair_samples["ELP1_to_ELP2"],
    required=True
)

R_rel_ov, T_rel_ov, stats_ov = robust_estimate(
    "OV9281 zu ELP2 ueber Marker ID 3",
    pair_samples["OV9281_to_ELP2"],
    required=True
)

R_ov_to_elp1, T_ov_to_elp1, stats_ov_elp1 = robust_estimate(
    "OV9281 zu ELP1 ueber Marker ID 10",
    pair_samples["OV9281_to_ELP1"],
    required=False
)

if R_ov_to_elp1 is not None:
    R_ov_to_elp2_via_elp1 = R_rel_elp1 @ R_ov_to_elp1
    T_ov_to_elp2_via_elp1 = R_rel_elp1 @ T_ov_to_elp1 + T_rel_elp1

    triangle_t_diff_cm = float(np.linalg.norm(T_rel_ov - T_ov_to_elp2_via_elp1) * 100)
    triangle_r_diff_deg = rotation_diff_deg(R_rel_ov, R_ov_to_elp2_via_elp1)

    print(
        f"Dreiecks-Konsistenz Diagnose: "
        f"Translation={triangle_t_diff_cm:.1f} cm | "
        f"Rotation={triangle_r_diff_deg:.2f} Grad"
    )

    if triangle_t_diff_cm > TRIANGLE_WARN_TRANSLATION_CM or triangle_r_diff_deg > TRIANGLE_WARN_ROTATION_DEG:
        print(
            "WARNUNG: ArUco Dreieck ist noch nicht ideal konsistent. "
            "Kalibrierung wird gespeichert, aber spaeter erneut pruefen."
        )
else:
    print(
        "WARNUNG: Die optionale Dreiecksverbindung OV9281 zu ELP1 ueber ID 10 "
        "konnte nicht stabil berechnet werden. "
        "Die zwei Hauptbeziehungen werden trotzdem gespeichert."
    )

np.save(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser(), R_rel_elp1)
np.save(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser(), T_rel_elp1)

np.save(Path("~/anthro3d/R_rel_ov9281_to_elp2.npy").expanduser(), R_rel_ov)
np.save(Path("~/anthro3d/T_rel_ov9281_to_elp2.npy").expanduser(), T_rel_ov)

OV_OK = True

print("ArUco Kalibrierung gespeichert.")
print(f"  ELP1 zu ELP2 T: {T_rel_elp1*100} cm")
print(f"  OV9281 zu ELP2 T: {T_rel_ov*100} cm")

print()
print("Positionskalibrierung abgeschlossen.")
print("Hintergrundaufnahme wird in calibrate_positions.py bewusst uebersprungen.")
print("Gespeichert wurden nur die ArUco-Positionsdaten.")

try:
    cap2.release()
    cap1.release()
    capovL.release()
    capovR.release()
except Exception:
    pass

raise SystemExit(0)


print("Scan-Korridor wird berechnet...")

# Kamerapositionen im ELP2-Koordinatensystem
cam_positions = np.array([
    [0.0, 0.0, 0.0],          # ELP2 = Ursprung
        T_rel_elp1.flatten(),      # ELP1
])
if OV_OK:
    cam_positions = np.vstack([cam_positions, T_rel_ov])

# Schwerpunkt der Kameras = Zentrum des Korridors
center = cam_positions.mean(axis=0)

# Konvexe Hülle der Kamerapositionen in XZ-Ebene (Draufsicht)
# Korridor = Dreieck zwischen den 3 Kameras, leicht nach innen versetzt
from scipy.spatial import ConvexHull
try:
    hull_pts = cam_positions[:, [0,2]]  # nur X und Z
    hull = ConvexHull(hull_pts)
    hull_vertices = hull_pts[hull.vertices]
    print(f"  Korridor-Dreieck: {len(hull.vertices)} Ecken")
    # 20cm nach innen versetzen (Puffer damit Kameras selbst nicht mitgescannt werden)
    INSET = 0.20
    hull_inset = []
    for v in hull_vertices:
        direction = center[[0,2]] - v
        direction = direction / np.linalg.norm(direction)
        hull_inset.append(v + direction * INSET)
    hull_inset = np.array(hull_inset)
    CORRIDOR_OK = True
except Exception as e:
    print(f"  ⚠ Korridor-Berechnung fehlgeschlagen: {e}")
    hull_inset = None
    CORRIDOR_OK = False

# Y-Bereich: Boden bis Decke (realistisch -0.5m bis +2.5m)
Y_MIN = -0.3  # 30cm unter Kamera
Y_MAX = 2.5   # 2.5m über Boden

def point_in_corridor(pts):
    """Prüft ob Punkte im Scan-Korridor liegen."""
    if not CORRIDOR_OK or hull_inset is None:
        return np.ones(len(pts), dtype=bool)
    # Y-Filter
    y_ok = (pts[:,1] > -Y_MAX) & (pts[:,1] < Y_MIN)
    # XZ-Polygon-Filter
    from matplotlib.path import Path as MplPath
    polygon = MplPath(hull_inset)
    xz_ok = polygon.contains_points(pts[:,[0,2]])
    return y_ok & xz_ok

np.save(Path("~/anthro3d/scan_corridor_hull.npy").expanduser(),
        hull_inset if CORRIDOR_OK else np.zeros((0,2)))
np.save(Path("~/anthro3d/scan_corridor_y.npy").expanduser(),
        np.array([Y_MIN, Y_MAX]))
print(f"  ✓ Scan-Korridor gespeichert")

# Schritt 2: Hintergrund-Disparitätskarten aufnehmen
print("Schritt 2: Hintergrund aufnehmen (30 Frames)...")
f2s=[]; f1s=[]; fovs=[]
for i in range(30):
    d=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2,lm,rm,wls)
    if d is not None: f2s.append(d)
    d=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2,lm,rm,wls)
    if d is not None: f1s.append(d)
    if OV_OK:
        retL,fL=capovL.read(); retR,fR=capovR.read()
        if retL and retR:
            fl=cv2.remap(fL,mlov1,mlov2,cv2.INTER_LINEAR)
            fr=cv2.remap(fR,mrov1,mrov2,cv2.INTER_LINEAR)
            fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
            fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
            gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
            gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            dl=lm_ov.compute(gl,gr); dr=rm_ov.compute(gr,gl)
            d=wls_ov.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
            d[d<4]=0
            fovs.append(d)
    if i%10==0: print(f"  {i+1}/30")

bg2=np.median(f2s,axis=0).astype(np.float32)
bg1=np.median(f1s,axis=0).astype(np.float32)
st2=(np.std(f2s,axis=0)<3.0)
st1=(np.std(f1s,axis=0)<3.0)
print(f"  ELP2 stabil: {st2.mean()*100:.0f}%  ELP1 stabil: {st1.mean()*100:.0f}%")

bgov=stov=None
if fovs:
    bgov=np.median(fovs,axis=0).astype(np.float32)
    stov=(np.std(fovs,axis=0)<3.0)
    print(f"  OV9281 stabil: {stov.mean()*100:.0f}%")

# Schritt 3: Leeren Scan-Korridor als 3D-Punktwolke aufnehmen
print("Schritt 3: Stabilen Hintergrund aufnehmen (Personen werden herausgefiltert)...")

def disp_to_pts(d, fx, cx, cy, bl, R_rel=None, T_rel=None):
    rows,ci=np.where(d>4)
    if len(rows)==0: return None
    dv=d[rows,ci]
    Z=fx*bl/dv; X=(ci-cx)*Z/fx; Y=(rows-cy)*Z/fx
    pts=np.stack([X,Y,Z],axis=1)
    zm=(Z>0.3)&(Z<5.0)
    pts=pts[zm]
    if R_rel is not None and len(pts)>0:
        pts=(R_rel@pts.T).T+T_rel.flatten()
    return pts

# Viele Frames aufnehmen — Personen bewegen sich, Hintergrund bleibt stabil
N_FRAMES = 60
voxel = 0.05  # 5cm Voxelgröße
print(f"  {N_FRAMES} Frames aufnehmen...")

# Voxel-Zähler: wie oft wurde jeder Voxel gesehen?
voxel_count = {}
voxel_pts = {}

for fi in range(N_FRAMES):
    d2=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2,lm,rm,wls)
    d1=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2,lm,rm,wls)
    frame_pts = []
    if d2 is not None:
        p=disp_to_pts(d2,fx2,cx2,cy2,bl2)
        if p is not None: frame_pts.append(p)
    if d1 is not None:
        p=disp_to_pts(d1,fx1,cx1,cy1,bl1,R_rel_elp1,T_rel_elp1)
        if p is not None: frame_pts.append(p)
    if OV_OK and R_rel_ov is not None:
        retL,fL=capovL.read(); retR,fR=capovR.read()
        if retL and retR:
            fl=cv2.remap(fL,mlov1,mlov2,cv2.INTER_LINEAR)
            fr=cv2.remap(fR,mrov1,mrov2,cv2.INTER_LINEAR)
            fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
            fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
            gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
            gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            dl=lm_ov.compute(gl,gr); dr=rm_ov.compute(gr,gl)
            dov=wls_ov.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
            dov[dov<4]=0
            p=disp_to_pts(dov,fxov,cxov,cyov,blov)
            if p is not None:
                # OV9281-Hintergrundpunkte an scan3d.py angleichen:
                # Rectified Stereo zurückdrehen, Achsen korrigieren, dann nach ELP2 transformieren.
                pov_raw = (R1_ov.T @ p.T).T
                flip_ov = np.array([-1.0, 1.0, -1.0], dtype=float).reshape(1, 3)
                p = (R_rel_ov.T @ (pov_raw * flip_ov).T).T + T_rel_ov.reshape(1, 3)
            if p is not None: frame_pts.append(p)
    if frame_pts:
        all_p = np.vstack(frame_pts)
        idx = (all_p/voxel).astype(np.int32)
        keys = idx[:,0]*1000000+idx[:,1]*1000+idx[:,2]
        for k,pt in zip(keys, all_p):
            if k not in voxel_count:
                voxel_count[k] = 0
                voxel_pts[k] = pt
            voxel_count[k] += 1
    # Fortschrittsbalken
    pct = int((fi+1)/N_FRAMES*50)
    bar = '█'*pct + '░'*(50-pct)
    print(f"  [{bar}] {fi+1}/{N_FRAMES}", end='\r')

# Nur Voxels die in mehr als 50% der Frames gesehen wurden = stabil = Hintergrund
# Personen bewegen sich → selten gesehen → werden nicht als Hintergrund markiert
min_count = N_FRAMES * 0.4  # 40% der Frames
stable_keys = [k for k,c in voxel_count.items() if c >= min_count]
if stable_keys:
    bg_pts = np.array([voxel_pts[k] for k in stable_keys])
    print(f"  Stabile Hintergrund-Voxels: {len(bg_pts)} (von {len(voxel_count)} gesamt)")
    print(f"  Personen/Bewegung entfernt: {len(voxel_count)-len(bg_pts)} Voxels")
else:
    bg_pts = np.zeros((0,3))
    print("  ⚠ Keine stabilen Voxels gefunden")

# Hintergrund-Scanbox begrenzen
# Entfernt unrealistische Ausreißer, bevor bg_pts gespeichert wird.
# Werte sind in Metern. ELP2 bleibt Weltursprung.
if bg_pts is not None and len(bg_pts) > 0:
    before_bg_box = len(bg_pts)

    bg_box = (
        (bg_pts[:,0] > -2.50) & (bg_pts[:,0] < 2.50) &
        (bg_pts[:,1] > -3.00) & (bg_pts[:,1] < 1.50) &
        (bg_pts[:,2] > 0.20) & (bg_pts[:,2] < 5.50)
    )

    bg_pts = bg_pts[bg_box]

    print(
        f"  Hintergrund-Scanbox: {before_bg_box} → {len(bg_pts)} Punkte | "
        f"X=-250..250cm Y=-300..150cm Z=20..550cm"
    )

    if len(bg_pts) > 0:
        print(
            f"  bg_pts Bereich nach Scanbox: "
            f"X={bg_pts[:,0].min()*100:.0f}..{bg_pts[:,0].max()*100:.0f}cm "
            f"Y={bg_pts[:,1].min()*100:.0f}..{bg_pts[:,1].max()*100:.0f}cm "
            f"Z={bg_pts[:,2].min()*100:.0f}..{bg_pts[:,2].max()*100:.0f}cm"
        )


# Alles speichern
out=Path("~/anthro3d/calibration_bg.npz").expanduser()
save_dict=dict(bg2=bg2,bg1=bg1,st2=st2,st1=st1)
if bgov is not None:
    save_dict.update(bgov=bgov,stov=stov)
    save_dict['OV_OK']=np.array([True])
else:
    save_dict['OV_OK']=np.array([False])
save_dict['bg_pts'] = bg_pts
save_dict['bg_voxel'] = np.array([voxel])
np.savez(out,**save_dict)
print(f"✓ Hintergrund gespeichert: {out}")

cap2.release(); cap1.release(); capovL.release(); capovR.release()
print()
print("=== Kalibrierung abgeschlossen ===")
print("Starte jetzt: python3 ~/anthro3d/scan3d.py")
