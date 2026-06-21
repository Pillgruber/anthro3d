import cv2, numpy as np, yaml, sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor
import math

# MediaPipe-Personensegmentierung für sichtbare Körperteile
try:
    import mediapipe as _mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _seg_options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(Path("~/anthro3d/selfie_segmenter.tflite").expanduser())
        ),
        output_category_mask=True
    )

    seg2 = mp_vision.ImageSegmenter.create_from_options(_seg_options)
    seg1 = mp_vision.ImageSegmenter.create_from_options(_seg_options)
    segov = mp_vision.ImageSegmenter.create_from_options(_seg_options)

    USE_PERSON_SEG = True
    print("MediaPipe Personenmaske aktiv")
except Exception as e:
    seg2 = seg1 = segov = None
    USE_PERSON_SEG = False
    print(f"MediaPipe Personenmaske nicht verfügbar: {e}")


def make_person_mask(frame_bgr, seg, target_shape, name="cam", min_pixels=800):
    """
    Erzeugt eine robuste 2D-Personenmaske.
    Falls MediaPipe Hintergrund und Person vertauscht liefert, wird automatisch invertiert.
    Danach wird nur die größte zusammenhängende Fläche behalten.
    """
    if frame_bgr is None or seg is None or not USE_PERSON_SEG:
        print(f"Personenmaske {name}: nicht verfügbar, volle Maske")
        return np.ones(target_shape, dtype=np.uint8) * 255

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        result = seg.segment(mp_img)

        if result.category_mask is None:
            print(f"Personenmaske {name}: keine Maske")
            return np.zeros(target_shape, dtype=np.uint8)

        raw = result.category_mask.numpy_view().squeeze()

        cand_a = (raw > 0).astype(np.uint8) * 255
        cand_b = (raw == 0).astype(np.uint8) * 255

        if cand_a.shape != target_shape:
            cand_a = cv2.resize(cand_a, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            cand_b = cv2.resize(cand_b, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        total = target_shape[0] * target_shape[1]

        def score_candidate(mask):
            n = int((mask > 0).sum())
            ratio = n / total if total else 0

            if n < min_pixels:
                return -1

            # Person sollte normalerweise nicht fast das ganze Bild sein.
            if ratio > 0.75:
                return -1

            # Idealbereich grob 5 bis 55 Prozent des Bildes.
            target_ratio = 0.22
            return 1.0 - abs(ratio - target_ratio)

        score_a = score_candidate(cand_a)
        score_b = score_candidate(cand_b)

        if score_b > score_a:
            mask = cand_b
            chosen = "invertiert"
        else:
            mask = cand_a
            chosen = "normal"

        n0 = int((mask > 0).sum())

        if n0 < min_pixels:
            print(f"Personenmaske {name}: keine sinnvolle Maske, Pixel={n0}")
            return np.zeros(target_shape, dtype=np.uint8)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Nur größte zusammenhängende Komponente behalten.
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)

        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            best_area = int(stats[best, cv2.CC_STAT_AREA])

            clean = np.zeros_like(mask)
            clean[labels == best] = 255
            mask = clean
        else:
            best_area = int((mask > 0).sum())

        n = int((mask > 0).sum())
        ratio = n / total * 100 if total else 0

        if n < min_pixels:
            print(f"Personenmaske {name}: größte Fläche zu klein, Pixel={n}")
            return np.zeros(target_shape, dtype=np.uint8)

        print(f"Personenmaske {name}: {n} Pixel ({ratio:.1f}%) | Modus={chosen} | vorher={n0}")
        return mask

    except Exception as e:
        print(f"Personenmaske {name}: Fehler {e}")
        return np.zeros(target_shape, dtype=np.uint8)



with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_ov9281.yaml").expanduser()) as f:
    cfgov = yaml.safe_load(f)

def make_maps_split(cfg, size=(1600,1200), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,flags=flags,alpha=alpha)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    return ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0]),R1

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2,R1_2 = make_maps_split(cfg2, (1600,1200))
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1,R1_1 = make_maps_split(cfg1, (1600,1200))
mlov1,mlov2,mrov1,mrov2,fxov,cxov,cyov,blov,R1_ov = make_maps_split(cfgov, (1280,800), flags=0, alpha=-1)
print(f"OV9281 Rectify aktiv: fx={fxov:.1f} cx={cxov:.1f} cy={cyov:.1f} baseline={blov*100:.1f}cm flags=0")

# ELP1→ELP2 laden
R_rel_elp1 = np.load(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser())
T_rel_elp1 = np.load(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser())
print(f"ELP1→ELP2 T: {T_rel_elp1*100} cm")

# OV9281→ELP2 Transformation berechnen via Marker ID 10
# ELP2 sieht ID 10, OV9281-L sieht ID 10
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
MARKER_SIZE = 0.19
obj_pts = np.array([[-MARKER_SIZE/2, MARKER_SIZE/2,0],
                    [ MARKER_SIZE/2, MARKER_SIZE/2,0],
                    [ MARKER_SIZE/2,-MARKER_SIZE/2,0],
                    [-MARKER_SIZE/2,-MARKER_SIZE/2,0]], dtype=np.float32)

K2_raw  = np.array(cfg2['camera_matrix_l']);  d2_raw  = np.array(cfg2['dist_l'])
Kov_raw = np.array(cfgov['camera_matrix_l']); dov_raw = np.array(cfgov['dist_l'])

cap2_cal  = cv2.VideoCapture(0)
cap2_cal.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2_cal.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
capovL_cal = cv2.VideoCapture(1)
capovL_cal.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovL_cal.set(cv2.CAP_PROP_FRAME_HEIGHT,800)


# OV9281 Position pruefen
# Version 1 der robusten ArUco Positionskontrolle fuer OV9281 zu ELP2.
# Wichtig:
# Diese Version prueft die Beziehung OV9281 zu ELP2.
# Sie ist noch nicht die vollstaendige Dreieckslogik mit ELP1.
# Fuer OV9281 zu ELP2 werden alle gemeinsam sichtbaren, bekannten Marker geprueft.
# Danach wird der beste und stabilste Markerpfad verwendet.
# Neue Positionswerte werden erst nach mehreren stabilen Bestaetigungen dauerhaft gespeichert.

from pathlib import Path as _Path
import time as _time

KNOWN_MARKER_IDS = {2, 3, 10}

EXPECTED_VISIBLE_MARKERS = {
    "ELP2": {2, 3},
    "OV9281": {3, 10},
}

R_saved_path_ov = _Path("~/anthro3d/R_rel_ov9281_to_elp2.npy").expanduser()
T_saved_path_ov = _Path("~/anthro3d/T_rel_ov9281_to_elp2.npy").expanduser()
pending_path_ov = _Path("~/anthro3d/pending_ov9281_position.npz").expanduser()

if not (R_saved_path_ov.exists() and T_saved_path_ov.exists()):
    raise SystemExit(
        "FEHLER: Gespeicherte OV9281 Kalibrierung fehlt. "
        "Bitte zuerst python3 ~/anthro3d/calibrate.py ausfuehren."
    )

R_saved_ov = np.load(R_saved_path_ov)
T_saved_ov = np.load(T_saved_path_ov).reshape(3)

print(
    f"OV9281 gespeicherte Position: "
    f"T={T_saved_ov*100} cm | "
    f"Distanz={np.linalg.norm(T_saved_ov)*100:.1f} cm"
)

def marker_reprojection_error(obj_pts, corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    real = corners.reshape(-1, 2)
    err = np.sqrt(((proj - real) ** 2).sum(axis=1))
    return float(np.mean(err))

def marker_geometry_ok(corner, tvec, min_area=50*50, max_dist=6.0):
    dist_m = float(np.linalg.norm(tvec))
    if dist_m > max_dist:
        return False

    pts = corner.reshape(-1, 2)
    area = float(cv2.contourArea(pts.astype(np.float32)))
    if area < min_area:
        return False

    w = float(np.linalg.norm(pts[0] - pts[1]))
    h = float(np.linalg.norm(pts[1] - pts[2]))
    if h <= 0:
        return False

    ratio = w / h
    if ratio < 0.65 or ratio > 1.35:
        return False

    return True

def rotation_diff_deg(R_a, R_b):
    R_delta = R_a @ R_b.T
    trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))

def stable_pending_update(candidate_R, candidate_T):
    required_count = 3

    max_pending_t_diff_cm = 3.0
    max_pending_rot_diff_deg = 3.0

    count = 1

    if pending_path_ov.exists():
        try:
            old = np.load(pending_path_ov)
            old_R = old["R"]
            old_T = old["T"].reshape(3)
            old_count = int(old["count"][0])

            tdiff = float(np.linalg.norm(candidate_T - old_T) * 100)
            rdiff = rotation_diff_deg(candidate_R, old_R)

            if tdiff <= max_pending_t_diff_cm and rdiff <= max_pending_rot_diff_deg:
                count = old_count + 1
            else:
                count = 1
        except Exception:
            count = 1

    np.savez(
        pending_path_ov,
        R=candidate_R,
        T=candidate_T,
        count=np.array([count]),
        timestamp=np.array([_time.time()])
    )

    return count, required_count

print("OV9281 ArUco Positionskontrolle laeuft...")
print("Erlaubte IDs:", sorted(KNOWN_MARKER_IDS))
print("ELP2 erwartet IDs:", sorted(EXPECTED_VISIBLE_MARKERS["ELP2"]))
print("OV9281 erwartet IDs:", sorted(EXPECTED_VISIBLE_MARKERS["OV9281"]))

R_live_list = []
T_live_list = []
score_list = []
used_marker_ids = []

for _ in range(60):
    ret2, f2 = cap2_cal.read()
    retov, fov = capovL_cal.read()

    if not ret2 or not retov:
        continue

    fl2 = f2[:, :f2.shape[1]//2]

    g2 = cv2.cvtColor(fl2, cv2.COLOR_BGR2GRAY)
    gov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
    gov = cv2.equalizeHist(gov)

    c2, ids2, _ = detector.detectMarkers(g2)
    cov, idsov, _ = detector.detectMarkers(gov)

    if ids2 is None or idsov is None:
        continue

    ids2f = ids2.flatten()
    idsovf = idsov.flatten()

    ids2_valid = [int(x) for x in ids2f if int(x) in KNOWN_MARKER_IDS]
    idsov_valid = [int(x) for x in idsovf if int(x) in KNOWN_MARKER_IDS]

    common = sorted(set(ids2_valid) & set(idsov_valid))

    expected_common = EXPECTED_VISIBLE_MARKERS["ELP2"] & EXPECTED_VISIBLE_MARKERS["OV9281"]
    common = [mid for mid in common if mid in expected_common]

    if not common:
        continue

    frame_candidates = []

    for mid in common:
        i2 = np.where(ids2f == mid)[0][0]
        iov = np.where(idsovf == mid)[0][0]

        ok2, rvec2, tvec2 = cv2.solvePnP(obj_pts, c2[i2], K2_raw, d2_raw)
        okv, rvecv, tvecv = cv2.solvePnP(obj_pts, cov[iov], Kov_raw, dov_raw)

        if not (ok2 and okv):
            continue

        if not marker_geometry_ok(c2[i2], tvec2):
            continue

        if not marker_geometry_ok(cov[iov], tvecv):
            continue

        err2 = marker_reprojection_error(obj_pts, c2[i2], rvec2, tvec2, K2_raw, d2_raw)
        errv = marker_reprojection_error(obj_pts, cov[iov], rvecv, tvecv, Kov_raw, dov_raw)

        if err2 > 4.0 or errv > 4.0:
            continue

        R2m, _ = cv2.Rodrigues(rvec2)
        Rvm, _ = cv2.Rodrigues(rvecv)

        R_rel = R2m @ Rvm.T
        T_rel = tvec2.flatten() - R_rel @ tvecv.flatten()

        score = err2 + errv

        frame_candidates.append((score, mid, R_rel, T_rel))

    if not frame_candidates:
        continue

    frame_candidates.sort(key=lambda x: x[0])
    score, mid, R_rel, T_rel = frame_candidates[0]

    R_live_list.append(R_rel)
    T_live_list.append(T_rel)
    score_list.append(score)
    used_marker_ids.append(mid)

cap2_cal.release()
capovL_cal.release()

if len(T_live_list) < 10:
    raise SystemExit(
        "FEHLER: Kameraposition konnte nicht geprueft werden. "
        "Kein gemeinsamer bekannter ArUco Marker zwischen ELP2 und OV9281 wurde stabil erkannt. "
        "Bitte Sicht auf Marker pruefen."
    )

T_arr = np.array(T_live_list)
R_arr = np.array(R_live_list)
score_arr = np.array(score_list)

dists = np.linalg.norm(T_arr, axis=1)
med = np.median(dists)
mask_dist = dists < med * 1.3

T_arr = T_arr[mask_dist]
R_arr = R_arr[mask_dist]
score_arr = score_arr[mask_dist]
used_marker_ids_arr = np.array(used_marker_ids)[mask_dist]

if len(T_arr) < 10:
    raise SystemExit(
        "FEHLER: OV9281 ArUco Messung ist nach Distanzfilter instabil. "
        "Bitte Marker Sicht, Licht und Stativ pruefen."
    )

if len(T_arr) >= 20:
    score_limit = np.percentile(score_arr, 70)
    mask_score = score_arr <= score_limit
    T_arr = T_arr[mask_score]
    R_arr = R_arr[mask_score]
    score_arr = score_arr[mask_score]
    used_marker_ids_arr = used_marker_ids_arr[mask_score]

if len(T_arr) < 10:
    raise SystemExit(
        "FEHLER: OV9281 ArUco Messung ist nach Reprojection Filter instabil. "
        "Bitte Marker Sicht, Licht und Stativ pruefen."
    )

T_live = np.median(T_arr, axis=0)

rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_arr])
rvec_live = np.median(rvecs, axis=0)
R_live, _ = cv2.Rodrigues(rvec_live)

t_std_cm = T_arr.std(axis=0) * 100
t_std_max_cm = float(np.max(t_std_cm))

t_diff_cm = float(np.linalg.norm(T_live - T_saved_ov) * 100)
rot_diff = rotation_diff_deg(R_live, R_saved_ov)

unique_ids, id_counts = np.unique(used_marker_ids_arr, return_counts=True)
id_info = ", ".join([f"ID {int(i)}:{int(c)}" for i, c in zip(unique_ids, id_counts)])

print(
    f"OV9281 Live Position: "
    f"T={T_live*100} cm | "
    f"Distanz={np.linalg.norm(T_live)*100:.1f} cm | "
    f"Samples={len(T_arr)} | Marker={id_info}"
)

print(
    f"OV9281 ArUco Qualitaet: "
    f"Reprojection median={np.median(score_arr):.2f}px | "
    f"Reprojection max={np.max(score_arr):.2f}px"
)

print(
    f"OV9281 Abweichung zur gespeicherten Position: "
    f"{t_diff_cm:.1f} cm | "
    f"Rotation={rot_diff:.2f} Grad | "
    f"Streuung max={t_std_max_cm:.1f} cm"
)

TOL_TRANSLATION_CM = 3.0
TOL_ROTATION_DEG = 3.0
MAX_STABLE_STD_CM = 6.0

if t_std_max_cm > MAX_STABLE_STD_CM:
    raise SystemExit(
        "FEHLER: OV9281 ArUco Messung ist zu instabil. "
        "Bitte Marker Sicht, Licht und Stativ pruefen."
    )

if t_diff_cm <= TOL_TRANSLATION_CM and rot_diff <= TOL_ROTATION_DEG:
    R_rel_ov = R_saved_ov
    T_rel_ov = T_saved_ov
    OV_OK = True

    if pending_path_ov.exists():
        try:
            pending_path_ov.unlink()
        except Exception:
            pass

    print("OV9281 Position OK. Gespeicherte Kalibrierung wird verwendet.")
else:
    R_rel_ov = R_live
    T_rel_ov = T_live
    OV_OK = True

    count, required_count = stable_pending_update(R_live, T_live)

    print(
        f"OV9281 Positionsaenderung erkannt. "
        f"Neue Live Position wird fuer diesen Scan verwendet. "
        f"Stabilitaetsbestaetigung {count} von {required_count}."
    )

    if count >= required_count:
        np.save(R_saved_path_ov, R_live)
        np.save(T_saved_path_ov, T_live)

        try:
            pending_path_ov.unlink()
        except Exception:
            pass

        print("OV9281 neue Position wurde stabil bestaetigt und dauerhaft gespeichert.")
    else:
        print("OV9281 neue Position wurde noch nicht dauerhaft gespeichert.")

bg2=bg1=bgov=st2=st1=stov=None
all_pts=all_cols=None


# Hintergrund automatisch aus Kalibrierung laden
from pathlib import Path as _Path
_cal = _Path("~/anthro3d/calibration_bg.npz").expanduser()
if _cal.exists():
    _data = np.load(_cal)
    bg2=_data['bg2']; bg1=_data['bg1']
    st2=_data['st2']; st1=_data['st1']
    if _data['OV_OK'][0]:
        bgov=_data['bgov']; stov=_data['stov']
    print("✓ Hintergrund aus Kalibrierung geladen — direkt SPACE drücken")
else:
    print("⚠ Keine Kalibrierung gefunden — B drücken für Hintergrund")

print("Fenster anklicken → B=Hintergrund | SPACE=Scan | Q=Beenden")

while True:
    d2,fl2   = get_disp_split(cap2,  ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
    d1,fl1   = get_disp_split(cap1,  ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
    if OV_OK:
        dov,flov = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
    else:
        dov,flov = None,None
    if d2 is None or d1 is None: continue

    m2  = np.zeros(d2.shape,  np.uint8)
    m1  = np.zeros(d1.shape,  np.uint8)
    mov = np.zeros(dov.shape, np.uint8) if dov is not None else None
    if st2 is not None:
        m2 = get_mask(d2, bg2, st2, 800, 600)
        m1 = get_mask(d1, bg1, st1, 800, 600)
        if dov is not None and stov is not None:
            mov = get_mask(dov, bgov, stov, 640, 400)

    dv2 = (np.clip(d2,0,128)/128*255).astype(np.uint8)
    dv1 = (np.clip(d1,0,128)/128*255).astype(np.uint8)
    ov_count = int(mov.sum()//255) if mov is not None else 0
    status = "B=Hintergrund" if st2 is None else f"ELP2:{m2.sum()//255} ELP1:{m1.sum()//255} OV:{ov_count} — SPACE=Scan"
    row1=np.hstack([cv2.resize(fl2,(480,300)),cv2.resize(cv2.applyColorMap(dv2,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m2,cv2.COLOR_GRAY2BGR),(480,300))])
    row2=np.hstack([cv2.resize(fl1,(480,300)),cv2.resize(cv2.applyColorMap(dv1,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m1,cv2.COLOR_GRAY2BGR),(480,300))])
    out = np.vstack([row1,row2])
    cv2.putText(out,status,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("ELP2 | ELP1",out)

    key = cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('b'):
        print("Hintergrund aufnehmen...")
        f2s=[]; f1s=[]; fovs=[]
        for i in range(20):
            d,_ = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
            if d is not None: f2s.append(d)
            d,_ = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
            if d is not None: f1s.append(d)
            if OV_OK:
                d,_ = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
                if d is not None: fovs.append(d)
            if i%5==0: print(f"  {i+1}/20")
        bg2 = np.median(f2s,axis=0).astype(np.float32)
        bg1 = np.median(f1s,axis=0).astype(np.float32)
        st2 = np.std(f2s,axis=0)<3.0
        st1 = np.std(f1s,axis=0)<3.0
        if fovs:
            bgov = np.median(fovs,axis=0).astype(np.float32)
            stov = np.std(fovs,axis=0)<3.0
            ov_pct = f" OV:{stov.mean()*100:.0f}%"
        else:
            ov_pct = " OV:n/a"
        print(f"✓ ELP2:{st2.mean()*100:.0f}% ELP1:{st1.mean()*100:.0f}%{ov_pct} — Person hinstellen → SPACE")

    if key==ord(' ') and st2 is not None:
        # 5 Sekunden Countdown
        import time as _time
        for _i in range(3, 0, -1):
            d2,fl2 = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
            d1,fl1 = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
            if d2 is None or d1 is None: continue
            dv2 = (np.clip(d2,0,128)/128*255).astype(np.uint8)
            row1 = np.hstack([cv2.resize(fl2,(480,300)), cv2.resize(cv2.applyColorMap(dv2,cv2.COLORMAP_JET),(480,300)), np.zeros((300,480,3),np.uint8)])
            out = np.vstack([row1, np.zeros((300,1440,3),np.uint8)])
            cv2.putText(out, f"Scan in {_i}...", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.imshow("ELP2 | ELP1", out)
            cv2.waitKey(1)
            _time.sleep(1)
        # Frische Frames für den Scan holen
        d2,fl2 = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
        d1,fl1 = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
        if OV_OK:
            dov,flov = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
        # MINIMALMODUS:
        # Keine MediaPipe-Personenmaske.
        # Keine Tiefenverfeinerung gegen den Hintergrund.
        # Alle gültigen Disparity-Punkte werden zunächst übernommen.
        m2 = np.ones(d2.shape, dtype=np.uint8) * 255
        m1 = np.ones(d1.shape, dtype=np.uint8) * 255

        if OV_OK and dov is not None:
            mov = np.ones(dov.shape, dtype=np.uint8) * 255

        p2,c2   = disp_to_pts(d2,  fl2,  m2,  fx2,  cx2,  cy2,  bl2)
        p1,c1   = disp_to_pts(d1,  fl1,  m1,  fx1,  cx1,  cy1,  bl1)
        if p1 is not None: print(f"ELP1 roh: X={p1[:,0].min()*100:.0f}-{p1[:,0].max()*100:.0f} Y={p1[:,1].min()*100:.0f}-{p1[:,1].max()*100:.0f} Z={p1[:,2].min()*100:.0f}-{p1[:,2].max()*100:.0f}")
        # Echte Kamerafarben verwenden
        # Debug-Farben deaktiviert
        parts=[]; cparts=[]
        if p2 is not None:
            zmask2 = (p2[:,2] > 0.5) & (p2[:,2] < 4.0)
            xmask2 = (p2[:,0] > -1.5) & (p2[:,0] < 1.5)
            p2 = p2[zmask2 & xmask2]; c2 = c2[zmask2 & xmask2]
            parts.append(p2); cparts.append(c2)
            print(f"ELP2: {len(p2)}")
        if p1 is not None:
            p1_fix = p1.copy()
            p1_fix[:,0] *= -1
            p1_t = (R_rel_elp1 @ p1_fix.T).T + T_rel_elp1.flatten()
            # Z-Clipping: nur Punkte 0.5–4m vor ELP2 Ursprung
            zmask1 = (p1_t[:,2] > 0.5) & (p1_t[:,2] < 4.0)
            xmask1 = (p1_t[:,0] > -1.5) & (p1_t[:,0] < 1.5)
            mask1 = zmask1 & xmask1
            print(f"ELP1 transformiert: X={p1_t[:,0].min()*100:.0f}-{p1_t[:,0].max()*100:.0f} Y={p1_t[:,1].min()*100:.0f}-{p1_t[:,1].max()*100:.0f} Z={p1_t[:,2].min()*100:.0f}-{p1_t[:,2].max()*100:.0f}")
            p1_t = p1_t[mask1]; c1 = c1[mask1]
            parts.append(p1_t); cparts.append(c1)
            print(f"ELP1: {len(p1)} → {len(p1_t)} nach Clipping")
        if OV_OK and mov is not None and stov is not None:
            pov,cov_col = disp_to_pts(dov, flov, mov, fxov, cxov, cyov, blov)
            if pov is not None:
                print(f"OV9281 Z-Bereich: {pov[:,2].min()*100:.0f}–{pov[:,2].max()*100:.0f}cm")
                print(f"OV9281 X-Bereich: {pov[:,0].min()*100:.0f}–{pov[:,0].max()*100:.0f}cm")
                # OV9281-Punkte kommen aus dem rektifizierten Stereo-Koordinatensystem.
                # Für die ArUco-Transformation müssen sie zurück in das Rohkamera-Koordinatensystem.
                pov_raw = (R1_ov.T @ pov.T).T
                print(
                    f"OV9281 nach Unrectify: "
                    f"X={pov_raw[:,0].min()*100:.0f} bis {pov_raw[:,0].max()*100:.0f}cm "
                    f"Y={pov_raw[:,1].min()*100:.0f} bis {pov_raw[:,1].max()*100:.0f}cm "
                    f"Z={pov_raw[:,2].min()*100:.0f} bis {pov_raw[:,2].max()*100:.0f}cm"
                )
                # Z-Clipping: nur Punkte 0.3–3m vor OV9281
                # OV9281 in ELP2-Weltkoordinaten transformieren
                # OV9281 feste Transformation nach erfolgreichem Test
                # Gefundene passende Variante: R.T*flip(-1,1,-1)+T
                Tov = T_rel_ov.reshape(1, 3)
                flip_ov = np.array([-1.0, 1.0, -1.0], dtype=float).reshape(1, 3)
                pf = pov_raw * flip_ov
                pov_t = (R_rel_ov.T @ pf.T).T + Tov

                ov_mask = (
                    (pov_t[:,0] > -1.60) & (pov_t[:,0] < 1.60) &
                    (pov_t[:,1] > -2.60) & (pov_t[:,1] < 0.90) &
                    (pov_t[:,2] > 0.35) & (pov_t[:,2] < 4.00)
                )

                print(
                    f"OV9281 feste Transformation: "
                    f"X={pov_t[:,0].min()*100:.0f} bis {pov_t[:,0].max()*100:.0f}cm "
                    f"Y={pov_t[:,1].min()*100:.0f} bis {pov_t[:,1].max()*100:.0f}cm "
                    f"Z={pov_t[:,2].min()*100:.0f} bis {pov_t[:,2].max()*100:.0f}cm | "
                    f"inBox={int(ov_mask.sum())}"
                )

                before_ov = len(pov_t)
                pov_t = pov_t[ov_mask]
                cov_col = cov_col[ov_mask]

                if len(pov_t) > 500:
                    parts.append(pov_t)
                    cparts.append(cov_col)
                    print(
                        f"OV9281: {before_ov} zu {len(pov_t)} nach Clipping hinzugefügt | "
                        f"X={pov_t[:,0].min()*100:.0f} bis {pov_t[:,0].max()*100:.0f}cm "
                        f"Z={pov_t[:,2].min()*100:.0f} bis {pov_t[:,2].max()*100:.0f}cm"
                    )
                else:
                    print(f"OV9281: {before_ov} zu {len(pov_t)} nach Clipping, zu wenig Punkte, nicht hinzugefügt")
        if parts:
            all_pts = np.vstack(parts); all_cols = np.vstack(cparts)
            print(f"Gesamt: {len(all_pts)}")
            print(f"Breite:{(all_pts[:,0].max()-all_pts[:,0].min())*100:.0f}cm Höhe:{(all_pts[:,1].max()-all_pts[:,1].min())*100:.0f}cm")
            # Pro-Kamera Diagnose
            if p2 is not None and len(p2)>0: print(f"  ELP2  X:{p2[:,0].min()*100:.0f}–{p2[:,0].max()*100:.0f}cm  Z:{p2[:,2].min()*100:.0f}–{p2[:,2].max()*100:.0f}cm")
            if p1 is not None and len(p1_t)>0: print(f"  ELP1  X:{p1_t[:,0].min()*100:.0f}–{p1_t[:,0].max()*100:.0f}cm  Z:{p1_t[:,2].min()*100:.0f}–{p1_t[:,2].max()*100:.0f}cm")
            break

cap2.release(); cap1.release(); capovL.release(); capovR.release()
cv2.destroyAllWindows()

if all_pts is not None:
    pts=all_pts.copy(); cols=all_cols.copy()


    # PCA vorerst deaktiviert, weil sie vor Korridor und Hintergrundentfernung die Weltkoordinaten zerstört
    print("PCA: deaktiviert")

    # Kombi-Filter: schwacher Hintergrundfilter plus Human-Shape-Filter
    print("MINIMAL-Filter: nur grobe Scanbox, kein Hintergrund, keine Personenmaske")

    before_box = len(pts)

    scanbox = (
        (pts[:,0] > -2.20) & (pts[:,0] < 2.20) &
        (pts[:,1] > -3.00) & (pts[:,1] < 1.20) &
        (pts[:,2] > 0.25) & (pts[:,2] < 4.50)
    )

    pts = pts[scanbox]
    cols = cols[scanbox]

    print(f"Scanbox minimal: {before_box} zu {len(pts)} Punkte")
    print(f"Minimal-Filter Ergebnis: {len(pts)} Punkte")

    # Voxel-Fusion: nahe Punkte aus mehreren Kameras zu einer gemeinsamen Wolke mitteln
    if len(pts) > 1000:
        voxel_fuse = 0.012  # 1.2 cm. Feiner für Körperdetails und trotzdem Fusion.

        v = np.floor(pts / voxel_fuse).astype(np.int32)
        uniq, inv = np.unique(v, axis=0, return_inverse=True)

        sum_pts = np.zeros((len(uniq), 3), dtype=np.float64)
        sum_cols = np.zeros((len(uniq), 3), dtype=np.float64)
        counts = np.zeros(len(uniq), dtype=np.float64)

        np.add.at(sum_pts, inv, pts)
        np.add.at(sum_cols, inv, cols)
        np.add.at(counts, inv, 1)

        pts_fused = sum_pts / counts[:, None]
        cols_fused = sum_cols / counts[:, None]

        before_fuse = len(pts)
        pts = pts_fused
        cols = cols_fused

        print(
            f"Voxel-Fusion: {before_fuse} → {len(pts)} Punkte | "
            f"Voxel={voxel_fuse*100:.1f}cm | "
            f"mittlere Punkte pro Voxel={before_fuse/len(pts):.2f}"
        )

    # Statistisches Outlier-Removal (ohne sklearn, nur numpy)
    print("Outlier-Removal: im Minimalmodus übersprungen")
    print(f"DEBUG vor Viewer: {len(pts)} Punkte, pts shape={pts.shape}")
    pts[:,0]-=pts[:,0].mean(); pts[:,1]-=pts[:,1].mean(); pts[:,2]-=pts[:,2].mean()
    pts[:,1]=-pts[:,1]
    step=max(1,len(pts)//10000); pts=pts[::step]; cols=cols[::step]

    from PyQt6.QtWidgets import QSlider, QHBoxLayout, QVBoxLayout, QWidget as QW
    from PyQt6.QtCore import Qt as Qt2

    class Viewer(QW):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ELP1+ELP2+OV9281 — Maus=Drehen | Slider=Zoom")
            self.resize(1000,1000)
            self.rx=15; self.ry=0; self.last=None
            global pts, cols
            if len(pts) == 0: pts = np.zeros((1,3)); cols = np.zeros((1,3))
            self.scale=300/max(pts[:,1].max()-pts[:,1].min(),0.1)

            # Layout: Canvas + vertikaler Slider
            hlay = QHBoxLayout(self); hlay.setContentsMargins(0,0,0,0); hlay.setSpacing(0)

            self.canvas = QW(self)
            self.canvas.setMinimumSize(900,900)
            self.canvas.paintEvent = self._paint
            self.canvas.mousePressEvent = self._mpress
            self.canvas.mouseMoveEvent = self._mmove
            self.canvas.mouseReleaseEvent = self._mrelease
            hlay.addWidget(self.canvas, 1)

            # Vertikaler Zoom-Schieberegler
            self.slider = QSlider(Qt2.Orientation.Vertical, self)
            self.slider.setMinimum(50); self.slider.setMaximum(2000)
            self.slider.setValue(int(self.scale))
            self.slider.setFixedWidth(40)
            self.slider.setToolTip("Zoom")
            self.slider.valueChanged.connect(self._on_zoom)
            hlay.addWidget(self.slider)

        def _on_zoom(self, val):
            self.scale = val; self.canvas.update()

        def _paint(self,e):
            p=QPainter(self.canvas); p.fillRect(self.canvas.rect(),QColor(20,20,20))
            cx_=self.canvas.width()//2; cy_=self.canvas.height()//2
            rx=math.radians(self.rx); ry=math.radians(self.ry)
            cX,sX=math.cos(rx),math.sin(rx); cY,sY=math.cos(ry),math.sin(ry)
            proj=[]
            for pt,c in zip(pts,cols):
                x,y,z=pt
                x2=x*cY+z*sY; z2=-x*sY+z*cY
                y2=y*cX-z2*sX; z3=y*sX+z2*cX
                proj.append((z3,int(cx_+x2*self.scale),int(cy_-y2*self.scale),c))
            proj.sort(key=lambda v:v[0])
            for _,sx,sy,c in proj:
                p.setPen(QColor(int(c[0]),int(c[1]),int(c[2]))); p.drawPoint(sx,sy)
            p.setPen(QColor(100,100,100))
            p.drawText(10,20,f"Punkte: {len(pts)} | Zoom: {int(self.scale)}")

        def _mpress(self,e): self.last=e.position()
        def _mmove(self,e):
            if self.last:
                dx=e.position().x()-self.last.x(); dy=e.position().y()-self.last.y()
                self.ry+=dx*0.5; self.rx+=dy*0.5; self.last=e.position()
                self.canvas.update()
        def _mrelease(self,e): self.last=None

    app=QApplication(sys.argv); w=Viewer(); w.show(); app.exec()
