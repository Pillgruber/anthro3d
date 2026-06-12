#!/usr/bin/env python3
"""
ANTHRO3D — Live Stereo Tiefenkarte mit automatischer Rektifizierung
Berechnet Tiefe aus zwei Kamerabildern ohne perfekte Kalibrierung.
"""
import cv2, numpy as np, yaml, os

BASE = os.path.expanduser("~/anthro3d")
CAM_L, CAM_R = 1, 2

# Stereo-Konfiguration laden falls vorhanden
K_l = K_r = d_l = d_r = R = T = None
maps_ok = False
try:
    with open(f"{BASE}/stereo_config.yaml") as f:
        cfg = yaml.safe_load(f)
    K_l = np.array(cfg['K_left'])
    d_l = np.array(cfg['dist_left'])
    K_r = np.array(cfg['K_right'])
    d_r = np.array(cfg['dist_right'])
    R   = np.array(cfg['R'])
    T   = np.array(cfg['T'])
    baseline_mm = cfg.get('baseline_mm', 80)

    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K_l,d_l,K_r,d_r,(1280,720),R,T,
                                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5)
    map_l1,map_l2 = cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,(1280,720),cv2.CV_32F)
    map_r1,map_r2 = cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,(1280,720),cv2.CV_32F)
    maps_ok = True
    print(f"Kalibrierung geladen ✓ Baseline={baseline_mm}mm")
except Exception as e:
    print(f"Keine Kalibrierung — verwende rohe Bilder: {e}")
    baseline_mm = 80

# SGBM — Semi-Global Block Matching (viel besser als BM)
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8*3*7**2,
    P2=32*3*7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# WLS Filter für glattere Tiefenkarte
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
sgbm_r = cv2.ximgproc.createRightMatcher(sgbm)
wls.setLambda(8000)
wls.setSigmaColor(1.5)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

cap_l = cv2.VideoCapture(CAM_L)
cap_r = cv2.VideoCapture(CAM_R)

print("Tiefenkarte — Q=Beenden | R=Rektifizierung an/aus | W=WLS an/aus")
use_rect = maps_ok
use_wls  = True

while True:
    rl,fl=cap_l.read(); rr,fr=cap_r.read()
    if not rl or not rr: continue

    # Rektifizierung anwenden
    if use_rect and maps_ok:
        fl = cv2.remap(fl, map_l1, map_l2, cv2.INTER_LINEAR)
        fr = cv2.remap(fr, map_r1, map_r2, cv2.INTER_LINEAR)

    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    # CLAHE Normalisierung
    gl = clahe.apply(gl); gr = clahe.apply(gr)

    # Disparität berechnen
    disp_l = sgbm.compute(gl, gr)

    if use_wls:
        disp_r = sgbm_r.compute(gr, gl)
        disp_filt = wls.filter(disp_l, gl, disparity_map_right=disp_r)
        disp_show = disp_filt.astype(np.float32) / 16.0
    else:
        disp_show = disp_l.astype(np.float32) / 16.0

    disp_show[disp_show < 1] = 0

    # In cm umrechnen wenn Kalibrierung vorhanden
    depth_cm = np.zeros_like(disp_show)
    if maps_ok and K_l is not None:
        focal = K_l[0,0]
        valid = disp_show > 1
        depth_cm[valid] = (focal * baseline_mm/10) / disp_show[valid]
        depth_cm = np.clip(depth_cm, 0, 500)

    # Visualisierung
    disp_norm = cv2.normalize(disp_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    # Epipolarlinien einzeichnen (zeigt wie gut die Ausrichtung ist)
    H = depth_color.shape[0]
    for y in range(0, H, 60):
        cv2.line(depth_color, (0,y), (depth_color.shape[1],y), (255,255,255), 1)

    # Info
    valid_px = int(np.sum(disp_show > 1))
    coverage = valid_px * 100 // (disp_show.shape[0]*disp_show.shape[1])
    mode_txt = f"{'RECT' if use_rect else 'RAW'} | {'WLS' if use_wls else 'SGBM'} | {coverage}% Abdeckung"
    cv2.putText(depth_color, mode_txt, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if maps_ok:
        # Tiefe in Bildmitte anzeigen
        mid_y, mid_x = H//2, depth_color.shape[1]//2
        d = depth_cm[mid_y, mid_x]
        if d > 0:
            cv2.putText(depth_color, f"Mitte: {d:.0f}cm",
                        (mid_x-60, mid_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.circle(depth_color, (mid_x, mid_y), 8, (0,255,255), 2)

    # Layout
    h_small = 240
    fl_s = cv2.resize(fl, (426, h_small))
    fr_s = cv2.resize(fr, (426, h_small))
    depth_s = cv2.resize(depth_color, (428, h_small))
    top = np.hstack([fl_s, fr_s, depth_s])

    depth_big = cv2.resize(depth_color, (1280, 480))
    full = np.vstack([top, depth_big])

    cv2.imshow("ANTHRO3D — Stereo Tiefe", full)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'): use_rect = not use_rect; print(f"Rektifizierung: {use_rect}")
    elif key == ord('w'): use_wls = not use_wls; print(f"WLS: {use_wls}")

cv2.destroyAllWindows()
cap_l.release(); cap_r.release()
