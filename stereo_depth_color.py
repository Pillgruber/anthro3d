#!/usr/bin/env python3
"""
ANTHRO3D — Farbtiefenkarte
Berechnet Tiefenkarte aus ELP Stereo Farbbild + überlagert sie.
"""
import cv2, numpy as np, yaml, os
from elp_stereo import ELPStereo

BASE = os.path.expanduser("~/anthro3d")

# SGBM
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8*3*49, P2=32*3*49,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
wls    = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
sgbm_r = cv2.ximgproc.createRightMatcher(sgbm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

# Stereo-Konfiguration laden
K_l = baseline_mm = None
try:
    with open(f"{BASE}/stereo_config.yaml") as f:
        cfg = yaml.safe_load(f)
    K_l = np.array(cfg['K_left'])
    baseline_mm = cfg.get('baseline_mm', 65)
    print(f"Kalibrierung geladen — Baseline={baseline_mm}mm")
except:
    print("Keine Kalibrierung — Tiefe relativ")

def compute_depth(fl, fr):
    """Berechnet Tiefenkarte aus zwei Farbbildern."""
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    disp_l = sgbm.compute(gl, gr)
    disp_r = sgbm_r.compute(gr, gl)
    disp_f = wls.filter(disp_l, gl, disparity_map_right=disp_r)
    disp   = np.clip(disp_f.astype(np.float32)/16.0, 0, None)
    # In cm umrechnen
    depth_cm = np.zeros_like(disp)
    if K_l is not None and baseline_mm:
        valid = disp > 1
        depth_cm[valid] = (K_l[0,0] * baseline_mm/10) / disp[valid]
        depth_cm = np.clip(depth_cm, 0, 500)
    return disp, depth_cm

def overlay_depth(color_frame, depth_cm, alpha=0.4):
    """Überlagert Tiefenkarte auf Farbbild."""
    disp_norm = cv2.normalize(depth_cm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)
    depth_color = cv2.resize(depth_color, (color_frame.shape[1], color_frame.shape[0]))
    mask = (disp_norm > 5).astype(np.uint8)
    mask = cv2.resize(mask, (color_frame.shape[1], color_frame.shape[0]))
    result = color_frame.copy()
    result[mask>0] = cv2.addWeighted(
        color_frame, 1-alpha, depth_color, alpha, 0)[mask>0]
    return result

if __name__ == "__main__":
    stereo = ELPStereo()
    print("Farbtiefenkarte — Q=Beenden | D=Tiefe ein/aus")
    show_depth = True
    while True:
        ret, fl, fr = stereo.read()
        if not ret: continue
        _, depth_cm = compute_depth(fl, fr)
        if show_depth:
            result = overlay_depth(fl, depth_cm)
        else:
            result = fl.copy()
        # Tiefe in Bildmitte
        h, w = result.shape[:2]
        d = depth_cm[h//2, w//2]
        if d > 0:
            cv2.putText(result, f"{d:.0f}cm", (w//2-30, h//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.circle(result, (w//2, h//2), 8, (0,255,255), 2)
        cv2.imshow("Farb-Tiefe", cv2.resize(result,(1280,480)))
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('d'): show_depth=not show_depth
    cv2.destroyAllWindows()
    stereo.release()
