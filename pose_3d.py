#!/usr/bin/env python3
"""
ANTHRO3D — 3D Pose Detektor
Kombiniert MediaPipe Pose mit Tiefenkarte → echte 3D Koordinaten in cm.
"""
import cv2, numpy as np, yaml, os, math
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from elp_stereo import ELPStereo
from stereo_depth_color import compute_depth

BASE       = os.path.expanduser("~/anthro3d")
MODEL_PATH = os.path.join(BASE, "pose_landmarker.task")

# Kamera-Intrinsics laden
K_l = np.array([[800,0,800],[0,800,600],[0,0,1]], dtype=np.float64)
try:
    with open(f"{BASE}/stereo_config.yaml") as f:
        cfg = yaml.safe_load(f)
    K_l = np.array(cfg['K_left'])
except: pass

# MediaPipe
opts = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5)
landmarker = PoseLandmarker.create_from_options(opts)

# Landmark-Namen
LM_NAMES = {
    0:'Nase', 11:'Schulter L', 12:'Schulter R',
    13:'Ellbogen L', 14:'Ellbogen R',
    15:'Handgelenk L', 16:'Handgelenk R',
    23:'Hüfte L', 24:'Hüfte R',
    25:'Knie L', 26:'Knie R',
    27:'Knöchel L', 28:'Knöchel R'
}

def pixel_to_3d(px, py, depth_cm, K):
    """Konvertiert Pixel + Tiefe in 3D Koordinaten (cm)."""
    if depth_cm <= 0: return None
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    x = (px - cx) * depth_cm / fx
    y = (py - cy) * depth_cm / fy
    z = depth_cm
    return (round(x,1), round(y,1), round(z,1))

def dist3d(a, b):
    """3D Abstand in cm."""
    if a is None or b is None: return None
    return round(math.sqrt(sum((a[i]-b[i])**2 for i in range(3))), 1)

def get_landmarks_3d(frame, depth_cm):
    """Gibt 3D Koordinaten aller Landmarks zurück."""
    H, W = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks: return {}
    pts = {}
    for i, lm in enumerate(result.pose_landmarks[0]):
        px = int(lm.x * W)
        py = int(lm.y * H)
        px = max(0, min(W-1, px))
        py = max(0, min(H-1, py))
        d  = float(depth_cm[py, px])
        pt3d = pixel_to_3d(px, py, d, K_l)
        pts[i] = {'px': px, 'py': py, 'depth': d, '3d': pt3d,
                  'name': LM_NAMES.get(i, f'LM{i}')}
    return pts

def compute_measurements(pts):
    """Berechnet klinische Maße aus 3D Punkten."""
    meas = {}
    # Schulterbreite
    if 11 in pts and 12 in pts:
        meas['Schulterbreite'] = dist3d(pts[11]['3d'], pts[12]['3d'])
    # Hüftbreite
    if 23 in pts and 24 in pts:
        meas['Hüftbreite'] = dist3d(pts[23]['3d'], pts[24]['3d'])
    # Oberarm L
    if 11 in pts and 13 in pts:
        meas['Oberarm L'] = dist3d(pts[11]['3d'], pts[13]['3d'])
    # Unterarm L
    if 13 in pts and 15 in pts:
        meas['Unterarm L'] = dist3d(pts[13]['3d'], pts[15]['3d'])
    # Oberschenkel L
    if 23 in pts and 25 in pts:
        meas['Oberschenkel L'] = dist3d(pts[23]['3d'], pts[25]['3d'])
    # Unterschenkel L
    if 25 in pts and 27 in pts:
        meas['Unterschenkel L'] = dist3d(pts[25]['3d'], pts[27]['3d'])
    # Körpergröße (Nase → Knöchel Durchschnitt)
    if 0 in pts and 27 in pts and 28 in pts:
        ankle_y = (pts[27]['3d'][1] + pts[28]['3d'][1]) / 2 if pts[27]['3d'] and pts[28]['3d'] else None
        if ankle_y and pts[0]['3d']:
            meas['Körpergröße'] = round(abs(pts[0]['3d'][1] - ankle_y) * 1.08, 1)
    return meas

if __name__ == "__main__":
    stereo = ELPStereo()
    print("3D Pose — Q=Beenden")
    while True:
        ret, fl, fr = stereo.read()
        if not ret: continue
        _, depth_cm = compute_depth(fl, fr)
        pts  = get_landmarks_3d(fl, depth_cm)
        meas = compute_measurements(pts)
        # Zeichnen
        vis = fl.copy()
        for idx, pt in pts.items():
            cv2.circle(vis, (pt['px'], pt['py']), 5, (0,255,0), -1)
            if pt['3d']:
                cv2.putText(vis, f"{pt['depth']:.0f}cm",
                            (pt['px']+5, pt['py']-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        # Maße anzeigen
        y = 30
        for name, val in meas.items():
            if val:
                cv2.putText(vis, f"{name}: {val}cm", (10,y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                y += 25
        cv2.imshow("3D Pose", cv2.resize(vis,(1280,480)))
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cv2.destroyAllWindows()
    stereo.release()
