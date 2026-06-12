#!/usr/bin/env python3
"""
ANTHRO3D — 360° Multi-Kamera
ELP Stereo (vorne) + OV9281 (Seiten/Hinten) → vollständiger Körperscan.
"""
import cv2, numpy as np, yaml, os, math
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

BASE       = os.path.expanduser("~/anthro3d")
MODEL_PATH = os.path.join(BASE, "pose_landmarker.task")

# Kamera-Rollen — werden aus config.yaml geladen
CAM_ROLES = {
    'ELP Stereo': None,
    'Seite L':    None,
    'Seite R':    None,
    'Hinten':     None,
}

def load_config():
    """Lädt Kamera-Zuordnung aus config.yaml."""
    try:
        with open(os.path.join(BASE, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        for cam in cfg['cameras']['tracking']:
            name = cam['name']
            if name in CAM_ROLES:
                CAM_ROLES[name] = cam['device_index']
        print("Kamera-Config geladen:")
        for name, idx in CAM_ROLES.items():
            print(f"  {name}: Index {idx}")
    except Exception as e:
        print(f"Config nicht gefunden: {e}")

def load_positions():
    """Lädt Kamera-Positionen aus cam_positions.yaml."""
    try:
        with open(os.path.join(BASE, "cam_positions.yaml")) as f:
            cfg = yaml.safe_load(f)
        return cfg.get('cameras', {})
    except:
        return {}

class CameraView:
    """Eine Kamera-Ansicht mit Pose-Erkennung."""
    def __init__(self, name, index, is_stereo=False):
        self.name      = name
        self.is_stereo = is_stereo
        self.cap       = cv2.VideoCapture(index)
        if is_stereo:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # Eigener Landmarker pro Kamera
        opts = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5)
        self.lm = PoseLandmarker.create_from_options(opts)
        print(f"  {name}: Index {index} ({'Stereo' if is_stereo else 'Mono'})")

    def read(self):
        ret, frame = self.cap.read()
        if not ret: return False, None
        if self.is_stereo:
            frame = frame[:, :frame.shape[1]//2]  # Nur Links
        return True, frame

    def get_landmarks(self, frame):
        H, W = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.lm.detect(img)
        if not res.pose_landmarks: return {}
        return {i: (int(lm.x*W), int(lm.y*H))
                for i, lm in enumerate(res.pose_landmarks[0])}

    def release(self):
        self.cap.release()

# Wichtige Landmarks
KEY_LM = {
    0:'Nase', 11:'Schulter L', 12:'Schulter R',
    23:'Hüfte L', 24:'Hüfte R',
    25:'Knie L', 26:'Knie R',
    27:'Knöchel L', 28:'Knöchel R'
}

if __name__ == "__main__":
    load_config()
    positions = load_positions()

    # Kameras öffnen
    views = []
    for name, idx in CAM_ROLES.items():
        if idx is not None:
            is_stereo = (name == 'ELP Stereo')
            views.append(CameraView(name, idx, is_stereo))

    if not views:
        print("Keine Kameras konfiguriert — bitte config.yaml prüfen")
        exit(1)

    print(f"\n{len(views)} Kameras aktiv — Q=Beenden")

    while True:
        frames = []
        all_pts = {}

        for view in views:
            ret, frame = view.read()
            if not ret: continue
            pts = view.get_landmarks(frame)
            all_pts[view.name] = pts

            # Landmarks einzeichnen
            vis = frame.copy()
            for idx, (px, py) in pts.items():
                cv2.circle(vis, (px, py), 5, (0,255,0), -1)
            label = f"{view.name} ({len(pts)} Pts)"
            cv2.putText(vis, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            frames.append(cv2.resize(vis, (640, 400)))

        if frames:
            # Alle Ansichten nebeneinander
            while len(frames) < 4:
                frames.append(np.zeros((400,640,3), dtype=np.uint8))
            top = np.hstack(frames[:2])
            bot = np.hstack(frames[2:4])
            grid = np.vstack([top, bot])
            cv2.putText(grid, f"ANTHRO3D 360°", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("ANTHRO3D 360°", grid)

        if cv2.waitKey(1)&0xFF==ord('q'): break

    cv2.destroyAllWindows()
    for v in views: v.release()
