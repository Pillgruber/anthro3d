#!/usr/bin/env python3
"""
ANTHRO3D — Punktwolke
Erstellt eine farbige 3D Punktwolke aus ELP Stereo → speichert als .ply
Öffenbar in MeshLab, Blender, CloudCompare.
"""
import cv2, numpy as np, yaml, os, struct
from elp_stereo import ELPStereo
from stereo_depth_color import compute_depth

BASE = os.path.expanduser("~/anthro3d")

# Kamera-Intrinsics
K_l = np.array([[800,0,800],[0,800,600],[0,0,1]], dtype=np.float64)
baseline_mm = 65.0
try:
    with open(f"{BASE}/stereo_config.yaml") as f:
        cfg = yaml.safe_load(f)
    K_l = np.array(cfg['K_left'])
    baseline_mm = cfg.get('baseline_mm', 65)
except: pass

def build_pointcloud(color_frame, depth_cm, max_dist=300):
    """Erstellt Punktwolke aus Farbbild + Tiefenkarte."""
    H, W = depth_cm.shape
    fx, fy = K_l[0,0], K_l[1,1]
    cx, cy = K_l[0,2], K_l[1,2]

    points = []
    colors = []

    step = 2  # Jeden 2. Pixel — für Performance
    for y in range(0, H, step):
        for x in range(0, W, step):
            d = depth_cm[y, x]
            if d < 5 or d > max_dist: continue
            # 3D Koordinaten
            X = (x - cx) * d / fx
            Y = (y - cy) * d / fy
            Z = d
            points.append([X, Y, Z])
            # Farbe (BGR → RGB)
            b, g, r = color_frame[y, x]
            colors.append([r, g, b])

    return np.array(points), np.array(colors)

def save_ply(points, colors, path):
    """Speichert Punktwolke als PLY-Datei."""
    n = len(points)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Gespeichert: {path} ({n} Punkte)")

if __name__ == "__main__":
    stereo = ELPStereo()
    print("Punktwolke — SPACE=Aufnahme | Q=Beenden")
    snap_count = 0

    while True:
        ret, fl, fr = stereo.read()
        if not ret: continue

        _, depth_cm = compute_depth(fl, fr)

        # Vorschau
        vis = fl.copy()
        h, w = vis.shape[:2]
        valid = int(np.sum(depth_cm > 5) * 100 // (h*w))
        cv2.putText(vis, f"Abdeckung: {valid}% | SPACE=Snapshot",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Punktwolke Vorschau", cv2.resize(vis,(1280,480)))

        key = cv2.waitKey(1)&0xFF
        if key == ord('q'): break
        elif key == ord(' '):
            # Punktwolke aufnehmen
            points, colors = build_pointcloud(fl, depth_cm)
            path = os.path.join(BASE, f"pointcloud_{snap_count:03d}.ply")
            save_ply(points, colors, path)
            snap_count += 1
            print(f"  → In MeshLab öffnen: {path}")

    cv2.destroyAllWindows()
    stereo.release()
