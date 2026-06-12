#!/usr/bin/env python3
"""
ANTHRO3D — Automatische Multi-Kamera Positions-Kalibrierung
============================================================
Kameras erkennen gegenseitig ihre ArUco Marker und berechnen
relative Positionen ohne manuelles Drehen.

Setup:
  Index 0: Kamera 3   → Board ID 10
  Index 1: Stereo R   → Board ID 2
  Index 2: Stereo L   → Board ID 2
  Index 3: Kamera 2   → Board ID 3
"""

import cv2, numpy as np, yaml, time, os, math

BASE          = os.path.expanduser("~/anthro3d")
MARKER_SIZE_M = 0.19  # 19cm = A4 Marker

# Kamera → eigene Marker ID
CAM_OWN_MARKER = {
    0: 10,   # Kamera 3
    1:  2,   # Stereo R
    2:  2,   # Stereo L
    3:  3,   # Kamera 2
}

CAM_NAMES = {
    0: 'Kamera 3',
    1: 'Stereo R',
    2: 'Stereo L',
    3: 'Kamera 2',
}

# Geschätzte Kameramatrix für OV9281 1280x720
def default_K():
    return np.array([[800,0,640],[0,800,360],[0,0,1]], dtype=np.float64)

dist = np.zeros((4,1))

# ArUco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params     = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.03
detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

def detect_markers(frame, own_id):
    """Erkennt fremde Marker in einem Frame."""
    if frame is None: return {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None: return {}

    results = {}
    s = MARKER_SIZE_M / 2
    obj = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)

    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid == own_id: continue  # eigenen überspringen
        if mid not in [2, 3, 10]: continue  # nur bekannte IDs

        ok, rvec, tvec = cv2.solvePnP(obj, c[0].astype(np.float32),
                                       default_K(), dist)
        if ok:
            results[mid] = {
                'rvec': rvec.flatten(),
                'tvec': tvec.flatten(),
                'corners': c[0],
            }
    return results

def collect_observations(caps, n_frames=80, auto_mode=False):
    """Sammelt Beobachtungen von allen Kameras."""
    print(f"\nSammle {n_frames} Frames...")
    observations = {idx: [] for idx in caps}

    for frame_i in range(n_frames):
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if not ret: continue
            own_id = CAM_OWN_MARKER[idx]
            detected = detect_markers(frame, own_id)
            if detected:
                observations[idx].append(detected)

        # Status
        counts = {CAM_NAMES[i]: len(observations[i]) for i in caps}
        parts = [f"{n}:{c}" for n,c in counts.items()]
        print(f"\r  Frame {frame_i+1}/{n_frames} | {' | '.join(parts)}    ",
              end="", flush=True)

    print()
    return observations

def compute_positions(observations):
    """Berechnet Kamera-Positionen aus Beobachtungen."""
    positions = {}

    # Stereo R (Index 1) = Ursprung [0,0,0]
    positions[1] = {
        'name': 'Stereo R',
        'R': np.eye(3).tolist(),
        't': [0.0, 0.0, 0.0],
        'distance_cm': 0.0,
    }

    # Für jede Kamera: median tvec aus allen Beobachtungen
    for idx, obs_list in observations.items():
        if not obs_list: continue
        name = CAM_NAMES[idx]

        # Alle tvecs sammeln
        all_tvecs = []
        for obs in obs_list:
            for mid, data in obs.items():
                all_tvecs.append(data['tvec'])

        if not all_tvecs: continue

        arr = np.array(all_tvecs)
        med = np.median(arr, axis=0)
        dist_cm = float(np.linalg.norm(med)) * 100

        # Rotation aus rvec
        all_rvecs = []
        for obs in obs_list:
            for mid, data in obs.items():
                all_rvecs.append(data['rvec'])

        med_rvec = np.median(np.array(all_rvecs), axis=0)
        R, _ = cv2.Rodrigues(med_rvec)

        positions[idx] = {
            'name': name,
            'R': R.tolist(),
            't': med.tolist(),
            'distance_cm': round(dist_cm, 1),
            'n_frames': len(obs_list),
        }
        print(f"  {name}: {dist_cm:.1f}cm  ({len(obs_list)} Frames)")

    return positions

def save_positions(positions):
    """Speichert in cam_positions.yaml."""
    config = {
        'calibration': {
            'timestamp': time.time(),
            'marker_size_m': MARKER_SIZE_M,
            'method': 'multi_cam_auto',
        },
        'cameras': {}
    }

    for idx, pos in positions.items():
        config['cameras'][pos['name']] = {
            'index':       idx,
            'position_m':  pos['t'],
            'distance_cm': pos['distance_cm'],
            'R':           pos['R'],
        }

    path = os.path.join(BASE, 'cam_positions.yaml')
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nGespeichert: {path}")

def main(auto_mode=False):
    print("ANTHRO3D — Multi-Kamera Kalibrierung")
    print("=" * 40)
    print("\nKameras verbinden...")

    caps = {}
    for idx, name in CAM_NAMES.items():
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            caps[idx] = cap
            print(f"  ✓ {name} (Index {idx})")
        else:
            print(f"  ✗ {name} (Index {idx}) — nicht verfügbar")

    if not caps:
        print("Keine Kameras gefunden!")
        return

    print(f"\n{len(caps)} Kameras verbunden")
    print("Alle Kameras müssen die Boards der anderen sehen.")
    print("Stelle sicher dass die Boards gut sichtbar sind.\n")
    if not auto_mode:
        input("Enter zum Starten...")

    # Beobachtungen sammeln
    obs = collect_observations(caps, n_frames=80)

    # Positionen berechnen
    print("\nBerechne Positionen...")
    positions = compute_positions(obs)

    if len(positions) < 2:
        print("Zu wenige Erkennungen — Boards besser ausrichten!")
    else:
        save_positions(positions)
        print(f"\n✓ {len(positions)} Kameras kalibriert")

    for cap in caps.values(): cap.release()

if __name__ == "__main__":
    import sys
    auto_mode = "--auto" in sys.argv
    main(auto_mode=auto_mode)
