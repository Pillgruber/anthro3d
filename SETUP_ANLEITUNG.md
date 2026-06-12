# ANTHRO3D Setup Anleitung

## Dreieck-Aufstellung
        ELP1 (vorne, ID2)
           /          \\
    OV9281 (links,ID10)  ELP2 (rechts,ID3)

## Reihenfolge wenn Kameras ankommen
1. python3 ~/anthro3d/detect_cameras.py
2. python3 ~/anthro3d/elp_capture.py
3. python3 ~/anthro3d/aruco_stereo_calib.py
4. python3 ~/anthro3d/triangle_calib.py
5. python3 ~/anthro3d/triangulation.py
6. python3 ~/anthro3d/anthro3d_app.py

## ArUco IDs
ID  2 = ELP1 (vorne)
ID  3 = ELP2 (rechts)
ID 10 = OV9281 (links)
ID 16 = Patient (Boden)

## USB Tipp
ELP an USB 3.0 direkt (kein Hub)
OV9281 koennen Hub verwenden
