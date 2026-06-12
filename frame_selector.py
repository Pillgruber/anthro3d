#!/usr/bin/env python3
"""
ANTHRO3D — Ring-Buffer + Frame Selector + 3D Viewer
Nimmt kontinuierlich alle Kameras auf (Ring-Buffer, letzte 10 Sek.)
Arzt friert einen Frame ein, wählt den besten aus, berechnet 3D Mesh.

Integration in anthro3d_app.py:
- _toggle_rec() startet/stoppt RingBuffer
- Frame-Selector öffnet sich nach Stopp
- 3D Viewer öffnet sich nach Frame-Auswahl
"""
import cv2, numpy as np, yaml, os, time, threading
from collections import deque
from pathlib import Path

BASE     = Path("~/anthro3d").expanduser()
SAVE_DIR = Path("~/anthro3d/recordings").expanduser()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Ring-Buffer ───────────────────────────────────────────────────────────────
class RingBuffer:
    """Speichert die letzten N Frames pro Kamera."""
    def __init__(self, maxlen=300):  # 300 Frames ≈ 10 Sek @ 30fps
        self.maxlen  = maxlen
        self.buffers = {}   # {cam_name: deque([frame, ...])}
        self.timestamps = {}
        self.lock    = threading.Lock()

    def add(self, cam_name, frame, timestamp=None):
        with self.lock:
            if cam_name not in self.buffers:
                self.buffers[cam_name]    = deque(maxlen=self.maxlen)
                self.timestamps[cam_name] = deque(maxlen=self.maxlen)
            self.buffers[cam_name].append(frame.copy())
            self.timestamps[cam_name].append(timestamp or time.time())

    def get_frame(self, cam_name, idx):
        """Gibt Frame an Position idx zurück."""
        with self.lock:
            buf = self.buffers.get(cam_name)
            if buf is None or idx >= len(buf): return None
            return list(buf)[idx]

    def get_all_at(self, idx):
        """Gibt alle Kamera-Frames bei Index idx zurück."""
        with self.lock:
            result = {}
            for name, buf in self.buffers.items():
                buf_list = list(buf)
                if idx < len(buf_list):
                    result[name] = buf_list[idx]
            return result

    def length(self, cam_name=None):
        with self.lock:
            if cam_name:
                return len(self.buffers.get(cam_name, []))
            if not self.buffers: return 0
            return min(len(b) for b in self.buffers.values())

    def clear(self):
        with self.lock:
            self.buffers.clear()
            self.timestamps.clear()


# ── Frame Qualitäts-Score ─────────────────────────────────────────────────────
def frame_quality(frame):
    """
    Bewertet Frame-Qualität:
    - Schärfe (Laplacian Varianz)
    - Helligkeit (nicht über/unterbelichtet)
    - Marker-Erkennbarkeit
    """
    if frame is None: return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Schärfe
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Helligkeit (optimal ~128)
    brightness = float(gray.mean())
    bright_score = 1.0 - abs(brightness - 128) / 128

    # Kontrast
    contrast = float(gray.std())

    score = sharpness * 0.5 + bright_score * 100 + contrast * 0.3
    return score


def find_best_frame(ring_buffer, cam_name='ELP Stereo'):
    """Findet den Frame mit der besten Qualität."""
    n = ring_buffer.length(cam_name)
    if n == 0: return 0

    best_idx   = 0
    best_score = 0
    step = max(1, n // 30)  # Nur jeden x-ten Frame prüfen

    for i in range(0, n, step):
        frame = ring_buffer.get_frame(cam_name, i)
        if frame is None: continue
        score = frame_quality(frame)
        if score > best_score:
            best_score = score
            best_idx   = i

    return best_idx


# ── Frame Selector (OpenCV Fenster) ──────────────────────────────────────────
class FrameSelector:
    """
    OpenCV Fenster mit Slider zum Auswählen des besten Frames.
    Zeigt alle Kameras nebeneinander.
    """
    def __init__(self, ring_buffer, on_select_callback):
        self.ring    = ring_buffer
        self.on_select = on_select_callback
        self.idx     = 0
        self.running = True
        self.win     = "Frame auswählen — ENTER=Übernehmen | ESC=Abbrechen"

    def run(self):
        n = self.ring.length()
        if n == 0:
            print("Kein Ring-Buffer gefüllt")
            return

        # Besten Frame vorauswählen
        self.idx = find_best_frame(self.ring)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 600)

        def on_slider(val):
            self.idx = val

        cv2.createTrackbar("Frame", self.win, self.idx, max(n-1,1), on_slider)

        print(f"\nFrame Selector — {n} Frames verfügbar")
        print("← → = Frame vor/zurück | ENTER = Übernehmen | ESC = Abbrechen\n")

        while self.running:
            frames = self.ring.get_all_at(self.idx)
            if not frames:
                time.sleep(0.05)
                continue

            # Alle Kameras als Grid anzeigen
            previews = []
            for name, frame in frames.items():
                vis = cv2.resize(frame, (426, 240))
                q   = frame_quality(frame)
                col = (0,200,80) if q > 500 else (0,150,255) if q > 200 else (80,80,200)
                cv2.putText(vis, name, (5,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.putText(vis, f"Q:{q:.0f}", (5,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                previews.append(vis)

            # Auffüllen auf 4
            while len(previews) < 4:
                previews.append(np.zeros((240,426,3), dtype=np.uint8))

            top = np.hstack(previews[:2])
            bot = np.hstack(previews[2:4])
            grid = np.vstack([top, bot])

            # Info-Leiste
            info = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
            pct  = int(self.idx * 100 / max(n-1, 1))
            cv2.putText(info,
                        f"Frame {self.idx}/{n-1} ({pct}%) | "
                        f"ENTER=3D berechnen | ESC=Abbrechen | "
                        f"← → = navigieren",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200,200,200), 1)

            cv2.imshow(self.win, np.vstack([grid, info]))
            cv2.setTrackbarPos("Frame", self.win, self.idx)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:   # ENTER
                print(f"Frame {self.idx} ausgewählt")
                cv2.destroyWindow(self.win)
                self.on_select(self.idx)
                return
            elif key == 27:  # ESC
                cv2.destroyWindow(self.win)
                return
            elif key == 81 or key == ord('a'):  # ←
                self.idx = max(0, self.idx - 1)
            elif key == 83 or key == ord('d'):  # →
                self.idx = min(n-1, self.idx + 1)


# ── 3D Viewer ────────────────────────────────────────────────────────────────
class Viewer3D:
    """
    Öffnet Open3D Viewer mit:
    - Freiem Drehen
    - Körpermaßen als 3D Linien
    - Ein/Ausblenden einzelner Maße
    """
    def __init__(self):
        try:
            import open3d as o3d
            self.o3d = o3d
            self.ok  = True
        except ImportError:
            print("Open3D nicht verfügbar — pip install open3d")
            self.ok = False

    def build_measurement_lines(self, landmarks_3d, measurements):
        """Erstellt 3D Linien für Körpermaße."""
        if not self.ok: return None
        o3d = self.o3d

        # Farbzuweisung pro Maß
        colors_map = {
            'Schulterbreite': [0.2, 0.8, 0.2],
            'Hüftbreite':     [0.8, 0.4, 0.2],
            'Oberarm L':      [0.2, 0.6, 0.8],
            'Unterarm L':     [0.2, 0.6, 0.8],
            'Oberschenkel L': [0.8, 0.2, 0.6],
            'Unterschenkel L':[0.8, 0.2, 0.6],
            'Körpergröße':    [0.9, 0.9, 0.2],
        }

        # Landmark-Paare pro Maß
        lm_pairs = {
            'Schulterbreite': (11, 12),
            'Hüftbreite':     (23, 24),
            'Oberarm L':      (11, 13),
            'Unterarm L':     (13, 15),
            'Oberschenkel L': (23, 25),
            'Unterschenkel L':(25, 27),
        }

        points = []
        lines  = []
        colors = []
        labels = []

        for name, (i, j) in lm_pairs.items():
            if i not in landmarks_3d or j not in landmarks_3d: continue
            p1 = landmarks_3d[i]
            p2 = landmarks_3d[j]
            if p1 is None or p2 is None: continue

            idx = len(points)
            points += [p1, p2]
            lines.append([idx, idx+1])
            colors.append(colors_map.get(name, [1,1,1]))

            # Maß-Label Mitte
            mid = [(p1[k]+p2[k])/2 for k in range(3)]
            val = measurements.get(name, '?')
            labels.append((mid, f"{name}: {val}cm"))

        if not points: return None, []

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines  = o3d.utility.Vector2iVector(np.array(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        return line_set, labels

    def show(self, mesh, landmarks_3d=None, measurements=None):
        """Öffnet interaktiven 3D Viewer."""
        if not self.ok: return
        o3d = self.o3d

        geometries = []

        # Koordinatensystem
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=20, origin=[0,0,0])
        geometries.append(coord)

        # Mesh
        if mesh is not None:
            geometries.append(mesh)

        # Maß-Linien
        if landmarks_3d and measurements:
            lines, labels = self.build_measurement_lines(
                landmarks_3d, measurements)
            if lines:
                geometries.append(lines)
                # Labels als kleine Kugeln
                for pos, label in labels:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
                    sphere.translate(pos)
                    sphere.paint_uniform_color([1, 1, 0])
                    geometries.append(sphere)
                    print(f"  📏 {label}")

        print("\n3D Viewer — Maus: Drehen | Scroll: Zoom | Q: Schließen")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="ANTHRO3D 3D Viewer",
            width=1280, height=720,
            mesh_show_back_face=True)


# ── Haupt-Controller ─────────────────────────────────────────────────────────
class RecordingController:
    """
    Verbindet Ring-Buffer + Frame-Selector + 3D Viewer.
    Wird in anthro3d_app.py integriert.

    Verwendung:
        ctrl = RecordingController()
        # In CameraThread:
        ctrl.ring.add('ELP Stereo', frame)
        # Aufnahme stoppen:
        ctrl.stop_and_select()
    """
    def __init__(self):
        self.ring      = RingBuffer(maxlen=300)
        self.viewer    = Viewer3D()
        self.recording = False
        self._selected_frames = None

    def start(self):
        self.ring.clear()
        self.recording = True
        print("● Aufnahme gestartet (Ring-Buffer)")

    def stop_and_select(self):
        """Stoppt Aufnahme und öffnet Frame-Selector."""
        self.recording = False
        print(f"■ Aufnahme gestoppt — {self.ring.length()} Frames")

        def on_select(idx):
            self._selected_frames = self.ring.get_all_at(idx)
            print(f"Frame {idx} ausgewählt — starte 3D Berechnung...")
            self._compute_and_show()

        selector = FrameSelector(self.ring, on_select)
        # In separatem Thread damit App nicht blockiert
        t = threading.Thread(target=selector.run, daemon=True)
        t.start()

    def _compute_and_show(self):
        """Berechnet 3D Mesh aus gewähltem Frame und öffnet Viewer."""
        if not self._selected_frames:
            print("Keine Frames ausgewählt")
            return

        try:
            from mesh_reconstruction import reconstruct_mesh, depth_to_pointcloud
            from stereo_depth_color import compute_depth, SGBM_FRONT
            import yaml

            # Stereo-Config laden
            K_l = np.array([[800,0,800],[0,800,600],[0,0,1]], dtype=np.float64)
            baseline_mm = 65.0
            try:
                with open(BASE / "stereo_config.yaml") as f:
                    cfg = yaml.safe_load(f)
                K_l = np.array(cfg['K_left'])
                baseline_mm = cfg.get('baseline_mm', 65)
            except: pass

            pcds = []

            # ELP Tiefenkarte
            if 'ELP_L' in self._selected_frames and 'ELP_R' in self._selected_frames:
                fl = self._selected_frames['ELP_L']
                fr = self._selected_frames['ELP_R']
                _, depth = compute_depth(fl, fr, SGBM_FRONT, K_l, baseline_mm)
                pcd = depth_to_pointcloud(fl, depth, K_l, step=2)
                if pcd: pcds.append(pcd)

            if pcds:
                mesh = reconstruct_mesh(pcds)
                self.viewer.show(mesh)
            else:
                print("Keine Tiefendaten verfügbar")

        except Exception as e:
            print(f"3D Fehler: {e}")


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ANTHRO3D Frame Selector Test")
    print("Öffnet Ring-Buffer mit Test-Frames\n")

    # Test mit Webcam
    ctrl = RecordingController()
    cap  = cv2.VideoCapture(0)

    ctrl.start()
    print("Aufnahme läuft — 5 Sekunden...")

    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret:
            ctrl.ring.add('ELP Stereo', frame)
        time.sleep(0.033)

    cap.release()
    print("5 Sekunden aufgenommen")
    ctrl.stop_and_select()

    # Warten bis Selector geschlossen
    time.sleep(30)
