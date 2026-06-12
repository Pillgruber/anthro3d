# ANTHRO3D - Camera Manager
# Erkennt und verwaltet alle angeschlossenen Kameras
# Jede Kamera ist austauschbar via config.yaml

import cv2
import yaml
import time
import threading
import numpy as np

class Camera:
    def __init__(self, config):
        self.id = config["id"]
        self.name = config["name"]
        self.device_index = config["device_index"]
        self.resolution = config["resolution"]
        self.fps = config["fps"]
        self.type = config["type"]
        self.position = config["position"]
        self.enabled = config["enabled"]
        self.cap = None
        self.frame = None
        self.timestamp = None
        self.running = False
        self.lock = threading.Lock()

    def connect(self):
        print(f"  Verbinde {self.name} (Index {self.device_index})...")
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            print(f"  FEHLER: {self.name} nicht gefunden!")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        print(f"  OK: {self.name} verbunden")
        return True

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.perf_counter_ns()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.timestamp

    def disconnect(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print(f"  {self.name} getrennt")


class CameraManager:
    def __init__(self, config_path="config.yaml"):
        print("ANTHRO3D Camera Manager startet...")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.cameras = []
        self._load_cameras()

    def _load_cameras(self):
        # Tracking Kameras (Mono)
        tracking = self.config["cameras"]["tracking"]
        for cam_config in tracking:
            if cam_config["enabled"]:
                self.cameras.append(Camera(cam_config))

        # Referenz Kamera (Farbe) - momentan disabled
        reference = self.config["cameras"]["reference"]
        for cam_config in reference:
            if cam_config["enabled"]:
                self.cameras.append(Camera(cam_config))

    def connect_all(self):
        print(f"\nVerbinde {len(self.cameras)} Kameras:")
        connected = []
        for cam in self.cameras:
            if cam.connect():
                connected.append(cam)
        self.cameras = connected
        print(f"\n{len(self.cameras)} Kameras verbunden\n")
        return len(self.cameras) > 0

    def start_all(self):
        for cam in self.cameras:
            cam.start()
        print("Alle Kameras laufen")

    def get_synced_frames(self):
        frames = {}
        timestamps = {}
        for cam in self.cameras:
            frame, ts = cam.get_frame()
            if frame is not None:
                frames[cam.id] = frame
                timestamps[cam.id] = ts
        return frames, timestamps

    def check_sync(self, timestamps):
        if len(timestamps) < 2:
            return True, 0
        ts_values = list(timestamps.values())
        max_offset_ns = max(ts_values) - min(ts_values)
        max_offset_ms = max_offset_ns / 1_000_000
        max_allowed = self.config["sync"]["max_offset_ms"]
        in_sync = max_offset_ms <= max_allowed
        return in_sync, max_offset_ms

    def disconnect_all(self):
        for cam in self.cameras:
            cam.disconnect()
        print("Alle Kameras getrennt")

    def list_cameras(self):
        print("\nVerfuegbare Kameras:")
        for cam in self.cameras:
            print(f"  ID:{cam.id} | {cam.name} | {cam.position} | {cam.type}")


# TEST - wird ausgefuehrt wenn du python3 camera_manager.py tippst
if __name__ == "__main__":
    manager = CameraManager()
    manager.list_cameras()

    if not manager.connect_all():
        print("Keine Kameras gefunden - pruefe Verbindungen")
        exit(1)

    manager.start_all()
    print("Druecke Q zum Beenden")

    while True:
        frames, timestamps = manager.get_synced_frames()

        # Sync pruefen
        in_sync, offset_ms = manager.check_sync(timestamps)
        sync_status = "SYNC OK" if in_sync else f"SYNC FEHLER: {offset_ms:.1f}ms"

        for cam_id, frame in frames.items():
            # Status ins Bild schreiben
            cv2.putText(frame, f"Kamera {cam_id} | {sync_status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Kamera {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    manager.disconnect_all()
    cv2.destroyAllWindows()
