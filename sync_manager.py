# ANTHRO3D - Sync Manager
# Synchronisiert alle Kameras zeitlich
# Stellt sicher dass alle Frames zum gleichen Zeitpunkt aufgenommen wurden

import time
import numpy as np
from collections import deque


class SyncManager:
    def __init__(self, config):
        self.config = config
        self.max_offset_ms = config["sync"]["max_offset_ms"]
        self.buffer_size = config["sync"]["buffer_size"]

        # Frame-Buffer fuer jede Kamera
        self.buffers = {}

        # Statistiken
        self.sync_ok_count = 0
        self.sync_fail_count = 0
        self.offset_history = deque(maxlen=100)

        print("Sync Manager bereit")
        print(f"  Max. Zeitversatz: {self.max_offset_ms}ms")

    def add_camera(self, cam_id):
        self.buffers[cam_id] = deque(maxlen=self.buffer_size)
        print(f"  Kamera {cam_id} zum Sync hinzugefuegt")

    def add_frame(self, cam_id, frame, timestamp_ns):
        if cam_id in self.buffers:
            self.buffers[cam_id].append({
                "frame": frame,
                "timestamp": timestamp_ns
            })

    def get_synced_frames(self):
        # Pruefe ob alle Kameras Frames haben
        if not all(len(buf) > 0 for buf in self.buffers.values()):
            return None, None, False

        # Neuesten Frame jeder Kamera holen
        latest_frames = {}
        latest_timestamps = {}

        for cam_id, buffer in self.buffers.items():
            if len(buffer) > 0:
                entry = buffer[-1]
                latest_frames[cam_id] = entry["frame"]
                latest_timestamps[cam_id] = entry["timestamp"]

        # Zeitversatz berechnen
        in_sync, offset_ms = self._check_sync(latest_timestamps)
        self.offset_history.append(offset_ms)

        if in_sync:
            self.sync_ok_count += 1
        else:
            self.sync_fail_count += 1

        return latest_frames, latest_timestamps, in_sync

    def _check_sync(self, timestamps):
        if len(timestamps) < 2:
            return True, 0.0

        ts_values = list(timestamps.values())
        max_offset_ns = max(ts_values) - min(ts_values)
        offset_ms = max_offset_ns / 1_000_000

        in_sync = offset_ms <= self.max_offset_ms
        return in_sync, offset_ms

    def get_stats(self):
        total = self.sync_ok_count + self.sync_fail_count
        if total == 0:
            return {
                "sync_rate": 0,
                "avg_offset_ms": 0,
                "max_offset_ms": 0,
                "ok": 0,
                "fail": 0
            }

        avg_offset = np.mean(self.offset_history) if self.offset_history else 0
        max_offset = np.max(self.offset_history) if self.offset_history else 0

        return {
            "sync_rate": round(self.sync_ok_count / total * 100, 1),
            "avg_offset_ms": round(avg_offset, 2),
            "max_offset_ms": round(max_offset, 2),
            "ok": self.sync_ok_count,
            "fail": self.sync_fail_count
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\nSync Statistiken:")
        print(f"  Sync-Rate:         {stats['sync_rate']}%")
        print(f"  Avg. Versatz:      {stats['avg_offset_ms']}ms")
        print(f"  Max. Versatz:      {stats['max_offset_ms']}ms")
        print(f"  Frames OK:         {stats['ok']}")
        print(f"  Frames FEHLER:     {stats['fail']}")


class FrameTimestamper:
    """
    Hilfsklasse - gibt jedem Frame einen praezisen Zeitstempel
    Wird in camera_manager._capture_loop verwendet
    """
    def __init__(self):
        self.start_time = time.perf_counter_ns()

    def now_ns(self):
        return time.perf_counter_ns()

    def now_ms(self):
        return self.now_ns() / 1_000_000

    def elapsed_ms(self):
        return (self.now_ns() - self.start_time) / 1_000_000


# TEST
if __name__ == "__main__":
    import yaml
    import numpy as np

    print("ANTHRO3D Sync Manager Test")
    print("==========================\n")

    # Config laden
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Sync Manager erstellen
    sync = SyncManager(config)

    # Simuliere 3 Kameras
    print("\nSimuliere 3 Kameras...")
    sync.add_camera(1)
    sync.add_camera(2)
    sync.add_camera(3)

    # Simuliere 100 Frames mit verschiedenen Zeitversaetzen
    print("Simuliere 100 Frame-Paare...\n")
    timestamper = FrameTimestamper()

    for i in range(100):
        base_time = timestamper.now_ns()

        # Simuliere kleine Zeitversaetze zwischen Kameras (0-10ms)
        offsets = [0, np.random.randint(0, 8_000_000), np.random.randint(0, 8_000_000)]

        for cam_id, offset in zip([1, 2, 3], offsets):
            fake_frame = np.zeros((720, 1280), dtype=np.uint8)
            sync.add_frame(cam_id, fake_frame, base_time + offset)

        frames, timestamps, in_sync = sync.get_synced_frames()

        if i % 20 == 0:
            status = "OK" if in_sync else "FEHLER"
            print(f"  Frame {i:3d}: {status}")

    sync.print_stats()
    print("\nTest abgeschlossen!")
