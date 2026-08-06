#!/usr/bin/env python3

from __future__ import annotations

import re
import time

from camera_system.catalog import discover_video_devices
from camera_system.hotplug import CameraHotplugMonitor


def safe_id(text):
    value = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return value.strip("_") or "camera"


def connected(config, device):
    print(
        f"AKTIV: {config['camera_id']} | "
        f"{config['unique_id']}"
    )


def disconnected(config):
    print(
        f"PAUSIERT: {config['camera_id']} | "
        f"{config['unique_id']}"
    )


devices = []

for position, device in enumerate(discover_video_devices()):
    if "External" not in device.get("device_type", ""):
        continue

    devices.append(
        {
            "camera_id": (
                f"test_{position}_"
                f"{safe_id(device['name'])}"
            ),
            "unique_id": device["unique_id"],
            "enabled": True,
            "required": False,
        }
    )

print(f"Überwachte externe Kameras: {len(devices)}")

for device in devices:
    print(
        f"  {device['camera_id']} | "
        f"{device['unique_id']}"
    )

monitor = CameraHotplugMonitor(
    devices,
    poll_interval_seconds=1.0,
    on_connected=connected,
    on_disconnected=disconnected,
)

monitor.start()
monitor.print_status()

print()
print("Jetzt eine Kamera oder den USB-Hub abziehen.")
print("Mindestens fünf Sekunden warten und wieder verbinden.")
print("Beenden mit Ctrl+C.")

try:
    while True:
        time.sleep(1.0)

except KeyboardInterrupt:
    print()
    print("Test beendet.")

finally:
    monitor.stop()
