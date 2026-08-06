#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from pathlib import Path

from camera_system.hotplug import CameraHotplugMonitor
from camera_system.registry import load_resolved_registry


REGISTRY_PATH = Path("camera_registry.local.yaml")


def camera_connected(config, device):
    try:
        device_name = str(device.localizedName())
    except Exception:
        device_name = "Unbekannte Kamera"

    print()
    print(
        "Capture-Session kann gestartet werden:"
        f"\n  Rolle:    {config['camera_id']}"
        f"\n  Name:     {device_name}"
        f"\n  uniqueID: {config['unique_id']}"
    )


def camera_disconnected(config):
    print()
    print(
        "Capture-Session muss pausiert werden:"
        f"\n  Rolle:    {config['camera_id']}"
        f"\n  uniqueID: {config['unique_id']}"
    )


def main():
    if not REGISTRY_PATH.exists():
        raise SystemExit(
            "camera_registry.local.yaml fehlt.\n"
            "Ohne Registry weiß das System nicht, "
            "welche Kameras erwartet werden."
        )

    devices = load_resolved_registry()

    enabled_devices = [
        device
        for device in devices
        if device.get("enabled", True)
    ]

    if not enabled_devices:
        raise SystemExit(
            "In camera_registry.local.yaml ist keine Kamera aktiviert."
        )

    monitor = CameraHotplugMonitor(
        enabled_devices,
        poll_interval_seconds=1.0,
        on_connected=camera_connected,
        on_disconnected=camera_disconnected,
    )

    monitor.start()
    monitor.print_status()

    missing = monitor.missing_required()

    if missing:
        print()
        print("Anthro3D wartet auf fehlende Pflichtkameras:")

        for config in missing:
            print(
                f"  - {config['camera_id']} "
                f"({config['unique_id']})"
            )

        print()
        print(
            "Die fehlenden Kameras können jetzt angeschlossen werden."
        )

    try:
        if monitor.wait_for_required(timeout_seconds=None):
            print()
            print("Alle erforderlichen Kameras sind verbunden.")
            print("Der Scanbetrieb kann gestartet werden.")
            monitor.print_status()

        print()
        print(
            "Hot-Plug-Überwachung läuft weiter. "
            "Kamera zum Testen abziehen und wieder anschließen."
        )
        print("Beenden mit Ctrl+C.")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
        print("Hot-Plug-Test beendet.")

    finally:
        monitor.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        raise
