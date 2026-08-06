from __future__ import annotations

import json
import sys

from camera_system.main_thread_hotplug import (
    AVFoundationMainThreadHotplugObserver,
)
from camera_system.manager import CameraManager, CameraSpec
from camera_system.registry import load_resolved_registry
from camera_system.runtime import MainThreadCameraRuntime


def specs_from_registry() -> list[CameraSpec]:
    devices = load_resolved_registry()

    return [
        CameraSpec(
            camera_id=str(device["camera_id"]),
            unique_id=str(device["unique_id"]),
            profile=(
                None
                if device.get("profile") is None
                else str(device["profile"])
            ),
            enabled=bool(device.get("enabled", True)),
            required=bool(device.get("required", True)),
            metadata={
                key: value
                for key, value in device.items()
                if key
                not in {
                    "camera_id",
                    "unique_id",
                    "profile",
                    "enabled",
                    "required",
                }
            },
        )
        for device in devices
    ]


def print_snapshot(manager: CameraManager) -> None:
    print(json.dumps(manager.snapshot(), indent=2, ensure_ascii=False))


def main() -> None:
    manager = CameraManager(specs_from_registry())
    observer = AVFoundationMainThreadHotplugObserver(manager.event_queue)
    runtime = MainThreadCameraRuntime(manager, observer)

    initial_events = runtime.start(seed_existing=True)

    print(f"Initiale Geräteereignisse: {len(initial_events)}")
    print_snapshot(manager)
    print()
    print("Hauptthread-Hot-Plug läuft.")
    print("Eine konfigurierte Kamera abziehen und wieder anschließen.")
    print("Beenden mit Ctrl+C.")

    try:
        while True:
            events = runtime.pump_once(timeout=0.25)

            if events:
                print()
                for event in events:
                    print(
                        f"{event.event_type.value.upper()}: "
                        f"{event.unique_id} | {dict(event.metadata)}"
                    )
                print_snapshot(manager)
    except KeyboardInterrupt:
        print()
        print("Hauptthread-Hot-Plug-Test beendet.")
    finally:
        runtime.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        raise
