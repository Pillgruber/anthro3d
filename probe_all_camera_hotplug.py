#!/usr/bin/env python3

from __future__ import annotations

import time

from camera_system.catalog import discover_video_devices


def snapshot():
    return {
        device["unique_id"]: device
        for device in discover_video_devices()
    }


def print_devices(devices, title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for unique_id, device in sorted(devices.items()):
        print(
            f"{device['name']} | "
            f"{device['model_id']} | "
            f"{unique_id}"
        )


def main():
    previous = snapshot()
    print_devices(previous, "Beim Start erkannte Kameras")

    print()
    print("Jetzt Kameras einzeln abziehen und wieder anschließen.")
    print("Beenden mit Ctrl+C.")

    try:
        while True:
            time.sleep(1.0)
            current = snapshot()

            removed_ids = set(previous) - set(current)
            added_ids = set(current) - set(previous)

            for unique_id in sorted(removed_ids):
                device = previous[unique_id]

                print()
                print("KAMERA ENTFERNT:")
                print(f"  Name:      {device['name']}")
                print(f"  Modell:    {device['model_id']}")
                print(f"  uniqueID:  {unique_id}")

            for unique_id in sorted(added_ids):
                device = current[unique_id]

                print()
                print("KAMERA HINZUGEFÜGT:")
                print(f"  Name:      {device['name']}")
                print(f"  Modell:    {device['model_id']}")
                print(f"  uniqueID:  {unique_id}")

            previous = current

    except KeyboardInterrupt:
        print()
        print("Gerätebeobachtung beendet.")


if __name__ == "__main__":
    main()
