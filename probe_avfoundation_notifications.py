#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime

import AVFoundation
import Foundation
import objc


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def device_value(device, method_name, default="?"):
    if device is None:
        return default

    try:
        value = getattr(device, method_name)()
        return default if value is None else str(value)
    except Exception:
        return default


class CameraNotificationObserver(Foundation.NSObject):
    def cameraConnected_(self, notification):
        device = notification.object()

        print()
        print("=" * 72)
        print(f"{timestamp()}  KAMERA VERBUNDEN")
        print(f"Name:      {device_value(device, 'localizedName')}")
        print(f"Unique ID: {device_value(device, 'uniqueID')}")
        print(f"Modell:    {device_value(device, 'modelID')}")
        print("=" * 72, flush=True)

    def cameraDisconnected_(self, notification):
        device = notification.object()

        print()
        print("=" * 72)
        print(f"{timestamp()}  KAMERA GETRENNT")
        print(f"Name:      {device_value(device, 'localizedName')}")
        print(f"Unique ID: {device_value(device, 'uniqueID')}")
        print(f"Modell:    {device_value(device, 'modelID')}")
        print("=" * 72, flush=True)


def discover_devices():
    device_types = []

    for constant_name in (
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeExternalUnknown",
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    ):
        value = getattr(AVFoundation, constant_name, None)

        if value is not None and value not in device_types:
            device_types.append(value)

    discovery = (
        AVFoundation.AVCaptureDeviceDiscoverySession
        .discoverySessionWithDeviceTypes_mediaType_position_(
            device_types,
            AVFoundation.AVMediaTypeVideo,
            AVFoundation.AVCaptureDevicePositionUnspecified,
        )
    )

    return list(discovery.devices())


def main():
    # Der Observer muss während des gesamten Tests gehalten werden.
    observer = CameraNotificationObserver.alloc().init()
    center = Foundation.NSNotificationCenter.defaultCenter()

    center.addObserver_selector_name_object_(
        observer,
        b"cameraConnected:",
        AVFoundation.AVCaptureDeviceWasConnectedNotification,
        None,
    )

    center.addObserver_selector_name_object_(
        observer,
        b"cameraDisconnected:",
        AVFoundation.AVCaptureDeviceWasDisconnectedNotification,
        None,
    )

    print("Beim Start erkannte Geräte:")
    print()

    for device in discover_devices():
        print(
            f"  {device_value(device, 'localizedName')} | "
            f"{device_value(device, 'uniqueID')}"
        )

    print()
    print("Direkte AVFoundation-Benachrichtigungen sind aktiv.")
    print("Jetzt den OV9281-Hub abziehen, 10 Sekunden warten")
    print("und anschließend wieder verbinden.")
    print("Beenden mit Ctrl+C.")

    try:
        # Bewusst auf dem Hauptthread:
        # Foundation-Ereignisse werden hier direkt verarbeitet.
        while True:
            until = Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.25)

            Foundation.NSRunLoop.currentRunLoop().runMode_beforeDate_(
                Foundation.NSDefaultRunLoopMode,
                until,
            )

    except KeyboardInterrupt:
        print()
        print("Benachrichtigungstest beendet.")

    finally:
        center.removeObserver_(observer)


if __name__ == "__main__":
    main()
