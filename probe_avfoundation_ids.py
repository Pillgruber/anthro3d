#!/usr/bin/env python3

from __future__ import annotations

import AVFoundation


def safe_call(obj, method_name, default=""):
    try:
        result = getattr(obj, method_name)()
        return default if result is None else result
    except Exception:
        return default


def discover_video_devices():
    device_types = []

    for constant_name in (
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeExternalUnknown",
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    ):
        value = getattr(AVFoundation, constant_name, None)

        if value is not None and value not in device_types:
            device_types.append(value)

    if device_types:
        discovery = (
            AVFoundation.AVCaptureDeviceDiscoverySession
            .discoverySessionWithDeviceTypes_mediaType_position_(
                device_types,
                AVFoundation.AVMediaTypeVideo,
                AVFoundation.AVCaptureDevicePositionUnspecified,
            )
        )
        return list(discovery.devices())

    return list(
        AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
    )


devices = discover_video_devices()

print(f"Gefundene Videogeräte: {len(devices)}")
print()

for position, device in enumerate(devices):
    name = str(safe_call(device, "localizedName"))
    unique_id = str(safe_call(device, "uniqueID"))
    model_id = str(safe_call(device, "modelID"))
    manufacturer = str(safe_call(device, "manufacturer"))
    device_type = str(safe_call(device, "deviceType"))

    try:
        format_count = len(list(device.formats()))
    except Exception:
        format_count = 0

    print("=" * 78)
    print(f"Position:     {position}")
    print(f"Name:         {name}")
    print(f"Unique ID:    {unique_id}")
    print(f"Modell-ID:    {model_id}")
    print(f"Hersteller:   {manufacturer}")
    print(f"Gerätetyp:    {device_type}")
    print(f"Formatanzahl: {format_count}")
    print()
