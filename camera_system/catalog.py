from __future__ import annotations

from typing import Any

import AVFoundation
import CoreMedia


def _discover_device_objects() -> list[Any]:
    device_types = []

    for constant_name in (
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeExternalUnknown",
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    ):
        value = getattr(AVFoundation, constant_name, None)

        if value is not None and value not in device_types:
            device_types.append(value)

    if not device_types:
        raise RuntimeError(
            "Keine AVFoundation-Videogerätetypen verfügbar."
        )

    discovery = (
        AVFoundation.AVCaptureDeviceDiscoverySession
        .discoverySessionWithDeviceTypes_mediaType_position_(
            device_types,
            AVFoundation.AVMediaTypeVideo,
            AVFoundation.AVCaptureDevicePositionUnspecified,
        )
    )

    return list(discovery.devices())


def _dimensions(format_description: Any) -> tuple[int, int]:
    dimensions = CoreMedia.CMVideoFormatDescriptionGetDimensions(
        format_description
    )

    if hasattr(dimensions, "width"):
        return int(dimensions.width), int(dimensions.height)

    if isinstance(dimensions, (tuple, list)) and len(dimensions) >= 2:
        return int(dimensions[0]), int(dimensions[1])

    raise RuntimeError(
        f"Unbekannter Dimensionstyp: {dimensions!r}"
    )


def _fourcc_text(value: int) -> str:
    value = int(value)

    raw = bytes(
        (value >> shift) & 0xFF
        for shift in (24, 16, 8, 0)
    )

    if all(32 <= byte <= 126 for byte in raw):
        return raw.decode("ascii", errors="replace")

    return f"0x{value:08x}"


def discover_video_devices() -> list[dict[str, Any]]:
    """
    Erkennt beliebig viele AVFoundation-Videogeräte.

    Die Discovery-Position dient nur zur Anzeige.
    Identifiziert wird ausschließlich über unique_id.
    """
    result = []

    for discovery_position, device in enumerate(
        _discover_device_objects()
    ):
        formats = []

        for format_index, format_object in enumerate(
            list(device.formats())
        ):
            try:
                description = format_object.formatDescription()
                width, height = _dimensions(description)

                subtype = int(
                    CoreMedia.CMFormatDescriptionGetMediaSubType(
                        description
                    )
                )

                fps_ranges = []

                for rate_range in (
                    format_object.videoSupportedFrameRateRanges()
                ):
                    fps_ranges.append(
                        {
                            "min": float(
                                rate_range.minFrameRate()
                            ),
                            "max": float(
                                rate_range.maxFrameRate()
                            ),
                        }
                    )

                formats.append(
                    {
                        "format_index": format_index,
                        "width": width,
                        "height": height,
                        "fourcc": _fourcc_text(subtype),
                        "fourcc_value": subtype,
                        "fps_ranges": fps_ranges,
                    }
                )

            except Exception:
                continue

        result.append(
            {
                "discovery_position": discovery_position,
                "unique_id": str(device.uniqueID()),
                "name": str(device.localizedName()),
                "model_id": str(device.modelID()),
                "manufacturer": str(device.manufacturer()),
                "device_type": str(device.deviceType()),
                "formats": formats,
            }
        )

    return result


def catalog_by_unique_id() -> dict[str, dict[str, Any]]:
    return {
        device["unique_id"]: device
        for device in discover_video_devices()
    }
