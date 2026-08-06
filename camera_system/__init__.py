"""Generisches AVFoundation-Kamerasystem für ANTHRO3D."""

from .catalog import (
    catalog_by_unique_id,
    discover_video_devices,
)
from .hotplug import (
    CameraHotplugMonitor,
    CameraPresence,
)
from .readiness import (
    CameraReadinessReport,
    CameraSystemNotReadyError,
    assert_camera_system_ready,
    check_camera_readiness,
    format_readiness_report,
)
from .registry import load_resolved_registry

__all__ = [
    "CameraHotplugMonitor",
    "CameraPresence",
    "CameraReadinessReport",
    "CameraSystemNotReadyError",
    "assert_camera_system_ready",
    "catalog_by_unique_id",
    "check_camera_readiness",
    "discover_video_devices",
    "format_readiness_report",
    "load_resolved_registry",
]
