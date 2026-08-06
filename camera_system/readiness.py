from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .catalog import catalog_by_unique_id


class CameraSystemNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraReadinessReport:
    required_count: int
    present_required_count: int
    missing_required: tuple[dict[str, Any], ...]
    missing_optional: tuple[dict[str, Any], ...]
    unregistered_devices: tuple[dict[str, Any], ...]

    @property
    def ready(self) -> bool:
        return not self.missing_required


def check_camera_readiness(
    devices: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, dict[str, Any]] | None = None,
) -> CameraReadinessReport:
    if catalog is None:
        catalog = catalog_by_unique_id()

    required_count = 0
    present_required_count = 0
    missing_required = []
    missing_optional = []
    registered_ids = set()

    for device in devices:
        if device.get("enabled", True) is False:
            continue

        unique_id = str(device["unique_id"])
        required = bool(device.get("required", True))

        registered_ids.add(unique_id)

        if required:
            required_count += 1

        if unique_id in catalog:
            if required:
                present_required_count += 1
            continue

        if required:
            missing_required.append(dict(device))
        else:
            missing_optional.append(dict(device))

    unregistered = tuple(
        catalog_device
        for unique_id, catalog_device in catalog.items()
        if unique_id not in registered_ids
    )

    return CameraReadinessReport(
        required_count=required_count,
        present_required_count=present_required_count,
        missing_required=tuple(missing_required),
        missing_optional=tuple(missing_optional),
        unregistered_devices=unregistered,
    )


def format_readiness_report(
    report: CameraReadinessReport,
) -> str:
    lines = [
        (
            "Kamera-Systemprüfung: "
            f"{report.present_required_count}/"
            f"{report.required_count} Pflichtkameras vorhanden"
        )
    ]

    if report.missing_required:
        lines.append("")
        lines.append("Fehlende Pflichtkameras:")

        for device in report.missing_required:
            lines.append(
                f"  - {device['camera_id']} "
                f"({device['unique_id']})"
            )

    if report.missing_optional:
        lines.append("")
        lines.append("Fehlende optionale Kameras:")

        for device in report.missing_optional:
            lines.append(
                f"  - {device['camera_id']} "
                f"({device['unique_id']})"
            )

    return "\n".join(lines)


def assert_camera_system_ready(
    devices: Sequence[Mapping[str, Any]],
) -> CameraReadinessReport:
    report = check_camera_readiness(devices)
    message = format_readiness_report(report)

    print(message)

    if not report.ready:
        raise CameraSystemNotReadyError(message)

    return report
