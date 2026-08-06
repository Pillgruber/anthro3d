from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROFILES_PATH = PROJECT_DIR / "camera_profiles.yaml"
REGISTRY_PATH = PROJECT_DIR / "camera_registry.local.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Konfigurationsdatei fehlt: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)

    if not isinstance(document, dict):
        raise ValueError(
            f"{path.name} muss ein YAML-Objekt enthalten."
        )

    return document


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def load_resolved_registry(
    profiles_path: Path = PROFILES_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> list[dict[str, Any]]:
    """
    Lädt beliebig viele physische Kameras.

    Kameras können ein wiederverwendbares Profil verwenden und
    einzelne Werte anschließend lokal überschreiben.
    """
    profiles_document = _load_yaml(profiles_path)
    registry_document = _load_yaml(registry_path)

    profiles = profiles_document.get("profiles", {})
    devices = registry_document.get("devices", [])

    if not isinstance(profiles, dict):
        raise ValueError(
            "camera_profiles.yaml: profiles muss ein Objekt sein."
        )

    if not isinstance(devices, list):
        raise ValueError(
            "camera_registry.local.yaml: devices muss eine Liste sein."
        )

    resolved = []
    camera_ids = set()
    unique_ids = set()

    for position, entry in enumerate(devices):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Registry-Eintrag {position} ist ungültig."
            )

        profile_name = entry.get("profile")
        profile = {}

        if profile_name is not None:
            if profile_name not in profiles:
                raise ValueError(
                    f"Unbekanntes Kameraprofil: {profile_name!r}"
                )

            profile = profiles[profile_name]

            if not isinstance(profile, dict):
                raise ValueError(
                    f"Kameraprofil {profile_name!r} ist ungültig."
                )

        local_values = {
            key: value
            for key, value in entry.items()
            if key != "profile"
        }

        device = _deep_merge(profile, local_values)
        device["profile"] = profile_name
        device.setdefault("enabled", True)
        device.setdefault("required", True)

        camera_id = device.get("camera_id")
        unique_id = device.get("unique_id")

        if not camera_id:
            raise ValueError(
                f"Registry-Eintrag {position}: camera_id fehlt."
            )

        if not unique_id:
            raise ValueError(
                f"Kamera {camera_id!r}: unique_id fehlt."
            )

        camera_id = str(camera_id)
        unique_id = str(unique_id)

        if camera_id in camera_ids:
            raise ValueError(
                f"camera_id ist doppelt: {camera_id!r}"
            )

        if unique_id in unique_ids:
            raise ValueError(
                f"unique_id wird mehrfach verwendet: {unique_id}"
            )

        camera_ids.add(camera_id)
        unique_ids.add(unique_id)

        device["camera_id"] = camera_id
        device["unique_id"] = unique_id

        resolved.append(device)

    return resolved
