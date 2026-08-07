from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import yaml

from camera_system.backends.avfoundation import AVFoundationCapture


DEFAULT_REGISTRY_PATH = Path(
    "~/anthro3d/camera_registry.scan.local.yaml"
).expanduser()


def _normalize_camera_name(name: str) -> str:
    return "".join(
        character
        for character in str(name).casefold()
        if character.isalnum()
    )


@dataclass(frozen=True)
class CameraProfile:
    name: str
    unique_id: str
    width: int
    height: int
    fps: float
    role: str
    confirmed: bool = False

    @property
    def capture_size(self) -> tuple[int, int]:
        return self.width, self.height


def load_camera_profiles(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, CameraProfile]:
    registry_path = Path(path).expanduser()

    if not registry_path.exists():
        raise RuntimeError(
            f"Scan-Kamera-Registry fehlt: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    try:
        raw_cameras = data["cameras"]
    except (TypeError, KeyError) as exc:
        raise RuntimeError(
            f"root.cameras fehlt in {registry_path}"
        ) from exc

    if not isinstance(raw_cameras, dict):
        raise RuntimeError(
            f"root.cameras ist keine Mapping-Struktur: {registry_path}"
        )

    profiles: dict[str, CameraProfile] = {}
    unique_ids: set[str] = set()

    for raw_name, raw_profile in raw_cameras.items():
        if not isinstance(raw_profile, dict):
            continue

        unique_id = str(
            raw_profile.get("unique_id", "")
        ).strip()

        if not unique_id:
            raise RuntimeError(f"{raw_name}: unique_id fehlt.")

        if unique_id in unique_ids:
            raise RuntimeError(
                f"Doppelte unique_id in Scan-Registry: {unique_id}"
            )
        unique_ids.add(unique_id)

        width = int(raw_profile.get("width", 0))
        height = int(raw_profile.get("height", 0))
        fps = float(raw_profile.get("fps", 0.0))

        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(
                f"{raw_name}: ungültiges Capture-Profil."
            )

        name = str(raw_name)
        key = _normalize_camera_name(name)

        if key in profiles:
            raise RuntimeError(
                f"Doppelter Kameraname nach Normalisierung: {name}"
            )

        profiles[key] = CameraProfile(
            name=name,
            unique_id=unique_id,
            width=width,
            height=height,
            fps=fps,
            role=str(raw_profile.get("role", name)),
            confirmed=bool(raw_profile.get("confirmed", False)),
        )

    return profiles


def get_camera_profile(
    name: str,
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> CameraProfile:
    profiles = load_camera_profiles(path)
    key = _normalize_camera_name(name)

    if key not in profiles:
        available = ", ".join(
            sorted(profile.name for profile in profiles.values())
        )
        raise RuntimeError(
            f"Kamera '{name}' fehlt in Scan-Registry. "
            f"Vorhanden: {available}"
        )

    return profiles[key]


class AnthroCameraCapture:
    """
    cv2.VideoCapture-ähnliche Fassade für scan3d.py.

    Die Kameraauswahl erfolgt ausschließlich über die stabile
    AVFoundation uniqueID aus camera_registry.scan.local.yaml.
    """

    def __init__(
        self,
        name: str,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        *,
        read_timeout: float = 2.0,
        capture_factory: Callable[..., Any] = AVFoundationCapture,
        open_immediately: bool = True,
    ) -> None:
        self.name = str(name)
        self.registry_path = Path(registry_path).expanduser()
        self.profile = get_camera_profile(
            self.name,
            self.registry_path,
        )

        self._read_timeout = float(read_timeout)
        if self._read_timeout < 0:
            raise ValueError(
                "read_timeout darf nicht negativ sein."
            )

        self._capture = capture_factory(
            unique_id=self.profile.unique_id,
            width=self.profile.width,
            height=self.profile.height,
            fps=self.profile.fps,
        )

        if open_immediately:
            self.open()

    @property
    def unique_id(self) -> str:
        return self.profile.unique_id

    @property
    def capture_size(self) -> tuple[int, int]:
        return self.profile.capture_size

    def open(self) -> bool:
        self._capture.open()
        return self.isOpened()

    def isOpened(self) -> bool:
        return bool(self._capture.is_open)

    def read(self):
        return self._capture.read(timeout=self._read_timeout)

    def release(self) -> None:
        self._capture.release()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.profile.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.profile.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.profile.fps)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        # Die echten Captureparameter werden beim Sessionstart
        # aus der Scan-Registry gesetzt. set() validiert hier nur.
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return int(round(value)) == self.profile.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return int(round(value)) == self.profile.height
        if prop_id == cv2.CAP_PROP_FPS:
            return abs(float(value) - self.profile.fps) <= 0.02
        return False

    def __enter__(self) -> "AnthroCameraCapture":
        if not self.isOpened():
            self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
