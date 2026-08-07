from __future__ import annotations

import re
import threading
import time
from typing import Any

import AppKit
import AVFoundation
import CoreMedia
import Foundation
import Quartz
import dispatch
import cv2
import numpy as np
import objc


VideoDelegateProtocol = objc.protocolNamed(
    "AVCaptureVideoDataOutputSampleBufferDelegate"
)


def _safe_call(
    obj: Any,
    method_name: str,
    default: Any = None,
) -> Any:
    try:
        return getattr(obj, method_name)()
    except Exception:
        return default


def _unpack_error_result(result: Any) -> tuple[Any, Any]:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], result[1]
        if len(result) == 1:
            return result[0], None
    return result, None


def _format_dimensions(fmt: Any) -> tuple[int, int] | None:
    description = _safe_call(fmt, "formatDescription")
    if description is None:
        return None

    try:
        dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(
            description
        )
        return int(dims.width), int(dims.height)
    except Exception:
        return None


def _matching_frame_rate_range(
    fmt: Any,
    fps: float,
) -> Any | None:
    ranges = _safe_call(
        fmt,
        "videoSupportedFrameRateRanges",
        [],
    ) or []

    tolerance = 0.02
    candidates: list[tuple[float, Any]] = []

    for item in ranges:
        min_rate = _safe_call(item, "minFrameRate")
        max_rate = _safe_call(item, "maxFrameRate")

        if min_rate is None or max_rate is None:
            continue

        try:
            minimum = float(min_rate)
            maximum = float(max_rate)
        except Exception:
            continue

        if (
            minimum - tolerance
            <= fps
            <= maximum + tolerance
        ):
            midpoint = (minimum + maximum) / 2.0
            candidates.append(
                (abs(midpoint - fps), item)
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _format_supports_fps(fmt: Any, fps: float) -> bool:
    return _matching_frame_rate_range(fmt, fps) is not None


def _find_exact_format(
    device: Any,
    width: int,
    height: int,
    fps: float,
) -> Any:
    matches: list[Any] = []

    for fmt in list(_safe_call(device, "formats", []) or []):
        if _format_dimensions(fmt) != (width, height):
            continue

        if _format_supports_fps(fmt, fps):
            matches.append(fmt)

    if not matches:
        raise RuntimeError(
            f"Kein AVFoundation-Format für "
            f"{width}x{height} @ {fps:g} fps gefunden."
        )

    # Mehrere UVC-Formate können dieselbe Auflösung melden.
    # Für ANTHRO3D genügt zunächst das erste Format, das die
    # gewünschte Bildrate explizit unterstützt. Dadurch werden
    # z. B. die 2-fps-Duplikate der ELP-Kameras ausgeschlossen.
    return matches[0]


def _find_device_by_unique_id(unique_id: str) -> Any:
    devices = list(
        AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
    )

    for device in devices:
        if str(_safe_call(device, "uniqueID", "")) == unique_id:
            return device

    raise RuntimeError(
        f"AVFoundation-Kamera nicht gefunden: {unique_id}"
    )


def _queue_label(unique_id: str) -> bytes:
    safe = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        unique_id,
    ).strip("_")
    return f"at.anthro3d.capture.{safe or 'camera'}".encode(
        "utf-8"
    )


def _acquire_device_lock(device: Any) -> None:
    result = device.lockForConfiguration_(None)

    if isinstance(result, tuple):
        ok = bool(result[0])
        error = result[1] if len(result) > 1 else None

        if not ok:
            raise RuntimeError(
                "AVCaptureDevice konnte nicht dauerhaft "
                f"gesperrt werden: {error}"
            )

    elif result is False:
        raise RuntimeError(
            "AVCaptureDevice konnte nicht dauerhaft "
            "gesperrt werden."
        )


def _configure_device(
    device: Any,
    fmt: Any,
    fps: float,
) -> None:
    lock_result = device.lockForConfiguration_(None)

    if isinstance(lock_result, tuple):
        ok = bool(lock_result[0])
        error = (
            lock_result[1]
            if len(lock_result) > 1
            else None
        )
        if not ok:
            raise RuntimeError(
                "AVCaptureDevice konnte nicht für die "
                f"Konfiguration gesperrt werden: {error}"
            )
    elif lock_result is False:
        raise RuntimeError(
            "AVCaptureDevice konnte nicht für die "
            "Konfiguration gesperrt werden."
        )

    try:
        device.setActiveFormat_(fmt)

        frame_range = _matching_frame_rate_range(
            fmt,
            fps,
        )
        if frame_range is None:
            raise RuntimeError(
                f"Das gewählte Format unterstützt "
                f"{fps:g} fps nicht."
            )

        min_rate = float(
            _safe_call(frame_range, "minFrameRate", 0.0)
        )
        max_rate = float(
            _safe_call(frame_range, "maxFrameRate", 0.0)
        )

        # UVC-Kameras können diskrete Bildraten mit leicht
        # abweichender rationaler Zeitbasis melden, z. B.
        # 30.00003 fps. AVFoundation verlangt dann exakt
        # die vom AVFrameRateRange gemeldete CMTime.
        if abs(max_rate - min_rate) <= 0.02:
            duration = _safe_call(
                frame_range,
                "minFrameDuration",
            )
            if duration is None:
                raise RuntimeError(
                    "AVFrameRateRange liefert keine "
                    "minFrameDuration."
                )

            device.setActiveVideoMinFrameDuration_(duration)
            device.setActiveVideoMaxFrameDuration_(duration)

        else:
            duration = CoreMedia.CMTimeMakeWithSeconds(
                1.0 / fps,
                1_000_000_000,
            )
            device.setActiveVideoMinFrameDuration_(duration)
            device.setActiveVideoMaxFrameDuration_(duration)

    finally:
        device.unlockForConfiguration()


def _safe_memoryview(value: Any) -> memoryview | None:
    try:
        return memoryview(value)
    except Exception:
        return None


def _nsdata_to_bytes(data: Any) -> bytes:
    try:
        return bytes(data)
    except Exception:
        length = int(data.length())
        buffer = bytearray(length)
        data.getBytes_length_(buffer, length)
        return bytes(buffer)


def _pixel_buffer_to_bgr_via_coreimage(
    pixel_buffer: Any,
    ci_context: Any | None = None,
    render_buffer: bytearray | None = None,
    color_space: Any | None = None,
) -> np.ndarray:
    # CVPixelBuffer -> CIImage -> direkter BGRA8-Render -> NumPy BGR.
    # Kein CGImage, kein NSBitmapImageRep, kein PNG, kein cv2.imdecode.
    # Der große BGRA-Arbeitspuffer kann wiederverwendet werden.
    width = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
    height = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Ungültige Pixelbuffer-Größe: {width}x{height}"
        )

    ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(
        pixel_buffer
    )
    if ci_image is None:
        raise RuntimeError(
            "CIImage konnte nicht aus CVPixelBuffer erzeugt werden."
        )

    context = ci_context
    if context is None:
        context = Quartz.CIContext.contextWithOptions_(None)

    rgb_space = color_space
    if rgb_space is None:
        rgb_space = Quartz.CGColorSpaceCreateDeviceRGB()

    row_bytes = width * 4
    byte_count = row_bytes * height

    bitmap = render_buffer
    if bitmap is None:
        bitmap = bytearray(byte_count)

    if len(bitmap) != byte_count:
        raise RuntimeError(
            "CoreImage-Arbeitspuffer hat falsche Größe: "
            f"{len(bitmap)} statt {byte_count} Bytes."
        )

    bounds = Quartz.CGRectMake(
        0.0,
        0.0,
        float(width),
        float(height),
    )

    context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_image,
        bitmap,
        row_bytes,
        bounds,
        Quartz.kCIFormatBGRA8,
        rgb_space,
    )

    bgra = np.frombuffer(
        bitmap,
        dtype=np.uint8,
        count=byte_count,
    ).reshape(
        height,
        width,
        4,
    )

    # Eigene BGR-Kopie, weil bitmap beim nächsten Frame wiederverwendet wird.
    return bgra[:, :, :3].copy()


def _pixel_buffer_to_bgr(
    pixel_buffer: Any,
    ci_context: Any | None = None,
    render_buffer: bytearray | None = None,
    color_space: Any | None = None,
) -> np.ndarray:
    width = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
    height = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))
    bytes_per_row = int(
        Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Ungültige Pixelbuffer-Größe: {width}x{height}"
        )

    expected_row_bytes = width * 4
    if bytes_per_row < expected_row_bytes:
        raise RuntimeError(
            "CVPixelBuffer hat weniger Bytes pro Zeile als "
            "für BGRA erforderlich."
        )

    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)

    try:
        base_address = Quartz.CVPixelBufferGetBaseAddress(
            pixel_buffer
        )

        if base_address is not None:
            byte_count = height * bytes_per_row

            for candidate in (
                base_address,
                _safe_memoryview(base_address),
            ):
                if candidate is None:
                    continue

                try:
                    raw = np.frombuffer(
                        candidate,
                        dtype=np.uint8,
                        count=byte_count,
                    )
                except (
                    TypeError,
                    BufferError,
                    ValueError,
                ):
                    continue

                if raw.size < byte_count:
                    continue

                rows = raw.reshape(
                    height,
                    bytes_per_row,
                )
                packed = rows[:, :expected_row_bytes]
                bgra = packed.reshape(
                    height,
                    width,
                    4,
                )
                return bgra[:, :, :3].copy()

    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(
            pixel_buffer,
            0,
        )

    return _pixel_buffer_to_bgr_via_coreimage(
        pixel_buffer,
        ci_context,
        render_buffer,
        color_space,
    )


class _FrameDelegate(
    Foundation.NSObject,
    protocols=[VideoDelegateProtocol],
):
    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self,
        output,
        sample_buffer,
        connection,
    ):
        owner = getattr(self, "owner", None)
        if owner is None:
            return

        try:
            owner._handle_sample_buffer(sample_buffer)
        except Exception as exc:
            # Nie eine Python-Ausnahme aus einem libdispatch-
            # Objective-C-Callback entweichen lassen.
            owner._publish_callback_error(exc)


class AVFoundationCapture:
    """
    Direkter AVFoundation-Capture über stabile uniqueID.

    Die Klasse puffert ausschließlich das neueste Frame.
    Alte Frames werden verworfen, um Latenzaufbau zu vermeiden.
    """

    def __init__(
        self,
        unique_id: str,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> None:
        if not str(unique_id).strip():
            raise ValueError("unique_id darf nicht leer sein.")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(
                "width und height müssen größer als 0 sein."
            )
        if float(fps) <= 0:
            raise ValueError("fps muss größer als 0 sein.")

        self.unique_id = str(unique_id)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)

        self._condition = threading.Condition()
        self._is_open = False
        self._latest_frame: np.ndarray | None = None
        self._latest_sequence = 0
        self._last_returned_sequence = 0
        self._last_error: str | None = None

        # AVFoundation liefert mit Geräte-FPS. Die teure
        # CVPixelBuffer->BGR-Konvertierung erfolgt aber nur,
        # wenn read() tatsächlich ein neues Frame benötigt.
        self._frame_requested = False
        self._conversion_in_progress = False

        # CoreImage-Ressourcen pro Capture-Instanz wiederverwenden.
        self._ci_context = Quartz.CIContext.contextWithOptions_(None)
        self._ci_color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        self._ci_render_buffer = bytearray(
            self.width * self.height * 4
        )

        self._device: Any = None
        self._session: Any = None
        self._input: Any = None
        self._output: Any = None
        self._delegate: Any = None
        self._callback_queue: Any = None
        self._selected_format: Any = None
        self._device_lock_held = False

    @property
    def is_open(self) -> bool:
        with self._condition:
            return self._is_open

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    @property
    def selected_dimensions(self) -> tuple[int, int] | None:
        fmt = self._selected_format
        if fmt is None:
            return None
        return _format_dimensions(fmt)

    @property
    def device_name(self) -> str | None:
        if self._device is None:
            return None
        return str(
            _safe_call(
                self._device,
                "localizedName",
                "Kamera",
            )
        )

    def open(self) -> None:
        with self._condition:
            if self._is_open:
                return

        device = _find_device_by_unique_id(self.unique_id)
        fmt = _find_exact_format(
            device,
            self.width,
            self.height,
            self.fps,
        )

        input_result = (
            AVFoundation.AVCaptureDeviceInput
            .deviceInputWithDevice_error_(
                device,
                None,
            )
        )
        camera_input, input_error = _unpack_error_result(
            input_result
        )

        if camera_input is None:
            raise RuntimeError(
                "AVCaptureDeviceInput konnte nicht erzeugt werden"
                + (
                    f": {input_error}"
                    if input_error is not None
                    else "."
                )
            )

        session = AVFoundation.AVCaptureSession.alloc().init()

        video_output = (
            AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        )
        delegate = _FrameDelegate.alloc().init()
        delegate.owner = self

        callback_queue = dispatch.dispatch_queue_create(
            _queue_label(self.unique_id),
            None,
        )

        device_lock_held = False

        session.beginConfiguration()

        try:
            input_priority = getattr(
                AVFoundation,
                "AVCaptureSessionPresetInputPriority",
                None,
            )
            if (
                input_priority is not None
                and session.canSetSessionPreset_(input_priority)
            ):
                session.setSessionPreset_(input_priority)

            if not session.canAddInput_(camera_input):
                raise RuntimeError(
                    "AVCaptureSession kann Kamera-Input "
                    "nicht hinzufügen."
                )
            session.addInput_(camera_input)

            video_output.setAlwaysDiscardsLateVideoFrames_(True)
            video_output.setVideoSettings_(
                {
                    Quartz.kCVPixelBufferPixelFormatTypeKey:
                        Quartz.kCVPixelFormatType_32BGRA
                }
            )
            video_output.setSampleBufferDelegate_queue_(
                delegate,
                callback_queue,
            )

            if not session.canAddOutput_(video_output):
                raise RuntimeError(
                    "AVCaptureSession kann Video-Output "
                    "nicht hinzufügen."
                )
            session.addOutput_(video_output)

            _configure_device(
                device,
                fmt,
                self.fps,
            )

            _acquire_device_lock(device)
            device_lock_held = True

        finally:
            try:
                session.commitConfiguration()
            except Exception:
                if device_lock_held:
                    try:
                        device.unlockForConfiguration()
                    except Exception:
                        pass
                    device_lock_held = False
                raise

        try:
            session.startRunning()

            active_dimensions = _format_dimensions(
                _safe_call(device, "activeFormat")
            )
            if active_dimensions != (
                self.width,
                self.height,
            ):
                raise RuntimeError(
                    "AVFoundation hat nach startRunning() "
                    "ein anderes aktives Format gewählt: "
                    f"{active_dimensions}, erwartet "
                    f"{self.width}x{self.height}."
                )

        except Exception:
            try:
                if session.isRunning():
                    session.stopRunning()
            except Exception:
                pass

            try:
                video_output.setSampleBufferDelegate_queue_(
                    None,
                    None,
                )
            except Exception:
                pass

            delegate.owner = None

            if device_lock_held:
                try:
                    device.unlockForConfiguration()
                except Exception:
                    pass
                device_lock_held = False

            raise

        with self._condition:
            self._device = device
            self._session = session
            self._input = camera_input
            self._output = video_output
            self._delegate = delegate
            self._callback_queue = callback_queue
            self._selected_format = fmt
            self._device_lock_held = device_lock_held
            self._latest_frame = None
            self._latest_sequence = 0
            self._last_returned_sequence = 0
            self._last_error = None
            self._frame_requested = False
            self._conversion_in_progress = False
            self._is_open = True
            self._condition.notify_all()

    def _handle_sample_buffer(self, sample_buffer: Any) -> None:
        # Nur ein von read() angefordertes Frame wird konvertiert.
        # Alle anderen AVFoundation-Callbacks werden vor CoreImage,
        # PNG und OpenCV sofort verworfen.
        with self._condition:
            if not self._is_open:
                return

            if (
                not self._frame_requested
                or self._conversion_in_progress
            ):
                return

            self._conversion_in_progress = True

        try:
            # Der Callback läuft auf einer eigenen libdispatch-Queue.
            # Ohne expliziten Autorelease-Pool können temporäre
            # Objective-C-Objekte aus CoreImage/AppKit über viele
            # Frames hinweg im Thread hängen bleiben.
            with objc.autorelease_pool():
                pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(
                    sample_buffer
                )

                if pixel_buffer is None:
                    raise RuntimeError(
                        "CMSampleBuffer enthält keinen CVPixelBuffer."
                    )

                frame = _pixel_buffer_to_bgr(
                    pixel_buffer,
                    self._ci_context,
                    self._ci_render_buffer,
                    self._ci_color_space,
                )

                actual_height, actual_width = frame.shape[:2]
                if (
                    actual_width != self.width
                    or actual_height != self.height
                ):
                    raise RuntimeError(
                        "Falsche Framegröße: erwartet "
                        f"{self.width}x{self.height}, erhalten "
                        f"{actual_width}x{actual_height}."
                    )

        except Exception:
            with self._condition:
                self._conversion_in_progress = False
                self._frame_requested = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._conversion_in_progress = False

            if not self._is_open:
                self._frame_requested = False
                self._condition.notify_all()
                return

            self._latest_frame = frame
            self._latest_sequence += 1
            self._last_error = None
            self._frame_requested = False
            self._condition.notify_all()

    def _publish_callback_error(self, exc: Exception) -> None:
        with self._condition:
            self._last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self._conversion_in_progress = False
            self._frame_requested = False
            self._condition.notify_all()

    def read(
        self,
        timeout: float | None = None,
    ) -> tuple[bool, np.ndarray | None]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout darf nicht negativ sein.")

        deadline = (
            None
            if timeout is None
            else time.monotonic() + timeout
        )

        with self._condition:
            if not self._is_open:
                return False, None

            if (
                self._latest_sequence
                <= self._last_returned_sequence
            ):
                self._frame_requested = True

            while (
                self._latest_sequence
                <= self._last_returned_sequence
            ):
                if self._last_error is not None:
                    self._frame_requested = False
                    return False, None

                if timeout == 0:
                    self._frame_requested = False
                    return False, None

                remaining = (
                    None
                    if deadline is None
                    else deadline - time.monotonic()
                )

                if (
                    remaining is not None
                    and remaining <= 0
                ):
                    self._frame_requested = False
                    return False, None

                self._condition.wait(remaining)

                if not self._is_open:
                    self._frame_requested = False
                    return False, None

            frame = self._latest_frame
            if frame is None:
                self._frame_requested = False
                return False, None

            self._last_returned_sequence = (
                self._latest_sequence
            )

            return True, frame

    def release(self) -> None:
        with self._condition:
            output = self._output
            delegate = self._delegate
            session = self._session
            device = self._device
            device_lock_held = self._device_lock_held

            self._is_open = False
            self._frame_requested = False
            self._conversion_in_progress = False
            self._condition.notify_all()

        if output is not None:
            try:
                output.setSampleBufferDelegate_queue_(
                    None,
                    None,
                )
            except Exception:
                pass

        if delegate is not None:
            try:
                delegate.owner = None
            except Exception:
                pass

        if session is not None:
            try:
                if session.isRunning():
                    session.stopRunning()
            except Exception:
                pass

        if device is not None and device_lock_held:
            try:
                device.unlockForConfiguration()
            except Exception:
                pass

        with self._condition:
            self._device = None
            self._session = None
            self._input = None
            self._output = None
            self._delegate = None
            self._callback_queue = None
            self._selected_format = None
            self._device_lock_held = False
            self._latest_frame = None
            self._latest_sequence = 0
            self._last_returned_sequence = 0
            self._frame_requested = False
            self._conversion_in_progress = False
            self._condition.notify_all()

    def __enter__(self) -> "AVFoundationCapture":
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.release()
