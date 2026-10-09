"""AnthroPrecis: two-camera live preview and lossless countdown snapshot.

Uses the same Galaxy SDK environment and shared-memory preview helpers as
daheng_live_preview.py. Press SPACE for a countdown, then save the next valid
native-resolution frame from each independently free-running camera.
This is NOT exposure-synchronized stereo acquisition.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import sys
import time

from daheng_live_preview import make_panel, new_slot, render_window


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def optional_float(control, name):
    try:
        return float(control.get_float_feature(name).get())
    except Exception:
        return None


def camera_worker(serial, options, slot, zoom, start_gate, capture_gate,
                  capture_at, stop, messages, run_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    camera = None
    streaming = False
    restore = []
    errors = []
    report = {'serial': serial, 'valid_frames': 0, 'preview_updates': 0,
              'missing_frame_ids': 0, 'frame_id_anomaly_count': 0}
    previous_id = None
    next_preview = 0.0
    snapshot = None
    try:
        import numpy as np
        import cv2 as cv
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy 1.26.4; found {np.__version__}')
        sdk_path = Path(options['sdk_python'])
        if not (sdk_path / 'gxipy' / '__init__.py').is_file():
            raise RuntimeError(f'gxipy not found under {sdk_path}')
        sys.path.insert(0, str(sdk_path))
        import gxipy as gx
        manager = gx.DeviceManager()
        _, devices = manager.update_all_device_list()
        matches = [device for device in devices if device['sn'] == serial]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one camera with serial {serial}; found {len(matches)}')
        camera = manager.open_device_by_sn(serial)
        control = camera.get_remote_device_feature_control()
        fmt = str(control.get_enum_feature('PixelFormat').get()[1])
        if not re.fullmatch(r'Mono(8|10|12|14|16)', fmt, re.IGNORECASE):
            raise RuntimeError(f'Unsupported lossless image format {fmt}; use unpacked Mono8/10/12/14/16')
        report.update(model=matches[0]['model_name'],
                      width=int(control.get_int_feature('Width').get()),
                      height=int(control.get_int_feature('Height').get()),
                      pixel_format=fmt,
                      exposure_time_us=optional_float(control, 'ExposureTime'),
                      gain=optional_float(control, 'Gain'),
                      configured_fps=optional_float(control, 'AcquisitionFrameRate'))
        for name, value in [('AcquisitionMode', 'Continuous'), ('TriggerMode', 'Off')]:
            feature = control.get_enum_feature(name)
            old_value = feature.get()[1]
            if old_value != value:
                restore.append((feature, old_value))
                feature.set(value)
        messages.put({'kind': 'ready', 'serial': serial, 'report': dict(report)})
        if not start_gate.wait(options['setup_timeout_s'] + 5):
            raise RuntimeError('Timed out waiting for software start')
        if stop.is_set():
            return
        streaming = True
        camera.stream_on()
        stream = camera.data_stream[0]
        import numpy as np
        pixels = np.frombuffer(slot['pixels'], dtype=np.uint8).reshape(
            options['panel_height'], options['panel_width'])
        while not stop.is_set():
            image = stream.dq_buf(options['timeout_ms'])
            received = time.perf_counter()
            received_utc = utc_now()
            if image is None:
                if stop.is_set():
                    break
                raise RuntimeError('No frame received before timeout')
            owned = None
            frame_info = None
            try:
                frame = image.frame_data
                if int(frame.status) != int(gx.GxFrameStatusList.SUCCESS):
                    raise RuntimeError(f'Incomplete frame: {frame.frame_id} status {frame.status}')
                if (int(frame.width), int(frame.height)) != (report['width'], report['height']):
                    raise RuntimeError('Frame dimensions changed')
                frame_id = int(frame.frame_id)
                if previous_id is not None and frame_id != previous_id + 1:
                    report['frame_id_anomaly_count'] += 1
                    report['missing_frame_ids'] += max(0, frame_id - previous_id - 1)
                previous_id = frame_id
                report['valid_frames'] += 1
                need_capture = capture_gate.is_set() and received >= capture_at.value
                need_preview = received >= next_preview
                if need_capture or need_preview:
                    array = image.get_numpy_array()
                    if array is None:
                        raise RuntimeError('SDK returned no pixel array')
                    array = np.asarray(array)
                    expect_dtype = np.uint8 if fmt.lower() == 'mono8' else np.uint16
                    if array.shape != (report['height'], report['width']) or array.dtype != expect_dtype:
                        raise RuntimeError(f'Unexpected frame shape/dtype: {array.shape}, {array.dtype}')
                    if need_capture:
                        # Copy before returning SDK-owned frame; 16-bit data stay 16-bit.
                        owned = array.copy()
                        frame_info = {
                            'serial': serial, 'filename': f'camera_{serial}.png',
                            'frame_id': frame_id,
                            'camera_timestamp_raw': int(frame.timestamp),
                            'host_received_perf_s': received,
                            'host_received_utc': received_utc,
                            'width': report['width'], 'height': report['height'],
                            'pixel_format': fmt, 'dtype': str(owned.dtype),
                            'exposure_time_us': report['exposure_time_us'],
                            'gain': report['gain'],
                            'configured_fps': report['configured_fps'],
                        }
                    if need_preview:
                        preview = make_panel(array, fmt, options['panel_width'],
                                             options['panel_height'], bool(zoom.value), np, cv)
                        with slot['lock']:
                            pixels[:] = preview
                            slot['updates'].value += 1
                            report['preview_updates'] += 1
                        next_preview = received + 1.0 / options['preview_fps']
                with slot['lock']:
                    slot['valid_frames'].value = report['valid_frames']
                    slot['frame_id'].value = frame_id
                    slot['last_receive'].value = received
                    slot['anomalies'].value = report['frame_id_anomaly_count']
            finally:
                stream.q_buf(image)
            if owned is not None:
                target = Path(run_path) / frame_info['filename']
                if target.exists():
                    raise RuntimeError(f'Refusing to overwrite {target}')
                if not cv.imwrite(str(target), owned, [cv.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f'Could not save {target}')
                frame_info['file_bytes'] = target.stat().st_size
                snapshot = frame_info
                messages.put({'kind': 'saved', 'serial': serial, 'report': frame_info})
                break
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        stop.set()
    finally:
        if camera is not None:
            if streaming:
                try:
                    camera.stream_off()
                except Exception as exc:
                    errors.append(f'Stopping acquisition failed: {exc}')
            for feature, previous in reversed(restore):
                try:
                    feature.set(previous)
                except Exception as exc:
                    errors.append(f'Restoring {feature} failed: {exc}')
            try:
                camera.close_device()
            except Exception as exc:
                errors.append(f'Closing camera failed: {exc}')
        if report['frame_id_anomaly_count']:
            errors.append('Frame-ID gaps, duplicates, or reversed IDs seen during run')
        report.update(saved=snapshot is not None, snapshot=snapshot, errors=errors,
                      passed=snapshot is not None and not errors)
        if errors:
            stop.set()
        messages.put({'kind': 'done', 'serial': serial, 'report': report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serials', nargs=2, required=True)
    parser.add_argument('--countdown-seconds', type=int, default=5)
    parser.add_argument('--auto', action='store_true',
                        help='Start countdown automatically once both previews have images')
    parser.add_argument('--panel-width', type=int, default=640)
    parser.add_argument('--panel-height', type=int, default=375)
    parser.add_argument('--preview-fps', type=float, default=15)
    parser.add_argument('--timeout-ms', type=int, default=2000)
    parser.add_argument('--setup-timeout-s', type=float, default=45)
    parser.add_argument('--capture-timeout-s', type=float, default=10)
    parser.add_argument('--sdk-python', type=Path, default=Path(os.environ.get(
        'GALAXY_SDK_PYTHON',
        r'C:\Program Files\Daheng Imaging\GalaxySDK\Development\Samples\Python')))
    args = parser.parse_args()
    if len(set(args.serials)) != 2 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in args.serials):
        parser.error('Supply two distinct camera serial numbers')
    if not 1 <= args.countdown_seconds <= 30:
        parser.error('Countdown must be 1..30 seconds')
    if not 160 <= args.panel_width <= 2048 or not 120 <= args.panel_height <= 1200:
        parser.error('Panel dimensions out of range')
    if not (math.isfinite(args.preview_fps) and 1 <= args.preview_fps <= 60):
        parser.error('Preview FPS must be 1..60')
    if not 1 <= args.timeout_ms <= 10000:
        parser.error('Timeout must be 1..10000 ms')
    if not (math.isfinite(args.setup_timeout_s) and args.setup_timeout_s > 0 and
            math.isfinite(args.capture_timeout_s) and args.capture_timeout_s > 0):
        parser.error('Timeouts must be positive')
    try:
        import numpy as np
        import cv2 as cv
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'Expected NumPy 1.26.4; found {np.__version__}')
    except (ImportError, RuntimeError) as exc:
        print(f'Dependencies not ready: {exc}; install requirements-daheng-preview.txt')
        return 1
    output = Path(__file__).resolve().parent / 'daheng_test_results'
    output.mkdir(exist_ok=True)
    run = output / ('snapshot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    run.mkdir()
    options = vars(args).copy()
    options['sdk_python'] = str(args.sdk_python.resolve())
    ctx = mp.get_context('spawn')
    slots = {s: new_slot(ctx, args.panel_width, args.panel_height) for s in args.serials}
    zoom = ctx.Value('i', 0)
    start_gate, capture_gate, stop = ctx.Event(), ctx.Event(), ctx.Event()
    capture_at = ctx.Value('d', float('inf'))
    messages = ctx.Queue()
    workers = {}
    ready = {}
    saved = {}
    reports = {}
    errors = []
    setup_deadline = time.perf_counter() + args.setup_timeout_s
    started = False
    countdown_started = None
    request_at = None
    requested_utc = None
    window = 'AnthroPrecis - Countdown Aufnahme'
    window_created = False

    def handle(message):
        serial = message['serial']
        kind = message['kind']
        if kind == 'ready':
            ready[serial] = message['report']
            data = ready[serial]
            print(f"Ready: {serial} {data['width']}x{data['height']} {data['pixel_format']}", flush=True)
        elif kind == 'saved':
            saved[serial] = message['report']
            print(f"Saved: {serial} frame {saved[serial]['frame_id']}", flush=True)
        elif kind == 'done':
            reports[serial] = message['report']
            for error in reports[serial]['errors']:
                print(f'{serial}: {error}', flush=True)

    try:
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        window_created = True
        cv.resizeWindow(window, 2 * args.panel_width, args.panel_height + 160)
        for serial in args.serials:
            worker = ctx.Process(target=camera_worker,
                                 args=(serial, options, slots[serial], zoom,
                                       start_gate, capture_gate, capture_at,
                                       stop, messages, str(run)))
            worker.start()
            workers[serial] = worker
        print('Both cameras warming up. SPACE: countdown/photo; Z: zoom; ESC/Q: cancel.')
        while not stop.is_set():
            while True:
                try:
                    handle(messages.get_nowait())
                except queue.Empty:
                    break
            now = time.perf_counter()
            if not started:
                if len(ready) == 2:
                    started = True
                    start_gate.set()
                    print('Live preview running; press SPACE when ready.')
                elif now > setup_deadline:
                    raise RuntimeError('Camera setup timeout')
            if any(worker.exitcode is not None and s not in saved for s, worker in workers.items()):
                raise RuntimeError('A camera stopped before saving its image')
            ready_for_countdown = (started and all(slots[s]['updates'].value > 0 for s in args.serials))
            if args.auto and ready_for_countdown and countdown_started is None:
                countdown_started = now
                print(f'Starting {args.countdown_seconds}-second countdown.')
            if countdown_started is not None and request_at is None:
                if now - countdown_started >= args.countdown_seconds:
                    # Common request; cameras remain free-running, not hardware synchronized.
                    with capture_at.get_lock():
                        capture_at.value = time.perf_counter()
                    request_at = capture_at.value
                    requested_utc = utc_now()
                    capture_gate.set()
                    print('PHOTO request sent to both cameras.')
            if request_at is not None:
                if len(saved) == 2:
                    break
                if now - request_at > args.capture_timeout_s:
                    raise RuntimeError('Snapshot timeout: did not save both images')
            canvas = render_window(args.serials, ready, slots, options, bool(zoom.value), np, cv)
            canvas = cv.copyMakeBorder(canvas, 0, 50, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
            if request_at is not None:
                status = 'Aufnahme laeuft - Bilder speichern...'
            elif countdown_started is not None:
                remaining = max(1, math.ceil(args.countdown_seconds - (now - countdown_started)))
                status = f'FOTO IN {remaining} ...'
            else:
                status = 'LEERTASTE: Countdown + Foto | Z: Zoom | ESC/Q: Abbrechen'
            cv.putText(canvas, status, (12, args.panel_height + 145),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
            cv.imshow(window, canvas)
            key = cv.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print('Cancelled before capture.')
                break
            if key == ord(' ') and ready_for_countdown and countdown_started is None:
                countdown_started = time.perf_counter()
                print(f'Starting {args.countdown_seconds}-second countdown.')
            if key in (ord('z'), ord('Z')):
                zoom.value = 1 - zoom.value
            try:
                if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                    break
            except cv.error:
                break
    except KeyboardInterrupt:
        errors.append('Interrupted')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        start_gate.set()
        deadline = time.perf_counter() + args.timeout_ms / 1000 + 10
        while any(w.is_alive() for w in workers.values()) and time.perf_counter() < deadline:
            try:
                handle(messages.get(timeout=0.2))
            except queue.Empty:
                pass
        for serial, worker in workers.items():
            if worker.is_alive():
                errors.append(f'Forced stop for {serial}; settings restoration unconfirmed')
                worker.terminate()
            worker.join(timeout=2)
            if worker.exitcode not in (None, 0):
                errors.append(f'Worker {serial} exited with code {worker.exitcode}')
        while True:
            try:
                handle(messages.get_nowait())
            except queue.Empty:
                break
        messages.close()
        if window_created:
            try:
                cv.destroyAllWindows()
            except Exception as exc:
                errors.append(f'Window cleanup failed: {exc}')
    for serial in args.serials:
        if serial not in saved:
            errors.append(f'No saved image for camera {serial}')
        elif not reports.get(serial, {}).get('passed', False):
            errors.append(f'Camera {serial} completed with acquisition/cleanup errors')
    host_skew_ms = None
    if len(saved) == 2:
        host_skew_ms = abs(saved[args.serials[0]]['host_received_perf_s'] -
                           saved[args.serials[1]]['host_received_perf_s']) * 1000
    result = {
        'started_utc': requested_utc, 'finished_utc': utc_now(),
        'request_host_perf_s': request_at, 'serials': args.serials,
        'frames': saved, 'cameras': reports, 'errors': errors,
        'host_frame_receive_gap_ms': host_skew_ms,
        'host_gap_is_not_exposure_offset': True,
        'hardware_synchronized': False, 'software_request_after_countdown': True,
        'passed': not errors and len(saved) == 2,
    }
    with (run / 'capture.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Output directory: {run}')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
