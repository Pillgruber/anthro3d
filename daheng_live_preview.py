r"""Two-camera AnthroPrecis live preview using the installed Galaxy SDK.

  .\.venv\Scripts\python.exe daheng_live_preview.py --serials SN1 SN2

ESC/Q or closing the window stops the preview; Z toggles a native 1:1
center crop for focus inspection. Acquisition settings are preserved except
temporary Continuous/Trigger Off modes, which are restored on normal/error
exit. No image recording, calibration or hardware synchronization is done.
The camera resolution and frame-rate settings are never changed. Preview
refresh is limited independently (default 15 FPS); RX FPS are host arrivals.
"""

import argparse
from collections import deque
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


def make_panel(array, pixel_format, width, height, zoom, np, cv):
    """Copy image pixels into an owned display panel while the SDK owns its buffer."""
    match = re.fullmatch(r'Mono(8|10|12|14|16)', str(pixel_format), re.IGNORECASE)
    if not match:
        raise RuntimeError(f'Preview requires unpacked Mono8/10/12/14/16; found {pixel_format}')
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise RuntimeError(f'Unsupported SDK image layout: shape={array.shape}, dtype={array.dtype}')
    bits = int(match.group(1))
    if bits > 8:
        if array.dtype != np.uint16:
            raise RuntimeError(f'{pixel_format} did not provide a uint16 image')
        array = np.right_shift(array, bits - 8).clip(0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise RuntimeError('Mono8 did not provide a uint8 image')
    native_h, native_w = array.shape
    panel = np.zeros((height, width), dtype=np.uint8)
    if zoom:
        crop_w, crop_h = min(width, native_w), min(height, native_h)
        x, y = (native_w - crop_w) // 2, (native_h - crop_h) // 2
        view = array[y:y + crop_h, x:x + crop_w]
    else:
        scale = min(width / native_w, height / native_h, 1.0)
        size = (max(1, int(native_w * scale)), max(1, int(native_h * scale)))
        view = cv.resize(array, size, interpolation=cv.INTER_AREA)
    h, w = view.shape
    panel[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w] = view
    return panel


def camera_worker(serial, options, slot, zoom, start_gate, stop, messages):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    camera = None
    manager = None
    streaming = False
    restore = []
    errors = []
    report = {'serial': serial, 'valid_frames': 0, 'missing_frame_ids': 0,
              'frame_id_anomaly_count': 0, 'preview_updates': 0}
    previous_id = None
    times = deque(maxlen=120)
    next_view = 0.0
    try:
        import numpy as np
        import cv2 as cv
        if np.__version__ != '1.26.4':
            raise RuntimeError(f'NumPy {np.__version__} installed; use numpy==1.26.4')
        sdk = Path(options['sdk_python'])
        if not (sdk / 'gxipy' / '__init__.py').is_file():
            raise RuntimeError(f'gxipy not found in SDK directory: {sdk}')
        sys.path.insert(0, str(sdk))
        import gxipy as gx
        report['gxipy_path'] = gx.__file__
        manager = gx.DeviceManager()
        _, devices = manager.update_all_device_list()
        matches = [d for d in devices if d['sn'] == serial]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one camera {serial}; found {len(matches)}')
        camera = manager.open_device_by_sn(serial)
        control = camera.get_remote_device_feature_control()
        report.update(model=matches[0]['model_name'],
                      width=int(control.get_int_feature('Width').get()),
                      height=int(control.get_int_feature('Height').get()),
                      pixel_format=control.get_enum_feature('PixelFormat').get()[1])
        if not re.fullmatch(r'Mono(8|10|12|14|16)', str(report['pixel_format']), re.IGNORECASE):
            raise RuntimeError(f"Unsupported preview pixel format: {report['pixel_format']}; settings left unchanged")
        for name, value in [('AcquisitionMode', 'Continuous'), ('TriggerMode', 'Off')]:
            feature = control.get_enum_feature(name)
            old = feature.get()[1]
            if old != value:
                restore.append((feature, old))
                feature.set(value)
        messages.put({'kind': 'ready', 'serial': serial, 'report': dict(report)})
        if not start_gate.wait(options['setup_timeout_s'] + 5):
            raise RuntimeError('Timed out waiting for common software start')
        if stop.is_set():
            return
        streaming = True
        camera.stream_on()
        stream = camera.data_stream[0]
        target = np.frombuffer(slot['pixels'], dtype=np.uint8).reshape(
            options['panel_height'], options['panel_width'])
        while not stop.is_set():
            image = stream.dq_buf(options['timeout_ms'])
            received = time.perf_counter()
            if image is None:
                if stop.is_set():
                    break
                raise RuntimeError('Timeout waiting for image')
            try:
                frame = image.frame_data
                if int(frame.status) != int(gx.GxFrameStatusList.SUCCESS):
                    raise RuntimeError(f'Incomplete frame {frame.frame_id}: status {frame.status}')
                if (int(frame.width), int(frame.height)) != (report['width'], report['height']):
                    raise RuntimeError('Frame dimensions changed during preview')
                frame_id = int(frame.frame_id)
                if previous_id is not None and frame_id - previous_id != 1:
                    report['frame_id_anomaly_count'] += 1
                    report['missing_frame_ids'] += max(0, frame_id - previous_id - 1)
                previous_id = frame_id
                times.append(received)
                report['valid_frames'] += 1
                fps = (len(times)-1) / (times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
                panel = None
                if received >= next_view:
                    reader = getattr(image, 'get_numpy_array', None)
                    if not callable(reader):
                        raise RuntimeError('Installed SDK image has no get_numpy_array() method')
                    array = reader()
                    if array is None:
                        raise RuntimeError(f"SDK returned no NumPy image for {report['pixel_format']}")
                    if array.shape != (report['height'], report['width']):
                        raise RuntimeError(f'Unexpected SDK image shape: {array.shape}')
                    panel = make_panel(array, report['pixel_format'], options['panel_width'],
                                       options['panel_height'], bool(zoom.value), np, cv)
                    next_view = received + 1 / options['preview_fps']
                with slot['lock']:
                    slot['valid_frames'].value = report['valid_frames']
                    slot['frame_id'].value = frame_id
                    slot['arrival_fps'].value = fps
                    slot['last_receive'].value = received
                    slot['anomalies'].value = report['frame_id_anomaly_count']
                    if panel is not None:
                        target[:] = panel  # Copy finishes before q_buf returns borrowed SDK memory.
                        slot['updates'].value += 1
                        report['preview_updates'] += 1
            finally:
                stream.q_buf(image)
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
            for feature, old in reversed(restore):
                try:
                    feature.set(old)
                except Exception as exc:
                    errors.append(f'Restoring acquisition setting failed: {exc}')
            try:
                camera.close_device()
            except Exception as exc:
                errors.append(f'Closing camera failed: {exc}')
        if report['frame_id_anomaly_count']:
            errors.append('Frame ID gaps, duplicates or backward jumps detected during preview')
        if errors:
            stop.set()
        report.update(errors=errors, passed=not errors and report['preview_updates'] > 0)
        messages.put({'kind': 'done', 'serial': serial, 'report': report})


def new_slot(context, width, height):
    return {'pixels': context.RawArray('B', width*height), 'lock': context.Lock(),
            'valid_frames': context.Value('Q', 0, lock=False),
            'frame_id': context.Value('Q', 0, lock=False),
            'updates': context.Value('Q', 0, lock=False),
            'anomalies': context.Value('Q', 0, lock=False),
            'arrival_fps': context.Value('d', 0.0, lock=False),
            'last_receive': context.Value('d', 0.0, lock=False)}


def render_window(serials, info, slots, options, zoom, np, cv):
    width, height = options['panel_width'], options['panel_height']
    canvas = np.zeros((height+110, width*2, 3), dtype=np.uint8)
    for index, serial in enumerate(serials):
        slot = slots[serial]
        # A crashed worker must not freeze the GUI by leaving a lock held.
        if slot['lock'].acquire(timeout=0.05):
            try:
                pixels = np.frombuffer(slot['pixels'], dtype=np.uint8).reshape(height, width).copy()
                fps = slot['arrival_fps'].value
                frame_id, updates = slot['frame_id'].value, slot['updates'].value
                anomalies, last_receive = slot['anomalies'].value, slot['last_receive'].value
            finally:
                slot['lock'].release()
        else:
            pixels = np.zeros((height, width), dtype=np.uint8)
            fps = frame_id = updates = anomalies = last_receive = 0
        x = index*width
        canvas[70:70+height, x:x+width] = cv.cvtColor(pixels, cv.COLOR_GRAY2BGR)
        data = info.get(serial, {})
        native = f"{data.get('width', '?')} x {data.get('height', '?')}"
        cv.putText(canvas, serial, (x+10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1)
        line = f'{native} | RX {fps:.1f} FPS | Frame {frame_id} | Luecken {anomalies}'
        cv.putText(canvas, line, (x+10, 52), cv.FONT_HERSHEY_SIMPLEX, 0.43, (190, 210, 190), 1)
        if not updates:
            cv.putText(canvas, 'Warte auf Kamerabild...', (x+20, 110), cv.FONT_HERSHEY_SIMPLEX,
                       0.65, (0, 190, 250), 1)
        elif time.perf_counter() - last_receive > 1.0:
            cv.putText(canvas, 'KEIN AKTUELLES BILD', (x+20, 110), cv.FONT_HERSHEY_SIMPLEX,
                       0.65, (0, 0, 255), 2)
    mode = '1:1 Mitte' if zoom else 'Ganzes Bild'
    footer = f'ESC/Q: Beenden | Z: Zoom | {mode} | Vorschau bis {options["preview_fps"]:g} FPS | Freilauf'
    cv.putText(canvas, footer, (10, height+98), cv.FONT_HERSHEY_SIMPLEX, 0.47,
               (210, 210, 210), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serials', nargs=2, required=True)
    parser.add_argument('--panel-width', type=int, default=640)
    parser.add_argument('--panel-height', type=int, default=375)
    parser.add_argument('--preview-fps', type=float, default=15)
    parser.add_argument('--timeout-ms', type=int, default=1000)
    parser.add_argument('--setup-timeout-s', type=float, default=45)
    parser.add_argument('--duration-seconds', type=float, default=0)
    parser.add_argument('--headless', action='store_true', help='Process preview pixels without a window (requires a duration)')
    parser.add_argument('--sdk-python', type=Path, default=Path(os.environ.get(
        'GALAXY_SDK_PYTHON', r'C:\Program Files\Daheng Imaging\GalaxySDK\Development\Samples\Python')))
    args = parser.parse_args()
    if len(set(args.serials)) != 2 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', sn) for sn in args.serials):
        parser.error('Choose two distinct camera serials')
    if not (160 <= args.panel_width <= 2048 and 120 <= args.panel_height <= 1200):
        parser.error('Panel dimensions out of range')
    if not math.isfinite(args.preview_fps) or not 1 <= args.preview_fps <= 60:
        parser.error('preview-fps must be between 1 and 60')
    if not 1 <= args.timeout_ms <= 10000 or not math.isfinite(args.setup_timeout_s) or args.setup_timeout_s <= 0:
        parser.error('Invalid acquisition/setup timeout')
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 0 or (args.headless and args.duration_seconds <= 0):
        parser.error('Duration must be non-negative; headless requires a positive duration')
    try:
        import numpy as np
        import cv2 as cv
        if np.__version__ != '1.26.4':
            raise RuntimeError('Use numpy==1.26.4 in this project environment')
    except (ImportError, RuntimeError) as exc:
        print(f'Dependency error: {exc}\nInstall requirements-daheng-preview.txt with the project Python.')
        return 1
    options = vars(args).copy()
    options['sdk_python'] = str(args.sdk_python.resolve())
    context = mp.get_context('spawn')
    slots = {sn: new_slot(context, args.panel_width, args.panel_height) for sn in args.serials}
    zoom = context.Value('i', 0)
    start_gate, stop = context.Event(), context.Event()
    messages = context.Queue()
    workers, info, reports, errors = {}, {}, {}, []
    started = None
    setup_deadline = time.perf_counter() + args.setup_timeout_s
    window_created = False
    window_name = 'AnthroPrecis - Daheng Live'

    def handle(message):
        sn, data = message['serial'], message['report']
        if message['kind'] == 'ready':
            info[sn] = data
            print(f"Ready: {sn} | {data['width']} x {data['height']} | {data['pixel_format']}", flush=True)
        else:
            reports[sn] = data
            for error in data['errors']:
                print(f'{sn}: {error}', flush=True)

    try:
        if not args.headless:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            window_created = True
            cv.resizeWindow(window_name, args.panel_width*2, args.panel_height+110)
        for sn in args.serials:
            worker = context.Process(target=camera_worker, args=(sn, options, slots[sn], zoom,
                                                                start_gate, stop, messages))
            worker.start()
            workers[sn] = worker
        print('Preparing both cameras. ESC/Q: close; Z: 1:1 focus crop.', flush=True)
        while not stop.is_set():
            while True:
                try:
                    handle(messages.get_nowait())
                except queue.Empty:
                    break
            now = time.perf_counter()
            if started is None:
                if len(info) == 2:
                    started = now
                    start_gate.set()
                    print('Live preview started. RX FPS are host arrivals; preview refresh is separate.', flush=True)
                elif now >= setup_deadline:
                    raise RuntimeError('Camera setup timeout')
            elif args.duration_seconds and now - started >= args.duration_seconds:
                break
            if any(w.exitcode is not None for w in workers.values()):
                raise RuntimeError('Camera worker stopped unexpectedly')
            if not args.headless:
                cv.imshow(window_name, render_window(args.serials, info, slots, options, bool(zoom.value), np, cv))
                key = cv.waitKey(20) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key in (ord('z'), ord('Z')):
                    zoom.value = 1 - zoom.value
                try:
                    visible = cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)
                except cv.error:
                    # Some HighGUI backends remove a window immediately on X.
                    break
                if visible < 1:
                    break
            else:
                stop.wait(0.02)
    except KeyboardInterrupt:
        print('Closing preview...', flush=True)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        start_gate.set()
        deadline = time.perf_counter() + args.timeout_ms/1000 + 10
        while any(w.is_alive() for w in workers.values()) and time.perf_counter() < deadline:
            try:
                handle(messages.get(timeout=0.2))
            except queue.Empty:
                pass
        for sn, worker in workers.items():
            if worker.is_alive():
                errors.append(f'Forced stop of {sn}; settings restoration unconfirmed')
                worker.terminate()
            worker.join(timeout=2)
            if worker.exitcode not in (None, 0):
                errors.append(f'Worker {sn} exit code: {worker.exitcode}')
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
                errors.append(f'Closing preview window failed: {exc}')
    result = {'finished_utc': datetime.now(timezone.utc).isoformat(), 'cameras': reports,
              'errors': errors, 'hardware_synchronized': False,
              'passed': not errors and len(reports) == 2 and all(r['passed'] for r in reports.values())}
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
