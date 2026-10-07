r"""AnthroPrecis: concurrent Galaxy SDK acquisition probe for exactly two cameras.

Run beside daheng_camera_test.py with the same Python environment:
  .\.venv\Scripts\python.exe daheng_dual_camera_test.py --serials SN1 SN2

Defaults: 3000 frames per camera, current camera settings, max. 120 seconds.
Only AcquisitionMode and TriggerMode are changed temporarily and restored.
No image payload is saved. This measures concurrent host reception, not
synchronized exposures or a guaranteed sensor frame rate. Camera timestamps
are recorded as raw SDK values without assuming units or a shared clock.
Results: daheng_test_results/dual_<UTC>/summary.json and <serial>_frames.csv.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import sys
import time


FIELDS = ['frame_id', 'status', 'host_perf_s', 'camera_timestamp_raw',
          'width', 'height']


def summarize(rows, report, errors, requested, elapsed, success_status):
    anomalies = []
    for previous, current in zip(rows, rows[1:]):
        delta = current['frame_id'] - previous['frame_id']
        if delta != 1:
            anomalies.append({'previous': previous['frame_id'],
                              'current': current['frame_id'],
                              'missing': max(0, delta - 1)})
    if anomalies:
        errors.append('Frame ID gaps, duplicates or backward jumps detected')
    valid = sum(row['status'] == success_status for row in rows)
    span = rows[-1]['host_perf_s'] - rows[0]['host_perf_s'] if len(rows) > 1 else 0
    report.update(requested_frames=requested, received_frames=len(rows),
                  valid_frames=valid, elapsed_s=elapsed,
                  valid_frames_per_elapsed_s=valid / elapsed if elapsed else None,
                  arrival_fps=(len(rows) - 1) / span if span else None,
                  first_host_perf_s=rows[0]['host_perf_s'] if rows else None,
                  last_host_perf_s=rows[-1]['host_perf_s'] if rows else None,
                  missing_frame_ids=sum(a['missing'] for a in anomalies),
                  frame_id_anomalies=anomalies, errors=errors,
                  passed=not errors and valid == requested)


def capture(serial, options, run_path, start_gate, stop, messages):
    # Each camera owns its SDK instance in a separate process. Ctrl+C is
    # handled by the parent so workers can finish returning buffers/cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    camera = None
    manager = None
    streaming = False
    restore = []
    rows = []
    errors = []
    report = {'serial': serial}
    started = None
    elapsed = 0.0
    success_status = 0
    try:
        sdk_path = Path(options['sdk_python'])
        if not (sdk_path / 'gxipy' / '__init__.py').is_file():
            raise RuntimeError(f'gxipy not found in SDK directory: {sdk_path}')
        sys.path.insert(0, str(sdk_path))
        import numpy
        if numpy.__version__ != '1.26.4':
            raise RuntimeError(f'NumPy {numpy.__version__} installed; use numpy==1.26.4')
        import gxipy as gx
        success_status = int(gx.GxFrameStatusList.SUCCESS)
        report['gxipy_path'] = gx.__file__
        manager = gx.DeviceManager()
        _, devices = manager.update_all_device_list()
        matches = [device for device in devices if device['sn'] == serial]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one camera with serial {serial}; found {len(matches)}')
        camera = manager.open_device_by_sn(serial)
        control = camera.get_remote_device_feature_control()
        report.update(model=matches[0]['model_name'],
                      width=int(control.get_int_feature('Width').get()),
                      height=int(control.get_int_feature('Height').get()))
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
            raise RuntimeError('Test cancelled before acquisition')
        started = time.perf_counter()
        streaming = True  # Attempt stream_off even after partial stream_on failure.
        camera.stream_on()
        stream = camera.data_stream[0]
        for index in range(options['frames']):
            if stop.is_set():
                raise RuntimeError('Test stopped because of interruption or peer failure')
            if time.perf_counter() - started >= options['max_run_seconds']:
                raise RuntimeError('Maximum acquisition duration reached')
            image = stream.dq_buf(options['timeout_ms'])
            received = time.perf_counter()
            if image is None:
                raise RuntimeError(f'Timeout waiting for frame {index + 1}')
            try:
                frame = image.frame_data
                row = {'frame_id': int(frame.frame_id), 'status': int(frame.status),
                       'host_perf_s': received,
                       'camera_timestamp_raw': int(frame.timestamp),
                       'width': int(frame.width), 'height': int(frame.height)}
                rows.append(row)
                if row['status'] != success_status:
                    errors.append(f"Incomplete frame {row['frame_id']}: status {row['status']}")
                if (row['width'], row['height']) != (report['width'], report['height']):
                    errors.append(f"Unexpected dimensions on frame {row['frame_id']}")
            finally:
                stream.q_buf(image)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        stop.set()
    finally:
        if started is not None and not elapsed:
            elapsed = time.perf_counter() - started
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
        summarize(rows, report, errors, options['frames'], elapsed, success_status)
        # Write frame metadata only after acquisition to avoid disk I/O in
        # the dequeue/requeue loop. Never overwrite earlier measurements.
        try:
            filename = f'{serial}_frames.csv'
            with (Path(run_path) / filename).open('x', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            report['frames_csv'] = filename
        except Exception as exc:
            errors.append(f'Writing frame metadata failed: {exc}')
            report['passed'] = False
        if errors:
            stop.set()
        messages.put({'kind': 'done', 'serial': serial, 'report': report})


def add_overlap(reports, run):
    if len(reports) != 2 or any(r.get('first_host_perf_s') is None or
                                r.get('last_host_perf_s') is None for r in reports.values()):
        return {'overlap_s': 0.0, 'shorter_capture_overlap_fraction': 0.0,
                'sufficient_overlap': False}
    start = max(r['first_host_perf_s'] for r in reports.values())
    end = min(r['last_host_perf_s'] for r in reports.values())
    overlap_s = max(0.0, end - start)
    shorter_span = min(r['last_host_perf_s'] - r['first_host_perf_s']
                       for r in reports.values())
    fraction = overlap_s / shorter_span if shorter_span > 0 else 0.0
    for report in reports.values():
        with (run / report['frames_csv']).open(newline='', encoding='utf-8') as handle:
            shared_rows = [row for row in csv.DictReader(handle)
                           if start <= float(row['host_perf_s']) <= end]
        span = (float(shared_rows[-1]['host_perf_s']) - float(shared_rows[0]['host_perf_s'])
                if len(shared_rows) > 1 else 0.0)
        report['frames_in_overlap'] = len(shared_rows)
        report['arrival_fps_in_overlap'] = (len(shared_rows) - 1) / span if span else None
    return {'overlap_s': overlap_s, 'shorter_capture_overlap_fraction': fraction,
            'sufficient_overlap': overlap_s > 0 and fraction >= 0.90 and
            all(r['frames_in_overlap'] >= 2 for r in reports.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serials', nargs=2, required=True)
    parser.add_argument('--frames', type=int, default=3000)
    parser.add_argument('--timeout-ms', type=int, default=2000)
    parser.add_argument('--setup-timeout-s', type=float, default=45)
    parser.add_argument('--max-run-seconds', type=float, default=120)
    parser.add_argument('--sdk-python', type=Path, default=Path(os.environ.get(
        'GALAXY_SDK_PYTHON', r'C:\Program Files\Daheng Imaging\GalaxySDK\Development\Samples\Python')))
    args = parser.parse_args()
    if args.frames < 2 or not 1 <= args.timeout_ms <= 60000:
        parser.error('frames must be >= 2; timeout-ms must be between 1 and 60000')
    if args.setup_timeout_s <= 0 or args.max_run_seconds <= 0:
        parser.error('timeout durations must be positive')
    if len(set(args.serials)) != 2 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', sn)
                                         for sn in args.serials):
        parser.error('Choose two distinct serial numbers (letters, digits, underscore, hyphen)')
    output = Path(__file__).resolve().parent / 'daheng_test_results'
    output.mkdir(exist_ok=True)
    run = output / ('dual_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    run.mkdir()
    options = vars(args).copy()
    options['sdk_python'] = str(args.sdk_python.resolve())
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'requested_frames_per_camera': args.frames, 'serials': args.serials,
              'purpose': 'Concurrent USB acquisition probe; metadata only',
              'hardware_synchronized': False, 'fps_setting_changed': False}
    context = mp.get_context('spawn')
    start_gate, stop = context.Event(), context.Event()
    messages = context.Queue()
    workers, reports, ready, errors = {}, {}, set(), []
    deadline = time.perf_counter() + args.setup_timeout_s
    started = False
    shutdown_deadline = None

    def handle(message):
        serial = message['serial']
        if message['kind'] == 'ready':
            ready.add(serial)
            info = message['report']
            print(f"Ready: {serial} | {info['model']} | {info['width']} x {info['height']}", flush=True)
        elif message['kind'] == 'done':
            reports[serial] = message['report']
            print(f"Finished: {serial} | {reports[serial]['valid_frames']}/{args.frames} valid frames", flush=True)

    try:
        for serial in args.serials:
            worker = context.Process(target=capture, args=(serial, options, str(run),
                                                           start_gate, stop, messages))
            worker.start()
            workers[serial] = worker
        print(f'Preparing both cameras; {args.frames} frames each...', flush=True)
        while len(reports) < 2:
            try:
                handle(messages.get(timeout=0.2))
            except queue.Empty:
                pass
            now = time.perf_counter()
            for serial, worker in workers.items():
                if worker.exitcode is not None and serial not in reports:
                    # A worker can exit just after queue.get times out.
                    while True:
                        try:
                            handle(messages.get_nowait())
                        except queue.Empty:
                            break
                    if serial not in reports:
                        errors.append(f'Worker {serial} exited without a report (exit {worker.exitcode})')
                        reports[serial] = {'serial': serial, 'passed': False,
                                           'errors': ['Worker ended without a report']}
                        stop.set()
            if not started and len(ready) == 2 and not stop.is_set():
                started = True
                deadline = now + args.max_run_seconds + args.timeout_ms / 1000 + 5
                print('Starting both cameras together (software start). Please wait...', flush=True)
                start_gate.set()
            if now >= deadline and not stop.is_set():
                errors.append('Acquisition deadline exceeded' if started else 'Camera setup deadline exceeded')
                stop.set()
            if stop.is_set():
                start_gate.set()
                if shutdown_deadline is None:
                    shutdown_deadline = now + args.timeout_ms / 1000 + 10
                if now >= shutdown_deadline:
                    break
    except KeyboardInterrupt:
        errors.append('Test interrupted by user')
        stop.set()
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        stop.set()
    finally:
        start_gate.set()
        # Drain messages while workers clean up; do not join a producer
        # before draining its queue (large anomaly reports can fill pipes).
        cleanup_deadline = time.perf_counter() + args.timeout_ms / 1000 + 10
        while any(worker.is_alive() for worker in workers.values()) and time.perf_counter() < cleanup_deadline:
            try:
                handle(messages.get(timeout=0.2))
            except queue.Empty:
                pass
        for serial, worker in workers.items():
            if worker.is_alive():
                errors.append(f'Forced stop of unresponsive worker {serial}; camera-setting restoration unconfirmed')
                worker.terminate()
            worker.join(timeout=2)
            if worker.exitcode not in (None, 0):
                errors.append(f'Worker {serial} exit code: {worker.exitcode}')
        while True:
            try:
                handle(messages.get_nowait())
            except queue.Empty:
                break
        messages.close()
        for serial in args.serials:
            reports.setdefault(serial, {'serial': serial, 'passed': False,
                                        'errors': ['No final report received']})
        try:
            overlap = add_overlap(reports, run)
        except Exception as exc:
            errors.append(f'Overlap evaluation failed: {exc}')
            overlap = {'overlap_s': 0.0, 'sufficient_overlap': False}
        if not overlap['sufficient_overlap']:
            errors.append('Insufficient concurrent host reception (need >=90% of shorter capture and >=2 frames per camera)')
        report.update(cameras=reports, concurrent_reception=overlap, errors=errors,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      passed=not errors and all(r.get('passed', False) for r in reports.values()))
        with (run / 'summary.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Results: {run}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
