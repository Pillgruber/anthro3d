"""Isolated AnthroPrecis acquisition check using the installed Galaxy SDK."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-python', type=Path, default=Path(os.environ.get(
        'GALAXY_SDK_PYTHON', r'C:\Program Files\Daheng Imaging\GalaxySDK\Development\Samples\Python')))
    parser.add_argument('--serial', help='Select one camera by serial number')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--timeout-ms', type=int, default=2000)
    args = parser.parse_args()
    if args.frames < 2 or not 1 <= args.timeout_ms <= 60000:
        parser.error('frames must be >= 2; timeout-ms must be between 1 and 60000')

    camera = None
    streaming = False
    restore = []
    rows = []
    errors = []
    report = {'requested_frames': args.frames, 'started_utc': datetime.now(timezone.utc).isoformat()}
    started = None
    elapsed = 0.0
    try:
        if not (args.sdk_python / 'gxipy' / '__init__.py').is_file():
            raise RuntimeError(f'gxipy not found in SDK directory: {args.sdk_python}')
        sys.path.insert(0, str(args.sdk_python))
        import numpy
        if numpy.__version__ != '1.26.4':
            raise RuntimeError(f'NumPy {numpy.__version__} installed; use numpy==1.26.4')
        import gxipy as gx
        report['gxipy_path'] = gx.__file__
        manager = gx.DeviceManager()
        count, devices = manager.update_all_device_list()
        print(f'Daheng cameras found: {count}')
        for device in devices:
            print(f"  {device['model_name']} | serial: {device['sn']}")
        if not count:
            raise RuntimeError('No camera found. Check USB3 cable, power and Galaxy driver.')
        if args.serial:
            selected = next((d for d in devices if d['sn'] == args.serial), None)
            if selected is None:
                raise RuntimeError(f'Camera serial {args.serial} not found')
        elif count == 1:
            selected = devices[0]
        else:
            raise RuntimeError('Multiple cameras found; select exactly one with --serial.')
        camera = manager.open_device_by_sn(selected['sn'])
        control = camera.get_remote_device_feature_control()
        report.update(model=selected['model_name'], serial=selected['sn'],
                      width=control.get_int_feature('Width').get(),
                      height=control.get_int_feature('Height').get())
        print(f"Selected: {report['model']} | {report['serial']} | {report['width']} x {report['height']}")
        # Change only acquisition mode; preserve image size, exposure, gain and pixel format.
        for name, value in [('AcquisitionMode', 'Continuous'), ('TriggerMode', 'Off')]:
            feature = control.get_enum_feature(name)
            old = feature.get()[1]
            if old != value:
                restore.append((feature, old))
                feature.set(value)
        started = time.perf_counter()
        streaming = True  # Attempt stream_off even if stream_on partially fails.
        camera.stream_on()
        stream = camera.data_stream[0]
        for index in range(args.frames):
            image = stream.dq_buf(args.timeout_ms)
            received = time.perf_counter()
            if image is None:
                raise RuntimeError(f'Timeout waiting for frame {index + 1}')
            try:
                frame = image.frame_data
                rows.append({'frame_id': int(frame.frame_id), 'status': int(frame.status),
                             'host_elapsed_s': received - started,
                             'camera_timestamp_raw': int(frame.timestamp),
                             'width': int(frame.width), 'height': int(frame.height)})
                if frame.status != gx.GxFrameStatusList.SUCCESS:
                    errors.append(f'Incomplete frame {frame.frame_id}: status {frame.status}')
            finally:
                stream.q_buf(image)
        elapsed = time.perf_counter() - started
    except KeyboardInterrupt:
        errors.append('Acquisition interrupted by user')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
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

    gaps = []
    for previous, current in zip(rows, rows[1:]):
        delta = current['frame_id'] - previous['frame_id']
        if delta != 1:
            gaps.append({'previous': previous['frame_id'], 'current': current['frame_id'],
                         'missing': max(0, delta - 1)})
    if gaps:
        errors.append('Frame ID gaps, duplicates or backward jumps detected')
    valid = sum(row['status'] == 0 for row in rows)
    span = rows[-1]['host_elapsed_s'] - rows[0]['host_elapsed_s'] if len(rows) > 1 else 0
    report.update(received_frames=len(rows), valid_frames=valid, elapsed_s=elapsed,
                  valid_frames_per_elapsed_s=valid / elapsed if elapsed else None,
                  arrival_fps=(len(rows) - 1) / span if span else None,
                  missing_frame_ids=sum(g['missing'] for g in gaps),
                  frame_id_anomalies=gaps, errors=errors,
                  passed=not errors and valid == args.frames)
    output = Path(__file__).resolve().parent / 'daheng_test_results'
    output.mkdir(exist_ok=True)
    run = output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir()  # Unique run directory; never overwrite earlier measurements.
    with (run / 'frames.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame_id', 'status', 'host_elapsed_s',
                                                   'camera_timestamp_raw', 'width', 'height'])
        writer.writeheader()
        writer.writerows(rows)
    with (run / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Results: {run}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
