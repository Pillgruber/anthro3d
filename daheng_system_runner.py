"""Run a Daheng camera pair by logical ANTHRO3D system ID.

Examples:
  python daheng_system_runner.py --system 1 --mode preview
  python daheng_system_runner.py --system 1 --mode capture
  python daheng_system_runner.py --system 1 --mode capture --countdown-seconds 3

Camera IDs are logical software identifiers, not ArUco board/marker IDs.
The preview does not adjust exposure; capture automatically adjusts and locks
physical exposure/gain before the photo. Focus and geometry are unchanged.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="1", help="Logical camera-system number")
    parser.add_argument("--mode", choices=("preview", "capture"), default="preview")
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    with (root / "daheng_systems.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    system = config.get("systems", {}).get(str(args.system))
    if system is None:
        parser.error(f"Unknown camera system: {args.system}")
    cameras = system.get("cameras", [])
    if len(cameras) != 2:
        parser.error("Current Daheng tools require exactly two cameras per system")
    ids = [c.get("id") for c in cameras]
    serials = [c.get("serial") for c in cameras]
    if (len(set(ids)) != 2 or len(set(serials)) != 2 or
            not all(isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in serials)):
        parser.error("Camera IDs and serial numbers must be present and unique")
    script = "daheng_live_preview.py" if args.mode == "preview" else "daheng_countdown_capture.py"
    print(f"ANTHRO3D {system.get('name', args.system)}:")
    for camera in cameras:
        print(f"  Camera ID {camera['id']} -> serial {camera['serial']}")
    print(f"Mode: {args.mode} (hardware synchronization: none)", flush=True)
    return subprocess.run([sys.executable, str(root / script), "--serials", *serials, *extra],
                          cwd=str(root), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
