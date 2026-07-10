import yaml
import math
from pathlib import Path

BASE = Path.home() / "anthro3d"

ROLE_NAMES = ["OV9281 L", "OV9281 R", "ELP2", "ELP1"]

def load_yaml(path):
    p = Path(path)
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or {}

def print_camera_roles():
    cfg = load_yaml(BASE / "config.yaml")
    print("Kamera-Rollen aus config.yaml")
    print("=============================")
    if not cfg:
        print("FEHLER: config.yaml fehlt oder ist leer")
        return {}
    roles = {}
    for cam in cfg.get("cameras", {}).get("tracking", []):
        name = cam.get("name")
        idx = cam.get("device_index")
        enabled = cam.get("enabled")
        roles[name] = idx
        print(f"{name}: Index {idx} | enabled={enabled}")
    print("")
    missing = [name for name in ROLE_NAMES if name not in roles]
    if missing:
        print("WARNUNG: fehlende Rollen:", ", ".join(missing))
    else:
        print("Rollenstatus: OK")
    print("")
    return roles

def print_stereo_config(label, filename):
    d = load_yaml(BASE / filename)
    print(label)
    print("=" * len(label))
    if not d:
        print(f"FEHLT: {filename}")
        print("")
        return
    baseline = d.get("baseline_cm")
    method = (d.get("charuco_board") or {}).get("method")
    k_l = d.get("camera_matrix_l") or [[None]]
    k_r = d.get("camera_matrix_r") or [[None]]
    print(f"datei: {filename}")
    print(f"baseline_cm: {baseline}")
    print(f"method: {method}")
    print(f"fx_l: {k_l[0][0]}")
    print(f"fx_r: {k_r[0][0]}")
    print("")

def print_elp2_scale():
    d = load_yaml(BASE / "aruco_state" / "elp2_id3_id30_board_calibration.yaml")
    print("ELP2 ID3/ID30 lokale Skalenkorrektur")
    print("====================================")
    if not d:
        print("FEHLT: aruco_state/elp2_id3_id30_board_calibration.yaml")
        print("Status: NICHT VERFUEGBAR")
        print("")
        return

    real_dist = float(d.get("center_distance_cm_real"))
    measured_dist = float(d.get("distance_cm_median"))
    dist_mad = float(d.get("distance_cm_mad"))
    real_marker = float(d.get("marker_size_cm_real"))
    top_marker = float(d.get("top_marker_size_cm_median"))
    bottom_marker = float(d.get("bottom_marker_size_cm_median"))
    scale_distance = float(d.get("scale_from_distance"))
    scale_top = float(d.get("scale_from_top_marker"))
    scale_bottom = float(d.get("scale_from_bottom_marker"))
    scale_median = float(d.get("scale_median"))
    samples = int(d.get("samples"))
    z_cm = float(d.get("z_cm_median"))

    corrected_by_distance = measured_dist * scale_distance
    corrected_by_median = measured_dist * scale_median
    error_median = corrected_by_median - real_dist

    print(f"samples: {samples}")
    print(f"z_median_cm: {z_cm:.1f}")
    print(f"real_distance_cm: {real_dist:.2f}")
    print(f"measured_distance_cm: {measured_dist:.2f}")
    print(f"distance_mad_cm: {dist_mad:.2f}")
    print(f"corrected_distance_by_distance_scale_cm: {corrected_by_distance:.2f}")
    print(f"corrected_distance_by_median_scale_cm: {corrected_by_median:.2f}")
    print(f"median_scale_error_cm: {error_median:.2f}")
    print(f"real_marker_cm: {real_marker:.2f}")
    print(f"top_marker_measured_cm: {top_marker:.2f}")
    print(f"bottom_marker_measured_cm: {bottom_marker:.2f}")
    print(f"scale_from_distance: {scale_distance:.4f}")
    print(f"scale_from_top_marker: {scale_top:.4f}")
    print(f"scale_from_bottom_marker: {scale_bottom:.4f}")
    print(f"scale_median: {scale_median:.4f}")

    spread = max(scale_distance, scale_top, scale_bottom) - min(scale_distance, scale_top, scale_bottom)
    ok = samples >= 60 and dist_mad <= 0.5 and spread <= 0.04 and abs(error_median) <= 1.0

    print(f"scale_spread: {spread:.4f}")
    print(f"Status: {'OK' if ok else 'PRUEFEN'}")
    print("Hinweis: Diese Korrektur gilt lokal fuer ELP2 im ID3/ID30-Bereich, nicht global.")
    print("")

def main():
    print("")
    print("ANTHRO3D Rollen- und Skalen-Diagnose")
    print("Diese Diagnose veraendert keine Kalibrierdateien.")
    print("")
    print_camera_roles()
    print_stereo_config("ELP2 stereo_config", "stereo_config.yaml")
    print_stereo_config("ELP1 stereo_config", "stereo_config_elp1.yaml")
    print_stereo_config("OV9281 stereo_config", "stereo_config_ov9281.yaml")
    print_elp2_scale()
    print("Fertig.")

if __name__ == "__main__":
    main()
