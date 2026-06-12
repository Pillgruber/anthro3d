#!/usr/bin/env python3
"""
ANTHRO3D — Stereo Kalibrierung
================================
Kalibriert zwei OV9281 Kameras als Stereopaar.

Ablauf:
  1. Beide Kameras öffnen
  2. Schachbrett-Muster erkennen (live)
  3. 25 gute Bildpaare sammeln
  4. OpenCV stereoCalibrate()
  5. Rektifizierung berechnen
  6. Alles in stereo_config.yaml speichern

Verwendung:
  python3 stereo_calibrator.py

Benötigt:
  pip3 install opencv-python numpy pyyaml --break-system-packages

Schachbrett:
  A4 ausdrucken: 9x6 Felder, Feldgröße 25mm
  Download: https://calib.io/pages/camera-calibration-pattern-generator
"""

import cv2
import numpy as np
import yaml
import time
import os
from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
CAMERA_LEFT_IDX  = 1          # OV9281 Links  (USB Index)
CAMERA_RIGHT_IDX = 2          # OV9281 Rechts (USB Index)
RESOLUTION       = (1280, 720)
FPS              = 60

# Schachbrett
BOARD_W          = 9          # innere Ecken horizontal
BOARD_H          = 6          # innere Ecken vertikal
SQUARE_SIZE_MM   = 25.0       # Feldgröße in mm

# Kalibrierung
MIN_PAIRS        = 25         # Mindestanzahl guter Bildpaare
CAPTURE_DELAY    = 0.8        # Sekunden zwischen automatischen Captures
SAVE_PATH        = Path("~/anthro3d/stereo_config.yaml").expanduser()
PREVIEW_SCALE    = 0.6

# ── FARBEN ────────────────────────────────────────────────────────────────────
GREEN  = (80, 180, 80)
RED    = (60, 60, 200)
YELLOW = (40, 200, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)


class StereoCalibrator:

    def __init__(self):
        self.cap_l = None
        self.cap_r = None

        # Schachbrett 3D-Punkte (in mm)
        self.objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE_MM

        self.objpoints   = []
        self.imgpoints_l = []
        self.imgpoints_r = []

        self.last_capture = 0
        self.calibrated   = False
        self.result       = {}

        self.cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                         cv2.CALIB_CB_NORMALIZE_IMAGE +
                         cv2.CALIB_CB_FAST_CHECK)

    # ── KAMERAS ───────────────────────────────────────────────────────────────
    def connect(self):
        print("\nVerbinde Kameras...")
        self.cap_l = cv2.VideoCapture(CAMERA_LEFT_IDX)
        self.cap_r = cv2.VideoCapture(CAMERA_RIGHT_IDX)

        for cap, name, idx in [
            (self.cap_l, "Links",  CAMERA_LEFT_IDX),
            (self.cap_r, "Rechts", CAMERA_RIGHT_IDX)
        ]:
            if not cap.isOpened():
                print(f"  FEHLER: Kamera {name} (Index {idx}) nicht gefunden!")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS,          FPS)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print(f"  OK: Kamera {name} verbunden (Index {idx})")
        return True

    def disconnect(self):
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()

    def grab_frames(self):
        """Beide Frames so synchron wie möglich lesen (grab → retrieve)."""
        ok_l = self.cap_l.grab()
        ok_r = self.cap_r.grab()
        if not ok_l or not ok_r:
            return None, None
        _, fl = self.cap_l.retrieve()
        _, fr = self.cap_r.retrieve()
        return fl, fr

    # ── SCHACHBRETT ───────────────────────────────────────────────────────────
    def find_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_W, BOARD_H), self.cb_flags)
        if found:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        return found, corners

    def draw_overlay(self, frame, found, corners, label, count):
        out = frame.copy()
        h, w = out.shape[:2]
        if found and corners is not None:
            cv2.drawChessboardCorners(out, (BOARD_W, BOARD_H), corners, found)
        col = GREEN if found else RED
        cv2.rectangle(out, (0, 0), (w, 44), BLACK, -1)
        cv2.rectangle(out, (0, 0), (w, 44), col, 2)
        status = "ERKANNT ✓" if found else "Suche..."
        cv2.putText(out, f"{label}  {status}", (10, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        bar_w = int((count / MIN_PAIRS) * (w - 20))
        cv2.rectangle(out, (10, 26), (w - 10, 38), (40, 40, 40), -1)
        cv2.rectangle(out, (10, 26), (10 + bar_w, 38), GREEN, -1)
        cv2.putText(out, f"{count}/{MIN_PAIRS}", (w - 60, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)
        return out

    # ── KALIBRIERUNG ──────────────────────────────────────────────────────────
    def calibrate(self):
        print(f"\nKalibriere mit {len(self.objpoints)} Paaren...")
        h, w = RESOLUTION[1], RESOLUTION[0]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        print("  Kamera Links...")
        err_l, K_l, D_l, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, (w, h), None, None)
        print(f"    Fehler: {err_l:.3f}px")

        print("  Kamera Rechts...")
        err_r, K_r, D_r, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, (w, h), None, None)
        print(f"    Fehler: {err_r:.3f}px")

        print("  Stereo-Kalibrierung...")
        err_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_l, self.imgpoints_r,
            K_l, D_l, K_r, D_r,
            (w, h), criteria=crit,
            flags=cv2.CALIB_FIX_INTRINSIC)
        baseline_mm = float(np.linalg.norm(T))
        print(f"    Fehler: {err_s:.3f}px  Baseline: {baseline_mm/10:.1f}cm")

        print("  Rektifizierung...")
        R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, (w, h), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        map_l1, map_l2 = cv2.initUndistortRectifyMap(
            K_l, D_l, R_l, P_l, (w, h), cv2.CV_32FC1)
        map_r1, map_r2 = cv2.initUndistortRectifyMap(
            K_r, D_r, R_r, P_r, (w, h), cv2.CV_32FC1)

        self.result = {
            "error_stereo": float(err_s),
            "error_left":   float(err_l),
            "error_right":  float(err_r),
            "baseline_mm":  baseline_mm,
            "baseline_cm":  baseline_mm / 10,
            "image_size":   [w, h],
            "K_left":   K_l.tolist(), "D_left":  D_l.tolist(),
            "K_right":  K_r.tolist(), "D_right": D_r.tolist(),
            "R": R.tolist(), "T": T.tolist(),
            "E": E.tolist(), "F": F.tolist(),
            "R_left":  R_l.tolist(), "R_right": R_r.tolist(),
            "P_left":  P_l.tolist(), "P_right": P_r.tolist(),
            "Q": Q.tolist(),
            "roi_left": list(roi_l), "roi_right": list(roi_r),
            "_maps": (map_l1, map_l2, map_r1, map_r2),
        }
        self.calibrated = True
        return self.result

    def save(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        maps_path = SAVE_PATH.parent / "stereo_maps.npz"
        maps = self.result.pop("_maps")
        np.savez(str(maps_path),
                 map_l1=maps[0], map_l2=maps[1],
                 map_r1=maps[2], map_r2=maps[3])
        data = {
            "anthro3d_stereo_calibration": True,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_left_index":  CAMERA_LEFT_IDX,
            "camera_right_index": CAMERA_RIGHT_IDX,
            "maps_file": str(maps_path),
            **self.result
        }
        with open(SAVE_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"\n  Gespeichert: {SAVE_PATH}")
        print(f"  Maps:        {maps_path}")

    def load_maps(self):
        maps_path = SAVE_PATH.parent / "stereo_maps.npz"
        if not maps_path.exists():
            return None
        d = np.load(str(maps_path))
        return d["map_l1"], d["map_l2"], d["map_r1"], d["map_r2"]

    def preview_rectification(self, maps):
        map_l1, map_l2, map_r1, map_r2 = maps
        print("\nRektifizierungs-Vorschau (ESC zum Beenden)")
        print("Gelbe Linien müssen auf BEIDEN Seiten gleich hoch sein.")
        while True:
            fl, fr = self.grab_frames()
            if fl is None:
                break
            rl = cv2.remap(fl, map_l1, map_l2, cv2.INTER_LINEAR)
            rr = cv2.remap(fr, map_r1, map_r2, cv2.INTER_LINEAR)
            combined = np.hstack([rl, rr])
            combined = cv2.resize(combined, (
                int(combined.shape[1] * PREVIEW_SCALE),
                int(combined.shape[0] * PREVIEW_SCALE)))
            # Epipolar-Linien
            for y in range(0, combined.shape[0], 40):
                cv2.line(combined, (0, y), (combined.shape[1], y), YELLOW, 1)
            cv2.putText(combined, "REKTIFIZIERT — Linien = Epipolarlinien",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
            cv2.putText(combined, "ESC = Beenden",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            cv2.imshow("ANTHRO3D — Rektifizierung", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    # ── HAUPTSCHLEIFE ─────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "="*52)
        print("  ANTHRO3D — Stereo Kalibrierung")
        print("="*52)
        print(f"""
  Schachbrett:  {BOARD_W}x{BOARD_H} innere Ecken, {SQUARE_SIZE_MM:.0f}mm Felder
  Ziel:         {MIN_PAIRS} Bildpaare
  Kameras:      Links=Index {CAMERA_LEFT_IDX}, Rechts=Index {CAMERA_RIGHT_IDX}

  Anleitung:
    1. Schachbrett-A4 ausdrucken (plan, scharf)
    2. Vor BEIDE Kameras gleichzeitig halten
    3. Grüner Rahmen = erkannt → automatischer Capture
    4. Verschiedene Winkel, Abstände, Positionen verwenden
    5. Mitte des Bildes UND Ecken abdecken

  Tasten:
    SPACE = manueller Capture
    R     = Reset
    Q/ESC = Beenden
""")
        input("  ENTER zum Starten...")

        if not self.connect():
            print("\n  Kameras nicht verfügbar.")
            print("  Warte auf OV9281 Lieferung :)")
            return

        print(f"\n  Kalibrierung läuft...\n")

        while True:
            fl, fr = self.grab_frames()
            if fl is None:
                break

            found_l, corners_l = self.find_corners(fl)
            found_r, corners_r = self.find_corners(fr)
            both = found_l and found_r
            count = len(self.objpoints)

            disp_l = self.draw_overlay(fl, found_l, corners_l, "LINKS",  count)
            disp_r = self.draw_overlay(fr, found_r, corners_r, "RECHTS", count)

            # Auto-Capture
            now = time.time()
            if both and (now - self.last_capture) > CAPTURE_DELAY:
                self.objpoints.append(self.objp)
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)
                self.last_capture = now
                print(f"  [{count+1:02d}/{MIN_PAIRS}] Paar gespeichert ✓")
                # Flash
                cv2.rectangle(disp_l, (0,0), (disp_l.shape[1], disp_l.shape[0]), GREEN, 8)
                cv2.rectangle(disp_r, (0,0), (disp_r.shape[1], disp_r.shape[0]), GREEN, 8)

            if count >= MIN_PAIRS:
                break

            s = PREVIEW_SCALE
            w2 = int(RESOLUTION[0] * s)
            h2 = int(RESOLUTION[1] * s)
            dl = cv2.resize(disp_l, (w2, h2))
            dr = cv2.resize(disp_r, (w2, h2))
            combined = np.hstack([dl, dr])

            mid = combined.shape[1] // 2
            cv2.line(combined, (mid, 0), (mid, combined.shape[0]), WHITE, 1)

            hint = "BEIDE ERKANNT — halte ruhig halten" if both else "Schachbrett vor beide Kameras halten"
            cv2.putText(combined, hint,
                        (mid - 190, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        GREEN if both else YELLOW, 1)

            cv2.imshow("ANTHRO3D — Stereo Kalibrierung", combined)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("\n  Abgebrochen.")
                cv2.destroyAllWindows()
                self.disconnect()
                return
            elif key == ord(' ') and both:
                self.objpoints.append(self.objp)
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)
                self.last_capture = time.time()
                print(f"  [{len(self.objpoints):02d}/{MIN_PAIRS}] Manuell gespeichert ✓")
            elif key == ord('r'):
                self.objpoints.clear()
                self.imgpoints_l.clear()
                self.imgpoints_r.clear()
                print("  Reset — alle Paare gelöscht")

        cv2.destroyAllWindows()

        # Kalibrierung
        result = self.calibrate()
        self.save()

        # Ergebnis
        print("\n" + "="*52)
        print("  ERGEBNIS")
        print("="*52)
        err = result["error_stereo"]
        print(f"""
  Stereo-Fehler: {err:.3f}px
  Links-Fehler:  {result['error_left']:.3f}px
  Rechts-Fehler: {result['error_right']:.3f}px
  Baseline:      {result['baseline_cm']:.1f}cm
""")
        if   err < 0.5: print("  ✓ AUSGEZEICHNET — sehr genaue Tiefenmessung")
        elif err < 1.0: print("  ✓ GUT           — gute Tiefenmessung")
        elif err < 2.0: print("  ⚠ AKZEPTABEL    — Tiefe eingeschränkt genau")
        else:
            print("  ✗ SCHLECHT      — bitte neu kalibrieren")
            print("    → Mehr/bessere Bildpaare verwenden")
            print("    → Schachbrett plan und scharf halten")

        # Vorschau
        print("\nRektifizierungs-Vorschau? (j/n) ", end="")
        if input().strip().lower() == "j":
            maps = self.load_maps()
            if maps:
                self.preview_rectification(maps)

        self.disconnect()
        print("\n  Fertig! Nächster Schritt: python3 pose_detector.py\n")


if __name__ == "__main__":
    cal = StereoCalibrator()
    cal.run()
