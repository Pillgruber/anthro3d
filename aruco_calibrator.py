# ANTHRO3D - ArUco Calibrator
# Kameras erkennen sich gegenseitig via ArUco Marker
# und kalibrieren sich vollautomatisch
#
# Jedes Stativ hat einen ArUco Marker:
# Kamera 1 Stativ → Marker ID 1
# Kamera 2 Stativ → Marker ID 2
# Kamera 3 Stativ → Marker ID 3
#
# Jede Kamera sieht die Marker der anderen Kameras
# → System berechnet automatisch Position & Winkel aller Kameras

import cv2
import numpy as np
import yaml
import json
import os
from datetime import datetime
from collections import deque


class ArucoCalibrator:
    def __init__(self, config):
        self.config = config
        self.marker_size_cm = config["calibration"]["marker_size_cm"]
        self.marker_size_m = self.marker_size_cm / 100.0
        self.stability_frames = config["calibration"]["stability_frames"]
        self.tolerance = config["calibration"]["tolerance"]
        self.ema_alpha = config["calibration"]["ema_alpha"]

        # ArUco Dictionary - 4x4 mit 50 Markern
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Kalibrierungsstatus pro Kamera
        self.calibration_data = {}
        self.is_calibrated = False

        # EMA und Stabilitaet pro Kamera
        self.ema_values = {}
        self.stability_counts = {}
        self.detection_history = {}

        # Kamera-Positionen im 3D-Raum (nach Kalibrierung)
        self.camera_positions = {}
        self.camera_rotations = {}

        # Standard Kamera-Matrix (wird spaeter mit echten Werten ersetzt)
        self.camera_matrix = np.array([
            [700, 0, 640],
            [0, 700, 360],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

        print("ArUco Calibrator bereit")
        print(f"  Marker Groesse: {self.marker_size_cm} cm")
        print(f"  Stabilitaet:    {self.stability_frames} Frames")
        print(f"  Toleranz:       {self.tolerance*100}%")

    def detect_markers(self, frame):
        """
        Erkennt alle ArUco Marker in einem Frame
        Gibt Liste von erkannten Markern zurueck
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, rejected = self.detector.detectMarkers(gray)

        markers = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                markers.append({
                    "id": int(marker_id),
                    "corners": corners[i]
                })

        return markers

    def calculate_px_per_mm(self, corners):
        """
        Berechnet px/mm Faktor aus Marker-Ecken
        """
        c = corners[0]
        d1 = np.linalg.norm(c[0] - c[1])
        d2 = np.linalg.norm(c[1] - c[2])
        d3 = np.linalg.norm(c[2] - c[3])
        d4 = np.linalg.norm(c[3] - c[0])
        avg_px = (d1 + d2 + d3 + d4) / 4
        px_per_mm = avg_px / (self.marker_size_cm * 10)
        return px_per_mm

    def update_ema(self, cam_id, marker_id, px_per_mm):
        """
        Exponentieller gleitender Durchschnitt fuer stabilen px/mm Wert
        """
        key = f"{cam_id}_{marker_id}"

        if key not in self.ema_values:
            self.ema_values[key] = px_per_mm
            self.stability_counts[key] = 0
        else:
            # EMA berechnen
            self.ema_values[key] = (
                self.ema_alpha * px_per_mm +
                (1 - self.ema_alpha) * self.ema_values[key]
            )

        # Stabilitaet pruefen
        deviation = abs(px_per_mm - self.ema_values[key]) / self.ema_values[key]
        if deviation < self.tolerance:
            self.stability_counts[key] = min(
                self.stability_counts.get(key, 0) + 1,
                self.stability_frames
            )
        else:
            self.stability_counts[key] = max(
                self.stability_counts.get(key, 0) - 1,
                0
            )

        return self.ema_values[key], self.stability_counts[key]

    def process_frame(self, cam_id, frame):
        """
        Verarbeitet einen Frame einer Kamera
        Erkennt Marker, berechnet Kalibrierung
        Gibt Status und annotiertes Frame zurueck
        """
        markers = self.detect_markers(frame)
        annotated = frame.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        results = {}

        for marker in markers:
            marker_id = marker["id"]
            corners = marker["corners"]

            # px/mm berechnen
            px_per_mm = self.calculate_px_per_mm(corners)

            # EMA updaten
            ema_val, stability = self.update_ema(cam_id, marker_id, px_per_mm)

            results[marker_id] = {
                "px_per_mm": ema_val,
                "stability": stability,
                "stable": stability >= self.stability_frames
            }

            # Marker zeichnen
            color = (0, 255, 255)  # Gelb - nicht stabil
            if stability >= self.stability_frames:
                color = (0, 255, 0)  # Gruen - stabil

            # Rahmen zeichnen
            pts = corners[0].astype(int)
            cv2.polylines(annotated, [pts], True, color, 2)

            # Mittelpunkt
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            # Info anzeigen
            stability_pct = int(stability / self.stability_frames * 100)
            cv2.putText(annotated,
                       f"ID:{marker_id} {ema_val:.3f}px/mm",
                       (cx - 60, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(annotated,
                       f"Stabilitaet: {stability_pct}%",
                       (cx - 60, cy + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return annotated, results

    def check_calibration_complete(self):
        """
        Prueft ob alle Kameras ausreichend kalibriert sind
        """
        if len(self.calibration_data) == 0:
            return False

        all_stable = all(
            data.get("stable", False)
            for data in self.calibration_data.values()
        )
        return all_stable

    def save_calibration(self, path="calibration.json"):
        """
        Speichert Kalibrierungsdaten fuer spaetere Verwendung
        """
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "marker_size_cm": self.marker_size_cm,
            "cameras": {}
        }

        for key, data in self.calibration_data.items():
            save_data["cameras"][str(key)] = {
                "px_per_mm": data.get("px_per_mm", 0),
                "stable": data.get("stable", False)
            }

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Kalibrierung gespeichert: {path}")

    def load_calibration(self, path="calibration.json"):
        """
        Laedt gespeicherte Kalibrierungsdaten
        """
        if not os.path.exists(path):
            print(f"Keine gespeicherte Kalibrierung gefunden: {path}")
            return False

        with open(path, "r") as f:
            data = json.load(f)

        self.calibration_data = data.get("cameras", {})
        self.is_calibrated = True
        print(f"Kalibrierung geladen: {path}")
        print(f"  Zeitstempel: {data.get('timestamp', 'unbekannt')}")
        return True

    def draw_status(self, frame, cam_id):
        """
        Zeichnet Kalibrierungs-Status ins Bild
        """
        h, w = frame.shape[:2]

        # Hintergrund
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        if self.is_calibrated:
            text = f"CAM {cam_id} | KALIBRIERT"
            color = (0, 255, 0)
        else:
            text = f"CAM {cam_id} | Warte auf Marker..."
            color = (0, 165, 255)

        cv2.putText(frame, text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


# TEST
if __name__ == "__main__":
    import yaml

    print("ANTHRO3D ArUco Calibrator Test")
    print("==============================\n")

    # Config laden
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    calibrator = ArucoCalibrator(config)

    print("\nTest mit eingebauter Kamera...")
    print("Halte einen ArUco 4x4 Marker (ID 0-3) vor die Kamera")
    print("Druecke Q zum Beenden\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Keine Kamera gefunden!")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame verarbeiten
        annotated, results = calibrator.process_frame(1, frame)

        # Status anzeigen
        annotated = calibrator.draw_status(annotated, 1)

        # Ergebnisse ausgeben
        if results:
            for marker_id, data in results.items():
                status = "STABIL" if data["stable"] else f"{data['stability']}/{calibrator.stability_frames}"
                print(f"\r  Marker {marker_id}: {data['px_per_mm']:.3f} px/mm | {status}    ", end="")

        cv2.imshow("ArUco Kalibrierung - Druecke Q zum Beenden", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\nTest abgeschlossen!")
