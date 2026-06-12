#!/usr/bin/env python3
"""
ANTHRO3D — Background Removal Modul
=====================================
Zwei Anzeigemodi umschaltbar per Tastendruck:

  Modus 0 — NORMAL:   echtes Kamerabild
  Modus 1 — KLINISCH: Person als saubere Silhouette im ANTHRO3D Design
                       (wie SCANECA — hellgrau, neutraler Hintergrund)

Verwendung:
  from background_removal import BackgroundRemoval
  bg = BackgroundRemoval()
  bg.load()

  # Im Frame-Loop:
  result = bg.process(frame, results)  # results = MediaPipe Pose Ergebnis
  cv2.imshow('ANTHRO3D', result)

  # Modus wechseln:
  bg.next_mode()   # Normal → Klinisch → Normal ...
  bg.set_mode(1)   # direkt setzen
"""

import cv2
import numpy as np
import mediapipe as mp
from enum import IntEnum

class Mode(IntEnum):
    NORMAL   = 0   # echtes Kamerabild
    CLINICAL = 1   # ANTHRO3D klinisches Design

# ── DESIGN-FARBEN (aus ANTHRO3D Design-Sprache) ───────────────────────────────
DESIGN = {
    # Klinischer Modus — helles, blasses Grün wie Setup Wizard
    "bg_color":       (241, 244, 240),  # --bg: #f0f4f1 (BGR)
    "body_light":     (230, 235, 228),  # Körper hell (wie SCANECA Hellgrau)
    "body_shadow":    (195, 210, 198),  # Körper Schatten/Tiefe
    "body_edge":      (160, 180, 165),  # Körperkante
    "edge_thickness": 2,
    "blur_body":      True,             # Körper weich zeichnen
    "blur_bg":        False,
    # Normal-Modus Hintergrund-Weichzeichner (optional)
    "normal_bg_blur": False,
}

class BackgroundRemoval:
    def __init__(self):
        self.mode       = Mode.NORMAL
        self.segmentor  = None
        self.loaded     = False
        self._prev_mask = None   # für zeitliche Glättung

    def load(self):
        """MediaPipe Selfie Segmentation laden."""
        try:
            mp_seg = mp.solutions.selfie_segmentation
            self.segmentor = mp_seg.SelfieSegmentation(model_selection=1)
            self.loaded = True
            print("  Background Removal ✓ (MediaPipe Segmentation)")
        except Exception as e:
            print(f"  Background Removal FEHLER: {e}")
            self.loaded = False
        return self.loaded

    def next_mode(self):
        self.mode = Mode((self.mode + 1) % len(Mode))
        names = {Mode.NORMAL: "Normal", Mode.CLINICAL: "Klinisch"}
        print(f"  Anzeigemodus: {names[self.mode]}")
        return self.mode

    def set_mode(self, mode: int):
        self.mode = Mode(mode)

    def mode_name(self):
        return {Mode.NORMAL: "NORMAL", Mode.CLINICAL: "KLINISCH"}[self.mode]

    def process(self, frame, pose_results=None):
        """
        Verarbeitet einen Frame je nach aktivem Modus.
        frame: BGR numpy array
        pose_results: MediaPipe Pose Ergebnis (optional, für bessere Maske)
        Gibt verarbeiteten Frame zurück.
        """
        if self.mode == Mode.NORMAL:
            return frame.copy()

        if not self.loaded or self.segmentor is None:
            return frame.copy()

        h, w = frame.shape[:2]

        # ── Segmentierungsmaske berechnen ─────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_result = self.segmentor.process(rgb)

        if seg_result.segmentation_mask is None:
            return frame.copy()

        raw_mask = seg_result.segmentation_mask  # float32, 0–1

        # Zeitliche Glättung (EMA) für ruhigere Maske
        if self._prev_mask is not None and self._prev_mask.shape == raw_mask.shape:
            raw_mask = 0.7 * raw_mask + 0.3 * self._prev_mask
        self._prev_mask = raw_mask.copy()

        # Maske schärfen + morphologisch bereinigen
        mask = (raw_mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Weiche Kante (Feathering)
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        mask_3ch  = np.stack([mask_blur]*3, axis=2)

        # ── Klinischer Modus ──────────────────────────────────────────────────
        return self._clinical(frame, mask, mask_3ch, h, w)

    def _clinical(self, frame, mask, mask_3ch, h, w):
        """
        ANTHRO3D klinisches Design — wie SCANECA:
        - Neutraler heller Hintergrund (--bg Farbe)
        - Person als weiche Grauton-Silhouette mit Tiefensimulation
        - Subtile Körperkante
        """
        D = DESIGN

        # 1. Hintergrund — einheitliche Designfarbe
        bg = np.full((h, w, 3), D["bg_color"], dtype=np.uint8)

        # 2. Körper-Rendering — Graustufenversion mit simulierter Tiefe
        #    Konvertiere Person zu Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalisiere Helligkeit der Person auf Design-Palette
        # Hell: body_light, Dunkel: body_shadow
        gray_norm = gray.astype(np.float32) / 255.0

        body_light  = np.array(D["body_light"],  dtype=np.float32)
        body_shadow = np.array(D["body_shadow"], dtype=np.float32)

        # Interpoliere zwischen Schatten und Licht
        body_colored = (body_shadow[np.newaxis, np.newaxis, :] +
                        gray_norm[:, :, np.newaxis] *
                        (body_light - body_shadow)[np.newaxis, np.newaxis, :])
        body_colored = np.clip(body_colored, 0, 255).astype(np.uint8)

        # Körper leicht weichzeichnen (wirkt plastischer)
        if D["blur_body"]:
            body_colored = cv2.bilateralFilter(body_colored, 9, 75, 75)

        # 3. Kante (Edge) für plastischen Effekt
        edge = cv2.Canny(mask.astype(np.uint8)*255, 50, 150)
        edge_dilated = cv2.dilate(edge, np.ones((D["edge_thickness"],
                                                  D["edge_thickness"]), np.uint8))
        edge_3ch = np.stack([edge_dilated]*3, axis=2)
        edge_col  = np.array(D["body_edge"], dtype=np.uint8)

        # 4. Compositing
        # Hintergrund
        result = bg.copy()
        # Körper einfügen
        result = (result * (1 - mask_3ch) + body_colored * mask_3ch).astype(np.uint8)
        # Kante einblenden
        edge_mask = (edge_3ch > 0).astype(np.float32) * mask_3ch
        result = (result * (1 - edge_mask * 0.5) +
                  edge_col * edge_mask * 0.5).astype(np.uint8)

        # 5. Subtile Vignette (wirkt professioneller)
        vignette = self._make_vignette(h, w)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)

        # 6. Modus-Label einblenden
        self._draw_mode_label(result, "KLINISCH")

        return result

    def _make_vignette(self, h, w):
        """Erzeugt eine subtile Vignette (dunkler Rand)."""
        Y, X = np.mgrid[0:h, 0:w]
        cx, cy = w//2, h//2
        dist = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
        vignette = np.clip(1.0 - dist*0.15, 0.85, 1.0)
        return vignette[:, :, np.newaxis]

    def _draw_mode_label(self, frame, label):
        """Kleines Modus-Label oben rechts."""
        h, w = frame.shape[:2]
        txt = f"[ {label} ]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x, y = w - tw - 10, 18
        cv2.putText(frame, txt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 180, 165), 1, cv2.LINE_AA)


# ── INTEGRATION IN POSE_DETECTOR / SESSION_RECORDER ──────────────────────────
"""
In pose_detector.py / session_recorder.py ergänzen:

from background_removal import BackgroundRemoval, Mode

# In __init__:
self.bg_removal = BackgroundRemoval()
self.bg_removal.load()

# Im Frame-Loop (nach MediaPipe):
display_frame = self.bg_removal.process(display_frame, results)

# Taste 'V' = View-Modus wechseln:
elif key == ord('v'):
    self.bg_removal.next_mode()

# Tasten anzeigen:
# V = Ansicht (Normal / Klinisch)
"""


if __name__ == "__main__":
    """Test-Modus: Webcam öffnen und Modi durchschalten."""
    import sys

    print("ANTHRO3D — Background Removal Test")
    print("Taste V = Modus wechseln  |  Q = Beenden")

    bg  = BackgroundRemoval()
    if not bg.load():
        print("Fehler: MediaPipe nicht verfügbar")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Gespiegelt anzeigen
        frame = cv2.flip(frame, 1)

        # Pose detection
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Background Removal
        out = bg.process(frame, results)

        # Skelett drüber (wenn Pose erkannt)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                out, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(90,140,100) if bg.mode==Mode.CLINICAL else (80,200,80),
                    thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(70,120,80) if bg.mode==Mode.CLINICAL else (40,160,40),
                    thickness=2))

        # Info-Text
        cv2.putText(out, f"Modus: {bg.mode_name()}  |  V = wechseln  Q = beenden",
                    (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (140,160,145), 1, cv2.LINE_AA)

        # Skaliert anzeigen
        h, w = out.shape[:2]
        cv2.imshow("ANTHRO3D", cv2.resize(out, (int(w*0.65), int(h*0.65))))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('v'): bg.next_mode()

    cap.release()
    cv2.destroyAllWindows()
