#!/usr/bin/env python3
"""
ANTHRO3D — ELP Stereo Farb-Kamera
Öffnet die ELP Stereokamera (3200x1200) und teilt das Bild in Links/Rechts.
Automatische Erkennung des Kamera-Index.
"""
import cv2, numpy as np, yaml, os

BASE = os.path.expanduser("~/anthro3d")

def find_elp_index():
    """Findet den Index der ELP Stereokamera (3200x1200)."""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened(): cap.release(); continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w >= 3200 and h >= 1200:
            print(f"ELP gefunden: Index {i} ({w}x{h})")
            return i
    print("ELP nicht gefunden — verwende Index 0")
    return 0

class ELPStereo:
    def __init__(self, index=None):
        self.index = index or find_elp_index()
        self.cap   = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3200)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.half_w = self.W // 2
        print(f"ELP: {self.W}x{self.H} @ Index {self.index}")

    def read(self):
        """Liest einen Frame und teilt ihn in Links/Rechts."""
        ret, frame = self.cap.read()
        if not ret: return False, None, None
        fl = frame[:, :self.half_w]
        fr = frame[:, self.half_w:]
        return True, fl, fr

    def release(self):
        self.cap.release()

if __name__ == "__main__":
    stereo = ELPStereo()
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    print("ELP Test — Q=Beenden")
    while True:
        ret, fl, fr = stereo.read()
        if not ret: continue
        disp = np.hstack([cv2.resize(fl,(640,400)), cv2.resize(fr,(640,400))])
        cv2.putText(disp,"Links",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(disp,"Rechts",(650,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("ELP Stereo",disp)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cv2.destroyAllWindows()
    stereo.release()
