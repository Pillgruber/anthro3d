#!/usr/bin/env python3
"""
ANTHRO3D — Session Recorder
=============================
Nimmt Stereo-Bildpaare, Tiefenkarten, Keypoints und Messwerte auf
und speichert alles in einer .anthro3d Session-Datei.

Starten:
  python3 session_recorder.py

Tasten (im Kamerafenster):
  SPACE = Aufnahme starten / stoppen
  F     = Framerate ändern (5 / 10 / 30 / 60 / 120 / 300 fps)
  S     = Snapshot (einzelner Frame)
  Q     = Beenden und Session speichern

Dateiformat:
  session_YYYY-MM-DD_HH-MM_Nachname_Vorname.anthro3d
  (ZIP-Archiv mit JSON + NPZ Frames)

Framerate:
  Standard:  10 fps
  Maximum:   300 fps (OV9281 nativ bis 120fps, Rest interpoliert)
  Änderbar:  jederzeit per 'F' Taste
"""

import cv2
import numpy as np
import mediapipe as mp
import yaml, json, time, math, zipfile, io, threading, os, datetime
from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
CAM_L          = 1
CAM_R          = 2
RES            = (1280, 720)
CAM_FPS        = 120           # OV9281 native max
CFG_PATH       = Path("~/anthro3d/stereo_config.yaml").expanduser()
MAPS_PATH      = Path("~/anthro3d/stereo_maps.npz").expanduser()
SAVE_DIR       = Path("~/anthro3d/sessions").expanduser()
PREVIEW_SCALE  = 0.6

# Framerate-Stufen (Standard = Index 1 = 10fps)
FPS_STEPS      = [5, 10, 30, 60, 120, 300]
FPS_DEFAULT    = 1             # Index in FPS_STEPS

# Kompression
JPG_QUALITY    = 85            # Bildqualität 0-100
DEPTH_SCALE    = 1000          # mm → m beim Laden

# Farben
GREEN  = (80, 180, 80)
RED    = (60,  60, 200)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
YELLOW = (40, 200, 220)
ORANGE = (40, 140, 220)

# MediaPipe Landmarks
LM = {
  "nose":0,"l_eye":2,"r_eye":5,"l_ear":7,"r_ear":8,
  "l_shoulder":11,"r_shoulder":12,"l_elbow":13,"r_elbow":14,
  "l_wrist":15,"r_wrist":16,"l_hip":23,"r_hip":24,
  "l_knee":25,"r_knee":26,"l_ankle":27,"r_ankle":28,
  "l_heel":29,"r_heel":30,
}


# ── STEREO ────────────────────────────────────────────────────────────────────
class Stereo:
    def __init__(self):
        self.ok=False; self.map_l1=self.map_l2=self.map_r1=self.map_r2=self.Q=None
        self.K_l=None; self.baseline_m=0.08; self.focal=700.0

    def load(self):
        if not CFG_PATH.exists(): return False
        with open(CFG_PATH) as f: cfg=yaml.safe_load(f)
        mp_path=Path(cfg.get("maps_file",str(MAPS_PATH)))
        if not mp_path.exists(): return False
        d=np.load(str(mp_path))
        self.map_l1,self.map_l2=d["map_l1"],d["map_l2"]
        self.map_r1,self.map_r2=d["map_r1"],d["map_r2"]
        self.Q=np.array(cfg["Q"])
        self.K_l=np.array(cfg["K_left"])
        self.baseline_m=cfg.get("baseline_cm",8.0)/100.0
        self.focal=self.K_l[0,0]
        self.ok=True
        print(f"  Stereo ✓  Baseline={self.baseline_m*100:.1f}cm")
        return True

    def rectify(self,fl,fr):
        return cv2.remap(fl,self.map_l1,self.map_l2,cv2.INTER_LINEAR), \
               cv2.remap(fr,self.map_r1,self.map_r2,cv2.INTER_LINEAR)

    def disparity(self,rl,rr):
        gl=cv2.cvtColor(rl,cv2.COLOR_BGR2GRAY)
        gr=cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)
        sgbm=cv2.StereoSGBM_create(
            minDisparity=0,numDisparities=96,blockSize=5,
            P1=8*3*25,P2=32*3*25,disp12MaxDiff=1,
            uniquenessRatio=10,speckleWindowSize=100,speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return sgbm.compute(gl,gr).astype(np.float32)/16.0

    def depth_map(self,disp):
        """Disparität → Tiefenkarte in mm (uint16 für kompakte Speicherung)."""
        with np.errstate(divide='ignore',invalid='ignore'):
            depth_m=np.where(disp>1.0,(self.focal*self.baseline_m)/disp,0)
        depth_mm=np.clip(depth_m*DEPTH_SCALE,0,65535).astype(np.uint16)
        return depth_mm

    def depth_at(self,disp,px,py):
        x,y=int(px),int(py)
        h,w=disp.shape
        if not(0<x<w-1 and 0<y<h-1): return None
        patch=disp[max(0,y-2):y+3,max(0,x-2):x+3]
        valid=patch[patch>1.0]
        if len(valid)==0: return None
        d=float(np.median(valid))
        z=(self.focal*self.baseline_m)/d
        return z if 0.3<z<5.0 else None

    def to_3d(self,px,py,z):
        K=self.K_l
        x=(px-K[0,2])*z/K[0,0]
        y=(py-K[1,2])*z/K[1,1]
        return [x,y,z]


# ── MESSUNGEN ─────────────────────────────────────────────────────────────────
def dist3d(a,b):
    if a is None or b is None: return None
    return float(np.linalg.norm(np.array(a)-np.array(b))*100)

def angle_deg(a,b):
    if a is None or b is None: return None
    return abs(math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])))

def ellipse_circ(w,d):
    if w is None or d is None: return None
    a,b=w/2,d/2
    return math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))

def compute_measurements(pts2d, pts3d, px_per_cm=None):
    has3d=any(v is not None for v in pts3d.values())

    def seg(a,b):
        if has3d and pts3d.get(a) and pts3d.get(b):
            return dist3d(pts3d[a],pts3d[b])
        elif px_per_cm and pts2d.get(a) and pts2d.get(b):
            p,q=pts2d[a],pts2d[b]
            return math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2)/px_per_cm
        return None

    def fmt(v): return round(v,1) if v is not None else None

    sl,sr=pts2d.get("l_shoulder"),pts2d.get("r_shoulder")
    hl,hr=pts2d.get("l_hip"),pts2d.get("r_hip")
    smid=((sl[0]+sr[0])/2,(sl[1]+sr[1])/2) if sl and sr else None
    hmid=((hl[0]+hr[0])/2,(hl[1]+hr[1])/2) if hl and hr else None

    nose=pts2d.get("nose")
    ankle=pts2d.get("l_ankle") or pts2d.get("r_ankle")
    height=None
    if nose and ankle:
        if has3d and pts3d.get("nose") and (pts3d.get("l_ankle") or pts3d.get("r_ankle")):
            a3=pts3d.get("l_ankle") or pts3d.get("r_ankle")
            h=dist3d(pts3d["nose"],a3)
            if h: height=h*1.06
        elif px_per_cm:
            height=abs(nose[1]-ankle[1])/px_per_cm*1.06

    def zdiff(a,b):
        if pts3d.get(a) and pts3d.get(b):
            return abs(pts3d[a][2]-pts3d[b][2])*100
        return None

    return {
        "height":     fmt(height),
        "shoulder_w": fmt(seg("l_shoulder","r_shoulder")),
        "hip_w":      fmt(seg("l_hip","r_hip")),
        "arm_l":      fmt(seg("l_shoulder","l_elbow")),
        "arm_r":      fmt(seg("r_shoulder","r_elbow")),
        "forearm_l":  fmt(seg("l_elbow","l_wrist")),
        "forearm_r":  fmt(seg("r_elbow","r_wrist")),
        "thigh_l":    fmt(seg("l_hip","l_knee")),
        "thigh_r":    fmt(seg("r_hip","r_knee")),
        "shin_l":     fmt(seg("l_knee","l_ankle")),
        "shin_r":     fmt(seg("r_knee","r_ankle")),
        "torso":      fmt(seg("l_shoulder","l_hip")),
        "shoulder_ang": fmt(angle_deg(sl,sr)),
        "hip_ang":      fmt(angle_deg(hl,hr)),
        "knee_ang":     fmt(angle_deg(pts2d.get("l_knee"),pts2d.get("r_knee"))),
        "hws":          fmt(angle_deg(nose,smid)),
        "bws":          fmt(angle_deg(smid,hmid)),
        "lws":          fmt(angle_deg(hmid,pts2d.get("l_ankle"))),
        "circ_chest":   fmt(ellipse_circ(seg("l_shoulder","r_shoulder"),zdiff("l_shoulder","r_shoulder"))),
        "circ_hip":     fmt(ellipse_circ(seg("l_hip","r_hip"),zdiff("l_hip","r_hip"))),
        "has_stereo":   has3d,
    }


# ── SESSION ───────────────────────────────────────────────────────────────────
class Session:
    """Verwaltet eine Aufnahmesession — Frames puffern und als .anthro3d speichern."""

    def __init__(self, patient_name="", target_fps=10):
        self.patient_name = patient_name
        self.target_fps   = target_fps
        self.frames       = []          # Liste von Frame-Dicts
        self.start_time   = None
        self.recording    = False
        self.frame_count  = 0
        self.lock         = threading.Lock()

    def start(self):
        self.start_time = time.time()
        self.recording  = True
        self.frames     = []
        self.frame_count= 0
        print(f"\n  ● REC gestartet  {self.target_fps}fps  Patient: {self.patient_name or '—'}")

    def stop(self):
        self.recording = False
        dur = time.time()-self.start_time if self.start_time else 0
        print(f"\n  ■ REC gestoppt  {len(self.frames)} Frames  {dur:.1f}s")

    def add_frame(self, rgb_left, depth_mm, pts2d, pts3d, measurements, timestamp):
        """Frame in den Puffer aufnehmen (thread-safe)."""
        if not self.recording: return

        # RGB komprimieren
        ok, jpg_buf = cv2.imencode('.jpg', rgb_left, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
        jpg_bytes = jpg_buf.tobytes() if ok else b''

        frame = {
            "index":       len(self.frames),
            "timestamp":   round(timestamp, 4),
            "fps_target":  self.target_fps,
            "jpg":         jpg_bytes,           # komprimiertes Bild
            "depth_mm":    depth_mm,            # uint16 numpy array (oder None)
            "pts2d":       pts2d,               # {name: [x,y]}
            "pts3d":       pts3d,               # {name: [x,y,z]}
            "measurements":measurements,
        }
        with self.lock:
            self.frames.append(frame)
        self.frame_count += 1

    def duration(self):
        if not self.start_time: return 0
        if self.recording: return time.time()-self.start_time
        return self.frames[-1]["timestamp"]-self.frames[0]["timestamp"] if len(self.frames)>1 else 0

    def save(self):
        """Session als .anthro3d Datei speichern (ZIP-Archiv)."""
        if not self.frames:
            print("  Keine Frames — nichts zu speichern.")
            return None

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        safe_name = self.patient_name.replace(" ","_").replace("/","_") or "Unbekannt"
        filename = f"session_{now.strftime('%Y-%m-%d_%H-%M')}_{safe_name}.anthro3d"
        filepath = SAVE_DIR / filename

        print(f"\n  Speichere {len(self.frames)} Frames → {filename}")

        with self.lock:
            frames_copy = list(self.frames)

        with zipfile.ZipFile(str(filepath), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:

            # 1. Session-Metadaten
            meta = {
                "format_version": "1.0",
                "anthro3d":       True,
                "patient":        self.patient_name,
                "recorded_at":    now.isoformat(),
                "fps_target":     self.target_fps,
                "frame_count":    len(frames_copy),
                "duration_s":     round(self.duration(), 2),
                "resolution":     list(RES),
                "has_stereo":     any(f["measurements"].get("has_stereo") for f in frames_copy),
                # Durchschnittswerte der gesamten Session
                "avg_measurements": self._avg_measurements(frames_copy),
            }
            zf.writestr("session.json", json.dumps(meta, indent=2, ensure_ascii=False))

            # 2. Frames
            for i, fr in enumerate(frames_copy):
                prefix = f"frames/{i:05d}"

                # RGB-Bild (JPEG)
                zf.writestr(f"{prefix}_rgb.jpg", fr["jpg"])

                # Tiefenkarte (komprimiertes NPZ)
                if fr["depth_mm"] is not None:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, depth=fr["depth_mm"])
                    zf.writestr(f"{prefix}_depth.npz", buf.getvalue())

                # Keypoints + Messungen (JSON)
                frame_meta = {
                    "index":        fr["index"],
                    "timestamp":    fr["timestamp"],
                    "pts2d":        fr["pts2d"],
                    "pts3d":        fr["pts3d"],
                    "measurements": fr["measurements"],
                }
                zf.writestr(f"{prefix}_data.json",
                            json.dumps(frame_meta, ensure_ascii=False))

                if i % 50 == 0:
                    print(f"    Frame {i+1}/{len(frames_copy)}...")

        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"  ✓ Gespeichert: {filepath}")
        print(f"  Größe: {size_mb:.1f} MB  ({size_mb/self.duration():.1f} MB/s)")
        return str(filepath)

    def _avg_measurements(self, frames):
        keys=["height","shoulder_w","hip_w","arm_l","arm_r","forearm_l","forearm_r",
              "thigh_l","thigh_r","shin_l","shin_r","torso",
              "shoulder_ang","hip_ang","knee_ang","hws","bws","lws",
              "circ_chest","circ_hip"]
        avg={}
        for k in keys:
            vals=[f["measurements"].get(k) for f in frames if f["measurements"].get(k) is not None]
            avg[k]=round(sum(vals)/len(vals),1) if vals else None
        return avg


# ── RECORDER ──────────────────────────────────────────────────────────────────
class Recorder:
    def __init__(self):
        self.stereo   = Stereo()
        self.session  = None
        self.fps_idx  = FPS_DEFAULT
        self.last_cap = 0.0
        self.disp     = None
        self.disp_t   = 0
        self.frame_n  = 0
        self.fps_val  = 0.0
        self.fps_t    = time.time()

        # MediaPipe
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
            smooth_landmarks=True)
        self.mp_draw  = mp.solutions.drawing_utils
        self.cap_l = self.cap_r = None

    @property
    def target_fps(self): return FPS_STEPS[self.fps_idx]

    @property
    def frame_interval(self): return 1.0 / self.target_fps

    def connect(self):
        print("\nVerbinde Kameras...")
        self.cap_l = cv2.VideoCapture(CAM_L)
        self.cap_r = cv2.VideoCapture(CAM_R)
        ok = True
        for cap, name, idx in [(self.cap_l,"L",CAM_L),(self.cap_r,"R",CAM_R)]:
            if not cap.isOpened():
                print(f"  FEHLER: Kamera {name} (Index {idx})")
                ok=False; continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])
            cap.set(cv2.CAP_PROP_FPS, min(CAM_FPS, 120))
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print(f"  OK: Kamera {name} (Index {idx})")
        return ok

    def grab(self):
        ok_l=self.cap_l.grab(); ok_r=self.cap_r.grab()
        if not ok_l or not ok_r: return None,None
        _,fl=self.cap_l.retrieve(); _,fr=self.cap_r.retrieve()
        return fl,fr

    def get_keypoints(self, results, w, h):
        pts2d, pts3d = {}, {}
        if not results.pose_landmarks: return pts2d, pts3d
        for name, idx in LM.items():
            lm = results.pose_landmarks.landmark[idx]
            if lm.visibility < 0.35: continue
            px, py = lm.x*w, lm.y*h
            pts2d[name] = [round(px,2), round(py,2)]
            if self.stereo.ok and self.disp is not None:
                z = self.stereo.depth_at(self.disp, px, py)
                if z: pts3d[name] = [round(v,4) for v in self.stereo.to_3d(px,py,z)]
        return pts2d, pts3d

    def draw_hud(self, frame, recording, pts2d):
        out = frame.copy()
        h, w = out.shape[:2]

        # Skelett
        col_conn=(80,180,80); col_pt=(60,160,60)
        CONNS=[("l_shoulder","r_shoulder"),("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
               ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),("l_shoulder","l_hip"),
               ("r_shoulder","r_hip"),("l_hip","r_hip"),("l_hip","l_knee"),
               ("l_knee","l_ankle"),("r_hip","r_knee"),("r_knee","r_ankle")]
        for a,b in CONNS:
            if a in pts2d and b in pts2d:
                cv2.line(out,tuple(int(v) for v in pts2d[a]),
                             tuple(int(v) for v in pts2d[b]),col_conn,2,cv2.LINE_AA)
        for nm,(px,py) in pts2d.items():
            cv2.circle(out,(int(px),int(py)),4,col_pt,-1)
            cv2.circle(out,(int(px),int(py)),4,(255,255,255),1)

        # REC Indikator
        rec_col = RED if recording else (100,100,100)
        cv2.circle(out, (20,20), 10, rec_col, -1)
        rec_txt = "REC" if recording else "STAND-BY"
        cv2.putText(out, rec_txt, (36,26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_col, 1, cv2.LINE_AA)

        # Frame + FPS + Dauer
        if recording and self.session:
            frames = self.session.frame_count
            dur    = self.session.duration()
            size_est = frames * (JPG_QUALITY * 8 + 512) / 1024  # KB grob
            info = f"{frames} Frames  {dur:.1f}s  ~{size_est/1024:.1f}MB"
            cv2.putText(out, info, (50,26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, WHITE, 1, cv2.LINE_AA)

        # Framerate
        fps_txt = f"{self.target_fps}fps (Kamera {self.fps_val:.0f}fps)"
        cv2.putText(out, fps_txt, (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)

        # Stereo status
        st_txt = "STEREO 3D" if self.stereo.ok else "2D MODUS"
        st_col = GREEN if self.stereo.ok else ORANGE
        cv2.putText(out, st_txt, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, st_col, 1, cv2.LINE_AA)

        # Steuerung
        cmds = "SPACE=REC  F=FPS  S=Snapshot  Q=Beenden+Speichern"
        cv2.putText(out, cmds, (w//2-200, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)

        return out

    def update_fps(self):
        self.frame_n += 1
        now = time.time()
        if now-self.fps_t >= 1.0:
            self.fps_val = self.frame_n/(now-self.fps_t)
            self.frame_n = 0; self.fps_t = now

    def run(self):
        print("\n"+"="*52)
        print("  ANTHRO3D — Session Recorder")
        print("="*52)
        print(f"\n  Standard-Framerate: {self.target_fps}fps")
        print(f"  Maximale Framerate: {FPS_STEPS[-1]}fps")
        print(f"  Speicherort: {SAVE_DIR}")

        # Patientenname
        print("\nPatientenname (Enter = überspringen): ", end="", flush=True)
        patient = input().strip()

        # Stereo laden
        print("\nLade Stereo-Kalibrierung...")
        if not self.stereo.load():
            print("  Kein Stereo — nur 2D-Aufnahme")

        # Kameras
        print("\nVerbinde Kameras...")
        if not self.connect():
            print("\nFEHLER: Kameras nicht verfügbar.")
            return

        print(f"""
Bereit!
  SPACE = Aufnahme starten/stoppen
  F     = Framerate wechseln (aktuell {self.target_fps}fps)
  S     = Einzelner Snapshot
  Q     = Beenden und speichern
""")

        pts2d, pts3d, meas = {}, {}, {}

        while True:
            fl, fr = self.grab()
            if fl is None: break
            self.update_fps()
            now = time.time()

            # Rektifizieren
            display = fl.copy()
            if self.stereo.ok:
                rl, rr = self.stereo.rectify(fl, fr)
                display = rl
                # Disparität alle 0.1s aktualisieren
                if now - self.disp_t > 0.1:
                    self.disp = self.stereo.disparity(rl, rr)
                    self.disp_t = now

            # MediaPipe
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            pts2d, pts3d = self.get_keypoints(results, RES[0], RES[1])
            meas = compute_measurements(pts2d, pts3d)

            # Frame aufnehmen wenn REC aktiv und Intervall erreicht
            recording = self.session is not None and self.session.recording
            if recording and (now - self.last_cap) >= self.frame_interval:
                depth_mm = None
                if self.stereo.ok and self.disp is not None:
                    depth_mm = self.stereo.depth_map(self.disp)
                self.session.add_frame(
                    display, depth_mm, pts2d, pts3d, meas, now)
                self.last_cap = now

            # HUD
            out = self.draw_hud(display, recording, pts2d)
            sw, sh = int(RES[0]*PREVIEW_SCALE), int(RES[1]*PREVIEW_SCALE)
            cv2.imshow("ANTHRO3D Recorder", cv2.resize(out, (sw, sh)))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                if not recording:
                    # Neue Session starten
                    self.session = Session(patient, self.target_fps)
                    self.session.start()
                    self.last_cap = 0
                else:
                    self.session.stop()

            elif key == ord('f'):
                # Framerate wechseln
                self.fps_idx = (self.fps_idx+1) % len(FPS_STEPS)
                new_fps = self.target_fps
                if self.session: self.session.target_fps = new_fps
                print(f"  Framerate: {new_fps}fps")

            elif key == ord('s'):
                # Einzelner Snapshot
                snap_path = SAVE_DIR / f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap_path), display)
                print(f"  Snapshot: {snap_path}")

        cv2.destroyAllWindows()
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()

        # Session speichern
        if self.session and self.session.frames:
            if self.session.recording: self.session.stop()
            saved = self.session.save()
            if saved:
                print(f"\n  Session bereit für Player:")
                print(f"  {saved}")
        print("\nAuf Wiedersehen!")


if __name__ == "__main__":
    Recorder().run()
