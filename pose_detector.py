#!/usr/bin/env python3
"""
ANTHRO3D — Pose Detector v2.0
===============================
MediaPipe Tasks API + Stereo → 3D Körpermaße → WebSocket → Live Dashboard

Starten:
  python3 pose_detector.py

Tasten (im Kamerafenster):
  L = Sprache umschalten (de/en)
  S = Skelett an/aus
  K = Kalman-Filter an/aus (Standard: AUS)
  Q = Beenden
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import yaml, json, time, math, asyncio, threading, websockets
from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
CAM_L        = 0
CAM_R        = 2
RES          = (1280, 720)
FPS          = 60
WS_HOST      = "localhost"
WS_PORT      = 8767
CFG_PATH     = Path("~/anthro3d/stereo_config.yaml").expanduser()
MAPS_PATH    = Path("~/anthro3d/stereo_maps.npz").expanduser()
MODEL_PATH   = Path("~/anthro3d/pose_landmarker.task").expanduser()
SCALE        = 0.65
LANG         = ["de"]

# ── LABELS ────────────────────────────────────────────────────────────────────
T = {
  "de": {
    "height":"Körpergröße","shoulder_w":"Schulterbreite","hip_w":"Hüftbreite",
    "arm_l":"Oberarm L","arm_r":"Oberarm R","forearm_l":"Unterarm L","forearm_r":"Unterarm R",
    "thigh_l":"Oberschenkel L","thigh_r":"Oberschenkel R","shin_l":"Unterschenkel L","shin_r":"Unterschenkel R",
    "torso":"Rumpflänge","shoulder_ang":"Schulterachse","hip_ang":"Beckenachse","knee_ang":"Knieachse",
    "hws":"HWS","bws":"BWS","lws":"LWS",
    "circ_waist":"Taille","circ_hip":"Hüfte","circ_chest":"Brust",
    "searching":"Suche Person...","found":"Person erkannt",
  },
  "en": {
    "height":"Body Height","shoulder_w":"Shoulder Width","hip_w":"Hip Width",
    "arm_l":"Upper Arm L","arm_r":"Upper Arm R","forearm_l":"Forearm L","forearm_r":"Forearm R",
    "thigh_l":"Thigh L","thigh_r":"Thigh R","shin_l":"Shin L","shin_r":"Shin R",
    "torso":"Torso","shoulder_ang":"Shoulder Axis","hip_ang":"Pelvis Axis","knee_ang":"Knee Axis",
    "hws":"Cervical","bws":"Thoracic","lws":"Lumbar",
    "circ_waist":"Waist","circ_hip":"Hip","circ_chest":"Chest",
    "searching":"Searching...","found":"Person detected",
  }
}

# MediaPipe PoseLandmarker Indizes (33 Landmarks)
LM = {
  "nose":0,"l_eye":2,"r_eye":5,"l_ear":7,"r_ear":8,
  "l_shoulder":11,"r_shoulder":12,"l_elbow":13,"r_elbow":14,
  "l_wrist":15,"r_wrist":16,"l_hip":23,"r_hip":24,
  "l_knee":25,"r_knee":26,"l_ankle":27,"r_ankle":28,
  "l_heel":29,"r_heel":30,
}

# Skeleton Verbindungen für Overlay
SKEL_CONNS = [
  ("l_shoulder","r_shoulder"),(("l_shoulder","l_elbow")),("l_elbow","l_wrist"),
  ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
  ("l_shoulder","l_hip"),("r_shoulder","r_hip"),
  ("l_hip","r_hip"),("l_hip","l_knee"),("l_knee","l_ankle"),
  ("r_hip","r_knee"),("r_knee","r_ankle"),
  ("nose","l_ear"),("nose","r_ear"),
]

# ── KALMAN-FILTER ─────────────────────────────────────────────────────────────
class KalmanFilter2D:
    def __init__(self, process_noise=5e-3, measurement_noise=2e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost      = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y):
        meas = np.array([[x],[y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePre  = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.initialized  = True
        self.kf.predict()
        est = self.kf.correct(meas)
        return float(est[0]), float(est[1])

    def reset(self): self.initialized = False

class KeypointSmoother:
    def __init__(self):
        self.enabled  = False
        self.filters  = {}
        self.conf_thr = 0.55

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            for f in self.filters.values(): f.reset()
        print(f"  Kalman-Filter: {'EIN' if self.enabled else 'AUS'}")
        return self.enabled

    def smooth(self, name, x, y, visibility):
        if visibility < self.conf_thr:
            if name in self.filters: self.filters[name].reset()
            return None
        if not self.enabled: return (x, y)
        if name not in self.filters:
            self.filters[name] = KalmanFilter2D()
        return self.filters[name].update(x, y)

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
        self.Q=np.array(cfg["Q"]); self.K_l=np.array(cfg["K_left"])
        self.baseline_m=cfg.get("baseline_mm",cfg.get("baseline_cm",80.0)*10)/1000.0; self.focal=self.K_l[0,0]
        self.ok=True; print(f"  Stereo ✓  Baseline={self.baseline_m*100:.1f}cm")
        return True

    def rectify(self,fl,fr):
        return cv2.remap(fl,self.map_l1,self.map_l2,cv2.INTER_LINEAR), \
               cv2.remap(fr,self.map_r1,self.map_r2,cv2.INTER_LINEAR)

    def disparity(self,rl,rr):
        gl=cv2.cvtColor(rl,cv2.COLOR_BGR2GRAY); gr=cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)
        sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=5,
            P1=8*3*25,P2=32*3*25,disp12MaxDiff=1,uniquenessRatio=10,
            speckleWindowSize=100,speckleRange=32,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return sgbm.compute(gl,gr).astype(np.float32)/16.0

    def depth_at(self,disp,px,py):
        x,y=int(px),int(py); h,w=disp.shape
        if not(0<x<w-1 and 0<y<h-1): return None
        patch=disp[max(0,y-2):y+3,max(0,x-2):x+3]; valid=patch[patch>1.0]
        if len(valid)==0: return None
        d=float(np.median(valid)); z=(self.focal*self.baseline_m)/d
        return z if 0.3<z<5.0 else None

    def to_3d(self,px,py,z):
        K=self.K_l; x=(px-K[0,2])*z/K[0,0]; y=(py-K[1,2])*z/K[1,1]
        return [x,y,z]

# ── KAMERAS ───────────────────────────────────────────────────────────────────
class Cameras:
    def __init__(self): self.cap_l=self.cap_r=None

    def connect(self):
        self.cap_l=cv2.VideoCapture(CAM_L); self.cap_r=cv2.VideoCapture(CAM_R)
        ok=True
        for cap,name,idx in [(self.cap_l,"L",CAM_L),(self.cap_r,"R",CAM_R)]:
            if not cap.isOpened(): print(f"  FEHLER: Kamera {name} ({idx})"); ok=False; continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,RES[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,RES[1])
            cap.set(cv2.CAP_PROP_FPS,FPS); print(f"  OK: Kamera {name} (Index {idx})")
        return ok

    def grab(self):
        ok_l=self.cap_l.grab(); ok_r=self.cap_r.grab()
        if not ok_l or not ok_r: return None,None
        _,fl=self.cap_l.retrieve(); _,fr=self.cap_r.retrieve()
        return fl,fr

    def release(self):
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()

# ── WEBSOCKET ─────────────────────────────────────────────────────────────────
class WSServer:
    def __init__(self): self.clients=set(); self.last={}; self.loop=None; self.smoother=None

    def start(self, smoother=None):
        self.smoother = smoother
        threading.Thread(target=self._run,daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        self.loop=asyncio.new_event_loop(); asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async def handler(ws):
            self.clients.add(ws)
            try:
                if self.last: await ws.send(json.dumps(self.last))
                async for msg in ws:
                    try:
                        cmd = json.loads(msg)
                        if cmd.get('cmd') == 'kalman':
                            if self.smoother:
                                self.smoother.enabled = bool(cmd.get('enabled', False))
                                print(f"  Kalman-Filter: {'EIN' if self.smoother.enabled else 'AUS'} (Browser)")
                        elif cmd.get('cmd') == 'rec_start':
                            print("  REC: Start (Browser)")
                        elif cmd.get('cmd') == 'rec_stop':
                            print("  REC: Stop (Browser)")
                    except: pass
            finally: self.clients.discard(ws)
        async with websockets.serve(handler,WS_HOST,WS_PORT):
            print(f"  WebSocket: ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()

    def send(self,data):
        self.last=data
        if not self.clients or not self.loop: return
        asyncio.run_coroutine_threadsafe(self._broadcast(json.dumps(data)),self.loop)

    async def _broadcast(self,msg):
        dead=set()
        for ws in list(self.clients):
            try: await ws.send(msg)
            except: dead.add(ws)
        self.clients-=dead

# ── MESSUNGEN ─────────────────────────────────────────────────────────────────
def dist3d(a,b):
    if a is None or b is None: return None
    return float(np.linalg.norm(np.array(a)-np.array(b))*100)

def dist2d_cm(a,b,px_per_cm):
    if a is None or b is None or px_per_cm is None: return None
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)/px_per_cm

def angle_deg(a,b):
    if a is None or b is None: return None
    return abs(math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])))

def ellipse_circ(w,d):
    if w is None or d is None: return None
    a,b=w/2,d/2
    return math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))

def _diff(a,b):
    if a is None or b is None: return None
    return abs(a-b)

# ── HAUPT-DETEKTOR ────────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self):
        self.stereo    = Stereo()
        self.cams      = Cameras()
        self.ws        = WSServer()
        self.px_per_cm = None
        self.show_skel = True
        self.frame_n   = 0
        self.fps_t     = time.time()
        self.fps_val   = 0.0
        self.smoother  = KeypointSmoother()
        self.landmarker = None
        self.last_landmarks = None

    def init_landmarker(self):
        if not MODEL_PATH.exists():
            print(f"  FEHLER: Modell nicht gefunden: {MODEL_PATH}")
            print("  Download: curl -o ~/anthro3d/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
            return False
        options = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.50,
            output_segmentation_masks=False)
        self.landmarker = PoseLandmarker.create_from_options(options)
        print("  PoseLandmarker ✓")
        return True

    def get_landmarks(self, frame, timestamp_ms):
        if self.landmarker is None: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.last_landmarks = result.pose_landmarks[0]
            return result.pose_landmarks[0]
        return None

    def get_lm(self, landmarks, name, w, h):
        if landmarks is None: return None
        idx = LM.get(name)
        if idx is None or idx >= len(landmarks): return None
        lm = landmarks[idx]
        pt = self.smoother.smooth(name, lm.x*w, lm.y*h, lm.visibility)
        return pt

    def get_lm_3d(self, landmarks, name, disp, K, w, h):
        lm2d = self.get_lm(landmarks, name, w, h)
        if lm2d is None: return lm2d, None
        if disp is None or not self.stereo.ok: return lm2d, None
        z = self.stereo.depth_at(disp, lm2d[0], lm2d[1])
        if z is None: return lm2d, None
        return lm2d, [round(v,4) for v in self.stereo.to_3d(lm2d[0],lm2d[1],z)]

    def compute_measurements(self, landmarks, disp, K, w, h):
        L = T[LANG[0]]
        def p2(n): return self.get_lm(landmarks, n, w, h)
        def p3(n):
            _, pt = self.get_lm_3d(landmarks, n, disp, K, w, h)
            return pt

        pts2d = {k: p2(k) for k in LM}
        pts3d = {k: p3(k) for k in LM} if (disp is not None and self.stereo.ok) else {k:None for k in LM}
        has3d = any(v is not None for v in pts3d.values())

        def seg(a,b):
            if has3d and pts3d.get(a) and pts3d.get(b): return dist3d(pts3d[a],pts3d[b])
            if self.px_per_cm and pts2d.get(a) and pts2d.get(b): return dist2d_cm(pts2d[a],pts2d[b],self.px_per_cm)
            return None

        def fmt(v): return round(v,1) if v is not None else None

        sl,sr = pts2d.get("l_shoulder"),pts2d.get("r_shoulder")
        hl,hr = pts2d.get("l_hip"),pts2d.get("r_hip")
        smid = ((sl[0]+sr[0])/2,(sl[1]+sr[1])/2) if sl and sr else None
        hmid = ((hl[0]+hr[0])/2,(hl[1]+hr[1])/2) if hl and hr else None
        nose  = pts2d.get("nose")
        ankle = pts2d.get("l_ankle") or pts2d.get("r_ankle")

        height = None
        if nose and ankle:
            if has3d and pts3d.get("nose") and (pts3d.get("l_ankle") or pts3d.get("r_ankle")):
                a3 = pts3d.get("l_ankle") or pts3d.get("r_ankle")
                h_ = dist3d(pts3d["nose"],a3)
                if h_: height = h_*1.06
            elif self.px_per_cm:
                height = abs(nose[1]-ankle[1])/self.px_per_cm*1.06

        axis_x = smid[0] if smid else None
        def axis_off(pt):
            if pt and axis_x: return round((pt[0]-axis_x)/self.px_per_cm if self.px_per_cm else pt[0]-axis_x,1)
            return None

        return {
            "height":fmt(height),"shoulder_w":fmt(seg("l_shoulder","r_shoulder")),
            "hip_w":fmt(seg("l_hip","r_hip")),"arm_l":fmt(seg("l_shoulder","l_elbow")),
            "arm_r":fmt(seg("r_shoulder","r_elbow")),"forearm_l":fmt(seg("l_elbow","l_wrist")),
            "forearm_r":fmt(seg("r_elbow","r_wrist")),"thigh_l":fmt(seg("l_hip","l_knee")),
            "thigh_r":fmt(seg("r_hip","r_knee")),"shin_l":fmt(seg("l_knee","l_ankle")),
            "shin_r":fmt(seg("r_knee","r_ankle")),"torso":fmt(seg("l_shoulder","l_hip")),
            "shoulder_ang":fmt(angle_deg(sl,sr)),"hip_ang":fmt(angle_deg(hl,hr)),
            "knee_ang":fmt(angle_deg(pts2d.get("l_knee"),pts2d.get("r_knee"))),
            "hws":fmt(angle_deg(nose,smid)),"bws":fmt(angle_deg(smid,hmid)),
            "lws":fmt(angle_deg(hmid,pts2d.get("l_ankle"))),
            "axis_shoulder_l":axis_off(sl),"axis_shoulder_r":axis_off(sr),
            "axis_hip_l":axis_off(hl),"axis_hip_r":axis_off(hr),
            "axis_knee_l":axis_off(pts2d.get("l_knee")),"axis_knee_r":axis_off(pts2d.get("r_knee")),
            "has_stereo":has3d,"lang":LANG[0],"fps":round(self.fps_val,1),
            "timestamp":round(time.time(),3),
            "landmarks_2d":{k:[round(v[0]/w,4),round(v[1]/h,4)] for k,v in pts2d.items() if v},
        }

    def draw_overlay(self, frame, landmarks, meas):
        if not self.show_skel or landmarks is None: return frame
        out = frame.copy()
        h, w = out.shape[:2]

        def p(name):
            pt = self.get_lm(landmarks, name, w, h)
            return (int(pt[0]),int(pt[1])) if pt else None

        # Skelett-Linien
        for a,b in SKEL_CONNS:
            pa,pb = p(a),p(b)
            if pa and pb: cv2.line(out,pa,pb,(80,180,80),2,cv2.LINE_AA)

        # Keypoint-Dots
        for name in LM:
            pt = p(name)
            if pt:
                cv2.circle(out,pt,4,(60,160,60),-1)
                cv2.circle(out,pt,4,(255,255,255),1)

        # Achsenlinien
        sl,sr = p("l_shoulder"),p("r_shoulder")
        hl,hr = p("l_hip"),p("r_hip")
        kl,kr = p("l_knee"),p("r_knee")
        if sl and sr: cv2.line(out,sl,sr,(80,80,220),2,cv2.LINE_AA)
        if hl and hr: cv2.line(out,hl,hr,(200,80,200),2,cv2.LINE_AA)
        if kl and kr: cv2.line(out,kl,kr,(80,180,80),2,cv2.LINE_AA)

        # Körperlot
        nose,ankle = p("nose"),p("l_ankle") or p("r_ankle")
        if nose and ankle:
            cx = (nose[0]+ankle[0])//2
            cv2.line(out,(cx,0),(cx,h),(180,180,180),1,cv2.LINE_AA)

        # Info-Box
        L = T[LANG[0]]
        h_val = meas.get("height")
        kalman_txt = "KALMAN ✓" if self.smoother.enabled else ""
        stereo_txt = "STEREO 3D" if meas.get("has_stereo") else "2D"
        lines = [
            (f"ANTHRO3D  {stereo_txt}  [{LANG[0].upper()}]  {meas.get('fps',0):.0f}fps  {kalman_txt}", (200,220,200)),
            (f"{L.get('height','Groesse')}: {h_val:.1f}cm" if h_val else f"{L.get('height','Groesse')}: ---", (120,220,120)),
        ]
        y0 = 18
        for txt,col in lines:
            cv2.putText(out,txt,(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(out,txt,(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)
            y0 += 18
        return out

    def update_fps(self):
        self.frame_n += 1
        now = time.time()
        if now-self.fps_t >= 1.0:
            self.fps_val = self.frame_n/(now-self.fps_t)
            self.frame_n = 0; self.fps_t = now

    def run(self):
        print("\n"+"="*52)
        print("  ANTHRO3D — Pose Detector v2.0")
        print("="*52)
        print("\nLade Modell...")
        if not self.init_landmarker():
            return
        print("\nLade Stereo-Kalibrierung...")
        self.stereo.load()
        print("\nVerbinde Kameras...")
        if not self.cams.connect():
            print("\nFEHLER: Kameras nicht verfügbar.")
            return
        K = None
        if self.stereo.ok:
            with open(CFG_PATH) as f: cfg=yaml.safe_load(f)
            K = np.array(cfg["K_left"])
        self.ws.start(self.smoother)
        print(f"""
Tasten:
  L = Sprache ({LANG[0]})
  S = Skelett an/aus
  K = Kalman-Filter an/aus (Standard: AUS)
  Q = Beenden

Dashboard: http://localhost:8080/anthro3d_app.html
""")
        disp = None
        disp_t = 0
        timestamp_ms = 0

        while True:
            fl,fr = self.cams.grab()
            if fl is None: break
            self.update_fps()
            now = time.time()
            timestamp_ms += int(1000/FPS)

            display = fl.copy()
            if self.stereo.ok:
                rl,rr = self.stereo.rectify(fl,fr)
                display = rl
                if now-disp_t > 0.1:
                    disp = self.stereo.disparity(rl,rr); disp_t=now

            landmarks = self.get_landmarks(display, timestamp_ms)
            meas = {}
            if landmarks:
                meas = self.compute_measurements(landmarks,disp,K,RES[0],RES[1])
                self.ws.send({"type":"measurements","kalman_enabled":self.smoother.enabled,**meas})

            out = self.draw_overlay(display, landmarks, meas)
            sw,sh = int(RES[0]*SCALE),int(RES[1]*SCALE)
            cv2.imshow("ANTHRO3D",cv2.resize(out,(sw,sh)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27): break
            elif key == ord('l'):
                LANG[0]="en" if LANG[0]=="de" else "de"
                print(f"  Sprache: {LANG[0].upper()}")
            elif key == ord('s'): self.show_skel=not self.show_skel
            elif key == ord('k'): self.smoother.toggle()

        cv2.destroyAllWindows()
        self.cams.release()
        if self.landmarker: self.landmarker.close()
        print("\nBeendet.")

if __name__ == "__main__":
    PoseDetector().run()
