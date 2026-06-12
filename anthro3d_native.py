#!/usr/bin/env python3
"""
ANTHRO3D — Native Desktop App
================================
Vollständige App in einem OpenCV-Fenster.
Keine Browser, kein WebSocket nötig.

Starten:
  python3 anthro3d_native.py

Maus:  Buttons klicken
Tasten:
  K = Kalman-Filter an/aus
  S = Skelett an/aus
  L = Sprache DE/EN
  V = Normal/Klinisch
  SPACE = Aufnahme starten/stoppen
  Q = Beenden
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import yaml, json, time, math, datetime
from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
CAM_L      = 0
CAM_R      = 2
RES        = (1280, 720)
FPS        = 60
MODEL_PATH = Path("~/anthro3d/pose_landmarker.task").expanduser()
SAVE_DIR   = Path("~/anthro3d/sessions").expanduser()

# ArUco
ARUCO_DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS   = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
ARUCO_ID       = 1       # Marker ID Patient
ARUCO_SIZE_CM  = 10.0    # Marker Größe in cm

# ── FARBEN (ANTHRO3D Design) ───────────────────────────────────────────────────
C = {
    'bg':       (242, 245, 243),   # Hintergrund
    'panel':    (255, 255, 255),   # Panel
    'border':   (212, 224, 214),   # Rand
    'g1':       (80,  107, 61),    # Grün dunkel
    'g2':       (90,  138, 74),    # Grün mittel
    'g3':       (106, 154, 90),    # Grün hell
    'text':     (26,  46,  32),    # Text
    'dim':      (96,  122, 100),   # Gedimmt
    'muted':    (176, 200, 176),   # Gemuted
    'red':      (64,  64,  192),   # Rot (BGR)
    'blue':     (160, 96,  48),    # Blau (BGR)
    'white':    (255, 255, 255),
    'black':    (0,   0,   0),
    'skel':     (80,  180, 80),    # Skelett grün
    'axis_s':   (220, 88,  68),    # Schulterachse (BGR)
    'axis_h':   (200, 80,  160),   # Beckenachse
    'axis_k':   (80,  180, 80),    # Knieachse
    'lot':      (48,  128, 192),   # Körperlot
}

# ── LAYOUT ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 780
CAM_X,  CAM_Y  = 240, 0
CAM_W,  CAM_H  = 800, 580
PANEL_L_W = 240
PANEL_R_X = CAM_X + CAM_W
PANEL_R_W = WIN_W - PANEL_R_X
BAR_Y     = CAM_H
BAR_H     = WIN_H - CAM_H

# MediaPipe Landmark Indizes
LM = {
    "nose":0,"l_eye":2,"r_eye":5,"l_ear":7,"r_ear":8,
    "l_shoulder":11,"r_shoulder":12,"l_elbow":13,"r_elbow":14,
    "l_wrist":15,"r_wrist":16,"l_hip":23,"r_hip":24,
    "l_knee":25,"r_knee":26,"l_ankle":27,"r_ankle":28,
}

SKEL_CONNS = [
    ("l_shoulder","r_shoulder"),("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
    ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
    ("l_shoulder","l_hip"),("r_shoulder","r_hip"),("l_hip","r_hip"),
    ("l_hip","l_knee"),("l_knee","l_ankle"),("r_hip","r_knee"),("r_knee","r_ankle"),
]

LABELS = {
    'de': {
        'height':'Körpergröße','shoulder_w':'Schulterbreite','hip_w':'Hüftbreite',
        'arm_l':'Oberarm L','arm_r':'Oberarm R','forearm_l':'Unterarm L','forearm_r':'Unterarm R',
        'thigh_l':'Oberschenkel L','thigh_r':'Oberschenkel R',
        'shin_l':'Unterschenkel L','shin_r':'Unterschenkel R','torso':'Rumpf',
        'shoulder_ang':'Schulterachse','hip_ang':'Beckenachse','knee_ang':'Knieachse',
    },
    'en': {
        'height':'Body Height','shoulder_w':'Shoulder Width','hip_w':'Hip Width',
        'arm_l':'Upper Arm L','arm_r':'Upper Arm R','forearm_l':'Forearm L','forearm_r':'Forearm R',
        'thigh_l':'Thigh L','thigh_r':'Thigh R',
        'shin_l':'Shin L','shin_r':'Shin R','torso':'Torso',
        'shoulder_ang':'Shoulder Axis','hip_ang':'Pelvis Axis','knee_ang':'Knee Axis',
    }
}

# ── KALMAN ────────────────────────────────────────────────────────────────────
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.processNoiseCov   = np.eye(4,dtype=np.float32)*5e-3
        self.kf.measurementNoiseCov = np.eye(2,dtype=np.float32)*2e-1
        self.kf.errorCovPost      = np.eye(4,dtype=np.float32)
        self.init = False
    def update(self,x,y):
        m=np.array([[x],[y]],np.float32)
        if not self.init:
            self.kf.statePre=self.kf.statePost=np.array([[x],[y],[0],[0]],np.float32)
            self.init=True
        self.kf.predict(); e=self.kf.correct(m)
        return float(e[0]),float(e[1])
    def reset(self): self.init=False

class Smoother:
    def __init__(self): self.enabled=False; self.filters={}
    def toggle(self): self.enabled=not self.enabled
    def smooth(self,name,x,y,vis):
        if vis<0.4: 
            if name in self.filters: self.filters[name].reset()
            return None
        if not self.enabled: return (x,y)
        if name not in self.filters: self.filters[name]=KalmanFilter2D()
        return self.filters[name].update(x,y)

# ── BUTTON KLASSE ─────────────────────────────────────────────────────────────
class Button:
    def __init__(self,x,y,w,h,label,toggle=False,active=True):
        self.x,self.y,self.w,self.h=x,y,w,h
        self.label=label; self.toggle=toggle; self.active=active
    def hit(self,mx,my): return self.x<=mx<=self.x+self.w and self.y<=my<=self.y+self.h
    def draw(self,canvas):
        col_bg  = C['g2'] if self.active else C['border']
        col_txt = C['white'] if self.active else C['dim']
        cv2.rectangle(canvas,(self.x,self.y),(self.x+self.w,self.y+self.h),col_bg,-1)
        cv2.rectangle(canvas,(self.x,self.y),(self.x+self.w,self.y+self.h),C['g1'],1)
        tw,th = cv2.getTextSize(self.label,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)[0]
        tx = self.x+(self.w-tw)//2; ty = self.y+(self.h+th)//2
        cv2.putText(canvas,self.label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.38,col_txt,1,cv2.LINE_AA)

# ── HAUPTAPP ──────────────────────────────────────────────────────────────────
class ANTHRO3DApp:
    def __init__(self):
        self.lang      = 'de'
        self.smoother  = Smoother()
        self.show_skel = True
        self.show_axis = True
        self.show_lot  = True
        self.show_seg  = True
        self.show_ang  = True
        self.clinical  = False
        self.px_per_cm = None
        self.meas      = {}
        self.pts       = {}
        self.fps_val   = 0.0
        self.frame_n   = 0
        self.fps_t     = time.time()
        self.recording = False
        self.rec_start = None
        self.frames    = []
        self.landmarker = None
        self.cap_l = self.cap_r = None
        self.timestamp_ms = 0
        self.canvas    = np.zeros((WIN_H,WIN_W,3),np.uint8)
        self._init_buttons()

    def _init_buttons(self):
        # Linkes Panel — Toggle Buttons
        bx,by,bw,bh,gap = 10,120,100,22,28
        self.btn_skel  = Button(bx,by,     bw,bh,"Skelett",   True,True)
        self.btn_axis  = Button(bx,by+gap, bw,bh,"Achsen",    True,True)
        self.btn_lot   = Button(bx,by+gap*2,bw,bh,"Koerperlot",True,True)
        self.btn_seg   = Button(bx,by+gap*3,bw,bh,"Laengen",   True,True)
        self.btn_ang   = Button(bx,by+gap*4,bw,bh,"Winkel",    True,True)
        self.btn_clin  = Button(bx,by+gap*5,bw,bh,"Klinisch",  True,False)
        self.btn_kalm  = Button(bx,by+gap*6,bw,bh,"Kalman",    True,False)
        self.btn_lang  = Button(bx,by+gap*7,bw,bh,"DE/EN",     False,True)
        # Bottom Bar
        self.btn_rec   = Button(CAM_X+CAM_W//2-80,BAR_Y+15,160,40,"● Aufnahme starten",False,True)
        self.all_btns  = [self.btn_skel,self.btn_axis,self.btn_lot,self.btn_seg,
                          self.btn_ang,self.btn_clin,self.btn_kalm,self.btn_lang,self.btn_rec]

    def init_landmarker(self):
        if not MODEL_PATH.exists():
            print(f"FEHLER: {MODEL_PATH} nicht gefunden")
            return False
        opts = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = PoseLandmarker.create_from_options(opts)
        print("  PoseLandmarker ✓")
        return True

    def connect_cams(self):
        self.cap_l = cv2.VideoCapture(CAM_L)
        self.cap_r = cv2.VideoCapture(CAM_R)
        for cap,name,idx in [(self.cap_l,"L",CAM_L),(self.cap_r,"R",CAM_R)]:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,RES[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,RES[1])
                cap.set(cv2.CAP_PROP_FPS,FPS)
                print(f"  Kamera {name} (Index {idx}) ✓")
            else:
                print(f"  Kamera {name} (Index {idx}) FEHLER")

    def grab(self):
        if not self.cap_l or not self.cap_l.isOpened(): return None
        self.cap_l.grab()
        if self.cap_r and self.cap_r.isOpened(): self.cap_r.grab()
        _,frame = self.cap_l.retrieve()
        return frame

    def detect_aruco(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_ = ARUCO_DETECTOR.detectMarkers(gray)
        if ids is None: return
        for i,mid in enumerate(ids.flatten()):
            if mid == ARUCO_ID:
                c = corners[i][0]
                px_size = (np.linalg.norm(c[1]-c[0])+np.linalg.norm(c[3]-c[0]))/2
                self.px_per_cm = px_size / ARUCO_SIZE_CM
                # Marker einzeichnen
                cv2.aruco.drawDetectedMarkers(frame,[corners[i]],np.array([[mid]]))

    def get_landmarks(self,frame):
        self.timestamp_ms += 33
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        res = self.landmarker.detect_for_video(img,self.timestamp_ms)
        if res.pose_landmarks and len(res.pose_landmarks)>0:
            return res.pose_landmarks[0]
        return None

    def extract_pts(self,landmarks,w,h):
        pts = {}
        if landmarks is None: return pts
        for name,idx in LM.items():
            if idx>=len(landmarks): continue
            lm = landmarks[idx]
            pt = self.smoother.smooth(name,lm.x*w,lm.y*h,lm.visibility)
            if pt: pts[name]=pt
        return pts

    def compute(self,pts):
        def seg(a,b):
            if not pts.get(a) or not pts.get(b): return None
            if self.px_per_cm:
                d=math.sqrt((pts[a][0]-pts[b][0])**2+(pts[a][1]-pts[b][1])**2)
                return round(d/self.px_per_cm,1)
            return None
        def ang(a,b):
            if not pts.get(a) or not pts.get(b): return None
            return round(abs(math.degrees(math.atan2(pts[b][1]-pts[a][1],pts[b][0]-pts[a][0]))),1)

        sl,sr = pts.get('l_shoulder'),pts.get('r_shoulder')
        hl,hr = pts.get('l_hip'),pts.get('r_hip')
        nose  = pts.get('nose')
        ankle = pts.get('l_ankle') or pts.get('r_ankle')

        height = None
        if nose and ankle and self.px_per_cm:
            height = round(abs(nose[1]-ankle[1])/self.px_per_cm*1.06,1)

        return {
            'height':height,
            'shoulder_w':seg('l_shoulder','r_shoulder'),
            'hip_w':seg('l_hip','r_hip'),
            'arm_l':seg('l_shoulder','l_elbow'),
            'arm_r':seg('r_shoulder','r_elbow'),
            'forearm_l':seg('l_elbow','l_wrist'),
            'forearm_r':seg('r_elbow','r_wrist'),
            'thigh_l':seg('l_hip','l_knee'),
            'thigh_r':seg('r_hip','r_knee'),
            'shin_l':seg('l_knee','l_ankle'),
            'shin_r':seg('r_knee','r_ankle'),
            'torso':seg('l_shoulder','l_hip'),
            'shoulder_ang':ang(sl,sr),
            'hip_ang':ang(hl,hr),
            'knee_ang':ang(pts.get('l_knee'),pts.get('r_knee')),
        }

    def update_fps(self):
        self.frame_n+=1
        now=time.time()
        if now-self.fps_t>=1.0:
            self.fps_val=self.frame_n/(now-self.fps_t)
            self.frame_n=0; self.fps_t=now

    # ── ZEICHNEN ──────────────────────────────────────────────────────────────
    def draw_frame(self,cam_frame):
        canvas = np.full((WIN_H,WIN_W,3),C['bg'],np.uint8)

        # ── Kamerabild ────────────────────────────────────────────────────────
        if cam_frame is not None:
            h,w = cam_frame.shape[:2]
            # Klinischer Modus
            if self.clinical:
                gray = cv2.cvtColor(cam_frame,cv2.COLOR_BGR2GRAY)
                g3   = cv2.merge([gray,gray,gray])
                # Auf ANTHRO3D Palette mappen
                lut  = np.zeros((256,1,3),np.uint8)
                for i in range(256):
                    t=i/255
                    lut[i,0,0]=int(195+t*33)
                    lut[i,0,1]=int(210+t*25)
                    lut[i,0,2]=int(198+t*30)
                display = cv2.LUT(g3,lut.reshape(256,3))
            else:
                display = cam_frame.copy()

            # Skalieren auf CAM-Bereich
            disp_scaled = cv2.resize(display,(CAM_W,CAM_H))
            canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = disp_scaled

            # Scale-Faktor für Landmark-Koordinaten
            sx = CAM_W/w; sy = CAM_H/h

            # ── Skelett ───────────────────────────────────────────────────────
            if self.show_skel and self.pts:
                def cp(name):
                    if name not in self.pts: return None
                    x,y = self.pts[name]
                    return (int(x*sx)+CAM_X, int(y*sy)+CAM_Y)

                for a,b in SKEL_CONNS:
                    pa,pb=cp(a),cp(b)
                    if pa and pb:
                        cv2.line(canvas,pa,pb,C['skel'],2,cv2.LINE_AA)
                for name in LM:
                    p=cp(name)
                    if p:
                        cv2.circle(canvas,p,4,C['g2'],-1)
                        cv2.circle(canvas,p,4,C['white'],1)

                # ── Achsenlinien ──────────────────────────────────────────────
                if self.show_axis:
                    sl,sr = cp('l_shoulder'),cp('r_shoulder')
                    hl,hr = cp('l_hip'),cp('r_hip')
                    kl,kr = cp('l_knee'),cp('r_knee')

                    def draw_axis(a,b,col,ang_val):
                        if not a or not b: return
                        # Weiße Outline
                        cv2.line(canvas,a,b,(255,255,255),5,cv2.LINE_AA)
                        # Farbige Linie
                        cv2.line(canvas,a,b,col,2,cv2.LINE_AA)
                        if self.show_ang and ang_val is not None:
                            mx,my=(a[0]+b[0])//2,(a[1]+b[1])//2
                            txt=f"{ang_val:.1f}"
                            cv2.putText(canvas,txt,(mx+4,my-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1,cv2.LINE_AA)

                    draw_axis(sl,sr,C['axis_s'],self.meas.get('shoulder_ang'))
                    draw_axis(hl,hr,C['axis_h'],self.meas.get('hip_ang'))
                    draw_axis(kl,kr,C['axis_k'],self.meas.get('knee_ang'))

                # ── Körperlot ─────────────────────────────────────────────────
                if self.show_lot:
                    nose=cp('nose'); ankle=cp('l_ankle') or cp('r_ankle')
                    if nose and ankle:
                        cx=(nose[0]+ankle[0])//2
                        cv2.line(canvas,(cx,CAM_Y),(cx,CAM_Y+CAM_H),C['lot'],1,cv2.LINE_AA)

                # ── Segmentlängen ─────────────────────────────────────────────
                if self.show_seg and self.px_per_cm:
                    segs=[('l_shoulder','l_elbow','arm_l'),('r_shoulder','r_elbow','arm_r'),
                          ('l_hip','l_knee','thigh_l'),('r_hip','r_knee','thigh_r'),
                          ('l_knee','l_ankle','shin_l'),('r_knee','r_ankle','shin_r')]
                    for a,b,key in segs:
                        pa,pb=cp(a),cp(b)
                        val=self.meas.get(key)
                        if pa and pb and val:
                            mx,my=(pa[0]+pb[0])//2,(pa[1]+pb[1])//2
                            txt=f"{val:.0f}cm"
                            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
                            cv2.rectangle(canvas,(mx-2,my-th-2),(mx+tw+2,my+2),(255,255,255),-1)
                            cv2.putText(canvas,txt,(mx,my),cv2.FONT_HERSHEY_SIMPLEX,
                                0.38,C['g1'],1,cv2.LINE_AA)

        # ── Kamerarahmen ──────────────────────────────────────────────────────
        cv2.rectangle(canvas,(CAM_X,CAM_Y),(CAM_X+CAM_W,CAM_Y+CAM_H),C['border'],1)

        # ── Linkes Panel ──────────────────────────────────────────────────────
        cv2.rectangle(canvas,(0,0),(PANEL_L_W,WIN_H),C['panel'],-1)
        cv2.rectangle(canvas,(0,0),(PANEL_L_W,WIN_H),C['border'],1)

        # Logo
        cv2.putText(canvas,"ANTHRO3D",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,C['g1'],2,cv2.LINE_AA)
        cv2.putText(canvas,"v2.0",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.35,C['muted'],1,cv2.LINE_AA)

        # Status
        aruco_txt = f"ArUco {self.px_per_cm:.1f}px/cm" if self.px_per_cm else "ArUco: ID1 zeigen"
        aruco_col = C['g2'] if self.px_per_cm else C['red']
        cv2.putText(canvas,aruco_txt,(10,75),cv2.FONT_HERSHEY_SIMPLEX,0.32,aruco_col,1,cv2.LINE_AA)
        cv2.putText(canvas,f"{self.fps_val:.0f} fps",(160,75),cv2.FONT_HERSHEY_SIMPLEX,0.32,C['muted'],1,cv2.LINE_AA)

        # Abschnitt-Label
        cv2.putText(canvas,"ANSICHT",(10,108),cv2.FONT_HERSHEY_SIMPLEX,0.32,C['muted'],1,cv2.LINE_AA)

        # Buttons aktualisieren und zeichnen
        self.btn_skel.active = self.show_skel
        self.btn_axis.active = self.show_axis
        self.btn_lot.active  = self.show_lot
        self.btn_seg.active  = self.show_seg
        self.btn_ang.active  = self.show_ang
        self.btn_clin.active = self.clinical
        self.btn_kalm.active = self.smoother.enabled

        for btn in self.all_btns:
            btn.draw(canvas)

        # ── Rechtes Panel — Messwerte ──────────────────────────────────────────
        cv2.rectangle(canvas,(PANEL_R_X,0),(WIN_W,WIN_H),C['panel'],-1)
        cv2.rectangle(canvas,(PANEL_R_X,0),(WIN_W,WIN_H),C['border'],1)

        L = LABELS[self.lang]
        rx = PANEL_R_X + 8

        # Körpergröße prominent
        h_val = self.meas.get('height')
        cv2.putText(canvas,L['height'],(rx,28),cv2.FONT_HERSHEY_SIMPLEX,0.38,C['dim'],1,cv2.LINE_AA)
        h_txt = f"{h_val:.1f} cm" if h_val else "--- cm"
        cv2.putText(canvas,h_txt,(rx,58),cv2.FONT_HERSHEY_SIMPLEX,0.9,C['g1'],2,cv2.LINE_AA)
        cv2.line(canvas,(rx,68),(WIN_W-8,68),C['border'],1)

        # Messwerte Liste
        keys = [
            ('shoulder_w',None),('hip_w',None),
            ('arm_l','arm_r'),('forearm_l','forearm_r'),
            ('thigh_l','thigh_r'),('shin_l','shin_r'),
            ('torso',None),
        ]
        ry = 82
        for k1,k2 in keys:
            v1 = self.meas.get(k1)
            lbl1 = L.get(k1,k1)
            val1 = f"{v1:.1f}cm" if v1 else "---"
            cv2.putText(canvas,lbl1[:14],(rx,ry),cv2.FONT_HERSHEY_SIMPLEX,0.3,C['dim'],1,cv2.LINE_AA)
            cv2.putText(canvas,val1,(rx,ry+13),cv2.FONT_HERSHEY_SIMPLEX,0.38,C['g1'],1,cv2.LINE_AA)
            if k2:
                v2 = self.meas.get(k2)
                lbl2 = L.get(k2,k2)
                val2 = f"{v2:.1f}cm" if v2 else "---"
                cv2.putText(canvas,lbl2[:14],(rx+80,ry),cv2.FONT_HERSHEY_SIMPLEX,0.3,C['dim'],1,cv2.LINE_AA)
                cv2.putText(canvas,val2,(rx+80,ry+13),cv2.FONT_HERSHEY_SIMPLEX,0.38,C['g1'],1,cv2.LINE_AA)
            ry += 32
            cv2.line(canvas,(rx,ry-4),(WIN_W-8,ry-4),C['border'],1)

        # Winkel
        cv2.putText(canvas,"WINKEL",(rx,ry+10),cv2.FONT_HERSHEY_SIMPLEX,0.32,C['muted'],1,cv2.LINE_AA)
        ry += 24
        for key in ['shoulder_ang','hip_ang','knee_ang']:
            v = self.meas.get(key)
            lbl = L.get(key,key)
            val = f"{v:.1f}" if v else "---"
            cv2.putText(canvas,lbl[:16],(rx,ry),cv2.FONT_HERSHEY_SIMPLEX,0.3,C['dim'],1,cv2.LINE_AA)
            cv2.putText(canvas,val,(rx+110,ry),cv2.FONT_HERSHEY_SIMPLEX,0.38,C['blue'],1,cv2.LINE_AA)
            ry += 22

        # ── Bottom Bar ────────────────────────────────────────────────────────
        cv2.rectangle(canvas,(CAM_X,BAR_Y),(CAM_X+CAM_W,WIN_H),C['panel'],-1)
        cv2.rectangle(canvas,(CAM_X,BAR_Y),(CAM_X+CAM_W,WIN_H),C['border'],1)

        # Aufnahme-Button
        if self.recording:
            dur = time.time()-self.rec_start
            m,s = int(dur//60),int(dur%60)
            self.btn_rec.label = f"■ Stopp  {m}:{s:02d}"
            self.btn_rec.active = True
        else:
            self.btn_rec.label = "● Aufnahme starten"
            self.btn_rec.active = False
        self.btn_rec.draw(canvas)

        # Tastenkürzel
        keys_txt = "K=Kalman  S=Skelett  V=Klinisch  L=Sprache  SPACE=REC  Q=Beenden"
        cv2.putText(canvas,keys_txt,(CAM_X+10,WIN_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.3,C['muted'],1,cv2.LINE_AA)

        return canvas

    def on_click(self,event,mx,my,flags,param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.btn_skel.hit(mx,my): self.show_skel=not self.show_skel
        elif self.btn_axis.hit(mx,my): self.show_axis=not self.show_axis
        elif self.btn_lot.hit(mx,my):  self.show_lot=not self.show_lot
        elif self.btn_seg.hit(mx,my):  self.show_seg=not self.show_seg
        elif self.btn_ang.hit(mx,my):  self.show_ang=not self.show_ang
        elif self.btn_clin.hit(mx,my): self.clinical=not self.clinical
        elif self.btn_kalm.hit(mx,my): self.smoother.toggle()
        elif self.btn_lang.hit(mx,my): self.lang='en' if self.lang=='de' else 'de'
        elif self.btn_rec.hit(mx,my):  self.toggle_rec()

    def toggle_rec(self):
        self.recording = not self.recording
        if self.recording:
            self.rec_start = time.time()
            self.frames = []
            print("  ● REC gestartet")
        else:
            print(f"  ■ REC gestoppt — {len(self.frames)} Frames")

    def run(self):
        print("\n"+"="*52)
        print("  ANTHRO3D — Native App v2.0")
        print("="*52)
        print("\nLade Modell...")
        if not self.init_landmarker(): return
        print("\nVerbinde Kameras...")
        self.connect_cams()

        cv2.namedWindow("ANTHRO3D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ANTHRO3D", WIN_W, WIN_H)
        cv2.setMouseCallback("ANTHRO3D", self.on_click)

        print(f"\nBereit! Fenster: ANTHRO3D")

        while True:
            frame = self.grab()
            self.update_fps()

            if frame is not None:
                self.detect_aruco(frame)
                lm = self.get_landmarks(frame)
                self.pts  = self.extract_pts(lm, frame.shape[1], frame.shape[0])
                self.meas = self.compute(self.pts)

            canvas = self.draw_frame(frame)
            cv2.imshow("ANTHRO3D", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27): break
            elif key == ord('k'): self.smoother.toggle()
            elif key == ord('s'): self.show_skel=not self.show_skel
            elif key == ord('v'): self.clinical=not self.clinical
            elif key == ord('l'): self.lang='en' if self.lang=='de' else 'de'
            elif key == ord(' '): self.toggle_rec()

        cv2.destroyAllWindows()
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()
        if self.landmarker: self.landmarker.close()
        print("\nBeendet.")

if __name__ == "__main__":
    ANTHRO3DApp().run()
