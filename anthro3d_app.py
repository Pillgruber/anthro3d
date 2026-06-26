#!/usr/bin/env python3
"""
ANTHRO3D — Native Mac App (PyQt6)
===================================
Vollständige Desktop-App im ANTHRO3D Design.
Exakt gleiche Optik wie der Browser.

Starten:
  python3 anthro3d_app.py
"""

import sys, time, math, json, datetime
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
CAM_L      = 0
CAM_R      = 2
MODEL_PATH = Path("~/anthro3d/pose_landmarker.task").expanduser()
SAVE_DIR   = Path("~/anthro3d/sessions").expanduser()
ARUCO_ID   = 1
ARUCO_CM   = 10.0

# ANTHRO3D Farben
COLORS = {
    'bg':      '#f0f4f1',
    'surface': '#ffffff',
    'border':  '#d4e0d6',
    'g1':      '#3d6b50',
    'g2':      '#5a8a6a',
    'g3':      '#6a9a7a',
    'g4':      '#c2dbc8',
    'g5':      '#e8f3ea',
    'text':    '#1a2e20',
    'dim':     '#7a9880',
    'muted':   '#b0c8b4',
    'red':     '#c04040',
    'blue':    '#3060a0',
    'warn':    '#b87030',
}

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
        'shin_l':'Unterschenkel L','shin_r':'Unterschenkel R','torso':'Rumpflänge',
        'shoulder_ang':'Schulterachse','hip_ang':'Beckenachse','knee_ang':'Knieachse',
        'hws':'HWS','bws':'BWS','lws':'LWS',
        'lengths':'Segmentlängen','angles':'Achsenwinkel',
        'view':'Ansicht','meas':'Messwerte','quality':'Qualität',
    },
    'en': {
        'height':'Body Height','shoulder_w':'Shoulder Width','hip_w':'Hip Width',
        'arm_l':'Upper Arm L','arm_r':'Upper Arm R','forearm_l':'Forearm L','forearm_r':'Forearm R',
        'thigh_l':'Thigh L','thigh_r':'Thigh R',
        'shin_l':'Shin L','shin_r':'Shin R','torso':'Torso',
        'shoulder_ang':'Shoulder Axis','hip_ang':'Pelvis Axis','knee_ang':'Knee Axis',
        'hws':'Cervical','bws':'Thoracic','lws':'Lumbar',
        'lengths':'Segment Lengths','angles':'Axis Angles',
        'view':'View','meas':'Measurements','quality':'Quality',
    }
}

# ── KALMAN ────────────────────────────────────────────────────────────────────
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.processNoiseCov    = np.eye(4,dtype=np.float32)*5e-3
        self.kf.measurementNoiseCov= np.eye(2,dtype=np.float32)*2e-1
        self.kf.errorCovPost       = np.eye(4,dtype=np.float32)
        self.init=False
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

# ── KAMERA THREAD ─────────────────────────────────────────────────────────────
class MeshViewer3D(QWidget):
    """3D Mesh Viewer mit Kamera-Textur — frei drehbar."""
    def __init__(self, ply_path, texture_frame=None):
        super().__init__()
        self.setWindowTitle("ANTHRO3D — 3D Mesh Viewer")
        self.resize(1100, 750)
        self.setStyleSheet("background:#1a1a1a;")

        self.rot_x    = 15.0
        self.rot_y    = -20.0
        self.zoom     = 1.0
        self.last_pos = None
        self.show_texture = texture_frame is not None
        self.show_wire    = True

        # PLY + Textur laden
        self.pts, self.triangles, self.uvs = self._load_ply(ply_path)
        self.texture = self._prepare_texture(texture_frame)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # Toolbar
        bar = QHBoxLayout()
        bar.setContentsMargins(12,6,12,6)
        title = QLabel("🔲  3D Mesh Viewer")
        title.setStyleSheet("color:#1D9E75;font-weight:700;font-size:14px;")
        bar.addWidget(title)
        bar.addSpacing(20)

        # Ansicht Buttons
        for label, rx, ry in [("Frontal",0,0),("Seitlich",0,-90),("Dorsal",0,180),("Oben",-90,0)]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet(f"""QPushButton{{background:#2a2a2a;color:#aaa;border:1px solid #444;
                border-radius:4px;font-size:11px;padding:0 10px;}}
                QPushButton:hover{{background:#333;color:#fff;}}""")
            btn.clicked.connect(lambda _,rx=rx,ry=ry: self._set_view(rx,ry))
            bar.addWidget(btn)

        bar.addStretch()

        # Toggle Textur/Wire
        self.tex_btn = QPushButton("🖼 Textur AN")
        self.tex_btn.setFixedHeight(24)
        self.tex_btn.setCheckable(True)
        self.tex_btn.setChecked(self.show_texture)
        self.tex_btn.setStyleSheet("""QPushButton{background:#1D9E75;color:white;border:none;
            border-radius:4px;font-size:11px;padding:0 10px;}
            QPushButton:!checked{background:#2a2a2a;color:#aaa;border:1px solid #444;}""")
        self.tex_btn.clicked.connect(self._toggle_texture)
        bar.addWidget(self.tex_btn)

        wire_btn = QPushButton("🔲 Drahtgitter")
        wire_btn.setFixedHeight(24)
        wire_btn.setCheckable(True)
        wire_btn.setChecked(True)
        wire_btn.setStyleSheet("""QPushButton{background:#1D9E75;color:white;border:none;
            border-radius:4px;font-size:11px;padding:0 10px;}
            QPushButton:!checked{background:#2a2a2a;color:#aaa;border:1px solid #444;}""")
        wire_btn.clicked.connect(lambda v: setattr(self,'show_wire',v) or self._render())
        bar.addWidget(wire_btn)

        bar.addSpacing(12)
        info = QLabel("Maus: Drehen  |  Scroll: Zoom  |  ESC: Schließen")
        info.setStyleSheet("color:#555;font-size:10px;")
        bar.addWidget(info)

        bar_widget = QWidget()
        bar_widget.setLayout(bar)
        bar_widget.setStyleSheet("background:#111;")
        bar_widget.setFixedHeight(38)
        layout.addWidget(bar_widget)

        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas, 1)

        self.setMouseTracking(True)
        self._render()

    def _prepare_texture(self, frame):
        """Kamera-Frame als QImage Textur vorbereiten."""
        if frame is None: return None
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data.tobytes(), w, h, 3*w, QImage.Format.Format_RGB888).copy()

    def _load_ply(self, path):
        pts, cols = [], []
        with open(path) as f:
            hdr = True
            for line in f:
                if line.strip() == 'end_header': hdr = False; continue
                if hdr: continue
                v = line.strip().split()
                if len(v) >= 3:
                    pts.append([float(v[0]),float(v[1]),float(v[2])])
                    if len(v) >= 6:
                        cols.append([int(v[3]),int(v[4]),int(v[5])])

        pts = np.array(pts, dtype=np.float32)
        # Ausreißer entfernen
        mask = np.ones(len(pts), dtype=bool)
        for ax in range(3):
            m, s = pts[:,ax].mean(), pts[:,ax].std()
            mask &= np.abs(pts[:,ax]-m) < 2.5*s
        pts = pts[mask]

        # Zentrieren
        pts -= pts.mean(axis=0)
        pts /= max(np.abs(pts).max(), 1)

        # UV-Koordinaten aus XY (Frontalprojektion)
        x_min,x_max = pts[:,0].min(), pts[:,0].max()
        y_min,y_max = pts[:,1].min(), pts[:,1].max()
        uvs = np.zeros((len(pts),2), dtype=np.float32)
        uvs[:,0] = (pts[:,0]-x_min) / max(x_max-x_min, 0.001)
        uvs[:,1] = 1.0 - (pts[:,1]-y_min) / max(y_max-y_min, 0.001)

        # ConvexHull
        triangles, tri_uvs = None, None
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            triangles = pts[hull.simplices]
            tri_uvs   = uvs[hull.simplices]
        except Exception as e:
            print(f"Hull Fehler: {e}")

        return pts, triangles, tri_uvs

    def _set_view(self, rx, ry):
        self.rot_x, self.rot_y = rx, ry
        self._render()

    def _toggle_texture(self):
        self.show_texture = self.tex_btn.isChecked()
        self.tex_btn.setText("🖼 Textur AN" if self.show_texture else "🖼 Textur AUS")
        self._render()

    def _project(self, pts, W, H):
        import math
        rx,ry = math.radians(self.rot_x), math.radians(self.rot_y)
        p = pts.copy()
        # Rotation X
        cx,sx = math.cos(rx),math.sin(rx)
        y,z = p[:,1]*cx-p[:,2]*sx, p[:,1]*sx+p[:,2]*cx
        p[:,1],p[:,2] = y,z
        # Rotation Y
        cy,sy = math.cos(ry),math.sin(ry)
        x,z = p[:,0]*cy+p[:,2]*sy, -p[:,0]*sy+p[:,2]*cy
        p[:,0],p[:,2] = x,z
        # Perspektive
        scale = min(W,H)*self.zoom*0.38
        zz = np.maximum(p[:,2]+3, 0.1)
        sx2 = p[:,0]/zz*scale + W/2
        sy2 = -p[:,1]/zz*scale + H/2
        return np.stack([sx2,sy2,p[:,2]], axis=1)

    def _sample_texture(self, u, v):
        """Farbe aus Textur an UV-Koordinate."""
        if self.texture is None: return QColor(29,158,117,40)
        tw = self.texture.width()
        th = self.texture.height()
        px = int(min(max(u,0),1) * (tw-1))
        py = int(min(max(v,0),1) * (th-1))
        c = QColor(self.texture.pixel(px,py))
        c.setAlpha(220)
        return c

    def _render(self):
        W = max(self.canvas.width(), 860)
        H = max(self.canvas.height(), 660)
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(QColor(22,22,22))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.triangles is not None and len(self.triangles) > 0:
            proj = self._project(self.triangles.reshape(-1,3), W, H)
            tris     = proj.reshape(-1,3,3)
            tri_uvs  = self.uvs if self.uvs is not None else None

            # Painter's algorithm — Tiefe sortieren
            depths = tris[:,:,2].mean(axis=1)
            order  = np.argsort(-depths)

            for i in order:
                t = tris[i]
                poly = QPolygonF()
                for pt in t:
                    poly.append(QPointF(float(pt[0]), float(pt[1])))

                # Textur-Farbe aus UV-Zentrum
                if self.show_texture and tri_uvs is not None:
                    uv = tri_uvs[i].mean(axis=0)
                    face_col = self._sample_texture(float(uv[0]), float(uv[1]))
                else:
                    face_col = QColor(29,158,117,30)

                if self.show_wire:
                    painter.setPen(QPen(QColor(29,158,117,120), 0.4))
                else:
                    painter.setPen(Qt.PenStyle.NoPen)

                painter.setBrush(QBrush(face_col))
                painter.drawPolygon(poly)

        # Info
        painter.setPen(QColor(80,80,80))
        painter.setFont(QFont("Arial",9))
        painter.drawText(12, H-12, f"Dreiecke: {len(self.triangles) if self.triangles is not None else 0}  |  Rot: {self.rot_x:.0f}°/{self.rot_y:.0f}°  |  Zoom: {self.zoom:.1f}x")

        painter.end()
        self.canvas.setPixmap(QPixmap.fromImage(img))

    def resizeEvent(self, e):
        self.canvas.resize(self.width(), self.height()-38)
        self._render()

    def mousePressEvent(self, e):
        self.last_pos = e.position()

    def mouseMoveEvent(self, e):
        if self.last_pos and e.buttons():
            dx = e.position().x()-self.last_pos.x()
            dy = e.position().y()-self.last_pos.y()
            self.rot_y += dx*0.5
            self.rot_x += dy*0.5
            self.last_pos = e.position()
            self._render()

    def mouseReleaseEvent(self, e):
        self.last_pos = None

    def wheelEvent(self, e):
        self.zoom *= 1.1 if e.angleDelta().y()>0 else 0.9
        self.zoom = max(0.2, min(8.0, self.zoom))
        self._render()

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape: self.close()


class Video3DRecorder:
    """Nimmt Frames + Tiefenkarten auf und baut eine 3D Video Sequenz."""
    def __init__(self):
        self.frames    = []
        self.recording = False
        self._lock     = __import__('threading').Lock()
        self._stereo_ready = False

    def start(self):
        with self._lock:
            self.frames    = []
            self.recording = True

    def stop(self):
        with self._lock:
            self.recording = False

    def add_frame(self, frame, pts):
        if not self.recording: return
        with self._lock:
            self.frames.append({'frame': frame.copy(), 'pts': dict(pts) if pts else {}, 'depth': None})

    def _compute_depth(self, frame):
        try:
            import yaml
            cfg_path = Path("~/anthro3d/stereo_config.yaml").expanduser()
            if not cfg_path.exists(): return None
            if not self._stereo_ready:
                with open(cfg_path) as f: cfg = yaml.safe_load(f)
                K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
                K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
                R=np.array(cfg['R']); T=np.array(cfg['T'])
                sz=(1280,720)
                R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,sz,R,T,alpha=0.5)
                self._map_l1,self._map_l2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,sz,cv2.CV_32F)
                self._map_r1,self._map_r2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,sz,cv2.CV_32F)
                self._K_l=K_l; self._baseline=cfg.get('baseline_mm',68.7)
                self._sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=7,
                    P1=8*3*49,P2=32*3*49,disp12MaxDiff=1,uniquenessRatio=10,
                    speckleWindowSize=100,speckleRange=32,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                self._stereo_ready = True
            cap_r = cv2.VideoCapture(2)
            ret, fr = cap_r.read(); cap_r.release()
            if not ret: return None
            fl=cv2.remap(frame,self._map_l1,self._map_l2,cv2.INTER_LINEAR)
            fr=cv2.remap(fr,self._map_r1,self._map_r2,cv2.INTER_LINEAR)
            gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY); gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            return np.clip(self._sgbm.compute(gl,gr).astype(np.float32)/16.0, 0, None)
        except Exception:
            return None

    def build_pointcloud(self, idx):
        if idx >= len(self.frames): return None, None
        f = self.frames[idx]
        depth = f['depth']; frame = f['frame']
        if depth is None or not self._stereo_ready: return None, None
        try:
            fx=self._K_l[0,0]; cx=self._K_l[0,2]; cy_=self._K_l[1,2]
            H,W=depth.shape; pts,cols=[],[]
            for y in range(0,H,4):
                for x in range(0,W,4):
                    d=depth[y,x]
                    if d<1: continue
                    Z=(fx*self._baseline/10)/d
                    if Z<10 or Z>400: continue
                    pts.append([(x-cx)*Z/fx,(y-cy_)*Z/fx,Z])
                    b,g,r=frame[y,x] if len(frame.shape)==3 else (frame[y,x],)*3
                    cols.append([int(r),int(g),int(b)])
            return (np.array(pts,dtype=np.float32) if pts else None), cols
        except Exception:
            return None, None

    def save(self):
        if not self.frames: return None
        base=Path("~/anthro3d/video3d").expanduser(); base.mkdir(exist_ok=True)
        ts=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        folder=base/ts; folder.mkdir(exist_ok=True)
        saved=0
        for i,f in enumerate(self.frames):
            cv2.imwrite(str(folder/f"frame_{i:04d}.png"), f['frame'])
            pts,cols=self.build_pointcloud(i)
            if pts is not None and len(pts)>0:
                ply=folder/f"depth_{i:04d}.ply"
                with open(ply,'w') as fp:
                    fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                             "property float x\nproperty float y\nproperty float z\n"
                             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                    for p,c in zip(pts,cols): fp.write(f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {c[0]} {c[1]} {c[2]}\n")
                saved+=1
        print(f"Gespeichert: {folder} ({saved} PLY, {len(self.frames)} Frames)")
        return str(folder)


class Video3DViewer(QWidget):
    """3D Video Viewer — frei drehbar, Play/Pause, Scrubber."""
    def __init__(self, recorder):
        super().__init__()
        self.recorder=recorder; self.frames=recorder.frames
        self.n_frames=len(self.frames); self.cur_idx=0
        self.playing=False; self.rot_x=15.0; self.rot_y=-20.0
        self.zoom=1.0; self.last_pos=None; self._cache={}
        self.setWindowTitle(f"ANTHRO3D — 3D Video ({self.n_frames} Frames)")
        self.resize(1100,780); self.setStyleSheet("background:#1a1a1a;")
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)

        # Toolbar
        bar=QHBoxLayout(); bar.setContentsMargins(12,6,12,6)
        t=QLabel("🎬  3D Video Viewer"); t.setStyleSheet("color:#1D9E75;font-weight:700;font-size:14px;")
        bar.addWidget(t); bar.addSpacing(16)
        for lbl,rx,ry in [("Frontal",0,0),("Seitlich",0,-90),("Dorsal",0,180),("Oben",-90,0)]:
            btn=QPushButton(lbl); btn.setFixedHeight(24)
            btn.setStyleSheet("QPushButton{background:#2a2a2a;color:#aaa;border:1px solid #444;"
                              "border-radius:4px;font-size:11px;padding:0 10px;}"
                              "QPushButton:hover{background:#333;color:#fff;}")
            btn.clicked.connect(lambda _,rx=rx,ry=ry: self._set_view(rx,ry))
            bar.addWidget(btn)
        bar.addStretch()
        self.frame_lbl=QLabel(f"Frame 0/{self.n_frames-1}")
        self.frame_lbl.setStyleSheet("color:#666;font-size:11px;")
        bar.addWidget(self.frame_lbl); bar.addSpacing(8)
        sb=QPushButton("💾 Speichern"); sb.setFixedHeight(24)
        sb.setStyleSheet("QPushButton{background:#3d6b50;color:white;border:none;"
                         "border-radius:4px;font-size:11px;padding:0 12px;}"
                         "QPushButton:hover{background:#5a8a6a;}")
        sb.clicked.connect(self._save); bar.addWidget(sb)
        bw=QWidget(); bw.setLayout(bar); bw.setStyleSheet("background:#111;"); bw.setFixedHeight(38)
        layout.addWidget(bw)

        self.canvas=QLabel(); self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas,1)

        # Controls
        ctrl=QHBoxLayout(); ctrl.setContentsMargins(12,6,12,6)
        self.play_btn=QPushButton("▶ Play"); self.play_btn.setFixedSize(80,28)
        self.play_btn.setStyleSheet("QPushButton{background:#1D9E75;color:white;border:none;"
                                    "border-radius:4px;font-size:12px;}"
                                    "QPushButton:hover{background:#5a8a6a;}")
        self.play_btn.clicked.connect(self._toggle_play); ctrl.addWidget(self.play_btn)
        ctrl.addSpacing(8)
        self.slider=QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,max(self.n_frames-1,1)); self.slider.setValue(0)
        self.slider.setStyleSheet("QSlider::groove:horizontal{background:#333;height:4px;border-radius:2px;}"
                                  "QSlider::handle:horizontal{background:#1D9E75;width:14px;height:14px;"
                                  "margin:-5px 0;border-radius:7px;}"
                                  "QSlider::sub-page:horizontal{background:#1D9E75;border-radius:2px;}")
        self.slider.valueChanged.connect(self._on_slider); ctrl.addWidget(self.slider,1)
        cw=QWidget(); cw.setLayout(ctrl); cw.setStyleSheet("background:#111;"); cw.setFixedHeight(44)
        layout.addWidget(cw)

        self._timer=QTimer(); self._timer.timeout.connect(self._next_frame); self._timer.setInterval(50)
        self.setMouseTracking(True); self._render()

    def _set_view(self,rx,ry): self.rot_x,self.rot_y=rx,ry; self._render()

    def _toggle_play(self):
        self.playing=not self.playing
        self.play_btn.setText("⏸ Pause" if self.playing else "▶ Play")
        self._timer.start() if self.playing else self._timer.stop()

    def _next_frame(self):
        self.cur_idx=(self.cur_idx+1)%self.n_frames
        self.slider.blockSignals(True); self.slider.setValue(self.cur_idx); self.slider.blockSignals(False)
        self._render()

    def _on_slider(self,val): self.cur_idx=val; self._render()

    def _get_pts(self,idx):
        if idx not in self._cache: self._cache[idx]=self.recorder.build_pointcloud(idx)
        return self._cache[idx]

    def _project(self,pts,W,H):
        import math
        rx,ry=math.radians(self.rot_x),math.radians(self.rot_y)
        p=pts.copy()
        cx_,sx=math.cos(rx),math.sin(rx)
        y,z=p[:,1]*cx_-p[:,2]*sx,p[:,1]*sx+p[:,2]*cx_; p[:,1],p[:,2]=y,z
        cy_,sy=math.cos(ry),math.sin(ry)
        x,z=p[:,0]*cy_+p[:,2]*sy,-p[:,0]*sy+p[:,2]*cy_; p[:,0],p[:,2]=x,z
        sc=min(W,H)*self.zoom*0.38; zz=np.maximum(p[:,2]+3,0.1)
        return np.stack([p[:,0]/zz*sc+W/2,-p[:,1]/zz*sc+H/2,p[:,2]],axis=1)

    def _render(self):
        W=max(self.canvas.width(),860); H=max(self.canvas.height(),600)
        img=QImage(W,H,QImage.Format.Format_RGB32); img.fill(QColor(22,22,22))
        painter=QPainter(img); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        f=self.frames[self.cur_idx]; frame=f['frame']


        # Kamerabild als Vollbild-Hintergrund
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
            rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            fh, fw = rgb.shape[:2]
            scale = min(W/fw, H/fh)
            nw, nh = int(fw*scale), int(fh*scale)
            ox, oy = (W-nw)//2, (H-nh)//2
            rs = cv2.resize(rgb, (nw, nh))
            qi = QImage(rs.data.tobytes(), nw, nh, 3*nw, QImage.Format.Format_RGB888)
            painter.drawImage(QPointF(float(ox), float(oy)), qi)

        pts,cols=self._get_pts(self.cur_idx)
        if pts is not None and len(pts)>0:
            pn=pts.copy(); pn-=pn.mean(axis=0); mx=np.abs(pn).max()
            if mx>0: pn/=mx
            proj=self._project(pn,W,H)
            step=max(1,len(proj)//4000)
            for i in range(0,len(proj),step):
                px_,py_=proj[i,0],proj[i,1]
                if 0<=px_<W and 0<=py_<H:
                    c=QColor(29,200,120,180)
                    painter.setPen(QPen(c,2)); painter.drawPoint(QPointF(float(px_),float(py_)))

        painter.setPen(QColor(70,70,70)); painter.setFont(QFont("Arial",9))
        painter.drawText(12,H-8,f"Frame {self.cur_idx}/{self.n_frames-1}  |  "
                         f"Rot {self.rot_x:.0f}°/{self.rot_y:.0f}°  |  Zoom {self.zoom:.1f}x")
        painter.end(); self.canvas.setPixmap(QPixmap.fromImage(img))
        self.frame_lbl.setText(f"Frame {self.cur_idx} / {self.n_frames-1}")

    def _save(self):
        p=self.recorder.save()
        if p: self.setWindowTitle(f"ANTHRO3D — Gespeichert: {p}")

    def resizeEvent(self,e): self.canvas.resize(self.width(),self.height()-82); self._render()
    def mousePressEvent(self,e): self.last_pos=e.position()
    def mouseMoveEvent(self,e):
        if self.last_pos and e.buttons():
            dx=e.position().x()-self.last_pos.x(); dy=e.position().y()-self.last_pos.y()
            self.rot_y+=dx*0.5; self.rot_x+=dy*0.5; self.last_pos=e.position(); self._render()
    def mouseReleaseEvent(self,e): self.last_pos=None
    def wheelEvent(self,e):
        self.zoom*=1.1 if e.angleDelta().y()>0 else 0.9
        self.zoom=max(0.2,min(8.0,self.zoom)); self._render()
    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_Escape: self.close()
        elif e.key()==Qt.Key.Key_Space: self._toggle_play()
        elif e.key()==Qt.Key.Key_Right: self._on_slider(min(self.cur_idx+1,self.n_frames-1))
        elif e.key()==Qt.Key.Key_Left: self._on_slider(max(self.cur_idx-1,0))


class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object, object)  # frame, pts, meas

    def __init__(self):
        super().__init__()
        self.running    = True
        self.smoother   = Smoother()
        self.px_per_cm  = None
        self.landmarker = None
        self.cap_l = self.cap_r = None
        self.timestamp_ms = 0
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_det    = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.fps_val   = 0.0
        self.frame_n   = 0
        self.fps_t     = time.time()

    def init(self):
        # Landmarker
        opts = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = PoseLandmarker.create_from_options(opts)
        # Kameras
        self.cap_l = cv2.VideoCapture(CAM_L)
        self.cap_r = cv2.VideoCapture(CAM_R)
        for cap,idx in [(self.cap_l,CAM_L),(self.cap_r,CAM_R)]:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                cap.set(cv2.CAP_PROP_FPS,60)
        # Entzerrungsmaps laden
        self.undist_map_l1 = self.undist_map_l2 = None
        self.undist_map_r1 = self.undist_map_r2 = None
        try:
            import yaml
            with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
                scfg = yaml.safe_load(f)
            K_l  = np.array(scfg['camera_matrix_l'])
            d_l  = np.array(scfg['dist_l'])
            K_r  = np.array(scfg['camera_matrix_r'])
            d_r  = np.array(scfg['dist_r'])
            sz   = tuple(scfg.get('img_size', [1280, 720]))
            # Optimale Kameramatrix berechnen (beschneidet schwarze Ränder)
            K_l_opt, _ = cv2.getOptimalNewCameraMatrix(K_l, d_l, sz, 0.5)
            K_r_opt, _ = cv2.getOptimalNewCameraMatrix(K_r, d_r, sz, 0.5)
            self.undist_map_l1, self.undist_map_l2 = cv2.initUndistortRectifyMap(
                K_l, d_l, None, K_l_opt, sz, cv2.CV_32F)
            self.undist_map_r1, self.undist_map_r2 = cv2.initUndistortRectifyMap(
                K_r, d_r, None, K_r_opt, sz, cv2.CV_32F)
            print("  Entzerrung geladen ✓")
        except Exception as e:
            print(f"  Entzerrung nicht verfügbar: {e}")

    def undistort(self, frame, map1, map2):
        """Fisheye-Verzerrung entfernen."""
        if frame is None or map1 is None: return frame
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    def detect_aruco(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners,ids,_ = self.aruco_det.detectMarkers(gray)
        if ids is None: return
        for i,mid in enumerate(ids.flatten()):
            if mid==ARUCO_ID:
                c=corners[i][0]
                px=(np.linalg.norm(c[1]-c[0])+np.linalg.norm(c[3]-c[0]))/2
                self.px_per_cm = px/ARUCO_CM

    def get_landmarks(self,frame):
        self.timestamp_ms += 33
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        res = self.landmarker.detect_for_video(img,self.timestamp_ms)
        if res.pose_landmarks and len(res.pose_landmarks)>0:
            return res.pose_landmarks[0]
        return None

    def extract_pts(self,lm,w,h):
        pts={}
        if lm is None: return pts
        for name,idx in LM.items():
            if idx>=len(lm): continue
            l=lm[idx]
            pt=self.smoother.smooth(name,l.x*w,l.y*h,l.visibility)
            if pt: pts[name]=pt
        return pts

    def compute(self,pts):
        def seg(a,b):
            if not pts.get(a) or not pts.get(b) or not self.px_per_cm: return None
            d=math.sqrt((pts[a][0]-pts[b][0])**2+(pts[a][1]-pts[b][1])**2)
            return round(d/self.px_per_cm,1)
        def ang(a,b):
            if not pts.get(a) or not pts.get(b): return None
            return round(abs(math.degrees(math.atan2(pts[b][1]-pts[a][1],pts[b][0]-pts[a][0]))),1)
        sl,sr=pts.get('l_shoulder'),pts.get('r_shoulder')
        hl,hr=pts.get('l_hip'),pts.get('r_hip')
        nose=pts.get('nose'); ankle=pts.get('l_ankle') or pts.get('r_ankle')
        height=None
        if nose and ankle and self.px_per_cm:
            height=round(abs(nose[1]-ankle[1])/self.px_per_cm*1.06,1)
        return {
            'height':height,'shoulder_w':seg('l_shoulder','r_shoulder'),
            'hip_w':seg('l_hip','r_hip'),'arm_l':seg('l_shoulder','l_elbow'),
            'arm_r':seg('r_shoulder','r_elbow'),'forearm_l':seg('l_elbow','l_wrist'),
            'forearm_r':seg('r_elbow','r_wrist'),'thigh_l':seg('l_hip','l_knee'),
            'thigh_r':seg('r_hip','r_knee'),'shin_l':seg('l_knee','l_ankle'),
            'shin_r':seg('r_knee','r_ankle'),'torso':seg('l_shoulder','l_hip'),
            'shoulder_ang':ang(sl,sr),'hip_ang':ang(hl,hr),
            'knee_ang':ang(pts.get('l_knee'),pts.get('r_knee')),
        }

    def update_fps(self):
        self.frame_n+=1
        now=time.time()
        if now-self.fps_t>=1.0:
            self.fps_val=self.frame_n/(now-self.fps_t)
            self.frame_n=0; self.fps_t=now

    def normalize_frame(self, frame):
        """Histogram-Normalisierung für bessere Erkennung bei wechselndem Licht."""
        if frame is None: return frame
        # In LAB konvertieren — nur L-Kanal (Helligkeit) normalisieren
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE — Contrast Limited Adaptive Histogram Equalization
        if not hasattr(self, '_clahe'):
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def apply_clinical_background(self, frame, pts):
        """Person vom Hintergrund trennen — custom PNG Hintergrund."""
        if frame is None: return frame
        H, W = frame.shape[:2]

        # Graustufen zu BGR
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Hintergrund laden (einmal cachen)
        if not hasattr(self, '_bg_cache') or self._bg_cache is None:
            bg_path = Path("~/anthro3d/background.png").expanduser()
            if bg_path.exists():
                bg_raw = cv2.imread(str(bg_path))
                self._bg_cache = cv2.resize(bg_raw, (W, H))
            else:
                # Fallback: reines Weiß
                self._bg_cache = np.full((H, W, 3), 248, dtype=np.uint8)

        bg = self._bg_cache
        if bg.shape[:2] != (H, W):
            bg = cv2.resize(bg, (W, H))

        # Segmentierungsmaske aus Landmarks
        if pts and len(pts) >= 8:
            lm_pts = []
            for pt in pts.values():
                if pt is not None and len(pt) >= 2:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < W and 0 <= y < H:
                        lm_pts.append([x, y])

            if len(lm_pts) >= 5:
                arr = np.array(lm_pts)
                center = arr.mean(axis=0)
                expanded = ((arr - center) * 1.32 + center).astype(np.int32)
                hull = cv2.convexHull(expanded)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [hull], 255)
                mask = cv2.GaussianBlur(mask, (21, 21), 7)
                mask = np.clip(mask.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
                m = mask.astype(np.float32)[:,:,None] / 255.0
                result = (frame.astype(np.float32)*m +
                          bg.astype(np.float32)*(1-m)).astype(np.uint8)
                return result
        return frame

    def run(self):
        self.init()
        while self.running:
            if not self.cap_l or not self.cap_l.isOpened():
                time.sleep(0.1); continue
            self.cap_l.grab()
            if self.cap_r and self.cap_r.isOpened(): self.cap_r.grab()
            _,frame=self.cap_l.retrieve()
            if frame is None: continue
            self.update_fps()
            frame_norm = self.normalize_frame(frame)
            self.detect_aruco(frame_norm)
            lm   = self.get_landmarks(frame_norm)
            pts  = self.extract_pts(lm,frame.shape[1],frame.shape[0])
            meas = self.compute(pts)
            # Klinischer Hintergrund (nur wenn aktiv)
            try:
                win = next((w for w in QApplication.topLevelWidgets()
                           if hasattr(w,'clinical_bg')), None)
                if win and win.clinical_bg:
                    frame_out = self.apply_clinical_background(frame_norm, pts)
                else:
                    frame_out = frame_norm
            except Exception:
                frame_out = frame_norm
            self.frame_ready.emit(frame_out,pts,meas)
            # Ring-Buffer füllen wenn Aufnahme läuft
            try:
                win = next((w for w in QApplication.topLevelWidgets()
                           if hasattr(w,'recording') and w.recording), None)
                if win and hasattr(win,'_rec_ctrl') and win._rec_ctrl.recording:
                    win._rec_ctrl.ring.add('ELP Stereo', frame_out)
            except Exception:
                pass
            except Exception:
                pass

    def stop(self):
        self.running=False
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()
        if self.landmarker: self.landmarker.close()

# ── KAMERA WIDGET ─────────────────────────────────────────────────────────────
class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap      = None
        self.pts         = {}
        self.meas        = {}
        self.show_skel   = True
        self.show_axis   = True
        self.show_lot    = True
        self.show_seg    = True
        self.show_ang    = True
        self.clinical    = False
        self.px_per_cm   = None
        self.setMinimumSize(640,480)

    def update_frame(self,frame,pts,meas,px_per_cm):
        self.pts=pts; self.meas=meas; self.px_per_cm=px_per_cm
        h,w=frame.shape[:2]
        if self.clinical:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # ANTHRO3D Grauton-Palette
            r=np.clip(195+(gray.astype(np.float32)/255*33),0,255).astype(np.uint8)
            g=np.clip(210+(gray.astype(np.float32)/255*25),0,255).astype(np.uint8)
            b_ch=np.clip(198+(gray.astype(np.float32)/255*30),0,255).astype(np.uint8)
            display=cv2.merge([b_ch,g,r])
        else:
            display=frame.copy()
        rgb=cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
        qimg=QImage(rgb.data,w,h,w*3,QImage.Format.Format_RGB888)
        self.pixmap=QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self,e):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w,h=self.width(),self.height()

        # Hintergrund
        painter.fillRect(0,0,w,h,QColor(COLORS['bg']))

        if self.pixmap is None: return

        # Video skaliert (aspect ratio beibehalten)
        scaled=self.pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        vw,vh=scaled.width(),scaled.height()
        ox=(w-vw)//2; oy=(h-vh)//2
        painter.drawPixmap(ox,oy,scaled)

        if not self.pts: return

        # Scale-Faktoren
        pw=self.pixmap.width(); ph=self.pixmap.height()
        sx=vw/pw; sy=vh/ph

        def cp(name):
            if name not in self.pts: return None
            x,y=self.pts[name]
            return QPointF(x*sx+ox, y*sy+oy)

        # ── Skelett ───────────────────────────────────────────────────────────
        if self.show_skel:
            pen=QPen(QColor('#5a8a6a'),2,Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            for a,b in SKEL_CONNS:
                pa,pb=cp(a),cp(b)
                if pa and pb: painter.drawLine(pa,pb)
            for name in LM:
                p=cp(name)
                if p:
                    painter.setBrush(QColor('#5a8a6a'))
                    painter.setPen(QPen(QColor('white'),1))
                    painter.drawEllipse(p,4,4)

        # ── Achsenlinien ──────────────────────────────────────────────────────
        if self.show_axis:
            axes=[
                ('l_shoulder','r_shoulder','#3060a0','shoulder_ang'),
                ('l_hip',     'r_hip',     '#a060a0','hip_ang'),
                ('l_knee',    'r_knee',    '#3d6b50','knee_ang'),
            ]
            for a,b,col,ang_key in axes:
                pa,pb=cp(a),cp(b)
                if pa and pb:
                    # Weiße Outline
                    painter.setPen(QPen(QColor(255,255,255,180),5,Qt.PenStyle.SolidLine))
                    painter.drawLine(pa,pb)
                    # Farbige Linie
                    pen=QPen(QColor(col),2.5,Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(pa,pb)

                    # Abweichung von Horizontal berechnen
                    # pa = linke Seite, pb = rechte Seite
                    # Bildschirm: Y wächst nach unten
                    # Wenn pa.y() < pb.y() → linke Seite höher (negativer Y = höher)
                    dy = pa.y() - pb.y()  # positiv wenn links höher
                    dx = pb.x() - pa.x()
                    if abs(dx) > 1:
                        dev = round(math.degrees(math.atan2(dy, dx)), 1)
                    else:
                        dev = 0.0

                    # Nur anzeigen wenn Abweichung >= 1°
                    if abs(dev) >= 1.0:
                        if dev > 0:
                            dev_txt = f"L +{dev:.1f}\u00b0"
                        else:
                            dev_txt = f"R +{abs(dev):.1f}\u00b0"
                        painter.setFont(QFont('Helvetica', 10, QFont.Weight.Bold))
                        painter.setPen(QPen(QColor(255,255,255,230)))
                        painter.drawText(QPointF(pa.x()+3, pa.y()-10), dev_txt)
                        painter.setPen(QPen(QColor('#c04040')))
                        painter.drawText(QPointF(pa.x()+2, pa.y()-11), dev_txt)

        # ── Körperlot ─────────────────────────────────────────────────────────
        if self.show_lot:
            nose=cp('nose'); ankle=cp('l_ankle') or cp('r_ankle')
            if nose and ankle:
                cx=(nose.x()+ankle.x())/2
                pen=QPen(QColor(COLORS['warn']),1,Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(cx,oy),QPointF(cx,oy+vh))

        # ── Segmentlängen ─────────────────────────────────────────────────────
        if self.show_seg and self.px_per_cm:
            segs=[('l_shoulder','l_elbow','arm_l'),('r_shoulder','r_elbow','arm_r'),
                  ('l_hip','l_knee','thigh_l'),('r_hip','r_knee','thigh_r'),
                  ('l_knee','l_ankle','shin_l'),('r_knee','r_ankle','shin_r')]
            for a,b,key in segs:
                pa,pb=cp(a),cp(b)
                val=self.meas.get(key)
                if pa and pb and val:
                    mx=(pa.x()+pb.x())/2; my=(pa.y()+pb.y())/2
                    txt=f"{val:.0f}cm"
                    painter.setFont(QFont('Helvetica',8))
                    fm=painter.fontMetrics()
                    tw=fm.horizontalAdvance(txt); th=fm.height()
                    painter.fillRect(QRectF(mx-2,my-th,tw+4,th+2),QColor(255,255,255,200))
                    painter.setPen(QPen(QColor(COLORS['g1'])))
                    painter.drawText(QPointF(mx,my),txt)

# ── TOGGLE BUTTON ─────────────────────────────────────────────────────────────
class ToggleButton(QPushButton):
    def __init__(self,label,parent=None):
        super().__init__(label,parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedHeight(26)
        self._update_style()
        self.toggled.connect(lambda: self._update_style())

    def _update_style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton{{background:{COLORS['g2']};color:white;border:1px solid {COLORS['g1']};
                border-radius:5px;font-size:11px;font-weight:500;padding:2px 8px;}}
                QPushButton:hover{{background:{COLORS['g1']};}}""")
        else:
            self.setStyleSheet(f"""
                QPushButton{{background:{COLORS['bg']};color:{COLORS['dim']};border:1px solid {COLORS['border']};
                border-radius:5px;font-size:11px;padding:2px 8px;}}
                QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")

# ── MESSWERT KARTE ─────────────────────────────────────────────────────────────
class MeasCard(QFrame):
    def __init__(self,label,parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""QFrame{{background:{COLORS['bg']};border:1px solid {COLORS['border']};
            border-radius:6px;}}""")
        lay=QVBoxLayout(self); lay.setContentsMargins(8,5,8,5); lay.setSpacing(1)
        self.lbl=QLabel(label)
        self.lbl.setStyleSheet(f"font-size:9px;color:{COLORS['muted']};border:none;background:transparent;")
        self.val=QLabel("—")
        self.val.setStyleSheet(f"font-size:13px;font-weight:700;color:{COLORS['muted']};border:none;background:transparent;")
        lay.addWidget(self.lbl); lay.addWidget(self.val)

    def set_value(self,v,unit='cm'):
        if v is not None:
            self.val.setText(f"{v:.1f} {unit}")
            self.val.setStyleSheet(f"font-size:13px;font-weight:700;color:{COLORS['g1']};border:none;background:transparent;")
        else:
            self.val.setText("—")
            self.val.setStyleSheet(f"font-size:13px;color:{COLORS['muted']};border:none;background:transparent;")

# ── KALIBRIERUNGS-THREAD ──────────────────────────────────────────────────────
class CalibThread(QThread):
    progress  = pyqtSignal(dict)   # {name: {'count': int, 'detected': bool}}
    finished  = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.05
        params.errorCorrectionRate    = 0.5
        self.detector     = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.PATIENT_ID   = 16
        self.MARKER_SIZE  = 0.1865
        self.K            = np.array([[1400,0,960],[0,1400,540],[0,0,1]], dtype=np.float64)
        self.dist         = np.zeros((4,1))
        self.cam_map      = {}

    def stop(self):
        self.running = False

    def run(self):
        import yaml
        try:
            with open(Path("~/anthro3d/config.yaml").expanduser()) as f:
                cfg = yaml.safe_load(f)
            self.cam_map = {c['device_index']: c['name']
                           for c in cfg['cameras']['tracking']}
        except Exception:
            self.cam_map = {0:'ELP2', 1:'OV9281 R', 2:'OV9281 L', 3:'ELP1'}

        caps      = {}
        positions = {name: [] for name in self.cam_map.values()}
        rotations = {name: [] for name in self.cam_map.values()}
        detected  = {name: False for name in self.cam_map.values()}

        for idx, name in self.cam_map.items():
            cap = cv2.VideoCapture(idx)
            if cap.isOpened(): caps[idx] = cap

        while self.running:
            frame_detected = {name: False for name in self.cam_map.values()}

            for idx, name in self.cam_map.items():
                cap = caps.get(idx)
                if not cap: continue
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)
                if ids is None: continue
                for c, mid in zip(corners, ids.flatten()):
                    if int(mid) != self.PATIENT_ID: continue
                    frame_detected[name] = True
                    s = self.MARKER_SIZE / 2
                    obj = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)
                    ok, rvec, tvec = cv2.solvePnP(obj, c[0].astype(np.float32),
                                                   self.K, self.dist)
                    if ok:
                        positions[name].append(tvec.flatten().tolist())
                        rotations[name].append(rvec.flatten().tolist())

            # detected = ob dieser Frame einen Marker sah
            for name in detected:
                if frame_detected[name]:
                    detected[name] = True

            counts = {name: {'count': len(positions[name]),
                             'detected': frame_detected[name]}
                      for name in positions}
            self.progress.emit(counts)

            if all(v['count'] >= 50 for v in counts.values()):
                self.running = False

            time.sleep(0.05)

        result = {}
        for idx, name in self.cam_map.items():
            pts = positions.get(name, [])
            if not pts: continue
            arr = np.array(pts)
            med = np.median(arr, axis=0)
            result[name] = {
                'index':       idx,
                'position_m':  med.tolist(),
                'distance_cm': round(float(np.linalg.norm(med))*100, 1),
                'n_frames':    len(pts),
            }

        for cap in caps.values(): cap.release()
        self.finished.emit(result)

# ── HAUPTFENSTER ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang='de'
        self.setWindowTitle("ANTHRO3D")
        self.setMinimumSize(1280,800)
        self.clinical_bg = False  # Standard: aus
        self.recording=False
        self.rec_start=None
        self.rec_timer=QTimer()
        self.rec_timer.timeout.connect(self._update_rec_time)
        self._build_ui()
        self._start_camera()

    def _build_ui(self):
        # Globales Style
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{COLORS['bg']};color:{COLORS['text']};font-family:'Helvetica Neue',Arial,sans-serif;}}
            QLabel{{font-size:12px;}}
            QScrollArea{{border:none;}}
        """)

        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── Linke Sidebar ─────────────────────────────────────────────────────
        sidebar=QWidget(); sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(f"background:{COLORS['surface']};border-right:1px solid {COLORS['border']};")
        sb_lay=QVBoxLayout(sidebar); sb_lay.setContentsMargins(14,16,14,14); sb_lay.setSpacing(6)

        # Logo
        logo=QLabel("ANTHRO3D")
        logo.setStyleSheet(f"font-size:20px;font-weight:800;color:{COLORS['g1']};letter-spacing:2px;")
        sb_lay.addWidget(logo)

        # Status
        self.status_lbl=QLabel("● Verbinde...")
        self.status_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};")
        sb_lay.addWidget(self.status_lbl)

        self.aruco_lbl=QLabel("ArUco: ID1 vor Kamera halten")
        self.aruco_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['warn']};")
        self.aruco_lbl.setWordWrap(True)
        sb_lay.addWidget(self.aruco_lbl)

        sep=QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{COLORS['border']};"); sb_lay.addWidget(sep)

        # Ansicht
        self._section_label(sb_lay,"ANSICHT")
        self.btn_skel=ToggleButton("Skelett"); sb_lay.addWidget(self.btn_skel)
        self.btn_axis=ToggleButton("Achsenlinien"); sb_lay.addWidget(self.btn_axis)
        self.btn_lot =ToggleButton("Körperlot"); sb_lay.addWidget(self.btn_lot)
        self.btn_seg =ToggleButton("Segmentlängen"); sb_lay.addWidget(self.btn_seg)
        self.btn_ang =ToggleButton("Winkelangaben"); sb_lay.addWidget(self.btn_ang)
        self.btn_clin=ToggleButton("Klinisch"); self.btn_clin.setChecked(False); sb_lay.addWidget(self.btn_clin)

        self._section_label(sb_lay,"3D")
        self.mesh3d_btn_sb = QPushButton("🔲 3D Mesh anzeigen")
        self.mesh3d_btn_sb.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:8px;font-size:11px;padding:8px 12px;text-align:left;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}
            QPushButton:disabled{{color:{COLORS['border']};}}""")
        self.mesh3d_btn_sb.clicked.connect(self._show_3d_mesh)
        self.mesh3d_btn_sb.setEnabled(False)
        sb_lay.addWidget(self.mesh3d_btn_sb)

        sep2=QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{COLORS['border']};"); sb_lay.addWidget(sep2)

        # Patienten
        self._section_label(sb_lay,"PATIENTEN")
        btn_patients = QPushButton("👤  Patienten")
        btn_patients.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:8px;font-size:11px;padding:8px 12px;text-align:left;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")
        btn_patients.clicked.connect(self._open_patient_list)
        sb_lay.addWidget(btn_patients)

        sep2b=QFrame(); sep2b.setFrameShape(QFrame.Shape.HLine)
        sep2b.setStyleSheet(f"color:{COLORS['border']};"); sb_lay.addWidget(sep2b)

        # Qualität
        self._section_label(sb_lay,"QUALITÄT")
        self.btn_kalm=ToggleButton("Kalman-Filter"); self.btn_kalm.setChecked(False)
        self.btn_kalm.setToolTip("Glättet Keypoints — Standard AUS")
        sb_lay.addWidget(self.btn_kalm)

        sep3=QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"color:{COLORS['border']};"); sb_lay.addWidget(sep3)

        # Info
        self._section_label(sb_lay,"INFO")
        self.info_lbl=QLabel("Modus: 2D\nPunkte: —\nKalman: AUS")
        self.info_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['dim']};line-height:1.8;")
        sb_lay.addWidget(self.info_lbl)

        sb_lay.addStretch()

        # Sprache
        lang_btn=QPushButton("DE / EN")
        lang_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:5px;font-size:11px;padding:4px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        lang_btn.clicked.connect(self._toggle_lang)
        sb_lay.addWidget(lang_btn)

        root.addWidget(sidebar)

        # ── Mitte: QTabWidget ─────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane{{border:none;background:{COLORS['bg']};}}
            QTabBar::tab{{background:{COLORS['surface']};color:{COLORS['dim']};
                padding:8px 20px;font-size:11px;border:none;
                border-bottom:2px solid transparent;}}
            QTabBar::tab:selected{{color:{COLORS['g1']};font-weight:600;
                border-bottom:2px solid {COLORS['g1']};background:{COLORS['bg']};}}
            QTabBar::tab:hover{{color:{COLORS['g1']};}}
        """)

        # ── Tab 1: Messung ────────────────────────────────────────────────────
        meas_tab = QWidget()
        meas_lay = QVBoxLayout(meas_tab)
        meas_lay.setContentsMargins(0,0,0,0); meas_lay.setSpacing(0)

        # Ansicht Tabs (Frontal/Seitlich/Dorsal) + FPS + Vollbild
        tabs_bar=QWidget()
        tabs_bar.setStyleSheet(f"background:{COLORS['surface']};border-bottom:1px solid {COLORS['border']};")
        tabs_lay=QHBoxLayout(tabs_bar); tabs_lay.setContentsMargins(12,8,12,8); tabs_lay.setSpacing(4)
        for t in ["Frontal","Seitlich","Dorsal"]:
            btn=QPushButton(t)
            btn.setStyleSheet(f"""QPushButton{{background:{''+COLORS['g5'] if t=='Frontal' else 'transparent'};
                color:{COLORS['g1'] if t=='Frontal' else COLORS['dim']};
                border:1px solid {''+COLORS['g4'] if t=='Frontal' else 'transparent'};
                border-radius:6px;font-size:11px;padding:4px 14px;}}
                QPushButton:hover{{border-color:{COLORS['g4']};color:{COLORS['g1']};}}""")
            tabs_lay.addWidget(btn)
        tabs_lay.addStretch()
        self.fps_lbl=QLabel("— fps")
        self.fps_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['dim']};padding:2px 8px;background:{COLORS['bg']};border:1px solid {COLORS['border']};border-radius:10px;")
        tabs_lay.addWidget(self.fps_lbl)
        self.btn_fullscreen=QPushButton("⛶")
        self.btn_fullscreen.setToolTip("Vollbild (F)")
        self.btn_fullscreen.setFixedSize(28,28)
        self.btn_fullscreen.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:6px;font-size:14px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        self.btn_fullscreen.clicked.connect(self._toggle_cam_fullscreen)
        tabs_lay.addWidget(self.btn_fullscreen)
        meas_lay.addWidget(tabs_bar)

        # Kamera Widget
        self.cam_widget=CameraWidget()
        meas_lay.addWidget(self.cam_widget,1)

        # Aufnahme-Leiste
        rec_bar=QWidget()
        rec_bar.setStyleSheet(f"background:rgba(255,255,255,0.9);border-top:1px solid {COLORS['border']};")
        rec_bar.setFixedHeight(60)
        rec_lay=QHBoxLayout(rec_bar); rec_lay.setContentsMargins(20,10,20,10)
        rec_lay.addStretch()
        self.rec_btn=QPushButton("● Aufnahme starten")
        self.rec_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:20px;font-size:12px;font-weight:500;padding:8px 24px;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")
        self.rec_btn.clicked.connect(self._toggle_rec)
        rec_lay.addWidget(self.rec_btn)
        self.rec_time_lbl=QLabel("")
        self.rec_time_lbl.setStyleSheet(f"font-size:14px;font-weight:700;color:{COLORS['g1']};margin-left:12px;")
        rec_lay.addWidget(self.rec_time_lbl)
        rec_lay.addStretch()

        snap_btn=QPushButton("📸 Snapshot")
        snap_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:20px;font-size:11px;padding:8px 16px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        snap_btn.clicked.connect(self._snapshot)
        rec_lay.addWidget(snap_btn)

        self.mesh3d_btn = QPushButton("🔲 3D Mesh")
        self.mesh3d_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:20px;font-size:11px;padding:8px 16px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}
            QPushButton:disabled{{opacity:0.4;}}""")
        self.mesh3d_btn.clicked.connect(self._show_3d_mesh)
        self.mesh3d_btn.setEnabled(False)
        rec_lay.addWidget(self.mesh3d_btn)
        meas_lay.addWidget(rec_bar)

        self.tabs.addTab(meas_tab, "📷  Messung")

        # ── Tab 2: Kalibrierung ───────────────────────────────────────────────
        calib_tab = self._build_calib_tab()
        self.tabs.addTab(calib_tab, "🎯  Kalibrierung")

        # ── Tab 3: Kamera Setup ───────────────────────────────────────────────
        setup_tab = self._build_camera_setup_tab()
        self.tabs.addTab(setup_tab, "⚙️  Kamera Setup")

        root.addWidget(self.tabs, 1)
        self.rec_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:20px;font-size:12px;font-weight:500;padding:8px 24px;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")

        self.rec_time_lbl=QLabel("")
        self.rec_time_lbl.setStyleSheet(f"font-size:14px;font-weight:700;color:{COLORS['g1']};margin-left:12px;")

        snap_btn=QPushButton("📸 Snapshot")
        snap_btn.setStyleSheet(f"""QPushButton{{background:transparent;color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:20px;font-size:11px;padding:8px 16px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        snap_btn.clicked.connect(self._snapshot)

        # ── Rechtes Panel: Messwerte ───────────────────────────────────────────
        right=QScrollArea(); right.setFixedWidth(270)
        right.setWidgetResizable(True)
        right.setStyleSheet(f"background:{COLORS['surface']};border-left:1px solid {COLORS['border']};")
        right_content=QWidget()
        right_content.setStyleSheet(f"background:{COLORS['surface']};")
        right_lay=QVBoxLayout(right_content); right_lay.setContentsMargins(14,14,14,14); right_lay.setSpacing(0)

        # Körpergröße prominent
        h_box=QFrame()
        h_box.setStyleSheet(f"background:{COLORS['g5']};border-bottom:1px solid {COLORS['border']};")
        h_box_lay=QVBoxLayout(h_box); h_box_lay.setContentsMargins(12,12,12,12); h_box_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_height=QLabel("Körpergröße")
        self.lbl_height.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};text-transform:uppercase;letter-spacing:1px;")
        self.lbl_height.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_height=QLabel("—")
        self.val_height.setStyleSheet(f"font-size:36px;font-weight:800;color:{COLORS['g1']};")
        self.val_height.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_height_mode=QLabel("ArUco ID1 vor Kamera halten")
        self.val_height_mode.setStyleSheet(f"font-size:9px;color:{COLORS['muted']};")
        self.val_height_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_box_lay.addWidget(self.lbl_height); h_box_lay.addWidget(self.val_height)
        h_box_lay.addWidget(self.val_height_mode)
        right_lay.addWidget(h_box)

        # Segmentlängen
        self._r_section(right_lay,"📐 Segmentlängen")
        grid=QGridLayout(); grid.setSpacing(4)
        self.cards={}
        meas_keys=[('shoulder_w','Schulterbreite'),('hip_w','Hüftbreite'),
                   ('arm_l','Oberarm L'),('arm_r','Oberarm R'),
                   ('forearm_l','Unterarm L'),('forearm_r','Unterarm R'),
                   ('thigh_l','Oberschenkel L'),('thigh_r','Oberschenkel R'),
                   ('shin_l','Unterschenkel L'),('shin_r','Unterschenkel R'),
                   ('torso','Rumpflänge')]
        for i,(key,lbl) in enumerate(meas_keys):
            card=MeasCard(lbl); self.cards[key]=card
            grid.addWidget(card,i//2,i%2)
        right_lay.addLayout(grid)

        # Winkel
        self._r_section(right_lay,"📐 Achsenwinkel")
        ang_grid=QGridLayout(); ang_grid.setSpacing(4)
        for i,(key,lbl) in enumerate([('shoulder_ang','Schulterachse'),
                                        ('hip_ang','Beckenachse'),('knee_ang','Knieachse')]):
            card=MeasCard(lbl); self.cards[key]=card
            ang_grid.addWidget(card,i//2,i%2)
        right_lay.addLayout(ang_grid)
        right_lay.addStretch()

        right.setWidget(right_content)
        root.addWidget(right)

        # Verbinde Buttons
        self.btn_skel.toggled.connect(lambda v: setattr(self.cam_widget,'show_skel',v))
        self.btn_axis.toggled.connect(lambda v: setattr(self.cam_widget,'show_axis',v))
        self.btn_lot.toggled.connect(lambda v:  setattr(self.cam_widget,'show_lot',v))
        self.btn_seg.toggled.connect(lambda v:  setattr(self.cam_widget,'show_seg',v))
        self.btn_ang.toggled.connect(lambda v:  setattr(self.cam_widget,'show_ang',v))
        self.btn_clin.toggled.connect(lambda v: setattr(self.cam_widget,'clinical',v))
        self.btn_clin.toggled.connect(self._toggle_clinical_bg)
        self.btn_kalm.toggled.connect(lambda v: setattr(self.cam_thread.smoother,'enabled',v))

    def _build_calib_tab(self):
        """Kalibrierungs-Tab — automatische Selbstkalibrierung beim Start."""
        tab = QWidget()
        tab.setStyleSheet(f"background:{COLORS['bg']};")
        lay = QVBoxLayout(tab); lay.setContentsMargins(40,40,40,40); lay.setSpacing(16)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Kamera Kalibrierung")
        title.setStyleSheet(f"font-size:20px;font-weight:700;color:{COLORS['g1']};")
        lay.addWidget(title)

        desc = QLabel(
            "Halte den Patienten-Marker (ID 16) vor dich\n"
            "und drehe dich langsam einmal im Kreis.\n\n"
            "Das System berechnet automatisch die Position\n"
            "aller Kameras zueinander."
        )
        desc.setStyleSheet(f"font-size:13px;color:{COLORS['dim']};line-height:1.6;")
        lay.addWidget(desc)

        # Status Box
        self.calib_status_box = QFrame()
        self.calib_status_box.setStyleSheet(f"background:{COLORS['surface']};border:1px solid {COLORS['border']};border-radius:10px;")
        sb_lay = QVBoxLayout(self.calib_status_box); sb_lay.setContentsMargins(20,16,20,16); sb_lay.setSpacing(8)

        self.calib_status_lbl = QLabel("Bereit zum Starten")
        self.calib_status_lbl.setStyleSheet(f"font-size:14px;font-weight:600;color:{COLORS['text']};")
        sb_lay.addWidget(self.calib_status_lbl)

        # Fortschritt pro Kamera
        self.calib_bars = {}
        for name in ['OV9281 R', 'OV9281 L', 'ELP1', 'ELP2']:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(80)
            lbl.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
            bar = QProgressBar()
            bar.setRange(0, 50)
            bar.setValue(0)
            bar.setFixedHeight(8)
            bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                QProgressBar::chunk{{background:{COLORS['g2']};border-radius:4px;}}""")
            count_lbl = QLabel("0")
            count_lbl.setFixedWidth(30)
            count_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};")
            row.addWidget(lbl); row.addWidget(bar); row.addWidget(count_lbl)
            sb_lay.addLayout(row)
            self.calib_bars[name] = (bar, count_lbl)

        lay.addWidget(self.calib_status_box)

        # Ergebnis
        self.calib_result_lbl = QLabel("")
        self.calib_result_lbl.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};line-height:1.8;")
        self.calib_result_lbl.setWordWrap(True)
        lay.addWidget(self.calib_result_lbl)

        lay.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self.calib_btn = QPushButton("▶  Kalibrierung starten")
        self.calib_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:10px;font-size:13px;font-weight:600;padding:12px 32px;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")
        self.calib_btn.clicked.connect(self._toggle_calibration)

        self.calib_apply_btn = QPushButton("✓  Übernehmen & Messen")
        self.calib_apply_btn.setEnabled(False)
        self.calib_apply_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['border']};color:{COLORS['muted']};
            border:none;border-radius:10px;font-size:13px;padding:12px 32px;}}
            QPushButton:enabled{{background:{COLORS['g1']};color:white;font-weight:600;}}
            QPushButton:enabled:hover{{background:{COLORS['g2']};}}""")
        self.calib_apply_btn.clicked.connect(self._apply_calibration)

        btn_row.addWidget(self.calib_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(self.calib_apply_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # ── Stereo Kalibrierung ───────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{COLORS['border']};margin-top:8px;")
        lay.addWidget(sep)

        # ── Automatische Kamera-Positions-Kalibrierung ────────────────────────
        auto_title = QLabel("Automatische Positions-Kalibrierung")
        auto_title.setStyleSheet(f"font-size:16px;font-weight:700;color:{COLORS['g1']};margin-top:8px;")
        lay.addWidget(auto_title)

        auto_desc = QLabel(
            "Kameras erkennen gegenseitig ihre Boards — keine manuelle Aktion nötig.\n"
            "Stelle sicher dass alle Boards der anderen Kameras sichtbar sind."
        )
        auto_desc.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        lay.addWidget(auto_desc)

        auto_btn_row = QHBoxLayout()
        self.auto_calib_btn = QPushButton("▶  Positions-Kalibrierung starten")
        self.auto_calib_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:10px;font-size:13px;font-weight:600;padding:12px 32px;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")
        self.auto_calib_btn.clicked.connect(self._start_auto_calib)

        self.auto_calib_status = QLabel("")
        self.auto_calib_status.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")

        auto_btn_row.addWidget(self.auto_calib_btn)
        auto_btn_row.addSpacing(12)
        auto_btn_row.addWidget(self.auto_calib_status)
        auto_btn_row.addStretch()
        lay.addLayout(auto_btn_row)

        # Fortschrittsbalken Auto-Kalibrierung
        auto_prog_row = QHBoxLayout()
        auto_prog_lbl = QLabel("Frames:")
        auto_prog_lbl.setFixedWidth(60)
        auto_prog_lbl.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        self.auto_calib_bar = QProgressBar()
        self.auto_calib_bar.setRange(0, 80)
        self.auto_calib_bar.setValue(0)
        self.auto_calib_bar.setFixedHeight(10)
        self.auto_calib_bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:5px;border:none;}}
            QProgressBar::chunk{{background:{COLORS['g2']};border-radius:5px;}}""")
        self.auto_calib_count = QLabel("0 / 80")
        self.auto_calib_count.setFixedWidth(55)
        self.auto_calib_count.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};")
        auto_prog_row.addWidget(auto_prog_lbl)
        auto_prog_row.addWidget(self.auto_calib_bar)
        auto_prog_row.addWidget(self.auto_calib_count)
        lay.addLayout(auto_prog_row)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"color:{COLORS['border']};margin-top:8px;")
        lay.addWidget(sep3)

        stereo_title = QLabel("Stereo-Kamera ausrichten")
        stereo_title = QLabel("Stereo-Kalibrierung")
        stereo_title = QLabel("Stereo-Kalibrierung")
        stereo_title.setStyleSheet(f"font-size:16px;font-weight:700;color:{COLORS['g1']};margin-top:8px;")
        lay.addWidget(stereo_title)

        align_desc = QLabel(
            "Öffnet ein Live-Fenster das zeigt wie gut beide Kameras übereinstimmen.\n"
            "Richte die Kameras so aus bis die Überlagerung grün wird."
        )
        align_desc.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        lay.addWidget(align_desc)

        align_btn_row = QHBoxLayout()
        self.align_btn = QPushButton("🎯  Stereo ausrichten")
        self.align_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['surface']};color:{COLORS['text']};
            border:1px solid {COLORS['border']};border-radius:10px;font-size:13px;font-weight:600;padding:12px 32px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        self.align_btn.clicked.connect(self._start_stereo_align)
        self.align_status = QLabel("")
        self.align_status.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        align_btn_row.addWidget(self.align_btn)
        align_btn_row.addSpacing(12)
        align_btn_row.addWidget(self.align_status)
        align_btn_row.addStretch()
        lay.addLayout(align_btn_row)

        sep4 = QFrame(); sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setStyleSheet(f"color:{COLORS['border']};margin-top:8px;")
        lay.addWidget(sep4)

        stereo_title2 = QLabel("Stereo-Kalibrierung")

        stereo_desc = QLabel(
            "Halte das ChArUco Board (9×6, 25mm) vor beide Stereokameras\n"
            "gleichzeitig — verschiedene Winkel und Abstände verwenden.\n"
            "Ziel: 50 Bildpaare → Tiefenmessung ±5mm"
        )
        stereo_desc.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        lay.addWidget(stereo_desc)

        stereo_btn_row = QHBoxLayout()
        self.stereo_btn = QPushButton("▶  Stereo-Kalibrierung starten")
        self.stereo_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['blue']};color:white;
            border:none;border-radius:10px;font-size:13px;font-weight:600;padding:12px 32px;}}
            QPushButton:hover{{background:#4070b0;}}""")
        self.stereo_btn.clicked.connect(self._start_stereo_calib)

        self.stereo_status = QLabel("")
        self.stereo_status.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")

        stereo_btn_row.addWidget(self.stereo_btn)
        stereo_btn_row.addSpacing(12)
        stereo_btn_row.addWidget(self.stereo_status)
        stereo_btn_row.addStretch()
        lay.addLayout(stereo_btn_row)

        # Fortschrittsbalken Stereo
        stereo_prog_row = QHBoxLayout()
        stereo_prog_lbl = QLabel("Bildpaare:")
        stereo_prog_lbl.setFixedWidth(70)
        stereo_prog_lbl.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        self.stereo_bar = QProgressBar()
        self.stereo_bar.setRange(0, 50)
        self.stereo_bar.setValue(0)
        self.stereo_bar.setFixedHeight(10)
        self.stereo_bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:5px;border:none;}}
            QProgressBar::chunk{{background:{COLORS['blue']};border-radius:5px;}}""")
        self.stereo_count_lbl = QLabel("0 / 50")
        self.stereo_count_lbl.setFixedWidth(50)
        self.stereo_count_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};")
        stereo_prog_row.addWidget(stereo_prog_lbl)
        stereo_prog_row.addWidget(self.stereo_bar)
        stereo_prog_row.addWidget(self.stereo_count_lbl)
        lay.addLayout(stereo_prog_row)

        self.calib_thread = None
        self.calib_positions = None
        return tab

    def _start_auto_calib(self):
        """Positions-Kalibrierung — Kameras erkennen sich gegenseitig."""
        import yaml
        self.auto_calib_btn.setEnabled(False)
        self.auto_calib_status.setText("Erkennung läuft...")
        self.auto_calib_bar.setValue(0)
        self.auto_calib_count.setText("0 / 80")

        # Kamera-Mapping laden
        try:
            with open(Path("~/anthro3d/config.yaml").expanduser()) as f:
                cfg = yaml.safe_load(f)
            cam_map = {c['device_index']: c['name'] for c in cfg['cameras']['tracking']}
        except Exception:
            cam_map = {0:'ELP2', 1:'OV9281 R', 2:'OV9281 L', 3:'ELP1'}

        # Welches Board hängt an welcher Kamera (eigene ID)
        CAM_OWN_MARKER = {'OV9281 R': 2, 'OV9281 L': 2, 'ELP1': 3, 'ELP2': 10}

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.05
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        MARKER_SIZE = 0.1865
        K = np.array([[1400,0,960],[0,1400,540],[0,0,1]], dtype=np.float64)
        dist_c = np.zeros((4,1))

        positions = {name: [] for name in cam_map.values()}
        rotations = {name: [] for name in cam_map.values()}
        caps = {}
        for idx in cam_map:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened(): caps[idx] = cap

        frame_count = [0]
        TARGET = 80

        # Fortschrittsbalken für jede Kamera im auto_calib Bereich
        # (nutzt die calib_bars aus der oberen Kalibrierung als Anzeige)

        def auto_step():
            frame_count[0] += 1
            detected_now = {name: False for name in cam_map.values()}

            for idx, name in cam_map.items():
                cap = caps.get(idx)
                if not cap: continue
                ret, frame = cap.read()
                if not ret: continue
                # Normalisierung
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b_ch = cv2.split(lab)
                if not hasattr(self, '_clahe_auto'):
                    self._clahe_auto = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = self._clahe_auto.apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None: continue

                own_id = CAM_OWN_MARKER.get(name, -1)
                for c, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    if mid == own_id: continue  # eigenes Board überspringen
                    if mid not in [2, 3, 4, 20, 30, 40]: continue
                    detected_now[name] = True
                    sv = MARKER_SIZE / 2
                    obj = np.array([[-sv,sv,0],[sv,sv,0],[sv,-sv,0],[-sv,-sv,0]], dtype=np.float32)
                    ok, rvec, tvec = cv2.solvePnP(obj, c[0].astype(np.float32), K, dist_c)
                    if ok:
                        positions[name].append(tvec.flatten().tolist())
                        rotations[name].append(rvec.flatten().tolist())

            # Fortschrittsbalken updaten
            self.auto_calib_bar.setValue(min(frame_count[0], TARGET))
            self.auto_calib_count.setText(f"{frame_count[0]} / {TARGET}")

            # Status pro Kamera anzeigen
            parts = []
            for name in cam_map.values():
                cnt = len(positions[name])
                ok = detected_now[name]
                parts.append(f"{'✓' if ok else '⚠'} {name}:{cnt}")
            self.auto_calib_status.setText("  ".join(parts))
            if any(not detected_now[n] for n in cam_map.values()):
                self.auto_calib_status.setStyleSheet("font-size:10px;color:#c04040;")
            else:
                self.auto_calib_status.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};")

            if frame_count[0] >= TARGET:
                self._auto_calib_timer.stop()
                for cap in caps.values(): cap.release()
                # Positionen berechnen und speichern
                result = {}
                for idx, name in cam_map.items():
                    pts = positions.get(name, [])
                    if not pts: continue
                    arr = np.array(pts)
                    med = np.median(arr, axis=0)
                    dist_cm = round(float(np.linalg.norm(med))*100, 1)
                    rvecs = rotations.get(name, [])
                    rot_med = np.median(np.array(rvecs), axis=0).tolist() if rvecs else [0,0,0]
                    import cv2 as _cv2; R_mat, _ = _cv2.Rodrigues(np.array(rot_med))
                    result[name] = {'index': idx, 'position_m': med.tolist(),
                                    'rotation_vec': rot_med,
                                    'rotation_mat': R_mat.tolist(),
                                    'distance_cm': dist_cm, 'n_frames': len(pts)}
                    print(f"  {name}: {dist_cm}cm ({len(pts)} Frames)")

                if result:
                    config = {'calibration': {'timestamp': time.time(), 'marker_size_m': MARKER_SIZE},
                              'cameras': result}
                    with open(Path("~/anthro3d/cam_positions.yaml").expanduser(), 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    self.auto_calib_status.setText(f"✓ Positions-Kalibrierung gespeichert — {len(result)} Kameras — kein Hintergrund")
                    self.auto_calib_status.setStyleSheet(f"font-size:11px;color:{COLORS['g2']};font-weight:600;")
                    # Positions-Kalibrierung endet hier bewusst.
                    # calibrate.py wird NICHT automatisch gestartet.
                    # Dadurch wird kein Hintergrund aufgenommen.

                else:
                    self.auto_calib_status.setText("⚠ Keine Erkennung — Boards ausrichten!")
                    self.auto_calib_status.setStyleSheet("font-size:11px;color:#c04040;")
                self.auto_calib_btn.setEnabled(True)

        self._auto_calib_timer = QTimer()
        self._auto_calib_timer.timeout.connect(auto_step)
        self._auto_calib_timer.start(50)

    def _start_stereo_align(self):
        """Öffnet Live-Ausrichtungsfenster für Stereokameras."""
        import subprocess, sys
        self.align_btn.setEnabled(False)
        self.align_status.setText("Fenster geöffnet — Q zum Schließen")

        script = """
import cv2, numpy as np, sys
CAM_L, CAM_R = 1, 2
cap_l = cv2.VideoCapture(CAM_L)
cap_r = cv2.VideoCapture(CAM_R)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
while True:
    rl,fl=cap_l.read(); rr,fr=cap_r.read()
    if not rl or not rr: continue
    fl_g=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
    fr_g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    fl_g=clahe.apply(fl_g); fr_g=clahe.apply(fr_g)
    H,W=fl_g.shape
    overlay=np.zeros((H,W,3),dtype=np.uint8)
    overlay[:,:,1]=fl_g
    overlay[:,:,2]=fr_g
    overlay[:,:,0]=fr_g
    diff=cv2.absdiff(fl_g,fr_g)
    score=max(0,100-int(np.mean(diff)*2))
    color=(0,200,80) if score>70 else (0,150,255) if score>40 else (80,80,220)
    disp=cv2.resize(overlay,(1280,400))
    W2=disp.shape[1]
    bar_w=int(W2*score/100)
    cv2.rectangle(disp,(0,370),(W2,400),(40,40,40),-1)
    cv2.rectangle(disp,(0,370),(bar_w,400),color,-1)
    cv2.putText(disp,f"Uebereinstimmung: {score}%",(10,395),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    msg="GUT ausgerichtet!" if score>70 else ("Fast — weiter anpassen" if score>40 else "Kameras ausrichten")
    cv2.putText(disp,msg,(W2//2-200,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2)
    cv2.putText(disp,"Gruen=Links  Magenta=Rechts  Weiss=Ueberlappung",(10,360),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
    side=np.hstack([cv2.resize(fl,(640,200)),cv2.resize(fr,(640,200))])
    cv2.putText(side,"OV9281 R",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,100),2)
    cv2.putText(side,"OV9281 L",(650,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
    full=np.vstack([disp,side])
    cv2.imshow("Stereo ausrichten — Q zum Schliessen",full)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cv2.destroyAllWindows()
cap_l.release(); cap_r.release()
"""
        proc = subprocess.Popen([sys.executable, '-c', script])

        def check_done():
            if proc.poll() is not None:
                self._align_timer.stop()
                self.align_btn.setEnabled(True)
                self.align_status.setText("")
        self._align_timer = QTimer()
        self._align_timer.timeout.connect(check_done)
        self._align_timer.start(500)

    def _start_stereo_calib(self):
        """Startet Stereo-Kalibrierung — eine ELP auswählen, L/R wird automatisch gesplittet."""
        import subprocess
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel as _QLabel, QRadioButton, QButtonGroup

        # Nur ELPs aus config laden
        try:
            import yaml
            cfg = yaml.safe_load(open(Path("~/anthro3d/config.yaml").expanduser()))
            cams = cfg.get("cameras", {}).get("tracking", [])
            elps = [c for c in cams if "ELP" in c.get("name", "")]
        except Exception:
            elps = []

        # Kalibrierungsstatus laden (beide ELP Configs prüfen)
        calibrated = {}
        import yaml as _yaml
        for _cfg_name in ["stereo_config.yaml", "stereo_config_elp1.yaml"]:
            _p = Path(f"~/anthro3d/{_cfg_name}").expanduser()
            if _p.exists():
                try:
                    sc = _yaml.safe_load(open(_p))
                    calibrated.update(sc.get("calibrated_elps", {}))
                except Exception:
                    pass

        # Dialog — eine ELP auswählen
        dlg = QDialog(self)
        dlg.setWindowTitle("Stereo-Kalibrierung — ELP auswählen")
        dlg.setMinimumWidth(380)
        vlay = QVBoxLayout(dlg)
        vlay.setSpacing(10)
        vlay.addWidget(_QLabel("<b>Welche ELP-Kamera kalibrieren?</b>"))
        vlay.addWidget(_QLabel("<small style='color:gray'>Jede ELP hat L/R eingebaut — Board vor eine ELP halten.</small>"))

        btn_group = QButtonGroup(dlg)
        radios = []
        for c in elps:
            name = c["name"]
            idx  = c["device_index"]
            status = " ✓ bereits kalibriert" if name in calibrated else ""
            rb = QRadioButton(f"{name}  (Index {idx}){status}")
            rb.setStyleSheet("font-size:13px; padding:4px;")
            if not radios:
                rb.setChecked(True)
            btn_group.addButton(rb)
            radios.append((rb, idx, name))
            vlay.addWidget(rb)

        if not radios:
            vlay.addWidget(_QLabel("Keine ELP-Kameras gefunden!"))

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vlay.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        sel = [(rb, idx, name) for rb, idx, name in radios if rb.isChecked()]
        if not sel:
            return
        elp_index = sel[0][1]
        elp_name  = sel[0][2]

        self.stereo_status.setText(f"Läuft — Board vor {elp_name} halten...")
        self.stereo_status.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        self.stereo_btn.setEnabled(False)
        self.stereo_bar.setValue(0)
        self.stereo_count_lbl.setText("0 / 50")

        script = Path("~/anthro3d").expanduser() / "stereo_charuco_calib.py"
        # QThread für Kalibrierung — kein Subprocess, cv2.imshow im Hauptprozess
        from PyQt6.QtCore import QThread, pyqtSignal as _Signal
        import cv2 as _cv2
        import numpy as _np
        import yaml as _yaml

        class StereoCalibThread(QThread):
            progress = _Signal(int)
            frame_ready = _Signal(object)
            finished = _Signal(bool, str)

            def __init__(self, elp_index, elp_name, target=80):
                super().__init__()
                self.elp_index = elp_index
                self.elp_name  = elp_name
                self.target    = target
                self._stop     = False

            def stop(self): self._stop = True

            def run(self):
                aruco_dict = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_ARUCO_ORIGINAL)
                board      = _cv2.aruco.CharucoBoard((9,6), 0.020, 0.015, aruco_dict)
                detector   = _cv2.aruco.ArucoDetector(aruco_dict, _cv2.aruco.DetectorParameters())
                cap = _cv2.VideoCapture(self.elp_index)
                cap.set(_cv2.CAP_PROP_FRAME_WIDTH, 3200)
                cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 1200)
                all_cl,all_il,all_cr,all_ir = [],[],[],[]
                count = 0
                base = Path("~/anthro3d").expanduser()

                # Geführte Positionen
                POSITIONS = [
                    ("Mitte — nah (30cm)",          (640, 300), 0),
                    ("Links oben",                   (200, 100), 0),
                    ("Rechts oben",                  (1100, 100), 0),
                    ("Links unten",                  (200, 500), 0),
                    ("Rechts unten",                 (1100, 500), 0),
                    ("Mitte — mittel (60cm)",        (640, 300), 0),
                    ("Links oben — kippen",          (200, 100), 20),
                    ("Rechts oben — kippen",         (1100, 100), -20),
                    ("Links unten — kippen",         (200, 500), -20),
                    ("Rechts unten — kippen",        (1100, 500), 20),
                    ("Mitte — links kippen 30°",     (640, 300), -30),
                    ("Mitte — rechts kippen 30°",    (640, 300), 30),
                    ("Mitte — weit (100cm)",         (640, 300), 0),
                    ("Links Mitte",                  (150, 300), 10),
                    ("Rechts Mitte",                 (1150, 300), -10),
                    ("Oben Mitte",                   (640, 80), 0),
                    ("Unten Mitte",                  (640, 550), 0),
                    ("Ganz nah (20cm)",              (640, 300), 0),
                    ("Links oben nah",               (200, 100), 25),
                    ("Rechts unten nah",             (1100, 500), -25),
                    ("Sehr weit (150cm)",            (640, 300), 0),
                    ("Links oben weit",              (200, 100), 0),
                    ("Rechts unten weit",            (1100, 500), 0),
                    ("Mitte kippen diagonal",        (640, 300), 45),
                    ("Mitte kippen diagonal 2",      (640, 300), -45),
                    ("Links unten weit",             (150, 500), 15),
                    ("Rechts oben weit",             (1150, 80), -15),
                    ("Mitte — sehr nah (15cm)",      (640, 300), 0),
                    ("Oben links kippen stark",      (200, 80), 35),
                    ("Unten rechts kippen stark",    (1100, 550), -35),
                ]

                def draw_guide(img, pos_label, target_x, target_y, angle, detected, count, total):
                    h, w = img.shape[:2]
                    # Zielrahmen
                    bw, bh = 300, 220
                    tx = min(max(target_x - bw//2, 10), w - bw - 10)
                    ty = min(max(target_y - bh//2, 10), h - bh - 10)
                    color = (0,255,0) if detected else (0,165,255)
                    _cv2.rectangle(img, (tx,ty), (tx+bw,ty+bh), color, 3)
                    # Winkelanzeige
                    if abs(angle) > 5:
                        arrow_x = tx + bw//2
                        arrow_y = ty + bh//2
                        dx = int(_np.sin(_np.radians(angle)) * 60)
                        _cv2.arrowedLine(img, (arrow_x,arrow_y), (arrow_x+dx,arrow_y), (255,200,0), 3)
                    # Text
                    _cv2.rectangle(img, (0,0), (w,50), (0,0,0), -1)
                    _cv2.putText(img, f"Position: {pos_label}", (10,22),
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    _cv2.putText(img, f"Board in Rahmen halten — {count}/{total} Paare",
                                (10,42), _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,255,0) if detected else (0,165,255), 1)
                    return img

                pos_idx = 0
                while not self._stop:
                    ret, frame = cap.read()
                    if not ret or frame is None: continue
                    w  = frame.shape[1] // 2
                    fl = frame[:, :w].copy()
                    fr = frame[:, w:].copy()
                    gl = _cv2.cvtColor(fl, _cv2.COLOR_BGR2GRAY)
                    gr = _cv2.cvtColor(fr, _cv2.COLOR_BGR2GRAY)
                    cl,il,_ = detector.detectMarkers(gl)
                    cr,ir,_ = detector.detectMarkers(gr)
                    ok_l=ok_r=False
                    if il is not None and len(il)>=4:
                        r2,ccl,cil = _cv2.aruco.interpolateCornersCharuco(cl,il,gl,board)
                        if r2 and r2>=6: _cv2.aruco.drawDetectedCornersCharuco(fl,ccl,cil,(0,255,0)); ok_l=True
                    if ir is not None and len(ir)>=4:
                        r2,ccr,cir = _cv2.aruco.interpolateCornersCharuco(cr,ir,gr,board)
                        if r2 and r2>=6: _cv2.aruco.drawDetectedCornersCharuco(fr,ccr,cir,(0,255,0)); ok_r=True
                    if ok_l and ok_r:
                        all_cl.append(ccl);all_il.append(cil);all_cr.append(ccr);all_ir.append(cir)
                        count+=1
                        self.progress.emit(count)
                        pos_idx = (pos_idx+1) % len(POSITIONS)
                        self.msleep(600)
                    # Guide zeichnen
                    p_label, (px,py), p_angle = POSITIONS[min(pos_idx, len(POSITIONS)-1)]
                    fl = draw_guide(fl, p_label, px, py, p_angle, ok_l and ok_r, count, self.target)
                    fr = draw_guide(fr, p_label, px, py, p_angle, ok_l and ok_r, count, self.target)
                    disp = _np.hstack([_cv2.resize(fl,(800,480)),_cv2.resize(fr,(800,480))])
                    self.frame_ready.emit(disp)
                    if count >= self.target: break

                cap.release()

                if count >= 10:
                    try:
                        print(f"Kalibriere mit {count} Paaren...")
                        # Objektpunkte direkt aus Board-Geometrie
                        obj_pts_l, obj_pts_r, img_pts_l, img_pts_r = [], [], [], []
                        ch_corners = board.getChessboardCorners()
                        for ccl, cil, ccr, cir in zip(all_cl, all_il, all_cr, all_ir):
                            ids_l = cil.flatten()
                            ids_r = cir.flatten()
                            common = sorted(set(ids_l) & set(ids_r))
                            if len(common) < 6: continue
                            o, pl, pr = [], [], []
                            for cid in common:
                                il_i = _np.where(ids_l==cid)[0]
                                ir_i = _np.where(ids_r==cid)[0]
                                if len(il_i)==0 or len(ir_i)==0: continue
                                o.append(ch_corners[cid])
                                pl.append(ccl[il_i[0]][0])
                                pr.append(ccr[ir_i[0]][0])
                            if len(o) >= 6:
                                obj_pts_l.append(_np.array(o, dtype=_np.float32))
                                obj_pts_r.append(_np.array(o, dtype=_np.float32))
                                img_pts_l.append(_np.array(pl, dtype=_np.float32))
                                img_pts_r.append(_np.array(pr, dtype=_np.float32))

                        print(f"Gültige Paare: {len(obj_pts_l)}")
                        if len(obj_pts_l) >= 8:
                            rms_l,K_l,d_l,_,_ = _cv2.calibrateCamera(obj_pts_l,img_pts_l,(1600,1200),None,None)
                            rms_r,K_r,d_r,_,_ = _cv2.calibrateCamera(obj_pts_r,img_pts_r,(1600,1200),None,None)
                            print(f"RMS L:{rms_l:.3f} R:{rms_r:.3f}")
                            print(f"fx_l={K_l[0,0]:.0f} fy_l={K_l[1,1]:.0f}")
                            print(f"fx_r={K_r[0,0]:.0f} fy_r={K_r[1,1]:.0f}")
                            rms_s,_,_,_,_,R,T,_,_ = _cv2.stereoCalibrate(
                                obj_pts_l,img_pts_l,img_pts_r,
                                K_l,d_l,K_r,d_r,(1600,1200),
                                flags=_cv2.CALIB_FIX_INTRINSIC)
                            print(f"Stereo RMS:{rms_s:.3f} Baseline:{_np.linalg.norm(T)*100:.1f}cm")
                            import yaml as _y
                            cfg = {"camera_matrix_l":K_l.tolist(),"dist_l":d_l.tolist(),
                                   "camera_matrix_r":K_r.tolist(),"dist_r":d_r.tolist(),
                                   "R":R.tolist(),"T":T.tolist(),
                                   "calibrated_elps":{self.elp_name:True},
                                   "calibrated_indices":[self.elp_index]}
                            cfg_name = "stereo_config.yaml" if self.elp_index == 0 else "stereo_config_elp1.yaml"
                            with open(str(base/cfg_name),"w") as f: _y.dump(cfg,f)
                            self.finished.emit(True, self.elp_name)
                            return
                    except Exception as e:
                        print("Kalibrierungsfehler:", e)
                        import traceback; traceback.print_exc()
                self.finished.emit(False, self.elp_name)

        self._stereo_thread = StereoCalibThread(elp_index, elp_name)
        self._stereo_win    = None

        def on_frame(img):
            from PyQt6.QtWidgets import QLabel
            from PyQt6.QtGui import QImage, QPixmap
            from PyQt6.QtCore import Qt
            if self._stereo_win is None:
                self._stereo_win = QLabel()
                self._stereo_win.setWindowTitle(elp_name + " Stereo Kalibrierung — Schließen zum Abbrechen")
                self._stereo_win.resize(1280, 400)
                self._stereo_win.show()
            h,w,c = img.shape
            qi = QImage(img.data, w, h, w*c, QImage.Format.Format_BGR888)
            self._stereo_win.setPixmap(QPixmap.fromImage(qi))

        def on_progress(val):
            self.stereo_bar.setValue(min(val, 50))
            self.stereo_count_lbl.setText(f"{val} / 50")

        def on_finished(ok, name):
            if self._stereo_win:
                self._stereo_win.close()
                self._stereo_win = None
            if ok:
                self.stereo_bar.setValue(50)
                self.stereo_count_lbl.setText("50 / 50")
                self.stereo_status.setText(f"✓ {name} kalibriert!")
                self.stereo_status.setStyleSheet(f"font-size:11px;color:{COLORS['g2']};font-weight:600;")
            else:
                self.stereo_status.setText("Abgebrochen oder zu wenig Paare")
            self.stereo_btn.setEnabled(True)

        self._stereo_thread.frame_ready.connect(on_frame)
        self._stereo_thread.progress.connect(on_progress)
        self._stereo_thread.finished.connect(on_finished)
        self._stereo_thread.start()

    def _create_stereo_script(self, path, elp_index=0, elp_name="ELP"):
        """Erstellt stereo_charuco_calib.py — ELP Bild wird in L/R gesplittet."""
        script = """import cv2, numpy as np, yaml, time
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
board      = cv2.aruco.CharucoBoard((6,9), 0.025, 0.018, aruco_dict)
detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
ELP_INDEX = __ELP_INDEX__
ELP_NAME  = "__ELP_NAME__"
TARGET = 80
PROGRESS_FILE = "/tmp/stereo_progress.txt"
cap = cv2.VideoCapture(ELP_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
all_cl,all_il,all_cr,all_ir = [],[],[],[]
count = 0
print("ChArUco Stereo " + ELP_NAME + " (Index " + str(ELP_INDEX) + ") — Ziel: " + str(TARGET) + " Paare | Q=Beenden")
while True:
    ret, frame = cap.read()
    if not ret or frame is None: continue
    w = frame.shape[1] // 2
    fl = frame[:, :w].copy()
    fr = frame[:, w:].copy()
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cl,il,_ = detector.detectMarkers(gl); cr,ir,_ = detector.detectMarkers(gr)
    ok_l=ok_r=False
    if il is not None and len(il)>=4:
        ret2,ccl,cil = cv2.aruco.interpolateCornersCharuco(cl,il,gl,board)
        if ret2 and ret2>=4: cv2.aruco.drawDetectedCornersCharuco(fl,ccl,cil,(0,255,0)); ok_l=True
    if ir is not None and len(ir)>=4:
        ret2,ccr,cir = cv2.aruco.interpolateCornersCharuco(cr,ir,gr,board)
        if ret2 and ret2>=4: cv2.aruco.drawDetectedCornersCharuco(fr,ccr,cir,(0,255,0)); ok_r=True
    if ok_l and ok_r:
        all_cl.append(ccl);all_il.append(cil);all_cr.append(ccr);all_ir.append(cir)
        count+=1
        open(PROGRESS_FILE,"w").write(str(count))
        time.sleep(0.4)
        cv2.putText(fl,"OK "+str(count)+"/"+str(TARGET),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(fl,"L:"+("OK" if ok_l else "--")+" R:"+("OK" if ok_r else "--")+" "+str(count)+"/"+str(TARGET),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,255),2)
    cv2.imshow(ELP_NAME + " Stereo Kalib — Q=Beenden",np.hstack([cv2.resize(fl,(640,400)),cv2.resize(fr,(640,400))]))
    key=cv2.waitKey(1)&0xFF
    if key in (ord('q'),27): break
    if count>=TARGET: break
cv2.destroyAllWindows(); cap.release()
if count>=5:
    print("Kalibriere mit " + str(count) + " Paaren...")
    err_l,K_l,d_l,_,_ = cv2.aruco.calibrateCameraCharuco(all_cl,all_il,board,(1600,1200),None,None)
    err_r,K_r,d_r,_,_ = cv2.aruco.calibrateCameraCharuco(all_cr,all_ir,board,(1600,1200),None,None)
""".replace("__ELP_INDEX__", str(elp_index)).replace("__ELP_NAME__", elp_name)
        with open(path, "w") as f:
            f.write(script)


    def _toggle_calibration(self):
        if self.calib_thread and self.calib_thread.isRunning():
            self.calib_thread.stop()
            self.calib_btn.setText("▶  Kalibrierung starten")
            self.calib_status_lbl.setText("Gestoppt")
        else:
            self._start_calibration()

    def _start_calibration(self):
        self.calib_btn.setText("■  Stoppen")
        self.calib_status_lbl.setText("Drehe dich mit Marker ID 16 im Kreis...")
        self.calib_apply_btn.setEnabled(False)
        self.calib_result_lbl.setText("")
        for bar, lbl in self.calib_bars.values():
            bar.setValue(0)
            lbl.setText("0")
            lbl.setStyleSheet(f"font-size:10px;color:{COLORS['muted']};")
            bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                QProgressBar::chunk{{background:{COLORS['g4']};border-radius:4px;}}""")

        # Kalibrierung direkt im Main Thread über QTimer
        import yaml
        try:
            with open(Path("~/anthro3d/config.yaml").expanduser()) as f:
                cfg = yaml.safe_load(f)
            cam_map = {c['device_index']: c['name'] for c in cfg['cameras']['tracking']}
        except Exception:
            cam_map = {0:'ELP2', 1:'OV9281 R', 2:'OV9281 L', 3:'ELP1'}

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.05
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        PATIENT_ID = 16
        MARKER_SIZE = 0.1865
        K = np.array([[1400,0,960],[0,1400,540],[0,0,1]], dtype=np.float64)
        dist_c = np.zeros((4,1))

        positions = {name: [] for name in cam_map.values()}
        rotations = {name: [] for name in cam_map.values()}
        caps = {}
        for idx in cam_map:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened(): caps[idx] = cap

        self._calib_state = {
            'cam_map': cam_map, 'detector': detector, 'caps': caps,
            'positions': positions, 'PATIENT_ID': PATIENT_ID,
            'MARKER_SIZE': MARKER_SIZE, 'K': K, 'dist': dist_c,
        }

        self._calib_timer2 = QTimer()
        self._calib_timer2.timeout.connect(self._calib_step)
        self._calib_timer2.start(50)

    def _calib_step(self):
        s = self._calib_state
        cam_map = s['cam_map']; detector = s['detector']
        caps = s['caps']; positions = s['positions']
        PATIENT_ID = s['PATIENT_ID']
        MARKER_SIZE = s['MARKER_SIZE']; K = s['K']; dist_c = s['dist']

        frame_detected = {name: False for name in cam_map.values()}

        for idx, name in cam_map.items():
            cap = caps.get(idx)
            if not cap: continue
            ret, frame = cap.read()
            if not ret: continue
            # Normalisierung für bessere Marker-Erkennung
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            if not hasattr(self, '_clahe'):
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None: continue
            for c, mid in zip(corners, ids.flatten()):
                if int(mid) != PATIENT_ID: continue
                frame_detected[name] = True
                sv = MARKER_SIZE / 2
                obj = np.array([[-sv,sv,0],[sv,sv,0],[sv,-sv,0],[-sv,-sv,0]], dtype=np.float32)
                ok, rvec, tvec = cv2.solvePnP(obj, c[0].astype(np.float32), K, dist_c)
                if ok:
                    positions[name].append(tvec.flatten().tolist())
                    rotations[name].append(rvec.flatten().tolist())

        # UI updaten
        all_done = True
        for name, (bar, lbl) in self.calib_bars.items():
            count = len(positions.get(name, []))
            detected = frame_detected.get(name, False)
            bar.setValue(min(count, 50))
            if count >= 50:
                lbl.setText(str(count))
                lbl.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};font-weight:600;")
                bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                    QProgressBar::chunk{{background:{COLORS['g2']};border-radius:4px;}}""")
            elif detected:
                lbl.setText(str(count))
                lbl.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};")
                bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                    QProgressBar::chunk{{background:{COLORS['g2']};border-radius:4px;}}""")
            else:
                lbl.setText("⚠")
                lbl.setStyleSheet("font-size:12px;color:#c04040;font-weight:700;")
                bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                    QProgressBar::chunk{{background:#e08030;border-radius:4px;}}""")
            if count < 50: all_done = False

        if all_done:
            self._calib_timer2.stop()
            for cap in caps.values(): cap.release()
            self._finish_calibration(positions, cam_map)

    def _finish_calibration(self, positions, cam_map):
        import yaml
        self.calib_btn.setText("▶  Nochmal kalibrieren")
        self.calib_status_lbl.setText("✓  Kalibrierung abgeschlossen!")
        self.calib_status_lbl.setStyleSheet(f"font-size:14px;font-weight:600;color:{COLORS['g2']};")
        self.calib_apply_btn.setEnabled(True)

        result = {}
        for idx, name in cam_map.items():
            pts = positions.get(name, [])
            if not pts: continue
            arr = np.array(pts)
            med = np.median(arr, axis=0)
            result[name] = {
                'index': idx, 'position_m': med.tolist(),
                'distance_cm': round(float(np.linalg.norm(med))*100, 1),
                'n_frames': len(pts),
            }

        lines = [f"{n}: {d['distance_cm']:.0f}cm ({d['n_frames']} Frames)"
                 for n, d in result.items()]
        self.calib_result_lbl.setText("\n".join(lines))
        self.calib_positions = result

        config = {
            'calibration': {'timestamp': time.time(), 'marker_size_m': 0.19},
            'cameras': {k: {'index': v['index'], 'position_m': v['position_m'],
                            'distance_cm': v['distance_cm'], 'n_frames': v['n_frames']}
                        for k, v in result.items()}
        }
        with open(Path("~/anthro3d/cam_positions.yaml").expanduser(), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Kalibrierung gespeichert")

    def _on_calib_progress(self, counts):
        for name, data in counts.items():
            if name in self.calib_bars:
                bar, lbl = self.calib_bars[name]
                count    = data['count']
                detected = data['detected']
                bar.setValue(min(count, 50))
                if detected:
                    lbl.setText(str(count))
                    lbl.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};")
                    bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                        QProgressBar::chunk{{background:{COLORS['g2']};border-radius:4px;}}""")
                else:
                    lbl.setText("⚠")
                    lbl.setStyleSheet("font-size:12px;color:#c04040;")
                    bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                        QProgressBar::chunk{{background:#c08040;border-radius:4px;}}""")

    def _on_calib_finished(self, positions):
        self.calib_positions = positions
        self.calib_btn.setText("▶  Nochmal kalibrieren")
        self.calib_status_lbl.setText("✓  Kalibrierung abgeschlossen!")
        self.calib_status_lbl.setStyleSheet(f"font-size:14px;font-weight:600;color:{COLORS['g2']};")
        self.calib_apply_btn.setEnabled(True)

        # Alle Balken auf 100% setzen
        for name, (bar, lbl) in self.calib_bars.items():
            if name in positions:
                bar.setValue(50)
                lbl.setText(str(positions[name].get('n_frames', 50)))
                bar.setStyleSheet(f"""QProgressBar{{background:{COLORS['bg']};border-radius:4px;border:none;}}
                    QProgressBar::chunk{{background:{COLORS['g2']};border-radius:4px;}}""")

        # Ergebnis anzeigen
        lines = []
        for name, data in positions.items():
            lines.append(f"{name}: {data['distance_cm']:.0f}cm  ({data['n_frames']} Frames)")
        self.calib_result_lbl.setText("\n".join(lines))

        # Automatisch speichern
        import yaml, time
        config = {
            'calibration': {'timestamp': time.time(), 'marker_size_m': 0.19},
            'cameras': {k: {'index': v['index'], 'position_m': v['position_m'],
                            'distance_cm': v['distance_cm'], 'n_frames': v['n_frames']}
                        for k, v in positions.items()}
        }
        path = Path("~/anthro3d/cam_positions.yaml").expanduser()
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Kalibrierung gespeichert: {path}")

    def _apply_calibration(self):
        """Kalibrierung übernehmen und zum Messen wechseln."""
        if self.calib_positions:
            # px_per_cm aus Median-Abstand berechnen
            if hasattr(self, 'cam_thread') and self.cam_thread:
                dists = [v['distance_cm'] for v in self.calib_positions.values()]
                avg = sum(dists) / len(dists) if dists else None
                print(f"Mittlerer Kameraabstand: {avg:.0f}cm" if avg else "")
        self.tabs.setCurrentIndex(0)  # Zu Messung wechseln

    def _build_camera_setup_tab(self):
        """Kamera Setup Tab — Kameras den richtigen Rollen zuordnen."""
        tab = QWidget()
        tab.setStyleSheet(f"background:{COLORS['bg']};")
        lay = QVBoxLayout(tab); lay.setContentsMargins(20,20,20,20); lay.setSpacing(12)

        # Titel
        title = QLabel("Kamera Setup")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{COLORS['g1']};")
        lay.addWidget(title)

        desc = QLabel("Weise jedem Kamera-Index die richtige Rolle zu.\nDas Live-Bild hilft dir die Kameras zu identifizieren.")
        desc.setStyleSheet(f"font-size:11px;color:{COLORS['dim']};")
        lay.addWidget(desc)

        # Kamera-Rollen
        self.cam_roles = ['OV9281 R', 'OV9281 L', 'ELP1', 'ELP2', 'Nicht verwendet']

        # Grid für Kameras
        self.setup_rows = []
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;")
        grid_widget = QWidget()
        grid_widget.setStyleSheet(f"background:{COLORS['bg']};")
        grid_lay = QVBoxLayout(grid_widget); grid_lay.setSpacing(8)

        for i in range(6):
            row = QFrame()
            row.setStyleSheet(f"background:{COLORS['surface']};border:1px solid {COLORS['border']};border-radius:8px;")
            row_lay = QHBoxLayout(row); row_lay.setContentsMargins(12,8,12,8); row_lay.setSpacing(12)

            # Index Label
            idx_lbl = QLabel(f"Index {i}")
            idx_lbl.setFixedWidth(55)
            idx_lbl.setStyleSheet(f"font-size:12px;font-weight:600;color:{COLORS['text']};")
            row_lay.addWidget(idx_lbl)

            # Live Vorschau
            preview = QLabel("Nicht verfügbar")
            preview.setFixedSize(160, 90)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setStyleSheet(f"background:{COLORS['bg']};border:1px solid {COLORS['border']};border-radius:4px;font-size:9px;color:{COLORS['muted']};")
            preview.setCursor(Qt.CursorShape.PointingHandCursor)
            preview.setProperty("cam_index", i)
            preview.installEventFilter(self)
            row_lay.addWidget(preview)

            # Auflösung
            res_lbl = QLabel("—")
            res_lbl.setFixedWidth(100)
            res_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['dim']};")
            row_lay.addWidget(res_lbl)

            # Dropdown Rolle
            combo = QComboBox()
            combo.addItems(self.cam_roles)
            combo.setFixedWidth(160)
            combo.setStyleSheet(f"""QComboBox{{background:{COLORS['bg']};border:1px solid {COLORS['border']};
                border-radius:6px;padding:4px 8px;font-size:11px;color:{COLORS['text']};}}
                QComboBox:hover{{border-color:{COLORS['g2']};}}""")
            # Standard-Zuweisung aus config.yaml laden
            default_map = {0:'OV9281 R', 1:'OV9281 L', 2:'ELP1', 3:'ELP2'}
            # Gespeicherte config laden
            try:
                import yaml as _yaml
                with open(Path("~/anthro3d/config.yaml").expanduser()) as _f:
                    _cfg = _yaml.safe_load(_f)
                for _cam in _cfg.get('cameras',{}).get('tracking',[]):
                    default_map[_cam['device_index']] = _cam['name']
            except Exception:
                pass
            if i in default_map and default_map[i] in self.cam_roles:
                combo.setCurrentIndex(self.cam_roles.index(default_map[i]))
            else:
                combo.setCurrentIndex(4)  # Nicht verwendet
            row_lay.addWidget(combo)

            # ArUco Status Label
            aruco_lbl = QLabel("—")
            aruco_lbl.setFixedWidth(180)
            aruco_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['dim']};")
            row_lay.addWidget(aruco_lbl)
            row_lay.addStretch()

            self.setup_rows.append({'preview': preview, 'res': res_lbl, 'combo': combo, 'index': i, 'idx_lbl': idx_lbl, 'frame': row, 'aruco_lbl': aruco_lbl})
            grid_lay.addWidget(row)

        grid_lay.addStretch()
        scroll.setWidget(grid_widget)
        lay.addWidget(scroll, 1)

        # Buttons
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("↺ Kameras suchen")
        refresh_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['bg']};color:{COLORS['dim']};
            border:1px solid {COLORS['border']};border-radius:8px;padding:8px 20px;font-size:11px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        refresh_btn.clicked.connect(self._refresh_camera_setup)

        expose_btn = QPushButton("⚡ Belichtung angleichen")
        expose_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['surface']};color:{COLORS['text']};
            border:1px solid {COLORS['border']};border-radius:8px;padding:8px 20px;font-size:11px;}}
            QPushButton:hover{{border-color:{COLORS['g2']};color:{COLORS['g1']};}}""")
        expose_btn.clicked.connect(self._auto_expose_cameras)

        save_btn = QPushButton("✓ Speichern")
        save_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
            border:none;border-radius:8px;padding:8px 24px;font-size:12px;font-weight:600;}}
            QPushButton:hover{{background:{COLORS['g2']};}}""")
        save_btn.clicked.connect(self._save_camera_setup)

        self.setup_status = QLabel("")
        self.setup_status.setStyleSheet(f"font-size:11px;color:{COLORS['g2']};")

        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(expose_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.setup_status)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        # Timer für Live-Vorschau
        self.setup_timer = QTimer()
        self.setup_timer.timeout.connect(self._update_setup_previews)
        self.setup_caps = {}

        # Vorschau starten wenn Tab gewechselt
        self.tabs.currentChanged.connect(self._on_tab_changed)

        return tab

    def _on_tab_changed(self, idx):
        if idx == 2:  # Kamera Setup Tab (Index 2)
            # CameraThread pausieren
            if hasattr(self, 'cam_thread') and self.cam_thread:
                self.cam_thread.running = False
                import time as _t; _t.sleep(0.3)
            self._refresh_camera_setup()
            self.setup_timer.start(500)
        else:
            self.setup_timer.stop()
            for cap in self.setup_caps.values(): cap.release()
            self.setup_caps.clear()
            # CameraThread wieder starten
            if hasattr(self, 'cam_thread') and self.cam_thread:
                self.cam_thread.running = True

    def _refresh_camera_setup(self):
        """Scannt alle verfügbaren Kameras — OV9281 zuerst."""
        for cap in self.setup_caps.values(): cap.release()
        self.setup_caps.clear()

        # Alle Kameras scannen und OV9281 erkennen
        found = []  # (index, w, h, is_ov9281)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # OV9281 erkennen: monochrom (B=G=R)
            for _ in range(3): cap.read()
            ret, frame = cap.read()
            is_ov = False
            if ret and frame is not None:
                b, g, r = cv2.split(frame)
                diff = float(np.mean(np.abs(b.astype(int) - r.astype(int))))
                is_ov = diff < 2.0
            cap.release()
            found.append((i, w, h, is_ov))

        # Gespeicherte Zuweisung aus config laden
        saved_map = {}  # {device_index: role_name}
        try:
            import yaml as _yaml
            with open(Path("~/anthro3d/config.yaml").expanduser()) as _f:
                _cfg = _yaml.safe_load(_f)
            for _cam in _cfg.get('cameras',{}).get('tracking',[]):
                saved_map[_cam['device_index']] = _cam['name']
        except Exception:
            pass

        # OV9281 zuerst sortieren
        found.sort(key=lambda x: (0 if x[3] else 1, x[0]))

        # Rows aktualisieren
        for row_idx, row in enumerate(self.setup_rows):
            if row_idx < len(found):
                idx, w, h, is_ov = found[row_idx]
                row['index'] = idx
                row['idx_lbl'].setText(f"Index {idx}")
                label = f"{w}×{h}"
                if is_ov:
                    label += " OV9281"
                    row['res'].setStyleSheet(f"font-size:10px;color:{COLORS['g2']};font-weight:600;")
                else:
                    label += " Webcam"
                    row['res'].setStyleSheet(f"font-size:10px;color:{COLORS['dim']};")
                row['res'].setText(label)
                row['frame'].setVisible(True)

                # Gespeicherte Zuweisung wiederherstellen
                if idx in saved_map and saved_map[idx] in self.cam_roles:
                    row['combo'].setCurrentIndex(self.cam_roles.index(saved_map[idx]))
                elif not is_ov:
                    row['combo'].setCurrentIndex(4)  # Webcam = Nicht verwendet

                # Kamera öffnen für Vorschau
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self.setup_caps[idx] = cap
                    # Preview property auf echten device index setzen
                    for row in self.setup_rows:
                        if row['index'] == idx:
                            row['preview'].setProperty("cam_index", idx)
            else:
                row['frame'].setVisible(False)

    def _update_setup_previews(self):
        """Aktualisiert Live-Vorschau im Setup Tab — nimmt kurze Snapshots."""
        for row in self.setup_rows:
            i = row['index']
            cap = self.setup_caps.get(i)
            if cap is None: continue
            ret, frame = cap.read()
            if not ret:
                # Kamera kurz neu öffnen
                cap.release()
                cap2 = cv2.VideoCapture(i)
                if cap2.isOpened():
                    ret, frame = cap2.read()
                    cap2.release()
                if not ret: continue
            frame = cv2.resize(frame, (160, 90))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg  = QImage(rgb.data, 160, 90, 160*3, QImage.Format.Format_RGB888)
            row['preview'].setPixmap(QPixmap.fromImage(qimg))
            row['preview'].setText("")

            # ArUco-Erkennung auf vollem Frame
            role = row['combo'].currentText()
            required = {'ELP2':[2,3],'ELP1':[2,10],'OV9281 L':[3,10],'OV9281 R':[3,10]}.get(role)
            if required and 'aruco_lbl' in row:
                try:
                    import cv2 as _cv2
                    _aruco_dict = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_ARUCO_ORIGINAL)
                    _params = _cv2.aruco.DetectorParameters()
                    _params.minMarkerPerimeterRate = 0.05  # Mindestgröße
                    _params.maxMarkerPerimeterRate = 0.5
                    _detector = _cv2.aruco.ArucoDetector(_aruco_dict, _params)
                    VALID_IDS = {2, 3, 4, 20, 30, 40}  # Nur Stativ-Marker
                    cap2 = self.setup_caps.get(i)
                    if cap2:
                        _ret, _fr = cap2.read()
                        if _ret:
                            _disp = _fr[:, :_fr.shape[1]//2] if _fr.shape[1] > 1600 else _fr
                            _gray = _cv2.cvtColor(_disp, _cv2.COLOR_BGR2GRAY)
                            _, _ids, _ = _detector.detectMarkers(_gray)
                            # Nur valide Marker (bekannte IDs + Mindestgröße)
                            _seen = []
                            if _ids is not None:
                                for _c, _mid in zip(_corners, _ids.flatten()):
                                    if int(_mid) not in VALID_IDS: continue
                                    _pts = _c[0]
                                    _area = _cv2.contourArea(_pts)
                                    if _area < 80*80: continue
                                    _w = float(np.linalg.norm(_pts[0]-_pts[1]))
                                    _h = float(np.linalg.norm(_pts[1]-_pts[2]))
                                    if _h > 0 and 0.7 < _w/_h < 1.3:
                                        _seen.append(int(_mid))
                            _ok = all(r in _seen for r in required)
                            _text = f"✓ ID {_seen}" if _ok else f"✗ fehlt {[r for r in required if r not in _seen]}"
                            _color = COLORS['g1'] if _ok else '#E05252'
                            row['aruco_lbl'].setText(_text)
                            row['aruco_lbl'].setStyleSheet(f"font-size:10px;color:{_color};font-weight:{'600' if _ok else '400'};")
                except Exception:
                    pass
            elif 'aruco_lbl' in row and row['combo'].currentText() == 'Nicht verwendet':
                row['aruco_lbl'].setText("—")
                row['aruco_lbl'].setStyleSheet(f"font-size:10px;color:{COLORS['dim']};")

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.MouseButtonPress:
            idx = obj.property("cam_index")
            if idx is not None:
                self._open_cam_fullscreen(idx)
                return True
        return super().eventFilter(obj, event)

    def _open_cam_fullscreen(self, idx):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel as _QL, QPushButton as _PB
        from PyQt6.QtCore import QTimer as _QT
        import cv2 as _cv2, numpy as _np
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Kamera Index {idx} — Live")
        dlg.resize(820, 640)
        dlg.setStyleSheet("background:#111;")
        vlay = QVBoxLayout(dlg); vlay.setContentsMargins(8,8,8,8); vlay.setSpacing(6)
        lbl = _QL()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("background:#000;border-radius:4px;")
        lbl.setScaledContents(False)
        vlay.addWidget(lbl, 1)
        info = _QL("ArUco: —"); info.setStyleSheet("color:#1D9E75;font-size:12px;font-weight:600;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlay.addWidget(info)
        close_btn = _PB("✕ Schließen")
        close_btn.setStyleSheet("background:#1D9E75;color:white;border:none;border-radius:6px;padding:6px 20px;font-size:12px;")
        close_btn.clicked.connect(dlg.close)
        vlay.addWidget(close_btn)
        aruco_dict = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_ARUCO_ORIGINAL)
        detector = _cv2.aruco.ArucoDetector(aruco_dict, _cv2.aruco.DetectorParameters())
        cap = self.setup_caps.get(idx)
        _own_cap = False
        if cap is None:
            cap = _cv2.VideoCapture(idx); _own_cap = True
        def _update():
            if not cap or not cap.isOpened(): return
            ret, frame = cap.read()
            if not ret: return
            # ELP: L+R nebeneinander anzeigen, OV9281/Webcam: volles Bild
            if frame.shape[1] > 1600:
                fl = frame[:, :frame.shape[1]//2]
                fr = frame[:, frame.shape[1]//2:]
                # Beide Hälften verkleinern und nebeneinander zeigen
                h = fl.shape[0]
                fl_s = _cv2.resize(fl, (640, int(640*h/fl.shape[1])))
                fr_s = _cv2.resize(fr, (640, int(640*h/fr.shape[1])))
                sep = _np.zeros((fl_s.shape[0], 4, 3), dtype=_np.uint8)
                disp = _np.hstack([fl_s, sep, fr_s])
            else:
                disp = frame.copy()
            # Sicherstellen dass Bild nicht zu groß für Fenster
            max_w, max_h = 800, 560
            h, w = disp.shape[:2]
            if w > max_w or h > max_h:
                scale = min(max_w/w, max_h/h)
                disp = cv2.resize(disp, (int(w*scale), int(h*scale)))
            # ArUco: bei ELP nur linke Hälfte, sonst volles Bild
            aruco_frame = disp[:, :disp.shape[1]//2] if disp.shape[1] > 1600 else disp
            gray = _cv2.cvtColor(aruco_frame, _cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                _cv2.aruco.drawDetectedMarkers(disp, corners, ids)
                info.setText(f"ArUco sichtbar: {ids.flatten().tolist()}")
                info.setStyleSheet("color:#1D9E75;font-size:12px;font-weight:700;")
            else:
                info.setText("ArUco: kein Marker sichtbar")
                info.setStyleSheet("color:#888;font-size:12px;")
            h, w = disp.shape[:2]
            tw, th = lbl.width() or 800, lbl.height() or 560
            scale = min(tw/w, th/h)
            nw, nh = int(w*scale), int(h*scale)
            disp = _cv2.resize(disp, (nw, nh))
            rgb = _cv2.cvtColor(disp, _cv2.COLOR_BGR2RGB)
            from PyQt6.QtGui import QImage as _QI, QPixmap as _QP
            qimg = _QI(rgb.data, nw, nh, nw*3, _QI.Format.Format_RGB888)
            lbl.setPixmap(_QP.fromImage(qimg))
        timer = _QT(dlg); timer.timeout.connect(_update); timer.start(66)
        def _on_close():
            timer.stop()
            if _own_cap and cap: cap.release()
        dlg.finished.connect(_on_close)
        dlg.exec()

    def _auto_expose_cameras(self):
        """Belichtung, Kontrast und Helligkeit aller Kameras automatisch angleichen."""
        self.setup_status.setText("Messe Belichtung...")

        # Frames von allen Kameras lesen und Helligkeit messen
        brightness = {}
        for row in self.setup_rows:
            idx = row['index']
            cap = self.setup_caps.get(idx)
            if cap is None: continue
            samples = []
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    samples.append(float(np.mean(gray)))
            if samples:
                brightness[idx] = np.mean(samples)

        if not brightness:
            self.setup_status.setText("Fehler — keine Kameras")
            return

        # Ziel-Helligkeit = Median aller Kameras
        target = float(np.median(list(brightness.values())))
        print(f"Ziel-Helligkeit: {target:.1f}")

        # Belichtung pro Kamera anpassen
        for idx, bright in brightness.items():
            cap = self.setup_caps.get(idx)
            if cap is None: continue

            # Auto-Exposure ausschalten
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

            # Aktuelle Exposure lesen
            current_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
            if current_exp <= 0: current_exp = 100

            # Neue Exposure berechnen (linear)
            if bright > 0:
                new_exp = current_exp * (target / bright)
                new_exp = max(10, min(500, new_exp))
                cap.set(cv2.CAP_PROP_EXPOSURE, new_exp)
                print(f"  Index {idx}: {bright:.1f} → {target:.1f}, exp {current_exp:.0f}→{new_exp:.0f}")

            # Kontrast und Helligkeit normalisieren
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            cap.set(cv2.CAP_PROP_CONTRAST, 128)

        # Kurz warten dann Status zeigen
        QTimer.singleShot(500, lambda: self.setup_status.setText(
            f"✓ {len(brightness)} Kameras angeglichen (Ziel: {target:.0f})"))

    def _save_camera_setup(self):
        """Speichert Kamera-Zuordnung in config.yaml."""
        import yaml
        role_to_id = {'OV9281 R': 1, 'OV9281 L': 1, 'ELP1': 2, 'ELP2': 3}
        tracking = []
        for row in self.setup_rows:
            role = row['combo'].currentText()
            if role == 'Nicht verwendet': continue
            tracking.append({
                'id':           role_to_id.get(role, 0),
                'device_index': row['index'],
                'enabled':      True,
                'name':         role,
            })
        cfg = {
            'calibration': {
                'marker_size_cm':   10.0,
                'stability_frames': 30,
                'ema_alpha':        0.3,
            },
            'cameras': {'tracking': tracking}
        }
        config_path = Path("~/anthro3d/config.yaml").expanduser()
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        self.setup_status.setText(f"✓ Gespeichert — {len(tracking)} Kameras")
        print(f"Kamera Setup gespeichert: {config_path}")

    def _section_label(self,lay,txt):
        l=QLabel(txt)
        l.setStyleSheet(f"font-size:9px;color:{COLORS['muted']};letter-spacing:1px;margin-top:6px;")
        lay.addWidget(l)

    def _r_section(self,lay,txt):
        f=QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color:{COLORS['border']};margin-top:8px;")
        lay.addWidget(f)
        l=QLabel(txt)
        l.setStyleSheet(f"font-size:11px;font-weight:600;color:{COLORS['text']};margin:6px 0 4px 0;")
        lay.addWidget(l)

    def _start_camera(self):
        self.cam_thread=CameraThread()
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.start()
        self.status_lbl.setText("● Live")
        self.status_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};")

    @pyqtSlot(object,object,object)
    def _on_frame(self,frame,pts,meas):
        self._last_frame = frame
        # Button aktivieren wenn erster Frame kommt
        if not self.rec_btn.isEnabled():
            self.rec_btn.setEnabled(True)
            self.rec_btn.setText('● Aufnahme starten')
        # Log schreiben
        if not hasattr(self,'_fc2'): self._fc2=0
        self._fc2+=1
        if self._fc2%30==0:
            v=getattr(getattr(self,'_video3d',None),'recording','NO')
            msg='f='+str(self._fc2)+' rec='+str(v)+' en='+str(self.rec_btn.isEnabled())+'\n'
            try:
                if not hasattr(self,'_log_f'):
                    self._log_f=open('/Users/alexanderpillgruber/anthro3d/debug.log','w',buffering=1)
                self._log_f.write(msg)
            except Exception:
                pass
        # 3D Video Frames sammeln
        if hasattr(self,'_video3d') and self._video3d and self._video3d.recording:
            self._video3d.add_frame(frame, pts)
        px=self.cam_thread.px_per_cm
        self.cam_widget.update_frame(frame,pts,meas,px)
        self.fps_lbl.setText(f"{self.cam_thread.fps_val:.0f} fps")

        # Messwerte
        h=meas.get('height')
        if h:
            self.val_height.setText(f"{h:.1f} cm")
            self.val_height.setStyleSheet(f"font-size:36px;font-weight:800;color:{COLORS['g1']};")
            self.val_height_mode.setText("2D · ArUco Kalibrierung")
        else:
            self.val_height.setText("—")

        for key,card in self.cards.items():
            v=meas.get(key)
            unit='°' if 'ang' in key else 'cm'
            card.set_value(v,unit)

        # ArUco Status
        if px:
            self.aruco_lbl.setText(f"ArUco ✓  {px:.1f} px/cm")
            self.aruco_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['g2']};")
        else:
            self.aruco_lbl.setText("ArUco: ID1 vor Kamera halten")
            self.aruco_lbl.setStyleSheet(f"font-size:10px;color:{COLORS['warn']};")

        # Info
        n=len(pts)
        k="EIN" if self.cam_thread.smoother.enabled else "AUS"
        self.info_lbl.setText(f"Modus: 2D\nPunkte: {n}/17\nKalman: {k}")

    def _toggle_cam_fullscreen(self):
        if self.cam_widget.parent() == self:
            # Vollbild — als eigenes Fenster
            self.cam_widget.setParent(None)
            self.cam_widget.setWindowTitle("ANTHRO3D — Vollbild")
            self.cam_widget.showFullScreen()
            self.btn_fullscreen.setText("✕")
            self.btn_fullscreen.setToolTip("Vollbild beenden (F oder ESC)")
        else:
            # Zurück in Layout
            self.cam_widget.showNormal()
            self.cam_widget.setParent(None)
            # Neu einfügen
            self._reinsert_cam_widget()
            self.btn_fullscreen.setText("⛶")
            self.btn_fullscreen.setToolTip("Vollbild (F)")

    def _reinsert_cam_widget(self):
        # Kamera-Widget zurück ins center Layout einfügen
        center = self.centralWidget().layout().itemAt(1).widget()
        lay = center.layout()
        lay.insertWidget(1, self.cam_widget, 1)

    def _toggle_clinical_bg(self, checked=None):
        if checked is None:
            self.clinical_bg = not self.clinical_bg
        else:
            self.clinical_bg = checked

    def _toggle_rec(self):
        if self.recording:
            # STOPP
            self.recording = False
            self.rec_timer.stop()
            self.rec_btn.setText("● Aufnahme starten")
            self.rec_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['g1']};color:white;
                border:none;border-radius:20px;font-size:12px;font-weight:500;padding:8px 24px;}}
                QPushButton:hover{{background:{COLORS['g2']};}}""")
            self.rec_time_lbl.setText("")
            if hasattr(self, '_video3d') and self._video3d:
                self._video3d.recording = False
                n = len(self._video3d.frames)
                print(f"■ Aufnahme gestoppt — {n} Frames")
                if n > 0:
                    self._open_video3d_viewer()
            self._check_for_ply()
        else:
            # START
            self.recording = True
            self.rec_start = time.time()
            self.rec_btn.setText("■ Aufnahme stoppen")
            self.rec_btn.setStyleSheet(f"""QPushButton{{background:{COLORS['red']};color:white;
                border:none;border-radius:20px;font-size:12px;font-weight:500;padding:8px 24px;}}""")
            self.rec_timer.start(1000)
            self._video3d = Video3DRecorder()
            self._video3d.recording = True
            print(f"● Aufnahme gestartet — recording={self._video3d.recording}")

    def _open_video3d_viewer(self):
        """Öffnet den 3D Video Viewer nach der Aufnahme."""
        self._v3d_win = Video3DViewer(self._video3d)
        self._v3d_win.show()
        # Automatisch Patienten-Dialog öffnen
        QTimer.singleShot(500, self._open_patient_dialog)

    def _open_patient_dialog(self):
        """Öffnet Patienten-Dialog zum Speichern der Aufnahme."""
        try:
            from patient_dialog import PatientDialog
            meas = getattr(self, '_last_measurements', {})
            dlg = PatientDialog(
                parent=self,
                video3d=self._video3d,
                measurements=meas,
            )
            dlg.patient_selected.connect(self._on_patient_saved)
            dlg.exec()
        except Exception as e:
            print(f"Patienten-Dialog Fehler: {e}")

    def _on_patient_saved(self, patient):
        """Wird aufgerufen wenn Aufnahme für Patient gespeichert."""
        print(f"Aufnahme gespeichert für: {patient.vollname}")

    def _open_patient_list(self):
        """Öffnet Patientenliste."""
        try:
            from patient_dialog import PatientListDialog
            dlg = PatientListDialog(parent=self)
            dlg.exec()
        except Exception as e:
            print(f"Patientenliste Fehler: {e}")

    def _update_rec_time(self):
        dur=time.time()-self.rec_start
        m,s=int(dur//60),int(dur%60)
        self.rec_time_lbl.setText(f"{m}:{s:02d}")

    def _check_for_ply(self):
        """Prüft ob PLY Dateien vorhanden sind und aktiviert 3D Button."""
        import glob
        plys = glob.glob(str(Path("~/anthro3d/scan_*.ply").expanduser()))
        if plys:
            self._latest_ply = sorted(plys)[-1]
            if hasattr(self, 'mesh3d_btn'):
                self.mesh3d_btn.setEnabled(True)
            if hasattr(self, 'mesh3d_btn_sb'):
                self.mesh3d_btn_sb.setEnabled(True)
                self.mesh3d_btn_sb.setText(f"🔲 3D Mesh ({len(plys)} Scans)")

    def _show_3d_mesh(self):
        """Zeigt 3D Mesh mit Kamera-Textur als PyQt6 Fenster."""
        ply = getattr(self, '_latest_ply', None)
        if not ply:
            import glob
            plys = sorted(glob.glob(str(Path("~/anthro3d/scan_*.ply").expanduser())))
            if not plys: print("Keine PLY Datei"); return
            ply = plys[-1]
        # Letzten Frame als Textur mitgeben
        frame = getattr(self, '_last_frame', None)
        self._mesh_win = MeshViewer3D(ply, texture_frame=frame)
        self._mesh_win.show()

    def _snapshot(self):
        if self.cam_widget.pixmap:
            path=SAVE_DIR/f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            SAVE_DIR.mkdir(parents=True,exist_ok=True)
            self.cam_widget.pixmap.save(str(path))
            print(f"Snapshot: {path}")

    def _toggle_lang(self):
        self.lang='en' if self.lang=='de' else 'de'
        L=LABELS[self.lang]
        self.lbl_height.setText(L['height'])
        for key,card in self.cards.items():
            if key in L: card.lbl.setText(L[key])

    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_Q: self.close()
        elif e.key()==Qt.Key.Key_F: self._toggle_cam_fullscreen()
        elif e.key()==Qt.Key.Key_Escape:
            if self.cam_widget.parent() != self: self._toggle_cam_fullscreen()
        elif e.key()==Qt.Key.Key_K: self.btn_kalm.setChecked(not self.btn_kalm.isChecked())
        elif e.key()==Qt.Key.Key_S: self.btn_skel.setChecked(not self.btn_skel.isChecked())
        elif e.key()==Qt.Key.Key_V: self.btn_clin.setChecked(not self.btn_clin.isChecked())
        elif e.key()==Qt.Key.Key_Space: self._toggle_rec()

    def closeEvent(self,e):
        self.cam_thread.stop()
        self.cam_thread.wait()
        super().closeEvent(e)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setApplicationName("ANTHRO3D")
    win=MainWindow()
    win.show()
    sys.exit(app.exec())
