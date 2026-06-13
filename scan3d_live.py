import cv2, numpy as np, yaml, sys, threading, time
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import QTimer
import math
import mediapipe as mp

# ── Konfiguration ──────────────────────────────────────────────
Z_MIN, Z_MAX = 1.0, 2.8      # Tiefenbereich in Metern
VOXEL = 0.05                  # 3cm Voxel-Gitter
TARGET_FPS = 5                # Ziel-Framerate

# ── Kalibrierungen laden ───────────────────────────────────────
with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)

R_rel_elp1 = np.load(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser())
T_rel_elp1 = np.load(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser())

def make_maps(cfg, size=(1600,1200)):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,alpha=0.5)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    return ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0])

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2 = make_maps(cfg2)
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1 = make_maps(cfg1)

# ── SGBM ──────────────────────────────────────────────────────
lm=cv2.StereoSGBM_create(minDisparity=4,numDisparities=96,blockSize=7,
    P1=8*3*49,P2=32*3*49,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm=cv2.ximgproc.createRightMatcher(lm)
wls=cv2.ximgproc.createDisparityWLSFilter(lm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

# ── MediaPipe Segmentierung ────────────────────────────────────
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _opt = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(Path("~/anthro3d/selfie_segmenter.tflite").expanduser())),
        output_category_mask=True)
    seg2 = mp_vision.ImageSegmenter.create_from_options(_opt)
    seg1 = mp_vision.ImageSegmenter.create_from_options(_opt)
    USE_MP = True
    print("MediaPipe Tasks API aktiv")
except Exception as e:
    seg2 = seg1 = None
    USE_MP = False
    print(f"MediaPipe nicht verfügbar ({e}) — Z-Clipping aktiv")

# ── Kameras ───────────────────────────────────────────────────
cap2=cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
cap1=cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)

def get_pts(cap, ml1, ml2, mr1, mr2, seg, fx, cx, cy, bl, R_rel=None, T_rel=None):
    ret,frame=cap.read()
    if not ret: return None,None
    w=frame.shape[1]//2
    fl=cv2.remap(frame[:,:w],ml1,ml2,cv2.INTER_LINEAR)
    fr=cv2.remap(frame[:,w:],mr1,mr2,cv2.INTER_LINEAR)
    # Helligkeit erhöhen
    fl_bright = cv2.convertScaleAbs(fl, alpha=2.0, beta=20)
    gl=cv2.cvtColor(fl_bright,cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(cv2.convertScaleAbs(fr,alpha=2.0,beta=20),cv2.COLOR_BGR2GRAY)
    dl=lm.compute(gl,gr); dr=rm.compute(gr,gl)
    d=wls.filter(dl,fl_bright,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4]=0
    # MediaPipe Maske oder volles Bild
    if seg is not None and USE_MP:
        import mediapipe as _mp
        rgb=cv2.cvtColor(fl_bright,cv2.COLOR_BGR2RGB)
        mp_img=_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        result=seg.segment(mp_img)
        mask_data=result.category_mask.numpy_view()
        mask=(mask_data.squeeze()>0).astype(np.uint8)
        rows,ci=np.where((mask>0)&(d>4))
    else:
        rows,ci=np.where(d>4)
    if len(rows)==0: return None,None
    dv=d[rows,ci]
    Z=fx*bl/dv; X=(ci-cx)*Z/fx; Y=(rows-cy)*Z/fx
    pts=np.stack([X,Y,Z],axis=1)
    cols=cv2.cvtColor(fl_bright,cv2.COLOR_BGR2RGB)[rows,ci]
    # Z-Clipping
    zm=(Z>Z_MIN)&(Z<Z_MAX)
    pts=pts[zm]; cols=cols[zm]
    # Transformation
    if R_rel is not None and len(pts)>0:
        pts=(R_rel @ pts.T).T + T_rel
    return pts,cols

def voxel_downsample(pts, cols, voxel_size):
    if len(pts)==0: return pts,cols
    idx=(pts/voxel_size).astype(np.int32)
    keys=idx[:,0]*1000000+idx[:,1]*1000+idx[:,2]
    _,ui=np.unique(keys,return_index=True)
    return pts[ui],cols[ui]

# ── Shared State ──────────────────────────────────────────────
lock=threading.Lock()
shared_pts=np.zeros((0,3)); shared_cols=np.zeros((0,3),dtype=np.uint8)
running=True; fps_actual=0

def scan_loop():
    global shared_pts, shared_cols, running, fps_actual
    while running:
        t0=time.time()
        p2,c2=get_pts(cap2,ml2_1,ml2_2,mr2_1,mr2_2,seg2,fx2,cx2,cy2,bl2)
        p1,c1=get_pts(cap1,ml1_1,ml1_2,mr1_1,mr1_2,seg1,fx1,cx1,cy1,bl1,R_rel_elp1,T_rel_elp1)
        parts=[]; cparts=[]
        if p2 is not None: parts.append(p2); cparts.append(c2)
        if p1 is not None: parts.append(p1); cparts.append(c1)
        if parts:
            all_pts=np.vstack(parts); all_cols=np.vstack(cparts)
            all_pts,all_cols=voxel_downsample(all_pts,all_cols,VOXEL)
            # Zentrieren
            if len(all_pts)>0:
                all_pts[:,0]-=all_pts[:,0].mean()
                all_pts[:,1]-=all_pts[:,1].mean()
                all_pts[:,2]-=all_pts[:,2].mean()
                all_pts[:,1]=-all_pts[:,1]
            with lock:
                shared_pts=all_pts; shared_cols=all_cols
        fps_actual=1.0/(time.time()-t0+0.001)

# ── Qt Viewer ─────────────────────────────────────────────────
class LiveViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANTHRO3D Live 3D — Maus=Drehen | Scroll=Zoom | Q=Beenden")
        self.resize(900,1000); self.rx=15; self.ry=0; self.last=None; self.scale=300
        self.timer=QTimer(self); self.timer.timeout.connect(self.update); self.timer.start(100)
    def paintEvent(self,e):
        with lock:
            pts=shared_pts.copy(); cols=shared_cols.copy()
        p=QPainter(self); p.fillRect(self.rect(),QColor(20,20,20))
        if len(pts)==0:
            p.setPen(QColor(100,100,100))
            p.drawText(self.width()//2-80,self.height()//2,"Warte auf Punkte...")
            return
        cx_=self.width()//2; cy_=self.height()//2
        rx=math.radians(self.rx); ry=math.radians(self.ry)
        cX,sX=math.cos(rx),math.sin(rx); cY,sY=math.cos(ry),math.sin(ry)
        proj=[]
        for pt,c in zip(pts,cols):
            x,y,z=pt
            x2=x*cY+z*sY; z2=-x*sY+z*cY
            y2=y*cX-z2*sX; z3=y*sX+z2*cX
            proj.append((z3,int(cx_+x2*self.scale),int(cy_-y2*self.scale),c))
        proj.sort(key=lambda v:v[0])
        for _,sx,sy,c in proj:
            p.setPen(QColor(int(c[0]),int(c[1]),int(c[2]))); p.drawPoint(sx,sy)
        p.setPen(QColor(0,255,0))
        p.drawText(10,20,f"Punkte: {len(pts)} | FPS: {fps_actual:.1f}")
    def mousePressEvent(self,e): self.last=e.position()
    def mouseMoveEvent(self,e):
        if self.last:
            dx=e.position().x()-self.last.x(); dy=e.position().y()-self.last.y()
            self.ry+=dx*0.5; self.rx+=dy*0.5; self.last=e.position()
    def mouseReleaseEvent(self,e): self.last=None
    def wheelEvent(self,e):
        self.scale*=1.1 if e.angleDelta().y()>0 else 0.9
    def keyPressEvent(self,e):
        if e.text()=='q': self.close()
    def closeEvent(self,e):
        global running; running=False
        cap2.release(); cap1.release()
        if seg2: seg2.close()
        if seg1: seg1.close()

t=threading.Thread(target=scan_loop,daemon=True); t.start()
app=QApplication(sys.argv); w=LiveViewer(); w.show(); app.exec()
