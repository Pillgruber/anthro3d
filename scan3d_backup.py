import cv2, numpy as np, yaml, sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor
import math

with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)

def make_maps(cfg):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,(1600,1200),R,T,alpha=0.5)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,(1600,1200),cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,(1600,1200),cv2.CV_32F)
    return ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0])

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2=make_maps(cfg2)
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1=make_maps(cfg1)

# Relative Transformation ELP1→ELP2 laden
R_rel = np.load('/tmp/R_rel.npy')
T_rel = np.load('/tmp/T_rel.npy')
print(f"ELP1→ELP2 T: {T_rel*100} cm")

lm=cv2.StereoSGBM_create(minDisparity=4,numDisparities=128,blockSize=9,
    P1=8*3*81,P2=32*3*81,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=150,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm=cv2.ximgproc.createRightMatcher(lm)
wls=cv2.ximgproc.createDisparityWLSFilter(lm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

cap2=cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
cap1=cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)

def get_disp(cap,ml1,ml2,mr1,mr2):
    ret,frame=cap.read()
    if not ret: return None,None
    w=frame.shape[1]//2
    fl=cv2.remap(frame[:,:w],ml1,ml2,cv2.INTER_LINEAR)
    fr=cv2.remap(frame[:,w:],mr1,mr2,cv2.INTER_LINEAR)
    gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    dl=lm.compute(gl,gr); dr=rm.compute(gr,gl)
    d=wls.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4]=0
    return d,fl

def get_mask(disp,bg,stable):
    closer=((disp-bg)>4)&(disp>4)&stable
    raw=closer.astype(np.uint8)*255
    raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((30,30),np.uint8))
    raw=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((10,10),np.uint8))
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(raw)
    if n<=1: return raw
    best=-1; best_s=float('inf')
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<300: continue
        dx=centroids[i][0]-800; dy=centroids[i][1]-600
        s=np.sqrt(dx*dx+dy*dy)/np.sqrt(stats[i,cv2.CC_STAT_AREA])
        if s<best_s: best_s=s; best=i
    return (labels==best).astype(np.uint8)*255 if best>0 else raw

def disp_to_pts(disp,fl,mask,fx,cx,cy,bl):
    rows,ci=np.where(mask>0)
    if len(rows)==0: return None,None
    d=disp[rows,ci]; v=d>4
    rows=rows[v]; ci=ci[v]; d=d[v]
    if len(rows)==0: return None,None
    Z=fx*bl/d; X=(ci-cx)*Z/fx; Y=(rows-cy)*Z/fx
    pts=np.stack([X,Y,Z],axis=1)
    colors=cv2.cvtColor(fl,cv2.COLOR_BGR2RGB)
    cols=colors[rows,ci]
    z5,z95=np.percentile(Z,[5,95]); m=(Z>=z5)&(Z<=z95)
    return pts[m],cols[m]

bg2=bg1=st2=st1=None
all_pts=all_cols=None

print("Fenster anklicken → B=Hintergrund | SPACE=Scan | Q=Beenden")

while True:
    d2,fl2=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2)
    d1,fl1=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2)
    if d2 is None or d1 is None: continue

    m2=np.zeros(d2.shape,np.uint8)
    m1=np.zeros(d1.shape,np.uint8)
    if st2 is not None:
        m2=get_mask(d2,bg2,st2)
        m1=get_mask(d1,bg1,st1)

    dv2=np.clip(d2,0,128); dv2=(dv2/128*255).astype(np.uint8)
    dv1=np.clip(d1,0,128); dv1=(dv1/128*255).astype(np.uint8)
    status="B=Hintergrund" if st2 is None else f"SPACE=Scan ELP2:{m2.sum()//255} ELP1:{m1.sum()//255}"
    row1=np.hstack([cv2.resize(fl2,(480,300)),cv2.resize(cv2.applyColorMap(dv2,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m2,cv2.COLOR_GRAY2BGR),(480,300))])
    row2=np.hstack([cv2.resize(fl1,(480,300)),cv2.resize(cv2.applyColorMap(dv1,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m1,cv2.COLOR_GRAY2BGR),(480,300))])
    out=np.vstack([row1,row2])
    cv2.putText(out,status,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("ELP2 | ELP1",out)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('b'):
        print("Hintergrund aufnehmen...")
        f2s=[]; f1s=[]
        for i in range(20):
            d,_=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2)
            if d is not None: f2s.append(d)
            d,_=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2)
            if d is not None: f1s.append(d)
            if i%5==0: print(f"  {i+1}/20")
        bg2=np.median(f2s,axis=0).astype(np.float32)
        bg1=np.median(f1s,axis=0).astype(np.float32)
        st2=np.std(f2s,axis=0)<3.0; st1=np.std(f1s,axis=0)<3.0
        print(f"✓ ELP2:{st2.mean()*100:.0f}% ELP1:{st1.mean()*100:.0f}% — Person hinstellen → SPACE")
    if key==ord(' ') and st2 is not None:
        p2,c2=disp_to_pts(d2,fl2,m2,fx2,cx2,cy2,bl2)
        p1,c1=disp_to_pts(d1,fl1,m1,fx1,cx1,cy1,bl1)
        parts=[]; cparts=[]
        if p2 is not None:
            parts.append(p2); cparts.append(c2)
            print(f"ELP2: {len(p2)}")
        if p1 is not None:
            # ELP1 in ELP2 Koordinatensystem transformieren
            p1_t = (R_rel @ p1.T).T + T_rel
            parts.append(p1_t); cparts.append(c1)
            print(f"ELP1: {len(p1)} → transformiert")
        if parts:
            all_pts=np.vstack(parts); all_cols=np.vstack(cparts)
            print(f"Gesamt: {len(all_pts)}")
            print(f"Breite:{(all_pts[:,0].max()-all_pts[:,0].min())*100:.0f}cm Höhe:{(all_pts[:,1].max()-all_pts[:,1].min())*100:.0f}cm")
            break

cap2.release(); cap1.release(); cv2.destroyAllWindows()

if all_pts is not None:
    pts=all_pts.copy(); cols=all_cols.copy()
    pts[:,0]-=pts[:,0].mean(); pts[:,1]-=pts[:,1].mean(); pts[:,2]-=pts[:,2].mean()
    pts[:,1]=-pts[:,1]
    step=max(1,len(pts)//10000); pts=pts[::step]; cols=cols[::step]

    class Viewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ELP1+ELP2 ausgerichtet — Maus=Drehen | Scroll=Zoom")
            self.resize(900,1000); self.rx=15; self.ry=0; self.last=None
            self.scale=300/max(pts[:,1].max()-pts[:,1].min(),0.1)
        def paintEvent(self,e):
            p=QPainter(self); p.fillRect(self.rect(),QColor(20,20,20))
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
        def mousePressEvent(self,e): self.last=e.position()
        def mouseMoveEvent(self,e):
            if self.last:
                dx=e.position().x()-self.last.x(); dy=e.position().y()-self.last.y()
                self.ry+=dx*0.5; self.rx+=dy*0.5; self.last=e.position(); self.update()
        def mouseReleaseEvent(self,e): self.last=None
        def wheelEvent(self,e):
            self.scale*=1.1 if e.angleDelta().y()>0 else 0.9; self.update()

    app=QApplication(sys.argv); w=Viewer(); w.show(); app.exec()
