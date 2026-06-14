
import cv2, numpy as np, sys, math
import yaml
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor

with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)

def make_maps(cfg, size=(1600,1200)):
    K_l=np.array(cfg["camera_matrix_l"]); d_l=np.array(cfg["dist_l"])
    K_r=np.array(cfg["camera_matrix_r"]); d_r=np.array(cfg["dist_r"])
    R=np.array(cfg["R"]); T=np.array(cfg["T"])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,alpha=0.5)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    return ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0])

ml1,ml2,mr1,mr2,fx,cx,cy,bl = make_maps(cfg1)

lm = cv2.StereoSGBM_create(minDisparity=4,numDisparities=128,blockSize=9,
    P1=8*3*81,P2=32*3*81,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=150,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm = cv2.ximgproc.createRightMatcher(lm)
wls = cv2.ximgproc.createDisparityWLSFilter(lm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)

print("SPACE=Scan | Q=Beenden")
pts = cols = None

while True:
    ret,frame = cap.read()
    if not ret: continue
    w = frame.shape[1]//2
    fl = cv2.remap(frame[:,:w], ml1, ml2, cv2.INTER_LINEAR)
    fr = cv2.remap(frame[:,w:], mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    dl = lm.compute(gl,gr); dr = rm.compute(gr,gl)
    d = wls.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4]=0
    dv = (np.clip(d,0,128)/128*255).astype(np.uint8)
    cv2.imshow("ELP1 RAW", cv2.resize(cv2.applyColorMap(dv,cv2.COLORMAP_JET),(800,600)))
    key = cv2.waitKey(1)&0xFF
    if key==ord("q"): break
    if key==ord(" "):
        rows,ci = np.where(d>4)
        dv2 = d[rows,ci]
        Z=fx*bl/dv2; X=(ci-cx)*Z/fx; Y=(rows-cy)*Z/fx
        pts = np.stack([X,Y,Z],axis=1)
        cols = cv2.cvtColor(fl,cv2.COLOR_BGR2RGB)[rows,ci]
        print(f"ELP1 RAW: {len(pts)} Punkte")
        print(f"X: {pts[:,0].min()*100:.0f}–{pts[:,0].max()*100:.0f}cm")
        print(f"Y: {pts[:,1].min()*100:.0f}–{pts[:,1].max()*100:.0f}cm")
        print(f"Z: {pts[:,2].min()*100:.0f}–{pts[:,2].max()*100:.0f}cm")
        break

cap.release(); cv2.destroyAllWindows()

if pts is not None:
    pts[:,1]=-pts[:,1]
    pts[:,0]-=pts[:,0].mean(); pts[:,1]-=pts[:,1].mean(); pts[:,2]-=pts[:,2].mean()
    step=max(1,len(pts)//15000); pts=pts[::step]; cols=cols[::step]

    class Viewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ELP1 RAW")
            self.resize(900,900)
            self.rx=15; self.ry=0; self.last=None
            self.scale=200
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
        def wheelEvent(self,e): self.scale+=e.angleDelta().y()//10; self.update()

    app=QApplication(sys.argv); w=Viewer(); w.show(); app.exec()
