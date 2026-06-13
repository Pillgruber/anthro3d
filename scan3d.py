import cv2, numpy as np, yaml, sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor
import math

with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_ov9281.yaml").expanduser()) as f:
    cfgov = yaml.safe_load(f)

def make_maps_split(cfg, size=(1600,1200)):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,alpha=0.5)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    return ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0])

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2 = make_maps_split(cfg2, (1600,1200))
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1 = make_maps_split(cfg1, (1600,1200))
mlov1,mlov2,mrov1,mrov2,fxov,cxov,cyov,blov = make_maps_split(cfgov, (1280,800))

# ELP1→ELP2 laden
R_rel_elp1 = np.load(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser())
T_rel_elp1 = np.load(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser())
print(f"ELP1→ELP2 T: {T_rel_elp1*100} cm")

# OV9281→ELP2 Transformation berechnen via Marker ID 10
# ELP2 sieht ID 10, OV9281-L sieht ID 10
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
MARKER_SIZE = 0.19
obj_pts = np.array([[-MARKER_SIZE/2, MARKER_SIZE/2,0],
                    [ MARKER_SIZE/2, MARKER_SIZE/2,0],
                    [ MARKER_SIZE/2,-MARKER_SIZE/2,0],
                    [-MARKER_SIZE/2,-MARKER_SIZE/2,0]], dtype=np.float32)

K2_raw  = np.array(cfg2['camera_matrix_l']);  d2_raw  = np.array(cfg2['dist_l'])
Kov_raw = np.array(cfgov['camera_matrix_l']); dov_raw = np.array(cfgov['dist_l'])

cap2_cal  = cv2.VideoCapture(0)
cap2_cal.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2_cal.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
capovL_cal = cv2.VideoCapture(1)
capovL_cal.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovL_cal.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

R_ov_list=[]; T_ov_list=[]
print("OV9281→ELP2 kalibrieren (Marker ID 10, 50 Frames)...")
for _ in range(50):
    ret2,f2   = cap2_cal.read()
    retov,fov = capovL_cal.read()
    if not ret2 or not retov: continue
    fl2 = f2[:, :f2.shape[1]//2]
    g2  = cv2.cvtColor(fl2, cv2.COLOR_BGR2GRAY)
    gov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
    # Helligkeit anpassen (OV9281 ist manchmal dunkel)
    gov = cv2.equalizeHist(gov)
    c2,ids2,_   = detector.detectMarkers(g2)
    cov,idsov,_ = detector.detectMarkers(gov)
    if ids2 is None or idsov is None: continue
    ids2f=ids2.flatten(); idsouf=idsov.flatten()
    common = set(ids2f) & set(idsouf)
    for mid in common:
        i2  = np.where(ids2f==mid)[0][0]
        iov = np.where(idsouf==mid)[0][0]
        ok2, rvec2, tvec2 = cv2.solvePnP(obj_pts, c2[i2],   K2_raw,  d2_raw)
        okv, rvecv, tvecv = cv2.solvePnP(obj_pts, cov[iov], Kov_raw, dov_raw)
        if ok2 and okv:
            R2m,_ = cv2.Rodrigues(rvec2); Rvm,_ = cv2.Rodrigues(rvecv)
            R_rel = R2m @ Rvm.T
            T_rel = tvec2.flatten() - R_rel @ tvecv.flatten()
            R_ov_list.append(R_rel); T_ov_list.append(T_rel)

cap2_cal.release(); capovL_cal.release()

if R_ov_list:
    T_arr = np.array(T_ov_list)
    R_arr = np.array(R_ov_list)
    # Ausreißer filtern
    dists = np.linalg.norm(T_arr, axis=1)
    med = np.median(dists)
    mask = dists < med * 1.3
    T_arr = T_arr[mask]; R_arr = R_arr[mask]
    # Rotation korrekt über Rodrigues-Vektoren mitteln
    import cv2 as _cv2
    rvecs = np.array([_cv2.Rodrigues(R)[0].flatten() for R in R_arr])
    rvec_mean = np.mean(rvecs, axis=0)
    R_rel_ov, _ = _cv2.Rodrigues(rvec_mean)
    T_rel_ov = np.mean(T_arr, axis=0)
    print(f"OV9281→ELP2 T: {T_rel_ov*100} cm  Distanz: {np.linalg.norm(T_rel_ov)*100:.1f} cm ({mask.sum()}/{len(mask)} gut)")
    print(f"  det(R)={np.linalg.det(R_rel_ov):.4f}")
    OV_OK = True
else:
    print("WARNUNG: OV9281→ELP2 Marker nicht gefunden — OV9281 deaktiviert")
    R_rel_ov = T_rel_ov = None; OV_OK = False

# SGBM für ELP (hohe Auflösung)
lm_elp = cv2.StereoSGBM_create(minDisparity=4,numDisparities=128,blockSize=9,
    P1=8*3*81,P2=32*3*81,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=150,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm_elp = cv2.ximgproc.createRightMatcher(lm_elp)
wls_elp = cv2.ximgproc.createDisparityWLSFilter(lm_elp)
wls_elp.setLambda(8000); wls_elp.setSigmaColor(1.5)

# SGBM für OV9281 (kleinere Auflösung, kürzere Baseline)
lm_ov = cv2.StereoSGBM_create(minDisparity=2,numDisparities=64,blockSize=7,
    P1=8*3*49,P2=32*3*49,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm_ov = cv2.ximgproc.createRightMatcher(lm_ov)
wls_ov = cv2.ximgproc.createDisparityWLSFilter(lm_ov)
wls_ov.setLambda(8000); wls_ov.setSigmaColor(1.5)

# Kameras öffnen
cap2  = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,3200);  cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
cap1  = cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,3200);  cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
capovL = cv2.VideoCapture(1)
capovL.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovL.set(cv2.CAP_PROP_FRAME_HEIGHT,800)
capovR = cv2.VideoCapture(2)
capovR.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovR.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

def get_disp_split(cap, ml1, ml2, mr1, mr2, lm_, rm_, wls_, alpha=2.5, beta=30):
    """ELP: ein Frame mit L+R nebeneinander"""
    ret,frame = cap.read()
    if not ret: return None,None
    w = frame.shape[1]//2
    fl = cv2.remap(frame[:,:w], ml1, ml2, cv2.INTER_LINEAR)
    fr = cv2.remap(frame[:,w:], mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    dl = lm_.compute(gl,gr); dr = rm_.compute(gr,gl)
    d  = wls_.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4] = 0
    return d, fl

def get_disp_dual(capL, capR, ml1, ml2, mr1, mr2, lm_, rm_, wls_):
    """OV9281: zwei separate Kamera-Feeds"""
    retL,fL = capL.read()
    retR,fR = capR.read()
    if not retL or not retR: return None,None
    fl = cv2.remap(fL, ml1, ml2, cv2.INTER_LINEAR)
    fr = cv2.remap(fR, mr1, mr2, cv2.INTER_LINEAR)
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    dl = lm_.compute(gl,gr); dr = rm_.compute(gr,gl)
    d  = wls_.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4] = 0
    return d, fl

def get_mask(disp, bg, stable, cx_img, cy_img):
    closer = ((disp-bg)>4)&(disp>4)&stable
    raw = closer.astype(np.uint8)*255
    raw = cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((30,30),np.uint8))
    raw = cv2.morphologyEx(raw,cv2.MORPH_OPEN, np.ones((10,10),np.uint8))
    n,labels,stats,centroids = cv2.connectedComponentsWithStats(raw)
    if n<=1: return raw
    best=-1; best_s=float('inf')
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<300: continue
        dx=centroids[i][0]-cx_img; dy=centroids[i][1]-cy_img
        s=np.sqrt(dx*dx+dy*dy)/np.sqrt(stats[i,cv2.CC_STAT_AREA])
        if s<best_s: best_s=s; best=i
    return (labels==best).astype(np.uint8)*255 if best>0 else raw

def disp_to_pts(disp, fl, mask, fx, cx, cy, bl):
    rows,ci = np.where(mask>0)
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

bg2=bg1=bgov=st2=st1=stov=None
all_pts=all_cols=None

print("Fenster anklicken → B=Hintergrund | SPACE=Scan | Q=Beenden")

while True:
    d2,fl2   = get_disp_split(cap2,  ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
    d1,fl1   = get_disp_split(cap1,  ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
    if OV_OK:
        dov,flov = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
    else:
        dov,flov = None,None
    if d2 is None or d1 is None: continue

    m2  = np.zeros(d2.shape,  np.uint8)
    m1  = np.zeros(d1.shape,  np.uint8)
    mov = np.zeros(dov.shape, np.uint8) if dov is not None else None
    if st2 is not None:
        m2 = get_mask(d2, bg2, st2, 800, 600)
        m1 = get_mask(d1, bg1, st1, 800, 600)
        if dov is not None and stov is not None:
            mov = get_mask(dov, bgov, stov, 640, 400)

    dv2 = (np.clip(d2,0,128)/128*255).astype(np.uint8)
    dv1 = (np.clip(d1,0,128)/128*255).astype(np.uint8)
    ov_count = int(mov.sum()//255) if mov is not None else 0
    status = "B=Hintergrund" if st2 is None else f"ELP2:{m2.sum()//255} ELP1:{m1.sum()//255} OV:{ov_count} — SPACE=Scan"
    row1=np.hstack([cv2.resize(fl2,(480,300)),cv2.resize(cv2.applyColorMap(dv2,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m2,cv2.COLOR_GRAY2BGR),(480,300))])
    row2=np.hstack([cv2.resize(fl1,(480,300)),cv2.resize(cv2.applyColorMap(dv1,cv2.COLORMAP_JET),(480,300)),cv2.resize(cv2.cvtColor(m1,cv2.COLOR_GRAY2BGR),(480,300))])
    out = np.vstack([row1,row2])
    cv2.putText(out,status,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("ELP2 | ELP1",out)

    key = cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('b'):
        print("Hintergrund aufnehmen...")
        f2s=[]; f1s=[]; fovs=[]
        for i in range(20):
            d,_ = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
            if d is not None: f2s.append(d)
            d,_ = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
            if d is not None: f1s.append(d)
            if OV_OK:
                d,_ = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
                if d is not None: fovs.append(d)
            if i%5==0: print(f"  {i+1}/20")
        bg2 = np.median(f2s,axis=0).astype(np.float32)
        bg1 = np.median(f1s,axis=0).astype(np.float32)
        st2 = np.std(f2s,axis=0)<3.0
        st1 = np.std(f1s,axis=0)<3.0
        if fovs:
            bgov = np.median(fovs,axis=0).astype(np.float32)
            stov = np.std(fovs,axis=0)<3.0
            ov_pct = f" OV:{stov.mean()*100:.0f}%"
        else:
            ov_pct = " OV:n/a"
        print(f"✓ ELP2:{st2.mean()*100:.0f}% ELP1:{st1.mean()*100:.0f}%{ov_pct} — Person hinstellen → SPACE")

    if key==ord(' ') and st2 is not None:
        # 5 Sekunden Countdown
        import time as _time
        for _i in range(5, 0, -1):
            d2,fl2 = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
            d1,fl1 = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
            if d2 is None or d1 is None: continue
            dv2 = (np.clip(d2,0,128)/128*255).astype(np.uint8)
            row1 = np.hstack([cv2.resize(fl2,(480,300)), cv2.resize(cv2.applyColorMap(dv2,cv2.COLORMAP_JET),(480,300)), np.zeros((300,480,3),np.uint8)])
            out = np.vstack([row1, np.zeros((300,1440,3),np.uint8)])
            cv2.putText(out, f"Hinstellen! Scan in {_i} Sekunden...", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.imshow("ELP2 | ELP1", out)
            cv2.waitKey(1)
            _time.sleep(1)
        # Frische Frames für den Scan holen
        d2,fl2 = get_disp_split(cap2, ml2_1,ml2_2,mr2_1,mr2_2, lm_elp,rm_elp,wls_elp)
        d1,fl1 = get_disp_split(cap1, ml1_1,ml1_2,mr1_1,mr1_2, lm_elp,rm_elp,wls_elp)
        if OV_OK:
            dov,flov = get_disp_dual(capovL,capovR, mlov1,mlov2,mrov1,mrov2, lm_ov,rm_ov,wls_ov)
        if st2 is not None:
            m2 = get_mask(d2, bg2, st2, 800, 600)
            m1 = get_mask(d1, bg1, st1, 800, 600)
            if OV_OK and dov is not None and stov is not None:
                mov = get_mask(dov, bgov, stov, 640, 400)
        p2,c2   = disp_to_pts(d2,  fl2,  m2,  fx2,  cx2,  cy2,  bl2)
        p1,c1   = disp_to_pts(d1,  fl1,  m1,  fx1,  cx1,  cy1,  bl1)
        parts=[]; cparts=[]
        if p2 is not None:
            zmask2 = (p2[:,2] > 0.5) & (p2[:,2] < 4.0)
            xmask2 = (p2[:,0] > -1.5) & (p2[:,0] < 1.5)
            p2 = p2[zmask2 & xmask2]; c2 = c2[zmask2 & xmask2]
            parts.append(p2); cparts.append(c2)
            print(f"ELP2: {len(p2)}")
        if p1 is not None:
            p1_t = (R_rel_elp1 @ p1.T).T + T_rel_elp1
            # Z-Clipping: nur Punkte 0.5–4m vor ELP2 Ursprung
            zmask1 = (p1_t[:,2] > 0.5) & (p1_t[:,2] < 4.0)
            # X-Clipping: ±1.5m um Bildmitte
            xmask1 = (p1_t[:,0] > -1.5) & (p1_t[:,0] < 1.5)
            mask1 = zmask1 & xmask1
            p1_t = p1_t[mask1]; c1 = c1[mask1]
            parts.append(p1_t); cparts.append(c1)
            print(f"ELP1: {len(p1)} → {len(p1_t)} nach Clipping")
        if OV_OK and mov is not None and stov is not None:
            pov,cov_col = disp_to_pts(dov, flov, mov, fxov, cxov, cyov, blov)
            if pov is not None:
                print(f"OV9281 Z-Bereich: {pov[:,2].min()*100:.0f}–{pov[:,2].max()*100:.0f}cm")
                print(f"OV9281 X-Bereich: {pov[:,0].min()*100:.0f}–{pov[:,0].max()*100:.0f}cm")
                # Z-Clipping: nur Punkte 0.3–3m vor OV9281
                zmask = (pov[:,2] > 0.3) & (pov[:,2] < 3.0)
                pov = pov[zmask]; cov_col = cov_col[zmask]
                pov_t = (R_rel_ov @ pov.T).T + T_rel_ov
                parts.append(pov_t); cparts.append(cov_col)
                print(f"OV9281: {len(pov)} → transformiert")
        if parts:
            all_pts = np.vstack(parts); all_cols = np.vstack(cparts)
            print(f"Gesamt: {len(all_pts)}")
            print(f"Breite:{(all_pts[:,0].max()-all_pts[:,0].min())*100:.0f}cm Höhe:{(all_pts[:,1].max()-all_pts[:,1].min())*100:.0f}cm")
            # Pro-Kamera Diagnose
            if p2 is not None and len(p2)>0: print(f"  ELP2  X:{p2[:,0].min()*100:.0f}–{p2[:,0].max()*100:.0f}cm  Z:{p2[:,2].min()*100:.0f}–{p2[:,2].max()*100:.0f}cm")
            if p1 is not None and len(p1_t)>0: print(f"  ELP1  X:{p1_t[:,0].min()*100:.0f}–{p1_t[:,0].max()*100:.0f}cm  Z:{p1_t[:,2].min()*100:.0f}–{p1_t[:,2].max()*100:.0f}cm")
            break

cap2.release(); cap1.release(); capovL.release(); capovR.release()
cv2.destroyAllWindows()

if all_pts is not None:
    pts=all_pts.copy(); cols=all_cols.copy()

    # 3D ROI — Objekt in der Mitte isolieren
    # Schritt 1: Grober Z-Cluster — dichteste Tiefenebene finden
    z_hist, z_edges = np.histogram(pts[:,2], bins=50)
    z_peak_idx = np.argmax(z_hist)
    z_center = (z_edges[z_peak_idx] + z_edges[z_peak_idx+1]) / 2
    Z_MARGIN = 0.35  # ±35cm Tiefe

    # Schritt 2: Innerhalb dieser Tiefe den X/Y Schwerpunkt finden
    zmask = np.abs(pts[:,2] - z_center) < Z_MARGIN
    if zmask.sum() > 50:
        x_center = np.median(pts[zmask, 0])
        y_center = np.median(pts[zmask, 1])
        X_MARGIN = 0.60  # ±60cm links/rechts
        Y_MARGIN = 0.80  # ±80cm oben/unten

        roi = zmask &               (np.abs(pts[:,0] - x_center) < X_MARGIN) &               (np.abs(pts[:,1] - y_center) < Y_MARGIN)
        pts = pts[roi]; cols = cols[roi]
        print(f"ROI: Zentrum=({x_center*100:.0f},{y_center*100:.0f},{z_center*100:.0f})cm → {len(pts)} Punkte")

    # Statistisches Outlier-Removal (ohne sklearn, nur numpy)
    print(f"Outlier-Removal: {len(pts)} Punkte...")
    RADIUS = 0.05; N_NEIGHBORS = 10
    # Subsample für Speed, dann auf alle anwenden
    step_r = max(1, len(pts)//5000)
    ref = pts[::step_r]
    keep = np.zeros(len(pts), dtype=bool)
    for i in range(0, len(pts), 2000):
        chunk = pts[i:i+2000]
        diffs = chunk[:,None,:] - ref[None,:,:]
        dists = np.sqrt((diffs**2).sum(axis=2))
        counts = (dists < RADIUS).sum(axis=1)
        keep[i:i+2000] = counts >= N_NEIGHBORS
    pts = pts[keep]; cols = cols[keep]
    print(f"  Nach Filter: {len(pts)} Punkte ({keep.sum()*100//len(keep)}% behalten)")

    pts[:,0]-=pts[:,0].mean(); pts[:,1]-=pts[:,1].mean(); pts[:,2]-=pts[:,2].mean()
    pts[:,1]=-pts[:,1]
    step=max(1,len(pts)//10000); pts=pts[::step]; cols=cols[::step]

    class Viewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ELP1+ELP2+OV9281 — Maus=Drehen | Scroll=Zoom")
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
