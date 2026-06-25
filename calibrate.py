import cv2, numpy as np, yaml, sys
from pathlib import Path

print("=== ANTHRO3D Kalibrierung ===")
print("Raum muss LEER sein — keine Person, keine beweglichen Objekte")
print()

# Kalibrierungen laden
with open(Path("~/anthro3d/stereo_config.yaml").expanduser()) as f:
    cfg2 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser()) as f:
    cfg1 = yaml.safe_load(f)
with open(Path("~/anthro3d/stereo_config_ov9281.yaml").expanduser()) as f:
    cfgov = yaml.safe_load(f)

R_rel_elp1 = np.load(Path("~/anthro3d/R_rel_elp1_to_elp2.npy").expanduser())
T_rel_elp1 = np.load(Path("~/anthro3d/T_rel_elp1_to_elp2.npy").expanduser()) * 100

def make_maps(cfg, size=(1600,1200), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5, return_R1=False):
    K_l=np.array(cfg['camera_matrix_l']); d_l=np.array(cfg['dist_l'])
    K_r=np.array(cfg['camera_matrix_r']); d_r=np.array(cfg['dist_r'])
    R=np.array(cfg['R']); T=np.array(cfg['T'])
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,size,R,T,flags=flags,alpha=alpha)
    ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,size,cv2.CV_32F)
    mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,size,cv2.CV_32F)
    out = (ml1,ml2,mr1,mr2,P1[0,0],P1[0,2],P1[1,2],abs(T.flatten()[0]))
    if return_R1:
        return out + (R1,)
    return out

ml2_1,ml2_2,mr2_1,mr2_2,fx2,cx2,cy2,bl2 = make_maps(cfg2)
ml1_1,ml1_2,mr1_1,mr1_2,fx1,cx1,cy1,bl1 = make_maps(cfg1)
mlov1,mlov2,mrov1,mrov2,fxov,cxov,cyov,blov,R1_ov = make_maps(cfgov,(1280,800), flags=0, alpha=-1, return_R1=True)
print(f"OV9281 Rectify Kalibrierung: fx={fxov:.1f} cx={cxov:.1f} cy={cyov:.1f} baseline={blov*100:.1f}cm flags=0")

lm=cv2.StereoSGBM_create(minDisparity=4,numDisparities=128,blockSize=9,
    P1=8*3*81,P2=32*3*81,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=150,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm=cv2.ximgproc.createRightMatcher(lm)
wls=cv2.ximgproc.createDisparityWLSFilter(lm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

lm_ov=cv2.StereoSGBM_create(minDisparity=2,numDisparities=64,blockSize=7,
    P1=8*3*49,P2=32*3*49,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
rm_ov=cv2.ximgproc.createRightMatcher(lm_ov)
wls_ov=cv2.ximgproc.createDisparityWLSFilter(lm_ov)
wls_ov.setLambda(8000); wls_ov.setSigmaColor(1.5)

# Kameras
cap2=cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
cap1=cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
capovL=cv2.VideoCapture(1)
capovL.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovL.set(cv2.CAP_PROP_FRAME_HEIGHT,800)
capovR=cv2.VideoCapture(2)
capovR.set(cv2.CAP_PROP_FRAME_WIDTH,1280); capovR.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

def get_disp(cap, ml1, ml2, mr1, mr2, lm_, rm_, wls_, split=True):
    ret,frame=cap.read()
    if not ret: return None
    if split:
        w=frame.shape[1]//2
        fl=cv2.remap(frame[:,:w],ml1,ml2,cv2.INTER_LINEAR)
        fr=cv2.remap(frame[:,w:],mr1,mr2,cv2.INTER_LINEAR)
    else:
        fl=cv2.remap(frame,ml1,ml2,cv2.INTER_LINEAR)
        fr=cv2.remap(frame,mr1,mr2,cv2.INTER_LINEAR)
    fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
    fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
    gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    dl=lm_.compute(gl,gr); dr=rm_.compute(gr,gl)
    d=wls_.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
    d[d<4]=0
    return d

# OV9281 Transformation
aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
params=cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate=0.05  # Mindestgröße ~80px bei 1600px Bild
params.maxMarkerPerimeterRate=0.5   # Maxgröße
detector=cv2.aruco.ArucoDetector(aruco_dict,params)
MARKER_SIZE = 0.1865
VALID_IDS={2,3,4,20,30,40}          # Nur bekannte Stativ-Marker
MAX_DIST=5.0                # Max 5m Entfernung
MIN_AREA=80*80              # Mindest-Pixelfläche

def validate_marker(corner, tvec):
    """Prüft ob Marker valide ist (Größe, Distanz, Aspect-Ratio)."""
    # Distanz prüfen
    dist = np.linalg.norm(tvec)
    if dist > MAX_DIST: return False
    # Pixelfläche prüfen
    pts = corner[0]
    area = cv2.contourArea(pts)
    if area < MIN_AREA: return False
    # Aspect-Ratio prüfen (muss quadratisch sein ±20%)
    w = np.linalg.norm(pts[0]-pts[1])
    h = np.linalg.norm(pts[1]-pts[2])
    if h == 0: return False
    ratio = w/h
    if ratio < 0.7 or ratio > 1.3: return False
    return True
obj_pts=np.array([[-MARKER_SIZE/2,MARKER_SIZE/2,0],[MARKER_SIZE/2,MARKER_SIZE/2,0],
                   [MARKER_SIZE/2,-MARKER_SIZE/2,0],[-MARKER_SIZE/2,-MARKER_SIZE/2,0]],dtype=np.float32)
K2=np.array(cfg2['camera_matrix_l']); d2=np.array(cfg2['dist_l'])
Kov=np.array(cfgov['camera_matrix_l']); dov=np.array(cfgov['dist_l'])

print("Schritt 1: OV9281→ELP2 Transformation berechnen...")
R_ov_list=[]; T_ov_list=[]
for _ in range(60):
    ret2,f2=cap2.read(); retov,fov=capovL.read()
    if not ret2 or not retov: continue
    fl2=f2[:,:f2.shape[1]//2]
    g2=cv2.cvtColor(fl2,cv2.COLOR_BGR2GRAY)
    gov=cv2.cvtColor(fov,cv2.COLOR_BGR2GRAY)
    c2,ids2,_=detector.detectMarkers(g2)
    cov,idsov,_=detector.detectMarkers(gov)
    if ids2 is None or idsov is None: continue
    common=set(ids2.flatten())&set(idsov.flatten())
    for mid in common:
        i2=np.where(ids2.flatten()==mid)[0][0]
        iov=np.where(idsov.flatten()==mid)[0][0]
        ok2,rv2,tv2=cv2.solvePnP(obj_pts,c2[i2],K2,d2)
        okv,rvv,tvv=cv2.solvePnP(obj_pts,cov[iov],Kov,dov)
        if ok2 and okv:
            R2m,_=cv2.Rodrigues(rv2); Rvm,_=cv2.Rodrigues(rvv)
            R_ov_list.append(R2m@Rvm.T)
            T_ov_list.append(tv2.flatten()-(R2m@Rvm.T)@tvv.flatten())

OV_OK=False
if R_ov_list:
    T_arr=np.array(T_ov_list)
    dists=np.linalg.norm(T_arr,axis=1)
    mask=dists<np.median(dists)*1.3
    rvecs=np.array([cv2.Rodrigues(R)[0].flatten() for R in np.array(R_ov_list)[mask]])
    R_rel_ov,_=cv2.Rodrigues(np.mean(rvecs,axis=0))
    T_rel_ov=np.mean(T_arr[mask],axis=0)
    np.save(Path("~/anthro3d/R_rel_ov9281_to_elp2.npy").expanduser(),R_rel_ov)
    np.save(Path("~/anthro3d/T_rel_ov9281_to_elp2.npy").expanduser(),T_rel_ov)
    print(f"  ✓ OV9281→ELP2: {T_rel_ov*100} cm  det={np.linalg.det(R_rel_ov):.4f}")
    OV_OK=True
else:
    print("  ✗ OV9281 Marker nicht gefunden")

# Scan-Korridor aus Kamerapositionen berechnen
print("Scan-Korridor wird berechnet...")

# Kamerapositionen im ELP2-Koordinatensystem
cam_positions = np.array([
    [0.0, 0.0, 0.0],          # ELP2 = Ursprung
        T_rel_elp1.flatten(),      # ELP1
])
if OV_OK:
    cam_positions = np.vstack([cam_positions, T_rel_ov])

# Schwerpunkt der Kameras = Zentrum des Korridors
center = cam_positions.mean(axis=0)

# Konvexe Hülle der Kamerapositionen in XZ-Ebene (Draufsicht)
# Korridor = Dreieck zwischen den 3 Kameras, leicht nach innen versetzt
from scipy.spatial import ConvexHull
try:
    hull_pts = cam_positions[:, [0,2]]  # nur X und Z
    hull = ConvexHull(hull_pts)
    hull_vertices = hull_pts[hull.vertices]
    print(f"  Korridor-Dreieck: {len(hull.vertices)} Ecken")
    # 20cm nach innen versetzen (Puffer damit Kameras selbst nicht mitgescannt werden)
    INSET = 0.20
    hull_inset = []
    for v in hull_vertices:
        direction = center[[0,2]] - v
        direction = direction / np.linalg.norm(direction)
        hull_inset.append(v + direction * INSET)
    hull_inset = np.array(hull_inset)
    CORRIDOR_OK = True
except Exception as e:
    print(f"  ⚠ Korridor-Berechnung fehlgeschlagen: {e}")
    hull_inset = None
    CORRIDOR_OK = False

# Y-Bereich: Boden bis Decke (realistisch -0.5m bis +2.5m)
Y_MIN = -0.3  # 30cm unter Kamera
Y_MAX = 2.5   # 2.5m über Boden

def point_in_corridor(pts):
    """Prüft ob Punkte im Scan-Korridor liegen."""
    if not CORRIDOR_OK or hull_inset is None:
        return np.ones(len(pts), dtype=bool)
    # Y-Filter
    y_ok = (pts[:,1] > -Y_MAX) & (pts[:,1] < Y_MIN)
    # XZ-Polygon-Filter
    from matplotlib.path import Path as MplPath
    polygon = MplPath(hull_inset)
    xz_ok = polygon.contains_points(pts[:,[0,2]])
    return y_ok & xz_ok

np.save(Path("~/anthro3d/scan_corridor_hull.npy").expanduser(),
        hull_inset if CORRIDOR_OK else np.zeros((0,2)))
np.save(Path("~/anthro3d/scan_corridor_y.npy").expanduser(),
        np.array([Y_MIN, Y_MAX]))
print(f"  ✓ Scan-Korridor gespeichert")

# Schritt 2: Hintergrund-Disparitätskarten aufnehmen
print("Schritt 2: Hintergrund aufnehmen (30 Frames)...")
f2s=[]; f1s=[]; fovs=[]
for i in range(30):
    d=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2,lm,rm,wls)
    if d is not None: f2s.append(d)
    d=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2,lm,rm,wls)
    if d is not None: f1s.append(d)
    if OV_OK:
        retL,fL=capovL.read(); retR,fR=capovR.read()
        if retL and retR:
            fl=cv2.remap(fL,mlov1,mlov2,cv2.INTER_LINEAR)
            fr=cv2.remap(fR,mrov1,mrov2,cv2.INTER_LINEAR)
            fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
            fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
            gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
            gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            dl=lm_ov.compute(gl,gr); dr=rm_ov.compute(gr,gl)
            d=wls_ov.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
            d[d<4]=0
            fovs.append(d)
    if i%10==0: print(f"  {i+1}/30")

bg2=np.median(f2s,axis=0).astype(np.float32)
bg1=np.median(f1s,axis=0).astype(np.float32)
st2=(np.std(f2s,axis=0)<3.0)
st1=(np.std(f1s,axis=0)<3.0)
print(f"  ELP2 stabil: {st2.mean()*100:.0f}%  ELP1 stabil: {st1.mean()*100:.0f}%")

bgov=stov=None
if fovs:
    bgov=np.median(fovs,axis=0).astype(np.float32)
    stov=(np.std(fovs,axis=0)<3.0)
    print(f"  OV9281 stabil: {stov.mean()*100:.0f}%")

# Schritt 3: Leeren Scan-Korridor als 3D-Punktwolke aufnehmen
print("Schritt 3: Stabilen Hintergrund aufnehmen (Personen werden herausgefiltert)...")

def disp_to_pts(d, fx, cx, cy, bl, R_rel=None, T_rel=None):
    rows,ci=np.where(d>4)
    if len(rows)==0: return None
    dv=d[rows,ci]
    Z=fx*bl/dv; X=(ci-cx)*Z/fx; Y=(rows-cy)*Z/fx
    pts=np.stack([X,Y,Z],axis=1)
    zm=(Z>0.3)&(Z<5.0)
    pts=pts[zm]
    if R_rel is not None and len(pts)>0:
        pts=(R_rel@pts.T).T+T_rel.flatten()
    return pts

# Viele Frames aufnehmen — Personen bewegen sich, Hintergrund bleibt stabil
N_FRAMES = 60
voxel = 0.05  # 5cm Voxelgröße
print(f"  {N_FRAMES} Frames aufnehmen...")

# Voxel-Zähler: wie oft wurde jeder Voxel gesehen?
voxel_count = {}
voxel_pts = {}

for fi in range(N_FRAMES):
    d2=get_disp(cap2,ml2_1,ml2_2,mr2_1,mr2_2,lm,rm,wls)
    d1=get_disp(cap1,ml1_1,ml1_2,mr1_1,mr1_2,lm,rm,wls)
    frame_pts = []
    if d2 is not None:
        p=disp_to_pts(d2,fx2,cx2,cy2,bl2)
        if p is not None: frame_pts.append(p)
    if d1 is not None:
        p=disp_to_pts(d1,fx1,cx1,cy1,bl1,R_rel_elp1,T_rel_elp1)
        if p is not None: frame_pts.append(p)
    if OV_OK and R_rel_ov is not None:
        retL,fL=capovL.read(); retR,fR=capovR.read()
        if retL and retR:
            fl=cv2.remap(fL,mlov1,mlov2,cv2.INTER_LINEAR)
            fr=cv2.remap(fR,mrov1,mrov2,cv2.INTER_LINEAR)
            fl=cv2.convertScaleAbs(fl,alpha=2.5,beta=30)
            fr=cv2.convertScaleAbs(fr,alpha=2.5,beta=30)
            gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
            gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            dl=lm_ov.compute(gl,gr); dr=rm_ov.compute(gr,gl)
            dov=wls_ov.filter(dl,fl,disparity_map_right=dr).astype(np.float32)/16.0
            dov[dov<4]=0
            p=disp_to_pts(dov,fxov,cxov,cyov,blov)
            if p is not None:
                # OV9281-Hintergrundpunkte an scan3d.py angleichen:
                # Rectified Stereo zurückdrehen, Achsen korrigieren, dann nach ELP2 transformieren.
                pov_raw = (R1_ov.T @ p.T).T
                flip_ov = np.array([-1.0, 1.0, -1.0], dtype=float).reshape(1, 3)
                p = (R_rel_ov.T @ (pov_raw * flip_ov).T).T + T_rel_ov.reshape(1, 3)
            if p is not None: frame_pts.append(p)
    if frame_pts:
        all_p = np.vstack(frame_pts)
        idx = (all_p/voxel).astype(np.int32)
        keys = idx[:,0]*1000000+idx[:,1]*1000+idx[:,2]
        for k,pt in zip(keys, all_p):
            if k not in voxel_count:
                voxel_count[k] = 0
                voxel_pts[k] = pt
            voxel_count[k] += 1
    # Fortschrittsbalken
    pct = int((fi+1)/N_FRAMES*50)
    bar = '█'*pct + '░'*(50-pct)
    print(f"  [{bar}] {fi+1}/{N_FRAMES}", end='\r')

# Nur Voxels die in mehr als 50% der Frames gesehen wurden = stabil = Hintergrund
# Personen bewegen sich → selten gesehen → werden nicht als Hintergrund markiert
min_count = N_FRAMES * 0.4  # 40% der Frames
stable_keys = [k for k,c in voxel_count.items() if c >= min_count]
if stable_keys:
    bg_pts = np.array([voxel_pts[k] for k in stable_keys])
    print(f"  Stabile Hintergrund-Voxels: {len(bg_pts)} (von {len(voxel_count)} gesamt)")
    print(f"  Personen/Bewegung entfernt: {len(voxel_count)-len(bg_pts)} Voxels")
else:
    bg_pts = np.zeros((0,3))
    print("  ⚠ Keine stabilen Voxels gefunden")

# Hintergrund-Scanbox begrenzen
# Entfernt unrealistische Ausreißer, bevor bg_pts gespeichert wird.
# Werte sind in Metern. ELP2 bleibt Weltursprung.
if bg_pts is not None and len(bg_pts) > 0:
    before_bg_box = len(bg_pts)

    bg_box = (
        (bg_pts[:,0] > -2.50) & (bg_pts[:,0] < 2.50) &
        (bg_pts[:,1] > -3.00) & (bg_pts[:,1] < 1.50) &
        (bg_pts[:,2] > 0.20) & (bg_pts[:,2] < 5.50)
    )

    bg_pts = bg_pts[bg_box]

    print(
        f"  Hintergrund-Scanbox: {before_bg_box} → {len(bg_pts)} Punkte | "
        f"X=-250..250cm Y=-300..150cm Z=20..550cm"
    )

    if len(bg_pts) > 0:
        print(
            f"  bg_pts Bereich nach Scanbox: "
            f"X={bg_pts[:,0].min()*100:.0f}..{bg_pts[:,0].max()*100:.0f}cm "
            f"Y={bg_pts[:,1].min()*100:.0f}..{bg_pts[:,1].max()*100:.0f}cm "
            f"Z={bg_pts[:,2].min()*100:.0f}..{bg_pts[:,2].max()*100:.0f}cm"
        )


# Alles speichern
out=Path("~/anthro3d/calibration_bg.npz").expanduser()
save_dict=dict(bg2=bg2,bg1=bg1,st2=st2,st1=st1)
if bgov is not None:
    save_dict.update(bgov=bgov,stov=stov)
    save_dict['OV_OK']=np.array([True])
else:
    save_dict['OV_OK']=np.array([False])
save_dict['bg_pts'] = bg_pts
save_dict['bg_voxel'] = np.array([voxel])
np.savez(out,**save_dict)
print(f"✓ Hintergrund gespeichert: {out}")

cap2.release(); cap1.release(); capovL.release(); capovR.release()
print()
print("=== Kalibrierung abgeschlossen ===")
print("Starte jetzt: python3 ~/anthro3d/scan3d.py")
