#!/usr/bin/env python3
"""
ANTHRO3D — ArUco Stereo Kalibrierung
Berechnet R, T zwischen Stereo L und R via ArUco Marker.
Verwendung: python3 aruco_stereo_calib.py
"""
import cv2, numpy as np, yaml, time
from pathlib import Path

BASE = Path("~/anthro3d").expanduser()
CAM_L, CAM_R = 1, 2
ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.minMarkerPerimeterRate = 0.03
DETECTOR     = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
MARKER_SIZE  = 0.19
REF_IDS      = [16, 2, 3, 10]

def load_K(idx):
    try:
        with open(BASE/"stereo_config.yaml") as f: cfg = yaml.safe_load(f)
        k = "K_left" if idx==CAM_L else "K_right"
        d = "dist_left" if idx==CAM_L else "dist_right"
        return np.array(cfg[k]), np.array(cfg[d])
    except:
        K = np.array([[900,0,640],[0,900,360],[0,0,1]],dtype=np.float64)
        return K, np.zeros((4,1))

def detect(frame, K, d):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    result = {}
    if ids is None: return result
    s = MARKER_SIZE/2
    obj = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],dtype=np.float32)
    for i,mid in enumerate(ids.flatten()):
        ok,rvec,tvec = cv2.solvePnP(obj,corners[i][0].astype(np.float32),K,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok: result[int(mid)]=(rvec.flatten(),tvec.flatten())
    return result

def relative_pose(rl,tl,rr,tr):
    Rl,_=cv2.Rodrigues(rl); Rr,_=cv2.Rodrigues(rr)
    R=Rr@Rl.T; T=tr-R@tl
    return R,T

def run(n=50):
    print("ArUco Stereo Kalibrierung — Marker ID16 vor beide Kameras halten")
    K_l,d_l=load_K(CAM_L); K_r,d_r=load_K(CAM_R)
    cap_l=cv2.VideoCapture(CAM_L); cap_r=cv2.VideoCapture(CAM_R)
    R_s,T_s=[],[]
    while True:
        ret_l,fl=cap_l.read(); ret_r,fr=cap_r.read()
        if not ret_l or not ret_r: continue
        pl=detect(fl,K_l,d_l); pr=detect(fr,K_r,d_r)
        common=set(pl)&set(pr)&set(REF_IDS)
        for mid in common:
            R,T=relative_pose(*pl[mid],*pr[mid])
            R_s.append(R); T_s.append(T)
        vis=np.hstack([cv2.resize(fl,(640,360)),cv2.resize(fr,(640,360))])
        pct=len(R_s)*100//n
        bar="["+"█"*(pct//5)+"░"*(20-pct//5)+f"] {len(R_s)}/{n}"
        col=(0,200,80) if common else (80,80,200)
        cv2.putText(vis,bar,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        cv2.putText(vis,f"Marker L:{list(pl.keys())} R:{list(pr.keys())}",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),1)
        cv2.imshow("ArUco Kalib — Q=Abbrechen",vis)
        if len(R_s)>=n: break
        if cv2.waitKey(1)&0xFF==ord("q"): cap_l.release();cap_r.release();cv2.destroyAllWindows();return
    cv2.destroyAllWindows(); cap_l.release(); cap_r.release()
    R_m=np.mean(R_s,axis=0); T_m=np.mean(T_s,axis=0)
    U,_,Vt=np.linalg.svd(R_m); R_f=U@Vt
    bl=np.linalg.norm(T_m)*1000
    rv,_=cv2.Rodrigues(R_f); ang=np.degrees(rv.flatten())
    print(f"Baseline: {bl:.1f}mm  Rotation: {ang[0]:.2f} {ang[1]:.2f} {ang[2]:.2f} Grad")
    cfg={}
    try:
        with open(BASE/"stereo_config.yaml") as f: cfg=yaml.safe_load(f) or {}
    except: pass
    cfg.update({"R":R_f.tolist(),"T":T_m.tolist(),"baseline_mm":float(bl),"calib_method":"aruco"})
    with open(BASE/"stereo_config.yaml","w") as f: yaml.dump(cfg,f)
    print(f"Gespeichert! Baseline={bl:.1f}mm")

if __name__=="__main__": run()
