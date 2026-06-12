#!/usr/bin/env python3
"""
ANTHRO3D — Triangulation
Berechnet 3D Körperpunkte aus zwei Einzelkameras.
Verwendung: python3 triangulation.py
Q=Beenden | S=Snapshot | M=Maße
"""
import cv2, numpy as np, yaml, time
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

BASE=Path("~/anthro3d").expanduser()
try:
    from circumference import CircumferenceCalculator
    CIRC=CircumferenceCalculator(); HAS_CIRC=True
except: HAS_CIRC=False
SAVE=Path("~/anthro3d/scans").expanduser(); SAVE.mkdir(exist_ok=True)
CAM_L,CAM_R=1,2
CONNECTIONS=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def load_calib():
    with open(BASE/"stereo_config.yaml") as f: cfg=yaml.safe_load(f)
    K_l=np.array(cfg["K_left"]); d_l=np.array(cfg["dist_left"])
    K_r=np.array(cfg["K_right"]); d_r=np.array(cfg["dist_right"])
    R=np.array(cfg["R"]); T=np.array(cfg["T"]).reshape(3,1)
    P1=K_l@np.hstack([np.eye(3),np.zeros((3,1))])
    P2=K_r@np.hstack([R,T])
    return K_l,d_l,K_r,d_r,P1,P2,float(cfg.get("baseline_mm",80))

def make_lm(model_path):
    opts=PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,min_tracking_confidence=0.5)
    return PoseLandmarker.create_from_options(opts)

def get_lms(lm,frame,ts):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    res=lm.detect_for_video(img,ts)
    if not res.pose_landmarks or not res.pose_landmarks[0]: return {}
    h,w=frame.shape[:2]
    return {i:(l.x*w,l.y*h,l.visibility) for i,l in enumerate(res.pose_landmarks[0])}

def triangulate(lms_l,lms_r,P1,P2):
    common={i for i in set(lms_l)&set(lms_r) if lms_l[i][2]>0.4 and lms_r[i][2]>0.4}
    if not common: return {}
    ids=sorted(common)
    pl=np.array([[lms_l[i][0],lms_l[i][1]] for i in ids],dtype=np.float32).T
    pr=np.array([[lms_r[i][0],lms_r[i][1]] for i in ids],dtype=np.float32).T
    p4=cv2.triangulatePoints(P1,P2,pl,pr); p4/=p4[3]
    pts=p4[:3].T
    return {ids[j]:(pts[j][0]*100,pts[j][1]*100,pts[j][2]*100) for j in range(len(ids)) if 30<pts[j][2]*100<500}

def measure(p):
    def d(a,b):
        if a not in p or b not in p: return None
        return round(float(np.linalg.norm(np.array(p[a])-np.array(p[b]))),1)
    h=None
    if 0 in p and (27 in p or 28 in p):
        a=p.get(27) or p.get(28)
        h=round(abs(p[0][1]-a[1])*1.06,1)
    return {"Groesse":h,"Schulter":d(11,12),"Huefte":d(23,24),
            "OA_L":d(11,13),"OA_R":d(12,14),"UA_L":d(13,15),"UA_R":d(14,16),
            "OS_L":d(23,25),"OS_R":d(24,26),"US_L":d(25,27),"US_R":d(26,28)}

def run():
    print("Triangulation — lade Kalibrierung...")
    K_l,d_l,K_r,d_r,P1,P2,bl=load_calib()
    print(f"Baseline: {bl:.1f}mm")
    lm_l=make_lm(BASE/"pose_landmarker.task")
    lm_r=make_lm(BASE/"pose_landmarker.task")
    cap_l=cv2.VideoCapture(CAM_L); cap_r=cv2.VideoCapture(CAM_R)
    ts_l=ts_r=0; smooth={}; ALPHA=0.35; show_m=True
    print("Q=Beenden | S=Snapshot | M=Maße")
    while True:
        rl,fl=cap_l.read(); rr,fr=cap_r.read()
        if not rl or not rr: continue
        ts_l+=33; ts_r+=33
        ll=get_lms(lm_l,fl,ts_l); lr=get_lms(lm_r,fr,ts_r)
        p3=triangulate(ll,lr,P1,P2)
        for idx,pt in p3.items():
            smooth[idx]=tuple(ALPHA*pt[k]+(1-ALPHA)*smooth[idx][k] if idx in smooth else pt[k] for k in range(3))
        m=measure(smooth)
        circ=CIRC.compute_from_pointclouds({"main":np.array(list(smooth.values()),dtype=np.float32)},smooth) if HAS_CIRC and len(smooth)>=10 else {}
        vl=fl.copy(); vr=fr.copy()
        for a,b in CONNECTIONS:
            for lms,vis in [(ll,vl),(lr,vr)]:
                if a in lms and b in lms and lms[a][2]>0.4 and lms[b][2]>0.4:
                    cv2.line(vis,(int(lms[a][0]),int(lms[a][1])),(int(lms[b][0]),int(lms[b][1])),(0,200,80),2)
        if show_m:
            y=30
            for k,v in m.items():
                if v: cv2.putText(vl,f"{k}:{v}cm",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,100),1); y+=20
        n=len(p3)
        col=(0,200,80) if n>=10 else (0,150,255) if n>=5 else (80,80,200)
        cv2.putText(vl,f"3D:{n}/33",(10,vl.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
        if 0 in smooth: cv2.putText(vl,f"Dist:{smooth[0][2]:.0f}cm",(vl.shape[1]-180,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),2)
        cv2.imshow("Triangulation",np.hstack([cv2.resize(vl,(640,360)),cv2.resize(vr,(640,360))]))
        k=cv2.waitKey(1)&0xFF
        if k==ord("q"): break
        elif k==ord("s") and smooth:
            ts=int(time.time())
            with open(SAVE/f"tri_{ts}.ply","w") as f:
                pts=list(smooth.values())
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                for x,y,z in pts: f.write(f"{x:.2f} {y:.2f} {z:.2f}\n")
            print(f"PLY: tri_{ts}.ply ({len(pts)} Punkte)")
        elif k==ord("m"): show_m=not show_m
        if show_m:
            yy=300
            for k2,v2 in circ.items():
                cv2.putText(vl,f"{k2}:{v2}cm",(10,yy),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,180,255),1); yy+=18
    cap_l.release(); cap_r.release(); lm_l.close(); lm_r.close(); cv2.destroyAllWindows()

if __name__=="__main__": run()
