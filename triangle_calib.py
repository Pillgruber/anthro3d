#!/usr/bin/env python3
import cv2,numpy as np,yaml,time
from pathlib import Path
BASE=Path("~/anthro3d").expanduser()
ARUCO_DICT=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS=cv2.aruco.DetectorParameters()
ARUCO_PARAMS.minMarkerPerimeterRate=0.03
DETECTOR=cv2.aruco.ArucoDetector(ARUCO_DICT,ARUCO_PARAMS)
MARKER_SIZE=0.19

def load_K(idx):
    try:
        with open(BASE/"stereo_config.yaml") as f: cfg=yaml.safe_load(f)
        K=np.array(cfg["K_left"]); d=np.array(cfg["dist_left"])
        return K,d
    except:
        K=np.array([[800,0,640],[0,800,360],[0,0,1]],dtype=np.float64)
        return K,np.zeros((4,1))

def detect(frame,K,d):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
    corners,ids,_=DETECTOR.detectMarkers(gray)
    result={}
    if ids is None: return result
    s=MARKER_SIZE/2
    obj=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],dtype=np.float32)
    for i,mid in enumerate(ids.flatten()):
        ok,rvec,tvec=cv2.solvePnP(obj,corners[i][0].astype(np.float32),K,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok: result[int(mid)]=(rvec.flatten(),tvec.flatten())
    return result

def rel_pose(rl,tl,rr,tr):
    Rl,_=cv2.Rodrigues(rl); Rr,_=cv2.Rodrigues(rr)
    R=Rr@Rl.T; T=tr-R@tl; return R,T

def run(n=60):
    print("Dreieck Kalibrierung — alle 3 ArUco Marker sichtbar halten")
    try:
        with open(BASE/"config.yaml") as f: cfg=yaml.safe_load(f)
        tracking=cfg["cameras"]["tracking"]
    except:
        print("config.yaml nicht gefunden — detect_cameras.py ausfuehren!"); return
    caps={}
    for cam in tracking:
        cap=cv2.VideoCapture(cam["device_index"])
        if cap.isOpened(): caps[cam["name"]]=cap; print(f"  {cam['name']} OK")
    if len(caps)<2: print("Zu wenige Kameras"); return
    Ks={name:load_K(0)[0] for name in caps}
    Ds={name:load_K(0)[1] for name in caps}
    names=list(caps.keys())
    pairs=[(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))]
    samples={p:{"R":[],"T":[]} for p in pairs}
    print(f"Paare: {pairs}  Q=Abbrechen")
    while True:
        frames={n:cap.read()[1] for n,cap in caps.items()}
        frames={n:f for n,f in frames.items() if f is not None}
        poses={n:detect(f,Ks[n],Ds[n]) for n,f in frames.items()}
        for a,b in pairs:
            if a not in poses or b not in poses: continue
            common=set(poses[a])&set(poses[b])-{2,3,10}
            for mid in common:
                R,T=rel_pose(*poses[a][mid],*poses[b][mid])
                samples[(a,b)]["R"].append(R); samples[(a,b)]["T"].append(T)
        vis_list=[]
        for name,frame in list(frames.items())[:3]:
            v=frame.copy() if len(frame.shape)==3 else cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            ids=list(poses.get(name,{}).keys())
            cv2.putText(v,f"{name}:{ids}",(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,80),1)
            vis_list.append(cv2.resize(v,(426,240)))
        while len(vis_list)<3: vis_list.append(np.zeros((240,426,3),dtype=np.uint8))
        grid=np.hstack(vis_list)
        info=np.zeros((50,grid.shape[1],3),dtype=np.uint8)
        status=" | ".join([f"{a[:3]}-{b[:3]}:{len(s['R'])}/{n}" for (a,b),s in samples.items()])
        cv2.putText(info,status,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
        cv2.imshow("Dreieck Kalib",np.vstack([grid,info]))
        done=all(len(s["R"])>=n for s in samples.values())
        if done: print("Fertig!"); break
        if cv2.waitKey(1)&0xFF==ord("q"): break
    cv2.destroyAllWindows()
    for cap in caps.values(): cap.release()
    result={}
    for (a,b),data in samples.items():
        if not data["R"]: continue
        Rm=np.mean(data["R"],axis=0); Tm=np.mean(data["T"],axis=0)
        U,_,Vt=np.linalg.svd(Rm); Rf=U@Vt
        bl=np.linalg.norm(Tm)*1000
        print(f"{a}-{b}: {bl:.1f}mm")
        result[f"{a}_{b}"]={"R":Rf.tolist(),"T":Tm.tolist(),"baseline_mm":float(bl)}
    with open(BASE/"triangle_config.yaml","w") as f: yaml.dump(result,f)
    print(f"triangle_config.yaml gespeichert")

if __name__=="__main__": run()
