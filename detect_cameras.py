#!/usr/bin/env python3
import cv2,numpy as np,yaml
from pathlib import Path
BASE=Path("~/anthro3d").expanduser()

def detect_all():
    print("ANTHRO3D — Kamera Erkennung")
    found=[]
    for idx in range(16):
        cap=cv2.VideoCapture(idx)
        if not cap.isOpened(): cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
        ret,frame=cap.read()
        if not ret or frame is None: cap.release(); continue
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_elp=(w>=2000)
        if is_elp: typ="ELP 3D Stereo"; wd=f"{w}x{h} (L+R = {w//2}x{h} je)"
        else:
            b,g,r=frame[:,:,0],frame[:,:,1],frame[:,:,2]
            typ="Farbe" if (np.std(b.astype(int)-g.astype(int))>5) else "Mono (OV9281)"
            wd=f"{w}x{h}"
        print(f"  Index {idx}: {typ} — {wd}")
        found.append({'index':idx,'type':'elp' if is_elp else 'ov9281','color':is_elp,'width':w,'height':h,'is_elp':is_elp})
        cap.release()
    print(f"\n{len(found)} Kameras gefunden")
    elp=[c for c in found if c['is_elp']]
    ov=[c for c in found if not c['is_elp']]
    print(f"ELP:{len(elp)}  OV9281:{len(ov)}")
    tracking=[]
    for i,c in enumerate(elp[:2]):
        tracking.append({'device_index':c['index'],'name':f'ELP{i+1}','id':[3,10][i],'enabled':True,'type':'elp'})
    for i,c in enumerate(ov[:4]):
        tracking.append({'device_index':c['index'],'name':f'OV{i+1}','id':[2,2,2,2][i],'enabled':True,'type':'ov9281'})
    cfg={'cameras':{'tracking':tracking},'calibration':{'ema_alpha':0.3,'marker_size_cm':19.0,'stability_frames':30},'aruco':{'dict':'DICT_ARUCO_ORIGINAL','marker_size_m':0.19,'ids':{'elp1':3,'elp2':10,'ov9281':2,'patient':16}}}
    with open(BASE/"config.yaml",'w') as f: yaml.dump(cfg,f)
    print(f"config.yaml gespeichert")
    print("\nNaechster Schritt: python3 ~/anthro3d/elp_capture.py")

if __name__=="__main__": detect_all()
