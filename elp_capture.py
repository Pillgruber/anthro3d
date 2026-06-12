#!/usr/bin/env python3
import cv2,numpy as np,time
from pathlib import Path
SAVE=Path("~/anthro3d/sessions").expanduser(); SAVE.mkdir(exist_ok=True)

def read_elp(cap):
    ret,frame=cap.read()
    if not ret: return None,None
    w=frame.shape[1]; return frame[:,:w//2],frame[:,w//2:]

def run():
    print("ELP Capture Test — suche Kameras...")
    elp_caps=[]
    for idx in range(10):
        cap=cv2.VideoCapture(idx)
        if not cap.isOpened(): cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200); cap.set(cv2.CAP_PROP_FPS,60)
        ret,frame=cap.read()
        if ret and frame is not None and frame.shape[1]>=2000:
            print(f"  ELP gefunden: Index {idx} ({frame.shape[1]}x{frame.shape[0]})")
            elp_caps.append((idx,cap))
        else: cap.release()
        if len(elp_caps)>=2: break
    if not elp_caps: print("Keine ELP gefunden! USB 3.0 pruefen"); return
    print(f"{len(elp_caps)} ELP(s) — Q=Beenden S=Snapshot")
    n=0
    while True:
        frames=[]
        for i,(idx,cap) in enumerate(elp_caps):
            fl,fr=read_elp(cap)
            if fl is None: continue
            fl_s=cv2.resize(fl,(640,400)); fr_s=cv2.resize(fr,(640,400))
            cv2.putText(fl_s,f"ELP{i+1} L",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,80),2)
            cv2.putText(fr_s,f"ELP{i+1} R",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,80),2)
            frames.extend([fl_s,fr_s])
        if not frames: time.sleep(0.05); continue
        if len(frames)==4: grid=np.vstack([np.hstack(frames[:2]),np.hstack(frames[2:])])
        else: grid=np.hstack(frames[:2]) if len(frames)>=2 else frames[0]
        cv2.imshow("ELP Capture",grid)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        elif k==ord('s'):
            ts=int(time.time())
            for i,(idx,cap) in enumerate(elp_caps):
                fl,fr=read_elp(cap)
                if fl is not None:
                    cv2.imwrite(str(SAVE/f"elp{i+1}_L_{ts}.png"),fl)
                    cv2.imwrite(str(SAVE/f"elp{i+1}_R_{ts}.png"),fr)
            print(f"Snapshot {n} gespeichert"); n+=1
    for _,cap in elp_caps: cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__": run()
