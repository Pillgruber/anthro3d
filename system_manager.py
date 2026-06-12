#!/usr/bin/env python3
"""
ANTHRO3D — System Manager
Verwaltet alle Kamera-Systeme modular und erweiterbar.
"""
import cv2,numpy as np,yaml,time
from pathlib import Path
BASE=Path("~/anthro3d").expanduser()

class CameraSystem:
    def __init__(self,name,cfg):
        self.name=name; self.cfg=cfg
        self.type=cfg.get("type","unknown")
        self.position=cfg.get("position","unknown")
        self.aruco_id=cfg.get("aruco_id")
        self.color=cfg.get("color",False)
        self.fps=cfg.get("fps",30)
        self.baseline=cfg.get("baseline_mm",80)
        self.priority=cfg.get("priority",99)
        self.enabled=cfg.get("enabled",False)
        self.cap_l=None; self.cap_r=None
        self.K_l=None; self.K_r=None
        self.R=None; self.T=None
        self.P1=None; self.P2=None
    def is_elp(self): return self.type=="elp"
    def is_ov_pair(self): return self.type=="ov9281_pair"
    def open(self,idx_l,idx_r=None):
        self.cap_l=cv2.VideoCapture(idx_l)
        if not self.cap_l.isOpened(): return False
        w,h=self.cfg.get("resolution",[1280,720])
        if self.is_elp(): w=w*2
        for cap in [self.cap_l]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
            cap.set(cv2.CAP_PROP_FPS,self.fps)
        if idx_r is not None:
            self.cap_r=cv2.VideoCapture(idx_r)
            if self.cap_r.isOpened():
                self.cap_r.set(cv2.CAP_PROP_FRAME_WIDTH,w)
                self.cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
                self.cap_r.set(cv2.CAP_PROP_FPS,self.fps)
        return True
    def read(self):
        if self.cap_l is None: return None,None
        ret,frame=self.cap_l.read()
        if not ret: return None,None
        if self.is_elp():
            w=frame.shape[1]//2; return frame[:,:w],frame[:,w:]
        elif self.cap_r is not None:
            ret_r,fr=self.cap_r.read(); return frame,(fr if ret_r else None)
        return frame,None
    def release(self):
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()
    def load_calibration(self):
        try:
            p=BASE/"stereo_config.yaml"
            if p.exists():
                with open(p) as f: cfg=yaml.safe_load(f)
                self.K_l=np.array(cfg.get("K_left",[[800,0,640],[0,800,360],[0,0,1]]))
                self.K_r=np.array(cfg.get("K_right",[[800,0,640],[0,800,360],[0,0,1]]))
                self.R=np.array(cfg.get("R",np.eye(3).tolist()))
                self.T=np.array(cfg.get("T",[self.baseline/1000,0,0]))
                return True
        except: pass
        fx=800.0
        self.K_l=np.array([[fx,0,640],[0,fx,360],[0,0,1]],dtype=np.float64)
        self.K_r=self.K_l.copy()
        self.R=np.eye(3); self.T=np.array([self.baseline/1000,0,0])
        return False
    def build_projection_matrices(self):
        if self.K_l is None: self.load_calibration()
        T=self.T.reshape(3,1) if self.T is not None else np.zeros((3,1))
        self.P1=self.K_l@np.hstack([np.eye(3),np.zeros((3,1))])
        self.P2=self.K_r@np.hstack([self.R,T]) if self.R is not None else self.P1.copy()
    def __repr__(self): return f"CameraSystem({self.name},{self.type},{self.position},en={self.enabled})"

class SystemManager:
    def __init__(self,config_path=None):
        self.cfg_path=config_path or (BASE/"config_standard.yaml")
        self.systems={}; self._load_config()
    def _load_config(self):
        try:
            with open(self.cfg_path) as f: cfg=yaml.safe_load(f)
            for name,sc in cfg.get("systems",{}).items():
                self.systems[name]=CameraSystem(name,sc)
            print(f"Config: {len(self.systems)} Systeme")
        except Exception as e:
            print(f"Config Fehler: {e}"); self._defaults()
    def _defaults(self):
        defaults={"ELP1":{"type":"elp","position":"front","aruco_id":2,"enabled":True,"color":True,"fps":27,"resolution":[2560,720],"baseline_mm":60,"priority":1},"ELP2":{"type":"elp","position":"back","aruco_id":3,"enabled":True,"color":True,"fps":27,"resolution":[2560,720],"baseline_mm":60,"priority":2},"OV1":{"type":"ov9281_pair","position":"left","aruco_id":10,"enabled":True,"color":False,"fps":60,"resolution":[1280,720],"baseline_mm":80,"priority":3}}
        for n,c in defaults.items(): self.systems[n]=CameraSystem(n,c)
    def detect(self):
        print("Kamera-Erkennung...")
        found=[]
        for idx in range(16):
            cap=cv2.VideoCapture(idx)
            if not cap.isOpened(): cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200); ret,frame=cap.read(); cap.release()
            if not ret or frame is None: continue
            w=frame.shape[1]; is_elp=(w>=2000)
            print(f"  Index {idx}: {'ELP' if is_elp else 'OV9281'} {w}px")
            found.append({"index":idx,"is_elp":is_elp})
        elp=[c for c in found if c["is_elp"]]; ov=[c for c in found if not c["is_elp"]]
        elp_sys=sorted([s for s in self.systems.values() if s.is_elp()],key=lambda s:s.priority)
        for i,sys in enumerate(elp_sys):
            if i<len(elp): sys.cfg["device_index"]=elp[i]["index"]; sys.enabled=True; print(f"  {sys.name}→{elp[i]['index']}")
        ov_sys=sorted([s for s in self.systems.values() if s.is_ov_pair()],key=lambda s:s.priority)
        oi=0
        for sys in ov_sys:
            if oi+1<len(ov): sys.cfg["device_index_l"]=ov[oi]["index"]; sys.cfg["device_index_r"]=ov[oi+1]["index"]; sys.enabled=True; print(f"  {sys.name}→L:{ov[oi]['index']} R:{ov[oi+1]['index']}"); oi+=2
        active=[s for s in self.systems.values() if s.enabled]
        print(f"{len(active)} aktiv: {[s.name for s in active]}"); return active
    def open_all(self):
        opened={}
        for name,sys in self.systems.items():
            if not sys.enabled: continue
            if sys.is_elp():
                idx=sys.cfg.get("device_index")
                if idx is not None and sys.open(idx): sys.load_calibration(); sys.build_projection_matrices(); opened[name]=sys; print(f"  {name} OK")
            elif sys.is_ov_pair():
                il=sys.cfg.get("device_index_l"); ir=sys.cfg.get("device_index_r")
                if il is not None and sys.open(il,ir): sys.load_calibration(); sys.build_projection_matrices(); opened[name]=sys; print(f"  {name} OK")
        return opened
    def get_active(self): return sorted([s for s in self.systems.values() if s.enabled],key=lambda s:s.priority)
    def add_system(self,name,cfg): self.systems[name]=CameraSystem(name,cfg); print(f"{name} hinzugefuegt")
    def enable(self,name):
        if name in self.systems: self.systems[name].enabled=True; print(f"{name} aktiviert")
    def disable(self,name):
        if name in self.systems: self.systems[name].enabled=False; print(f"{name} deaktiviert")
    def status(self):
        print(f"{'Name':10} {'Typ':15} {'Position':10} {'Aktiv':6} {'Prio':4}")
        print("-"*50)
        for n,s in sorted(self.systems.items(),key=lambda x:x[1].priority):
            print(f"{n:10} {s.type:15} {s.position:10} {'✓' if s.enabled else '✗':6} {s.priority:4}")
    def save_config(self):
        cfg={"version":2,"systems":{n:s.cfg for n,s in self.systems.items()}}
        with open(self.cfg_path,"w") as f: yaml.dump(cfg,f)
        print(f"Gespeichert: {self.cfg_path}")

if __name__=="__main__":
    sm=SystemManager(); sm.status()
    sm.detect()
    systems=sm.open_all()
    if not systems: print("Keine Kameras"); exit()
    print("Q=Beenden")
    while True:
        frames=[]
        for name,sys in systems.items():
            fl,fr=sys.read()
            if fl is None: continue
            vis=cv2.resize(fl,(426,240)); cv2.putText(vis,f"{name} L",(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,80),2); frames.append(vis)
            if fr is not None:
                vr=cv2.resize(fr,(426,240)); cv2.putText(vr,f"{name} R",(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,80),2); frames.append(vr)
        if not frames: time.sleep(0.05); continue
        while len(frames)%2!=0: frames.append(np.zeros((240,426,3),dtype=np.uint8))
        rows=[np.hstack(frames[i:i+2]) for i in range(0,len(frames),2)]
        cv2.imshow("System Manager",np.vstack(rows))
        if cv2.waitKey(1)&0xFF==ord("q"): break
    for s in systems.values(): s.release()
    cv2.destroyAllWindows()
