#!/usr/bin/env python3
import yaml,shutil,datetime
from pathlib import Path
BASE=Path("~/anthro3d").expanduser()
DB_DIR=BASE/"patients"; DB_DIR.mkdir(parents=True,exist_ok=True)

class Patient:
    def __init__(self,pid,data):
        self.pid=pid; self.vorname=data.get("vorname",""); self.nachname=data.get("nachname","")
        self.geburtsdatum=data.get("geburtsdatum",""); self.kartennummer=data.get("kartennummer","")
        self.notizen=data.get("notizen",""); self.erstellt=data.get("erstellt","")
    @property
    def vollname(self): return f"{self.vorname} {self.nachname}".strip()
    @property
    def folder(self): return DB_DIR/self.pid
    def to_dict(self): return {"pid":self.pid,"vorname":self.vorname,"nachname":self.nachname,"geburtsdatum":self.geburtsdatum,"kartennummer":self.kartennummer,"notizen":self.notizen,"erstellt":self.erstellt}
    def save(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        with open(self.folder/"info.yaml","w") as f: yaml.dump(self.to_dict(),f,allow_unicode=True)
    def get_sessions(self):
        sessions=[]
        for d in sorted(self.folder.iterdir(),reverse=True):
            if not d.is_dir() or not d.name.startswith("2"): continue
            meas={}
            mf=d/"measurements.yaml"
            if mf.exists():
                with open(mf) as f: meas=yaml.safe_load(f) or {}
            sessions.append({"folder":d,"datum":d.name,"meas":meas,"has_video":(d/"video3d").exists(),"has_report":(d/"report.pdf").exists()})
        return sessions
    def last_session(self):
        s=self.get_sessions(); return s[0] if s else None

class Database:
    def __init__(self): self.db_dir=DB_DIR
    def create_patient(self,vorname,nachname,geburtsdatum="",kartennummer="",notizen=""):
        existing=self.search(vorname=vorname,nachname=nachname,geburtsdatum=geburtsdatum)
        if existing: return existing[0]
        ts=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        p=Patient(f"P{ts}",{"vorname":vorname.strip(),"nachname":nachname.strip(),"geburtsdatum":geburtsdatum.strip(),"kartennummer":kartennummer.strip(),"notizen":notizen.strip(),"erstellt":datetime.datetime.now().strftime("%d.%m.%Y %H:%M")})
        p.save(); print(f"Patient angelegt: {p.vollname}"); return p
    def get_patient(self,pid):
        f=self.db_dir/pid/"info.yaml"
        if not f.exists(): return None
        with open(f) as ff: d=yaml.safe_load(ff)
        return Patient(pid,d)
    def search(self,vorname="",nachname="",geburtsdatum="",kartennummer=""):
        results=[]
        for pd in self.db_dir.iterdir():
            if not pd.is_dir(): continue
            inf=pd/"info.yaml"
            if not inf.exists(): continue
            try:
                with open(inf) as f: d=yaml.safe_load(f)
                p=Patient(pd.name,d); ok=True
                if vorname and vorname.lower() not in p.vorname.lower(): ok=False
                if nachname and nachname.lower() not in p.nachname.lower(): ok=False
                if geburtsdatum and geburtsdatum not in p.geburtsdatum: ok=False
                if kartennummer and kartennummer not in p.kartennummer: ok=False
                if ok: results.append(p)
            except: continue
        return sorted(results,key=lambda p:p.nachname)
    def all_patients(self): return self.search()
    def save_session(self,patient,video3d=None,measurements=None):
        import cv2
        ts=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sd=patient.folder/ts; sd.mkdir(parents=True,exist_ok=True)
        if measurements:
            with open(sd/"measurements.yaml","w") as f: yaml.dump(measurements,f,allow_unicode=True)
        if video3d and video3d.frames:
            vd=sd/"video3d"; vd.mkdir(exist_ok=True)
            for i,fd in enumerate(video3d.frames):
                fr=fd.get("frame")
                if fr is not None: cv2.imwrite(str(vd/f"frame_{i:04d}.png"),fr)
            print(f"  {len(video3d.frames)} Frames gespeichert")
        print(f"Session: {sd}"); return sd
