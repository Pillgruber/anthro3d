#!/usr/bin/env python3
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from database import Database, Patient

COLORS={'bg':'#f0f4f0','white':'#ffffff','g1':'#0F6E56','g2':'#1D9E75','border':'#d0d8d0','dim':'#888780','text':'#1a1a1a'}

def inp_style(): return f"QLineEdit,QTextEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;font-size:13px;}}QLineEdit:focus,QTextEdit:focus{{border-color:{COLORS['g2']};}}"

class PatientDialog(QDialog):
    patient_selected=pyqtSignal(object)
    def __init__(self,parent=None,video3d=None,measurements=None):
        super().__init__(parent)
        self.db=Database(); self.video3d=video3d; self.measurements=measurements or {}; self._selected=None
        self.setWindowTitle("Aufnahme speichern — Patient")
        self.setMinimumSize(680,520)
        self.setStyleSheet(f"background:{COLORS['bg']};color:{COLORS['text']};")
        self._build()
    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(24,24,24,24); lay.setSpacing(16)
        t=QLabel("Aufnahme speichern"); t.setStyleSheet(f"font-size:20px;font-weight:700;color:{COLORS['g1']};"); lay.addWidget(t)
        n=len(self.video3d.frames) if self.video3d else 0
        lay.addWidget(QLabel(f"{n} Frames  |  {len(self.measurements)} Messwerte"))
        tabs=QTabWidget(); lay.addWidget(tabs,1)
        tabs.addTab(self._search_tab(),"Suchen")
        tabs.addTab(self._new_tab(),"Neuer Patient")
        br=QHBoxLayout(); br.addStretch()
        self.btn_save=QPushButton("Speichern"); self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:8px;padding:10px 32px;font-size:13px;font-weight:600;}}QPushButton:hover{{background:{COLORS['g2']};}}QPushButton:disabled{{background:#ccc;}}")
        self.btn_save.clicked.connect(self._save)
        bc=QPushButton("Abbrechen"); bc.clicked.connect(self.reject)
        bc.setStyleSheet(f"QPushButton{{background:transparent;color:{COLORS['dim']};border:1px solid {COLORS['border']};border-radius:8px;padding:10px 24px;}}")
        br.addWidget(bc); br.addWidget(self.btn_save); lay.addLayout(br)
    def _search_tab(self):
        w=QWidget(); l=QVBoxLayout(w); l.setContentsMargins(16,16,16,16)
        self.search_inp=QLineEdit(); self.search_inp.setPlaceholderText("Name oder Kartennummer..."); self.search_inp.setStyleSheet(inp_style())
        self.search_inp.textChanged.connect(self._search); l.addWidget(self.search_inp)
        self.result_list=QListWidget()
        self.result_list.setStyleSheet(f"QListWidget{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:8px;}}QListWidget::item{{padding:10px;border-radius:6px;}}QListWidget::item:selected{{background:{COLORS['g2']};color:white;}}")
        self.result_list.itemClicked.connect(self._select); l.addWidget(self.result_list,1)
        self._load_all(); return w
    def _new_tab(self):
        w=QWidget(); l=QFormLayout(w); l.setContentsMargins(16,16,16,16); l.setSpacing(12)
        s=inp_style()
        self.inp_v=QLineEdit(); self.inp_v.setStyleSheet(s)
        self.inp_n=QLineEdit(); self.inp_n.setStyleSheet(s)
        self.inp_g=QLineEdit(); self.inp_g.setPlaceholderText("TT.MM.JJJJ"); self.inp_g.setStyleSheet(s)
        self.inp_k=QLineEdit(); self.inp_k.setStyleSheet(s)
        self.inp_not=QTextEdit(); self.inp_not.setMaximumHeight(80); self.inp_not.setStyleSheet(s)
        l.addRow("Vorname *",self.inp_v); l.addRow("Nachname *",self.inp_n)
        l.addRow("Geburtsdatum",self.inp_g); l.addRow("Kartennummer",self.inp_k); l.addRow("Notizen",self.inp_not)
        btn=QPushButton("Patient anlegen + speichern")
        btn.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:8px;padding:10px;font-size:13px;font-weight:600;}}QPushButton:hover{{background:{COLORS['g2']};}}")
        btn.clicked.connect(self._create_save); l.addRow("",btn); return w
    def _load_all(self):
        self.result_list.clear()
        for p in self.db.all_patients():
            s=p.get_sessions(); last=f" | Letzte: {s[0]['datum'][:10]}" if s else ""
            item=QListWidgetItem(f"{p.vollname}  |  {p.geburtsdatum}  |  Nr:{p.kartennummer or '—'}{last}")
            item.setData(Qt.ItemDataRole.UserRole,p); self.result_list.addItem(item)
        if not self.result_list.count(): self.result_list.addItem("Keine Patienten — neuen anlegen")
    def _search(self,txt):
        self.result_list.clear()
        if not txt.strip(): self._load_all(); return
        pts=self.db.search(nachname=txt)+self.db.search(vorname=txt)+self.db.search(kartennummer=txt)
        seen=set(); unique=[]
        for p in pts:
            if p.pid not in seen: seen.add(p.pid); unique.append(p)
        for p in unique:
            item=QListWidgetItem(f"{p.vollname}  |  {p.geburtsdatum}  |  Nr:{p.kartennummer or '—'}")
            item.setData(Qt.ItemDataRole.UserRole,p); self.result_list.addItem(item)
        if not unique: self.result_list.addItem("Nicht gefunden")
    def _select(self,item):
        p=item.data(Qt.ItemDataRole.UserRole)
        if p and isinstance(p,Patient):
            self._selected=p; self.btn_save.setEnabled(True); self.btn_save.setText(f"Speichern für {p.vollname}")
    def _create_save(self):
        v=self.inp_v.text().strip(); n=self.inp_n.text().strip()
        if not v or not n: QMessageBox.warning(self,"Fehler","Vorname und Nachname Pflicht!"); return
        self._selected=self.db.create_patient(v,n,self.inp_g.text(),self.inp_k.text(),self.inp_not.toPlainText())
        self._save()
    def _save(self):
        if not self._selected: return
        sd=self.db.save_session(self._selected,self.video3d,self.measurements)
        self.patient_selected.emit(self._selected)
        QMessageBox.information(self,"Gespeichert",f"Gespeichert für:\n{self._selected.vollname}\n\n{sd}")
        self.accept()

class PatientListDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.db=Database(); self._current=None
        self.setWindowTitle("ANTHRO3D — Patienten")
        self.setMinimumSize(900,600)
        self.setStyleSheet(f"background:{COLORS['bg']};color:{COLORS['text']};")
        self._build()
    def _build(self):
        lay=QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        left=QWidget(); left.setFixedWidth(280)
        left.setStyleSheet(f"background:{COLORS['white']};border-right:1px solid {COLORS['border']};")
        ll=QVBoxLayout(left); ll.setContentsMargins(12,12,12,12); ll.setSpacing(8)
        t=QLabel("Patienten"); t.setStyleSheet(f"font-size:16px;font-weight:700;color:{COLORS['g1']};"); ll.addWidget(t)
        s=QLineEdit(); s.setPlaceholderText("Suchen..."); s.setStyleSheet(f"QLineEdit{{background:{COLORS['bg']};border:1px solid {COLORS['border']};border-radius:6px;padding:6px;}}")
        btn_a=QPushButton("Anlegen")
        btn_a.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:6px;padding:6px 12px;font-size:11px;font-weight:600;}}QPushButton:hover{{background:{COLORS['g2']};}}")
        btn_a.clicked.connect(self._open_full_akte); s.textChanged.connect(self._search); ll.addWidget(s)
        ll.addWidget(btn_a)
        self.plist=QListWidget()
        self.plist.setStyleSheet(f"QListWidget{{background:transparent;border:none;}}QListWidget::item{{padding:8px;border-radius:6px;margin:1px;}}QListWidget::item:selected{{background:{COLORS['g2']};color:white;}}")
        self.plist.itemClicked.connect(self._show); ll.addWidget(self.plist,1); lay.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); rl.setContentsMargins(24,24,24,24); rl.setSpacing(12)
        self.dtitle=QLabel("Patient auswählen"); self.dtitle.setStyleSheet(f"font-size:18px;font-weight:700;color:{COLORS['g1']};"); rl.addWidget(self.dtitle)
        self.dinfo=QLabel(""); self.dinfo.setStyleSheet(f"color:{COLORS['dim']};font-size:12px;"); rl.addWidget(self.dinfo)
        self.slay=QVBoxLayout(); sw=QWidget(); sw.setLayout(self.slay)
        sc=QScrollArea(); sc.setWidget(sw); sc.setWidgetResizable(True); sc.setStyleSheet("border:none;"); rl.addWidget(sc,1)
        br=QHBoxLayout(); br.addStretch()
        self.btn_rep=QPushButton("Report drucken"); self.btn_rep.setEnabled(False)
        self.btn_rep.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:8px;padding:10px 24px;}}QPushButton:disabled{{background:#ccc;}}")
        self.btn_rep.clicked.connect(self._report); br.addWidget(self.btn_rep)
        self.btn_akte=QPushButton("Akte oeffnen")
        self.btn_akte.setEnabled(False)
        self.btn_akte.setStyleSheet(f"QPushButton{{background:#185FA5;color:white;border:none;border-radius:8px;padding:10px 24px;}}QPushButton:disabled{{background:#ccc;}}")
        self.btn_akte.clicked.connect(self._open_akte); br.addWidget(self.btn_akte); rl.addLayout(br)
        lay.addWidget(right,1); self._load()
    def _new_patient_dialog(self):
        dlg=QDialog(self); dlg.setWindowTitle("Neuer Patient"); dlg.setMinimumWidth(420)
        dlg.setStyleSheet(f"background:{COLORS['bg']};")
        l=QFormLayout(dlg); l.setContentsMargins(20,20,20,20); l.setSpacing(12)
        s=f"QLineEdit{{background:white;border:1px solid {COLORS['border']};border-radius:6px;padding:8px;font-size:13px;}}"
        iv=QLineEdit(); iv.setStyleSheet(s)
        in_=QLineEdit(); in_.setStyleSheet(s)
        ig=QLineEdit(); ig.setPlaceholderText("TT.MM.JJJJ"); ig.setStyleSheet(s)
        ik=QLineEdit(); ik.setStyleSheet(s)
        l.addRow("Vorname *",iv); l.addRow("Nachname *",in_)
        l.addRow("Geburtsdatum",ig); l.addRow("Kartennummer",ik)
        btn=QPushButton("Patient anlegen")
        btn.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:8px;padding:10px;font-size:13px;font-weight:600;}}QPushButton:hover{{background:{COLORS['g2']};}}")
        def do_save():
            v=iv.text().strip(); n=in_.text().strip()
            if not v or not n: QMessageBox.warning(dlg,"Fehler","Vorname und Nachname sind Pflichtfelder!"); return
            self.db.create_patient(v,n,ig.text(),ik.text()); dlg.accept(); self._load()
        btn.clicked.connect(do_save); l.addRow("",btn); dlg.exec()

    def _load(self):
        self.plist.clear()
        for p in self.db.all_patients():
            s=p.get_sessions(); item=QListWidgetItem(f"{p.vollname}\n{p.geburtsdatum}  |  {len(s)} Messungen")
            item.setData(Qt.ItemDataRole.UserRole,p); self.plist.addItem(item)
    def _search(self,txt):
        self.plist.clear()
        pts=self.db.search(nachname=txt) if txt else self.db.all_patients()
        for p in pts:
            item=QListWidgetItem(f"{p.vollname}\n{p.geburtsdatum}")
            item.setData(Qt.ItemDataRole.UserRole,p); self.plist.addItem(item)
    def _show(self,item):
        p=item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(p,Patient): return
        self._current=p; self.dtitle.setText(p.vollname)
        self.dinfo.setText(f"Geb: {p.geburtsdatum}  |  Nr: {p.kartennummer or '—'}  |  Angelegt: {p.erstellt}")
        while self.slay.count():
            i=self.slay.takeAt(0)
            if i.widget(): i.widget().deleteLater()
        sessions=p.get_sessions()
        for s in sessions:
            card=QFrame(); card.setStyleSheet(f"QFrame{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:10px;padding:12px;}}")
            cl=QVBoxLayout(card)
            cl.addWidget(QLabel(f"📅  {s['datum'][:10]}  {s['datum'][11:16]}"))
            meas=s.get("meas",{})
            if meas:
                mr=QHBoxLayout()
                for k in ["Koerpergroesse","Schulterbreite","Huefte","Brustumfang","Taillenumfang"]:
                    v=meas.get(k)
                    if v:
                        col=QVBoxLayout()
                        vl=QLabel(str(v)); vl.setStyleSheet(f"font-size:15px;font-weight:700;color:{COLORS['g1']};")
                        kl=QLabel(k[:8]); kl.setStyleSheet(f"font-size:10px;color:{COLORS['dim']};")
                        col.addWidget(vl); col.addWidget(kl); mr.addLayout(col); mr.addSpacing(12)
                mr.addStretch(); cl.addLayout(mr)
            # Click-Handler für Session-Karte
            def _open_session(checked, session=s):
                from pathlib import Path
                import cv2, numpy as np
                folder = Path(session['folder']) if session.get('folder') else None
                if not folder or not folder.exists(): return
                video_folder = folder / "video3d"
                if not video_folder.exists():
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Info", "Kein Video in dieser Aufnahme")
                    return
                try:
                    from anthro3d_app import Video3DRecorder, Video3DViewer
                    rec = Video3DRecorder()
                    pngs = sorted(video_folder.glob("frame_*.png"))
                    rec.frames = [{"frame": cv2.imread(str(p)), "pts": {}, "depth": None} for p in pngs]
                    self._v3d_win = Video3DViewer(rec)
                    self._v3d_win.show()
                    self._v3d_win.raise_()
                    self._v3d_win.activateWindow()
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Fehler", str(e))
            card.mousePressEvent = _open_session
            card.setCursor(__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.CursorShape.PointingHandCursor)
            self.slay.addWidget(card)
        if not sessions: self.slay.addWidget(QLabel("Noch keine Messungen"))
        self.slay.addStretch(); self.btn_rep.setEnabled(True)
        if hasattr(self,'btn_akte'): self.btn_akte.setEnabled(True)
    def _open_full_akte(self):
        from patienten_akte import PatientenAkte
        p = self._current if hasattr(self,'_current') and self._current else None
        dlg=PatientenAkte(self,p); dlg.exec(); self._load()

    def _open_akte(self):
        if not self._current: return
        try:
            from patienten_akte import PatientenAkte
            dlg=PatientenAkte(self,self._current); dlg.exec()
        except Exception as e: QMessageBox.warning(self,"Fehler",str(e))

    def _report(self):
        if not self._current: return
        try:
            from report_generator import ReportGenerator
            p=ReportGenerator().generate(self._current)
            if p:
                import subprocess; subprocess.Popen(["open",p])
        except Exception as e: QMessageBox.warning(self,"Fehler",str(e))

if __name__=="__main__":
    import sys
    app=QApplication(sys.argv)
    d=PatientListDialog(); d.show()
    sys.exit(app.exec())
