#!/usr/bin/env python3
"""ANTHRO3D — Patientenakte"""
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from database import Database, Patient
import yaml, datetime
from pathlib import Path

BASE = Path("~/anthro3d").expanduser()
COLORS = {"bg":"#f0f4f0","white":"#ffffff","g1":"#0F6E56","g2":"#1D9E75",
          "border":"#d0d8d0","dim":"#888780","text":"#1a1a1a","light":"#e8f4f0"}

def inp(ph=""):
    w=QLineEdit(); w.setPlaceholderText(ph)
    w.setStyleSheet(f"QLineEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;font-size:12px;}}QLineEdit:focus{{border-color:{COLORS['g2']};}}")
    return w

def txt(ph="",h=80):
    w=QTextEdit(); w.setPlaceholderText(ph); w.setFixedHeight(h)
    w.setStyleSheet(f"QTextEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;font-size:12px;}}QTextEdit:focus{{border-color:{COLORS['g2']};}}")
    return w

def combo(items):
    w=QComboBox(); w.addItems(items)
    w.setStyleSheet(f"QComboBox{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:6px 8px;font-size:12px;}}")
    return w

def section(title):
    w=QLabel(title)
    w.setStyleSheet(f"font-size:13px;font-weight:700;color:{COLORS['g1']};background:{COLORS['light']};border-radius:6px;padding:8px 12px;margin-top:8px;")
    return w

class PatientenAkte(QDialog):
    def __init__(self,parent=None,patient=None):
        super().__init__(parent)
        self.db=Database(); self.patient=patient
        self.setWindowTitle("ANTHRO3D — Patientenakte")
        self.setMinimumSize(900,700)
        self.setStyleSheet(f"background:{COLORS['bg']};color:{COLORS['text']};")
        self._build()
        if patient: self._load_data(patient)

    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        hdr=QWidget(); hdr.setStyleSheet(f"background:{COLORS['g1']};"); hdr.setFixedHeight(60)
        hl=QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0)
        t=QLabel("Patientenakte"); t.setStyleSheet("color:white;font-size:18px;font-weight:700;")
        hl.addWidget(t); hl.addStretch()
        self.header_info=QLabel(""); self.header_info.setStyleSheet("color:white;font-size:12px;")
        hl.addWidget(self.header_info); lay.addWidget(hdr)
        main=QHBoxLayout(); main.setContentsMargins(0,0,0,0); main.setSpacing(0)
        self.tab_list=QListWidget(); self.tab_list.setFixedWidth(180)
        self.tab_list.setStyleSheet(f"QListWidget{{background:{COLORS['white']};border:none;border-right:1px solid {COLORS['border']};padding:8px 0;}}QListWidget::item{{padding:12px 16px;font-size:12px;color:{COLORS['dim']};border-left:3px solid transparent;}}QListWidget::item:selected{{color:{COLORS['g1']};font-weight:600;background:{COLORS['light']};border-left:3px solid {COLORS['g1']};}}QListWidget::item:hover{{background:{COLORS['bg']};}}")
        for t2 in ["Stammdaten","Anamnese","Befunde","Diagnosen","Therapie","Medikamente","Verlauf","Aufklaerung","Entlassung"]:
            self.tab_list.addItem(t2)
        self.tab_list.setCurrentRow(0); main.addWidget(self.tab_list)
        self.stack=QStackedWidget()
        for tab in [self._tab_stammdaten(),self._tab_anamnese(),self._tab_befunde(),self._tab_diagnosen(),self._tab_therapie(),self._tab_medikamente(),self._tab_verlauf(),self._tab_aufklaerung(),self._tab_entlassung()]:
            self.stack.addWidget(tab)
        self.tab_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        main.addWidget(self.stack,1)
        mw=QWidget(); mw.setLayout(main); lay.addWidget(mw,1)
        footer=QWidget(); footer.setStyleSheet(f"background:{COLORS['white']};border-top:1px solid {COLORS['border']};"); footer.setFixedHeight(60)
        fl=QHBoxLayout(footer); fl.setContentsMargins(20,0,20,0)
        self.lbl_saved=QLabel(""); self.lbl_saved.setStyleSheet(f"color:{COLORS['g2']};font-size:12px;"); fl.addWidget(self.lbl_saved); fl.addStretch()
        bc=QPushButton("Schließen"); bc.setStyleSheet(f"QPushButton{{background:transparent;color:{COLORS['dim']};border:1px solid {COLORS['border']};border-radius:8px;padding:8px 20px;}}QPushButton:hover{{border-color:{COLORS['g2']};}}"); bc.clicked.connect(self.reject); fl.addWidget(bc)
        fl.addSpacing(8)
        bs=QPushButton("Speichern"); bs.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:8px;padding:8px 24px;font-size:13px;font-weight:600;}}QPushButton:hover{{background:{COLORS['g2']};}}"); bs.clicked.connect(self._save); fl.addWidget(bs)
        lay.addWidget(footer)

    def _scroll(self,w):
        s=QScrollArea(); s.setWidget(w); s.setWidgetResizable(True); s.setStyleSheet("border:none;background:transparent;"); return s
    def _fw(self):
        w=QWidget(); w.setStyleSheet(f"background:{COLORS['bg']};"); return w

    def _tab_stammdaten(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Persönliche Daten"))
        g=QGridLayout(); g.setSpacing(10)
        self.s_vorname=inp("Vorname"); g.addWidget(QLabel("Vorname *"),0,0); g.addWidget(self.s_vorname,0,1)
        self.s_nachname=inp("Nachname"); g.addWidget(QLabel("Nachname *"),1,0); g.addWidget(self.s_nachname,1,1)
        self.s_geb=inp("TT.MM.JJJJ"); g.addWidget(QLabel("Geburtsdatum"),2,0); g.addWidget(self.s_geb,2,1)
        self.s_geschl=combo(["—","Männlich","Weiblich","Divers"]); g.addWidget(QLabel("Geschlecht"),3,0); g.addWidget(self.s_geschl,3,1)
        l.addLayout(g)
        l.addWidget(section("Kontakt"))
        g2=QGridLayout(); g2.setSpacing(10)
        self.s_strasse=inp("Straße + Nr."); g2.addWidget(QLabel("Adresse"),0,0); g2.addWidget(self.s_strasse,0,1)
        self.s_plz=inp("PLZ"); self.s_ort=inp("Ort")
        pr=QHBoxLayout(); pr.addWidget(self.s_plz); pr.addWidget(self.s_ort)
        pw=QWidget(); pw.setLayout(pr); g2.addWidget(QLabel("PLZ / Ort"),1,0); g2.addWidget(pw,1,1)
        self.s_tel=inp("+43"); g2.addWidget(QLabel("Telefon"),2,0); g2.addWidget(self.s_tel,2,1)
        self.s_email=inp("email@"); g2.addWidget(QLabel("E-Mail"),3,0); g2.addWidget(self.s_email,3,1)
        l.addLayout(g2)
        l.addWidget(section("Versicherung & Notfall"))
        g3=QGridLayout(); g3.setSpacing(10)
        self.s_kasse=inp("Krankenkasse"); g3.addWidget(QLabel("Krankenkasse"),0,0); g3.addWidget(self.s_kasse,0,1)
        self.s_vnr=inp("Versicherungsnr."); g3.addWidget(QLabel("Versicherungs-Nr."),1,0); g3.addWidget(self.s_vnr,1,1)
        self.s_karten=inp("Kartennummer"); g3.addWidget(QLabel("Kartennummer"),2,0); g3.addWidget(self.s_karten,2,1)
        self.s_notfall=inp("Name + Tel"); g3.addWidget(QLabel("Notfallkontakt"),3,0); g3.addWidget(self.s_notfall,3,1)
        l.addLayout(g3); l.addStretch(); return self._scroll(w)

    def _tab_anamnese(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Aktuelle Beschwerden")); self.a_beschwerden=txt("Beschwerden, Beginn, Verlauf...",100); l.addWidget(self.a_beschwerden)
        l.addWidget(section("Vorerkrankungen")); self.a_vorerkrank=txt("Bekannte Vorerkrankungen...",80); l.addWidget(self.a_vorerkrank)
        l.addWidget(QLabel("Operationen")); self.a_operationen=txt("Frühere Operationen...",80); l.addWidget(self.a_operationen)
        l.addWidget(section("Medikamente & Allergien")); self.a_medikamente=txt("Aktuelle Medikamente...",80); l.addWidget(self.a_medikamente)
        l.addWidget(QLabel("Allergien")); self.a_allergien=txt("Allergien, Unverträglichkeiten...",60); l.addWidget(self.a_allergien)
        l.addWidget(section("Familienanamnese")); self.a_familie=txt("Erbliche Erkrankungen...",80); l.addWidget(self.a_familie)
        l.addWidget(section("Sozialanamnese"))
        g=QGridLayout(); g.setSpacing(10)
        self.a_beruf=inp("Beruf"); g.addWidget(QLabel("Beruf"),0,0); g.addWidget(self.a_beruf,0,1)
        self.a_sport=inp("Sport"); g.addWidget(QLabel("Sport"),1,0); g.addWidget(self.a_sport,1,1)
        self.a_nikotin=combo(["—","Nichtraucher","Raucher","Ex-Raucher"]); g.addWidget(QLabel("Nikotin"),2,0); g.addWidget(self.a_nikotin,2,1)
        self.a_alkohol=combo(["—","Kein","Gelegentlich","Regelmäßig"]); g.addWidget(QLabel("Alkohol"),3,0); g.addWidget(self.a_alkohol,3,1)
        l.addLayout(g); l.addStretch(); return self._scroll(w)

    def _tab_befunde(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Vitalparameter"))
        g=QGridLayout(); g.setSpacing(10)
        self.b_groesse=inp("cm"); g.addWidget(QLabel("Körpergröße"),0,0); g.addWidget(self.b_groesse,0,1)
        self.b_gewicht=inp("kg"); g.addWidget(QLabel("Gewicht"),1,0); g.addWidget(self.b_gewicht,1,1)
        self.b_rr=inp("mmHg"); g.addWidget(QLabel("Blutdruck"),2,0); g.addWidget(self.b_rr,2,1)
        self.b_puls=inp("/min"); g.addWidget(QLabel("Puls"),3,0); g.addWidget(self.b_puls,3,1)
        self.b_temp=inp("C"); g.addWidget(QLabel("Temperatur"),4,0); g.addWidget(self.b_temp,4,1)
        self.b_spo2=inp("%"); g.addWidget(QLabel("SpO2"),5,0); g.addWidget(self.b_spo2,5,1)
        l.addLayout(g)
        l.addWidget(section("Klinische Untersuchung")); self.b_klinisch=txt("Befundbeschreibung...",120); l.addWidget(self.b_klinisch)
        l.addWidget(section("Laborwerte")); self.b_labor=txt("Laborwerte, Datum...",80); l.addWidget(self.b_labor)
        l.addWidget(section("Bildgebung / 3D Scan")); self.b_bildgebung=txt("Befund...",100); l.addWidget(self.b_bildgebung)
        l.addWidget(section("ANTHRO3D Messwerte"))
        g2=QGridLayout(); g2.setSpacing(8)
        felder=[("Schulterbreite","cm"),("Huefte","cm"),("Koerpergroesse","cm"),("Schulterachse","Grad"),("Beckenachse","Grad"),("Brustumfang","cm"),("Taillenumfang","cm"),("Hueftumfang","cm")]
        self.b_meas={}
        for i,(name,unit) in enumerate(felder):
            row,col=divmod(i,2); self.b_meas[name]=inp(unit)
            g2.addWidget(QLabel(name),row,col*2); g2.addWidget(self.b_meas[name],row,col*2+1)
        l.addLayout(g2); l.addStretch(); return self._scroll(w)

    def _tab_diagnosen(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Hauptdiagnose")); self.d_haupt=inp("ICD-10 + Beschreibung"); l.addWidget(self.d_haupt)
        self.d_haupt_txt=txt("Detailbeschreibung...",80); l.addWidget(self.d_haupt_txt)
        l.addWidget(section("Nebendiagnosen")); self.d_neben=txt("ICD-10 Codes...",100); l.addWidget(self.d_neben)
        l.addWidget(section("Verdachtsdiagnosen")); self.d_verdacht=txt("Verdachtsdiagnosen...",80); l.addWidget(self.d_verdacht)
        l.addStretch(); return self._scroll(w)

    def _tab_therapie(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Aerztliche Anordnungen")); self.t_anordnung=txt("Angeordnete Massnahmen...",80); l.addWidget(self.t_anordnung)
        l.addWidget(section("Physiotherapie")); self.t_physio=txt("Physiotherapeutische Massnahmen...",80); l.addWidget(self.t_physio)
        l.addWidget(section("Komplikationen")); self.t_komplik=txt("Komplikationen...",80); l.addWidget(self.t_komplik)
        # SMART Ziele
        l.addWidget(section("SMART Ziele"))
        g_smart=QGridLayout(); g_smart.setSpacing(8); g_smart.setColumnStretch(1,1)
        for row,(lbl,attr,ph) in enumerate([
            ("Spezifisch","t_spez","Was genau soll erreicht werden?"),
            ("Messbar","t_mess","Wie wird der Fortschritt gemessen?"),
            ("Attraktiv","t_attr","Warum ist dieses Ziel wichtig?"),
            ("Realistisch","t_real","Ist das Ziel erreichbar?"),
        ]):
            setattr(self,attr,txt(ph,60))
            g_smart.addWidget(QLabel(lbl),row,0)
            g_smart.addWidget(getattr(self,attr),row,1)
        l.addLayout(g_smart)
        lbl_term=QLabel("Terminiert")
        self.t_term=QLineEdit(); self.t_term.setPlaceholderText("Datum (z.B. 30.06.2026)")
        self.t_term.setStyleSheet(f"QLineEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;}}")
        g_term=QGridLayout(); g_term.setColumnStretch(1,1)
        g_term.addWidget(lbl_term,0,0); g_term.addWidget(self.t_term,0,1)
        l.addLayout(g_term)
        # Bio-Psycho-Sozial
        l.addWidget(section("Bio-Psycho-Sozial"))
        g_bps=QGridLayout(); g_bps.setSpacing(8); g_bps.setColumnStretch(1,1)
        for row,(lbl,attr,ph) in enumerate([
            ("Bio","t_bio","Biologische Faktoren..."),
            ("Psycho","t_psy","Psychologische Faktoren..."),
            ("Sozial","t_soz","Soziale Faktoren..."),
        ]):
            setattr(self,attr,txt(ph,70))
            g_bps.addWidget(QLabel(lbl),row,0)
            g_bps.addWidget(getattr(self,attr),row,1)
        l.addLayout(g_bps)
        # ICF Ebenen
        l.addWidget(section("ICF Ebenen"))
        g_icf=QGridLayout(); g_icf.setSpacing(8); g_icf.setColumnStretch(1,1)
        for row,(lbl,attr,ph) in enumerate([
            ("Aktivitaet","t_aktiv","Aktivitaet und Teilhabe..."),
            ("Partizipation","t_part","Partizipation im Alltag..."),
        ]):
            setattr(self,attr,txt(ph,70))
            g_icf.addWidget(QLabel(lbl),row,0)
            g_icf.addWidget(getattr(self,attr),row,1)
        l.addLayout(g_icf)
        l.addStretch(); return self._scroll(w)

    def _tab_medikamente(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Medikamentenplan"))
        self.t_med_table=QTableWidget(10,4); self.t_med_table.setHorizontalHeaderLabels(["Medikament","Dosierung","Frequenz","Dauer"])
        self.t_med_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.t_med_table.setStyleSheet(f"QTableWidget{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;}}QHeaderView::section{{background:{COLORS['light']};color:{COLORS['g1']};font-weight:600;padding:6px;border:none;}}")
        self.t_med_table.setMinimumHeight(300); l.addWidget(self.t_med_table)
        btn_row=QHBoxLayout()
        btn_add_row=QPushButton("+ Zeile hinzufügen")
        btn_add_row.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:6px;padding:8px 16px;}}QPushButton:hover{{background:{COLORS['g2']};}}")
        btn_add_row.clicked.connect(lambda: self.t_med_table.setRowCount(self.t_med_table.rowCount()+1))
        btn_row.addWidget(btn_add_row); btn_row.addStretch(); l.addLayout(btn_row)
        def med_double_click(item):
            row=item.row()
            menu=QMenu(self.t_med_table)
            menu.setStyleSheet(f"QMenu{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:4px;}}QMenu::item{{padding:8px 20px;border-radius:4px;}}QMenu::item:selected{{background:{COLORS['light']};color:{COLORS['g1']};}}")
            act_del=menu.addAction("🗑  Zeile löschen")
            act_clear=menu.addAction("✖  Zeile leeren")
            action=menu.exec(self.t_med_table.viewport().mapToGlobal(self.t_med_table.visualItemRect(item).center()))
            if action==act_del: self.t_med_table.removeRow(row)
            elif action==act_clear:
                for col in range(self.t_med_table.columnCount()):
                    it=self.t_med_table.item(row,col)
                    if it: it.setText("")
        self.t_med_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        def med_select_row(item):
            self.t_med_table.selectRow(item.row())
        self.t_med_table.itemDoubleClicked.connect(med_select_row)
        self.t_med_table.customContextMenuRequested.connect(lambda pos: med_double_click(self.t_med_table.itemAt(pos)) if self.t_med_table.itemAt(pos) else None)
        l.addWidget(section("Hinweise / Unvertraeglichkeiten"))
        self.t_med_hinweis=txt("Hinweise zu Medikamenten, Unvertraeglichkeiten...",100); l.addWidget(self.t_med_hinweis)
        l.addStretch(); return self._scroll(w)

    def _tab_verlauf(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Verlaufseintrag"))
        dr=QHBoxLayout()
        self.v_datum=QLineEdit(); self.v_datum.setText(datetime.datetime.now().strftime("%d.%m.%Y"))
        self.v_uhr=QLineEdit(); self.v_uhr.setText(datetime.datetime.now().strftime("%H:%M"))
        for w2 in [self.v_datum,self.v_uhr]: w2.setStyleSheet(f"QLineEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;}}")
        dr.addWidget(QLabel("Datum:")); dr.addWidget(self.v_datum); dr.addWidget(QLabel("Uhrzeit:")); dr.addWidget(self.v_uhr); dr.addStretch()
        l.addLayout(dr)
        self.v_eintrag=txt("Tageseintrag...",120); l.addWidget(self.v_eintrag)
        btn_add=QPushButton("+ Eintrag hinzufuegen")
        btn_add.setStyleSheet(f"QPushButton{{background:{COLORS['g1']};color:white;border:none;border-radius:6px;padding:8px 16px;}}QPushButton:hover{{background:{COLORS['g2']};}}")
        btn_add.clicked.connect(self._add_verlauf); l.addWidget(btn_add)
        l.addWidget(section("Bisherige Eintraege"))
        self.v_history=QTextEdit(); self.v_history.setReadOnly(True); self.v_history.setFixedHeight(200)
        self.v_history.setStyleSheet(f"QTextEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;font-size:11px;}}")
        l.addWidget(self.v_history); l.addStretch(); return self._scroll(w)

    def _tab_aufklaerung(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Aufklaerungsgespraech"))
        g=QGridLayout(); g.setSpacing(10)
        self.au_datum=QLineEdit(); self.au_datum.setText(datetime.datetime.now().strftime("%d.%m.%Y"))
        self.au_datum.setStyleSheet(f"QLineEdit{{background:{COLORS['white']};border:1px solid {COLORS['border']};border-radius:6px;padding:8px;}}")
        self.au_arzt=inp("Name des Arztes")
        g.addWidget(QLabel("Datum"),0,0); g.addWidget(self.au_datum,0,1)
        g.addWidget(QLabel("Arzt"),1,0); g.addWidget(self.au_arzt,1,1)
        l.addLayout(g)
        self.au_inhalt=txt("Inhalt...",100); l.addWidget(self.au_inhalt)
        l.addWidget(section("Einwilligungen"))
        self.au_einwill=txt("Einwilligungen...",80); l.addWidget(self.au_einwill)
        l.addWidget(section("Datenschutz"))
        self.au_dsgvo=QCheckBox("Patient hat der Datenverarbeitung gemaess DSGVO zugestimmt")
        self.au_foto=QCheckBox("Patient hat der 3D Bildaufnahme zugestimmt")
        for cb in [self.au_dsgvo,self.au_foto]: cb.setStyleSheet(f"font-size:12px;color:{COLORS['text']};"); l.addWidget(cb)
        l.addStretch(); return self._scroll(w)

    def _tab_entlassung(self):
        w=self._fw(); l=QVBoxLayout(w); l.setContentsMargins(24,16,24,24); l.setSpacing(8)
        l.addWidget(section("Entlassung"))
        g=QGridLayout(); g.setSpacing(10)
        self.e_datum=inp("TT.MM.JJJJ")
        self.e_status=combo(["—","Beschwerdefrei","Gebessert","Unveraendert","Verschlechtert"])
        g.addWidget(QLabel("Entlassungsdatum"),0,0); g.addWidget(self.e_datum,0,1)
        g.addWidget(QLabel("Zustand"),1,0); g.addWidget(self.e_status,1,1)
        l.addLayout(g)
        l.addWidget(section("Entlassungsbericht")); self.e_bericht=txt("Zusammenfassung...",120); l.addWidget(self.e_bericht)
        l.addWidget(section("Weiterbehandlung")); self.e_empfehlung=txt("Empfehlungen...",80); l.addWidget(self.e_empfehlung)
        l.addWidget(section("Medikamentenplan")); self.e_medplan=txt("Medikamente...",80); l.addWidget(self.e_medplan)
        l.addStretch(); return self._scroll(w)

    def _load_data(self,patient):
        self.header_info.setText(patient.vollname + "  |  " + patient.geburtsdatum)
        akte=patient.folder/"akte.yaml"
        if not akte.exists():
            self.s_vorname.setText(patient.vorname); self.s_nachname.setText(patient.nachname)
            self.s_geb.setText(patient.geburtsdatum); self.s_karten.setText(patient.kartennummer)
            return
        try:
            with open(akte) as f: d=yaml.safe_load(f) or {}
            s=d.get("stammdaten",{})
            self.s_vorname.setText(s.get("vorname",patient.vorname)); self.s_nachname.setText(s.get("nachname",patient.nachname))
            self.s_geb.setText(s.get("geburtsdatum","")); self.s_karten.setText(s.get("kartennummer",""))
            self.s_strasse.setText(s.get("strasse","")); self.s_plz.setText(s.get("plz","")); self.s_ort.setText(s.get("ort",""))
            self.s_tel.setText(s.get("telefon","")); self.s_email.setText(s.get("email",""))
            self.s_kasse.setText(s.get("kasse","")); self.s_vnr.setText(s.get("vnr","")); self.s_notfall.setText(s.get("notfall",""))
            a=d.get("anamnese",{})
            self.a_beschwerden.setPlainText(a.get("beschwerden","")); self.a_vorerkrank.setPlainText(a.get("vorerkrankungen",""))
            self.a_operationen.setPlainText(a.get("operationen","")); self.a_medikamente.setPlainText(a.get("medikamente",""))
            self.a_allergien.setPlainText(a.get("allergien","")); self.a_familie.setPlainText(a.get("familie",""))
            self.a_beruf.setText(a.get("beruf","")); self.a_sport.setText(a.get("sport",""))
            idx=["—","Nichtraucher","Raucher","Ex-Raucher"]
            if a.get("nikotin","") in idx: self.a_nikotin.setCurrentIndex(idx.index(a.get("nikotin","")))
            idx2=["—","Kein","Gelegentlich","Regelmäßig"]
            if a.get("alkohol","") in idx2: self.a_alkohol.setCurrentIndex(idx2.index(a.get("alkohol","")))
            b=d.get("befunde",{})
            self.b_groesse.setText(b.get("groesse","")); self.b_gewicht.setText(b.get("gewicht",""))
            self.b_rr.setText(b.get("rr","")); self.b_puls.setText(b.get("puls",""))
            self.b_temp.setText(b.get("temp","")); self.b_spo2.setText(b.get("spo2",""))
            self.b_klinisch.setPlainText(b.get("klinisch","")); self.b_labor.setPlainText(b.get("labor",""))
            self.b_bildgebung.setPlainText(b.get("bildgebung",""))
            for k,v in b.get("meas",{}).items():
                if k in self.b_meas: self.b_meas[k].setText(str(v) if v else "")
            di=d.get("diagnosen",{})
            self.d_haupt.setText(di.get("hauptdiagnose","")); self.d_haupt_txt.setPlainText(di.get("hauptdiagnose_txt",""))
            self.d_neben.setPlainText(di.get("nebendiagnosen","")); self.d_verdacht.setPlainText(di.get("verdachtsdiagnosen",""))
            th=d.get("therapie",{})
            self.t_anordnung.setPlainText(th.get("anordnungen","")); self.t_physio.setPlainText(th.get("physiotherapie",""))
            self.t_komplik.setPlainText(th.get("komplikationen",""))
            sm=th.get("smart",{})
            self.t_spez.setPlainText(sm.get("spezifisch","")); self.t_mess.setPlainText(sm.get("messbar",""))
            self.t_attr.setPlainText(sm.get("attraktiv","")); self.t_real.setPlainText(sm.get("realistisch",""))
            self.t_term.setText(sm.get("terminiert",""))
            bps=th.get("bps",{})
            self.t_bio.setPlainText(bps.get("bio","")); self.t_psy.setPlainText(bps.get("psycho",""))
            self.t_soz.setPlainText(bps.get("sozial",""))
            icf=th.get("icf",{})
            self.t_aktiv.setPlainText(icf.get("aktivitaet","")); self.t_part.setPlainText(icf.get("partizipation",""))
            meds=th.get("medikamente",[])
            if meds:
                self.t_med_table.setRowCount(max(10,len(meds)))
                for r,m in enumerate(meds):
                    for c,key in enumerate(["medikament","dosierung","frequenz","dauer"]):
                        self.t_med_table.setItem(r,c,QTableWidgetItem(str(m.get(key,""))))
            self.t_med_hinweis.setPlainText(th.get("med_hinweis",""))
            au=d.get("aufklaerung",{})
            self.au_datum.setText(au.get("datum","")); self.au_arzt.setText(au.get("arzt",""))
            self.au_inhalt.setPlainText(au.get("inhalt","")); self.au_einwill.setPlainText(au.get("einwill",""))
            self.au_dsgvo.setChecked(bool(au.get("dsgvo",False))); self.au_foto.setChecked(bool(au.get("foto",False)))
            en=d.get("entlassung",{})
            self.e_datum.setText(en.get("datum",""))
            stati=["—","Beschwerdefrei","Gebessert","Unveraendert","Verschlechtert"]
            if en.get("status","") in stati: self.e_status.setCurrentIndex(stati.index(en.get("status","")))
            self.e_bericht.setPlainText(en.get("bericht","")); self.e_empfehlung.setPlainText(en.get("empfehlung",""))
            self.e_medplan.setPlainText(en.get("medplan",""))
            verlauf=d.get("verlauf",[])
            if verlauf:
                hist=""
                for e in verlauf:
                    hist += "-- " + str(e.get("datum","")) + " " + str(e.get("uhrzeit","")) + " --\n"
                    hist += str(e.get("eintrag","")) + "\n\n"
                self.v_history.setPlainText(hist)
        except Exception as ex: print("Akte laden:", ex)

    def _add_verlauf(self):
        e=self.v_eintrag.toPlainText().strip()
        if not e: return
        d=self.v_datum.text(); u=self.v_uhr.text()
        cur=self.v_history.toPlainText()
        self.v_history.setPlainText("-- " + d + " " + u + " --\n" + e + "\n\n" + cur)
        self.v_eintrag.clear()

    def _get_verlauf_list(self):
        result=[]
        text=self.v_history.toPlainText()
        for block in text.split("\n\n"):
            if "--" in block and len(block.split("--"))>=3:
                parts=block.split("--")
                dt_parts=parts[1].strip().split()
                datum=dt_parts[0] if dt_parts else ""
                uhrzeit=dt_parts[1] if len(dt_parts)>1 else ""
                eintrag=parts[2].strip()
                if eintrag: result.append({"datum":datum,"uhrzeit":uhrzeit,"eintrag":eintrag})
        return result

    def _save(self):
        if not self.patient:
            v=self.s_vorname.text().strip(); n=self.s_nachname.text().strip()
            if not v or not n: QMessageBox.warning(self,"Fehler","Vorname und Nachname eingeben!"); self.tab_list.setCurrentRow(0); return
            self.patient=self.db.create_patient(v,n,self.s_geb.text(),self.s_karten.text())
        meds=[]
        for row in range(self.t_med_table.rowCount()):
            med={["medikament","dosierung","frequenz","dauer"][col]:(self.t_med_table.item(row,col).text() if self.t_med_table.item(row,col) else "") for col in range(4)}
            if any(med.values()): meds.append(med)
        data={
            "stammdaten":{"vorname":self.s_vorname.text(),"nachname":self.s_nachname.text(),"geburtsdatum":self.s_geb.text(),"geschlecht":self.s_geschl.currentText(),"strasse":self.s_strasse.text(),"plz":self.s_plz.text(),"ort":self.s_ort.text(),"telefon":self.s_tel.text(),"email":self.s_email.text(),"kasse":self.s_kasse.text(),"vnr":self.s_vnr.text(),"kartennummer":self.s_karten.text(),"notfall":self.s_notfall.text()},
            "anamnese":{"beschwerden":self.a_beschwerden.toPlainText(),"vorerkrankungen":self.a_vorerkrank.toPlainText(),"operationen":self.a_operationen.toPlainText(),"medikamente":self.a_medikamente.toPlainText(),"allergien":self.a_allergien.toPlainText(),"familie":self.a_familie.toPlainText(),"beruf":self.a_beruf.text(),"sport":self.a_sport.text(),"nikotin":self.a_nikotin.currentText(),"alkohol":self.a_alkohol.currentText()},
            "befunde":{"groesse":self.b_groesse.text(),"gewicht":self.b_gewicht.text(),"rr":self.b_rr.text(),"puls":self.b_puls.text(),"temp":self.b_temp.text(),"spo2":self.b_spo2.text(),"klinisch":self.b_klinisch.toPlainText(),"labor":self.b_labor.toPlainText(),"bildgebung":self.b_bildgebung.toPlainText(),"meas":{k:v.text() for k,v in self.b_meas.items()}},
            "diagnosen":{"hauptdiagnose":self.d_haupt.text(),"hauptdiagnose_txt":self.d_haupt_txt.toPlainText(),"nebendiagnosen":self.d_neben.toPlainText(),"verdachtsdiagnosen":self.d_verdacht.toPlainText()},
            "therapie":{"anordnungen":self.t_anordnung.toPlainText(),"physiotherapie":self.t_physio.toPlainText(),"medikamente":meds,"komplikationen":self.t_komplik.toPlainText(),"smart":{"spezifisch":self.t_spez.toPlainText(),"messbar":self.t_mess.toPlainText(),"attraktiv":self.t_attr.toPlainText(),"realistisch":self.t_real.toPlainText(),"terminiert":self.t_term.text()},"bps":{"bio":self.t_bio.toPlainText(),"psycho":self.t_psy.toPlainText(),"sozial":self.t_soz.toPlainText()},"icf":{"aktivitaet":self.t_aktiv.toPlainText(),"partizipation":self.t_part.toPlainText()}},
            "aufklaerung":{"datum":self.au_datum.text(),"arzt":self.au_arzt.text(),"inhalt":self.au_inhalt.toPlainText(),"einwill":self.au_einwill.toPlainText(),"dsgvo":self.au_dsgvo.isChecked(),"foto":self.au_foto.isChecked()},
            "verlauf":self._get_verlauf_list(),
            "entlassung":{"datum":self.e_datum.text(),"status":self.e_status.currentText(),"bericht":self.e_bericht.toPlainText(),"empfehlung":self.e_empfehlung.toPlainText(),"medplan":self.e_medplan.toPlainText()},
            "gespeichert":datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
        }
        self.patient.folder.mkdir(parents=True,exist_ok=True)
        with open(self.patient.folder/"akte.yaml","w") as f: yaml.dump(data,f,allow_unicode=True)
        self.patient.vorname=self.s_vorname.text(); self.patient.nachname=self.s_nachname.text()
        self.patient.geburtsdatum=self.s_geb.text(); self.patient.kartennummer=self.s_karten.text()
        self.patient.save()
        self.lbl_saved.setText("Gespeichert " + datetime.datetime.now().strftime("%H:%M"))
        self.header_info.setText(self.patient.vollname + "  |  " + self.patient.geburtsdatum)
        print("Akte gespeichert:", self.patient.vollname)

if __name__=="__main__":
    import sys
    app=QApplication(sys.argv); dlg=PatientenAkte(); dlg.show(); sys.exit(app.exec())
