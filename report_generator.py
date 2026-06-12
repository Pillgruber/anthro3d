#!/usr/bin/env python3
import datetime
from pathlib import Path
BASE=Path("~/anthro3d").expanduser()
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor,white
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,HRFlowable,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    HAS_RL=True
except: HAS_RL=False; print("pip install reportlab --break-system-packages")

class ReportGenerator:
    G=HexColor("#0F6E56") if HAS_RL else None
    BG=HexColor("#f8f9fa") if HAS_RL else None
    LT=HexColor("#e8f4f0") if HAS_RL else None

    def _v(self,v):
        """Wert ausgefüllt?"""
        if v is None or v=="": return False
        if isinstance(v,str) and v.strip() in ("","—","-"): return False
        if isinstance(v,list) and len(v)==0: return False
        if isinstance(v,dict) and not any(self._v(x) for x in v.values()): return False
        return True

    def _hs(self): return ParagraphStyle("h",fontSize=12,fontName="Helvetica-Bold",textColor=self.G,spaceAfter=3)
    def _ns(self): return ParagraphStyle("n",fontSize=10,fontName="Helvetica",spaceAfter=2)
    def _ss(self): return ParagraphStyle("s",fontSize=9,fontName="Helvetica",textColor=HexColor("#555555"),spaceAfter=1)

    def _section(self,title,rows,story):
        """Sektion mit Tabelle — nur wenn Zeilen vorhanden"""
        if not rows: return
        block=[]
        block.append(Paragraph(title,self._hs()))
        t=Table(rows,colWidths=[55*mm,115*mm])
        t.setStyle(TableStyle([
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),10),
            ("PADDING",(0,0),(-1,-1),5),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[white,self.BG]),
            ("BOX",(0,0),(-1,-1),0.5,self.G),
            ("INNERGRID",(0,0),(-1,-1),0.25,HexColor("#cccccc")),
        ]))
        block.append(t)
        block.append(Spacer(1,6*mm))
        story.append(KeepTogether(block))

    def generate(self,patient,session=None):
        if not HAS_RL: print("reportlab fehlt!"); return None
        import yaml
        akte_path=patient.folder/"akte.yaml"
        akte={}
        if akte_path.exists():
            with open(akte_path) as f: akte=yaml.safe_load(f) or {}

        pdf=patient.folder/"report.pdf"
        doc=SimpleDocTemplate(str(pdf),pagesize=A4,
            leftMargin=20*mm,rightMargin=20*mm,topMargin=20*mm,bottomMargin=20*mm)
        story=[]

        # ── HEADER ──────────────────────────────────────────────
        story.append(Paragraph(
            f"<font color='#0F6E56'><b>ANTHRO3D</b></font>  —  Patientenbericht",
            ParagraphStyle("t",fontSize=18,fontName="Helvetica-Bold")))
        story.append(Paragraph(
            datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
            ParagraphStyle("d",fontSize=10,textColor=HexColor("#888888"))))
        story.append(Spacer(1,4*mm))
        story.append(HRFlowable(width="100%",thickness=2,color=self.G))
        story.append(Spacer(1,6*mm))

        # ── STAMMDATEN ───────────────────────────────────────────
        s=akte.get("stammdaten",{})
        LABELS_S=[
            ("vorname","Vorname"),("nachname","Nachname"),("geburtsdatum","Geburtsdatum"),
            ("geschlecht","Geschlecht"),("strasse","Adresse"),("plz","PLZ"),("ort","Ort"),
            ("telefon","Telefon"),("email","E-Mail"),("kasse","Krankenkasse"),
            ("kartennummer","Kartennummer"),("vnr","Versicherungsnummer"),("notfall","Notfallkontakt"),
        ]
        rows=[[lb,str(s[k])] for k,lb in LABELS_S if k in s and self._v(s.get(k))]
        self._section("Stammdaten",rows,story)

        # ── ANAMNESE ─────────────────────────────────────────────
        a=akte.get("anamnese",{})
        LABELS_A=[
            ("beschwerden","Beschwerden"),("vorerkrankungen","Vorerkrankungen"),
            ("operationen","Operationen"),("medikamente","Medikamente"),
            ("allergien","Allergien"),("beruf","Beruf"),("sport","Sport"),
            ("nikotin","Nikotin"),("alkohol","Alkohol"),("familie","Familie"),
        ]
        rows=[[lb,str(a[k])] for k,lb in LABELS_A if k in a and self._v(a.get(k))]
        self._section("Anamnese",rows,story)

        # ── BEFUNDE ──────────────────────────────────────────────
        b=akte.get("befunde",{})
        LABELS_B=[
            ("groesse","Größe (cm)"),("gewicht","Gewicht (kg)"),("rr","Blutdruck"),
            ("puls","Puls"),("spo2","SpO2"),("temp","Temperatur"),
            ("labor","Labor"),("bildgebung","Bildgebung"),("klinisch","Klinischer Befund"),
        ]
        rows=[[lb,str(b[k])] for k,lb in LABELS_B if k in b and self._v(b.get(k))]
        meas=b.get("meas",{})
        for k,v in meas.items():
            if self._v(v): rows.append([k,f"{v} cm"])
        self._section("Befunde",rows,story)

        # ── DIAGNOSEN ────────────────────────────────────────────
        d=akte.get("diagnosen",{})
        LABELS_D=[
            ("hauptdiagnose","Hauptdiagnose ICD"),("hauptdiagnose_txt","Hauptdiagnose"),
            ("nebendiagnosen","Nebendiagnosen"),("verdachtsdiagnosen","Verdachtsdiagnosen"),
        ]
        rows=[[lb,str(d[k])] for k,lb in LABELS_D if k in d and self._v(d.get(k))]
        self._section("Diagnosen",rows,story)

        # ── THERAPIE ─────────────────────────────────────────────
        th=akte.get("therapie",{})
        LABELS_TH=[
            ("anordnungen","Ärztliche Anordnungen"),("physiotherapie","Physiotherapie"),
            ("komplikationen","Komplikationen"),
        ]
        rows=[[lb,str(th[k])] for k,lb in LABELS_TH if k in th and self._v(th.get(k))]
        # Medikamente
        meds=th.get("medikamente",[])
        if meds:
            for m in meds:
                if isinstance(m,dict) and any(self._v(v) for v in m.values()):
                    rows.append(["Medikament",f"{m.get('medikament','')} | {m.get('dosierung','')} | {m.get('frequenz','')} | {m.get('dauer','')}"])
        # SMART
        sm=th.get("smart",{})
        LABELS_SM=[("spezifisch","SMART — Spezifisch"),("messbar","SMART — Messbar"),
                   ("attraktiv","SMART — Attraktiv"),("realistisch","SMART — Realistisch"),
                   ("terminiert","SMART — Terminiert")]
        for k,lb in LABELS_SM:
            if self._v(sm.get(k)): rows.append([lb,str(sm[k])])
        # BPS
        bps=th.get("bps",{})
        for k,lb in [("bio","Bio"),("psycho","Psycho"),("sozial","Sozial")]:
            if self._v(bps.get(k)): rows.append([f"Bio-Psycho-Sozial — {lb}",str(bps[k])])
        # ICF
        icf=th.get("icf",{})
        for k,lb in [("aktivitaet","Aktivität"),("partizipation","Partizipation")]:
            if self._v(icf.get(k)): rows.append([f"ICF — {lb}",str(icf[k])])
        if self._v(th.get("med_hinweis")):
            rows.append(["Medikamenten-Hinweise", str(th["med_hinweis"])])
        self._section("Therapie",rows,story)

        # ── VERLAUF ──────────────────────────────────────────────
        vl=akte.get("verlauf",[])
        if isinstance(vl,list) and vl:
            block=[]
            block.append(Paragraph("Verlauf",self._hs()))
            for e in vl:
                if isinstance(e,dict):
                    dt=e.get("datum",""); txt=e.get("eintrag","")
                    if self._v(txt):
                        block.append(Paragraph(f"<b>{dt}</b>",self._ss()))
                        block.append(Paragraph(str(txt),self._ns()))
                        block.append(Spacer(1,3*mm))
            block.append(Spacer(1,3*mm))
            story.append(KeepTogether(block))

        # ── AUFKLÄRUNG ───────────────────────────────────────────
        au=akte.get("aufklaerung",{})
        rows=[]
        if self._v(au.get("datum")): rows.append(["Datum",str(au["datum"])])
        if self._v(au.get("arzt")): rows.append(["Arzt",str(au["arzt"])])
        if self._v(au.get("inhalt")): rows.append(["Inhalt",str(au["inhalt"])])
        if self._v(au.get("einwill")): rows.append(["Einwilligungen",str(au["einwill"])])
        if au.get("dsgvo"): rows.append(["DSGVO","Zugestimmt "])
        if au.get("foto"): rows.append(["3D Bildaufnahme","Zugestimmt "])
        self._section("Aufklärung",rows,story)

        # ── ENTLASSUNG ───────────────────────────────────────────
        en=akte.get("entlassung",{})
        LABELS_E=[("datum","Entlassungsdatum"),("status","Status"),
                  ("bericht","Entlassungsbericht"),("empfehlung","Empfehlungen"),("medplan","Medikamentenplan")]
        rows=[[lb,str(en[k])] for k,lb in LABELS_E if k in en and self._v(en.get(k))]
        self._section("Entlassung",rows,story)

        # ── MESSUNGEN ────────────────────────────────────────────
        sessions=patient.get_sessions()
        if sessions:
            if session is None: session=sessions[0]
            meas=session.get("meas",{})
            if meas and any(self._v(v) for v in meas.values()):
                rows=[["Messung","Wert"]]+[[k,f"{v} cm"] for k,v in meas.items() if self._v(v)]
                block=[]
                block.append(Paragraph("3D Messungen",self._hs()))
                t2=Table(rows,colWidths=[100*mm,70*mm])
                t2.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),self.G),("TEXTCOLOR",(0,0),(-1,0),white),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
                    ("PADDING",(0,0),(-1,-1),5),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,self.BG]),
                    ("BOX",(0,0),(-1,-1),0.5,self.G),
                    ("INNERGRID",(0,0),(-1,-1),0.25,HexColor("#cccccc"))]))
                block.append(t2); block.append(Spacer(1,8*mm))
                story.append(KeepTogether(block))

        # ── UNTERSCHRIFT ─────────────────────────────────────────
        story.append(Spacer(1,16*mm))
        story.append(HRFlowable(width="100%",thickness=0.5,color=HexColor("#cccccc")))
        story.append(Spacer(1,4*mm))
        sig=Table([["Datum / Unterschrift Arzt:","Datum / Unterschrift Patient:"],["_"*35,"_"*35]],
                  colWidths=[85*mm,85*mm])
        sig.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),9),("PADDING",(0,0),(-1,-1),6),
                                  ("TOPPADDING",(0,1),(-1,1),12)]))
        story.append(sig)

        if not story: story.append(Paragraph("Keine Daten vorhanden.",ParagraphStyle("n",fontSize=10)))
        doc.build(story)
        print(f"Report: {pdf}"); return str(pdf)

if __name__=="__main__":
    print("ReportLab:",HAS_RL)
