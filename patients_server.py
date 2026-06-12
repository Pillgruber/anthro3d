#!/usr/bin/env python3
"""
ANTHRO3D — Patienten-Server
=============================
HTTP + WebSocket Server für Patientenverwaltung.
Speichert Patienten in SQLite, stellt REST-API bereit.

Starten:
  python3 patients_server.py

API:
  GET  /api/patients           → alle Patienten
  GET  /api/patients?q=name    → Suche (Autocomplete)
  GET  /api/patients/{id}      → einzelner Patient
  POST /api/patients           → neuer Patient
  PUT  /api/patients/{id}      → Patient bearbeiten
  GET  /api/patients/{id}/sessions → Sessions eines Patienten
"""

import sqlite3, json, os, re, datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

DB_PATH      = Path("~/anthro3d/patients.db").expanduser()
SESSIONS_DIR = Path("~/anthro3d/sessions").expanduser()
PORT         = 8765

# ── DATENBANK ─────────────────────────────────────────────────────────────────
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id           TEXT PRIMARY KEY,
            vorname      TEXT NOT NULL,
            nachname     TEXT NOT NULL,
            geburtsdatum TEXT,
            strasse      TEXT,
            plz          TEXT,
            ort          TEXT,
            land         TEXT DEFAULT 'Österreich',
            telefon      TEXT,
            email        TEXT,
            notizen      TEXT,
            erstellt_am  TEXT NOT NULL,
            geaendert_am TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            patient_id  TEXT NOT NULL,
            datei       TEXT NOT NULL,
            aufnahme_at TEXT NOT NULL,
            fps         INTEGER,
            frames      INTEGER,
            dauer_s     REAL,
            has_stereo  INTEGER DEFAULT 0,
            notizen     TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );
    """)
    con.commit()
    con.close()
    print(f"  Datenbank: {DB_PATH}")

def get_db():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con

def next_patient_id():
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) as n FROM patients")
    n = cur.fetchone()["n"]
    con.close()
    return f"PAT{n+1:04d}"

def row_to_dict(row):
    return dict(row) if row else None

def rows_to_list(rows):
    return [dict(r) for r in rows]


# ── HTTP HANDLER ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Kein Logging im Terminal

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",  "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
            ext = Path(path).suffix.lower()
            mime = {".html":"text/html",".js":"application/javascript",
                    ".css":"text/css",".json":"application/json"}.get(ext,"text/plain")
            self.send_response(200)
            self.send_header("Content-Type",   mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_json({"error": "Nicht gefunden"}, 404)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length > 0 else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)

        # ── Statische Dateien
        if path == "/" or path == "/patients":
            self.send_file(Path("~/anthro3d/patients.html").expanduser())
            return

        # ── API: Alle Patienten / Suche
        if path == "/api/patients":
            q = qs.get("q", [None])[0]
            con = get_db()
            if q:
                like = f"%{q}%"
                rows = con.execute(
                    """SELECT * FROM patients
                       WHERE nachname LIKE ? OR vorname LIKE ?
                          OR (nachname || ' ' || vorname) LIKE ?
                          OR (vorname || ' ' || nachname) LIKE ?
                          OR id LIKE ?
                       ORDER BY nachname, vorname LIMIT 20""",
                    (like, like, like, like, like)).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM patients ORDER BY nachname, vorname").fetchall()
            con.close()
            self.send_json(rows_to_list(rows))
            return

        # ── API: Einzelner Patient
        m = re.match(r"^/api/patients/(PAT\d+)$", path)
        if m:
            pid = m.group(1)
            con = get_db()
            pat = con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
            if not pat:
                con.close(); self.send_json({"error": "Patient nicht gefunden"}, 404); return
            result = row_to_dict(pat)
            # Sessions dazu
            result["sessions"] = rows_to_list(
                con.execute("SELECT * FROM sessions WHERE patient_id=? ORDER BY aufnahme_at DESC", (pid,)).fetchall())
            con.close()
            self.send_json(result)
            return

        # ── API: Sessions eines Patienten
        m = re.match(r"^/api/patients/(PAT\d+)/sessions$", path)
        if m:
            pid = m.group(1)
            con = get_db()
            rows = con.execute(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY aufnahme_at DESC", (pid,)).fetchall()
            con.close()
            self.send_json(rows_to_list(rows))
            return

        # ── API: Session-Dateien scannen und registrieren
        if path == "/api/scan-sessions":
            self._scan_sessions()
            return

        self.send_json({"error": "Nicht gefunden"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        # ── Neuen Patienten anlegen
        if path == "/api/patients":
            data = self.read_body()
            if not data.get("vorname") or not data.get("nachname"):
                self.send_json({"error": "Vor- und Nachname erforderlich"}, 400)
                return
            now = datetime.datetime.now().isoformat()
            pid = next_patient_id()
            con = get_db()
            try:
                con.execute("""INSERT INTO patients
                    (id, vorname, nachname, geburtsdatum, strasse, plz, ort, land,
                     telefon, email, notizen, erstellt_am, geaendert_am)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    pid,
                    data.get("vorname","").strip(),
                    data.get("nachname","").strip(),
                    data.get("geburtsdatum",""),
                    data.get("strasse",""),
                    data.get("plz",""),
                    data.get("ort",""),
                    data.get("land","Österreich"),
                    data.get("telefon",""),
                    data.get("email",""),
                    data.get("notizen",""),
                    now, now))
                con.commit()
                pat = row_to_dict(con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone())
                con.close()
                print(f"  Neuer Patient: {pid} — {data.get('vorname')} {data.get('nachname')}")
                self.send_json(pat, 201)
            except Exception as e:
                con.close()
                self.send_json({"error": str(e)}, 500)
            return

        # ── Session registrieren
        if path == "/api/sessions":
            data = self.read_body()
            con = get_db()
            try:
                now = datetime.datetime.now().isoformat()
                con.execute("""INSERT OR REPLACE INTO sessions
                    (id, patient_id, datei, aufnahme_at, fps, frames, dauer_s, has_stereo, notizen)
                    VALUES (?,?,?,?,?,?,?,?,?)""", (
                    data.get("id", f"SES{int(datetime.datetime.now().timestamp())}"),
                    data["patient_id"],
                    data["datei"],
                    data.get("aufnahme_at", now),
                    data.get("fps", 10),
                    data.get("frames", 0),
                    data.get("dauer_s", 0),
                    1 if data.get("has_stereo") else 0,
                    data.get("notizen", "")))
                con.commit()
                con.close()
                self.send_json({"ok": True})
            except Exception as e:
                con.close()
                self.send_json({"error": str(e)}, 500)
            return

        self.send_json({"error": "Nicht gefunden"}, 404)

    def do_PUT(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        m = re.match(r"^/api/patients/(PAT\d+)$", path)
        if m:
            pid  = m.group(1)
            data = self.read_body()
            now  = datetime.datetime.now().isoformat()
            con  = get_db()
            try:
                con.execute("""UPDATE patients SET
                    vorname=?, nachname=?, geburtsdatum=?, strasse=?, plz=?, ort=?,
                    land=?, telefon=?, email=?, notizen=?, geaendert_am=?
                    WHERE id=?""", (
                    data.get("vorname","").strip(),
                    data.get("nachname","").strip(),
                    data.get("geburtsdatum",""),
                    data.get("strasse",""),
                    data.get("plz",""),
                    data.get("ort",""),
                    data.get("land","Österreich"),
                    data.get("telefon",""),
                    data.get("email",""),
                    data.get("notizen",""),
                    now, pid))
                con.commit()
                pat = row_to_dict(con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone())
                con.close()
                self.send_json(pat)
            except Exception as e:
                con.close()
                self.send_json({"error": str(e)}, 500)
            return
        self.send_json({"error": "Nicht gefunden"}, 404)

    def _scan_sessions(self):
        """Scannt sessions/ Ordner und gibt Dateiliste zurück."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        files = list(SESSIONS_DIR.glob("*.anthro3d"))
        result = [{"name": f.name, "size_mb": round(f.stat().st_size/1024/1024, 1),
                   "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
                  for f in sorted(files, reverse=True)]
        self.send_json(result)


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  ANTHRO3D — Patienten-Server")
    print("="*52)
    init_db()
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"  Server: http://localhost:{PORT}")
    print(f"  Browser: http://localhost:{PORT}/patients")
    print("\n  Ctrl+C zum Beenden\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Beendet.")
