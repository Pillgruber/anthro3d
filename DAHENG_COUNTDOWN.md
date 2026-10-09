# AnthroPrecis – Daheng Countdown-Aufnahme

## Windows-Start

Galaxy Viewer schließen, beide Daheng-Kameras anschließen. PowerShell im Projektordner öffnen:

```powershell
git switch feature/daheng-camera
git pull --ff-only
.\.venv\Scripts\python.exe daheng_system_runner.py --system 1 --mode capture
```

Kamerasystem 1: Kamera-ID 3 (FHK26060051) und Kamera-ID 30 (FHK26080099).

Leertaste: Countdown von 5 Sekunden und Aufnahme. Z: 1:1-Ansicht. ESC/Q: Abbrechen.
Optional: `--countdown-seconds 3` oder `--auto`.

## Speicherort – ein Ordner direkt am Desktop

Das Programm erstellt bei Bedarf automatisch **AnthroPrecis-Aufnahmen** auf dem
Windows-Desktop (auch bei Desktop-Umleitung durch OneDrive).

**Keine Aufnahme-Unterordner.** Alle Dateien liegen direkt in diesem Verzeichnis:
- `20261009_160001_123456_camera_FHK26060051.png`
- `20261009_160001_123456_camera_FHK26080099.png`
- `20261009_160001_123456_capture.json`

Jede Aufnahme erhält einen einzigartigen Datums-/Uhrzeit-Präfix.
Im Windows Explorer nach **Name → Absteigend** sortieren, um die neuesten
Aufnahmen zuerst zu sehen. Alternativ **Änderungsdatum → Absteigend**.

Das Verzeichnis kann auf Wunsch überschrieben werden: `--output-dir "D:\Aufnahmen"`.
Die bisherigen Messdateien in `daheng_test_results` bleiben erhalten.

## Aufnahmetechnik

Mono8 wird als verlustfreie 8-Bit-PNG gespeichert, Mono10/12/14/16 unverändert
als 16-Bit-PNG. Die JSON-Datei hält Frame-IDs, Zeitstempel, Kameraeinstellungen
und eventuelle Fehler fest.

Die beiden Kameras laufen frei; ihre Belichtungszeitpunkte sind nicht
hardware-synchronisiert. Host-Empfangszeitdifferenzen sind keine gemessenen
Belichtungszeitdifferenzen. Belichtung, Gain, Bildgröße und Kamera-FPS
werden nicht aktiv verändert. Die neue Countdown-Funktion muss noch
auf der tatsächlichen Hardware getestet werden.
