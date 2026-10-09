# AnthroPrecis – Daheng Countdown-Aufnahme

## Windows-Start

Galaxy Viewer schließen, beide Daheng-Kameras anschließen. PowerShell im Projektordner öffnen:

Vor dem ersten Hardwaretest können die Tests der Helligkeitslogik ausgeführt werden:

```powershell
.\.venv\Scripts\python.exe -m unittest -q test_daheng_auto_brightness
```

```powershell
git switch feature/daheng-camera
git pull --ff-only
.\.venv\Scripts\python.exe daheng_system_runner.py --system 1 --mode capture
```

Kamerasystem 1: Kamera-ID 3 (FHK26060051) und Kamera-ID 30 (FHK26080099).

Leertaste: Countdown von 3 Sekunden und Aufnahme. Z: 1:1-Ansicht. ESC/Q: Abbrechen.
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
hardware-synchronisiert. Host-Empfangszeitdifferenzen sind keine gemessenen Belichtungszeitdifferenzen.

## Automatische Belichtung und 3-Sekunden-Countdown

Schon bei der Live-Vorschau regelt jede Kamera Belichtungszeit und bei Bedarf
Gain automatisch. Die laufenden Anpassungen werden insbesondere **während des
3-Sekunden-Countdowns** fortgesetzt. Beide Kameras verwenden denselben
vorläufigen Helligkeits-Zielwert (105 / 255), dürfen wegen unterschiedlicher
Lichtverhältnisse aber andere Belichtungszeiten haben.

Nach 3 Sekunden friert das Programm die Einstellungen ein und wartet nur auf
je ein bestätigendes Bild nach dieser Fixierung (Timeout standardmäßig 2 s).
**Das frühere Warten auf exakt ±5 % gleiche Helligkeit entfällt.** Eine nicht
erreichbare Übereinstimmung wird in `capture.json` protokolliert und verhindert
die Aufnahme nicht, solange beide Bilder ausreichend verwertbare Grauwerte
enthalten. Vollständig dunkle bzw. ausgebrannte Aufnahmen werden weiterhin
abgelehnt. Diese Plausibilitätsprüfung ist bewusst vorläufig: Ohne erkannte
Person und Kalibrierung kann sie die Qualität einer Körpervermessung nicht
verlässlich beurteilen.

Als Messbereich dient vorläufig die Bildmitte (60 % Breite, 70 % Höhe). Die
Regelung ignoriert sehr helle Pixel für den Steuerwert und bewertet zusätzlich
die Zahl der Zwischentöne. Helle Fenster können dadurch weniger stark in die
Belichtung eingreifen, aber Software kann verlorene Details nicht zurückholen.

Die Regelung beschränkt die Belichtungszeit vorerst auf maximal 8 ms und Gain
auf 8 dB. Bei fehlender Bildinformation bzw. Kameraausfall erscheint nur eine
kurze, verständliche Fehlermeldung. Detaildiagnostik, der Helligkeitsabgleich
und die finalen Kameraeinstellungen bleiben im JSON-Protokoll.

Die Anzeige für empfangene Bilder pro Sekunde (RX FPS) wird jetzt im
Countdown-Modus korrekt aktualisiert. Die Aufnahme ist weiterhin eine
**Doppel-Fotoaufnahme**, noch kein kontinuierlicher 3D-Scan. Außerdem gibt es
noch **keine Hardware-Synchronisation** der Belichtungszeitpunkte.

Die neue Version muss noch mit den realen Kameras unter verschiedenen
Lichtverhältnissen geprüft werden. Tests ohne Kameras:

```powershell
.\.venv\Scripts\python.exe -m unittest -q test_daheng_auto_brightness
```
