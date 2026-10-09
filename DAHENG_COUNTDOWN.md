# AnthroPrecis – Daheng Countdown-Aufnahme

## Windows-Start

Galaxy Viewer schließen, beide Daheng-Kameras anschließen. PowerShell im Projektordner öffnen:

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

## Automatischer Helligkeitsabgleich vor dem Foto

Mit Betätigung der Leertaste beginnt der 3-Sekunden-Countdown. Währenddessen
passen beide Kameras **automatisch und individuell** ihre physische Belichtungszeit
und bei Bedarf Gain an denselben Zielhelligkeitswert (105 / 255) an.
Die aktuelle erste Version bewertet einen mittleren Bildbereich
(60 % Bildbreite, 70 % Bildhöhe), blendet sehr helle Pixel bei der
Messwertberechnung weitgehend aus und prüft auch den überbelichteten Anteil.
Später ersetzen wir diese Näherung durch eine Körpermasken-basierte Regelung.

Die beiden Messwerte müssen jeweils höchstens 5 % vom gemeinsamen Sollwert
abweichen und dürfen sich um höchstens 5 % unterscheiden.
Die automatische Regelung wartet auf stabile Messwerte und fixiert dann
Belichtung und Gain für die Aufnahme. Nötigenfalls läuft der stille
Abgleich nach dem Countdown noch maximal 18 Sekunden weiter.
**Eine Fehlermeldung erscheint nur, wenn keine gültige Aufnahme möglich ist.**
Technische Auffälligkeiten, die die Messung nicht blockieren, werden nur
im JSON-Protokoll gespeichert.

Die Regelung beschränkt die Belichtungszeit vorerst auf max. 8 ms und Gain
auf 8 dB, um hohe Bildraten und geringes Rauschen zu priorisieren.
Der exakte Belichtungsabgleich ist derzeit ein erster Entwicklungsstand:
Messgenauigkeit, reale Kamerabedingungen und Pixelmasken müssen noch
auf dem Windows-PC validiert werden. Er ersetzt nicht die geometrische
Kamerakalibrierung oder eine echte Belichtungssynchronisation.

