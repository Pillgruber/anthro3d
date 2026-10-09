# AnthroPrecis – Daheng Countdown-Aufnahme

## Windows-Start

Galaxy Viewer schließen, beide Daheng-Kameras anschließen. PowerShell im Projektordner öffnen:

```powershell
git switch feature/daheng-camera
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements-daheng-preview.txt
.\.venv\Scripts\python.exe daheng_countdown_capture.py --serials FHK26080102 FHK26080092
```

Leertaste: Countdown von 5 Sekunden und Snapshot. Z: 1:1-Ansicht. ESC/Q: Abbrechen.
Optional: --countdown-seconds 3 oder --auto.

Jeder Lauf erzeugt einen neuen Ordner daheng_test_results/snapshot_<UTC> mit zwei verlustfreien PNGs und capture.json.
Mono8 wird als 8-Bit-PNG gespeichert, Mono10/12/14/16 unverändert als 16-Bit-PNG.
Die Kameras laufen frei; die Aufnahme ist nicht hardware-synchronisiert.
Host-Empfangszeitdifferenzen sind keine gemessenen Belichtungszeitdifferenzen.
Belichtung, Gain, Bildgröße und Kamera-FPS werden nicht aktiv verändert.
Das Skript ist noch nicht an realer Hardware validiert.