# AnthroPrecis: isolierter Daheng-Test

Voraussetzungen: Python 3.12, installierter Galaxy SDK inklusive USB3-Treiber,
angeschlossene Daheng-Kamera. Galaxy Viewer vor dem Test schliessen.

In PowerShell im Repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-daheng.txt
.\.venv\Scripts\python.exe daheng_camera_test.py
```

Bei mehreren Kameras fordert der Test die Auswahl einer Seriennummer:

```powershell
.\.venv\Scripts\python.exe daheng_camera_test.py --serial SERIENNUMMER
```

Der SDK-Python-Pfad kann mit `--sdk-python` oder `GALAXY_SDK_PYTHON`
gesetzt werden. gxipy wird direkt aus diesem SDK geladen, nicht aus PyPI.

Standard: 100 Frames, maximal 2000 ms Wartezeit pro Frame. Der Test verwendet
die aktuelle Aufloesung und setzt voruebergehend kontinuierliche Aufnahme ohne
Trigger. Danach werden die geaenderten Modi wiederhergestellt. Belichtung,
Gain, Pixel Format und FPS werden nicht eingestellt.

Jeder Lauf erzeugt einen eigenen Ordner unter `daheng_test_results/` mit
`frames.csv` und `summary.json`, auch bei Aufnahmefehlern. Bilddaten werden
bei diesem ersten Test nicht gespeichert. Kamera-Zeitstempel bleiben als
SDK-Rohwerte erhalten; ihre Einheit wird nicht vorausgesetzt.

Exitcode 0 bedeutet: alle angeforderten Frames erfolgreich, fortlaufende
Frame-IDs und keine Aufnahme- oder Freigabefehler. Exitcode 1 bedeutet einen
fehlgeschlagenen Test. Frame-ID-Luecken geben Hinweise auf fehlende Frames;
Verluste vor dem ersten oder nach dem letzten Frame sind damit nicht erkennbar.

`arrival_fps` wird aus den Host-Ankunftszeiten zwischen erstem und letztem
Frame berechnet. `valid_frames_per_elapsed_s` umfasst auch den Aufnahmestart.
Beide messen diesen kurzen Lauf, nicht die langfristige Stabilitaet oder
eine garantierte Sensor-Framerate. Das Ziel von 60 FPS wird separat eingestellt
und in spaeteren laengeren Tests geprueft.
