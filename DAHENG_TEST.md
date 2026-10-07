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

## Zwei Kameras gleichzeitig

```powershell
.\.venv\Scripts\python.exe daheng_dual_camera_test.py --serials FHK26080102 FHK26080092
```

Der Test nimmt standardmaessig 3000 Frames pro Kamera auf. Beide Kameras
werden in getrennten Prozessen vorbereitet und ueber einen gemeinsamen
Softwarestart freigegeben. Der Start synchronisiert keine Belichtungen.
Je Kamera wird eine CSV mit Frame-Metadaten geschrieben; eine gemeinsame
summary.json enthaelt Aufnahmefehler, Frame-ID-Anomalien, Empfangs-FPS und
die zeitliche Ueberlappung. Fuer einen bestandenen gemeinsamen Test muessen
beide Einzelaufnahmen fehlerfrei sein und mindestens 90 Prozent der kuerzeren
Aufnahme mit mindestens zwei empfangenen Frames je Kamera ueberlappen.

## Bestaetigter Hardwarestand am 07.10.2026

Commit der getesteten Aufnahmeprogramme: 2dd6942.
Die folgenden Werte stammen aus Alexanders auf dem Windows-PC ausgefuehrtem
Testlauf (21:12 bis 21:13 UTC / 23:12 bis 23:13 Europe/Vienna).

| Kamera | Aufloesung | Gueltige Frames | Empfangs-FPS | Fehlende Frame-IDs |
| --- | --- | --- | --- | --- |
| MER2-240-159U3M / FHK26080092 | 2048 x 1200 | 3000 / 3000 | 98.7445 | 0 |
| MER2-240-159U3M / FHK26080102 | 2048 x 1200 | 3000 / 3000 | 98.7435 | 0 |

Gemeinsamer Empfang: 30.3713 Sekunden; keine gemeldeten Aufnahme- oder
Freigabefehler; Gesamtstatus passed=true. Die Kameras liefen mit ihren
vorhandenen Einstellungen. 60 FPS wurden dabei nicht eingestellt. Dieser
Nachweis betrifft die zwei getesteten Kameras und diesen Aufnahmeweg;
Bildanzeige, Bildspeicherung und Stereo-Synchronisation sind weitere Schritte.
Rohprotokolle liegen lokal unter daheng_test_results/dual_20261007T211242_899402Z/.

## Live-Vorschau

Zuerst die Anzeige-Abhaengigkeiten installieren. Der bestehende NumPy-Pin
bleibt bestehen; gxipy kommt weiterhin aus dem installierten Galaxy SDK.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-daheng-preview.txt
.\.venv\Scripts\python.exe daheng_live_preview.py --serials FHK26080102 FHK26080092
```

Galaxy Viewer und andere Kameraaufnahmeprogramme vorher schliessen.
Die beiden Bilder erscheinen nebeneinander. ESC/Q oder das Fensterschliessen
beendet die Aufnahme. Z wechselt zwischen Gesamtbild und einem unvergroesserten
1:1-Mittelausschnitt zum Scharfstellen. Die Kameraaufloesung, Belichtung, Gain,
Pixel Format und die eingestellte Kamera-Framerate werden nicht veraendert.
Wie im Aufnahmetest werden nur Continuous/Trigger Off voruebergehend gesetzt
und beim Beenden wiederhergestellt.

RX FPS zeigt die Empfangsrate am Host. Die Vorschau wird getrennt davon auf
maximal 15 Updates je Sekunde begrenzt und speichert keine Bilddaten. SDK-Puffer
werden nach Verarbeitung zurueckgegeben; die Anzeige verwendet eigene kopierte
Pixel in gemeinsamem Prozessspeicher. Uebersprungene Anzeige-Updates sind keine
fehlenden Aufnahmeframes. Die aktuelle Vorschau unterstuetzt ungepackte Mono8,
Mono10, Mono12, Mono14 und Mono16; hochbitige Bilder werden nur fuer die Anzeige
mit festem Bitshift auf 8 Bit abgebildet. Kamerazeitstempel werden nicht zur
Synchronisation der Anzeige verwendet; beide Kameras laufen frei.

Fuer einen begrenzten Lauf kann --duration-seconds 10 angegeben werden.
--headless verarbeitet dieselben Vorschaubilder ohne GUI und erfordert eine
positive Dauer. Der Abschlussbericht enthaelt Fehler, verarbeitete Frames und
Frame-ID-Anomalien. Eine erfolgreiche Vorschau ersetzt keinen Dauer-,
Speicher- oder metrischen Kalibrierungstest.

