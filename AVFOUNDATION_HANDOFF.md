# AVFoundation-Zwischenstand

## Funktioniert

- Eigene Python-Umgebung: `.venv-avfoundation`
- Python 3.14
- PyObjC 12.2.1
- AVFoundation, CoreMedia und Quartz verfügbar
- Kameras werden ohne OpenCV-Indizes über `uniqueID` erkannt
- Der direkte Hauptthread-Test `probe_avfoundation_notifications.py`
  erkennt Trennung und Wiederanschluss zuverlässig
- OV9281-Hub wurde erfolgreich getestet
- Kameraanzahl und Kameratypen sind im neuen System nicht fest begrenzt

## Erkannte Geräte

### ELP-Stereokameras

- `0x110000032e42b10`
- `0x210000032e42b10`

Die endgültige Zuordnung ELP1/ELP2 muss noch anhand direkter
Vorschaubilder bestätigt werden.

### OV9281

- `0x1300000c45636d`
- `0x1410000c45636d`

Die endgültige Zuordnung links/rechts muss noch anhand direkter
Vorschaubilder bestätigt werden.

### Interne Kamera

- FaceTime HD-Kamera:
  `47B4B64B-7067-4B9C-AD2B-AE273A71F4B5`

## Noch offen

Der direkte Notification-Test auf dem Hauptthread funktioniert.

Der generische `CameraHotplugMonitor` in `camera_system/hotplug.py`
erkennt Abziehen und Wiederanschließen während seiner Laufzeit noch
nicht zuverlässig. Wahrscheinliche Ursache ist die Thread- bzw.
Runloop-Architektur.

Nicht weiter an Discovery-Polling experimentieren, bevor geprüft wurde,
ob der Notification-Observer dauerhaft auf dem Foundation-Hauptthread
laufen und Ereignisse über eine thread-sichere Queue an den
CameraManager übergeben sollte.

## Empfohlener nächster Schritt

1. Hot-Plug-Observer auf dem Hauptthread belassen.
2. Notification-Ereignisse in eine thread-sichere Queue schreiben.
3. CameraManager verarbeitet diese Queue.
4. Danach direkten `AVFoundationCapture` bauen:
   - Öffnen über uniqueID
   - activeFormat auswählen
   - AVCaptureVideoDataOutput
   - Delegate
   - CVPixelBuffer zu NumPy/BGR
   - synchrones `read(timeout)`
5. Erst danach `scan3d.py` umstellen.

## Wichtige Regel

`camera_registry.local.yaml` enthält lokale Hardware-IDs und wird
nicht in Git eingecheckt.
