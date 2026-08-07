# ANTHRO3D – Stand 07.08.2026

## Funktioniert

- Vier Kameras werden stabil per AVFoundation uniqueID erkannt.
- ELP2, ELP1, OV9281_L und OV9281_R werden korrekt benannt.
- camera_alignment_view.py funktioniert.
- Live-Vorschaubilder und ArUco-Anzeige funktionieren.
- ArUco-System verwendet:
  - OV9281_STAND = ID2 / ID20
  - ELP1_STAND = ID3 / ID30
  - ELP2_STAND = ID4 / ID40
- marker_vertical_offsets.yaml ist verifizierte Hardware-Geometrie.
- marker_integrity_reference.yaml wurde erzeugt.
- redundante Markerintegrität funktioniert:
  - alle drei Stative OK
  - safe_for_auto_recalibration=True
  - marker_fault_detected=False
- Zwei-Marker-Fusion verbessert zeitliche Stabilität deutlich.

## Noch ungelöst

Der Kamera-Graph schließt geometrisch noch nicht ausreichend.

Gemessen:
- Dreiecksschluss ca. 12–14 cm Translation
- ca. 4–7 Grad Rotation

Einzelmarker sind nicht die Hauptursache.
Stereo R->L Transformationsformel wurde geprüft und ist mathematisch korrekt.
Nur linke oder nur rechte Kameraseiten lösen den Fehler ebenfalls nicht.

Nächster Ansatz:
Beide ArUcos eines Stativs als EIN starres 3D-Target benutzen.
Aus 2/20, 3/30 und 4/40 jeweils 8 bekannte 3D-Eckpunkte erzeugen und
eine gemeinsame Pose mit solvePnP/solvePnPGeneric bestimmen.

## Letzter Scan 07.08.2026 12:12

Scan wurde vollständig gespeichert.

ELP2:
- 140758 Rohpunkte
- Bereich ungefähr X -32 bis 22 cm
- Z 50 bis 73 cm

ELP1:
- 160493 Punkte
- nach Transformation X 28 bis 80 cm
- Z 79 bis 148 cm
- liegt derzeit deutlich falsch zu ELP2

OV9281:
- Z roh ca. 449 bis 2098 cm
- nach aktueller Transformation außerhalb Scanbox
- 52066 -> 0 Punkte
- wurde nicht zur Fusion hinzugefügt

Fusion:
- ELP1 + ELP2
- 301251 Punkte vor Filter
- 7414 Punkte finale Voxel/Outlier-Wolke

WICHTIG:
scan3d.py verwendet aktuell noch:
"OV9281->ELP2 kalibrieren (Marker ID 10...)"

ID10 ist Altbestand und muss auf das neue 2/20-3/30-4/40-System
umgestellt werden.

scan3d.py darf deshalb derzeit NICHT als endgültig geometrisch
kalibriert betrachtet werden.

## UI-Ziel

Später weiterhin eine gemeinsame ANTHRO3D-App mit:

- automatische uniqueID-Kameraerkennung
- Kamera-Livevorschau / Vollbild zur Ausrichtung
- ArUco-Sichtkontrolle
- selbstüberwachender Kamera-Graph
- automatische Neuberechnung nach Kameraverschiebung
- Markerfehler-Sperre
- 3D-Scan
- Körpervermessung
- Vermessungs-Vorschaubilder
- Patientenverwaltung
- Berichte / Speichern
