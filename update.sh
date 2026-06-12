#!/bin/bash
# ANTHRO3D — Update + Start Script
# Laedt neueste Version, patcht, startet mit Logging

ANTHRO=~/anthro3d
LOG=$ANTHRO/debug.log

# Neueste Version aus Downloads laden
LATEST=$(ls -t ~/Downloads/anthro3d_app*.py 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    cp "$LATEST" $ANTHRO/anthro3d_app.py
    echo "Update von: $LATEST"
    # Patches anwenden
    python3 -c "
with open('$ANTHRO/anthro3d_app.py','r') as f: c=f.read()
c=c.replace('        stereo_title.setStyleSheet(','        stereo_title = QLabel(\"Stereo-Kalibrierung\")\n        stereo_title.setStyleSheet(')
with open('$ANTHRO/anthro3d_app.py','w') as f: f.write(c)
print('Patched OK')
"
else
    echo "Kein Download gefunden — starte aktuelle Version"
fi

# Log rotieren (max 1MB)
if [ -f "$LOG" ] && [ $(wc -c < "$LOG") -gt 1048576 ]; then
    mv "$LOG" "$LOG.old"
    echo "Log rotiert"
fi

echo "=============================" >> $LOG
echo "Start: $(date)" >> $LOG
echo "=============================" >> $LOG

# Terminal 2 oeffnen mit Live-Log
osascript -e 'tell app "Terminal" to do script "echo ANTHRO3D Debug Log; tail -f ~/anthro3d/debug.log | grep -E --color=always \"ERROR|Fehler|WARNING|✓|✗|Frames|recording|Aufnahme|Kamera|fps\""' 2>/dev/null

# App starten mit Logging
echo "Starte App..."
cd $ANTHRO
python3 anthro3d_app.py 2>&1 | tee -a $LOG
