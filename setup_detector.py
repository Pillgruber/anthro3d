import cv2, numpy as np, yaml, json, threading, time, math, asyncio, websockets
from scipy.spatial.transform import Rotation

class SetupConfig:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.marker_size_cm   = cfg["calibration"]["marker_size_cm"]
        self.marker_size_m    = self.marker_size_cm / 100.0
        self.tolerance_cm     = 5.0
        self.stability_frames = cfg["calibration"]["stability_frames"]
        self.ema_alpha        = cfg["calibration"]["ema_alpha"]
        self.mode             = "compact"
        self.target_radius    = {"compact": 175, "pro": 250}
        self.target_angles    = {1: 270, 2: 30, 3: 150}
    def get_target_radius(self):
        return self.target_radius[self.mode]
    def get_target_position(self, cam_id):
        a = math.radians(self.target_angles[cam_id])
        r = self.get_target_radius()
        return np.array([r*math.cos(a), r*math.sin(a)])

class ArucoDetector:
    def __init__(self, marker_size_m):
        self.marker_size_m = marker_size_m
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector      = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        self.camera_matrix = np.array([[700,0,640],[0,700,360],[0,0,1]], dtype=np.float64)
        self.dist_coeffs   = np.zeros((4,1))
    def detect(self, frame):
        if frame is None: return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        markers = []
        if ids is None: return markers
        for i, mid in enumerate(ids.flatten()):
            c = corners[i]
            objPoints = np.array([[-self.marker_size_m/2, self.marker_size_m/2, 0],
                                  [ self.marker_size_m/2, self.marker_size_m/2, 0],
                                  [ self.marker_size_m/2,-self.marker_size_m/2, 0],
                                  [-self.marker_size_m/2,-self.marker_size_m/2, 0]], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(objPoints, corners[i][0].astype(np.float32),
                                               self.camera_matrix, self.dist_coeffs)
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            markers.append({
                "id": int(mid), "corners": c[0],
                "rvec": rvec, "tvec": tvec,
                "distance_m": np.linalg.norm(tvec),
                "center": c[0].mean(axis=0)
            })
        return markers
    def draw_markers(self, frame, markers):
        for m in markers:
            cv2.polylines(frame, [m["corners"].astype(int)], True, (80,150,100), 2)
            cx,cy = int(m["center"][0]), int(m["center"][1])
            cv2.putText(frame, f"ID:{m['id']} {m['distance_m']*100:.1f}cm",
                       (cx-50,cy-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,150,100), 1)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                             m["rvec"], m["tvec"], self.marker_size_m*0.5)
        return frame

class PositionCalculator:
    def __init__(self, config):
        self.config        = config
        self.ema_positions = {}
        self.stability     = {}
    def update(self, cam_id, markers):
        other = [m for m in markers if m["id"] != cam_id]
        if not other: return None
        best = min(other, key=lambda m: m["distance_m"])
        tx   = best["tvec"][0] * 100
        tz   = best["tvec"][2] * 100
        pos  = np.array([-tx, tz])
        key  = str(cam_id)
        if key not in self.ema_positions:
            self.ema_positions[key] = pos
            self.stability[key]     = 0
        else:
            a = self.config.ema_alpha
            self.ema_positions[key] = a*pos + (1-a)*self.ema_positions[key]
        dev = np.linalg.norm(pos - self.ema_positions[key])
        if dev < 2.0:
            self.stability[key] = min(self.stability[key]+1, self.config.stability_frames)
        else:
            self.stability[key] = max(self.stability[key]-1, 0)
        return self.ema_positions[key]
    def get_deviation(self, cam_id, pos):
        target  = self.config.get_target_position(cam_id)
        vec     = target - pos
        dist_cm = np.linalg.norm(vec)
        angle   = math.degrees(math.atan2(vec[1], vec[0]))
        tol     = self.config.tolerance_cm
        status  = "ok" if dist_cm<=tol else ("warn" if dist_cm<=tol*2 else "far")
        return {
            "deviation_cm":  round(dist_cm, 1),
            "direction":     (vec/(dist_cm+0.001)).tolist(),
            "angle_deg":     round(angle, 1),
            "status":        status,
            "target":        target.tolist(),
            "current":       pos.tolist(),
            "stability":     self.stability.get(str(cam_id), 0),
            "stability_max": self.config.stability_frames
        }

class CameraCapture:
    def __init__(self, cam_id, device_index):
        self.cam_id       = cam_id
        self.device_index = device_index
        self.cap          = None
        self.frame        = None
        self.running      = False
        self.lock         = threading.Lock()
        self.connected    = False
    def connect(self):
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            print(f"  FEHLER: Kamera {self.cam_id} nicht gefunden")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS,            60)
        self.connected = True
        print(f"  OK: Kamera {self.cam_id} verbunden")
        return True
    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
    def get_frame(self):
        with self.lock:
            return (self.frame.copy(), time.perf_counter_ns()) if self.frame is not None else (None, None)
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

class SetupDetector:
    def __init__(self, config_path="config.yaml"):
        print("\nANTHRO3D Setup Detector v2")
        print("=" * 40)
        self.config      = SetupConfig(config_path)
        self.detector    = ArucoDetector(self.config.marker_size_m)
        self.calculator  = PositionCalculator(self.config)
        self.cameras     = {}
        self.ws_clients  = set()
        self.latest_state= {}
        self.running     = False
        self.all_ready   = False
        self._load_cameras(config_path)
    def _load_cameras(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for c in cfg["cameras"]["tracking"]:
            if c["enabled"]:
                self.cameras[c["id"]] = CameraCapture(c["id"], c["device_index"])
    def set_mode(self, mode):
        self.config.mode = mode
        print(f"Modus: {mode.upper()} — Zielradius {self.config.get_target_radius()} cm")
    def connect_cameras(self):
        print(f"\nVerbinde {len(self.cameras)} Kameras:")
        connected = {cid: cam for cid, cam in self.cameras.items() if cam.connect()}
        self.cameras = connected
        if not self.cameras:
            print("Keine externen Kameras — nutze eingebaute Kamera (Demo)")
            demo = CameraCapture(1, 0)
            if demo.connect():
                self.cameras[1] = demo
        return len(self.cameras) > 0
    def _draw_arrow(self, frame, dev):
        h,w   = frame.shape[:2]
        cx,cy = w//2, h//2
        dx    = int(dev["direction"][0]*80)
        dy    = int(dev["direction"][1]*80)
        color = (160,130,50) if dev["status"]=="warn" else (160,70,70)
        cv2.arrowedLine(frame, (cx,cy), (cx+dx,cy+dy), color, 3, tipLength=0.3)
        cv2.putText(frame, f"{dev['deviation_cm']} cm",
                   (cx+dx//2-20, cy+dy//2-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    def _process_frame(self, cam_id, frame):
        markers   = self.detector.detect(frame)
        annotated = self.detector.draw_markers(frame.copy(), markers)
        position  = self.calculator.update(cam_id, markers)
        cam_state = {
            "cam_id": cam_id, "connected": True,
            "marker_count": len(markers),
            "markers": [{"id": m["id"], "distance_cm": round(m["distance_m"]*100,1)} for m in markers],
        }
        if position is not None:
            dev = self.calculator.get_deviation(cam_id, position)
            cam_state.update({
                "position": position.tolist(),
                "deviation_cm": dev["deviation_cm"],
                "angle_deg": dev["angle_deg"],
                "direction": dev["direction"],
                "status": dev["status"],
                "stability": dev["stability"],
                "stability_max": dev["stability_max"],
                "target": dev["target"],
            })
            colors = {"ok":(70,150,90),"warn":(160,130,50),"far":(160,70,70)}
            texts  = {"ok":f"OK {dev['deviation_cm']}cm","warn":f"Korrigieren {dev['deviation_cm']}cm","far":f"Zu weit {dev['deviation_cm']}cm"}
            color  = colors.get(dev["status"],(150,150,150))
            cv2.putText(annotated, f"CAM {cam_id}: {texts.get(dev['status'],'Suche...')}",
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if dev["status"] != "ok":
                self._draw_arrow(annotated, dev)
            cv2.putText(annotated, f"Stabilitaet: {dev['stability']}/{dev['stability_max']}",
                       (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cam_state["status"] = "searching"
            cv2.putText(annotated, f"CAM {cam_id}: Suche ArUco Marker...",
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 2)
        cv2.putText(annotated,
                   f"Modus: {self.config.mode.upper()} | Ziel: {self.config.get_target_radius()}cm",
                   (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,150,130), 1)
        return annotated, cam_state

    def _start_websocket_thread(self, host, port):
        async def handler(ws):
            self.ws_clients.add(ws)
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("cmd") == "set_mode":
                            self.set_mode(data["mode"])
                        elif data.get("cmd") == "get_state":
                            await ws.send(json.dumps(self.latest_state))
                    except Exception:
                        pass
            finally:
                self.ws_clients.discard(ws)
        async def broadcast_loop():
            while self.running:
                if self.ws_clients and self.latest_state:
                    msg = json.dumps(self.latest_state)
                    dead = set()
                    for ws in self.ws_clients:
                        try:
                            await ws.send(msg)
                        except Exception:
                            dead.add(ws)
                    self.ws_clients -= dead
                await asyncio.sleep(0.05)
        async def serve():
            async with websockets.serve(handler, host, port):
                print(f"WebSocket bereit: ws://{host}:{port}")
                await broadcast_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())

    def run(self, host="localhost", port=8766):
        self.running = True
        for cam in self.cameras.values():
            cam.start()
        threading.Thread(target=self._start_websocket_thread, args=(host,port), daemon=True).start()
        print(f"\nWebSocket: ws://{host}:{port}")
        print("Kameras laufen — Suche ArUco Marker...")
        print("Druecke Q zum Beenden\n")
        while self.running:
            state   = {}
            all_ok  = True
            for cam_id, cam in self.cameras.items():
                frame, _ = cam.get_frame()
                if frame is None:
                    all_ok = False
                    continue
                annotated, cam_state = self._process_frame(cam_id, frame)
                pass  # cv2.imshow deaktiviert
                state[str(cam_id)] = cam_state
                if cam_state.get("status") not in ("ok",):
                    all_ok = False
            self.all_ready     = all_ok and len(state) >= 1
            state["all_ready"] = self.all_ready
            state["mode"]      = self.config.mode
            state["timestamp"] = time.time()
            self.latest_state  = state
            parts = []
            for cid in sorted(self.cameras.keys()):
                s    = state.get(str(cid), {})
                icon = {"ok":"✓","warn":"~","far":"✗","searching":"?"}.get(s.get("status","?"),"?")
                parts.append(f"CAM{cid}:{icon} {s.get('deviation_cm','?')}cm")
            ready = "✓ BEREIT!" if self.all_ready else "Warte..."
            print(f"\r  {' | '.join(parts)} | {ready}    ", end="", flush=True)
            pass  # Q wird via CTRL+C beendet
        self.running = False
        for cam in self.cameras.values():
            cam.stop()
        pass  # cv2.destroyAllWindows deaktiviert
        print("\nSetup Detector beendet")

if __name__ == "__main__":
    import sys
    print("ANTHRO3D Setup Detector v2")
    print("==========================")
    print("\nModus waehlen:")
    print("  1 = COMPACT (Radius 175cm, Raum 4x4m)")
    print("  2 = PRO     (Radius 250cm, Raum 6x6m)")
    choice = input("\nEingabe (1/2): ").strip()
    mode   = "pro" if choice == "2" else "compact"
    detector = SetupDetector()
    detector.set_mode(mode)
    if not detector.connect_cameras():
        print("Fehler: Keine Kameras gefunden")
        sys.exit(1)
    detector.run()
