#!/usr/bin/env python3
"""
ANTHRO3D — 3D Mesh Rekonstruktion
Kombiniert Tiefenkarten aller Kameras zu einem geschlossenen 3D Mesh.

Installation:
    pip install open3d --break-system-packages

Verwendung:
    python3 mesh_reconstruction.py

Tasten:
    SPACE  = Mesh aufnehmen und speichern
    Q      = Beenden
    V      = Vorschau in Open3D Viewer
"""
import cv2, numpy as np, yaml, os, time
try:
    import open3d as o3d
    OPEN3D = True
except ImportError:
    print("Open3D nicht installiert — pip install open3d --break-system-packages")
    OPEN3D = False

BASE = os.path.expanduser("~/anthro3d")

# ── Kamera-Konfiguration ──────────────────────────────────────────────────────
# Wird aus config.yaml + cam_positions.yaml geladen
# ELP Stereo = Index der ELP Kamera (3200x1200)
# OV9281_*   = Indizes der OV9281 Kameras

def load_camera_config():
    cams = {}
    try:
        with open(os.path.join(BASE, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        for cam in cfg['cameras']['tracking']:
            cams[cam['name']] = cam['device_index']
    except:
        # Fallback
        cams = {'Stereo R': 0, 'Seite L': 1, 'Seite R': 2, 'Hinten': 3}
    return cams

def load_stereo_config():
    try:
        with open(os.path.join(BASE, "stereo_config.yaml")) as f:
            return yaml.safe_load(f)
    except:
        return None

def load_cam_positions():
    try:
        with open(os.path.join(BASE, "cam_positions.yaml")) as f:
            return yaml.safe_load(f)
    except:
        return {}

# ── Stereo Tiefenkarte ────────────────────────────────────────────────────────
def make_sgbm():
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=7,
        P1=8*3*49, P2=32*3*49,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    wls    = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
    sgbm_r = cv2.ximgproc.createRightMatcher(sgbm)
    wls.setLambda(8000); wls.setSigmaColor(1.5)
    return sgbm, sgbm_r, wls

SGBM_FRONT = make_sgbm()
SGBM_BACK  = make_sgbm()
SGBM_SIDE  = make_sgbm()

def compute_depth(fl, fr, sgbm_tuple, K, baseline_mm):
    """Tiefenkarte aus Stereopaar."""
    sgbm, sgbm_r, wls = sgbm_tuple
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gl = clahe.apply(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY))
    gr = clahe.apply(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    disp_l = sgbm.compute(gl, gr)
    disp_r = sgbm_r.compute(gr, gl)
    disp_f = wls.filter(disp_l, gl, disparity_map_right=disp_r)
    disp   = np.clip(disp_f.astype(np.float32)/16.0, 0, None)
    depth  = np.zeros_like(disp)
    valid  = disp > 1
    if K is not None and baseline_mm:
        depth[valid] = (K[0,0] * baseline_mm/10) / disp[valid]
        depth = np.clip(depth, 10, 400)
    return depth

# ── Punktwolke aus Tiefenkarte ────────────────────────────────────────────────
def depth_to_pointcloud(color_frame, depth_cm, K, step=3):
    """Konvertiert Tiefenkarte in Open3D Punktwolke mit Farbe."""
    if not OPEN3D: return None
    H, W = depth_cm.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    pts, cols = [], []
    for y in range(0, H, step):
        for x in range(0, W, step):
            d = depth_cm[y, x]
            if d < 5 or d > 350: continue
            X = (x - cx) * d / fx
            Y = (y - cy) * d / fy
            Z = d
            pts.append([X, Y, Z])
            b, g, r = color_frame[y, x]
            cols.append([r/255, g/255, b/255])

    if not pts: return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
    return pcd

# ── ICP Ausrichtung ───────────────────────────────────────────────────────────
def align_pointclouds(pcds, cam_positions):
    """Richtet alle Punktwolken mit ICP zueinander aus."""
    if not OPEN3D or not pcds: return pcds
    if len(pcds) == 1: return pcds

    # Erste Kamera = Referenz
    result = [pcds[0]]
    for i, pcd in enumerate(pcds[1:], 1):
        # Grobe Ausrichtung aus cam_positions.yaml
        T_init = np.eye(4)
        cam_name = list(cam_positions.keys())[i] if i < len(cam_positions) else None
        if cam_name and cam_name in cam_positions:
            pos = cam_positions[cam_name].get('position_m', [0,0,0])
            T_init[:3, 3] = [p*100 for p in pos]  # m → cm

        # ICP Feinausrichtung
        threshold = 5.0  # 5cm
        reg = o3d.pipelines.registration.registration_icp(
            pcd, pcds[0], threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

        pcd_aligned = pcd.transform(reg.transformation)
        result.append(pcd_aligned)
        print(f"  ICP Kamera {i}: fitness={reg.fitness:.3f}")

    return result

# ── Mesh Rekonstruktion ───────────────────────────────────────────────────────
def reconstruct_mesh(pcds):
    """Erstellt geschlossenes Mesh aus Punktwolken."""
    if not OPEN3D: return None

    # Alle Punktwolken zusammenführen
    combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined += pcd

    print(f"  Punkte gesamt: {len(combined.points)}")

    # Rauschen entfernen
    combined, _ = combined.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)

    # Normalen berechnen (nötig für Poisson)
    combined.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=30))
    combined.orient_normals_towards_camera_location(
        camera_location=np.array([0, 0, -100]))

    # Poisson Surface Reconstruction
    print("  Poisson Rekonstruktion...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined, depth=8, width=0, scale=1.1, linear_fit=False)

    # Niedrige Dichte-Bereiche entfernen (Ränder/Artefakte)
    densities_np = np.asarray(densities)
    thresh = np.percentile(densities_np, 5)
    verts_to_remove = densities_np < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)

    # Vereinfachen
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=50000)

    # Glätten
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    mesh.compute_vertex_normals()

    print(f"  Dreiecke: {len(mesh.triangles)}")
    return mesh

# ── Speichern ────────────────────────────────────────────────────────────────
def save_mesh(mesh, suffix=""):
    """Speichert Mesh als .obj und .ply."""
    if not OPEN3D or mesh is None: return
    ts = int(time.time())
    obj_path = os.path.join(BASE, f"mesh_{ts}{suffix}.obj")
    ply_path = os.path.join(BASE, f"mesh_{ts}{suffix}.ply")
    o3d.io.write_triangle_mesh(obj_path, mesh)
    o3d.io.write_triangle_mesh(ply_path, mesh)
    size_kb = os.path.getsize(obj_path) // 1024
    print(f"  Gespeichert: {obj_path} ({size_kb}KB)")
    print(f"  Gespeichert: {ply_path}")
    return obj_path

# ── Hauptprogramm ─────────────────────────────────────────────────────────────
def main():
    if not OPEN3D:
        print("Bitte installieren: pip install open3d --break-system-packages")
        return

    cam_config   = load_camera_config()
    stereo_cfg   = load_stereo_config()
    cam_positions = load_cam_positions().get('cameras', {})

    # Kamera-Intrinsics
    K_front = np.array([[800,0,800],[0,800,600],[0,0,1]], dtype=np.float64)
    K_side  = np.array([[800,0,640],[0,800,400],[0,0,1]], dtype=np.float64)
    baseline_front = 65.0
    baseline_side  = 60.0

    if stereo_cfg:
        K_front = np.array(stereo_cfg['K_left'])
        baseline_front = stereo_cfg.get('baseline_mm', 65)

    # Kameras öffnen
    caps = {}
    print("Öffne Kameras...")
    for name, idx in cam_config.items():
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if 'ELP' in name:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
            caps[name] = cap
            print(f"  ✓ {name} (Index {idx})")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    snap_count = 0

    print("\nSPACE=Mesh aufnehmen | V=Vorschau | Q=Beenden\n")

    while True:
        frames = {}

        # ELP Stereo lesen
        if 'Stereo R' in caps:
            ret, frame = caps['Stereo R'].read()
            if ret:
                w = frame.shape[1]
                frames['Stereo R'] = frame

        # OV9281 lesen
        for name in ['Seite L', 'Seite R', 'Hinten']:
            if name in caps:
                ret, frame = caps[name].read()
                if ret: frames[name] = frame

        # Vorschau
        previews = []
        for name, frame in list(frames.items())[:4]:
            vis = cv2.resize(frame, (320, 200))
            cv2.putText(vis, name, (5,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            previews.append(vis)
        while len(previews) < 4:
            previews.append(np.zeros((200,320,3), dtype=np.uint8))

        top = np.hstack(previews[:2])
        bot = np.hstack(previews[2:4])
        grid = np.vstack([top, bot])
        cv2.putText(grid, "SPACE=Mesh | Q=Beenden", (10,390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("ANTHRO3D Mesh", grid)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            print(f"\nAufnahme {snap_count+1}...")
            pcds = []

            # ELP Tiefenkarte
            if 'ELP_L' in frames and 'ELP_R' in frames:
                depth = compute_depth(
                    frames.get('Stereo R'), frames.get('Stereo L'),
                    SGBM_FRONT, K_front, baseline_front)
                pcd = depth_to_pointcloud(frames.get('Stereo R'), depth, K_front)
                if pcd: pcds.append(pcd)
                print(f"  ELP: {len(pcd.points) if pcd else 0} Punkte")

            # OV9281 Seite L + R als Stereopaar
            if 'Seite L' in frames and 'Seite R' in frames:
                depth = compute_depth(
                    frames['Seite L'], frames['Seite R'],
                    SGBM_SIDE, K_side, baseline_side)
                pcd = depth_to_pointcloud(frames['Seite L'], depth, K_side)
                if pcd: pcds.append(pcd)
                print(f"  Seite: {len(pcd.points) if pcd else 0} Punkte")

            if not pcds:
                print("  Keine Punktwolken — Kameras prüfen")
                continue

            # ICP Ausrichtung
            print("  ICP Ausrichtung...")
            pcds_aligned = align_pointclouds(pcds, cam_positions)

            # Mesh Rekonstruktion
            mesh = reconstruct_mesh(pcds_aligned)

            if mesh:
                path = save_mesh(mesh, f"_{snap_count:03d}")
                snap_count += 1

        elif key == ord('v') and OPEN3D:
            # Open3D Viewer
            print("Open3D Viewer — Q zum Schließen")
            if 'ELP_L' in frames and 'ELP_R' in frames:
                depth = compute_depth(
                    frames.get('Stereo R'), frames.get('Stereo L'),
                    SGBM_FRONT, K_front, baseline_front)
                pcd = depth_to_pointcloud(frames.get('Stereo R'), depth, K_front, step=2)
                if pcd:
                    o3d.visualization.draw_geometries(
                        [pcd],
                        window_name="ANTHRO3D Punktwolke",
                        width=1280, height=720)

    cv2.destroyAllWindows()
    for cap in caps.values(): cap.release()

if __name__ == "__main__":
    main()
