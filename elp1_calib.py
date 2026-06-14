import cv2, numpy as np, yaml
from pathlib import Path

SQUARES_X = 9; SQUARES_Y = 6
SQUARE_SIZE = 0.020; MARKER_SIZE = 0.015

print("Initialisiere...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters())
print("✓ ChArUco Board initialisiert")

print("Öffne Kamera Index 3...")
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
ret, frame = cap.read()
if not ret:
    print("✗ Kamera nicht erreichbar!")
    exit()
print(f"✓ Kamera OK: {frame.shape[1]}x{frame.shape[0]}")

# Helligkeit prüfen
w = frame.shape[1]//2
fl = frame[:,:w]
brightness = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).mean()
print(f"  Helligkeit: {brightness:.1f} (min 50 empfohlen)")

for _ in range(10): cap.read()

print("\nELP1 Kalibrierung — ENTER=Frame aufnehmen | q=Kalibrieren & Beenden")
print("Board vor LINKE Kamerahälfte halten, verschiedene Positionen & Winkel")

all_corners_l=[]; all_ids_l=[]; all_corners_r=[]; all_ids_r=[]
img_size=None; count=0

while True:
    cmd = input(f"\n[{count} Frames] ENTER=aufnehmen | q=kalibrieren: ").strip().lower()
    if cmd == 'q': break

    for _ in range(3): cap.read()
    ret, frame = cap.read()
    if not ret: print("  ✗ Kein Frame"); continue

    w = frame.shape[1]//2
    fl = frame[:,:w]; fr = frame[:,w:]
    if img_size is None: img_size = (fl.shape[1], fl.shape[0])

    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    brightness_l = gl.mean()
    brightness_r = gr.mean()
    print(f"  Helligkeit L:{brightness_l:.1f} R:{brightness_r:.1f}")

    cl,il,_,_ = charuco_detector.detectBoard(gl)
    cr,ir,_,_ = charuco_detector.detectBoard(gr)

    n_l = len(cl) if cl is not None else 0
    n_r = len(cr) if cr is not None else 0
    print(f"  Ecken erkannt: L={n_l} R={n_r} (mind. 7 nötig)")

    ok_l = cl is not None and len(cl) > 6
    ok_r = cr is not None and len(cr) > 6

    if ok_l and ok_r:
        all_corners_l.append(cl); all_ids_l.append(il)
        all_corners_r.append(cr); all_ids_r.append(ir)
        count += 1
        print(f"  ✓ Frame {count} gespeichert")
    else:
        if not ok_l: print(f"  ✗ Linke Kamera: Board neu positionieren")
        if not ok_r: print(f"  ✗ Rechte Kamera: Board neu positionieren")

cap.release()

if count < 10:
    print(f"\nZu wenig Frames ({count}), mindestens 10 nötig")
else:
    print(f"\nKalibriere mit {count} Frames...")
    err_l,K_l,d_l,_,_=cv2.aruco.calibrateCameraCharuco(all_corners_l,all_ids_l,board,img_size,None,None)
    err_r,K_r,d_r,_,_=cv2.aruco.calibrateCameraCharuco(all_corners_r,all_ids_r,board,img_size,None,None)
    print(f"  L: {err_l:.4f}px  R: {err_r:.4f}px  fx_L={K_l[0,0]:.1f}")
    obj_pts_all=[]; img_pts_l=[]; img_pts_r=[]
    for cl,il,cr,ir in zip(all_corners_l,all_ids_l,all_corners_r,all_ids_r):
        common=np.intersect1d(il.flatten(),ir.flatten())
        if len(common)<4: continue
        op=[]; ipl=[]; ipr=[]
        for cid in common:
            idx_l=np.where(il.flatten()==cid)[0][0]
            idx_r=np.where(ir.flatten()==cid)[0][0]
            op.append(board.getChessboardCorners()[cid])
            ipl.append(cl[idx_l][0]); ipr.append(cr[idx_r][0])
        if len(op)>=4:
            obj_pts_all.append(np.array(op,dtype=np.float32))
            img_pts_l.append(np.array(ipl,dtype=np.float32))
            img_pts_r.append(np.array(ipr,dtype=np.float32))
    stereo_err,K_l,d_l,K_r,d_r,R,T,E,F=cv2.stereoCalibrate(
        obj_pts_all,img_pts_l,img_pts_r,K_l,d_l,K_r,d_r,img_size,
        flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"  Stereo: {stereo_err:.4f}px  Baseline: {abs(T[0,0])*100:.1f}cm")
    cfg={'R':R.tolist(),'T':T.tolist(),'camera_matrix_l':K_l.tolist(),'dist_l':d_l.tolist(),
         'camera_matrix_r':K_r.tolist(),'dist_r':d_r.tolist(),
         'baseline_cm':float(abs(T[0,0])*100),'calibrated_elps':{'ELP1':True},'calibrated_indices':[3]}
    with open(Path("~/anthro3d/stereo_config_elp1.yaml").expanduser(),'w') as f:
        yaml.dump(cfg,f)
    print("✓ stereo_config_elp1.yaml gespeichert")
