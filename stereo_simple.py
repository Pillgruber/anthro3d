import cv2, numpy as np, yaml, os
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard((9,6), 0.025, 0.018, aruco_dict)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
BASE = os.path.expanduser("~/anthro3d")
cap_l = cv2.VideoCapture(1)
cap_r = cv2.VideoCapture(2)
all_cl,all_il,all_cr,all_ir = [],[],[],[]
count = 0; cooldown = 0
print("Board nah vor beide Kameras — Q=Beenden")
while True:
    rl,fl=cap_l.read(); rr,fr=cap_r.read()
    if not rl or not rr: continue
    gl=cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY); gr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    cl,il,_=detector.detectMarkers(gl); cr,ir,_=detector.detectMarkers(gr)
    ok_l=ok_r=False; ccl=cil=ccr=cir=None
    if il is not None and len(il)>=4:
        ret,ccl,cil=cv2.aruco.interpolateCornersCharuco(cl,il,gl,board)
        if ret and ret>=4: ok_l=True
    if ir is not None and len(ir)>=4:
        ret,ccr,cir=cv2.aruco.interpolateCornersCharuco(cr,ir,gr,board)
        if ret and ret>=4: ok_r=True
    dl=fl.copy(); dr=fr.copy()
    if ok_l: cv2.aruco.drawDetectedCornersCharuco(dl,ccl,cil,(0,220,80))
    if ok_r: cv2.aruco.drawDetectedCornersCharuco(dr,ccr,cir,(0,220,80))
    if ok_l and ok_r and cooldown==0:
        all_cl.append(ccl);all_il.append(cil);all_cr.append(ccr);all_ir.append(cir)
        count+=1; cooldown=12
        print(f"\r  {count}/20", end="", flush=True)
        if count>=20: break
    if cooldown>0: cooldown-=1
    col=(0,220,80) if (ok_l and ok_r) else (80,80,220)
    cv2.putText(dl,f"{'OK' if ok_l else '--'} {count}/20",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
    cv2.putText(dr,f"{'OK' if ok_r else '--'} {count}/20",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
    cv2.imshow('Stereo',np.hstack([cv2.resize(dl,(640,360)),cv2.resize(dr,(640,360))]))
    if cv2.waitKey(1)&0xFF==ord('q'): break
cv2.destroyAllWindows(); cap_l.release(); cap_r.release()
if count>=5:
    print(f"\nKalibriere {count} Paare...")
    _,K_l,d_l,_,_=cv2.aruco.calibrateCameraCharuco(all_cl,all_il,board,(1280,720),None,None)
    _,K_r,d_r,_,_=cv2.aruco.calibrateCameraCharuco(all_cr,all_ir,board,(1280,720),None,None)
    print(f"fx_L={K_l[0,0]:.0f} fx_R={K_r[0,0]:.0f}")
    obj,ipl,ipr=[],[],[]
    for a,b,c,d in zip(all_cl,all_il,all_cr,all_ir):
        cm=np.intersect1d(b.flatten(),d.flatten())
        if len(cm)<4: continue
        obj.append(board.getChessboardCorners()[cm].astype(np.float32))
        ipl.append(a[np.isin(b.flatten(),cm)]); ipr.append(c[np.isin(d.flatten(),cm)])
    if len(obj)>=5:
        err,K_l,d_l,K_r,d_r,R,T,E,F=cv2.stereoCalibrate(obj,ipl,ipr,K_l,d_l,K_r,d_r,(1280,720),flags=cv2.CALIB_FIX_INTRINSIC)
        bl=float(np.linalg.norm(T))*1000
        print(f"Fehler:{err:.3f}px Baseline:{bl:.1f}mm")
        R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K_l,d_l,K_r,d_r,(1280,720),R,T)
        ml1,ml2=cv2.initUndistortRectifyMap(K_l,d_l,R1,P1,(1280,720),cv2.CV_32F)
        mr1,mr2=cv2.initUndistortRectifyMap(K_r,d_r,R2,P2,(1280,720),cv2.CV_32F)
        cfg={'K_left':K_l.tolist(),'dist_left':d_l.tolist(),'K_right':K_r.tolist(),'dist_right':d_r.tolist(),'R':R.tolist(),'T':T.tolist(),'Q':Q.tolist(),'baseline_mm':round(bl,1),'reprojection_error':round(err,4),'img_size':[1280,720]}
        with open(f'{BASE}/stereo_config.yaml','w') as f: yaml.dump(cfg,f,default_flow_style=False)
        np.savez(f'{BASE}/stereo_maps.npz',map_l1=ml1,map_l2=ml2,map_r1=mr1,map_r2=mr2)
        print("Gespeichert!")
