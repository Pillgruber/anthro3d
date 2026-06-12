import cv2, numpy as np, yaml, time
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board      = cv2.aruco.CharucoBoard((9,6), 0.025, 0.018, aruco_dict)
detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
ELP_INDEX = 0
ELP_NAME  = "ELP2"
TARGET = 50
PROGRESS_FILE = "/tmp/stereo_progress.txt"
cap = cv2.VideoCapture(ELP_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
all_cl,all_il,all_cr,all_ir = [],[],[],[]
count = 0
print("ChArUco Stereo " + ELP_NAME + " (Index " + str(ELP_INDEX) + ") — Ziel: " + str(TARGET) + " Paare | Q=Beenden")
while True:
    ret, frame = cap.read()
    if not ret or frame is None: continue
    w = frame.shape[1] // 2
    fl = frame[:, :w].copy()
    fr = frame[:, w:].copy()
    gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cl,il,_ = detector.detectMarkers(gl); cr,ir,_ = detector.detectMarkers(gr)
    ok_l=ok_r=False
    if il is not None and len(il)>=4:
        ret2,ccl,cil = cv2.aruco.interpolateCornersCharuco(cl,il,gl,board)
        if ret2 and ret2>=4: cv2.aruco.drawDetectedCornersCharuco(fl,ccl,cil,(0,255,0)); ok_l=True
    if ir is not None and len(ir)>=4:
        ret2,ccr,cir = cv2.aruco.interpolateCornersCharuco(cr,ir,gr,board)
        if ret2 and ret2>=4: cv2.aruco.drawDetectedCornersCharuco(fr,ccr,cir,(0,255,0)); ok_r=True
    if ok_l and ok_r:
        all_cl.append(ccl);all_il.append(cil);all_cr.append(ccr);all_ir.append(cir)
        count+=1
        open(PROGRESS_FILE,"w").write(str(count))
        time.sleep(0.4)
        cv2.putText(fl,"OK "+str(count)+"/"+str(TARGET),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(fl,"L:"+("OK" if ok_l else "--")+" R:"+("OK" if ok_r else "--")+" "+str(count)+"/"+str(TARGET),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,255),2)
    cv2.imshow(ELP_NAME + " Stereo Kalib — Q=Beenden",np.hstack([cv2.resize(fl,(640,400)),cv2.resize(fr,(640,400))]))
    key=cv2.waitKey(1)&0xFF
    if key in (ord('q'),27): break
    if count>=TARGET: break
cv2.destroyAllWindows(); cap.release()
if count>=5:
    print("Kalibriere mit " + str(count) + " Paaren...")
    err_l,K_l,d_l,_,_ = cv2.aruco.calibrateCameraCharuco(all_cl,all_il,board,(1600,1200),None,None)
    err_r,K_r,d_r,_,_ = cv2.aruco.calibrateCameraCharuco(all_cr,all_ir,board,(1600,1200),None,None)
