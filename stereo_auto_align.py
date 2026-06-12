import cv2, numpy as np, yaml, os

BASE = os.path.expanduser("~/anthro3d")
CAM_L, CAM_R = 1, 2

cap_l = cv2.VideoCapture(CAM_L)
cap_r = cv2.VideoCapture(CAM_R)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

orb = cv2.ORB_create(nfeatures=2000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8*3*49, P2=32*3*49,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
sgbm_r = cv2.ximgproc.createRightMatcher(sgbm)
wls.setLambda(8000); wls.setSigmaColor(1.5)

H_warp = None  # Homographie für Rektifizierung
frame_count = 0

print("Auto-Rektifizierung — Q=Beenden")

while True:
    rl,fl=cap_l.read(); rr,fr=cap_r.read()
    if not rl or not rr: continue

    gl = clahe.apply(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY))
    gr = clahe.apply(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))

    # Alle 30 Frames Homographie neu berechnen
    if frame_count % 30 == 0:
        kp_l, des_l = orb.detectAndCompute(gl, None)
        kp_r, des_r = orb.detectAndCompute(gr, None)
        if des_l is not None and des_r is not None and len(des_l)>20 and len(des_r)>20:
            matches = bf.match(des_l, des_r)
            matches = sorted(matches, key=lambda x: x.distance)[:200]
            if len(matches) > 20:
                pts_l = np.float32([kp_l[m.queryIdx].pt for m in matches])
                pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches])
                H_warp, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 3.0)
                inliers = int(mask.sum()) if mask is not None else 0
                print(f"\r  Matches:{len(matches)} Inliers:{inliers}", end="", flush=True)

    frame_count += 1

    # Rektifizierung anwenden
    if H_warp is not None:
        fr_rect = cv2.warpPerspective(fr, H_warp, (fr.shape[1], fr.shape[0]))
        gr_rect = cv2.warpPerspective(gr.reshape(gr.shape[0],gr.shape[1],1),
                                       H_warp, (gr.shape[1], gr.shape[0])).squeeze()
    else:
        fr_rect = fr; gr_rect = gr

    # Tiefenkarte
    disp_l = sgbm.compute(gl, gr_rect)
    disp_r = sgbm_r.compute(gr_rect, gl)
    disp_f = wls.filter(disp_l, gl, disparity_map_right=disp_r)
    disp_show = np.clip(disp_f.astype(np.float32)/16.0, 0, None)

    disp_norm = cv2.normalize(disp_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    coverage = int(np.sum(disp_show>1)*100//(disp_show.shape[0]*disp_show.shape[1]))
    cv2.putText(depth_color, f"Auto-Rect | {coverage}% Abdeckung", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    side = np.hstack([cv2.resize(fl,(426,240)),
                      cv2.resize(fr_rect,(426,240)),
                      cv2.resize(depth_color,(428,240))])
    depth_big = cv2.resize(depth_color,(1280,480))
    cv2.imshow("Auto Stereo Tiefe", np.vstack([side, depth_big]))
    if cv2.waitKey(1)&0xFF==ord('q'): break

cv2.destroyAllWindows()
cap_l.release(); cap_r.release()
