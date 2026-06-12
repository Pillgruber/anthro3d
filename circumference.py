#!/usr/bin/env python3
import numpy as np
from pathlib import Path
try: from scipy.spatial import ConvexHull
except: ConvexHull=None

class CircumferenceCalculator:
    def __init__(self): self.thickness=3.0; self.alpha=0.3; self._last={}
    def compute_from_pointclouds(self,pointclouds,landmarks_3d=None):
        if not pointclouds or ConvexHull is None: return {}
        all_pts=np.vstack(list(pointclouds.values()))
        y=all_pts[:,1]; bot=np.percentile(y,5); top=np.percentile(y,95); h=top-bot
        if h<50: return {}
        keys={"Brustumfang":(0.77,(11,12)),"Taillenumfang":(0.62,(23,24)),"Hueftumfang":(0.52,(23,24)),"Oberschenkel_L":(0.42,(23,25)),"Oberschenkel_R":(0.42,(24,26)),"Knieumfang_L":(0.28,(25,27)),"Knieumfang_R":(0.28,(26,28))}
        res={}
        for name,(ratio,pair) in keys.items():
            ht=bot+h*ratio
            if landmarks_3d:
                a,b=pair
                pa=landmarks_3d.get(a); pb=landmarks_3d.get(b)
                if pa and pb: ht=(pa[1]+pb[1])/2
                if "Taillen" in name: ht+=5
                if "Hueft" in name: ht-=8
            c=self._circ(all_pts,ht)
            if c:
                if name in self._last: c=self.alpha*c+(1-self.alpha)*self._last[name]
                self._last[name]=c; res[name]=round(c,1)
        return res
    def _circ(self,pts,h):
        mask=np.abs(pts[:,1]-h)<self.thickness; sp=pts[mask]
        if len(sp)<6: return None
        xz=sp[:,[0,2]]; xc=xz[:,0].mean()
        left=xz[xz[:,0]<xc]; right=xz[xz[:,0]>=xc]
        if len(right)<3 and len(left)>=3:
            m=left.copy(); m[:,0]=2*xc-m[:,0]; xz=np.vstack([xz,m])
        try:
            hull=ConvexHull(xz); hp=xz[hull.vertices]; hp=np.vstack([hp,hp[0]])
            d=np.diff(hp,axis=0); return float(np.sum(np.sqrt((d**2).sum(axis=1))))
        except: return None
