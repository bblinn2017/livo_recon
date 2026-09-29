#!/usr/bin/env python3
"""Expensive offline reference builder. Uses Open3D iterative RANSAC if available.
It is intentionally not a production backend and may retain all stationary points.

R52 AUDIT/REPAIR (documented): the originally supplied version called
Open3D's segment_plane() once per iteration and accepted the ENTIRE inlier
set as one surface. RANSAC plane-fitting has no notion of spatial
connectivity -- two physically disconnected, parallel walls sharing (within
tolerance) the same normal and offset are a single "plane" to RANSAC and
would be reported as ONE reference surface with an unbounded/incorrect
finite-support region spanning both. This directly violates this round's
own explicit requirement ("must represent finite spatial support... avoid
merging disconnected parallel/coplanar surfaces").

Fix: after each RANSAC inlier set is found, cluster the inliers spatially
(DBSCAN on the 3D inlier points, not on the whole cloud) and emit ONE
surface row per connected component above --min-points, instead of one row
for the whole inlier set. Each surface's finite support is reported as its
own axis-aligned bounding box (min/max) plus its own point count -- not the
plane's global infinite extent. A component that ends up smaller than
--min-points is returned to the working cloud (NOT discarded and NOT kept
as an under-supported surface) so a later, different-orientation RANSAC
pass can still claim those points if they belong to a real surface;
without this, small leftover fragments are irrecoverably lost.
"""
import argparse,csv,glob,os,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--distance',type=float,default=.02);p.add_argument('--min-points',type=int,default=100);p.add_argument('--cluster-eps',type=float,default=.15);p.add_argument('--cluster-min-samples',type=int,default=10);a=p.parse_args()
 try: import open3d as o3d
 except ImportError as e: raise SystemExit('open3d required for reference-map analysis') from e
 xs=[np.load(f)['xyz'] for f in sorted(glob.glob(os.path.join(a.input,'obs_*.npz')))];x=np.concatenate(xs) if xs else np.empty((0,3));pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x));rows=[];sid=0
 stall_guard=0
 while len(pc.points)>=a.min_points and stall_guard<10000:
  stall_guard+=1
  model,idx=pc.segment_plane(a.distance,3,1000)
  if len(idx)<a.min_points:break
  inlier_pts=np.asarray(pc.points)[idx]
  n=np.array(model[:3]);n/=np.linalg.norm(n)
  # R52 fix: split the RANSAC inlier set into spatially-connected
  # components before accepting any of them as a "surface" -- see module
  # docstring. Open3D's own DBSCAN (cluster_dbscan) run on JUST the
  # inlier points, not the full cloud.
  inlier_pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
  labels=np.asarray(inlier_pc.cluster_dbscan(eps=a.cluster_eps,min_points=a.cluster_min_samples))
  accepted_mask=np.zeros(len(idx),dtype=bool)
  for lbl in sorted(set(labels.tolist())-{-1}):
   comp_mask=labels==lbl
   if comp_mask.sum()<a.min_points: continue  # too small; leave in working cloud for a later pass
   comp_pts=inlier_pts[comp_mask]
   c=comp_pts.mean(0)
   bb_min=comp_pts.min(0); bb_max=comp_pts.max(0)
   rows.append([sid,*c,*n,float(model[3]),int(comp_mask.sum()),*bb_min,*bb_max])
   sid+=1
   accepted_mask|=comp_mask
  # remove only the ACCEPTED (large-enough, connected) inliers from the
  # working cloud; rejected/undersized components and all non-inliers
  # stay available for subsequent RANSAC iterations.
  accepted_global_idx=[idx[i] for i in range(len(idx)) if accepted_mask[i]]
  if not accepted_global_idx:
   # nothing accepted this iteration (every component too small) --
   # remove just the raw inlier set to avoid an infinite loop on the same
   # degenerate plane fit.
   pc=pc.select_by_index(idx,invert=True)
  else:
   pc=pc.select_by_index(accepted_global_idx,invert=True)
 with open(a.output,'w',newline='') as f:
  w=csv.writer(f);w.writerow(['surface_id','cx','cy','cz','nx','ny','nz','d','points','bb_min_x','bb_min_y','bb_min_z','bb_max_x','bb_max_y','bb_max_z']);w.writerows(rows)
if __name__=='__main__':main()
