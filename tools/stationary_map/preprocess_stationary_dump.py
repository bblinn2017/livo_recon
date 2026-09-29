#!/usr/bin/env python3
"""Deterministic preprocessing caches, reproducing production's exact
cbk/lidar point-ingestion semantics (audited directly against
src/utils/lidar/lidar.cpp's buildLidarPoints() and src/processing/
cbk_processing.cpp's loadParameters(), not inferred from the fitted
behavior):

  1. point_filter_num decimation: keep point i iff i % pfn == 0 (matches
     `if (point_filter_num > 1 && (i % point_filter_num) != 0) continue;`
     exactly -- Python's x[::pfn] stride is index-identical).
  2. finite-position filter (`std::isfinite` on x/y/z) -- defensive,
     matches production, essentially always a no-op on real LiDAR data.
  3. blind-radius filter: discard any point with squaredNorm() <
     blind_sqr. Production's blind_sqr defaults to 0.5m^2 = 0.25 (radius
     0.5m, cbk/lidar/blind's own default -- confirmed in
     cbk_processing.cpp: `paramWarn<double>(pnh, "cbk/lidar/blind", blind,
     0.5); opts_.blind_sqr = blind * blind;`). Order relative to PFN does
     not matter mathematically (both are independent survivorship tests
     on the SAME raw index, so the intersection is order-invariant) --
     applied here after PFN purely for a smaller intermediate array.
  4. voxel-downsample: FIRST-mode only (matches PointXYZCovKeyFn's
     `floor(p/voxel_size)` key exactly; production's AVERAGE mode is a
     separate, non-default option not exercised by this round's 4
     canonical/dsoff x PFN1/PFN3 cache combinations).

R53 REPAIR: production's bootstrap path (src/processing/calib_processing.cpp:
508-519) converts every retained point with state_->lidarToImu(p) = R_l2i*p + t_l2i
BEFORE the voxel downsample, so voxel keys are formed in the IMU frame. R52 down-
sampled in the raw sensor frame; with t_l2i = (-0.050, 0, 0.055) (ntu_viral.yaml
extrinsics) the FIRST-mode voxel boundaries, and therefore which points survive,
differ. This script now applies --lidar-to-imu-R/--lidar-to-imu-t after the PFN/
finite/blind filters and before the downsample, and the cache is in the IMU frame.
The reference map must be built from a cache in this same frame.

R52 REPAIR (documented, from the original supplied script): the supplied
version omitted the blind-radius filter entirely (a genuine production/
harness semantic mismatch, not merely an omission of an optional feature
-- blind=0.5 is production's own non-zero DEFAULT, always active unless a
config explicitly overrides it to 0, which no config in this repository
does). Added here. The isfinite check was also missing; added
defensively (should be a no-op on real data, included for exact parity).

`measures_->isValid(timestamp)` (production's fourth survivorship
condition) is NOT replicated -- it gates on live estimator/queue
synchronization state, not a geometric property of the point cloud, and
has no meaning for an offline cached batch dump.
"""
import argparse,csv,hashlib,json,os,numpy as np

BLIND_SQ_DEFAULT = 0.25  # cbk/lidar/blind = 0.5m (production default), squared

def voxel_first(x,leaf):
 if leaf<=0:return x
 seen={};out=[]
 for p in x:
  k=tuple(np.floor(p/leaf).astype(np.int64));
  if k not in seen:seen[k]=1;out.append(p)
 return np.asarray(out)

def apply_production_filters(x,pfn,blind_sq):
 x=x[::max(1,pfn)]
 if len(x)==0:return x
 finite=np.isfinite(x).all(axis=1)
 x=x[finite]
 if len(x)==0:return x
 sqn=(x*x).sum(axis=1)
 return x[sqn>=blind_sq]

def to_imu(x,R,t):
 return x@np.asarray(R,dtype=np.float64).reshape(3,3).T+np.asarray(t,dtype=np.float64)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--pfn',type=int,required=True);p.add_argument('--ds',type=float,default=0);p.add_argument('--blind-sq',type=float,default=BLIND_SQ_DEFAULT);p.add_argument('--lidar-to-imu-R',type=float,nargs=9,default=[1,0,0,0,1,0,0,0,1]);p.add_argument('--lidar-to-imu-t',type=float,nargs=3,default=[0,0,0]);a=p.parse_args();os.makedirs(a.output,exist_ok=True)
 rows=[];H=hashlib.sha256();src=list(csv.DictReader(open(os.path.join(a.input,'manifest.csv'))))
 for r in src:
  x=np.load(os.path.join(a.input,r['file']))['xyz']
  x=apply_production_filters(x,a.pfn,a.blind_sq)
  if len(x):x=to_imu(x,a.lidar_to_imu_R,a.lidar_to_imu_t)
  x=voxel_first(x,a.ds);fn=r['file'];np.savez_compressed(os.path.join(a.output,fn),xyz=x,timestamp=float(r['timestamp']))
  # R52: also emit a trivial flat-binary sidecar (int64 N, then N*3
  # float64 xyz) alongside the .npz -- the C++ benchmark driver
  # (tools/stationary_map/stationary_map_cli.cpp) reads this format
  # directly rather than linking an npz/zip parser into the production
  # build for a research-only harness. Deliberate, documented
  # simplification, not a second source of truth: both files are written
  # from the exact same `x` array in the same call.
  bin_fn=fn.replace('.npz','.bin')
  with open(os.path.join(a.output,bin_fn),'wb') as bf:
   np.array([len(x)],dtype=np.int64).tofile(bf); x.astype(np.float64).tofile(bf)
  d=hashlib.sha256(x.tobytes()).hexdigest();H.update(x.tobytes());rows.append([r['observation_id'],r['timestamp'],len(x),fn,d,r['in_calibration']])
 with open(os.path.join(a.output,'manifest.csv'),'w',newline='') as f:w=csv.writer(f);w.writerow(['observation_id','timestamp','points','file','point_sha256','in_calibration']);w.writerows(rows)
 json.dump({'pfn':a.pfn,'ds':a.ds,'blind_sq':a.blind_sq,'lidar_to_imu_R':a.lidar_to_imu_R,'lidar_to_imu_t':a.lidar_to_imu_t,'frame':'imu','stream_sha256':H.hexdigest()},open(os.path.join(a.output,'metadata.json'),'w'),indent=2)
if __name__=='__main__':main()
