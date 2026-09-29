#!/usr/bin/env python3
"""Read a ROS bag directly and cache stationary LiDAR observations through --end-timestamp.
Output is deliberately backend-neutral: manifest.csv plus one compressed npz per scan.
The coding agent must determine --end-timestamp from Leica GT and record that evidence.
"""
import argparse,csv,hashlib,json,os
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--bag',required=True);p.add_argument('--lidar-topic',required=True);p.add_argument('--end-timestamp',type=float,required=True);p.add_argument('--output',required=True);p.add_argument('--calibration-end-timestamp',type=float,default=None);a=p.parse_args()
 try: import rosbag
 except ImportError as e: raise SystemExit('rosbag Python module required in coding-agent ROS environment') from e
 try:
  from sensor_msgs import point_cloud2
 except ImportError as e: raise SystemExit('sensor_msgs.point_cloud2 required') from e
 os.makedirs(a.output,exist_ok=True); rows=[]; h=hashlib.sha256(); obs=0
 with rosbag.Bag(a.bag) as bag:
  for _,msg,t in bag.read_messages(topics=[a.lidar_topic]):
   ts=t.to_sec()
   if ts>a.end_timestamp: break
   pts=[]
   for q in point_cloud2.read_points(msg,skip_nans=True): pts.append((float(q[0]),float(q[1]),float(q[2])))
   arr=np.asarray(pts,dtype=np.float64); fn=f'obs_{obs:06d}.npz'; np.savez_compressed(os.path.join(a.output,fn),xyz=arr,timestamp=ts)
   digest=hashlib.sha256(arr.tobytes()).hexdigest();h.update(arr.tobytes()); rows.append([obs,ts,len(arr),fn,digest,int(a.calibration_end_timestamp is not None and ts<=a.calibration_end_timestamp)]);obs+=1
 with open(os.path.join(a.output,'manifest.csv'),'w',newline='') as f:
  w=csv.writer(f);w.writerow(['observation_id','timestamp','raw_points','file','point_sha256','in_calibration']);w.writerows(rows)
 json.dump({'bag':os.path.abspath(a.bag),'lidar_topic':a.lidar_topic,'end_timestamp':a.end_timestamp,'calibration_end_timestamp':a.calibration_end_timestamp,'observations':obs,'stream_sha256':h.hexdigest()},open(os.path.join(a.output,'metadata.json'),'w'),indent=2)
if __name__=='__main__':main()
