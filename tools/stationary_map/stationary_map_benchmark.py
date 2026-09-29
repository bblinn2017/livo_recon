#!/usr/bin/env python3
"""Portable reference benchmark for cached observations.
The C++ stationary-map library is the production-speed implementation; this script provides an auditable oracle/output schema."""
import argparse,csv,glob,os,time,numpy as np

def fit(x):
 if len(x)<3:return None
 c=x.mean(0);C=(x-c).T@(x-c)/len(x);v,U=np.linalg.eigh(C);n=U[:,0];return c,n,-n@c,v

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--leaf',type=float,default=.25);a=p.parse_args();cells={};t0=time.time()
 for f in sorted(glob.glob(os.path.join(a.input,'obs_*.npz'))):
  for q in np.load(f)['xyz']:
   k=tuple(np.floor(q/a.leaf).astype(int));cells.setdefault(k,[]).append(q)
 with open(a.output,'w',newline='') as g:
  w=csv.writer(g);w.writerow(['patch_id','cx','cy','cz','nx','ny','nz','d','lambda0','lambda1','lambda2','points']);pid=0
  for _,v in sorted(cells.items()):
   z=fit(np.asarray(v));
   if z is None:continue
   c,n,d,e=z;w.writerow([pid,*c,*n,d,*e,len(v)]);pid+=1
 print('patches',pid,'seconds',time.time()-t0)
if __name__=='__main__':main()
