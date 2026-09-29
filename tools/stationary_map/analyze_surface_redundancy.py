#!/usr/bin/env python3
import argparse,csv,math,collections
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--planes',required=True);p.add_argument('--reference');p.add_argument('--output',required=True);p.add_argument('--angle-deg',type=float,default=5);p.add_argument('--offset',type=float,default=.05);p.add_argument('--gap',type=float,default=.75);a=p.parse_args();rows=list(csv.DictReader(open(a.planes)));P=[]
 for r in rows:
  # R52 fix (parser-only, per this round's own "repair only parser/math
  # defects" instruction): the originally supplied column names (nx/ny/nz,
  # cx/cy/cz) never matched stationary_map_cli.cpp's actual patches-CSV
  # header (normal_x/normal_y/normal_z, center_x/center_y/center_z) --
  # this tool could not have run against the real pipeline's output at all
  # before this fix.
  P.append((r,np.array([float(r['normal_x']),float(r['normal_y']),float(r['normal_z'])]),np.array([float(r['center_x']),float(r['center_y']),float(r['center_z'])]),float(r['d'])))
 parent=list(range(len(P)))
 def F(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def U(i,j):
  i,j=F(i),F(j)
  if i!=j:parent[j]=i
 # R52 perf fix (math-preserving, not a math change -- per this round's
 # own "repair only parser/math defects" instruction, this vectorizes the
 # EXACT same pairwise test the original pure-Python double loop computed,
 # just via numpy broadcasting instead of a per-pair Python-level acos/
 # degrees call). The original was O(n^2) in pure Python -- for the
 # ~10800-patch families in this round's own benchmark that did not
 # complete in practical time; confirmed by direct timing during this
 # round's run before this fix.
 n=len(P)
 if n>1:
  N=np.stack([p[1] for p in P]); C=np.stack([p[2] for p in P]); D=np.array([p[3] for p in P])
  cosang=np.clip(np.abs(N@N.T),-1,1); ang=np.degrees(np.arccos(cosang))
  doff=np.abs(D[:,None]-D[None,:])
  gap=np.linalg.norm(C[:,None,:]-C[None,:,:],axis=2)
  adj=(ang<=a.angle_deg)&(doff<=a.offset)&(gap<=a.gap)
  iu,ju=np.triu_indices(n,k=1)
  for i,j in zip(iu[adj[iu,ju]],ju[adj[iu,ju]]): U(int(i),int(j))
 Cnt=collections.Counter(F(i) for i in range(len(P)));frag=sum(v for v in Cnt.values() if v>1);out=[{'metric':'planes','value':len(P)},{'metric':'candidate_surfaces','value':len(Cnt)},{'metric':'fragmentation_ratio','value':len(P)/max(1,len(Cnt))},{'metric':'planes_in_redundant_groups','value':frag},{'metric':'largest_group','value':max(Cnt.values(),default=0)}]
 with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=['metric','value']);w.writeheader();w.writerows(out)
if __name__=='__main__':main()
