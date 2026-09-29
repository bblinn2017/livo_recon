#!/usr/bin/env python3
import argparse,csv,math,collections
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--planes',required=True);p.add_argument('--reference');p.add_argument('--output',required=True);p.add_argument('--angle-deg',type=float,default=5);p.add_argument('--offset',type=float,default=.05);p.add_argument('--gap',type=float,default=.75);a=p.parse_args();rows=list(csv.DictReader(open(a.planes)));P=[]
 for r in rows:
  P.append((r,np.array([float(r['nx']),float(r['ny']),float(r['nz'])]),np.array([float(r['cx']),float(r['cy']),float(r['cz'])]),float(r['d'])))
 parent=list(range(len(P)))
 def F(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def U(i,j):
  i,j=F(i),F(j)
  if i!=j:parent[j]=i
 for i in range(len(P)):
  for j in range(i+1,len(P)):
   _,ni,ci,di=P[i];_,nj,cj,dj=P[j];ang=math.degrees(math.acos(min(1,max(-1,abs(float(ni@nj))))))
   if ang<=a.angle_deg and abs(di-dj)<=a.offset and np.linalg.norm(ci-cj)<=a.gap:U(i,j)
 C=collections.Counter(F(i) for i in range(len(P)));frag=sum(v for v in C.values() if v>1);out=[{'metric':'planes','value':len(P)},{'metric':'candidate_surfaces','value':len(C)},{'metric':'fragmentation_ratio','value':len(P)/max(1,len(C))},{'metric':'planes_in_redundant_groups','value':frag},{'metric':'largest_group','value':max(C.values(),default=0)}]
 with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=['metric','value']);w.writeheader();w.writerows(out)
if __name__=='__main__':main()
