#!/usr/bin/env python3
"""Compute GT-direction metrics from timestamped per-iteration tail logs."""
import argparse,csv,math

def num(r,k):
 v=r.get(k,""); return float(v) if v not in ("","nan","NaN",None) else float("nan")
def load_gt(path,tcol,cols):
 with open(path,newline='') as f: rows=[(float(r[tcol]),tuple(float(r[c]) for c in cols)) for r in csv.DictReader(f)]
 return sorted(rows)
def interp(gt,t,max_gap):
 if not gt or t<gt[0][0] or t>gt[-1][0]: return None
 lo,hi=0,len(gt)-1
 while lo<hi:
  m=(lo+hi)//2
  if gt[m][0]<t: lo=m+1
  else: hi=m
 if abs(gt[lo][0]-t)<=1e-10*max(1.0,abs(t)): return gt[lo][1],gt[lo][0],gt[lo][0],0.0,0.0
 j,i=lo,lo-1
 if i<0:return None
 t0,p0=gt[i];t1,p1=gt[j];gap=t1-t0
 if gap<=0 or gap>max_gap:return None
 a=(t-t0)/gap;p=tuple(p0[k]+a*(p1[k]-p0[k]) for k in range(3))
 return p,t0,t1,a,gap
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--iterations',required=True);ap.add_argument('--gt',required=True);ap.add_argument('--output',required=True);ap.add_argument('--time-col',default='t_abs_iter');ap.add_argument('--gt-time-col',default='t');ap.add_argument('--gt-x',default='x');ap.add_argument('--gt-y',default='y');ap.add_argument('--gt-z',default='z');ap.add_argument('--gt-max-gap',type=float,default=.1);a=ap.parse_args();gt=load_gt(a.gt,a.gt_time_col,(a.gt_x,a.gt_y,a.gt_z));extra=['gt_valid','gt_t_lo','gt_t_hi','gt_fraction','gt_gap','gt_x','gt_y','gt_z','gt_error_pre','gt_error_post','gt_error_reduction','gt_cosine','gt_parallel_correction','gt_perpendicular_correction']
 with open(a.iterations,newline='') as fi,open(a.output,'w',newline='') as fo:
  rd=csv.DictReader(fi);fields=list(rd.fieldnames or []);fields += [x for x in extra if x not in fields];wr=csv.DictWriter(fo,fieldnames=fields);wr.writeheader()
  for r in rd:
   try:
    t=num(r,a.time_col);b=[num(r,k) for k in ('tail_p_before_x','tail_p_before_y','tail_p_before_z')];p=[num(r,k) for k in ('tail_p_after_x','tail_p_after_y','tail_p_after_z')];q=interp(gt,t,a.gt_max_gap) if all(math.isfinite(x) for x in [t,*b,*p]) else None
    if q is None:r.update({k:('0' if k=='gt_valid' else '') for k in extra});wr.writerow(r);continue
    (g,t0,t1,f,gap)=q;d=[p[i]-b[i] for i in range(3)];to=[g[i]-b[i] for i in range(3)];dn=math.sqrt(sum(x*x for x in d));tn=math.sqrt(sum(x*x for x in to));post=math.sqrt(sum((p[i]-g[i])**2 for i in range(3)));cos=sum(d[i]*to[i] for i in range(3))/(dn*tn) if dn>1e-15 and tn>1e-15 else 0.0;u=[x/tn for x in to] if tn>1e-15 else [0,0,0];par=sum(d[i]*u[i] for i in range(3));per=math.sqrt(max(0.0,dn*dn-par*par));r.update({'gt_valid':'1','gt_t_lo':t0,'gt_t_hi':t1,'gt_fraction':f,'gt_gap':gap,'gt_x':g[0],'gt_y':g[1],'gt_z':g[2],'gt_error_pre':tn,'gt_error_post':post,'gt_error_reduction':tn-post,'gt_cosine':cos,'gt_parallel_correction':par,'gt_perpendicular_correction':per});wr.writerow(r)
   except (KeyError,ValueError,TypeError):r.update({k:('0' if k=='gt_valid' else '') for k in extra});wr.writerow(r)
if __name__=='__main__':main()
