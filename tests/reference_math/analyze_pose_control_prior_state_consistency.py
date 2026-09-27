#!/usr/bin/env python3
"""Offline audit of first-frame EKF/spline covariance consistency.

This consumes production first-frame dumps; it does not run or change the estimator.
It reports: eigenspectra/rank, EKF vs spline conditional/marginal covariance,
position->velocity covariance gains, state/spline gaps, and exact reconstruction
of the logged continuous-IMU information from per-sample J/W dumps.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np

def parse_matrices(path):
    text = Path(path).read_text()
    out = []
    pos = 0
    pat = re.compile(r"matrix (\S+) rows=(\d+) cols=(\d+)\n")
    while True:
        m = pat.search(text, pos)
        if not m: break
        name, rows, cols = m.group(1), int(m.group(2)), int(m.group(3))
        start=m.end()
        n=rows*cols
        vals=[]
        # matrix rows are plain whitespace-separated numbers; consume until next labeled block
        endm=re.search(r"\n(?:matrix |vector |=== end_prior_audit ===|=== prior_audit ===)", text[start:])
        end=start+(endm.start() if endm else len(text)-start)
        vals=np.fromstring(text[start:end],sep=' ')
        if vals.size>=n:
            vals=vals[:n].reshape(rows,cols)
            out.append((name,vals))
        pos=end
    return out

def parse_snapshots(path):
    text=Path(path).read_text()
    snaps=[]
    starts=[m.start() for m in re.finditer(r"=== prior_audit ===",text)]
    for i,st in enumerate(starts):
        en=starts[i+1] if i+1<len(starts) else len(text)
        block=text[st:en]
        head={}
        for k,v in re.findall(r"^(architecture|scan_id|iteration|t_abs|var_acc|var_gyr|imu_samples)=(.+)$",block,re.M):
            head[k]=v
        mats={k:v for k,v in parse_matrices_from_text(block)}
        snaps.append((head,mats,block))
    return snaps

def parse_matrices_from_text(text):
    out=[]; pos=0
    pat=re.compile(r"matrix (\S+) rows=(\d+) cols=(\d+)\n")
    while True:
        m=pat.search(text,pos)
        if not m: break
        name,rows,cols=m.group(1),int(m.group(2)),int(m.group(3)); start=m.end()
        nxt=re.search(r"\n(?:matrix |vector |eigenvalues_|q_density_|tail_state_|=== end_prior_audit ===)",text[start:])
        end=start+(nxt.start() if nxt else len(text)-start)
        vals=np.fromstring(text[start:end],sep=' ')
        n=rows*cols
        if vals.size>=n: out.append((name,vals[:n].reshape(rows,cols)))
        pos=end
    return out

def eigstats(M,tol=1e-12):
    if M.size==0 or M.shape[0]!=M.shape[1]: return {}
    d=np.linalg.eigvalsh((M+M.T)/2)
    th=tol*max(abs(d[-1]),1.0)
    return {'min':float(d[0]),'max':float(d[-1]),'rank':int(np.sum(d>th)),'eigenvalues':d.tolist(),'threshold':th}

def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prior-audit',required=True)
    ap.add_argument('--state-chain',default='')
    ap.add_argument('--solve',default='')
    ap.add_argument('--knot-map',default='')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    snaps=parse_snapshots(args.prior_audit)
    result={'num_prior_audit_snapshots':len(snaps),'snapshots':[]}
    for head,mats,block in snaps:
        item=dict(head)
        item['spectra']={k:eigstats(mats[k]) for k in ['A_process_raw','A_ff_prior','Lambda_full_prior','P_z_marginal','P_z_conditional_given_head'] if k in mats}
        if 'P_EKF_RPV_after_IMU' in mats and 'P_tail_conditional_RPV' in mats:
            pe=mats['P_EKF_RPV_after_IMU']; pc=mats['P_tail_conditional_RPV']; pm=mats.get('P_tail_marginal_RPV')
            item['covariance_compare']={'cond_vs_ekf_fro':rel(pc,pe),'cond_trace':float(np.trace(pc)),'ekf_trace':float(np.trace(pe))}
            if pm is not None: item['covariance_compare'].update({'marg_vs_ekf_fro':rel(pm,pe),'marg_trace':float(np.trace(pm)),'marg_vs_cond_fro':rel(pm,pc)})
            def gain(M):
                return M[6:9,3:6]@np.linalg.pinv(M[3:6,3:6])
            item['v_given_p_gains']={'ekf':gain(pe).tolist(),'conditional':gain(pc).tolist()}
            if pm is not None: item['v_given_p_gains']['marginal']=gain(pm).tolist()
        if 'A_process_raw' in mats: item['a_process_trace']=float(np.trace(mats['A_process_raw']))
        # Exact J^T W J reconstruction from every logged IMU sample in this snapshot.
        js=re.findall(r"matrix J_imu rows=(\d+) cols=(\d+)\n(.*?)(?=matrix |vector Wdiag|=== end_prior_audit ===)",block,re.S)
        ws=re.findall(r"vector Wdiag size=6\n(.*?)(?=imu_sample=|=== end_prior_audit ===)",block,re.S)
        if js and len(js)==len(ws):
            Arec=None
            for jm,wm in zip(js,ws):
                r,c=int(jm[0]),int(jm[1]); J=np.fromstring(jm[2],sep=' ')[:r*c].reshape(r,c); w=np.fromstring(wm,sep=' ')[:6]
                W=np.diag(w); A=J.T@W@J; Arec=A if Arec is None else Arec+A
            if 'A_process_raw' in mats:
                item['continuous_imu_A_reconstruction_rel']=rel(Arec,mats['A_process_raw'])
                item['continuous_imu_A_reconstruction_norm']=float(np.linalg.norm(Arec-mats['A_process_raw']))
        # FIX (found this round): the supplied script builds `item` per
        # snapshot but never appends it to result['snapshots'] -- every
        # invocation reported num_prior_audit_snapshots=1 (correct) but an
        # empty snapshots list, silently discarding every computed field.
        result['snapshots'].append(item)
    result['state_chain_summary']={}
    if args.state_chain:
        rows=list(csv.DictReader(open(args.state_chain,newline='')))
        result['state_chain_summary']['rows']=len(rows)
        for r in rows:
            if r['phase']=='post_iteration':
                key=r['architecture']; result['state_chain_summary'].setdefault(key,[]).append({
                    'iteration':int(r['iteration']),
                    'imu_dp_norm':float(np.linalg.norm([float(r['imu_dp_x']),float(r['imu_dp_y']),float(r['imu_dp_z'])])),
                    'lio_dp_norm':float(np.linalg.norm([float(r['lio_dp_x']),float(r['lio_dp_y']),float(r['lio_dp_z'])])),
                    'delta_lio_dp_norm':float(np.linalg.norm([float(r['delta_lio_from_imu_x']),float(r['delta_lio_from_imu_y']),float(r['delta_lio_from_imu_z'])])),
                    'imu_dv_norm':float(np.linalg.norm([float(r['imu_dv_x']),float(r['imu_dv_y']),float(r['imu_dv_z'])])),
                    'lio_dv_norm':float(np.linalg.norm([float(r['lio_dv_x']),float(r['lio_dv_y']),float(r['lio_dv_z'])])),
                    'delta_lio_dv_norm':float(np.linalg.norm([float(r['delta_lio_vel_x']),float(r['delta_lio_vel_y']),float(r['delta_lio_vel_z'])])),
                    'lio_attitude_error_norm':float(np.linalg.norm([float(r['lio_dr_x']),float(r['lio_dr_y']),float(r['lio_dr_z'])])),
                })
    Path(args.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
