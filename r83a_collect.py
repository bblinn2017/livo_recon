#!/usr/bin/env python3
"""R83A collector. Copies requested files; never edits sources; stdlib only."""
import argparse, glob, hashlib, os, shutil, sys, tempfile, zipfile

PART = 28 * 1024 * 1024

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def arms(d):
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))) if os.path.isdir(d) else []

def tail(p, n=200):
    with open(p, 'rb') as f:
        return b''.join(f.readlines()[-n:])

def collect(a, stage):
    led = []
    def put(item, src, dst, text=None):
        d = os.path.join(stage, dst); os.makedirs(os.path.dirname(d), exist_ok=True)
        if text is not None:
            open(d, 'wb').write(text); led.append((item, dst, os.path.getsize(d), sha(d))); return
        if src and os.path.isfile(src):
            shutil.copy2(src, d); led.append((item, dst, os.path.getsize(d), sha(d)))
        else:
            led.append((item, dst, '-', 'ABSENT'))
    r3 = a.r83_runs
    for arm in arms(r3):
        base = os.path.join(r3, arm)
        for nm in ('plane_fit_trace.csv', 'plane_fit_trace.csv.gz'):
            if os.path.isfile(os.path.join(base, nm)): put('A1', os.path.join(base, nm), f'A1/{arm}/{nm}')
        if '_dst_' in arm:
            for nm in ('plane_fit_reasons.csv', 'residual_dump.csv', 'residual_dump_all.csv.gz', 'plane_fit_trace.csv'):
                put('A2', os.path.join(base, nm), f'A2/{arm}/{nm}')
        if 'refuse' in arm:
            lg = [p for p in (os.path.join(base, 'run.log'), os.path.join(base, 'stderr.txt')) if os.path.isfile(p)]
            put('A3', None, f'A3/{arm}/log_tail.txt', tail(lg[0]) if lg else b'ABSENT: no run.log or stderr.txt\n')
        if arm in ('r83_dc_sm_pcw_60s', 'r83_dc_sm_dst_60s'):
            for nm in ('profiler.txt', 'params_echo.txt', 'run.log'):
                put('A4', os.path.join(base, nm), f'A4/{arm}/{nm}')
        if '_mqs_' in arm:
            put('A6', os.path.join(base, 'motion_q_scan.csv'), f'A6/{arm}/motion_q_scan.csv')
    for arm in arms(a.r82_runs):
        for nm in ('exit_status.txt', 'engagement_imu.txt'):
            put('A5', os.path.join(a.r82_runs, arm, nm), f'A5/{arm}/{nm}')
    if a.imu_times:
        put('A7', a.imu_times, 'A7/imu_times.txt')
    if a.gates:
        put('A8', a.gates, 'A8/r83_gates_rerun.txt')
    if a.sweep:
        put('A9', a.sweep, 'A9/combined_sweep.csv')
    return led

def pack(stage, out, led):
    os.makedirs(out, exist_ok=True)
    lp = os.path.join(stage, 'A_ledger.tsv')
    with open(lp, 'w') as f:
        f.write('item\tpath\tbytes\tsha256\n')
        for r in led: f.write('\t'.join(map(str, r)) + '\n')
    z = os.path.join(tempfile.mkdtemp(), 'R83A_RETURN.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as Z:
        for root, _, fs in os.walk(stage):
            for n in fs:
                p = os.path.join(root, n); Z.write(p, os.path.relpath(p, stage))
    parts = []
    if os.path.getsize(z) <= PART:
        d = os.path.join(out, 'R83A_RETURN.zip'); shutil.copy2(z, d); parts = [d]
    else:
        with open(z, 'rb') as f:
            i = 0
            while True:
                b = f.read(PART)
                if not b: break
                d = os.path.join(out, 'R83A_RETURN.zip.%02d' % i); open(d, 'wb').write(b); parts.append(d); i += 1
    with open(os.path.join(out, 'R83A_RETURN.sha256'), 'w') as f:
        for p in parts: f.write(f'{sha(p)}  {os.path.basename(p)}\n')
    return parts

def selftest():
    t = tempfile.mkdtemp(); r3 = os.path.join(t, 'r3'); r2 = os.path.join(t, 'r2')
    for arm, files in {'r83_rt_dst_x_first_fq': ['plane_fit_trace.csv', 'plane_fit_reasons.csv'], 'r83_dc_refuse_a': ['run.log'], 'r83_rt_mqs_b': ['motion_q_scan.csv']}.items():
        os.makedirs(os.path.join(r3, arm))
        for n in files: open(os.path.join(r3, arm, n), 'w').write('x\n1\n' * 5)
    os.makedirs(os.path.join(r2, 'r82_a')); open(os.path.join(r2, 'r82_a', 'exit_status.txt'), 'w').write('0\n')
    class A: pass
    a = A(); a.r83_runs, a.r82_runs, a.imu_times, a.gates, a.sweep = r3, r2, None, None, None
    stage = os.path.join(t, 's'); os.makedirs(stage)
    led = collect(a, stage); parts = pack(stage, os.path.join(t, 'o'), led)
    items = [r[0] for r in led]
    assert items.count('A5') == 2 and any(r[3] == 'ABSENT' for r in led) and 'A3' in items and 'A6' in items, led
    assert len(parts) == 1 and open(os.path.join(t, 'o', 'R83A_RETURN.sha256')).read().count('\n') == 1
    global PART; PART = 200
    parts = pack(stage, os.path.join(t, 'o2'), led); assert len(parts) > 1
    assert b''.join(open(p, 'rb').read() for p in parts)[:2] == b'PK'
    print('selftest OK', len(led), 'ledger rows,', len(parts), 'parts when split')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--r83-runs'); ap.add_argument('--r83-return'); ap.add_argument('--r82-runs'); ap.add_argument('--r82-return')
    ap.add_argument('--imu-times'); ap.add_argument('--gates'); ap.add_argument('--sweep'); ap.add_argument('--out', default='R83A_RETURN')
    ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    a.r83_runs = a.r83_runs; 
    if a.selftest: selftest(); sys.exit(0)
    a.r83_runs = a.r83_runs or ''; a.r82_runs = a.r82_runs or ''
    stage = tempfile.mkdtemp(); led = collect(a, stage); parts = pack(stage, a.out, led)
    print('wrote', parts)
