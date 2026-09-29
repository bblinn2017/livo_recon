#!/usr/bin/env python3
"""Keep only some checkpoints of a (large) stationary_map_cli patches or summary CSV.
  subset_checkpoints.py --in big.csv --out small.csv --checkpoints 3,10,25,50,100,200,300,400,480
Streams the file; header is kept. Column 'checkpoint' selects rows."""
import argparse, csv
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--checkpoints', required=True)
a = ap.parse_args()
keep = {int(x) for x in a.checkpoints.split(',')}
n = m = 0
with open(a.inp, newline='') as f, open(a.out, 'w', newline='') as g:
    rd = csv.reader(f); w = csv.writer(g)
    hdr = next(rd); w.writerow(hdr); ci = hdr.index('checkpoint')
    for r in rd:
        n += 1
        if int(float(r[ci])) in keep:
            w.writerow(r); m += 1
print('kept', m, 'of', n, 'rows ->', a.out)
