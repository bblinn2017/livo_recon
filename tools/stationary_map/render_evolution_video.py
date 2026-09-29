#!/usr/bin/env python3
"""Video of how a stationary-map backend's planes/surfaces evolve, drawn over the
(undistorted) camera frame with the config intrinsics/extrinsics.

Per frame (one CLI checkpoint):
  top-left      VOXELS: every patch drawn as its finite-support quad, colour = the patch's OWN centre
  top-right     SURFACES: same quads, colour = one distinct hue per SURFACE (--surface-color id, default since R56;
                --surface-color mean gives the R55 mean-centre colour); each multi-member (merged) surface has an outline.
                A merge shows up as several voxels taking one colour.
  bottom-left   REFERENCE map (static, built from all scans), coloured the same way
  bottom-right  counters (scan, time, patches, surfaces, merged surfaces, largest surface,
                cumulative merges/splits/unmerges, points) and time-series with a moving cursor

Colour (--color-by): xyz = RGB from normalised body-frame x,y,z (DEFAULT; R=x, G=y, B=z, each scaled between the
                         bounds below, so nearby positions get similar colours)
                     x | y | z | range = turbo rainbow over that coordinate (range = |centre|)
Bounds: --bounds, else the 2-98 percentile of the LAST frame's patch centres. Pass the same
--bounds to every config (see --print-bounds) so colours are comparable across videos.
Invalid (rank-1) patches, if any, are drawn grey.

  render_evolution_video.py --patches P.csv --summary S.csv --reference reference_surfaces.csv \
     --camera-yaml ntu_viral.yaml --image camera_first.png --label "canonical / gaussian_surface" \
     --out video.mp4 [--stats-out stats.csv] [--stills-at 3,50,200,480 --stills-dir DIR]
     [--manifest cache/manifest.csv] [--stride 1] [--fps 15] [--scale 0.75] [--color-by xyz]
Patch CSV = stationary_map_cli --patches-out (rows grouped by checkpoint).
"""
import argparse, csv, os, shutil, subprocess, sys
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------- geometry
def quads(center, normal, bb_min, bb_max):
    n = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    t1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    bad = np.linalg.norm(t1, axis=1) < 1e-6
    t1[bad] = np.cross(n[bad], np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    ext = bb_max - bb_min
    e1 = np.maximum(0.5 * np.sum(np.abs(t1) * ext, axis=1), 0.03)
    e2 = np.maximum(0.5 * np.sum(np.abs(t2) * ext, axis=1), 0.03)
    cs = [center + s1 * e1[:, None] * t1 + s2 * e2[:, None] * t2
          for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.stack(cs, axis=1)


class Camera:
    """Undistorts the raw frame with K/D and projects body-frame points with a pinhole."""

    def __init__(self, cfg, raw_gray):
        ex = cfg['extrinsics']; cam = cfg['camera']
        self.R_li = np.array(ex['R_l2i']).reshape(3, 3); self.t_li = np.array(ex['t_l2i'])
        self.R_lc = np.array(ex['R_l2c']).reshape(3, 3); self.t_lc = np.array(ex['t_l2c'])
        K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.0]])
        D = np.array(cam['distortion_coeffs'])
        self.W, self.H = int(cam['width']), int(cam['height'])
        self.K, _ = cv2.getOptimalNewCameraMatrix(K, D, (self.W, self.H), 1.0, (self.W, self.H))
        self.img = cv2.undistort(raw_gray, K, D, None, self.K)

    def project(self, Q):
        """Q (N,4,3) body/IMU-frame quads -> uv (N,4,2), keep mask (all corners in front), depth."""
        pl = (Q.reshape(-1, 3) - self.t_li) @ self.R_li
        pc = pl @ self.R_lc.T + self.t_lc
        z = pc[:, 2]; zs = np.where(z > 1e-6, z, 1.0)
        uv = np.stack([self.K[0, 0] * pc[:, 0] / zs + self.K[0, 2],
                       self.K[1, 1] * pc[:, 1] / zs + self.K[1, 2]], axis=1).reshape(-1, 4, 2)
        zz = z.reshape(-1, 4)
        return uv, np.all(zz > 0.2, axis=1), zz.mean(axis=1)


# ---------------------------------------------------------------- colour
class Colourer:
    def __init__(self, mode, lo, hi):
        self.mode, self.lo, self.hi = mode, np.asarray(lo, float), np.asarray(hi, float)
        self.cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)

    def __call__(self, centres):
        """centres (N,3) -> BGR uint8 (N,3)."""
        c = np.asarray(centres, float)
        if self.mode == 'xyz':
            t = np.clip((c - self.lo) / np.maximum(self.hi - self.lo, 1e-9), 0, 1)
            rgb = (t * 255).astype(np.uint8)
            return np.ascontiguousarray(rgb[:, ::-1])
        if self.mode == 'range':
            v = np.linalg.norm(c, axis=1)
            lo, hi = np.linalg.norm(self.lo), np.linalg.norm(self.hi)
            lo = 0.0
        else:
            k = 'xyz'.index(self.mode)
            v, lo, hi = c[:, k], self.lo[k], self.hi[k]
        t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        u8 = (t * 255).astype(np.uint8).reshape(-1, 1)
        return cv2.applyColorMap(u8, self.cmap).reshape(-1, 3)


def surface_id_colours(patch_id, inv, n_surf):
    """Categorical colour per SURFACE (R56): hue from the golden-ratio sequence of a stable key (the smallest patch_id in
    the surface), with saturation/value cycling through a few levels, so neighbouring surfaces at the same height are told
    apart. Returns BGR uint8 (n_patches, 3). A surface keeps its colour while its lowest-id member stays in it."""
    key = np.full(n_surf, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(key, inv, patch_id.astype(np.int64))
    h = (key * 0.6180339887498949) % 1.0
    sat = 0.60 + 0.20 * (key % 3)            # 0.60, 0.80, 1.00
    val = 0.70 + 0.30 * ((key // 3) % 2)     # 0.70, 1.00
    hsv = np.stack([h * 179.0, sat * 255.0, val * 255.0], 1).astype(np.uint8).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return bgr[inv]


# ---------------------------------------------------------------- data
def read_patch_groups(path):
    """Stream a stationary_map_cli patches CSV (rows grouped by checkpoint) -> {cp: dict of arrays}."""
    want = ['patch_id', 'surface_id', 'point_count', 'center_x', 'center_y', 'center_z', 'normal_x', 'normal_y',
            'normal_z', 'bb_min_x', 'bb_min_y', 'bb_min_z', 'bb_max_x', 'bb_max_y', 'bb_max_z', 'rank2']
    groups, cur, rows = {}, None, []

    def flush():
        if cur is None:
            return
        a = np.array(rows, dtype=np.float64).reshape(len(rows), len(want))
        groups[cur] = {k: a[:, i] for i, k in enumerate(want)}

    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            cp = int(float(r['checkpoint']))
            if cp != cur:
                flush(); cur, rows = cp, []
            rows.append([float(r[k]) for k in want])
    flush()
    return groups


def read_reference(path):
    rows = list(csv.DictReader(open(path, newline='')))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {'center': np.stack([col('cx'), col('cy'), col('cz')], 1),
            'normal': np.stack([col('nx'), col('ny'), col('nz')], 1),
            'bb_min': np.stack([col('bb_min_x'), col('bb_min_y'), col('bb_min_z')], 1),
            'bb_max': np.stack([col('bb_max_x'), col('bb_max_y'), col('bb_max_z')], 1)}


def surface_means(sid, centre, w):
    u, inv = np.unique(sid, return_inverse=True)
    ws = np.bincount(inv, weights=w)
    m = np.stack([np.bincount(inv, weights=w * centre[:, k]) for k in range(3)], 1) / np.maximum(ws, 1e-12)[:, None]
    return m[inv], inv, np.bincount(inv)


# ---------------------------------------------------------------- drawing
def draw_polys(base, cam, uv, keep, depth, bgr, scale, alpha=0.65, hull_groups=None, hull_cols=None):
    over = base.copy()
    sel = [i for i in np.argsort(-depth) if keep[i]]
    for i in sel:
        p = np.round(uv[i] * scale).astype(np.int32)
        col = tuple(int(x) for x in bgr[i])
        cv2.fillConvexPoly(over, p, col)
        cv2.polylines(over, [p.reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)
    out = cv2.addWeighted(over, alpha, base, 1 - alpha, 0)
    if hull_groups is not None:
        for g, idx in hull_groups.items():
            idx = [i for i in idx if keep[i]]
            if len(idx) < 2:
                continue
            pts = np.round(uv[idx].reshape(-1, 2) * scale).astype(np.int32)
            h = cv2.convexHull(pts)
            cv2.polylines(out, [h], True, tuple(int(x) for x in hull_cols[idx[0]]), 2, cv2.LINE_AA)
    return out


def put(img, text, org, s=0.5, col=(255, 255, 255), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def sparkline(canvas, x0, y0, w, h, xs, ys, cur_x, label, col):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return
    xn = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-9)
    yn = (ys - 0) / max(ys.max(), 1e-9)
    pts = np.stack([x0 + xn * w, y0 + h - yn * (h - 4) - 2], 1).astype(np.int32)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
    cx = x0 + (cur_x - xs.min()) / max(xs.max() - xs.min(), 1e-9) * w
    cv2.line(canvas, (int(cx), y0), (int(cx), y0 + h), (255, 255, 255), 1)
    put(canvas, f"{label} (max {ys.max():.0f})", (x0 + 4, y0 + 14), 0.4, col)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patches', required=True); ap.add_argument('--summary', required=True)
    ap.add_argument('--reference', required=True); ap.add_argument('--camera-yaml', required=True)
    ap.add_argument('--image', required=True, help='raw (distorted) camera frame, PNG')
    ap.add_argument('--out', required=True); ap.add_argument('--label', default='')
    ap.add_argument('--stats-out'); ap.add_argument('--manifest')
    ap.add_argument('--stills-at', default=''); ap.add_argument('--stills-dir')
    ap.add_argument('--color-by', default='xyz', choices=['xyz', 'x', 'y', 'z', 'range'])
    ap.add_argument('--surface-color', default='id', choices=['id', 'mean'],
                    help="SURFACES panel colouring: id (default, R56) = a distinct hue per surface; mean = mean centre colour (R55)")
    ap.add_argument('--print-bounds', action='store_true', help='print 2-98 percentile bounds of the last frame (lox loy loz hix hiy hiz) and exit')
    ap.add_argument('--bounds', type=float, nargs=6, metavar=('LOX', 'LOY', 'LOZ', 'HIX', 'HIY', 'HIZ'))
    ap.add_argument('--stride', type=int, default=1); ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--dim', type=float, default=0.55)
    ap.add_argument('--max-frames', type=int, default=0, help='debug: render only the first N frames')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.camera_yaml))
    raw = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        sys.exit('cannot read image ' + a.image)
    cam = Camera(cfg, raw)
    W, H = int(cam.W * a.scale), int(cam.H * a.scale)
    base = cv2.cvtColor((cam.img.astype(np.float32) * a.dim).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)

    ref = read_reference(a.reference)
    groups = read_patch_groups(a.patches)
    cps = sorted(groups)
    last = groups[cps[-1]]
    lc = np.stack([last['center_x'], last['center_y'], last['center_z']], 1)
    lo = np.percentile(lc, 2, axis=0); hi = np.percentile(lc, 98, axis=0)
    if a.print_bounds:
        print(' '.join('%.4f' % v for v in list(lo) + list(hi))); return
    if a.bounds:
        lo, hi = np.array(a.bounds[:3]), np.array(a.bounds[3:])
    colr = Colourer(a.color_by, lo, hi)
    summ = {int(float(r['checkpoint'])): r for r in csv.DictReader(open(a.summary))}
    tstamp = {}
    if a.manifest:
        m = list(csv.DictReader(open(a.manifest)))
        t0 = float(m[0]['timestamp'])
        tstamp = {int(r['observation_id']) + 1: float(r['timestamp']) - t0 for r in m}

    # pass 1: per-checkpoint statistics (also the sparkline series)
    stats = []
    for cp in cps:
        g = groups[cp]; n = len(g['patch_id'])
        u, cnt = np.unique(g['surface_id'], return_counts=True)
        sm = summ.get(cp, {})
        stats.append({'checkpoint': cp, 'patches': n, 'surfaces': len(u), 'merged_surfaces': int((cnt > 1).sum()),
                      'largest_surface': int(cnt.max()) if n else 0, 'points_in_patches': float(g['point_count'].sum()),
                      'valid_fraction': float(g['rank2'].mean()) if n else 0.0,
                      'merges': float(sm.get('merges', 0)), 'splits': float(sm.get('splits', 0)),
                      'unmerges': float(sm.get('unmerges', 0)),
                      'points_ingested': float(sm.get('total_raw_points_ingested', 0))})
    if a.stats_out:
        run_id = os.path.splitext(os.path.basename(a.out))[0]
        arm_family = a.label.split('/')
        with open(a.stats_out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['row_type', 'run_id', 'arm', 'family', 'metric', 'value', 'detail'])
            for s in stats:
                for k, v in s.items():
                    if k != 'checkpoint':
                        w.writerow(['evolution_frame', run_id, arm_family[0].strip(), arm_family[-1].strip(), k, v, 'checkpoint=%d' % s['checkpoint']])

    # static reference panel
    rq = quads(ref['center'], ref['normal'], ref['bb_min'], ref['bb_max'])
    ruv, rkeep, rdep = cam.project(rq)
    ref_panel = draw_polys(base, cam, ruv, rkeep, rdep, colr(ref['center']), a.scale)
    put(ref_panel, 'REFERENCE (all scans; static)', (8, 20), 0.55)
    if a.stills_dir:
        os.makedirs(a.stills_dir, exist_ok=True)
        cv2.imwrite(os.path.join(a.stills_dir, 'reference.png'), ref_panel)

    def frame(k):
        cp = cps[k]; g = groups[cp]
        Q = quads(np.stack([g['center_x'], g['center_y'], g['center_z']], 1),
                  np.stack([g['normal_x'], g['normal_y'], g['normal_z']], 1),
                  np.stack([g['bb_min_x'], g['bb_min_y'], g['bb_min_z']], 1),
                  np.stack([g['bb_max_x'], g['bb_max_y'], g['bb_max_z']], 1))
        ctr = np.stack([g['center_x'], g['center_y'], g['center_z']], 1)
        uv, keep, dep = cam.project(Q)
        valid = g['rank2'] > 0.5
        cv = colr(ctr); cv[~valid] = (128, 128, 128)
        smean, inv, cnt = surface_means(g['surface_id'].astype(np.int64), ctr, np.maximum(g['point_count'], 1.0))
        if a.surface_color == 'id':
            cs = surface_id_colours(g['patch_id'], inv, len(cnt))
        else:
            cs = colr(smean)
        cs[~valid] = (128, 128, 128)
        groups_multi = {}
        for i, gi in enumerate(inv):
            if cnt[gi] > 1:
                groups_multi.setdefault(gi, []).append(i)
        A = draw_polys(base, cam, uv, keep, dep, cv, a.scale)
        B = draw_polys(base, cam, uv, keep, dep, cs, a.scale, hull_groups=groups_multi, hull_cols=cs)
        put(A, 'VOXELS: colour = own centre', (8, 20), 0.55)
        put(B, ('SURFACES: hue = surface id; outline = merged' if a.surface_color == 'id'
                else 'SURFACES: colour = surface mean centre; outline = merged'), (8, 20), 0.55)
        D = np.zeros_like(base)
        s = stats[k]
        t = tstamp.get(cp)
        lines = [f"{a.label}", f"scan {cp}" + (f"   t = {t:5.1f} s" if t is not None else ''),
                 f"patches {s['patches']}   surfaces {s['surfaces']}",
                 f"merged surfaces {s['merged_surfaces']}   largest {s['largest_surface']} patches",
                 f"cum. merge-ops {s['merges']:.0f}  split-ops {s['splits']:.0f}  unmerge-ops {s['unmerges']:.0f}",
                 f"points ingested {s['points_ingested']:.0f}"]
        y = 22
        for ln in lines:
            put(D, ln, (8, y), 0.5 * max(a.scale, 0.75)); y += int(20 * max(a.scale, 0.75))
        xs = [q['checkpoint'] for q in stats]
        hh = int(48 * a.scale + 10)
        yb = y + 6
        sparkline(D, 8, yb, W - 16, hh, xs, [q['patches'] for q in stats], cp, 'patches', (80, 200, 255))
        sparkline(D, 8, yb + hh + 8, W - 16, hh, xs, [q['surfaces'] for q in stats], cp, 'surfaces', (120, 255, 120))
        sparkline(D, 8, yb + 2 * (hh + 8), W - 16, hh, xs, [q['merged_surfaces'] for q in stats], cp, 'merged surfaces', (255, 160, 120))
        top = np.hstack([A, B]); bot = np.hstack([ref_panel, D])
        return np.vstack([top, bot])

    still_set = {int(x) for x in a.stills_at.split(',') if x.strip()}
    idxs = list(range(0, len(cps), max(1, a.stride)))
    if a.max_frames:
        idxs = idxs[:a.max_frames]
    Wc, Hc = 2 * W, 2 * H
    Wc -= Wc % 2; Hc -= Hc % 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = shutil.which('ffmpeg')
    proc = None; vw = None
    if ff:
        proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{Wc}x{Hc}',
                                 '-r', str(a.fps), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '27',
                                 '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    else:
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (Wc, Hc))
    for j, k in enumerate(idxs):
        fr = frame(k)[:Hc, :Wc]
        if proc:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        else:
            vw.write(fr)
        if a.stills_dir and cps[k] in still_set:
            cv2.imwrite(os.path.join(a.stills_dir, f'frame_{cps[k]:04d}.png'), fr)
        if j % 50 == 0:
            print(f'frame {j}/{len(idxs)} (scan {cps[k]})', flush=True)
    if proc:
        proc.stdin.close(); rc = proc.wait()
        if rc != 0:
            sys.exit('ffmpeg failed rc=%d' % rc)
    else:
        vw.release()
    print('wrote', a.out, f'{len(idxs)} frames', f'{Wc}x{Hc}')


if __name__ == '__main__':
    main()
