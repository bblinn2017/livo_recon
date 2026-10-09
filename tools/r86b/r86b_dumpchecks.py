#!/usr/bin/env python3
"""r86b_dumpchecks.py -- the R86B soundness checks of the dumps (planner's script; selftest on synthetic files only, which is not evidence, rule 1.1). Imported by r86b_gates.py.

What it does: every quantity R86B dumps twice, through two independent writers, is compared with itself, and the dumps are compared with the shipped R86A reference digests.
    PREIMU    state_trace.csv: exactly one pre_imu row (scan 1) before the post_imu row and the post_lio row of scan 1, all with the same t_abs
    CHAIN     pose_control_first_frame_state_chain.csv (state before IMU, after IMU, after every LIO iteration) against the state_trace rows
    ITERS     iterations 0..n-1 (n = completed_iterations of the post_lio row) present in residual_dump.csv, residual_dump_state.csv and the solve matrices, with equal
              residual counts, and the state after each step equal in the residual dump state and the solve matrices; the state before iteration k equals the state after k-1
    COVBUDGET pose_control_first_frame_covariance_budget.txt (P before IMU, after IMU, after LIO) against the state_trace diagonals; the .csv has n+2 rows
    DIGEST    frame-1 residual rows at iteration 0 and at the last iteration against R86A (sha256 of the canonical rows, plane_key excluded: it is a pointer)
    MAPLINK   every iteration-0 residual voxel is a plane voxel of post_calibration_voxelmap.csv with the same point count and layer
    MAPSAME   post_calibration_voxelmap.csv against R86A (all columns except the tangent basis, which differs in the last bits between two builds of the same map)
    SNAPF2    voxelmap_snapshot_f2.csv has fs_n_voxels rows and fs_n_voxels_is_plane plane voxels of scan 1 (short_scan_all.csv.gz)
Facts only. Usage by r86b_gates.py; from a shell:
    python3 tools/r86b/r86b_dumpchecks.py --make-refs --ret-dir R86A_RETURN --out r86b_manifest/ref_r86a_digests.json     (planner, once)
    python3 tools/r86b/r86b_dumpchecks.py --selftest
"""
import argparse, csv, gzip, hashlib, json, math, os, re, sys, tempfile

PFN_TAGS = ("m3r3", "m1r1", "m1r3", "m3r1")
RES_COLS = ["source_index", "r", "sigma_squared", "plane_var_term", "node_id", "tier", "hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z", "row_kind", "floor_term",
            "sigma_diag_squared", "s_prior_pose", "sigma_gate_squared", "voxel_point_count", "voxel_distinct_frames", "voxel_layer"]
TOL_POS = 1e-12          # printed with 17 significant digits by both writers
TOL_ROT = 1e-9           # measured on real R86A data: the conversion from the logged quaternion agrees with the node's Log() to 1.5e-13
REL_COV = 1e-4           # Eigen's operator<< prints 6 significant digits


def rd(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def resid_digests(path, frame=1):
    """{iteration: (rows, sha256)} of the canonical rows of one frame of residual_dump.csv(.gz), in file order."""
    h, n = {}, {}
    with rd(path) as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) != frame:
                continue
            it = int(r["iter"])
            h.setdefault(it, hashlib.sha256())
            n[it] = n.get(it, 0) + 1
            h[it].update(("|".join(r[c] for c in RES_COLS) + "\n").encode())
    return {it: (n[it], h[it].hexdigest()) for it in n}


def map_digest(path):
    """(rows, sha256) of a voxel-map snapshot: every column except tangent_*, tokens as written."""
    h = hashlib.sha256()
    with rd(path) as f:
        rows = csv.reader(f)
        hdr = next(rows)
        keep = [i for i, c in enumerate(hdr) if not c.startswith("tangent_")]
        h.update(("|".join(hdr[i] for i in keep) + "\n").encode())
        n = 0
        for r in rows:
            h.update(("|".join(r[i] for i in keep) + "\n").encode())
            n += 1
    return n, h.hexdigest()


def make_refs(ret, out):
    refs = {"note": "digests of the R86A return (frame 1); residual: iteration 0 and the last iteration; map: post_calibration_voxelmap.csv without tangent_* columns", "resid": {}, "map": {}}
    for t in PFN_TAGS:
        a = os.path.join(ret, "archive", "r86a_rt_sp_eee_01_v0p25_" + t)
        d = resid_digests(os.path.join(a, "residual_dump.csv.gz"))
        last = max(d)
        refs["resid"][t] = {"0": list(d[0]), "last_iter": last, "last": list(d[last])}
        refs["map"][t] = list(map_digest(os.path.join(a, "post_calibration_voxelmap.csv.gz")))
    json.dump(refs, open(out, "w"), indent=1)
    return refs


def state_rows(path):
    """rows of scan 1 and the count of pre_imu rows of a state_trace.csv(.gz)."""
    scan1, n_pre = [], 0
    with rd(path) as f:
        for r in csv.DictReader(f):
            if r["phase"] == "pre_imu":
                n_pre += 1
            if r["scan_id"] == "1":
                scan1.append(r)
    return scan1, n_pre


def rotvec(qw, qx, qy, qz):
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    s = math.sqrt(qx * qx + qy * qy + qz * qz)
    if s < 1e-300:
        return [0.0, 0.0, 0.0]
    ang = 2.0 * math.atan2(s, qw)
    return [qx / s * ang, qy / s * ang, qz / s * ang]


def solve_snapshots(path):
    snaps, cur = [], None
    keys = ("iteration=", "n_residuals=", "state_before_pos=", "state_after_pos=")
    with rd(path) as f:
        for ln in f:
            if ln.startswith("=== iekf_solve_snapshot"):
                cur = {}
                snaps.append(cur)
            elif cur is not None:
                for k in keys:
                    if ln.startswith(k) and k[:-1] not in cur:
                        cur[k[:-1]] = ln[len(k):].split()
    return snaps


def budget_blocks(path):
    lines = [x.rstrip("\n") for x in rd(path)]
    blocks, cur, i = [], None, 0
    while i < len(lines):
        m = re.match(r"=== covariance_budget (\S+) (\S+) iter=(-?\d+) t_abs=(\S+) ===", lines[i])
        if m:
            cur = {"phase": m.group(2), "iter": int(m.group(3)), "mats": {}}
            blocks.append(cur)
            i += 1
            continue
        if cur is not None and lines[i].strip() in ("P_before_IMU", "P_after_IMU", "P_after_LIO"):
            name, rows, j = lines[i].strip(), [], i + 1
            while j < len(lines) and len(rows) < 18:
                try:
                    rows.append([float(x) for x in lines[j].split()])
                except ValueError:
                    break
                j += 1
            cur["mats"][name] = rows
            i = j
            continue
        i += 1
    return blocks


class Checks:
    def __init__(self):
        self.rows = []     # crosscheck csv rows
        self.res = {}      # gate -> [ok, [messages], [missing]]

    def gate(self, g):
        return self.res.setdefault(g, [True, [], []])

    def missing(self, g, msg):
        self.gate(g)[2].append(msg)

    def num(self, g, arm, qty, a, b, tol, rel=False):
        d = abs(a - b)
        ok = d <= (tol * max(abs(a), abs(b)) + 1e-15 if rel else tol)
        self.rows.append(dict(check=g, arm=arm, quantity=qty, value_a=repr(a), value_b=repr(b), abs_diff=repr(d), tol=tol, verdict="PASS" if ok else "FAIL"))
        if not ok:
            self.fail(g, "%s %s: %r vs %r (diff %.3g, tol %g)" % (arm, qty, a, b, d, tol))
        return ok

    def true(self, g, arm, qty, cond, text):
        self.rows.append(dict(check=g, arm=arm, quantity=qty, value_a=text, value_b="", abs_diff="", tol="", verdict="PASS" if cond else "FAIL"))
        if not cond:
            self.fail(g, "%s %s: %s" % (arm, qty, text))
        return cond

    def fail(self, g, msg):
        r = self.gate(g)
        r[0] = False
        r[1].append(msg)

    def verdicts(self, gates):
        out = []
        for g in gates:
            ok, msgs, miss = self.gate(g)
            if miss:
                out.append((g, "MISSING", "; ".join(miss[:8])))
            elif not ok:
                out.append((g, "FAIL", "; ".join(msgs[:8]) + (" (+%d more)" % (len(msgs) - 8) if len(msgs) > 8 else "")))
            else:
                n = sum(1 for r in self.rows if r["check"] == g)
                out.append((g, "PASS", "%d comparisons, all within tolerance" % n))
        return out


GATES = ("PREIMU", "CHAIN", "ITERS", "COVBUDGET", "DIGEST", "MAPLINK", "MAPSAME", "SNAPF2")


def run_checks(ret, man, arms):
    """arms: {tag: arm_id}. Returns (verdict list, crosscheck rows)."""
    C = Checks()
    refs = json.load(open(os.path.join(man, "ref_r86a_digests.json")))
    for t, a in arms.items():
        ad = os.path.join(ret, "archive", a)
        P = lambda fn: os.path.join(ad, fn)
        have = lambda g, fn: os.path.isfile(P(fn)) or (C.missing(g, a + "/" + fn + " absent") and False)
        # ---- PREIMU
        post_lio = post_imu = pre = None
        n_iter = None
        if have("PREIMU", "state_trace.csv.gz"):
            scan1, n_pre = state_rows(P("state_trace.csv.gz"))
            ph = [r["phase"] for r in scan1]
            C.true("PREIMU", a, "pre_imu rows in the file", n_pre == 1, "%d (need exactly 1)" % n_pre)
            C.true("PREIMU", a, "phases of scan 1 in file order", ph == ["pre_imu", "post_imu", "post_lio"], str(ph))
            if ph == ["pre_imu", "post_imu", "post_lio"]:
                pre, post_imu, post_lio = scan1
                C.true("PREIMU", a, "t_abs equal in the three rows", pre["t_abs"] == post_imu["t_abs"] == post_lio["t_abs"], "%s %s %s" % (pre["t_abs"], post_imu["t_abs"], post_lio["t_abs"]))
                C.true("PREIMU", a, "pre_imu residual_count and completed_iterations", pre["residual_count"] == "0" and pre["completed_iterations"] == "0", "%s %s" % (pre["residual_count"], pre["completed_iterations"]))
                C.true("PREIMU", a, "pre_imu dim 18", pre["dim"] == "18", pre["dim"])
                dg = [float(pre["P_%d_%d" % (i, i)]) for i in range(18)]
                C.true("PREIMU", a, "pre_imu covariance diagonal finite and positive", all(math.isfinite(x) and x > 0 for x in dg), "min %.3g" % min(dg))
                n_iter = int(post_lio["completed_iterations"])
        # ---- CHAIN
        if post_lio is not None and have("CHAIN", "pose_control_first_frame_state_chain.csv.gz"):
            with rd(P("pose_control_first_frame_state_chain.csv.gz")) as f:
                ch = list(csv.DictReader(f))
            pu = [r for r in ch if r["phase"] == "pre_update"]
            pi = sorted([r for r in ch if r["phase"] == "post_iteration"], key=lambda r: int(r["iteration"]))
            C.true("CHAIN", a, "rows pre_update / post_iteration", len(pu) == 1 and len(pi) == n_iter, "%d / %d (post_iteration need %d)" % (len(pu), len(pi), n_iter))
            C.true("CHAIN", a, "iteration numbers", [int(r["iteration"]) for r in pi] == list(range(n_iter)), str([int(r["iteration"]) for r in pi]))
            if len(pu) == 1 and pi:
                r0 = pu[0]
                for k, ax in enumerate("xyz"):
                    C.num("CHAIN", a, "position before IMU %s: chain x0 vs pre_imu row" % ax, float(r0["x0_p" + ax]), float(pre["p" + ax]), TOL_POS)
                    C.num("CHAIN", a, "position after IMU %s: chain ximu vs post_imu row" % ax, float(r0["ximu_p" + ax]), float(post_imu["p" + ax]), TOL_POS)
                    C.num("CHAIN", a, "position after last iteration %s: chain xlio vs post_lio row" % ax, float(pi[-1]["xlio_p" + ax]), float(post_lio["p" + ax]), TOL_POS)
                    C.num("CHAIN", a, "velocity before IMU %s: chain x0 vs pre_imu row" % ax, float(r0["x0_v" + ax]), float(pre["v" + ax]), TOL_POS)
                    C.num("CHAIN", a, "velocity after IMU %s: chain ximu vs post_imu row" % ax, float(r0["ximu_v" + ax]), float(post_imu["v" + ax]), TOL_POS)
                rv = lambda row: rotvec(*(float(row[k]) for k in ("qw", "qx", "qy", "qz")))
                for k, ax in enumerate("xyz"):
                    C.num("CHAIN", a, "rotation vector before IMU %s: chain x0_r vs pre_imu quaternion" % ax, float(r0["x0_r_" + ax]), rv(pre)[k], TOL_ROT)
                    C.num("CHAIN", a, "rotation vector after IMU %s: chain ximu_r vs post_imu quaternion" % ax, float(r0["ximu_r_" + ax]), rv(post_imu)[k], TOL_ROT)
                    C.num("CHAIN", a, "rotation vector after last iteration %s: chain xlio_r vs post_lio quaternion" % ax, float(pi[-1]["xlio_r_" + ax]), rv(post_lio)[k], TOL_ROT)
        # ---- ITERS
        if n_iter is not None and have("ITERS", "residual_dump.csv.gz") and have("ITERS", "residual_dump_state.csv.gz") and have("ITERS", "pose_control_first_frame_solve_matrices.txt.gz"):
            cnt = {}
            with rd(P("residual_dump.csv.gz")) as f:
                for r in csv.DictReader(f):
                    if r["frame"] == "1":
                        cnt[int(r["iter"])] = cnt.get(int(r["iter"]), 0) + 1
            with rd(P("residual_dump_state.csv.gz")) as f:
                st = {int(r["iter"]): r for r in csv.DictReader(f) if r["frame"] == "1"}
            sn = solve_snapshots(P("pose_control_first_frame_solve_matrices.txt.gz"))
            C.true("ITERS", a, "iterations in residual_dump.csv", sorted(cnt) == list(range(n_iter)), "%s (need 0..%d)" % (sorted(cnt), n_iter - 1))
            C.true("ITERS", a, "iterations in residual_dump_state.csv", sorted(st) == list(range(n_iter)), "%s (need 0..%d)" % (sorted(st), n_iter - 1))
            C.true("ITERS", a, "snapshots in the solve matrices", len(sn) == n_iter and [int(s["iteration"][0]) for s in sn] == list(range(n_iter)), "%d snapshots (need %d)" % (len(sn), n_iter))
            if sorted(cnt) == list(range(n_iter)) and sorted(st) == list(range(n_iter)) and len(sn) == n_iter:
                for k in range(n_iter):
                    C.true("ITERS", a, "residual count iteration %d: rows / state file / solve matrices" % k, cnt[k] == int(st[k]["n_resid"]) == int(sn[k]["n_residuals"][0]),
                           "%d / %s / %s" % (cnt[k], st[k]["n_resid"], sn[k]["n_residuals"][0]))
                    after = [float(x) for x in sn[k]["state_after_pos"]]
                    before = [float(x) for x in sn[k]["state_before_pos"]]
                    for i, ax in enumerate("xyz"):
                        C.num("ITERS", a, "state after step %d %s: residual_dump_state vs solve matrices" % (k, ax), float(st[k]["p" + ax]), after[i], TOL_POS)
                        prev = float(st[k - 1]["p" + ax]) if k > 0 else float(post_imu["p" + ax])
                        C.num("ITERS", a, "state before step %d %s: %s" % (k, ax, "state after step k-1" if k > 0 else "post_imu row"), before[i], prev, TOL_POS)
                for i, ax in enumerate("xyz"):
                    C.num("ITERS", a, "state after the last step %s: residual_dump_state vs post_lio row" % ax, float(st[n_iter - 1]["p" + ax]), float(post_lio["p" + ax]), TOL_POS)
        # ---- COVBUDGET
        if post_lio is not None and have("COVBUDGET", "pose_control_first_frame_covariance_budget.txt.gz") and have("COVBUDGET", "pose_control_first_frame_covariance_budget.csv.gz"):
            with rd(P("pose_control_first_frame_covariance_budget.csv.gz")) as f:
                cr = list(csv.DictReader(f))
            C.true("COVBUDGET", a, "rows of the budget csv (pre_update + n post_iteration + post_covariance)", len(cr) == n_iter + 2, "%d (need %d)" % (len(cr), n_iter + 2))
            bl = budget_blocks(P("pose_control_first_frame_covariance_budget.txt.gz"))
            C.true("COVBUDGET", a, "blocks in the budget txt", len(bl) == n_iter + 2 and bl[0]["phase"] == "pre_update" and bl[-1]["phase"] == "post_covariance", str([b["phase"] for b in bl]))
            if len(bl) == n_iter + 2 and bl[0]["phase"] == "pre_update" and bl[-1]["phase"] == "post_covariance":
                for name, blk, row in (("P before IMU", bl[0]["mats"].get("P_before_IMU"), pre), ("P after IMU", bl[0]["mats"].get("P_after_IMU"), post_imu), ("P after LIO", bl[-1]["mats"].get("P_after_LIO"), post_lio)):
                    ok = blk is not None and len(blk) == 18 and all(len(r) == 18 for r in blk)
                    C.true("COVBUDGET", a, name + " is 18x18 in the txt", ok, "found" if ok else "not parsed")
                    if ok:
                        for i in range(18):
                            C.num("COVBUDGET", a, "%s diagonal %d: txt vs state_trace" % (name, i), blk[i][i], float(row["P_%d_%d" % (i, i)]), REL_COV, rel=True)
        # ---- DIGEST
        if have("DIGEST", "residual_dump.csv.gz"):
            d = resid_digests(P("residual_dump.csv.gz"))
            rf = refs["resid"][t]
            li = rf["last_iter"]
            C.true("DIGEST", a, "last iteration index (R86A %d)" % li, n_iter is not None and n_iter - 1 == li, "completed_iterations-1 = %s" % (None if n_iter is None else n_iter - 1))
            for key, it in (("0", 0), ("last", li)):
                got = d.get(it)
                want = tuple(rf[key])
                C.true("DIGEST", a, "frame-1 residual rows iteration %d: count and sha256 equal R86A" % it, got is not None and tuple(got) == want, "got %s want %s" % (got, want))
        # ---- MAPLINK, MAPSAME
        if have("MAPLINK", "residual_dump.csv.gz") and have("MAPLINK", "post_calibration_voxelmap.csv.gz"):
            vm = {}
            with rd(P("post_calibration_voxelmap.csv.gz")) as f:
                for r in csv.DictReader(f):
                    vm[r["node_id"]] = (r["is_plane"], r["point_count"], r["layer"])
            seen, bad = set(), []
            with rd(P("residual_dump.csv.gz")) as f:
                for r in csv.DictReader(f):
                    if r["frame"] != "1" or r["iter"] != "0" or r["node_id"] in seen:
                        continue
                    seen.add(r["node_id"])
                    m = vm.get(r["node_id"])
                    if m is None or m[0] != "1" or m[1] != r["voxel_point_count"] or m[2] != r["voxel_layer"]:
                        bad.append((r["node_id"], m, r["voxel_point_count"], r["voxel_layer"]))
            C.true("MAPLINK", a, "iteration-0 residual voxels: plane voxel of the post-calibration map with equal point count and layer", len(seen) > 0 and not bad,
                   "%d voxels, %d not matching %s" % (len(seen), len(bad), bad[:2]))
            nm, hm = map_digest(P("post_calibration_voxelmap.csv.gz"))
            C.true("MAPSAME", a, "post_calibration_voxelmap rows and sha256 (no tangent_*) equal R86A", [nm, hm] == refs["map"][t], "got %d %s want %s" % (nm, hm[:16], refs["map"][t][0:1] + [refs["map"][t][1][:16]]))
        # ---- SNAPF2
        sp = os.path.join(ret, "compact", a, "short_scan_all.csv.gz")
        if have("SNAPF2", "voxelmap_snapshot_f2.csv.gz") and os.path.isfile(sp):
            with rd(sp) as f:
                s1 = [r for r in csv.DictReader(f) if r["scan"] == "1"]
            nrow = npl = 0
            with rd(P("voxelmap_snapshot_f2.csv.gz")) as f:
                for r in csv.DictReader(f):
                    nrow += 1
                    npl += r["is_plane"] == "1"
            C.true("SNAPF2", a, "snapshot f2 voxels and plane voxels equal scan 1 of short_scan_all", len(s1) == 1 and nrow == int(float(s1[0]["fs_n_voxels"])) and npl == int(float(s1[0]["fs_n_voxels_is_plane"])),
                   "snapshot %d / %d, scan 1 %s / %s" % (nrow, npl, s1[0]["fs_n_voxels"] if s1 else "?", s1[0]["fs_n_voxels_is_plane"] if s1 else "?"))
        elif os.path.isfile(P("voxelmap_snapshot_f2.csv.gz")):
            C.missing("SNAPF2", a + ": short_scan_all.csv.gz absent")
    return C.verdicts(GATES), C.rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-refs", action="store_true"); ap.add_argument("--ret-dir"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import r86b_gates
        sys.exit(r86b_gates.selftest())
    if a.make_refs:
        r = make_refs(a.ret_dir, a.out)
        print(json.dumps({k: (v if k != "resid" else {t: {kk: vv for kk, vv in x.items()} for t, x in v.items()}) for k, v in r.items()}, indent=1)[:1500])
