#!/usr/bin/env python3
"""R70 bootstrap reducer + identity gates (planning agent's; not built or run by me).

  python3 bootstrap_reduce.py <arm_dir>                   -> <arm_dir>/bootstrap_summary.json
  python3 bootstrap_reduce.py --collect <runs_dir> --out <ret_dir>
        -> <ret_dir>/bootstrap_summary.csv   one row per arm: schema, hash, sensor_cov_trace_over_3_{mean,min,max}, zero count, deskew fallback flag, digest
        -> <ret_dir>/identity_gates.txt       PASS/FAIL per gate (see R69_INSTRUCTIONS.md section 6)
Gates (R69 build vs the R67 sensor_model runs; the R69 patch removes code and must not change arithmetic):
  G1 00_id_baseline_{48s,60s,full}, 00_id_motionq_full: blank-run-id state_trace digest == the R67 sensor_model digest
  G2 every non-refusal arm: bootstrap_preparation schema v2, flattened_prepared_population_hash == R67 sensor_model hash, zero count 0
  G3 00_id_default_posecov_60s digest == 00_id_baseline_60s digest (new compiled default sensor_only == the pinned value)
  G4 00_id_infl_whole_f1_60s digest == 00_id_baseline_60s digest (whole-run inflation at factor 1 is the identity)
  G5 00_id_baseline_full_rerun digest == 00_id_baseline_full digest (determinism)
  G8 diagnostics-on arms digest == the matching diagnostics-off arm (the new dumps are write-only)
  G9 (R70) block information scale: 10/10 equals the global F=10 at frame 1 iteration 0; 12/2 equals S HtH(F=1) S; control: 10/10 against F=1 must differ
  G11 (R70) arms that repeat an R69B arm reproduce its digest (surface, dense octree, combined x F cells)
  G13 (R70) no engagement.txt carries an INERT flag
  G6 r71_refuse jobs: each refused at startup and run.log contains the expected text
"""
import argparse, csv, json, os, re, sys

R67_DIGEST = {"00_id_baseline_48s": "694fadd31260f9fb262e7e32690e9cc7aad80079b08c22ed1d2f3f9ec321d1be",
              "00_id_baseline_60s": "8cb066436baaedb45f2af20921ff56337a1b553e1aa2d8fafe9d4170a9043f19",
              "00_id_baseline_full": "8bc89b6bb69aaf32d2283a969ad6cffad798836cf40ccf0def2fe9712d14e5c2",
              "00_id_motionq_full": "98f3855cb2146bb06e9dc15b101a8699804a395b4807144323f99e2e509dbad1"}
R67_HASH = "1957524258770630969"
R69B_DIGEST = {   # R69B returned digests (bootstrap_summary.csv); the R70 changes to these paths are write-only (counters, logs), so the digests must not move
    "r71_sf_cell0p25_default": ("r69_sf_cell0p25_default_aggregate", "a737153e343e54d16b3985bbbae140d942f3ced43570fe56cf62194c37dfab12"),
    "r71_sf_cell0p425_default": ("r69_sf_cell0p425_default_aggregate", "07469345620246247197821e8834f83105816d3131281a9dd121e46c381f0aac"),
    "r71_sf_cell0p425_dense": ("r69_sf_cell0p425_dense_aggregate", "56e5130a476bb553aaeb3bf7434299c76a22d8b0388157057c81006807dd6aeb"),
    "r71_sf_pfn1_dsoff_fixedq": ("r69_sf_pfn1_dsoff_fixedq", "ce63fd4dfb6936ff64947c63394ffb143e40bfb57684d9d8f6a33392e264a7f4"),
    "r71_sf_pfn1_dsoff_motionq": ("r69_sf_pfn1_dsoff_motionq", "8dbe66801976d63f0db08250731d6838896b3ccc60ae3eb8666c0657e5aecd63"),
    "r71_cb_eee_01_fq_f1": ("r69_cb_eee_01_fq_f1", "df1b9f66c7fcab75d265ce536922b292f07c3b9b5f9e5ac7a8cf6d0e61db1c5a"),
    "r71_cb_eee_01_fq_f10": ("r69_cb_eee_01_fq_f10", "157d610c87c2421d13f5608d5c43805b42fb4bfc69b058b47fa76e505b4f1287"),
    "r71_cb_eee_01_fq_f30": ("r69_cb_eee_01_fq_f30", "59b1a11f94a77bf54f99065ee323bd889af3cb520c9aaf95fd5374e790d07b99"),
    "r71_cb_eee_01_mq_f1": ("r69_cb_eee_01_mq_f1", "2f2e491e107719fd419b385fab6f7da656bd1daaa70756e223edcbca3aeda905"),
    "r71_cb_eee_01_mq_f10": ("r69_cb_eee_01_mq_f10", "5a1ee08d95ceb0791e21d7ededfc786bde7a91b10d8e58a6bb3bf082a654c295"),
    "r71_cb_eee_01_mq_f30": ("r69_cb_eee_01_mq_f30", "bbca8f11698cb2a4f141727e65b8c98f19d1de1f84911f8903fda65e2d92bda6"),
}
R70_DIGEST = {   # R70 returned digests of arms whose R71 counterpart differs only in write-only logs (full coupled diagnostics): G11b
    "r71_co_tail_fulldiag_60s": ("r70_co_tail_fulldiag_60s", "a54199a34bea4347cac71566fa57d24d68a6e814495d7b174422f3bbb735e9c1"),
    "r71_co_meas_fulldiag_60s": ("r70_co_meas_fulldiag_60s", "80434ab13524156cea5fcb95966829761e153a0a43a03d3fb5f3f1935509b685"),
    "r71_co_tail_gateon_knots8_fq": ("r70_co_tail_gateon_knots8_iters5_fixedq (minimal diagnostics; R71 has full coupled diagnostics on: tests that they are write-only)",
                                     "2f1f87d189cc7d814881f2e8161a828eb3bbb09bc0a5e91c930bec38f625fe72"),
}
REFUSE = {"r71_refuse_point_cov_zero": "removed key: calib/bootstrap/point_cov",
          "r71_refuse_spline_mode": ["[config/REFUSED]", "spline/mode"],
          "r71_refuse_estimator_decoupled": "RENAMED to 'iekf'",
          "r71_refuse_surface_cell0p10": "cell sizes <= 0.10 m are retired",
          "r71_refuse_octree_voxel0p10": "voxel sizes <= 0.10 m are retired",
          "r71_refuse_info_scale_inert": "is inert",
          "r71_refuse_info_scale_suboption": ["[config/REFUSED]", "lio/info_scale/rotation"],
          "r71_refuse_info_scale_coupled": "estimator/mode=iekf only",
          "r71_refuse_dump_frames_bad": "is not a positive integer frame index",
          "r71_refuse_renorm_without_woodbury": ["[config/REFUSED]", "lio/residual_redundancy/renorm"]}


def update_row(runs, arm, frame=1, it=0):
    """Frame <frame>, iteration <it> row of lio_update_diag_compact.csv.gz as a dict (HtH_ij, Htz_i), or None."""
    import gzip
    p = os.path.join(runs, arm, "lio_update_diag_compact.csv.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        for r in csv.DictReader(f):
            if int(float(r["frame"])) == frame and int(float(r["iter"])) == it:
                return {k: float(v) for k, v in r.items() if k.startswith("HtH_") or k.startswith("Htz_")}
    return None


def max_rel(a, b, keys, scale=None):
    """max over keys of |a-b| / max(|b|, tiny); scale(key) multiplies b first (expected ratio)."""
    m = 0.0
    for k in keys:
        e = b[k] * (scale(k) if scale else 1.0)
        m = max(m, abs(a[k] - e) / max(abs(e), 1e-300))
    return m


def parse_prep(path):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        p = ln.split()
        if len(p) == 2:
            d[p[0]] = p[1]
        elif len(p) == 1:
            d.setdefault("_flags", []).append(p[0])
        if p and p[0] == "observation":
            d["n_obs_lines"] = d.get("n_obs_lines", 0) + 1
    return d


def summarize(arm_dir):
    pre = os.path.join(arm_dir, "bootstrap_summary.json")
    if os.path.exists(pre):   # R69: written by the scorer while run.log still existed
        try:
            return json.load(open(pre))
        except Exception:
            pass
    d = parse_prep(os.path.join(arm_dir, "bootstrap_preparation.txt"))
    fb = False
    rl = os.path.join(arm_dir, "run.log")
    if os.path.exists(rl):
        fb = "poses-empty fallback used" in open(rl, errors="replace").read()
    dg = os.path.join(arm_dir, "state_trace_blank_runid.sha256")
    return {"arm": os.path.basename(os.path.abspath(arm_dir)), "schema": d.get("schema"),
            "hash": d.get("flattened_prepared_population_hash"), "total_prepared": d.get("total_prepared_point_count"),
            "tr_mean": d.get("sensor_cov_trace_over_3_mean"), "tr_min": d.get("sensor_cov_trace_over_3_min"),
            "tr_max": d.get("sensor_cov_trace_over_3_max"), "n_zero": d.get("sensor_cov_zero_point_count"),
            "deskew_fallback": fb, "digest": open(dg).read().strip() if os.path.exists(dg) else None}


def collect(runs, out, arm_table=None):
    os.makedirs(out, exist_ok=True)
    rows = [summarize(os.path.join(runs, a)) for a in sorted(os.listdir(runs)) if os.path.isdir(os.path.join(runs, a)) and not a.startswith("r71_refuse")]
    with open(os.path.join(out, "bootstrap_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {r["arm"]: r for r in rows}
    L = []
    def gate(name, ok, why=""):
        L.append("%s  %s  %s" % ("PASS" if ok else "FAIL", name, why))
    for base, dig in R67_DIGEST.items():
        r = by.get(base)
        if r is None:
            gate("G1 %s present" % base, False, "arm missing"); continue
        gate("G1 %s digest == R67 sensor_model digest" % base, r["digest"] == dig, "got %s" % r["digest"])
    # G2 (R69, fixes the R68 defect F-314): schema v2 and zero sensor_cov points for EVERY arm; the prepared-population hash must equal the
    # R67 value for eee_01 default sampling (pfn 3, ds first) and must be identical within each (seq, sampling) group for every other group.
    samp = {}
    if arm_table and os.path.exists(arm_table):
        for r in csv.DictReader(open(arm_table), delimiter="\t"):
            samp[r["arm"]] = (r["seq"], r.get("sampling", ""))
    bad = [r["arm"] for r in rows if not r["arm"].startswith("_") and not (r["schema"] == "bootstrap_preparation_v2" and r["n_zero"] == "0")]   # R71: skip non-arm directories such as _matrices
    gate("G2a every arm: schema v2, sensor_cov_zero_point_count 0", not bad, "failing arms: %s" % bad[:20])
    grp = {}
    for r in rows:
        grp.setdefault(samp.get(r["arm"], ("?", "?")), []).append(r)
    for k, rs in sorted(grp.items()):
        hs = sorted(set(x["hash"] for x in rs))
        if k == ("eee_01", "pfn3_dsfirst"):
            gate("G2b %s/%s: hash == R67 value (%d arms)" % (k[0], k[1], len(rs)), hs == [R67_HASH], "hashes %s" % hs)
        else:
            gate("G2b %s/%s: one hash within the group (%d arms)" % (k[0], k[1], len(rs)), len(hs) == 1, "hashes %s" % hs)
    def same(g, a_, b_):
        x, y = by.get(a_), by.get(b_)
        gate("%s %s digest == %s digest" % (g, a_, b_), x is not None and y is not None and x["digest"] is not None and x["digest"] == y["digest"],
             "got %s / %s" % (x and x["digest"], y and y["digest"]))
    same("G3", "00_id_default_posecov_60s", "00_id_baseline_60s")
    same("G4", "00_id_infl_whole_f1_60s", "00_id_baseline_60s")
    same("G5", "00_id_baseline_full_rerun", "00_id_baseline_full")
    same("G8a", "r71_diag_on_baseline_60s", "00_id_baseline_60s")
    same("G8b", "r71_mx_baseline_fq_60s", "00_id_baseline_60s")
    same("G8c", "r71_smoke_dump_on_baseline_60s", "00_id_baseline_60s")
    for arm, (prev_arm, dig) in R69B_DIGEST.items():
        r = by.get(arm)
        gate("G11 %s digest == R69B %s digest (R70 changes to this path are write-only or inert)" % (arm, prev_arm), r is not None and r["digest"] == dig,
             "got %s" % (r and r["digest"]))
    for arm, (prev_arm, dig) in R70_DIGEST.items():
        r = by.get(arm)
        gate("G11b %s digest == R70 %s digest (write-only logs; not a stop condition)" % (arm, prev_arm), r is not None and r["digest"] == dig, "got %s" % (r and r["digest"]))
    # G9 (F-320): the block information scale against the global scale, frame 1 iteration 0 (same map, same residuals, nothing has diverged yet)
    hk = ["HtH_%d%d" % (i, j) for i in range(6) for j in range(i, 6)]
    zk = ["Htz_%d" % i for i in range(6)]
    blk, glob, one = (update_row(runs, x) for x in ("r71_smoke_block_f10_10_60s", "r71_mx_cb_fq_f10_60s", "r71_mx_cb_fq_f1_60s"))
    if blk and glob and one:
        eh, ez = max_rel(blk, glob, hk), max_rel(blk, glob, zk)
        gate("G9a block 10/10 == global F=10 at frame 1 iteration 0 (HtH relative < 1e-9, Htz relative < 1e-6)", eh < 1e-9 and ez < 1e-6, "HtH %.3e Htz %.3e" % (eh, ez))
        ctl = max_rel(blk, one, hk)
        gate("G9c control: block 10/10 against the F=1 arm must DIFFER by a wide margin (the comparison can fail)", ctl > 0.5, "HtH relative difference %.3e" % ctl)
    else:
        gate("G9a/G9c need frame-1 rows of r71_smoke_block_f10_10_60s, r71_mx_cb_fq_f10_60s, r71_mx_cb_fq_f1_60s", False, "missing: %s" % [n for n, v in (("block", blk), ("global", glob), ("one", one)) if not v])
    b122 = update_row(runs, "r71_smoke_block_f12_2_60s")
    if b122 and one:
        def ratio(k):
            i, j = int(k[4]), int(k[5])
            fi = 12.0 if i < 3 else 2.0
            fj = 12.0 if j < 3 else 2.0
            return 1.0 / ((fi * fj) ** 0.5)
        eh = max_rel(b122, one, hk, ratio)
        gate("G9b block 12/2: HtH = S HtH(F=1) S at frame 1 iteration 0 (relative < 1e-5: the CSV prints 7 significant digits; R70 measured 5.5e-7)", eh < 1e-5, "HtH %.3e" % eh)
    else:
        gate("G9b needs frame-1 rows of r71_smoke_block_f12_2_60s and r71_mx_cb_fq_f1_60s", False, "missing")
    # G13: no engagement.txt of any arm may carry an INERT flag
    inert = []
    for arm in sorted(by):
        ep = os.path.join(runs, arm, "engagement.txt")
        if os.path.exists(ep) and "<-- INERT" in open(ep, errors="replace").read():
            inert.append(arm)
    gate("G13 no arm's engagement.txt carries an INERT flag", not inert, "arms: %s" % inert[:20])
    # G13a (R71, F-350): the iekf destructor now writes engagement.txt; if it is missing for an arm that ran to completion, G13 would again test nothing
    noeng = [arm for arm in sorted(by) if not arm.startswith("_") and not os.path.exists(os.path.join(runs, arm, "engagement.txt"))]
    gate("G13a every arm directory has an engagement.txt (G13 is not vacuous)", not noeng, "missing in %d arms: %s" % (len(noeng), noeng[:20]))
    for arm, txt in REFUSE.items():
        d = os.path.join(runs, arm)
        txts = txt if isinstance(txt, list) else [txt]
        have = any(os.path.exists(os.path.join(d, f)) and all(x in open(os.path.join(d, f), errors="replace").read() for x in txts) for f in ("run.log", "run_log_tail.txt"))
        gate("G6 %s refused at startup (run.log contains %r)" % (arm, txt), have, "run dir %s" % ("present" if os.path.isdir(d) else "missing"))
    fb = [r["arm"] for r in rows if r["deskew_fallback"]]
    L.append("INFO deskew poses-empty fallback used in %d arms: %s" % (len(fb), fb[:20]))
    open(os.path.join(out, "identity_gates.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not any(l.startswith("FAIL") for l in L) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("arm_dir", nargs="?"); ap.add_argument("--collect"); ap.add_argument("--out", default="R71_RETURN")
    ap.add_argument("--arm-table", default=None)
    a = ap.parse_args()
    if a.collect:
        sys.exit(collect(a.collect, a.out, a.arm_table))
    s = summarize(a.arm_dir); json.dump(s, open(os.path.join(a.arm_dir, "bootstrap_summary.json"), "w"), indent=1)
