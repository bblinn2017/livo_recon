#!/usr/bin/env python3
"""R79: build the job files, arm_table.tsv, listener specs, required_outputs.json and the size ledger from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r79_manifest.py --jobs r69_inputs/r66_jobs.json --out r79_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (only the counts / key checks in tools/r69/r79_gates.py --selftest and the check at the end of this file).
Reuses the R74 helpers (make_r74_manifest as M), the R77 line-mode tables (make_r77_manifest as R7) and the R78 constants (make_r78_manifest as R8): the same R66 base jobs, the same overrides.

Bryce 2026-10-05 (his words are in each arm's `asked` field and in COMPLIANCE_TABLE.md):
  F-463  0.35 m against 0.425 m on all nine bags: lines off and hard absolute (plane_equivalent), both Q, every 3rd point, whole bag: 9 x 2 x 2 x 2 = 72 arms (the 0.425 m arms are the baselines).
  F-401 / F-418  ZUPT mid (apply, gyro 0.02, acc 0.15, sigma_v 0.02, enter_scans 3) on both Q modes and NIS gate hi 1.3 on motion-Q, nine bags, 0.425 m, every 3rd, lines off: 3 x 9 = 27 arms.
  F-437  eee_02 under motion-Q: the lines-off 0.425 m fixed-Q and motion-Q arms inside F-463, with the motion-Q trace (motion_q_scan.csv) returned for the motion-Q arm.
  F-464  why the split (all points to the map, every 3rd to the IEKF) is worse than every 3rd everywhere: eee_01, lines off, 0.425 m, both Q, every-3rd-everywhere and split = 4 arms, 60 s,
         residual / first-frame dumps and a voxel-map snapshot at each of frames 1 to 5.
  F-441 / F-440  eee_01, 0.425 m, calibration samples 200 and 400, pfn 1 and 3, fixed-Q = 4 short arms, the same dumps and per-frame snapshots.
  Quaternion columns qw, qx, qy, qz in pos_trace_all.csv.gz (tools: r74_frame_errors.pos_trace).
Batches (queues):
  r79_id   5 arms (EF cell n1600 ds off pfn3 fq, 17 s): 3 identity arms, ONE determinism repeat, 1 smoke arm of the new snapshot key.
  r79_ref  3 refuse arms (5 s each, MUST FAIL at startup naming the key).
  r79_ef   8 early-frame arms (F-464 60 s, F-441 / F-440 10 and 11 s).
  r79_fb   100 whole-bag arms: 72 (F-463) + 27 (F-401 / F-418) + ONE determinism repeat.
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402
import make_r77_manifest as R7  # noqa: E402
import make_r78_manifest as R8  # noqa: E402

P = R7.P
VAR = R7.VAR
GAP = R7.GAP
LINE76 = R7.LINE76
WANT = R7.WANT
ENUM = R7.ENUM
DESC = R7.DESC
LOGKEY = R7.LOGKEY
CELL = R7.CELL
ID_DUR = R7.ID_DUR
OWN = R7.OWN
QS = ("fq", "mq")
BAGS = M.BAGS
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # bag_params.json n_scans
STREAM = R8.STREAM
EDGES = (0.35, 0.425)
LINES = ("off", "hardabs")
ZUPT_MID = M.ZUPT("apply")                       # gyro 0.02, acc 0.15, enter_scans 3, sigma_v 0.02 (the R74 "mid" lever)
NIS13 = M.GAIN("nis_gate", nis_hi=1.3)
F5DUMP = {"lio.first_frame_dump_frames": 5, "lio.residual_dump_frames": "1,2,3,4,5", "lio.update_diag_en": True}
SNAP5 = {"voxel_map.map.snapshot_post_calibration": True, "voxel_map.map.snapshot_frames": "2,3,4,5"}
SNAP_FRAMES_N = (1, 2, 3, 4, 5)
DX_DUR = 60
EC_N = (200, 400)
MQ_TRACE_ARM = "r79_nb_eee_02_off_v0p425_pfn3_mq"   # F-437: the one arm whose motion_q_scan.csv is returned (r69_collect.KEEP_IF_ARM_HAS)
IDENT = [("r79_id_off_v1p0", "off", 1.0, "r78_id_off_v1p0", {}),
         ("r79_id_hardabs_v0p425", "hardabs", 0.425, "r78_id_hardabs_v0p425", {}),
         ("r79_id_second_off_v1p0", "off", 1.0, "r78_id_second_off_v1p0", {"lio.dry_run_point_filter_num": 3, "lio.iekf_point_stream": "second"})]
ID_DET, ID_DET_ORIG = "r79_own_det_id_off_v1p0", "r79_id_off_v1p0"
SMOKE_ID = "r79_sm_snap_off_v0p425"
FB_DET_ORIG, FB_DET = "r79_nb_eee_03_pe_hardabs_v0p35_pfn3_mq", "r79_own_det_nb_eee_03_pe_hardabs_v0p35_pfn3_mq"
REFUSE = [("snapshot_frames_no_postcal", {"voxel_map.map.snapshot_frames": "2,3"}, ["voxel_map/map/snapshot_frames needs voxel_map/map/snapshot_post_calibration"], "00_id_baseline_60s"),
          ("snapshot_frames_one", {"voxel_map.map.snapshot_post_calibration": True, "voxel_map.map.snapshot_frames": "1,2"}, ["voxel_map/map/snapshot_frames: frame 1 is below 2"], "00_id_baseline_60s"),
          ("snapshot_frames_bogus", {"voxel_map.map.snapshot_post_calibration": True, "voxel_map.map.snapshot_frames": "2,x"}, ["voxel_map/map/snapshot_frames: 'x' is not an integer"], "00_id_baseline_60s")]
# whole-bag arm -> the R78 arm (eee_01, same cell) whose per-scan position and yaw it must reproduce exactly
IDREF_NB = {}
for _q in QS:
    IDREF_NB["r79_nb_eee_01_off_v0p425_pfn3_%s" % _q] = "r78_ct_off_v0p425_pfn3_%s" % _q
    IDREF_NB["r79_nb_eee_01_pe_hardabs_v0p425_pfn3_%s" % _q] = "r78_ct_pe_hardabs_v0p425_pfn3_%s" % _q
    IDREF_NB["r79_nb_eee_01_off_v0p35_pfn3_%s" % _q] = "r78_sa_off_v0p35_gap0_%s" % _q
    IDREF_NB["r79_nb_eee_01_pe_hardabs_v0p35_pfn3_%s" % _q] = "r78_sa_pe_hardabs_v0p35_gap0_%s" % _q
IDREF_DX = {}
for _q in QS:
    IDREF_DX["r79_dx_off_v0p425_pfn3_%s" % _q] = "r78_ct_off_v0p425_pfn3_%s" % _q
    IDREF_DX["r79_dx_split_off_v0p425_%s" % _q] = "r78_sp_off_v0p425_%s" % _q
DX_LAST_EXCLUDED = 3   # scans at the end of a 60 s arm that the identity gate does not decide on (the run stops at the duration; reported as INFO)
EXPECTED = {"residual_dump_all.csv.gz": 14000, "r73_f417_hth_frames.csv": 2000, "pos_trace_all.csv.gz": 56, "line_rows_per_scan_all.csv.gz": 0, "snapshot_frames_all.csv.gz": 0,
            "state_trace_early_all.csv.gz": 15000, "lio_update_diag_early_all.csv.gz": 8000, "residual_dump_state_all.csv.gz": 3000, "motion_q_scan_all.csv.gz": 160000}
MAXFILE = {"residual_dump_all.csv.gz": 3000000, "r73_f417_hth_frames.csv": 100000, "pos_trace_all.csv.gz": 330000, "line_rows_per_scan_all.csv.gz": 250000, "snapshot_frames_all.csv.gz": 1500000,
           "state_trace_early_all.csv.gz": 100000, "lio_update_diag_early_all.csv.gz": 100000, "residual_dump_state_all.csv.gz": 100000, "motion_q_scan_all.csv.gz": 400000}
POS_COLS = M.POS_COLS + ["qw", "qx", "qy", "qz"]
DUMP_COLS = M.DUMP_COLS + ["row_kind"]
LINE_ROWS_COLS = R7.LINE_ROWS_COLS
SNAP_COLS = ["map_frame"] + R8.SNAP_COLS[:]
STATE_EARLY_COLS = ["run_id", "scan_id", "t_abs", "phase", "qw", "qx", "qy", "qz", "px", "py", "pz", "vx", "vy", "vz", "P_0_0", "P_17_17"]
UPD_EARLY_COLS = ["frame", "iter", "n_resid", "HtH_00", "HtH_55", "Htz_0", "dtheta_x", "dt_x"]

ASK_463 = ("Bryce (2026-10-05): 'it seems like 0.35 is a contender for best ATE. we should try that over all bags.'; his answers: 'Lines off + hard absolute, both Q (Recommended)' = voxel edge 0.35 m against 0.425 m "
           "on all nine bags, lines off and hard absolute (plane_equivalent), fixed-Q and motion-Q, every 3rd point (pfn 3), whole bag; the same-batch 0.425 m arms are the baselines")
ASK_ALL = "Bryce (2026-10-05): 'i would like these experiments, i'd also like all our other open followups.'; his answer on scope: 'all of them'"
ASK_437 = ASK_ALL + "; F-437 (eee_02 under motion-Q: ATE 52.57 mm against 19.87 mm fixed-Q): the eee_02 lines-off 0.425 m fixed-Q and motion-Q arms of F-463, the motion-Q trace (motion_q_scan.csv) returned"
ASK_401 = (ASK_ALL + "; his answer on the motion-Q alternatives: 'ZUPT mid + NIS 1.3, nine bags, 0.425 m (Recommended)' = ZUPT mid (apply, gyro 0.02 rad/s, acc 0.15 m/s2, sigma_v 0.02 m/s, enter_scans 3) on both Q modes "
           "and the NIS gate with nis_hi 1.3 on motion-Q only, nine bags, 0.425 m, every 3rd point, lines off (controls: the F-463 lines-off 0.425 m arms of the same bag)")
ASK_464 = ("Bryce (2026-10-05): 'i still don't quite understand why using the same point set for the iekf but using all points for the map performs so much worse. i'd like to just look at those two settings (every third "
           "point always and every third point for iekf and all for the map and see how the map and the residuals and the covariance changes, perhaps we check the early frames first and also determine where the "
           "trajectories really diverge)'; his answers: 'Lines off 0.425 m, both Q (Recommended)', 'Frames 1 to 5 only, map at every frame', 'First 60 s (Recommended)'")
ASK_441 = (ASK_ALL + "; F-441 (why calibration samples 200 yields few planes) and F-440 (facade points labelled ground): his answer on the grid: 'eee_01, 0.425 m, samples 200/400, pfn 1/3, fixed-Q (Recommended)'; "
           "per-node snapshot plus the 5-frame residual dump joined by node_id (F-440's 'per-point residual of the fitted plane' is read as the voxel's smallest eigenvalue from the snapshot; per-point residuals would need new C++)")


def tagf(x):
    return M.tag(x)


def mkey(mode):
    return "off" if mode == "off" else "pe_" + mode


def nbarm(aid, bag, mode, v, q, group, asked, purpose, pfn=3, layer=2, extra=None, dump=None, dur=None, early=0, stream=False, dump5=False, snap5=False, tier="FB", batch="fb", attr=None):
    """One whole-bag (or duration-limited) arm: the R66 base job of the bag and Q mode plus voxel / layer / line mode (plane_equivalent when lines are on) / point filter and the optional keys."""
    d = dict(M.VOX(v, layer), **LINE76[mode], **LOGKEY, **{"cbk.lidar.point_filter_num": pfn})
    if mode != "off":
        d[VAR] = "plane_equivalent"
    if stream:
        d.update(STREAM)
    if extra:
        d.update(extra)
    if dump is None:
        dump = (mode != "off")
    if dump5:
        d.update(F5DUMP)
    elif dump:
        d.update(M.F1DUMP)
    if snap5:
        d.update(SNAP5)
    has_dump = bool(dump or dump5)
    a = dict(id=aid, base=M.base_of(bag, q), group=group, tier=tier, batch=batch, dur=dur, mode=mode, voxel=v, kind="fb", ref=None, delta=d, f417=1 if has_dump else 0, asked=asked, purpose=purpose,
             layer=layer, gap=False, stream="second" if stream else "primary", info3=False, pfn=(1 if stream else pfn), q=q, pe=(mode != "off"), dump=(5 if dump5 else (1 if dump else 0)),
             bag=bag, early=early, snap5=snap5, zupt=False, nis=False)
    if attr:
        a.update(attr)
    return a


def arms():
    A = []
    # ---- OWN: identity, determinism, smoke, refuse (standing rules 3.4, 3.6, 3.7)
    for aid, mode, v, ref, extra in IDENT:
        A.append(dict(id=aid, base=M.base_of("eee_01", "fq"), group="IDN", tier="EF", batch="id", dur=ID_DUR, mode=mode, voxel=v, kind="id", ref=ref,
                      delta=dict(M.VOX(v, 2), **LINE76[mode], **LOGKEY, **M.F1DUMP, **CELL, **extra), f417=1,
                      asked=OWN + "%s, root voxel %s, EF cell n1600 ds off pfn3 fq; %s; the state must reproduce %s exactly" % (
                          DESC[mode], v, "every new key at its default" if not extra else "lio/iekf_point_stream = second with lio/dry_run_point_filter_num 3 and the primary stream also pfn 3: the second stream equals the primary set",
                          ref),
                      purpose="R79 identity: %s at voxel %s%s vs %s" % (mode, v, " with the second stream equal to the primary one" if extra else " with the new keys at default", ref),
                      layer=2, gap=False, stream="second" if extra else "primary", info3=False, pfn=3, q="fq", pe=False, dump=1, bag="eee_01", early=0, snap5=False, zupt=False, nis=False))
    A.append(dict(id=ID_DET, base=M.base_of("eee_01", "fq"), group="DET", tier="EF", batch="id", dur=ID_DUR, mode="off", voxel=1.0, kind="det_id", ref=ID_DET_ORIG,
                  delta=dict(M.VOX(1.0, 2), **LINE76["off"], **LOGKEY, **M.F1DUMP, **CELL), f417=1,
                  asked=OWN + "rule 3.7 determinism control: an exact repeat of %s; its state-trace digest must equal that arm's" % ID_DET_ORIG,
                  purpose="OWN determinism control: repeat of %s" % ID_DET_ORIG, layer=2, gap=False, stream="primary", info3=False, pfn=3, q="fq", pe=False, dump=1, bag="eee_01", early=0, snap5=False, zupt=False, nis=False))
    A.append(dict(id=SMOKE_ID, base=M.base_of("eee_01", "fq"), group="SMK", tier="EF", batch="id", dur=ID_DUR, mode="off", voxel=0.425, kind="smoke", ref=None,
                  delta=merged(M.VOX(0.425, 2), LINE76["off"], LOGKEY, F5DUMP, SNAP5, CELL), f417=1,
                  asked=OWN + "rule 3.6 smoke arm for the new key voxel_map/map/snapshot_frames: EF cell, lines off, root voxel 0.425, snapshot_post_calibration plus snapshot_frames 2,3,4,5, residual dump frames 1 to 5; "
                           "the snapshot file of each of frames 1 to 5 must exist, hold nodes, and every node_id of the dump of frame N must be in the snapshot of frame N",
                  purpose="R79 smoke: per-frame snapshots (frames 1 to 5) and the 5-frame dump, EF cell, voxel 0.425", layer=2, gap=False, stream="primary", info3=False, pfn=3, q="fq", pe=False, dump=5,
                  bag="eee_01", early=1, snap5=True, zupt=False, nis=False))
    for suf, delta, needles, base in REFUSE:
        A.append(dict(id="r79_refuse_%s" % suf, base=base, group="REF", tier="REF", batch="ref", dur=5, mode=None, voxel=None, kind="refuse", ref=None, needles=needles, delta=dict(delta), f417=0,
                      asked=OWN + "rule 3.4 refusal: the contradictory or dead setting MUST FAIL at startup, text contains %s" % " and ".join(repr(n) for n in needles),
                      purpose="MUST FAIL: %s" % suf, layer=None, gap=False, stream="primary", info3=False, pfn=None, q=None, pe=False, dump=0, bag="eee_01", early=0, snap5=False, zupt=False, nis=False))
    # ---- F-463 (72): 9 bags x edge {0.35, 0.425} x Q x lines {off, hard absolute pe}, every 3rd point, whole bag
    for bag in BAGS:
        for e in EDGES:
            for q in QS:
                for mode in LINES:
                    aid = "r79_nb_%s_%s_v%s_pfn3_%s" % (bag, mkey(mode), tagf(e), q)
                    ask = ASK_463
                    if aid == MQ_TRACE_ARM or aid == MQ_TRACE_ARM.replace("_mq", "_fq"):
                        ask = ASK_463 + ". Also (F-437) " + ASK_437
                    A.append(nbarm(aid, bag, mode, e, q, "NB", "%s: %s, edge %s m, %s Q, %s%s" % (ask, bag, e, "fixed" if q == "fq" else "motion", DESC[mode], " with plane_equivalent variance" if mode != "off" else ""),
                                   "R79 F-463: %s, edge %s, %s, pfn 3, %s" % (bag, e, mode, q), pfn=3))
    # ---- F-401 / F-418 (27): ZUPT mid on both Q, NIS 1.3 on motion-Q, nine bags, 0.425, every 3rd, lines off
    for bag in BAGS:
        for nm, q, ex, key, desc in (("zupt", "fq", ZUPT_MID, "zupt", "ZUPT mid (apply, gyro 0.02, acc 0.15, sigma_v 0.02, enter_scans 3)"),
                                      ("zupt", "mq", ZUPT_MID, "zupt", "ZUPT mid (apply, gyro 0.02, acc 0.15, sigma_v 0.02, enter_scans 3)"),
                                      ("nis13", "mq", NIS13, "nis", "NIS gate with nis_hi 1.3")):
            aid = "r79_qv_%s_%s_%s" % (bag, nm, q)
            a = nbarm(aid, bag, "off", 0.425, q, "QV", "%s: %s, %s Q, %s, edge 0.425 m, every 3rd point, lines off" % (ASK_401, bag, "fixed" if q == "fq" else "motion", desc),
                      "R79 F-401/F-418: %s on %s, %s" % (nm, bag, q), pfn=3, extra=ex)
            a["zupt"] = (key == "zupt"); a["nis"] = (key == "nis")
            A.append(a)
    d = nbarm(FB_DET, "eee_03", "hardabs", 0.35, "mq", "DET", OWN + "rule 3.7 determinism control: an exact whole-bag repeat of %s; its state-trace digest must equal that arm's" % FB_DET_ORIG,
              "OWN determinism control: whole-bag repeat of %s" % FB_DET_ORIG, pfn=3)
    d["kind"] = "det_fb"; d["ref"] = FB_DET_ORIG
    A.append(d)
    # ---- F-464 (4): every 3rd everywhere and the split, lines off, 0.425 m, both Q, first 60 s, frames 1 to 5 in full, map snapshot at each of frames 1 to 5
    for q in QS:
        A.append(nbarm("r79_dx_off_v0p425_pfn3_%s" % q, "eee_01", "off", 0.425, q, "DX",
                       "%s: every 3rd point everywhere (point filter 3), lines off, edge 0.425 m, %s Q, first %d s; residual dump, first-frame solve dump (state, HtH, covariance) and a map snapshot at each of frames 1 to 5" % (
                           ASK_464, "fixed" if q == "fq" else "motion", DX_DUR),
                       "R79 F-464: every 3rd everywhere, %s, 60 s, frames 1-5 dumps and snapshots" % q, pfn=3, dur=DX_DUR, dump5=True, snap5=True, early=1, tier="EF", batch="ef"))
        A.append(nbarm("r79_dx_split_off_v0p425_%s" % q, "eee_01", "off", 0.425, q, "DX",
                       "%s: the split (map from every point, point filter 1; IEKF from the every-3rd set of the second stream), lines off, edge 0.425 m, %s Q, first %d s; the same dumps and snapshots" % (
                           ASK_464, "fixed" if q == "fq" else "motion", DX_DUR),
                       "R79 F-464: split (map every point, IEKF every 3rd), %s, 60 s, frames 1-5 dumps and snapshots" % q, pfn=1, stream=True, dur=DX_DUR, dump5=True, snap5=True, early=1, tier="EF", batch="ef"))
    # ---- F-441 / F-440 (4): eee_01, 0.425 m, calibration samples 200 / 400, pfn 1 / 3, fixed-Q, lines off, the same dumps and snapshots
    for ns in EC_N:
        dur = int(math.ceil(ns / 197.5 + 8.0))
        for pfn in (1, 3):
            ex = {"calib.stationary.num_samples": ns}
            A.append(nbarm("r79_ec_n%d_pfn%d_fq" % (ns, pfn), "eee_01", "off", 0.425, "fq", "EC",
                           "%s: calibration samples %d, point filter %d, edge 0.425 m, lines off, fixed-Q, imu.ds.mode first (the base job's setting; Bryce named no ds mode), %d s by the R72 formula ceil(ns/197.5+8); "
                           "residual dump of frames 1 to 5 and a map snapshot at each of frames 1 to 5 (post-calibration snapshot included)" % (ASK_441, ns, pfn, dur),
                           "R79 F-441/F-440: samples %d, pfn %d, 0.425, fq, frames 1-5 dumps and snapshots" % (ns, pfn), pfn=pfn, extra=ex, dur=dur, dump5=True, snap5=True, early=1, tier="EF", batch="ef"))
    return A


def merged(*ds):
    o = {}
    for x in ds:
        o.update(x)
    return o


def req_of(o, batch):
    r = M.requires_of(o, batch)
    if o.get("lio.line_rows_log_en"):
        r = r + ["line_rows_per_scan.csv"]
    if o.get("voxel_map.map.snapshot_frames"):
        r = r + ["voxelmap_snapshot_f%d.csv" % n for n in SNAP_FRAMES_N[1:]]
    return r


def dump_bytes(t):
    """expected gz bytes of the frames 1-5 residual dump (R78 yw arms, pfn 3: 419 KB; hard absolute 665 KB; pfn 1 about three times the rows; calibration samples 200: few residuals)."""
    if "_n200_" in t["arm"]:
        return 150000
    base = 665000 if t["pe"] == "1" else 420000
    return base * (3 if t["pfn"] == "1" and "_split_" not in t["arm"] else 1)


def required_outputs(table):
    sel = lambda f: [t for t in table if f(t)]
    byarm = lambda L: [t["arm"] for t in L]
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", expected_per=None, maxb=None):
        if where == "return" and arms_ and expected_per is not None:
            eb = int(sum(expected_per(a) for a in arms_))
        elif where == "return":
            eb = int(EXPECTED.get(file, 50000) * max(1, len(arms_)))
        else:
            eb = 0
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=eb,
                      max_bytes_per_file=maxb or MAXFILE.get(file, 400000)))
    by = {t["arm"]: t for t in table}
    nonref = [t for t in table if t["group"] != "REF"]
    fb = sel(lambda t: t["batch"] == "r79_fb")
    ident = sel(lambda t: t["group"] in ("IDN", "SMK") or t["arm"] == ID_DET)
    dumpers1 = sel(lambda t: t["f417"] == "1" and t["dump"] != "5")
    dumpers5 = sel(lambda t: t["dump"] == "5")
    early = sel(lambda t: t["early"] == "1")
    add("F-463", "per-scan posterior position, yaw and quaternion of every whole-bag arm (scan_id key)", byarm(fb), "pos_trace_all.csv.gz", POS_COLS, 1000, expected_per=lambda a: 56 * N_SCANS[by[a]["bag"]], maxb=330000)
    add("F-463", "per-scan line-row log (scans <= 800 plus every 10th), every whole-bag arm", byarm(fb), "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 100,
        expected_per=lambda a: (800 + max(N_SCANS[by[a]["bag"]] - 800, 0) / 10.0) * (47 if by[a]["pe"] == "1" else 14))
    add("F-463", "frame-1 residual dump with row_kind, the hard absolute arms (whole bag) and the identity arms", byarm([t for t in dumpers1 if t["group"] in ("NB", "DET", "IDN")]), "residual_dump_all.csv.gz", DUMP_COLS, 50,
        expected_per=lambda a: 14000)
    add("F-463", "residual report (HtH / Htz metrics) per dumped frame, the arms that dump frame 1", byarm([t for t in dumpers1 if t["group"] in ("NB", "DET", "IDN")]), "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1,
        expected_per=lambda a: 2000)
    add("F-464", "residual dump of frames 1 to 5 (with row_kind, weight, class, rotation-about-z row of the Jacobian) of the F-464, F-441 / F-440 and smoke arms", byarm(dumpers5), "residual_dump_all.csv.gz", DUMP_COLS, 500,
        expected_per=lambda a: dump_bytes(by[a]), maxb=3000000)
    add("F-464", "HtH / Htz report per dumped frame, frames 1 to 5", byarm(dumpers5), "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 5, expected_per=lambda a: 4000)
    add("F-464", "residual-dump state (state before / after each iteration) of frames 1 to 5", byarm(dumpers5), "residual_dump_state_all.csv.gz", [], 5, expected_per=lambda a: 3000)
    add("F-464", "every voxel-map snapshot of frames 1 to 5 (map_frame 1 = post-calibration; one row per node: point count, eigenvalues, plane test values, centre, normal), joined to the dump by node_id", byarm(early), "snapshot_frames_all.csv.gz", SNAP_COLS, 300,
        expected_per=lambda a: (60000 if by[a]["pfn"] == "1" else 40000) * 5)
    add("F-464", "state, covariance (18 x 18 upper triangle) of both phases, scans 0 to 10", byarm(early), "state_trace_early_all.csv.gz", STATE_EARLY_COLS, 10, expected_per=lambda a: 15000)
    add("F-464", "HtH, Htz, step per iteration, frames 1 to 10", byarm(early), "lio_update_diag_early_all.csv.gz", UPD_EARLY_COLS, 5, expected_per=lambda a: 8000)
    add("F-464", "per-scan position, yaw and quaternion of the early-frame arms (alignment-free divergence of the F-464 arms)", byarm(early), "pos_trace_all.csv.gz", POS_COLS, 5, expected_per=lambda a: 56 * (600 if by[a]["duration_s"] == "60" else 100))
    add("F-464", "per-scan line-row log of the early-frame arms", byarm(early), "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 5, expected_per=lambda a: 20000)
    add("F-437", "motion-Q trace (motion_q_scan.csv, every row) of the eee_02 lines-off 0.425 m motion-Q arm", [MQ_TRACE_ARM], "motion_q_scan_all.csv.gz", ["scan_index", "t_rel_s", "model", "acc_scale", "gyro_scale", "mean_q_acc_x"], 2000,
        expected_per=lambda a: 160000, maxb=400000)
    add("F-439", "per-scan position, yaw and quaternion of every identity arm", byarm(ident), "pos_trace_all.csv.gz", POS_COLS, 20, expected_per=lambda a: 56 * 126)
    add("F-452", "per-scan line-row log, identity and smoke arms", byarm(ident), "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 50, expected_per=lambda a: 14000)
    add("F-452", "gates: identity vs R78, determinism digests, refusals, effective config (voxel, layer, line, variance, gap floor, stream, ZUPT, NIS, information scale), engagement, quaternion check, snapshot join, early files", [], "r79_gates.txt", [], 1, source="generated")
    add("F-452", "identity comparison of every identity arm, the eight eee_01 F-463 arms and the four F-464 arms against their R78 counterparts (per-scan max abs difference of position and yaw)", [], "r79_identity_vs_old.csv",
        ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"], 15, source="generated")
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated (includes the ZUPT, q_gain and iekf_point_stream lines)", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 50, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm that is not a refuse arm", byarm(nonref), "engagement_imu.txt", [], 1, where="archive")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"] or e["source"] == "generated":
            continue
        new = e["file"] in ("snapshot_frames_all.csv.gz", "state_trace_early_all.csv.gz", "lio_update_diag_early_all.csv.gz", "residual_dump_state_all.csv.gz") or (e["file"] == "residual_dump_all.csv.gz" and e["min_rows"] >= 500)
        mult = 3 if new else 1.5
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * mult),
                    "basis": ("no measurement of the real file yet (estimate from the R78 data: snapshot 27-30 KB gz per 0.425 m node table; 5-frame dump 0.42 MB (pfn 3) and 0.67 MB (hard absolute) from the R78 yw arms); the agent "
                              "measures one arm of each with size_ledger.py measure; allowance 3x" if new else
                              "measured on the R78 return: pos_trace 137 KB per 3,980 scans plus 86 KB for the four quaternion columns (56 bytes per scan), line rows 15 KB (lines off) / 52 KB (hard absolute) per eee_01 arm; allowance 1.5x"),
                    "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    for pat, mx, tot in (("r73_f417_classes.csv", 20000, 4000000), ("r73_f417_consistency.csv", 5000, 1000000), ("r79_gates.txt", 900000, 900000), ("identity_controls.txt", 150000, 150000),
                         ("r79_identity_vs_old.csv", 40000, 40000), ("engagement_all.txt", 700000, 700000), ("completeness.csv", 300000, 300000),
                         ("phase_summary.csv", 4000000, 4000000), ("arm_scan_all.csv.gz", 40000000, 40000000), ("raw_sha256.txt", 20000, 20000),
                         ("R79_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000), ("listener_*.log", 5000000, 20000000), ("*.md", 1000000, 3000000),
                         ("*.json", 2000000, 10000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000), ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74-R78 ledger values or a round figure); not a measurement", "note": "non-arm file"})
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]
            y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r79_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {"id": [], "ref": [], "ef": [], "fb": []}; table = []; seen = set()
    for r in arms():
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 to R78)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r79_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        req = req_of(o, "refuse" if r["batch"] == "ref" else "x")
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r79_" + r["batch"], sampling=sampling, requires=" ".join(req),
                          f417=str(r["f417"]), plane_check="0", asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="1" if r["pe"] else "0", dump=str(r["dump"]), bag=r["bag"]))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r79_%s" % b, "results_root": "%s/r79_%s" % (a.results_root, b), "expected_jobs": n}
    for nm, rn, bs in (("id", "R79-id", ["id", "ref"]), ("fb", "R79-fb", ["ef", "fb"])):
        sp = M.listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r79_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["line_rows_per_scan.csv"] + ["voxelmap_snapshot_f%d.csv" % n for n in SNAP_FRAMES_N[1:]]   # all are in the lossless archive (BIG list of r74_score_job.py); the reducers run before the raw files are deleted
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r79_file_size_ledger.json"), "w"), indent=1)
    print("id %d arms (3 identity + 1 determinism + 1 smoke), ref %d refuse arms, ef %d arms (4 F-464 + 4 F-441/F-440), fb %d arms (72 F-463 + 27 F-401/F-418 + 1 determinism)" % (
        len(out["id"]), len(out["ref"]), len(out["ef"]), len(out["fb"])))


if __name__ == "__main__":
    main()
