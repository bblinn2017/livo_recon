#!/usr/bin/env python3
"""R78: build the job files, arm_table.tsv, listener specs, required_outputs.json and the size ledger from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r78_manifest.py --jobs r69_inputs/r66_jobs.json --out r78_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (only the counts / key checks in tools/r69/r78_gates.py --selftest and the check at the end of this file).
Reuses the R74 helpers (make_r74_manifest as M) and the R77 line-mode tables (make_r77_manifest as R7): the same R66 base jobs, the same overrides.

Bryce 2026-10-03 / 2026-10-04 (his words are in each arm's `asked` field and in COMPLIANCE_TABLE.md):
  Stage A (approved): voxel edges 0.25, 0.30, 0.35 m below 0.4 m, every 3rd point, fixed / motion Q, line settings {off, hard absolute, hard shape, hard relative + noise floor, smooth} with
      plane_equivalent variance, gap floor off and on crossed with every line setting: 3 x 2 x 5 x 2 = 60 arms.
  Split: all points to the map, the every-3rd set to the IEKF (new key lio/iekf_point_stream = second): edges {0.425, 1.0} x Q x lines {off, hard absolute pe} = 8 arms, plus 16 same-batch controls
      (every point everywhere and every 3rd everywhere, the same 8 cells each).
  F-460: information-matched thinning: block information scale F = 3 (rotation = position = 3) on every-point and every-3rd arms, the same 16 cells as the controls = 16 arms (F = 1 is the control set).
  F-457: per-frame yaw information by row class: the frame-by-frame residual dump for frames 1 to 20 on 6 arms (0.425 m, every 3rd, Q x lines {off, hard absolute pe, smooth pe}).
  Stage C: max octree layer 1 at 0.425 and 1.0 m, every 3rd, Q x lines {off, hard absolute pe} = 8 arms (the layer-2 controls are the control set).
  F-428: whole-bag root edge 1.4 m, every 3rd, Q x lines {off, hard absolute pe} = 4 arms, and 2 post-calibration snapshot arms (every node's point count and plane test values) at 0.425 and 1.0 m.
Batches (queues):
  r78_id   6 arms (EF cell n1600 ds off pfn3 fq, 16 s): 3 identity arms, ONE determinism repeat, 2 smoke arms of the new key.
  r78_ref  3 refuse arms (5 s each, MUST FAIL at startup naming the key).
  r78_fb   121 full-bag arms (eee_01, whole bag): 60 + 8 + 16 + 16 + 6 + 8 + 4 + 2 = 120, plus ONE determinism repeat.
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402
import make_r77_manifest as R7  # noqa: E402

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
QS = ("fq", "mq")
STREAM = {"cbk.lidar.point_filter_num": 1, "lio.dry_run_point_filter_num": 3, "lio.iekf_point_stream": "second"}   # map: every point; IEKF: the every-3rd set (second stream, deskewed and downsampled as the control's)
INFO3 = {"lio.info_scale.mode": "block", "lio.info_scale.rotation": 3.0, "lio.info_scale.position": 3.0}   # F_rot = F_pos = 3: the global information divisor (info_scale.h: equals HtH/3)
FRAMES20 = ",".join(str(i) for i in range(1, 21))
F1DUMP20 = {"lio.first_frame_dump_frames": 1, "lio.residual_dump_frames": FRAMES20, "lio.update_diag_en": True}
EDGES_A = (0.25, 0.3, 0.35)
EDGES_C = (0.425, 1.0)
LINES_A = ("off", "hardabs", "hardshape", "hardrelnoise", "smooth")
LINES_2 = ("off", "hardabs")
ROOT_EDGE = 1.4
IDENT = [("r78_id_off_v1p0", "off", 1.0, "r77_id_off_v1p0", {}),
         ("r78_id_hardabs_v0p425", "hardabs", 0.425, "r77_id_hardabs_v0p425", {}),
         ("r78_id_second_off_v1p0", "off", 1.0, "r77_id_off_v1p0", {"lio.dry_run_point_filter_num": 3, "lio.iekf_point_stream": "second"})]   # every key at default / the second stream equal to the primary one
ID_DET, ID_DET_ORIG = "r78_own_det_id_off_v1p0", "r78_id_off_v1p0"
SMOKE = [("r78_sm_split_off_v0p425", "off", 0.425), ("r78_sm_split_pe_hardabs_v1p0", "hardabs", 1.0)]
FB_DET_ORIG, FB_DET = "r78_ct_pe_hardabs_v1p0_pfn3_fq", "r78_own_det_fb_pe_hardabs_v1p0_pfn3_fq"
REFUSE = [("second_no_dryrun", {"lio.iekf_point_stream": "second"}, ["lio/iekf_point_stream = second needs lio/dry_run_point_filter_num"], "00_id_baseline_60s"),
          ("stream_bogus", {"lio.iekf_point_stream": "bogus"}, ["lio/iekf_point_stream = 'bogus' is not one of"], "00_id_baseline_60s"),
          ("second_coupled", {"lio.iekf_point_stream": "second", "lio.dry_run_point_filter_num": 3}, ["lio/iekf_point_stream = second is implemented for estimator/mode=iekf only"],
           "10_smoke_coupled_measurement_fulldiag_60s")]

ASK_A = ("Bryce (2026-10-03): 'i wonder if we want to try some of the better performing config options on voxel sizes less than 0.4? include our line modes, fixed/motion Q, gap floor on and off, any others we should try?'; "
         "(2026-10-04): 'i approve stage A' (Stage A of claude/r78-proposal-voxel-below-0p4.md: edges 0.25, 0.30, 0.35 m, every 3rd point, fixed and motion Q, line settings off / hard absolute / hard shape / "
         "hard relative + noise floor / smooth with plane_equivalent variance, gap floor off and on crossed with every line setting, 60 arms)")
ASK_SPLIT = ("Bryce (2026-10-04): 'can we perhaps try an experiment in which we use all points for map building but only use filtered points for the iekf?'; his answers: 'Every 3rd of the all-point set (Recommended)', "
             "then, after my correction that whole-bag arms downsample at 0.1 m: 'Exactly the pfn3 control set (Recommended)'; '8 arms (Recommended)' = edges 0.425 and 1.0 m x fixed / motion Q x lines {off, hard absolute "
             "plane_equivalent}, 'Controls (every point everywhere, every 3rd everywhere) are in the same batch' (the option text he chose)")
ASK_F460 = ("Bryce (2026-10-04): 'we can also try f-460' (information-matched thinning: every point with the information divided by 3 against every 3rd point); his answer '16 arms, lines off and hard absolute (Recommended)' "
            "= divisor F in {1, 3} x every point / every 3rd x Q x edges {0.425, 1.0} x lines {off, hard absolute pe}, minus the duplicates of the split grid (F = 1 is the control set)")
ASK_OPEN = ("Bryce (2026-10-04): 'are there any open followups you can design tests for that we haven't already? go ahead and design tests and generate the handoff'; his answers ticked 'F-457 yaw by row class', "
            "'Stage C max layer 1 vs 2', 'F-428 1 m root voxel'")
ASK_457 = ASK_OPEN + "; for F-457 he chose '6 arms at 0.425 m, every 3rd (Recommended)' (motion-Q and fixed-Q x lines {off, hard absolute pe, smooth pe}, per-frame yaw information log for frames 1 to 20, split by row class)"
ASK_C = ASK_OPEN + "; for Stage C he chose 'Layer 1 at 0.425 m and 1.0 m, 8 arms (Recommended)' (every 3rd, both Q, lines {off, hard absolute pe}; the layer-2 controls are the same-batch controls)"
ASK_428 = ASK_OPEN + "; for F-428 he chose 'Both of the above' = 'Whole-bag root-edge sweep, 1.4 m' (every 3rd, both Q, lines {off, hard absolute pe}: 4 arms) and 'Per-voxel statistics at 1.0 m' (the post-calibration snapshot with every node's point count and plane test values on 1.0 m and 0.425 m, every 3rd, lines off, motion-Q: 2 arms)"
OWN = R7.OWN
EXPECTED = {"residual_dump_all.csv.gz": 25000, "r73_f417_hth_frames.csv": 2000, "pos_trace_all.csv.gz": 140000, "line_rows_per_scan_all.csv.gz": 70000, "snapshot_nodes_all.csv.gz": 900000}
MAXFILE = {"residual_dump_all.csv.gz": 700000, "r73_f417_hth_frames.csv": 100000, "pos_trace_all.csv.gz": 220000, "line_rows_per_scan_all.csv.gz": 250000, "snapshot_nodes_all.csv.gz": 3000000}
DUMP_COLS = M.DUMP_COLS + ["row_kind"]
LINE_ROWS_COLS = R7.LINE_ROWS_COLS
SNAP_COLS = ["node_id", "layer", "status", "retired", "initialized", "is_plane", "full", "point_count", "distinct_frames", "eig0", "eig1", "eig2", "edge_m", "planarity_value", "planarity_pass",
             "second_eig_below_threshold", "eigengap_denom1", "eigengap_denom2", "ratio_e1_over_e0", "ratio_e2_over_e1", "rel_gap_threshold", "center_x", "center_y", "center_z",
             "normal_x", "normal_y", "normal_z", "d"]
FB_POS_EXPECTED = 110000
# groups: STA stage A, SPL split, CTL controls, IF3 information divisor 3, YAW F-457, ML1 max layer 1, ROOT root edge 1.4, SNP snapshot, IDN identity, SMK smoke, DET determinism, REF refuse


def merged(*ds):
    o = {}
    for x in ds:
        o.update(x)
    return o


def mkey(mode):
    return "off" if mode == "off" else "pe_" + mode


def fbarm(aid, mode, v, q, group, asked, purpose, pfn=3, layer=2, gap=False, stream=False, info3=False, dump=None, extra=None, attr=None):
    """One whole-bag eee_01 arm: the R66 base job of the Q mode plus voxel / layer / line mode (plane_equivalent when lines are on) / point filter and the optional new keys."""
    d = dict(M.VOX(v, layer), **LINE76[mode], **LOGKEY, **{"cbk.lidar.point_filter_num": pfn})
    if mode != "off":
        d[VAR] = "plane_equivalent"
    if gap:
        d[GAP] = True
    if stream:
        d.update(STREAM)
    if info3:
        d.update(INFO3)
    if extra:
        d.update(extra)
    if dump is None:
        dump = (mode != "off")
    if dump == 20:
        d.update(F1DUMP20)
    elif dump:
        d.update(M.F1DUMP)
    a = dict(id=aid, base=M.base_of("eee_01", q), group=group, tier="FB", batch="fb", dur=None, mode=mode, voxel=v, kind="fb", ref=None, delta=d, f417=1 if dump else 0, asked=asked, purpose=purpose,
             layer=layer, gap=gap, stream="second" if stream else "primary", info3=info3, pfn=(1 if stream else pfn), q=q, pe=(mode != "off"), dump=(20 if dump == 20 else (1 if dump else 0)))
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
                      purpose="R78 identity: %s at voxel %s%s vs %s" % (mode, v, " with the second stream equal to the primary one" if extra else " with the new keys at default", ref),
                      layer=2, gap=False, stream="second" if extra else "primary", info3=False, pfn=3, q="fq", pe=False, dump=1))
    A.append(dict(id=ID_DET, base=M.base_of("eee_01", "fq"), group="DET", tier="EF", batch="id", dur=ID_DUR, mode="off", voxel=1.0, kind="det_id", ref=ID_DET_ORIG,
                  delta=dict(M.VOX(1.0, 2), **LINE76["off"], **LOGKEY, **M.F1DUMP, **CELL), f417=1,
                  asked=OWN + "rule 3.7 determinism control: an exact repeat of %s; its state-trace digest must equal that arm's" % ID_DET_ORIG,
                  purpose="OWN determinism control: repeat of %s" % ID_DET_ORIG, layer=2, gap=False, stream="primary", info3=False, pfn=3, q="fq", pe=False, dump=1))
    for aid, mode, v in SMOKE:
        ex = dict(STREAM, **({VAR: "plane_equivalent"} if mode != "off" else {}))
        A.append(dict(id=aid, base=M.base_of("eee_01", "fq"), group="SMK", tier="EF", batch="id", dur=ID_DUR, mode=mode, voxel=v, kind="smoke", ref=None,
                      delta=merged(M.VOX(v, 2), LINE76[mode], ex, LOGKEY, M.F1DUMP, CELL, {"cbk.lidar.point_filter_num": 1}), f417=1,
                      asked=OWN + "rule 3.6 smoke arm for the new key lio/iekf_point_stream = second: EF cell, %s, root voxel %s, map from every point (pfn 1), IEKF from the every-3rd second stream; the engagement line must show "
                               "frames_second = frames, frames_fallback = 0 and mean_points_iekf below mean_points_map" % (DESC[mode], v),
                      purpose="R78 smoke: split (map every point, IEKF every 3rd), %s, voxel %s" % (mode, v), layer=2, gap=False, stream="second", info3=False, pfn=1, q="fq", pe=(mode != "off"), dump=1))
    for suf, delta, needles, base in REFUSE:
        A.append(dict(id="r78_refuse_%s" % suf, base=base, group="REF", tier="REF", batch="ref", dur=5, mode=None, voxel=None, kind="refuse", ref=None, needles=needles, delta=dict(delta), f417=0,
                      asked=OWN + "rule 3.4 refusal: the contradictory or dead setting MUST FAIL at startup, text contains %s" % " and ".join(repr(n) for n in needles),
                      purpose="MUST FAIL: %s" % suf, layer=None, gap=False, stream="primary", info3=False, pfn=None, q=None, pe=False, dump=0))
    # ---- Stage A: edges below 0.4 m x Q x line mode (pe) x gap floor off / on, every 3rd point (60)
    for e in EDGES_A:
        for q in QS:
            for mode in LINES_A:
                for gap in (False, True):
                    aid = "r78_sa_%s_v%s_gap%d_%s" % (mkey(mode), M.tag(e), 1 if gap else 0, q)
                    A.append(fbarm(aid, mode, e, q, "STA", "%s: edge %s m, every 3rd point, %s Q, %s%s, gap floor %s" % (
                        ASK_A, e, "fixed" if q == "fq" else "motion", DESC[mode], " with plane_equivalent variance" if mode != "off" else " (lines off)", "on (plane_gap_floor_en)" if gap else "off"),
                        "R78 Stage A: edge %s, %s, gap floor %s, pfn 3, %s" % (e, mode, "on" if gap else "off", q), pfn=3, gap=gap))
    # ---- controls (24 = 8 split + 16 controls) and F-460 (16)
    for v in EDGES_C:
        for q in QS:
            for mode in LINES_2:
                for pfn in (1, 3):
                    A.append(fbarm("r78_ct_%s_v%s_pfn%d_%s" % (mkey(mode), M.tag(v), pfn, q), mode, v, q, "CTL",
                                   "%s: same-batch control of the split and information-divisor grids: edge %s m, %s, %s Q, %s (the same cell as the R77 arm of that name; also the F = 1 half of the F-460 grid)" % (
                                       ASK_SPLIT, v, "every point" if pfn == 1 else "every 3rd point", "fixed" if q == "fq" else "motion", DESC[mode]),
                                   "R78 control: edge %s, %s, pfn %d, %s" % (v, mode, pfn, q), pfn=pfn))
                A.append(fbarm("r78_sp_%s_v%s_%s" % (mkey(mode), M.tag(v), q), mode, v, q, "SPL",
                               "%s: edge %s m, %s, %s Q: the map built from every point (point filter 1), the IEKF residuals from the every-3rd set of the second stream" % (
                                   ASK_SPLIT, v, DESC[mode], "fixed" if q == "fq" else "motion"),
                               "R78 split: map every point, IEKF every 3rd, edge %s, %s, %s" % (v, mode, q), stream=True))
                for pfn in (1, 3):
                    A.append(fbarm("r78_if3_%s_v%s_pfn%d_%s" % (mkey(mode), M.tag(v), pfn, q), mode, v, q, "IF3",
                                   "%s: information divided by 3 (lio/info_scale mode block, rotation = position = 3), edge %s m, %s, %s Q, %s" % (
                                       ASK_F460, v, "every point" if pfn == 1 else "every 3rd point", "fixed" if q == "fq" else "motion", DESC[mode]),
                                   "R78 F-460: information / 3, edge %s, %s, pfn %d, %s" % (v, mode, pfn, q), pfn=pfn, info3=True))
    # ---- F-457: frame-by-frame dump for frames 1 to 20 (6)
    for q in QS:
        for mode in ("off", "hardabs", "smooth"):
            A.append(fbarm("r78_yw_%s_v0p425_pfn3_%s" % (mkey(mode), q), mode, 0.425, q, "YAW",
                           "%s: edge 0.425 m, every 3rd point, %s Q, %s; lio/residual_dump_frames = 1..20 so the yaw information of frames 1 to 20 can be split by row class" % (
                               ASK_457, "fixed" if q == "fq" else "motion", DESC[mode]),
                           "R78 F-457: per-frame dump frames 1-20, 0.425, %s, pfn 3, %s" % (mode, q), pfn=3, dump=20))
    # ---- Stage C: max layer 1 (8)
    for v in EDGES_C:
        for q in QS:
            for mode in LINES_2:
                A.append(fbarm("r78_ml1_%s_v%s_pfn3_%s" % (mkey(mode), M.tag(v), q), mode, v, q, "ML1",
                               "%s: max octree layer 1 (the controls have 2), edge %s m, every 3rd point, %s Q, %s" % (ASK_C, v, "fixed" if q == "fq" else "motion", DESC[mode]),
                               "R78 Stage C: max_layer 1, edge %s, %s, pfn 3, %s" % (v, mode, q), pfn=3, layer=1))
    # ---- F-428: root edge 1.4 m (4) and the snapshot arms (2)
    for q in QS:
        for mode in LINES_2:
            A.append(fbarm("r78_rt_%s_v1p4_pfn3_%s" % (mkey(mode), q), mode, ROOT_EDGE, q, "ROOT",
                           "%s: whole-bag root edge 1.4 m, max_layer 2, every 3rd point, %s Q, %s" % (ASK_428, "fixed" if q == "fq" else "motion", DESC[mode]),
                           "R78 F-428: root edge 1.4, %s, pfn 3, %s" % (mode, q), pfn=3))
    for v in (0.425, 1.0):
        A.append(fbarm("r78_snap_off_v%s_pfn3_mq" % M.tag(v), "off", v, "mq", "SNP",
                       "%s: post-calibration snapshot (voxel_map.map.snapshot_post_calibration) at edge %s m, every 3rd point, motion Q, lines off: every node's point count and plane test values" % (ASK_428, v),
                       "R78 F-428: snapshot with test values, edge %s, lines off, pfn 3, mq" % v, pfn=3, dump=True, extra=dict(M.SNAP)))
    d = fbarm(FB_DET, "hardabs", 1.0, "fq", "DET", OWN + "rule 3.7 determinism control: an exact whole-bag repeat of %s; its state-trace digest must equal that arm's" % FB_DET_ORIG,
              "OWN determinism control: whole-bag repeat of %s" % FB_DET_ORIG, pfn=3)
    d["kind"] = "det_fb"; d["ref"] = FB_DET_ORIG
    A.append(d)
    return A


def required_outputs(table):
    sel = lambda f: [t["arm"] for t in table if f(t)]
    dumpers = sel(lambda t: t["f417"] == "1")
    yaw = sel(lambda t: t["group"] == "YAW")
    dump1 = [a for a in dumpers if a not in yaw]
    ident = sel(lambda t: t["group"] in ("IDN", "SMK") or t["arm"] == ID_DET)
    fb_all = sel(lambda t: t["batch"] == "r78_fb")
    snap = sel(lambda t: t["group"] == "SNP")
    idfb = sel(lambda t: t["group"] != "REF")
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", expected=None, maxb=None):
        eb = (expected if expected is not None else EXPECTED.get(file, 50000)) * max(1, len(arms_)) if where == "return" else 0
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=eb,
                      max_bytes_per_file=maxb or MAXFILE.get(file, 400000)))
    add("F-459", "per-residual dump of frame 1 with row_kind (0 plane, 1-2 hard line u0 u1, 3-5 smooth) for every arm that dumps, except the F-457 arms", dump1, "residual_dump_all.csv.gz", DUMP_COLS, 50)
    add("F-457", "per-residual dump of frames 1 to 20 with row_kind, weight, class and the rotation-about-z row of the Jacobian (hrot_z), the six F-457 arms", yaw, "residual_dump_all.csv.gz", DUMP_COLS, 1000, expected=500000, maxb=2500000)
    add("F-459", "residual report (HtH / Htz metrics) per dumped frame for every dumping arm", dumpers, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1,
        expected=2000)
    add("F-451", "per-scan line-row log, identity and smoke arms", ident, "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 50)
    add("F-451", "per-scan line-row log (scans <= 800 plus every 10th), every full-bag arm", fb_all, "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 100)
    add("F-459", "per-scan posterior position and yaw of every identity and smoke arm (scan_id key)", ident, "pos_trace_all.csv.gz", M.POS_COLS, 20)
    add("F-459", "per-scan posterior position and yaw of every full-bag arm", fb_all, "pos_trace_all.csv.gz", M.POS_COLS, 1000, expected=FB_POS_EXPECTED, maxb=200000)
    add("F-428", "every node of the post-calibration snapshot (point count, eigenvalues, plane test values, centre, normal), the two snapshot arms", snap, "snapshot_nodes_all.csv.gz", SNAP_COLS, 500)
    add("F-452", "gates: identity vs R77, determinism digests, refusals, effective config (voxel, layer, line, variance, gap floor, stream, information scale), engagement, whole-bag line-row gate", [], "r78_gates.txt", [], 1, source="generated")
    add("F-452", "identity comparison of every identity arm and every full-bag control against its R77 counterpart (per-scan max abs difference of position and yaw)", [], "r78_identity_vs_old.csv",
        ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"], 6, source="generated")
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated (includes the iekf_point_stream and information-scale lines)", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 50, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm that is not a refuse arm", idfb, "engagement_imu.txt", [], 1, where="archive")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"] or e["source"] == "generated":
            continue
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * 3 if e["file"] != "residual_dump_all.csv.gz" else e["expected_bytes"] * 2),
                    "basis": "estimate from the measured R74-R77 returns (median per file); the snapshot file and the 20-frame dump have no measurement yet, the agent measures one arm of each (size_ledger.py measure); %d arms" % len(e["arms"]),
                    "note": e["followup"]})
    for pat, mx, tot in (("r73_f417_classes.csv", 20000, 4000000), ("r73_f417_consistency.csv", 5000, 1000000), ("r78_gates.txt", 600000, 600000), ("identity_controls.txt", 150000, 150000),
                         ("r78_identity_vs_old.csv", 40000, 40000), ("engagement_all.txt", 600000, 600000), ("completeness.csv", 300000, 300000),
                         ("phase_summary.csv", 4000000, 4000000), ("arm_scan_all.csv.gz", 40000000, 40000000), ("raw_sha256.txt", 20000, 20000),
                         ("R78_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000), ("listener_*.log", 5000000, 20000000), ("*.md", 1000000, 3000000),
                         ("*.json", 2000000, 10000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000), ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74-R77 ledger values or a round figure); not a measurement", "note": "non-arm file"})
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
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r78_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {"id": [], "ref": [], "fb": []}; table = []; seen = set()
    for r in arms():
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 to R77)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r78_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        req = M.requires_of(o, "refuse" if r["batch"] == "ref" else "x")
        if o.get("lio.line_rows_log_en"):
            req = req + ["line_rows_per_scan.csv"]
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r78_" + r["batch"], sampling=sampling, requires=" ".join(req),
                          f417=str(r["f417"]), plane_check="0", asked=r["asked"]))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r78_%s" % b, "results_root": "%s/r78_%s" % (a.results_root, b), "expected_jobs": n}
    for nm, rn, bs in (("id", "R78-id", ["id", "ref"]), ("fb", "R78-fb", ["fb"])):
        sp = M.listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r78_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["line_rows_per_scan.csv"]   # reduced into the archive (BIG list of r74_score_job.py) and into line_rows_per_scan_all.csv.gz (R77)
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r78_file_size_ledger.json"), "w"), indent=1)
    print("id %d arms (3 identity + 1 determinism + 2 smoke), ref %d refuse arms, fb %d arms (60 stage A + 8 split + 16 controls + 16 F-460 + 6 F-457 + 8 stage C + 4 root + 2 snapshot + 1 determinism)" % (
        len(out["id"]), len(out["ref"]), len(out["fb"])))


if __name__ == "__main__":
    main()
