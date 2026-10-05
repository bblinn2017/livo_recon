#!/usr/bin/env python3
"""R79 gates: identity of the new build against the R78 arms (3 EF identity arms, the 8 eee_01 F-463 arms, the 4 F-464 arms), determinism digests, refusals, effective config (root voxel, max layer, line mode,
variance, gap floor, point stream, information scale, ZUPT, q gain), quaternion columns, whole-bag length and line-row engagement, the per-frame snapshots (presence, node counts, join to the residual dump of
the same frame), the early state / update files, the motion-Q trace, completeness.

    python3 tools/r69/r79_gates.py --runs <runs_dir> [--refs r79_manifest/identity_refs.json] [--refs-fb r79_manifest/identity_refs_fb.csv.gz] [--out r79_gates.txt] [--csv r79_identity_vs_old.csv]
    python3 tools/r69/r79_gates.py --selftest      fake arm dirs for every arm of the R79 manifest: good data all PASS, every single breakage FAILs its gate, empty dir all MISSING

runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py); the return's compact/<arm>/ files are read from the same directory (r69_collect copies them there before the gates run, as in R78).
PASS / FAIL / MISSING (absent: never a pass) / INFO (a fact reported without a verdict). Exit 1 on any FAIL.

Own choices (disclosed): (1) the snapshot join gate requires 99.9 percent of the dump rows of frame N to find their node_id in the snapshot of map frame N (the expected value is 100 percent; the 0.1 percent
allowance is for a row whose node was retired between the query and the snapshot, if that can happen); an INFO row reports the share found in the snapshot of frame N-1 so the reader can see whether the join
could tell the two maps apart. (2) ZUPT: the gate decides only that the ZUPT was evaluated (scans > 0) and configured as asked; how many scans were stationary or applied is an INFO row (R74: ZUPT mid was
inert on 5 of 9 bags). (3) q gain: for the NIS arms the gate decides that the Q step ran with the gain machinery on (steps > 0); the count of inflated steps is an INFO row (a gate that never inflates is a result,
not a defect); arms without the gain must show steps_with_gain_ne_1 = 0. (4) F-464 identity: the 60 s arms are compared with the R78 whole-bag arm on the common scans except the last 3 (the run stops at the duration);
the last 3 are an INFO row. (5) Length: a whole-bag arm must have at least 97 percent of the bag's scans.
"""
import shutil, argparse, io, json, os, re, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r79_manifest as M  # noqa: E402
from r74_gates import rd, csv, jsn, digest, G, report  # noqa: E402
from r77_identity_refs import dump_sigs, node_sig  # noqa: E402
import r77_gates as R  # noqa: E402
import r78_gates as R8  # noqa: E402

LINE_RE = R.LINE_RE
GAP_RE = R8.GAP_RE
GAP_RE_R77 = R8.GAP_RE_R77
PFN_RE = R8.PFN_RE
STREAM_RE = R8.STREAM_RE
INFO_RE = R8.INFO_RE
CTR = R.CTR
LINEC = R.LINEC
ROS_LINE = R8.ROS_LINE
ZUPT_RE = re.compile(r"lio/zupt mode=(\w+) gyro_thresh=([0-9.eE+-]+) acc_thresh=([0-9.eE+-]+) enter_scans=(\d+) sigma_v=([0-9.eE+-]+) scans=(\d+) stationary=(\d+) applied=(\d+)")
GAIN_RE = re.compile(r"q_gain steps=(\d+) steps_with_gain_ne_1=(\d+) max_gain_acc=([0-9.eE+-]+) max_gain_gyr=([0-9.eE+-]+)")
FRAMES5 = (1, 2, 3, 4, 5)
SNAP_JOIN_MIN = 0.999
LEN_MIN = 0.97
MIN_DUMP_ROWS = 10


def lrows(runs, arm):
    return csv(os.path.join(runs, arm), "line_rows_per_scan_all.csv.gz")


def eng_text(d):
    return (rd(d, "engagement.txt") or "") + "\n" + (rd(d, "run.log") or "")


def ref_rows(refs_fb, ref_arm):
    if refs_fb is None:
        return None
    r = refs_fb[refs_fb["arm"] == ref_arm]
    return None if r.empty else r


def compare_pos(P, rr, drop_last=0):
    """-> (n_common, max |dpos| mm, max |dyaw| mrad, same-scan table limited to scans that are decided) of the arm's pos trace against the reference rows."""
    new = P.drop_duplicates("scan_id").set_index("scan_id"); old = rr.drop_duplicates("scan_id").set_index("scan_id")
    com = old.index.intersection(new.index).sort_values()
    tail = com[len(com) - drop_last:] if drop_last else com[:0]
    dec = com[:len(com) - drop_last] if drop_last else com
    def mx(ix):
        if not len(ix):
            return float("nan"), float("nan")
        dp = float(np.max(np.abs(new.loc[ix, ["px", "py", "pz"]].to_numpy() - old.loc[ix, ["px", "py", "pz"]].to_numpy())) * 1e3)
        dy = new.loc[ix, "yaw_rad"].to_numpy() - old.loc[ix, "yaw_rad"].to_numpy()
        return dp, float(np.max(np.abs((dy + np.pi) % (2 * np.pi) - np.pi)) * 1e3)
    return len(com), mx(dec), mx(tail), len(dec)


def run_gates(runs, refs=None, refs_fb=None):
    g = G()
    A = lambda a: os.path.join(runs, a)
    table = M.arms()
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    notdone = []
    for det, orig in ((M.ID_DET, M.ID_DET_ORIG), (M.FB_DET, M.FB_DET_ORIG)):
        a, b = digest(runs, det), digest(runs, orig)
        miss = [n for n, x in ((det, a), (orig, b)) if x is None]
        g.add("DET %s digest == %s" % (det, orig), "MISSING" if miss else ("PASS" if a == b else "FAIL"), ("digest file absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    id_rows = []
    for t in table:
        arm = t["id"]
        if t["kind"] == "refuse":
            nm = "REFUSE %s dies at startup, text contains %s" % (arm, " and ".join(repr(n) for n in t["needles"]))
            if not present(arm):
                g.add(nm, "MISSING"); continue
            meta = jsn(A(arm), "arm.json") or {}
            tail = (rd(A(arm), "run_log_tail.txt") or "") + (rd(A(arm), "run.log") or "")
            has = [n in tail for n in t["needles"]]
            g.add(nm, "PASS" if meta.get("sentinel") == "FAILED" and all(has) else "FAIL", "sentinel=%s needles_found=%s" % (meta.get("sentinel"), has))
            continue
        if not present(arm):
            g.add("LINE %s engaged" % arm, "MISSING"); notdone.append(arm); continue
        if (jsn(A(arm), "arm.json") or {}).get("sentinel") != "DONE":
            notdone.append(arm)
        log = rd(A(arm), "run.log") or ""
        mv, ml = re.search(r"map/voxel_size:\s+([0-9.eE+-]+)", log), re.search(r"map/max_layer:\s+(\d+)", log)
        ok = bool(mv and ml) and abs(float(mv.group(1)) - t["voxel"]) < 1e-12 and int(ml.group(1)) == t["layer"]
        g.add("VOX %s root voxel %s max_layer %s in the effective config" % (arm, t["voxel"], t["layer"]), "PASS" if ok else ("MISSING" if not log else "FAIL"), "log: voxel %s max_layer %s" % (mv and mv.group(1), ml and ml.group(1)))
        want = M.WANT[t["mode"]]
        wvar = "plane_equivalent" if t["pe"] else "fit_only"
        l = LINE_RE.search(log)
        ok = bool(l) and (l.group(1), l.group(2), l.group(3)) == want and abs(float(l.group(4)) - 3.0) < 1e-12 and abs(float(l.group(5)) - 0.1) < 1e-12 and int(l.group(6)) == M.ENUM[t["mode"]] and l.group(7) == wvar
        g.add("LINE %s effective config enabled %s / form %s / rule %s (enum %d) / variance %s" % (arm, want[0], want[1], want[2], M.ENUM[t["mode"]], wvar), "PASS" if ok else ("MISSING" if not log else "FAIL"), l.group(0) if l else "no line")
        gm = GAP_RE.search(log)
        g.add("GAPCFG %s effective config plane_gap_floor_en false (application line only)" % arm, "PASS" if gm and gm.group(1) == "false" else ("MISSING" if not log else "FAIL"), gm.group(0).strip() if gm else "no line")
        pm = PFN_RE.search(log)
        g.add("PFN %s map point filter %s in the effective config" % (arm, t["pfn"]), "PASS" if pm and int(pm.group(1)) == t["pfn"] else ("MISSING" if not pm else "FAIL"), "log: %s" % (pm and pm.group(0)))
        et = eng_text(A(arm))
        sm, im, zm = STREAM_RE.search(et), INFO_RE.search(et), ZUPT_RE.search(et)
        gt = GAIN_RE.search(rd(A(arm), "engagement_imu.txt") or "")
        ident_second = t["kind"] == "id" and t["stream"] == "second"
        if t["stream"] == "second" and not ident_second:
            if not sm:
                g.add("STREAM %s engagement line for lio/iekf_point_stream = second" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                fr, f2, fb, mp, mi = int(sm.group(3)), int(sm.group(4)), int(sm.group(5)), float(sm.group(6)), float(sm.group(7))
                ok = sm.group(1) == "second" and int(sm.group(2)) == 3 and fr > 0 and f2 == fr and fb == 0 and 0 < mi < mp
                g.add("STREAM %s engaged: stream second, dry_run 3, frames_second == frames (%d of %d), no fallback (%d), IEKF set smaller than the map set (%.1f < %.1f)" % (arm, f2, fr, fb, mi, mp), "PASS" if ok else "FAIL", sm.group(0))
        elif not ident_second:
            g.add("NOSTREAM %s no second-stream engagement line (key at default)" % arm, "PASS" if not sm and et.strip() else ("MISSING" if not et.strip() else "FAIL"), sm.group(0) if sm else "")
        if ident_second:
            if not sm:
                g.add("STREAM %s engagement line (identity arm with the second stream equal to the primary)" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                fr, f2, fb, mp, mi = int(sm.group(3)), int(sm.group(4)), int(sm.group(5)), float(sm.group(6)), float(sm.group(7))
                g.add("STREAM %s engaged with an equal set: frames_second == frames, no fallback, mean_points_iekf == mean_points_map" % arm, "PASS" if fr > 0 and f2 == fr and fb == 0 and abs(mi - mp) < 1e-9 else "FAIL", sm.group(0))
        g.add("NOINFO3 %s information scale off (no engagement line)" % arm, "PASS" if not im and et.strip() else ("MISSING" if not et.strip() else "FAIL"), im.group(0) if im else "")
        if t["zupt"]:
            if not zm:
                g.add("ZUPT %s engagement line for lio/zupt mode apply" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                ok = zm.group(1) == "apply" and abs(float(zm.group(2)) - 0.02) < 1e-12 and abs(float(zm.group(3)) - 0.15) < 1e-12 and int(zm.group(4)) == 3 and abs(float(zm.group(5)) - 0.02) < 1e-12 and int(zm.group(6)) > 0
                g.add("ZUPT %s configured apply / gyro 0.02 / acc 0.15 / enter 3 / sigma_v 0.02 and evaluated (scans %s > 0)" % (arm, zm.group(6)), "PASS" if ok else "FAIL", zm.group(0))
                g.add("ZUPT %s scans evaluated, stationary, applied (a fact; ZUPT mid was inert on 5 of 9 bags in R74)" % arm, "INFO", "scans %s stationary %s applied %s" % (zm.group(6), zm.group(7), zm.group(8)))
        else:
            g.add("NOZUPT %s no ZUPT engagement line (mode off)" % arm, "PASS" if not zm and et.strip() else ("MISSING" if not et.strip() else "FAIL"), zm.group(0) if zm else "")
        if t["nis"]:
            if not gt:
                g.add("GAIN %s q_gain line in engagement_imu.txt (NIS gate on)" % arm, "MISSING" if rd(A(arm), "engagement_imu.txt") is None else "FAIL", "no line")
            else:
                g.add("GAIN %s the Q step ran with the gain machinery on (steps %s > 0)" % (arm, gt.group(1)), "PASS" if int(gt.group(1)) > 0 else "FAIL", gt.group(0))
                g.add("GAIN %s steps with gain != 1 and the largest gains (a fact: 0 means the NIS never exceeded 1.3)" % arm, "INFO", "steps_with_gain_ne_1 %s max_gain_acc %s max_gain_gyr %s" % (gt.group(2), gt.group(3), gt.group(4)))
        else:
            g.add("NOGAIN %s no gain applied (steps_with_gain_ne_1 == 0)" % arm, "PASS" if gt and int(gt.group(2)) == 0 else ("MISSING" if not gt else "FAIL"), gt.group(0) if gt else "no q_gain line")
        # ---- quaternion columns and length
        P = csv(A(arm), "pos_trace_all.csv.gz")
        if P is None or any(c not in P.columns for c in M.POS_COLS):
            g.add("QUAT %s pos_trace_all.csv.gz has qw, qx, qy, qz, unit norm, and yaw_rad equal to the yaw of the quaternion" % arm, "MISSING")
        else:
            q = P[["qw", "qx", "qy", "qz"]].to_numpy(float)
            nrm = float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)))
            w_, x_, y_, z_ = q.T
            yq = np.arctan2(2 * (x_ * y_ + z_ * w_), 1 - 2 * (y_ * y_ + z_ * z_))
            dy = float(np.max(np.abs((yq - P["yaw_rad"].to_numpy(float) + np.pi) % (2 * np.pi) - np.pi)))
            g.add("QUAT %s qw, qx, qy, qz present, unit norm within 1e-6, yaw of the quaternion equals yaw_rad within 1e-6 rad" % arm, "PASS" if nrm < 1e-6 and dy < 1e-6 else "FAIL", "max | |q|-1 | %.3g, max yaw difference %.3g rad" % (nrm, dy))
        if t["batch"] == "r79_fb" or t["bag"] != "eee_01" or t["kind"] == "fb" and t["early"] == 0:
            n = M.N_SCANS[t["bag"]]
            ok = P is not None and len(P) >= LEN_MIN * n
            g.add("LEN %s whole bag: at least %d percent of the %d scans of %s" % (arm, int(LEN_MIN * 100), n, t["bag"]), "PASS" if ok else ("MISSING" if P is None else "FAIL"), "rows %s" % (None if P is None else len(P)))
        if t["group"] == "DX":
            ok = P is not None and 560 <= len(P) <= 700
            g.add("DXLEN %s first 60 s: between 560 and 700 scans (about 595 expected)" % arm, "PASS" if ok else ("MISSING" if P is None else "FAIL"), "rows %s" % (None if P is None else len(P)))
        # ---- dump gates
        D = csv(A(arm), "residual_dump_all.csv.gz")
        if t["dump"] and t["mode"] != "off":
            if D is None or "row_kind" not in D.columns:
                g.add("LINE %s dump has row_kind and consistent row counts" % arm, "MISSING")
            else:
                n = D["row_kind"].value_counts().to_dict(); c = lambda k: int(n.get(k, 0))
                good = c(2) <= c(1) and c(3) == c(4) == c(5) == 0
                g.add("LINE %s dump row kinds consistent (hard: kind2 <= kind1, no smooth kinds)" % arm, "PASS" if good else "FAIL", "counts %s" % dict(sorted(n.items())))
        elif t["dump"] and D is not None and "row_kind" in D.columns:
            g.add("LINE %s lines off: every dump row is a plane row (row_kind 0)" % arm, "PASS" if int((D["row_kind"] != 0).sum()) == 0 else "FAIL", "%d of %d rows with kind != 0" % (int((D["row_kind"] != 0).sum()), len(D)))
        # ---- early-frame files
        if t["early"]:
            if D is None or "frame" not in D.columns:
                g.add("DUMPFR %s dump holds rows for every frame 1 to 5" % arm, "MISSING")
            else:
                cnt = {f: int((D["frame"] == f).sum()) for f in FRAMES5}
                g.add("DUMPFR %s dump holds at least %d rows for every frame 1 to 5" % (arm, MIN_DUMP_ROWS), "PASS" if all(v >= MIN_DUMP_ROWS for v in cnt.values()) else "FAIL", "rows per frame %s" % cnt)
            S = csv(A(arm), "snapshot_frames_all.csv.gz")
            if S is None:
                g.add("SNAPS %s snapshot_frames_all.csv.gz with map frames 1 to 5" % arm, "MISSING")
            else:
                miss = [c for c in M.SNAP_COLS if c not in S.columns]
                frs = sorted(int(x) for x in S["map_frame"].unique()) if "map_frame" in S.columns else []
                per = {f: int((S["map_frame"] == f).sum()) for f in frs} if frs else {}
                uniq = all(S[S["map_frame"] == f]["node_id"].is_unique for f in frs) if frs and "node_id" in S.columns else False
                ok = not miss and frs == list(FRAMES5) and all(v >= 100 for v in per.values()) and uniq and int(S["layer"].max()) <= t["layer"] and int(S["layer"].min()) >= 0 and bool((S["point_count"] >= 0).all())
                g.add("SNAPS %s snapshot file: the %d columns, map frames exactly 1 to 5, at least 100 nodes each, unique node_id within a frame, layers 0..%d, point_count >= 0" % (arm, len(M.SNAP_COLS), t["layer"]), "PASS" if ok else "FAIL",
                      "nodes per frame %s, missing columns %s, unique %s" % (per, miss[:4], uniq))
                if D is not None and "node_id" in D.columns and "frame" in D.columns and frs:
                    sets = {f: set(S[S["map_frame"] == f]["node_id"].astype("int64")) for f in frs}
                    for f in FRAMES5:
                        ids = D[D["frame"] == f]["node_id"].astype("int64")
                        if not len(ids):
                            continue
                        same = float(ids.isin(sets.get(f, set())).mean())
                        g.add("SNAPJOIN %s frame %d: share of the dump rows whose node_id is in the snapshot of map frame %d is at least %.3f" % (arm, f, f, SNAP_JOIN_MIN), "PASS" if same >= SNAP_JOIN_MIN else "FAIL", "share %.4f" % same)
                        if f >= 2:
                            prev = float(ids.isin(sets.get(f - 1, set())).mean())
                            g.add("SNAPJOIN %s frame %d: share found in the snapshot of map frame %d instead (a fact: how far the two maps differ)" % (arm, f, f - 1), "INFO", "share %.4f" % prev)
                else:
                    g.add("SNAPJOIN %s dump and snapshot joined by node_id" % arm, "MISSING")
            E = csv(A(arm), "state_trace_early_all.csv.gz")
            if E is None:
                g.add("EARLY %s state_trace_early_all.csv.gz holds scans 1 to 5 of both phases with the covariance upper triangle" % arm, "MISSING")
            else:
                pc = [c for c in E.columns if c.startswith("P_")]
                ph = {f: set(E[E["scan_id"] == f]["phase"]) for f in FRAMES5} if "scan_id" in E.columns and "phase" in E.columns else {}
                ok = len(pc) >= 171 and all(ph.get(f) == {"post_imu", "post_lio"} for f in FRAMES5) and all(c in E.columns for c in M.STATE_EARLY_COLS)
                g.add("EARLY %s state_trace_early_all.csv.gz: scans 1 to 5 each with post_imu and post_lio rows, %d covariance columns (>= 171)" % (arm, len(pc)), "PASS" if ok else "FAIL", "phases %s" % ph)
            U = csv(A(arm), "lio_update_diag_early_all.csv.gz")
            if U is None:
                g.add("EARLY %s lio_update_diag_early_all.csv.gz holds frames 1 to 5" % arm, "MISSING")
            else:
                fr = sorted(set(int(x) for x in U["frame"].unique())) if "frame" in U.columns else []
                g.add("EARLY %s lio_update_diag_early_all.csv.gz: every frame 1 to 5 has at least one iteration row, columns HtH / Htz present" % arm, "PASS" if all(f in fr for f in FRAMES5) and all(c in U.columns for c in M.UPD_EARLY_COLS) else "FAIL", "frames %s" % fr[:12])
        # ---- line rows
        L = lrows(runs, arm)
        if L is None or any(c not in L.columns for c in CTR + ["sum_tilt_var", "n_plane_rows"] + LINEC):
            g.add("LROWS %s line_rows_per_scan_all.csv.gz present with the R77 columns" % arm, "MISSING")
        else:
            cm = {c: float(L[c].max()) for c in CTR}
            tilt = float(L["sum_tilt_var"].sum()); nlin = float(L[LINEC].to_numpy().sum()); npl = float(L["n_plane_rows"].sum())
            per_scan = L[LINEC].sum(axis=1)
            if t["mode"] == "off":
                g.add("LROWS %s lines off: plane rows > 0 and no line rows over %d logged scans" % (arm, len(L)), "PASS" if npl > 0 and nlin == 0 else "FAIL", "plane rows %d, line rows %d" % (npl, nlin))
            else:
                g.add("LROWS %s whole-bag line engagement: some scan has line rows (%d of %d logged scans)" % (arm, int((per_scan > 0).sum()), len(L)), "PASS" if nlin > 0 else "FAIL", "line rows %d, share of scans %.3f" % (nlin, float((per_scan > 0).mean())))
                g.add("LROWS %s share of logged scans with line rows (a fact)" % arm, "INFO", "%.3f" % float((per_scan > 0).mean()))
            if t["pe"]:
                g.add("PE %s engaged: tilt_voxels moved (counter max %d) and tilt_fallbacks == 0 (max %d)" % (arm, cm["tilt_voxels"], cm["tilt_fallbacks"]), "PASS" if cm["tilt_voxels"] > 0 and cm["tilt_fallbacks"] == 0 else "FAIL", "counters %s" % cm)
            else:
                g.add("NOPE %s variance at default: tilt counters and tilt sum are exactly 0" % arm, "PASS" if cm["tilt_voxels"] == 0 and cm["tilt_floored_axes"] == 0 and cm["tilt_fallbacks"] == 0 and tilt == 0 else "FAIL",
                      "tilt_voxels %d, floored %d, fallbacks %d, sum %.6g" % (cm["tilt_voxels"], cm["tilt_floored_axes"], cm["tilt_fallbacks"], tilt))
            g.add("NOGAP %s gap floor off: gap counters are exactly 0" % arm, "PASS" if cm["gap_fits"] == 0 and cm["gap_d1"] == 0 and cm["gap_d2"] == 0 else "FAIL", "gap_fits %d d1 %d d2 %d" % (cm["gap_fits"], cm["gap_d1"], cm["gap_d2"]))
        # ---- identity
        if t["kind"] == "id" and refs and arm in refs:
            if D is None or P is None:
                g.add("IDENT %s state reproduces %s" % (arm, refs[arm]["ref_arm"]), "MISSING", "dump or pos_trace absent")
            else:
                r = R.ident_row(arm, refs[arm], D, P); id_rows.append(r)
                same = r["pos_max_abs_diff_mm"] == 0.0 and r["yaw_max_abs_diff_mrad"] == 0.0 and r["dump_equal"] == "yes"
                g.add("IDENT %s state reproduces %s" % (arm, r["ref_arm"]), "PASS" if same else "FAIL",
                      "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad over %s common scans, dump (all columns but plane_key, node_id) %s" % (r["pos_max_abs_diff_mm"], r["yaw_max_abs_diff_mrad"], r["n_common_scans"], r["dump_equal"]))
                g.add("IDENT-NODE %s node_id column equal to %s" % (arm, r["ref_arm"]), "INFO", "node_id equal: %s (a pointer-like id; reported, never a verdict)" % r["node_id_equal"])
        for tab, tag_, drop, minc in ((M.IDREF_NB, "IDENT-FB", 0, 1000), (M.IDREF_DX, "IDENT-DX", M.DX_LAST_EXCLUDED, 400)):
            if arm in tab and refs_fb is not None:
                ref_arm = tab[arm]
                rr = ref_rows(refs_fb, ref_arm)
                if rr is None or P is None:
                    g.add("%s %s state reproduces %s" % (tag_, arm, ref_arm), "MISSING", "reference rows or pos_trace absent")
                    continue
                ncom, (dp, dyaw), (tp, ty), ndec = compare_pos(P, rr, drop)
                id_rows.append(dict(arm=arm, ref_arm=ref_arm, n_common_scans=ncom, pos_max_abs_diff_mm=dp, yaw_max_abs_diff_mrad=dyaw, dump_equal="n/a", node_id_equal="n/a", digest_equal_if_known="n/a"))
                g.add("%s %s state reproduces %s over %d common scans%s" % (tag_, arm, ref_arm, ndec, (" (the last %d not decided)" % drop) if drop else ""), "PASS" if ndec > minc and dp == 0.0 and dyaw == 0.0 else "FAIL",
                      "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad" % (dp, dyaw))
                if drop:
                    g.add("%s %s the last %d common scans (a fact: the run stops at the duration)" % (tag_, arm, drop), "INFO", "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad" % (tp, ty))
    # ---- F-437 motion-Q trace
    Q = csv(A(M.MQ_TRACE_ARM), "motion_q_scan_all.csv.gz")
    g.add("MQSCAN %s motion_q_scan_all.csv.gz present, every scan (>= 2000 rows), columns scan_index, t_rel_s, model, acc_scale" % M.MQ_TRACE_ARM,
          "MISSING" if Q is None else ("PASS" if len(Q) >= 2000 and all(c in Q.columns for c in ("scan_index", "t_rel_s", "model", "acc_scale")) else "FAIL"), "rows %s" % (None if Q is None else len(Q)))
    runnable = [t for t in table if t["kind"] != "refuse"]
    g.add("ALL %d non-refuse arms finished (sentinel DONE)" % len(runnable),
          "MISSING" if len(notdone) == len(runnable) and not any(present(t["id"]) for t in runnable) else ("PASS" if not notdone else "FAIL"),
          "not done: %s" % ",".join(notdone[:6]) + (" ..." if len(notdone) > 6 else ""))
    for t in runnable:
        arm = t["id"]
        if not present(arm) or (jsn(A(arm), "arm.json") or {}).get("sentinel") == "FAILED":
            continue
        raw = rd(A(arm), "raw_sha256.txt") or ""
        g.add("ENGIMU %s wrote engagement_imu.txt" % arm, "PASS" if "engagement_imu.txt" in raw or os.path.exists(os.path.join(A(arm), "engagement_imu.txt")) else "FAIL")
    g.id_rows = id_rows
    return g


# --------------------------------------------------------------------------- selftest
def fake_pos_q(n, shift=0.0, qbad=None):
    t = np.arange(1, n + 1)
    yaw = 3.1 + t * 1e-5
    P = pd.DataFrame({"scan_id": t, "t_abs": 1.6e9, "t_rel_s": (t - 1) * 0.1, "px": t * 1e-3 + shift, "py": 0.0, "pz": 0.0, "yaw_rad": yaw})
    P["qw"], P["qx"], P["qy"], P["qz"] = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
    if qbad == "norm":
        P["qw"] = P["qw"] * 1.01
    if qbad == "yaw":
        P["qz"] = -P["qz"]
    if qbad == "cols":
        P = P.drop(columns=["qx"])
    return P


def fake_refs():
    refs = {}
    for aid, mode, v, ref, extra in M.IDENT:
        rt = lambda df: pd.read_csv(io.StringIO(df.to_csv(index=False)))
        D, P = rt(R._fake_dump(mode)), rt(R._fake_pos())
        refs[aid] = dict(ref_arm=ref, source="R78", pos={c: P[c].tolist() for c in ("scan_id", "px", "py", "pz", "yaw_rad")}, dump_rows=int(len(D)), row_kind_counts=None, dump_col_sha=dump_sigs(D), node_id_sha=node_sig(D))
    return refs


def fake_refs_fb():
    rows = []
    for tab, n in ((M.IDREF_NB, 3980), (M.IDREF_DX, 595)):
        for arm, ref in tab.items():
            P = pd.read_csv(io.StringIO(fake_pos_q(n).to_csv(index=False)))[["scan_id", "px", "py", "pz", "yaw_rad"]]
            P.insert(0, "arm", ref); rows.append(P)
    R_ = pd.concat(rows)
    return R_.drop_duplicates(["arm", "scan_id"])


FIRSTNB = "r79_nb_eee_01_pe_hardabs_v0p35_pfn3_fq"; FIRSTNB2 = "r79_nb_sbs_02_off_v0p425_pfn3_mq"; FIRSTZ = "r79_qv_nya_02_zupt_mq"; FIRSTN = "r79_qv_eee_02_nis13_mq"; FIRSTQ0 = "r79_qv_eee_03_zupt_fq"
FIRSTDXS = "r79_dx_split_off_v0p425_mq"; FIRSTDX = "r79_dx_off_v0p425_pfn3_fq"; FIRSTEC = "r79_ec_n200_pfn1_fq"; FIRSTID = "r79_id_off_v1p0"; FIRSTSM = M.SMOKE_ID


def mkfake(root, bad=None, only=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)
    for t in M.arms():
        arm = t["id"]
        if only is not None and arm not in only:
            continue
        if t["kind"] == "refuse":
            ok = not (bad == "refuse_ran" and arm == "r79_refuse_snapshot_frames_one")
            w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if ok else "DONE"}))
            w(arm, "run_log_tail.txt", "terminate: " + " ... ".join(t["needles"]) + "\n")
            continue
        w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if (bad == "arm_failed" and arm == FIRSTNB) else "DONE"}))
        if not (bad == "no_engimu" and arm == FIRSTNB):
            w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
        w(arm, "state_trace_blank_runid.sha256", ("b" if (bad == "det_fb" and arm == M.FB_DET) or (bad == "det_id" and arm == M.ID_DET) else "a") * 64 + "\n")
        mode, vox = t["mode"], t["voxel"]
        if arm == FIRSTNB and bad == "vox_value":
            vox = 0.425
        layer = t["layer"] + (1 if (arm == FIRSTNB and bad == "layer_value") else 0)
        en, form, rule = M.WANT[mode]
        if arm == FIRSTNB and bad == "rule":
            rule = "absolute" if rule != "absolute" else "smooth"
        enum = M.ENUM[mode] + (1 if (arm == FIRSTNB and bad == "enum") else 0)
        var = ("fit_only" if (arm == FIRSTNB and bad == "variance_cfg") else ("plane_equivalent" if t["pe"] else "fit_only"))
        gcfg = "true" if (arm == FIRSTNB2 and bad == "gap_cfg") else "false"
        pfn = t["pfn"] + (1 if (arm == FIRSTDXS and bad == "pfn_cfg") else 0)
        w(arm, "run.log", "ROS param dump: %s\n[params/voxel_map]\n  map/voxel_size:                  %s\n  map/max_layer:                   %s\n"
                          "  plane/line_residual:             enabled %s / form %s / rule %s / rel_sigmas 3 / shape_ratio 0.1 / enum %d / variance %s\n  plane/plane_gap_floor_en:        %s\n  cbk/lidar/point_filter_num: %d\n" % (
                              ROS_LINE, vox, layer, en, form, rule, enum, var, gcfg, pfn))
        eng = ""
        if t["stream"] == "second":
            fr = 595 if t["kind"] == "fb" else 126
            f2 = fr - (5 if (arm == FIRSTDXS and bad == "stream_fallback") else 0)
            fbk = fr - f2
            mp = 6000.0; mi = (mp if (arm == FIRSTDXS and bad == "stream_same_size") else (mp if t["group"] == "IDN" else 2100.0))
            eng += "\n  lio/iekf_point_stream=second dry_run_point_filter_num=3 frames=%d frames_second=%d frames_fallback=%d mean_points_map=%.6f mean_points_iekf=%.6f" % (fr, f2, fbk, mp, mi)
        if arm == FIRSTDX and bad == "stream_on_default":
            eng += "\n  lio/iekf_point_stream=second dry_run_point_filter_num=3 frames=10 frames_second=10 frames_fallback=0 mean_points_map=6000.0 mean_points_iekf=2000.0"
        if arm == FIRSTID and bad == "info3_on":
            eng += "\n  lio/info_scale mode=block F_rot=3 F_pos=3 iterations=10 mean_trace_ratio=0.333"
        if t["zupt"]:
            sc = 0 if (arm == FIRSTZ and bad == "zupt_inert") else 3900
            sg = 0.05 if (arm == FIRSTZ and bad == "zupt_value") else 0.02
            eng += "\n  lio/zupt mode=apply gyro_thresh=0.02 acc_thresh=0.15 enter_scans=3 sigma_v=%g scans=%d stationary=480 applied=470 mean_vnorm_before(applied)=0.01 max_vnorm_before=0.05" % (sg, sc)
        if arm == FIRSTID and bad == "zupt_on_default":
            eng += "\n  lio/zupt mode=apply gyro_thresh=0.02 acc_thresh=0.15 enter_scans=3 sigma_v=0.02 scans=10 stationary=5 applied=5 mean_vnorm_before(applied)=0.01 max_vnorm_before=0.05"
        w(arm, "engagement.txt", "[engagement]  did each flag actually do anything?" + eng + "\n")
        steps = 0 if (arm == FIRSTN and bad == "nis_noprop") else 4990
        ne1 = (7 if (arm == FIRSTQ0 and bad == "gain_moved") else (12 if t["nis"] else 0))
        if not (arm == FIRSTN and bad == "no_gainline") and not (arm == FIRSTNB and bad == "no_engimu"):
            w(arm, "engagement_imu.txt", "[engagement/imu] propagations=126\n  q_gain steps=%d steps_with_gain_ne_1=%d max_gain_acc=%g max_gain_gyr=%g\n" % (steps, ne1, 1.0 if not ne1 else 1.5, 1.0 if not ne1 else 1.4))
        # dump
        D = None
        if t["dump"]:
            D = R._fake_dump(mode, None, node=0)
            if t["early"]:
                D = pd.concat([D.assign(frame=f, node_id=(1000 * f + np.arange(len(D)) % 40)) for f in FRAMES5 if not (arm == FIRSTDX and bad == "dump_frames" and f == 4)])
            if arm == FIRSTNB and bad == "kinds":
                D = pd.concat([D, pd.DataFrame({"frame": 1, "iter": 0, "idx": 999, "r": 0.0, "row_kind": [2] * 4, "weight": 1.0, "cls": "x", "plane_key": 1, "node_id": 0})])
            if arm == FIRSTID and bad == "id_dump":
                D = D.assign(r=D["r"] + 1e-6)
            D.to_csv(os.path.join(root, arm, "residual_dump_all.csv.gz"), index=False, compression="gzip")
        # pos trace
        n = {"id": 126, "det_id": 126, "smoke": 126}.get(t["kind"], None)
        if n is None:
            n = (595 if t["group"] == "DX" else (100 if t["group"] == "EC" else M.N_SCANS[t["bag"]]))
        if arm == FIRSTNB2 and bad == "short_bag":
            n = int(M.N_SCANS[t["bag"]] * 0.8)
        if arm == FIRSTDX and bad == "dx_len":
            n = 300
        shift = 0.0
        if (arm == FIRSTNB and bad == "ctl_pos") or (arm == FIRSTDXS and bad == "dx_pos"):
            shift = 1e-4
        if t["kind"] == "id":
            P = R._fake_pos(1e-4 if (arm == FIRSTID and bad == "id_pos") else 0.0)
            for c, v in (("qw", np.cos(P["yaw_rad"] / 2)), ("qx", 0.0), ("qy", 0.0), ("qz", np.sin(P["yaw_rad"] / 2))):
                P[c] = v
        else:
            P = fake_pos_q(n, shift, bad if (arm == FIRSTNB2 and bad in ("norm", "yaw", "cols")) else None)
            if arm == FIRSTDXS and bad == "dx_last_only":      # a difference only in the last 3 scans: decided nowhere (INFO), must stay PASS
                P.loc[P.index[-3:], "px"] += 1e-3
        if not (arm == FIRSTSM and bad == "no_pos"):
            P.to_csv(os.path.join(root, arm, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        # early files
        if t["early"]:
            nn = 400
            S = []
            for f in FRAMES5:
                if arm == FIRSTEC and bad == "snap_frame_missing" and f == 3:
                    continue
                ids = (1000 * f + np.arange(nn)) if True else None
                Sf = pd.DataFrame({c: np.arange(nn) % 3 if c == "layer" else np.ones(nn) for c in M.R8.SNAP_COLS})
                Sf["node_id"] = np.arange(nn) + 1000 * f
                Sf["point_count"] = 10.0
                Sf.insert(0, "map_frame", f)
                S.append(Sf)
            S = pd.concat(S)
            if arm == FIRSTDX and bad == "snap_dup":
                S.loc[S.index[:2], "node_id"] = 5
            if arm == FIRSTDXS and bad == "snap_layer":
                S["layer"] = np.arange(len(S)) % 6
            if arm == FIRSTDXS and bad == "snap_join":
                S["node_id"] = S["node_id"] + 7
            if arm == FIRSTDXS and bad == "snap_rows":
                S = S[S["map_frame"] == 1].iloc[:50]
            if not (arm == FIRSTDX and bad == "no_snap"):
                S.to_csv(os.path.join(root, arm, "snapshot_frames_all.csv.gz"), index=False, compression="gzip")
            E = pd.DataFrame({"run_id": "x", "scan_id": np.repeat(np.arange(0, 11), 2), "t_abs": 1.6e9, "phase": ["post_imu", "post_lio"] * 11, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0,
                              "vx": 0.0, "vy": 0.0, "vz": 0.0})
            for i in range(18):
                for j in range(i, 18):
                    E["P_%d_%d" % (i, j)] = 1e-6
            if arm == FIRSTEC and bad == "early_phase":
                E = E[E["phase"] == "post_lio"]
            if arm == FIRSTDX and bad == "early_pcols":
                E = E[[c for c in E.columns if not c.startswith("P_") or c in ("P_0_0", "P_17_17")]]
            E.to_csv(os.path.join(root, arm, "state_trace_early_all.csv.gz"), index=False, compression="gzip")
            U = pd.DataFrame({"frame": np.repeat(np.arange(1, 11), 3), "iter": list(range(3)) * 10, "n_resid": 100, "HtH_00": 1.0, "HtH_55": 1.0, "Htz_0": 0.1, "dtheta_x": 0.0, "dt_x": 0.0})
            if arm == FIRSTSM and bad == "early_frames":
                U = U[U["frame"] != 3]
            U.to_csv(os.path.join(root, arm, "lio_update_diag_early_all.csv.gz"), index=False, compression="gzip")
        if arm == M.MQ_TRACE_ARM and bad != "no_mq":
            Q = pd.DataFrame({"scan_index": np.arange(1, (300 if bad == "mq_short" else 3200)), "t_rel_s": 0.1, "model": "isotropic", "acc_scale": 0.3})
            Q.to_csv(os.path.join(root, arm, "motion_q_scan_all.csv.gz"), index=False, compression="gzip")
        lb = None
        if bad == "pe_no_tilt" and arm == FIRSTNB: lb = bad
        if bad == "pe_fallback" and arm == FIRSTNB: lb = bad
        if bad == "default_moved" and arm == FIRSTNB2: lb = bad
        if bad == "gap_moved_off" and arm == FIRSTNB2: lb = bad
        if bad == "off_has_lines" and arm == FIRSTNB2: lb = bad
        if not (bad == "no_lrows" and arm == FIRSTNB):
            L = R._fake_lrows(mode, t["pe"], False, lb)
            if bad == "no_line_rows_anywhere" and arm == FIRSTNB:
                for c in LINEC:
                    L[c] = 0.0
            L.to_csv(os.path.join(root, arm, "line_rows_per_scan_all.csv.gz"), index=False, compression="gzip")


CASES = {"det_id": "DET r79_own_det_id", "det_fb": "DET r79_own_det_nb", "vox_value": "VOX " + FIRSTNB, "layer_value": "VOX " + FIRSTNB, "rule": "LINE " + FIRSTNB, "enum": "LINE " + FIRSTNB,
         "variance_cfg": "LINE " + FIRSTNB, "gap_cfg": "GAPCFG " + FIRSTNB2, "pfn_cfg": "PFN " + FIRSTDXS, "kinds": "LINE " + FIRSTNB + " dump row kinds", "arm_failed": "ALL", "no_engimu": "ENGIMU " + FIRSTNB,
         "refuse_ran": "REFUSE r79_refuse_snapshot_frames_one", "stream_fallback": "STREAM " + FIRSTDXS, "stream_same_size": "STREAM " + FIRSTDXS, "stream_on_default": "NOSTREAM " + FIRSTDX,
         "info3_on": "NOINFO3 " + FIRSTID, "zupt_inert": "ZUPT " + FIRSTZ, "zupt_value": "ZUPT " + FIRSTZ, "zupt_on_default": "NOZUPT " + FIRSTID, "nis_noprop": "GAIN " + FIRSTN, "no_gainline": "GAIN " + FIRSTN,
         "gain_moved": "NOGAIN " + FIRSTQ0, "norm": "QUAT " + FIRSTNB2, "yaw": "QUAT " + FIRSTNB2, "cols": "QUAT " + FIRSTNB2, "short_bag": "LEN " + FIRSTNB2, "dx_len": "DXLEN " + FIRSTDX,
         "dump_frames": "DUMPFR " + FIRSTDX, "no_snap": "SNAPS " + FIRSTDX, "snap_dup": "SNAPS " + FIRSTDX, "snap_layer": "SNAPS " + FIRSTDXS, "snap_rows": "SNAPS " + FIRSTDXS, "snap_frame_missing": "SNAPS " + FIRSTEC,
         "snap_join": "SNAPJOIN " + FIRSTDXS, "early_phase": "EARLY " + FIRSTEC, "early_pcols": "EARLY " + FIRSTDX, "early_frames": "EARLY " + FIRSTSM, "no_mq": "MQSCAN", "mq_short": "MQSCAN",
         "pe_no_tilt": "PE " + FIRSTNB, "pe_fallback": "PE " + FIRSTNB, "default_moved": "NOPE " + FIRSTNB2, "gap_moved_off": "NOGAP " + FIRSTNB2, "off_has_lines": "LROWS " + FIRSTNB2,
         "no_line_rows_anywhere": "LROWS " + FIRSTNB, "no_lrows": "LROWS " + FIRSTNB, "id_pos": "IDENT " + FIRSTID, "id_dump": "IDENT " + FIRSTID, "ctl_pos": "IDENT-FB " + FIRSTNB, "dx_pos": "IDENT-DX " + FIRSTDXS,
         "no_pos": "QUAT " + FIRSTSM}
INFO_OK = {"dx_last_only": "IDENT-DX " + FIRSTDXS}


def affected(bad, needle, ids):
    """the arms a breakage rewrites: the arm the gate names, a determinism arm and its partner, the first NB arm for ALL, the motion_q arm for MQSCAN."""
    out = {w for w in needle.split() if w in ids}
    for a in list(out):
        if a.startswith("r79_own_det_id_"): out.add("r79_id_" + a[len("r79_own_det_id_"):])
        if a.startswith("r79_own_det_nb_"): out.add("r79_nb_" + a[len("r79_own_det_nb_"):])
    if needle.startswith("DET "):
        out |= {a for a in ids if a.startswith(needle.split()[1])}
        out |= {"r79_id_" + a[len("r79_own_det_id_"):] for a in out if a.startswith("r79_own_det_id_")}
        out |= {"r79_nb_" + a[len("r79_own_det_nb_"):] for a in out if a.startswith("r79_own_det_nb_")}
    if needle == "ALL": out.add(FIRSTNB)
    if needle == "MQSCAN": out.add(M.MQ_TRACE_ARM)
    return out


def selftest():
    fails = []
    refs, refs_fb = fake_refs(), fake_refs_fb()
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good, refs, refs_fb)
        nb = [r for r in g.rows if r[1] not in ("PASS", "INFO")]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb[:10]])
        n = {k: sum(1 for x in M.arms() if x["batch"] == k) for k in ("id", "ref", "ef", "fb")}
        if n != {"id": 5, "ref": 3, "ef": 8, "fb": 100}:
            fails.append("manifest arm counts %s != id 5, ref 3, ef 8, fb 100" % n)
        ids = {x["id"] for x in M.arms()}
        for bad, needle in CASES.items():
            d = os.path.join(t, bad); shutil.copytree(good, d); aff = affected(bad, needle, ids); [shutil.rmtree(os.path.join(d, a), ignore_errors=True) for a in aff]; mkfake(d, bad, only=aff)
            failed = [r[0] for r in run_gates(d, refs, refs_fb).rows if r[1] in ("FAIL", "MISSING")]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL/MISSING a gate containing %r (failed: %s)" % (bad, needle, failed[:5]))
        for bad, needle in INFO_OK.items():
            d = os.path.join(t, bad); shutil.copytree(good, d); aff = affected(bad, needle, ids); [shutil.rmtree(os.path.join(d, a), ignore_errors=True) for a in aff]; mkfake(d, bad, only=aff)
            rows = run_gates(d, refs, refs_fb).rows
            if any(r[1] in ("FAIL", "MISSING") for r in rows) or not any(needle in r[0] and r[1] == "INFO" and "e-0" in r[2] or needle in r[0] and r[1] == "INFO" and "1" in r[2] for r in rows):
                fails.append("%s: a difference in the last scans only must stay PASS with an INFO row (failed: %s)" % (bad, [r[0] for r in rows if r[1] in ("FAIL", "MISSING")][:3]))
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] != "MISSING" for r in run_gates(e, refs, refs_fb).rows):
            fails.append("empty runs dir produced a non-MISSING gate: %s" % [r for r in run_gates(e, refs, refs_fb).rows if r[1] != "MISSING"][:3])
        R.write_identity_csv(g.id_rows, os.path.join(t, "idv.csv"))
        if len(pd.read_csv(os.path.join(t, "idv.csv"))) != 3 + 8 + 4:
            fails.append("identity csv has %d rows, want 15" % len(pd.read_csv(os.path.join(t, "idv.csv"))))
    if fails:
        print("SELFTEST r79_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r79_gates PASS (%d gates PASS/INFO on good data; %d single breakages each flipped their gate to FAIL/MISSING; a difference in the last 3 scans of a 60 s arm stays an INFO row; empty dir all MISSING)" % (len(g.rows), len(CASES)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refs", default=os.path.join(HERE, "..", "..", "r79_manifest", "identity_refs.json"))
    ap.add_argument("--refs-fb", default=os.path.join(HERE, "..", "..", "r79_manifest", "identity_refs_fb.csv.gz"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    refs = json.load(open(a.refs)) if os.path.exists(a.refs) else None
    refs_fb = pd.read_csv(a.refs_fb) if os.path.exists(a.refs_fb) else None
    g = run_gates(a.runs, refs, refs_fb)
    txt = report(g)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    if a.csv:
        R.write_identity_csv(g.id_rows, a.csv)
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
