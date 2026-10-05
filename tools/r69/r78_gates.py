#!/usr/bin/env python3
"""R78 gates: identity of the new build against the R77 arms (identity arms and the 16 full-bag controls), determinism digests, refusals, effective config (root voxel, max layer, line mode,
variance, gap floor, point stream, information scale), engagement of the second stream and of the information scale, the whole-bag line-row gate, the 20-frame dump of the F-457 arms,
the snapshot file of the F-428 arms, completeness.

    python3 tools/r69/r78_gates.py --runs <runs_dir> [--refs r78_manifest/identity_refs.json] [--refs-fb r78_manifest/identity_refs_fb.csv.gz] [--out r78_gates.txt] [--csv r78_identity_vs_old.csv]
    python3 tools/r69/r78_gates.py --selftest      fake arm dirs for every arm of the R78 manifest: good data all PASS, every single breakage FAILs its gate, empty dir all MISSING

runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py). PASS / FAIL / MISSING (absent: never a pass) / INFO (a fact reported without a verdict). Exit 1 on any FAIL.

F-454 fixes (the R77 gate defects, mine):
  (a) the gap-floor config gate matches the APPLICATION's own parameter line only (`^\\s*plane/plane_gap_floor_en:\\s+(true|false)\\s*$`, multi-line): the R77 regex `plane/plane_gap_floor_en:\\s+(\\w+)` also matched the
      ROS parameter dump (`/livo_recon/voxel_map/plane/plane_gap_floor_en: True`) when that came first, giving 9 false FAILs. The selftest writes a ROS-style line BEFORE the application line in every arm and
      shows the old regex reading the wrong value on it (the check fires on the old regex, passes on the new one).
  (b) the frame-1 line-engagement gate is no longer a verdict; the line engagement gate reads the whole-bag log line_rows_per_scan_all.csv.gz: FAIL when no scan of the arm has a line or smooth row (in a mode that
      labels any), INFO line with the share of scans that have one.
Own choices (disclosed): hard relative is not run in R78 (inert in R74), so no INFO exception remains; with the second stream engaged the IEKF set must be smaller than the map set and no frame may fall back;
with the information scale on, F_rot = F_pos = 3 and iterations > 0; every counter that a key off must leave at 0 is a FAIL when it moved.
"""
import argparse, io, json, os, re, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r78_manifest as M  # noqa: E402
from r74_gates import rd, csv, jsn, digest, G, report  # noqa: E402
from r77_identity_refs import dump_sigs, node_sig  # noqa: E402
import r77_gates as R  # noqa: E402  (ident_row, write_identity_csv, fake builders)

MIN_ROWS_FOR_ENGAGEMENT = 50
LINE_RE = R.LINE_RE
GAP_RE = re.compile(r"^\s*plane/plane_gap_floor_en:\s+(true|false)\s*$", re.M)          # F-454 (a): the application's own line only
GAP_RE_R77 = re.compile(r"plane/plane_gap_floor_en:\s+(\w+)")                            # the defective R77 regex, kept ONLY so the selftest can show it misreads a ROS-style line
PFN_RE = re.compile(r"cbk/lidar/point_filter_num:\s+(\d+)")
STREAM_RE = re.compile(r"lio/iekf_point_stream=(\w+) dry_run_point_filter_num=(\d+) frames=(\d+) frames_second=(\d+) frames_fallback=(\d+) mean_points_map=([0-9.eE+-]+) mean_points_iekf=([0-9.eE+-]+)")
INFO_RE = re.compile(r"lio/info_scale mode=block F_rot=([0-9.eE+-]+) F_pos=([0-9.eE+-]+) iterations=(\d+)")
CTR = R.CTR
LINEC = R.LINEC
NOISE = 4e-4
ROS_LINE = "/livo_recon/voxel_map/plane/plane_gap_floor_en: True"


def lrows(runs, arm):
    return csv(os.path.join(runs, arm), "line_rows_per_scan_all.csv.gz")


def eng_text(d):
    return (rd(d, "engagement.txt") or "") + "\n" + (rd(d, "run.log") or "")


def fb_ref_rows(refs_fb, ref_arm):
    if refs_fb is None:
        return None
    r = refs_fb[refs_fb["arm"] == ref_arm]
    return None if r.empty else r


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
        g.add("LINE %s effective config enabled %s / form %s / rule %s (enum %d, rel_sigmas 3, shape_ratio 0.1) / variance %s" % (arm, want[0], want[1], want[2], M.ENUM[t["mode"]], wvar),
              "PASS" if ok else ("MISSING" if not log else "FAIL"), l.group(0) if l else "no line")
        gm = GAP_RE.search(log); wgap = "true" if t["gap"] else "false"
        g.add("GAPCFG %s effective config plane_gap_floor_en %s (application line only)" % (arm, wgap), "PASS" if gm and gm.group(1) == wgap else ("MISSING" if not log else "FAIL"), gm.group(0).strip() if gm else "no line")
        if t["mode"] == "hardrelnoise":
            sg = re.search(r"plane/weight_floor/sigma_r2:\s+([0-9.eE+-]+)", log)
            g.add("NOISE %s sensor noise variance (the floor of the relative_noise test) is 4e-4 in the effective config" % arm,
                  "PASS" if sg and abs(float(sg.group(1)) - NOISE) < 1e-12 else ("MISSING" if not log else "FAIL"), "log: sigma_r2 %s" % (sg and sg.group(1)))
        pm = PFN_RE.search(log)
        g.add("PFN %s map point filter %s in the effective config" % (arm, t["pfn"]), "PASS" if pm and int(pm.group(1)) == t["pfn"] else ("MISSING" if not pm else "FAIL"), "log: %s" % (pm and pm.group(0)))
        et = eng_text(A(arm))
        sm, im = STREAM_RE.search(et), INFO_RE.search(et)
        ident_second = t["kind"] == "id" and t["stream"] == "second" and t["group"] == "IDN"
        if t["stream"] == "second" and not ident_second:
            if not sm:
                g.add("STREAM %s engagement line for lio/iekf_point_stream = second" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                fr, f2, fb, mp, mi = int(sm.group(3)), int(sm.group(4)), int(sm.group(5)), float(sm.group(6)), float(sm.group(7))
                ok = sm.group(1) == "second" and int(sm.group(2)) == 3 and fr > 0 and f2 == fr and fb == 0 and 0 < mi < mp
                g.add("STREAM %s engaged: stream second, dry_run 3, frames_second == frames (%d of %d), no fallback (%d), IEKF set smaller than the map set (%.1f < %.1f)" % (arm, f2, fr, fb, mi, mp),
                      "PASS" if ok else "FAIL", sm.group(0))
                g.add("STREAM %s ratio of the IEKF set to the map set" % arm, "INFO", "mean_points_iekf / mean_points_map = %.4f (%.1f / %.1f); a fact, not a verdict" % (mi / mp if mp else float("nan"), mi, mp))
        elif not ident_second:
            g.add("NOSTREAM %s no second-stream engagement line (key at default)" % arm, "PASS" if not sm and et.strip() else ("MISSING" if not et.strip() else "FAIL"), sm.group(0) if sm else "")
        if ident_second:
            if not sm:
                g.add("STREAM %s engagement line (identity arm with the second stream equal to the primary)" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                fr, f2, fb, mp, mi = int(sm.group(3)), int(sm.group(4)), int(sm.group(5)), float(sm.group(6)), float(sm.group(7))
                g.add("STREAM %s engaged with an equal set: frames_second == frames, no fallback, mean_points_iekf == mean_points_map" % arm,
                      "PASS" if fr > 0 and f2 == fr and fb == 0 and abs(mi - mp) < 1e-9 else "FAIL", sm.group(0))
        if t["info3"]:
            if not im:
                g.add("INFO3 %s engagement line for lio/info_scale mode block" % arm, "MISSING" if not et.strip() else "FAIL", "no line")
            else:
                g.add("INFO3 %s information scale F_rot = F_pos = 3 and applied (iterations %s)" % (arm, im.group(3)),
                      "PASS" if abs(float(im.group(1)) - 3.0) < 1e-12 and abs(float(im.group(2)) - 3.0) < 1e-12 and int(im.group(3)) > 0 else "FAIL", im.group(0))
        else:
            g.add("NOINFO3 %s information scale off (no engagement line)" % arm, "PASS" if not im and et.strip() else ("MISSING" if not et.strip() else "FAIL"), im.group(0) if im else "")
        D = csv(A(arm), "residual_dump_all.csv.gz")
        dump_on = "lio.residual_dump_frames" in t["delta"]
        if dump_on and t["mode"] != "off":
            if D is None or "row_kind" not in D.columns:
                g.add("LINE %s dump has row_kind and consistent row counts" % arm, "MISSING")
            else:
                n = D["row_kind"].value_counts().to_dict(); c = lambda k: int(n.get(k, 0))
                if t["mode"] == "smooth":
                    good = c(0) == c(1) == c(2) == 0 and c(3) == c(4) == c(5) > 0
                    why = "smooth: kinds 3, 4, 5 equal and > 0, no kinds 0-2"
                else:
                    good = c(2) <= c(1) and c(3) == c(4) == c(5) == 0
                    why = "hard: kind2 <= kind1, no smooth kinds"
                g.add("LINE %s dump row kinds consistent (%s)" % (arm, why), "PASS" if good else "FAIL", "counts %s" % dict(sorted(n.items())))
                nl = int((D["row_kind"] != 0).sum())
                g.add("LINE %s frame-1 share of rows labelled line or smooth (a fact; the engagement verdict is the whole-bag LROWS gate)" % arm, "INFO", "%d of %d rows, share %.3f" % (nl, len(D), nl / max(len(D), 1)))
        elif dump_on and D is not None and "row_kind" in D.columns:
            g.add("LINE %s lines off: every dump row is a plane row (row_kind 0)" % arm, "PASS" if int((D["row_kind"] != 0).sum()) == 0 else "FAIL", "%d of %d rows with kind != 0" % (int((D["row_kind"] != 0).sum()), len(D)))
        if t["group"] == "YAW":
            if D is None or "frame" not in D.columns:
                g.add("DUMPFR %s dump covers frames 1 to 20" % arm, "MISSING")
            else:
                fr = sorted(set(int(x) for x in D["frame"].unique()))
                need = list(range(1, 21))
                g.add("DUMPFR %s dump holds rows for every frame 1 to 20 (%d frames present)" % (arm, len(fr)), "PASS" if all(f in fr for f in need) and all(int((D["frame"] == f).sum()) >= MIN_ROWS_FOR_ENGAGEMENT for f in need) else "FAIL",
                      "frames present %s" % (fr[:25],))
        if t["group"] == "SNP":
            S = csv(A(arm), "snapshot_nodes_all.csv.gz")
            if S is None:
                g.add("SNAP %s snapshot_nodes_all.csv.gz with every node" % arm, "MISSING")
            else:
                miss = [c for c in M.SNAP_COLS if c not in S.columns]
                ok = not miss and len(S) >= 500 and int(S["layer"].max()) <= t["layer"] and int(S["layer"].min()) >= 0 and bool((S["point_count"] >= 0).all()) and S["node_id"].is_unique
                g.add("SNAP %s snapshot has the %d columns, >= 500 nodes, layers 0..%d, unique node_id, point_count >= 0" % (arm, len(M.SNAP_COLS), t["layer"]), "PASS" if ok else "FAIL",
                      "rows %d, missing columns %s, layer max %s" % (len(S), miss[:4], S["layer"].max() if "layer" in S.columns else None))
        L = lrows(runs, arm)
        if L is None or any(c not in L.columns for c in CTR + ["sum_tilt_var", "n_plane_rows"] + LINEC):
            g.add("LROWS %s line_rows_per_scan_all.csv.gz present with the R77 columns" % arm, "MISSING")
        else:
            cm = {c: float(L[c].max()) for c in CTR}
            tilt = float(L["sum_tilt_var"].sum()); nlin = float(L[LINEC].to_numpy().sum()); npl = float(L["n_plane_rows"].sum())
            per_scan = L[LINEC].sum(axis=1)
            share = float((per_scan > 0).mean())
            if t["mode"] == "off":
                g.add("LROWS %s lines off: plane rows > 0 and no line rows over %d logged scans" % (arm, len(L)), "PASS" if npl > 0 and nlin == 0 else "FAIL", "plane rows %d, line rows %d" % (npl, nlin))
            else:
                g.add("LROWS %s whole-bag line engagement: some scan has line or smooth rows (%d of %d logged scans)" % (arm, int((per_scan > 0).sum()), len(L)), "PASS" if nlin > 0 else "FAIL",
                      "line rows %d, share of scans %.3f" % (nlin, share))
                g.add("LROWS %s share of logged scans with line or smooth rows (a fact)" % arm, "INFO", "%.3f" % share)
            if t["pe"]:
                g.add("PE %s engaged: tilt_voxels moved (counter max %d) and tilt_fallbacks == 0 (max %d)" % (arm, cm["tilt_voxels"], cm["tilt_fallbacks"]), "PASS" if cm["tilt_voxels"] > 0 and cm["tilt_fallbacks"] == 0 else "FAIL", "counters %s" % cm)
                g.add("PE %s sum_tilt_var over scans > 0 (the tilt term reached the rows)" % arm, "PASS" if tilt > 0 else "FAIL", "sum %.6g" % tilt)
            else:
                g.add("NOPE %s variance at default: tilt counters and tilt sum are exactly 0" % arm, "PASS" if cm["tilt_voxels"] == 0 and cm["tilt_floored_axes"] == 0 and cm["tilt_fallbacks"] == 0 and tilt == 0 else "FAIL",
                      "tilt_voxels %d, floored %d, fallbacks %d, sum %.6g" % (cm["tilt_voxels"], cm["tilt_floored_axes"], cm["tilt_fallbacks"], tilt))
            if t["gap"]:
                g.add("GAP %s engaged: gap_fits > 0 and gap_d1 + gap_d2 > 0 (the floor changed at least one fit)" % arm, "PASS" if cm["gap_fits"] > 0 and cm["gap_d1"] + cm["gap_d2"] > 0 else "FAIL", "counters %s" % cm)
            else:
                g.add("NOGAP %s gap floor off: gap counters are exactly 0" % arm, "PASS" if cm["gap_fits"] == 0 and cm["gap_d1"] == 0 and cm["gap_d2"] == 0 else "FAIL", "gap_fits %d d1 %d d2 %d" % (cm["gap_fits"], cm["gap_d1"], cm["gap_d2"]))
        if t["kind"] == "id" and refs and arm in refs:
            P = csv(A(arm), "pos_trace_all.csv.gz")
            if D is None or P is None:
                g.add("IDENT %s state reproduces %s" % (arm, refs[arm]["ref_arm"]), "MISSING", "dump or pos_trace absent")
            else:
                r = R.ident_row(arm, refs[arm], D, P); id_rows.append(r)
                same = r["pos_max_abs_diff_mm"] == 0.0 and r["yaw_max_abs_diff_mrad"] == 0.0 and r["dump_equal"] == "yes"
                g.add("IDENT %s state reproduces %s" % (arm, r["ref_arm"]), "PASS" if same else "FAIL",
                      "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad over %s common scans, dump (all columns but plane_key, node_id) %s" % (r["pos_max_abs_diff_mm"], r["yaw_max_abs_diff_mrad"], r["n_common_scans"], r["dump_equal"]))
                g.add("IDENT-NODE %s node_id column equal to %s" % (arm, r["ref_arm"]), "INFO", "node_id equal: %s (a pointer-like id; reported, never a verdict)" % r["node_id_equal"])
        if t["group"] == "CTL" and refs_fb is not None:
            ref_arm = t["id"].replace("r78_ct_", "r77_fb_", 1).replace("_off_", "_off_")
            ref_arm = ref_arm.replace("r77_fb_pe_hardabs", "r77_fb_pe_hardabs")
            rr = fb_ref_rows(refs_fb, ref_arm)
            P = csv(A(arm), "pos_trace_all.csv.gz")
            if rr is None or P is None:
                g.add("IDENT-FB %s state reproduces %s" % (arm, ref_arm), "MISSING", "reference rows or pos_trace absent")
            else:
                new = P.drop_duplicates("scan_id").set_index("scan_id"); old = rr.drop_duplicates("scan_id").set_index("scan_id")
                com = old.index.intersection(new.index)
                dp = float(np.max(np.abs(new.loc[com, ["px", "py", "pz"]].to_numpy() - old.loc[com, ["px", "py", "pz"]].to_numpy())) * 1e3) if len(com) else float("nan")
                dy = new.loc[com, "yaw_rad"].to_numpy() - old.loc[com, "yaw_rad"].to_numpy()
                dyaw = float(np.max(np.abs((dy + np.pi) % (2 * np.pi) - np.pi)) * 1e3) if len(com) else float("nan")
                id_rows.append(dict(arm=arm, ref_arm=ref_arm, n_common_scans=int(len(com)), pos_max_abs_diff_mm=dp, yaw_max_abs_diff_mrad=dyaw, dump_equal="n/a", node_id_equal="n/a", digest_equal_if_known="n/a"))
                g.add("IDENT-FB %s state reproduces %s over %d common scans" % (arm, ref_arm, len(com)), "PASS" if len(com) > 1000 and dp == 0.0 and dyaw == 0.0 else "FAIL", "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad" % (dp, dyaw))
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
def ref_of(t):
    """R77 reference arm of an R78 full-bag control."""
    return t["id"].replace("r78_ct_", "r77_fb_", 1)


def fake_refs():
    refs = {}
    for aid, mode, v, ref, extra in M.IDENT:
        rt = lambda df: pd.read_csv(io.StringIO(df.to_csv(index=False)))
        D, P = rt(R._fake_dump(mode)), rt(R._fake_pos())
        refs[aid] = dict(ref_arm=ref, source="R77", pos={c: P[c].tolist() for c in ("scan_id", "px", "py", "pz", "yaw_rad")}, dump_rows=int(len(D)), row_kind_counts=None, dump_col_sha=dump_sigs(D), node_id_sha=node_sig(D))
    return refs


def fake_fb_pos(shift=0.0, n=3980):
    return pd.DataFrame({"scan_id": np.arange(1, n + 1), "t_abs": 1.6e9, "t_rel_s": np.arange(n) * 0.1, "px": np.arange(n) * 1e-3 + shift, "py": 0.0, "pz": 0.0, "yaw_rad": 3.1 + np.arange(n) * 1e-5})


def fake_refs_fb():
    rows = []
    for t in M.arms():
        if t["group"] == "CTL":
            P = pd.read_csv(io.StringIO(fake_fb_pos().to_csv(index=False)))
            P.insert(0, "arm", ref_of(t)); rows.append(P)
    return pd.concat(rows)


FIRST = "r78_sa_pe_hardshape_v0p3_gap1_fq"; FIRSTID = "r78_id_off_v1p0"; FIRSTSPL = "r78_sp_pe_hardabs_v1p0_mq"; FIRSTIF = "r78_if3_off_v0p425_pfn1_fq"; FIRSTCTL = "r78_ct_off_v1p0_pfn3_mq"
FIRSTYAW = "r78_yw_pe_smooth_v0p425_pfn3_mq"; FIRSTSNP = "r78_snap_off_v1p0_pfn3_mq"; FIRSTIDS = "r78_id_second_off_v1p0"; FIRSTGAP0 = "r78_sa_off_v0p25_gap0_fq"


def mkfake(root, bad=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)
    for t in M.arms():
        arm = t["id"]
        if t["kind"] == "refuse":
            ok = not (bad == "refuse_ran" and arm == "r78_refuse_stream_bogus")
            w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if ok else "DONE"}))
            w(arm, "run_log_tail.txt", "terminate: " + " ... ".join(t["needles"]) + "\n")
            continue
        w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if (bad == "arm_failed" and arm == FIRST) else "DONE"}))
        if not (bad == "no_engimu" and arm == FIRST):
            w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
        w(arm, "state_trace_blank_runid.sha256", ("b" if (bad == "det_fb" and arm == M.FB_DET) or (bad == "det_id" and arm == M.ID_DET) else "a") * 64 + "\n")
        mode, vox = t["mode"], t["voxel"]
        if arm == FIRST and bad == "vox_value":
            vox = 0.425
        layer = t["layer"] + (1 if (arm == FIRST and bad == "layer_value") else 0)
        en, form, rule = M.WANT[mode]
        if arm == FIRST and bad == "rule":
            rule = "absolute"
        enum = M.ENUM[mode] + (1 if (arm == FIRST and bad == "enum") else 0)
        pe, gap = t["pe"], t["gap"]
        var = ("fit_only" if (arm == FIRST and bad == "variance_cfg") else ("plane_equivalent" if pe else "fit_only"))
        gcfg = ("false" if (arm == FIRST and bad == "gap_cfg") else ("true" if gap else "false"))
        pfn = t["pfn"] + (1 if (arm == FIRSTSPL and bad == "pfn_cfg") else 0)
        # a ROS-style parameter line BEFORE the application's line, with the opposite capitalisation: the defect P4 of R77 (F-454)
        w(arm, "run.log", "ROS param dump: %s\n[params/voxel_map]\n  map/voxel_size:                  %s\n  map/max_layer:                   %s\n  plane/weight_floor/sigma_r2:     %s\n"
                          "  plane/line_residual:             enabled %s / form %s / rule %s / rel_sigmas 3 / shape_ratio 0.1 / enum %d / variance %s\n  plane/plane_gap_floor_en:        %s\n  cbk/lidar/point_filter_num: %d\n" % (
                              ROS_LINE, vox, layer, 0.0025 if (bad == "sigma" and mode == "hardrelnoise") else NOISE, en, form, rule, enum, var, gcfg, pfn))
        eng = ""
        if t["stream"] == "second":
            fr = 3980 if t["kind"] == "fb" else 126
            f2 = fr - (5 if (arm == FIRSTSPL and bad == "stream_fallback") else 0) - (fr if (arm == FIRSTSPL and bad == "stream_inert") else 0)
            fbk = fr - f2 if bad in ("stream_fallback", "stream_inert") and arm == FIRSTSPL else 0
            mp = 6000.0; mi = (mp if (arm == FIRSTSPL and bad == "stream_same_size") else (mp if t["group"] == "IDN" else 2100.0))
            eng += "\n  lio/iekf_point_stream=second dry_run_point_filter_num=3 frames=%d frames_second=%d frames_fallback=%d mean_points_map=%.6f mean_points_iekf=%.6f" % (fr, f2, fbk, mp, mi)
        if arm == FIRSTCTL and bad == "stream_on_default":
            eng += "\n  lio/iekf_point_stream=second dry_run_point_filter_num=3 frames=10 frames_second=10 frames_fallback=0 mean_points_map=6000.0 mean_points_iekf=2000.0"
        if t["info3"]:
            fval = 10.0 if (arm == FIRSTIF and bad == "info3_value") else 3.0
            it = 0 if (arm == FIRSTIF and bad == "info3_inert") else 9000
            eng += "\n  lio/info_scale mode=block F_rot=%g F_pos=%g iterations=%d mean_trace_ratio=0.333" % (fval, fval, it)
        if arm == FIRSTCTL and bad == "info3_on_default":
            eng += "\n  lio/info_scale mode=block F_rot=3 F_pos=3 iterations=10 mean_trace_ratio=0.333"
        w(arm, "engagement.txt", "[engagement]  did each flag actually do anything?" + eng + "\n")
        if "lio.residual_dump_frames" in t["delta"]:
            D = R._fake_dump(mode, None, node=0)
            if t["group"] == "YAW":
                D = pd.concat([D.assign(frame=f) for f in range(1, 21) if not (arm == FIRSTYAW and bad == "dump_frames" and f == 7)])
            if arm == FIRST and bad == "kinds":
                D = pd.concat([D, pd.DataFrame({"frame": 1, "iter": 0, "idx": 999, "r": 0.0, "row_kind": [2] * 4, "weight": 1.0, "cls": "x", "plane_key": 1, "node_id": 0})])
            if arm == FIRSTID and bad == "id_dump":
                D = D.assign(r=D["r"] + 1e-6)
            if arm == FIRSTID and bad == "id_node":
                D = D.assign(node_id=7)
            D.to_csv(os.path.join(root, arm, "residual_dump_all.csv.gz"), index=False, compression="gzip")
        if t["kind"] == "id":
            R._fake_pos(1e-4 if (arm == FIRSTID and bad == "id_pos") else 0.0).to_csv(os.path.join(root, arm, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        elif t["group"] == "CTL":
            fake_fb_pos(1e-4 if (arm == FIRSTCTL and bad == "ctl_pos") else 0.0).to_csv(os.path.join(root, arm, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        if t["group"] == "SNP":
            n = 2000
            S = pd.DataFrame({c: np.arange(n) % 3 if c == "layer" else np.ones(n) for c in M.SNAP_COLS})
            S["node_id"] = np.arange(n)
            S["point_count"] = 10.0
            if arm == FIRSTSNP and bad == "snap_layer":
                S["layer"] = np.arange(n) % 5
            if arm == FIRSTSNP and bad == "snap_rows":
                S = S.iloc[:100]
            if not (arm == FIRSTSNP and bad == "no_snap"):
                S.to_csv(os.path.join(root, arm, "snapshot_nodes_all.csv.gz"), index=False, compression="gzip")
        lb = None
        if bad == "pe_no_tilt" and arm == FIRST: lb = bad
        if bad == "pe_fallback" and arm == FIRST: lb = bad
        if bad == "gap_inert" and arm == FIRST: lb = bad
        if bad == "default_moved" and arm == FIRSTCTL: lb = bad
        if bad == "gap_moved_off" and arm == FIRSTGAP0: lb = bad
        if bad == "off_has_lines" and arm == FIRSTCTL: lb = bad
        if bad == "no_line_rows_anywhere" and arm == FIRST: lb = bad
        if not (bad == "no_lrows" and arm == FIRST):
            L = R._fake_lrows(mode, pe, gap, lb if lb != "no_line_rows_anywhere" else None)
            if lb == "no_line_rows_anywhere":
                for c in LINEC:
                    L[c] = 0.0
            L.to_csv(os.path.join(root, arm, "line_rows_per_scan_all.csv.gz"), index=False, compression="gzip")


CASES = {"det_id": "DET r78_own_det_id", "det_fb": "DET r78_own_det_fb", "vox_value": "VOX " + FIRST + " root voxel", "layer_value": "VOX " + FIRST + " root voxel",
         "rule": "LINE " + FIRST + " effective config", "enum": "LINE " + FIRST + " effective config", "variance_cfg": "LINE " + FIRST + " effective config", "gap_cfg": "GAPCFG " + FIRST,
         "pfn_cfg": "PFN " + FIRSTSPL, "sigma": "NOISE r78_sa_pe_hardrelnoise", "kinds": "LINE " + FIRST + " dump row kinds", "arm_failed": "ALL", "no_engimu": "ENGIMU " + FIRST,
         "refuse_ran": "REFUSE r78_refuse_stream_bogus", "stream_fallback": "STREAM " + FIRSTSPL, "stream_inert": "STREAM " + FIRSTSPL, "stream_same_size": "STREAM " + FIRSTSPL,
         "stream_on_default": "NOSTREAM " + FIRSTCTL, "info3_value": "INFO3 " + FIRSTIF, "info3_inert": "INFO3 " + FIRSTIF, "info3_on_default": "NOINFO3 " + FIRSTCTL,
         "dump_frames": "DUMPFR " + FIRSTYAW, "snap_layer": "SNAP " + FIRSTSNP, "snap_rows": "SNAP " + FIRSTSNP, "no_snap": "SNAP " + FIRSTSNP,
         "pe_no_tilt": "PE " + FIRST, "pe_fallback": "PE " + FIRST, "gap_inert": "GAP " + FIRST, "default_moved": "NOPE " + FIRSTCTL, "gap_moved_off": "NOGAP " + FIRSTGAP0,
         "off_has_lines": "LROWS " + FIRSTCTL, "no_line_rows_anywhere": "LROWS " + FIRST, "id_pos": "IDENT " + FIRSTID, "id_dump": "IDENT " + FIRSTID, "ctl_pos": "IDENT-FB " + FIRSTCTL, "no_lrows": "LROWS " + FIRST}
INFO_CASES = {"id_node": "IDENT-NODE " + FIRSTID}


def selftest():
    fails = []
    refs, refs_fb = fake_refs(), fake_refs_fb()
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good, refs, refs_fb)
        nb = [r for r in g.rows if r[1] not in ("PASS", "INFO")]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb[:10]])
        n = {k: sum(1 for x in M.arms() if x["batch"] == k) for k in ("id", "ref", "fb")}
        if n != {"id": 6, "ref": 3, "fb": 121}:
            fails.append("manifest arm counts %s != id 6, ref 3, fb 121" % n)
        # F-454 (a): the old regex misreads the ROS-style line that the good data carries first; the new one reads the application's line
        log = open(os.path.join(good, FIRST, "run.log")).read()
        if GAP_RE_R77.search(log).group(1) != "True" or GAP_RE.search(log).group(1) != "true":
            fails.append("the ROS-style line did not make the old regex misread (old %s, new %s)" % (GAP_RE_R77.search(log).group(1), GAP_RE.search(log).group(1)))
        for bad, needle in CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            failed = [r[0] for r in run_gates(d, refs, refs_fb).rows if r[1] in ("FAIL", "MISSING")]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL/MISSING a gate containing %r (failed: %s)" % (bad, needle, failed[:5]))
        for bad, needle in INFO_CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            rows = run_gates(d, refs, refs_fb).rows
            if not any(needle in r[0] and r[1] == "INFO" and "equal: no" in r[2] for r in rows) or any(r[1] == "FAIL" for r in rows):
                fails.append("node_id difference %s should show INFO 'equal: no' and nothing FAIL" % bad)
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] != "MISSING" for r in run_gates(e, refs, refs_fb).rows):
            fails.append("empty runs dir produced a non-MISSING gate: %s" % [r for r in run_gates(e, refs, refs_fb).rows if r[1] != "MISSING"][:3])
        R.write_identity_csv(g.id_rows, os.path.join(t, "idv.csv"))
        if len(pd.read_csv(os.path.join(t, "idv.csv"))) != 3 + 16:
            fails.append("identity csv has %d rows, want 19" % len(pd.read_csv(os.path.join(t, "idv.csv"))))
    if fails:
        print("SELFTEST r78_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r78_gates PASS (%d gates PASS/INFO on good data; %d single breakages each flipped their gate to FAIL/MISSING; %d node_id difference stays INFO; the old gap regex misreads the ROS-style line; empty dir all MISSING)" % (
        len(g.rows), len(CASES), len(INFO_CASES)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refs", default=os.path.join(HERE, "..", "..", "r78_manifest", "identity_refs.json"))
    ap.add_argument("--refs-fb", default=os.path.join(HERE, "..", "..", "r78_manifest", "identity_refs_fb.csv.gz"))
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
