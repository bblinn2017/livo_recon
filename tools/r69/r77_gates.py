#!/usr/bin/env python3
"""R77 gates (F-452): identity of the new build with every new key at its default against the R76 arms, determinism digests, refusals, effective config (line mode, variance, gap floor),
row kinds, engagement of the new counters (line_rows_per_scan.csv), completeness.

    python3 tools/r69/r77_gates.py --runs <runs_dir> [--refs r77_manifest/identity_refs.json] [--out r77_gates.txt] [--csv r77_identity_vs_old.csv]
    python3 tools/r69/r77_gates.py --selftest      fake arm dirs for every arm of the R77 manifest: good data all PASS, every single breakage FAILs its gate, empty dir all MISSING

runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py). PASS / FAIL / MISSING (absent: never a pass) / INFO (a fact reported without a verdict). Exit 1 on any FAIL.
F-452 fixes (R76 defects): (P1) the frame-1 dump minimum is 50 rows, not 100 (two whole-bag arms had 94); (P2) node_id is compared as its own INFO row and no longer decides the identity verdict:
the state (position, yaw per scan) and the dump columns except node_id and plane_key decide PASS / FAIL.
Own choices (disclosed): the engagement gates for hardrel (line rows, tilt sum) are INFO, not FAIL (R74 measured it as inert); engagement of the new variance and of the gap floor is FAIL when the
counters did not move ('inert when on'); with the keys off or at default every R77 counter and the tilt sum must be exactly 0.
"""
import argparse, io, json, os, re, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r77_manifest as M  # noqa: E402
from r74_gates import rd, csv, jsn, digest, G, report  # noqa: E402
from r77_identity_refs import dump_sigs, node_sig  # noqa: E402

MIN_ROWS_FOR_ENGAGEMENT = 50
LINE_RE = re.compile(r"plane/line_residual:\s+enabled (\w+) / form (\w+) / rule (\w+) / rel_sigmas ([0-9.eE+-]+) / shape_ratio ([0-9.eE+-]+) / enum (\d+) / variance (\w+)")
GAP_RE = re.compile(r"plane/plane_gap_floor_en:\s+(\w+)")
CTR = ["gap_fits", "gap_d1", "gap_d2", "tilt_voxels", "tilt_floored_axes", "tilt_fallbacks"]
LINEC = ["n_line_u0", "n_line_u1", "n_smooth_u0", "n_smooth_u1", "n_smooth_u2"]


def pe_of(arm):
    """True when the arm runs line_residual/variance = plane_equivalent (by arm name: _pe_ in the full-bag, smoke and sweep names)."""
    return "_pe_" in arm


def gap_of(arm):
    return "gap" in arm.split("r77_", 1)[-1] and "refuse" not in arm


def info_of(t):
    return dict(mode=t["mode"], voxel=t["voxel"])


def ident_row(arm, ref, D, P):
    """One row of r77_identity_vs_old.csv from the new arm's dump D and pos trace P against the reference dict ref (None parts -> NaN, never invented).
    dump_equal covers every dump column except plane_key and node_id; node_id_equal is its own column (F-452)."""
    r = dict(arm=arm, ref_arm=ref["ref_arm"], n_common_scans=np.nan, pos_max_abs_diff_mm=np.nan, yaw_max_abs_diff_mrad=np.nan, dump_equal="", node_id_equal="n/a",
             digest_equal_if_known="n/a (the reference arm wrote no digest)")
    if P is not None:
        old = pd.DataFrame(ref["pos"]).set_index("scan_id"); new = P.drop_duplicates("scan_id").set_index("scan_id")
        com = old.index.intersection(new.index)
        r["n_common_scans"] = int(len(com))
        if len(com):
            r["pos_max_abs_diff_mm"] = float(np.max(np.abs(new.loc[com, ["px", "py", "pz"]].to_numpy() - old.loc[com, ["px", "py", "pz"]].to_numpy())) * 1e3)
            dy = new.loc[com, "yaw_rad"].to_numpy() - old.loc[com, "yaw_rad"].to_numpy()
            r["yaw_max_abs_diff_mrad"] = float(np.max(np.abs((dy + np.pi) % (2 * np.pi) - np.pi)) * 1e3)
    if D is not None:
        ns = dump_sigs(D)
        common = [c for c in ns if c in ref["dump_col_sha"]]
        bad = [c for c in common if ns[c] != ref["dump_col_sha"][c]]
        same_n = len(D) == ref["dump_rows"]
        r["dump_equal"] = "yes" if (same_n and not bad and common) else "no: rows %d vs %d%s" % (len(D), ref["dump_rows"], ("; columns differing: " + ",".join(bad[:8])) if bad else "")
        if ref.get("node_id_sha") is not None and node_sig(D) is not None:
            r["node_id_equal"] = "yes" if node_sig(D) == ref["node_id_sha"] else "no"
    return r


def lrows(runs, arm):
    """line_rows_per_scan_all.csv.gz of the arm as a DataFrame (None when absent)."""
    return csv(os.path.join(runs, arm), "line_rows_per_scan_all.csv.gz")


def run_gates(runs, refs=None):
    g = G()
    A = lambda a: os.path.join(runs, a)
    table = M.arms()
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    notdone = []
    # ---- determinism controls (rule 3.7)
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
        ok = bool(mv and ml) and abs(float(mv.group(1)) - t["voxel"]) < 1e-12 and int(ml.group(1)) == 2
        g.add("VOX %s root voxel %s max_layer 2 in the effective config" % (arm, t["voxel"]), "PASS" if ok else ("MISSING" if not log else "FAIL"), "log: voxel %s max_layer %s" % (mv and mv.group(1), ml and ml.group(1)))
        want = M.WANT[t["mode"]]
        wvar = "plane_equivalent" if pe_of(arm) else "fit_only"
        l = LINE_RE.search(log)
        ok = bool(l) and (l.group(1), l.group(2), l.group(3)) == want and abs(float(l.group(4)) - 3.0) < 1e-12 and abs(float(l.group(5)) - 0.1) < 1e-12 and int(l.group(6)) == M.ENUM[t["mode"]] and l.group(7) == wvar
        g.add("LINE %s effective config enabled %s / form %s / rule %s (enum %d, rel_sigmas 3, shape_ratio 0.1) / variance %s" % (arm, want[0], want[1], want[2], M.ENUM[t["mode"]], wvar),
              "PASS" if ok else ("MISSING" if not log else "FAIL"), l.group(0) if l else "no line")
        gm = GAP_RE.search(log); wgap = "true" if gap_of(arm) else "false"
        g.add("GAPCFG %s effective config plane_gap_floor_en %s" % (arm, wgap), "PASS" if gm and gm.group(1) == wgap else ("MISSING" if not log else "FAIL"), gm.group(0) if gm else "no line")
        if t["mode"] == "hardrelnoise":
            sg = re.search(r"plane/weight_floor/sigma_r2:\s+([0-9.eE+-]+)", log)
            g.add("NOISE %s sensor noise variance (the floor of the relative_noise test) is 4e-4 in the effective config" % arm,
                  "PASS" if sg and abs(float(sg.group(1)) - 4e-4) < 1e-12 else ("MISSING" if not log else "FAIL"), "log: sigma_r2 %s" % (sg and sg.group(1)))
        D = csv(A(arm), "residual_dump_all.csv.gz")
        if "lio.residual_dump_frames" not in t["delta"]:
            pass   # the sweep arms and the lines-off controls dump nothing: no row-kind gate applies
        elif t["mode"] != "off":
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
                bad = len(D) >= MIN_ROWS_FOR_ENGAGEMENT and nl == 0
                g.add("LINE %s engagement: the mode labelled some rows as line/smooth (frame 1)" % arm, ("INFO" if t["mode"] == "hardrel" else "FAIL") if bad else "PASS",
                      "%d of %d rows (kind != 0), share %.3f%s" % (nl, len(D), nl / max(len(D), 1), "; hardrel was inert in R74, zero is an expected result" if bad and t["mode"] == "hardrel" else ""))
        elif D is not None and "row_kind" in D.columns:
            g.add("LINE %s lines off: every dump row is a plane row (row_kind 0)" % arm, "PASS" if int((D["row_kind"] != 0).sum()) == 0 else "FAIL", "%d of %d rows with kind != 0" % (int((D["row_kind"] != 0).sum()), len(D)))
        # ---- the F-451 log and the engagement of the new counters (F-453 variance, F-433 gap floor)
        L = lrows(runs, arm)
        if L is None or any(c not in L.columns for c in CTR + ["sum_tilt_var", "n_plane_rows"] + LINEC):
            g.add("LROWS %s line_rows_per_scan_all.csv.gz present with the R77 columns" % arm, "MISSING")
        else:
            cm = {c: float(L[c].max()) for c in CTR}
            tilt = float(L["sum_tilt_var"].sum()); nlin = float(L[LINEC].to_numpy().sum()); npl = float(L["n_plane_rows"].sum())
            g.add("LROWS %s rows by kind match the mode (plane rows > 0%s)" % (arm, "; no line rows" if t["mode"] == "off" else "; line rows > 0"),
                  ("PASS" if (npl > 0 or t["mode"] == "smooth") and nlin == 0 else "FAIL") if t["mode"] == "off" else
                  ("PASS" if nlin > 0 else ("INFO" if t["mode"] == "hardrel" else "FAIL")), "%d scans, plane rows %d, line rows %d" % (len(L), npl, nlin))
            if pe_of(arm):
                g.add("PE %s engaged: tilt_voxels moved (counter max %d) and tilt_fallbacks == 0 (max %d)" % (arm, cm["tilt_voxels"], cm["tilt_fallbacks"]), "PASS" if cm["tilt_voxels"] > 0 and cm["tilt_fallbacks"] == 0 else "FAIL", "counters %s" % cm)
                g.add("PE %s sum_tilt_var over scans > 0 (the tilt term reached the rows)" % arm, "PASS" if tilt > 0 else ("INFO" if t["mode"] == "hardrel" else "FAIL"), "sum %.6g" % tilt)
            else:
                g.add("NOPE %s variance at default: tilt counters and tilt sum are exactly 0" % arm, "PASS" if cm["tilt_voxels"] == 0 and cm["tilt_floored_axes"] == 0 and cm["tilt_fallbacks"] == 0 and tilt == 0 else "FAIL",
                      "tilt_voxels %d, floored %d, fallbacks %d, sum %.6g" % (cm["tilt_voxels"], cm["tilt_floored_axes"], cm["tilt_fallbacks"], tilt))
            if gap_of(arm):
                g.add("GAP %s engaged: gap_fits > 0 and gap_d1 + gap_d2 > 0 (the floor changed at least one fit)" % arm, "PASS" if cm["gap_fits"] > 0 and cm["gap_d1"] + cm["gap_d2"] > 0 else "FAIL", "counters %s" % cm)
            else:
                g.add("NOGAP %s gap floor off: gap counters are exactly 0" % arm, "PASS" if cm["gap_fits"] == 0 and cm["gap_d1"] == 0 and cm["gap_d2"] == 0 else "FAIL", "gap_fits %d d1 %d d2 %d" % (cm["gap_fits"], cm["gap_d1"], cm["gap_d2"]))
        if t["kind"] == "id" and refs and arm in refs:
            P = csv(A(arm), "pos_trace_all.csv.gz")
            if D is None or P is None:
                g.add("IDENT %s state reproduces %s" % (arm, refs[arm]["ref_arm"]), "MISSING", "dump or pos_trace absent")
            else:
                r = ident_row(arm, refs[arm], D, P); id_rows.append(r)
                same = r["pos_max_abs_diff_mm"] == 0.0 and r["yaw_max_abs_diff_mrad"] == 0.0 and r["dump_equal"] == "yes"
                g.add("IDENT %s state reproduces %s" % (arm, r["ref_arm"]), "PASS" if same else "FAIL",
                      "pos max |diff| %.6g mm, yaw max |diff| %.6g mrad over %s common scans, dump (all columns but plane_key, node_id) %s" % (r["pos_max_abs_diff_mm"], r["yaw_max_abs_diff_mrad"], r["n_common_scans"], r["dump_equal"]))
                g.add("IDENT-NODE %s node_id column equal to %s" % (arm, r["ref_arm"]), "INFO", "node_id equal: %s (a pointer-like id; differed at voxel 1.0 in R76; reported, never a verdict)" % r["node_id_equal"])
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


def write_identity_csv(rows, path):
    cols = ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


# --------------------------------------------------------------------------- selftest
def _fake_dump(mode, bad=None, node=0):
    kinds = {"off": [0] * 150, "smooth": [3, 4, 5] * 60, "hardrel": [0] * 150}.get(mode, [0] * 50 + [1, 2] * 5 + [1] * 3)
    if mode == "hardabs" and bad == "no_lines":
        kinds = [0] * 150
    return pd.DataFrame({"frame": 1, "iter": 0, "idx": range(len(kinds)), "r": np.linspace(-0.01, 0.01, len(kinds)), "row_kind": kinds, "weight": 1000.0, "cls": "ground", "plane_key": 123, "node_id": node})


def _fake_pos(shift=0.0):
    n = 126
    return pd.DataFrame({"scan_id": np.arange(1, n + 1), "t_abs": 1.6e9, "t_rel_s": np.arange(n) * 0.1, "px": np.arange(n) * 1e-3 + shift, "py": 0.0, "pz": 0.0, "yaw_rad": 3.1 + np.arange(n) * 1e-4})


def _fake_lrows(mode, pe, gap, bad=None):
    n = 60
    z = np.zeros(n)
    ramp = np.arange(1, n + 1)
    kinds = {"off": (0, 0, 0, 0, 0), "smooth": (0, 0, 40, 40, 40), "hardrel": (0, 0, 0, 0, 0)}.get(mode, (30, 30, 0, 0, 0))
    D = {"scan_id": np.arange(1, n + 1), "iter": 0, "n_rows": 200, "n_points": 100, "n_plane_rows": (100 if mode != "smooth" else 0) + z,
         "n_line_u0": kinds[0] + z, "n_line_u1": kinds[1] + z, "n_smooth_u0": kinds[2] + z, "n_smooth_u1": kinds[3] + z, "n_smooth_u2": kinds[4] + z, "n_pts_with_line_rows": 30 + z,
         "sum_fit_var_line": 1e-3 + z, "sum_tilt_var": (2e-4 if pe and mode != "hardrel" else 0.0) + z, "sum_sigma2_line": 5e-3 + z, "sum_sigma2_plane": 4e-3 + z,
         "gap_fits": (3 * ramp if gap else z), "gap_d1": (2 * ramp if gap else z), "gap_d2": (1 * ramp if gap else z),
         "tilt_voxels": (10 * ramp if pe else z), "tilt_floored_axes": (ramp if pe else z), "tilt_fallbacks": z.copy()}
    if mode == "smooth":
        D["n_plane_rows"] = z
    if bad == "pe_no_tilt":
        D["tilt_voxels"] = z; D["sum_tilt_var"] = z
    if bad == "pe_fallback":
        D["tilt_fallbacks"] = ramp
    if bad == "gap_inert":
        D["gap_fits"] = z; D["gap_d1"] = z; D["gap_d2"] = z
    if bad == "default_moved":
        D["tilt_voxels"] = ramp
    if bad == "gap_moved_off":
        D["gap_d1"] = ramp
    if bad == "off_has_lines":
        D["n_line_u0"] = 5 + z
    return pd.DataFrame(D)


def fake_refs():
    refs = {}
    for aid, mode, v, ref in M.IDENT:
        rt = lambda df: pd.read_csv(io.StringIO(df.to_csv(index=False)))   # through CSV text, as the real reference (a float parsed from text) and the new arm's file both are
        D, P = rt(_fake_dump(mode)), rt(_fake_pos())
        refs[aid] = dict(ref_arm=ref, source="R76", pos={c: P[c].tolist() for c in ("scan_id", "px", "py", "pz", "yaw_rad")}, dump_rows=int(len(D)), row_kind_counts=None, dump_col_sha=dump_sigs(D), node_id_sha=node_sig(D))
    return refs


FIRST = "r77_fb_pe_hardshape_v1p0_pfn1_fq"; FIRSTID = "r77_id_hardabs_v1p0"; FIRSTGAP = "r77_fb_gap_v1p0_pfn1_fq"; FIRSTOFF = "r77_fb_off_v1p0_pfn1_fq"; FIRSTSM = "r77_sm_pe_smooth_v0p425"


def mkfake(root, bad=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)
    for t in M.arms():
        arm = t["id"]
        if t["kind"] == "refuse":
            ok = not (bad == "refuse_ran" and arm == "r77_refuse_variance_bogus")
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
        en, form, rule = M.WANT[mode]
        if arm == FIRST and bad == "rule":
            rule = "absolute"
        enum = M.ENUM[mode] + (1 if (arm == FIRST and bad == "enum") else 0)
        rs, sr = (1.0 if (arm == FIRST and bad == "rel_sigmas") else 3.0), (0.5 if (arm == FIRST and bad == "shape_ratio") else 0.1)
        pe, gap = pe_of(arm), gap_of(arm)
        var = ("fit_only" if (arm == FIRST and bad == "variance_cfg") else ("plane_equivalent" if pe else "fit_only"))
        gcfg = ("false" if (arm == FIRSTGAP and bad == "gap_cfg") else ("true" if gap else "false"))
        w(arm, "run.log", "[params/voxel_map]\n  map/voxel_size:                  %s\n  map/max_layer:                   2\n  plane/weight_floor/sigma_r2:     %s\n"
                          "  plane/line_residual:             enabled %s / form %s / rule %s / rel_sigmas %s / shape_ratio %s / enum %d / variance %s\n  plane/plane_gap_floor_en:        %s\n" % (
                              vox, 0.0025 if (bad == "sigma" and mode == "hardrelnoise") else 0.0004, en, form, rule, rs, sr, enum, var, gcfg))
        if "lio.residual_dump_frames" in t["delta"]:
            D = _fake_dump(mode, bad if arm == FIRSTID else None, node=(7 if (arm == FIRSTID and bad == "id_node") else 0))
            if arm == FIRST and bad == "kinds":
                D = pd.concat([D, pd.DataFrame({"frame": 1, "iter": 0, "idx": 999, "r": 0.0, "row_kind": [2] * 4, "weight": 1.0, "cls": "x", "plane_key": 1, "node_id": 0})])
            if arm == FIRST and bad == "smooth_kinds":
                D = pd.concat([D, pd.DataFrame({"frame": 1, "iter": 0, "idx": 999, "r": 0.0, "row_kind": [3, 4, 5], "weight": 1.0, "cls": "x", "plane_key": 1, "node_id": 0})])
            if arm == FIRSTID and bad == "id_dump":
                D = D.assign(r=D["r"] + 1e-6)
            if arm == FIRSTGAP and bad == "gap_dump_lines":
                D = D.assign(row_kind=[1] + [0] * (len(D) - 1))
            D.to_csv(os.path.join(root, arm, "residual_dump_all.csv.gz"), index=False, compression="gzip")
        if t["kind"] == "id":
            _fake_pos(1e-4 if (arm == FIRSTID and bad == "id_pos") else 0.0).to_csv(os.path.join(root, arm, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        lb = None
        if bad in ("pe_no_tilt", "pe_fallback") and arm == FIRST: lb = bad
        if bad == "gap_inert" and arm == FIRSTGAP: lb = bad
        if bad == "default_moved" and arm == FIRSTOFF: lb = bad
        if bad == "gap_moved_off" and arm == FIRSTID: lb = bad
        if bad == "off_has_lines" and arm == FIRSTOFF: lb = bad
        if not (bad == "no_lrows" and arm == FIRST):
            _fake_lrows(mode, pe, gap, lb).to_csv(os.path.join(root, arm, "line_rows_per_scan_all.csv.gz"), index=False, compression="gzip")


CASES = {"det_id": "DET r77_own_det_id", "det_fb": "DET r77_own_det_fb", "vox_value": "VOX " + FIRST + " root voxel", "rule": "LINE " + FIRST + " effective config",
         "enum": "LINE " + FIRST + " effective config", "rel_sigmas": "LINE " + FIRST + " effective config", "shape_ratio": "LINE " + FIRST + " effective config",
         "variance_cfg": "LINE " + FIRST + " effective config", "gap_cfg": "GAPCFG " + FIRSTGAP,
         "kinds": "LINE " + FIRST + " dump row kinds", "smooth_kinds": "LINE " + FIRST + " dump row kinds", "sigma": "NOISE r77_fb_pe_hardrelnoise",
         "no_lines": "LINE " + FIRSTID + " engagement", "arm_failed": "ALL", "no_engimu": "ENGIMU " + FIRST, "refuse_ran": "REFUSE r77_refuse_variance_bogus",
         "pe_no_tilt": "PE " + FIRST, "pe_fallback": "PE " + FIRST, "gap_inert": "GAP " + FIRSTGAP, "default_moved": "NOPE " + FIRSTOFF, "gap_moved_off": "NOGAP " + FIRSTID,
         "off_has_lines": "LROWS " + FIRSTOFF, "gap_dump_lines": "GAPCFG-NO", "id_pos": "IDENT " + FIRSTID, "id_dump": "IDENT " + FIRSTID, "no_lrows": "LROWS " + FIRST}
CASES["gap_dump_lines"] = "lines off: every dump row is a plane row"
INFO_CASES = {"id_node": "IDENT-NODE " + FIRSTID}   # node_id differences are facts: the row is INFO and nothing FAILs (F-452, defect P2)


def selftest():
    fails = []
    refs = fake_refs()
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good, refs)
        nb = [r for r in g.rows if r[1] not in ("PASS", "INFO")]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb[:10]])
        if any(r[1] == "INFO" and not (r[0].startswith("IDENT-NODE") or r[0].startswith("LINE") or r[0].startswith("LROWS") or r[0].startswith("PE")) for r in g.rows):
            fails.append("good data: an unexpected INFO row")
        n = {k: sum(1 for x in M.arms() if x["batch"] == k) for k in ("id", "ref", "fb")}
        if n != {"id": 10, "ref": 5, "fb": 73}:
            fails.append("manifest arm counts %s != id 10, ref 5, fb 73" % n)
        for bad, needle in CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            failed = [r[0] for r in run_gates(d, refs).rows if r[1] in ("FAIL", "MISSING")]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL/MISSING a gate containing %r (failed: %s)" % (bad, needle, failed[:5]))
        for bad, needle in INFO_CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            rows = run_gates(d, refs).rows
            if not any(needle in r[0] and r[1] == "INFO" and "equal: no" in r[2] for r in rows) or any(r[1] == "FAIL" for r in rows):
                fails.append("node_id difference %s should show INFO 'equal: no' and nothing FAIL" % bad)
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] != "MISSING" for r in run_gates(e, refs).rows):
            fails.append("empty runs dir produced a non-MISSING gate: %s" % [r for r in run_gates(e, refs).rows if r[1] != "MISSING"][:3])
        write_identity_csv(g.id_rows, os.path.join(t, "idv.csv"))
        if len(pd.read_csv(os.path.join(t, "idv.csv"))) != 6:
            fails.append("identity csv has %d rows, want 6" % len(pd.read_csv(os.path.join(t, "idv.csv"))))
    if fails:
        print("SELFTEST r77_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r77_gates PASS (%d gates PASS/INFO on good data; %d single breakages each flipped their gate to FAIL/MISSING; %d node_id difference stays INFO; empty dir all MISSING)" % (len(g.rows), len(CASES), len(INFO_CASES)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refs", default=os.path.join(HERE, "..", "..", "r77_manifest", "identity_refs.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    refs = json.load(open(a.refs)) if os.path.exists(a.refs) else None
    g = run_gates(a.runs, refs)
    txt = report(g)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    if a.csv:
        write_identity_csv(g.id_rows, a.csv)
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
