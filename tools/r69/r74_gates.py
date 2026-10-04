#!/usr/bin/env python3
"""R74 gates: identity and determinism, engagement of the new options (voxel size, line mode, process-noise switch), plane test values, F-425 opt-in, refusals, completeness.

    python3 tools/r69/r74_gates.py --runs <runs_dir> [--out r74_gates.txt]     runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py)
    python3 tools/r69/r74_gates.py --selftest                                   builds fake arm dirs for every arm of the manifest: every gate PASS on good data, every single breakage FAILs its gate

PASS / FAIL / MISSING (arm or file absent: never a pass) / INFO (a fact reported without a verdict). Exit 1 if any FAIL.
The expected arms come from make_r74_manifest.arms() (the same list that built the job files), so a gate cannot silently skip an arm.
Decision recorded (Bryce 2026-10-01, F-419): identity compares ATE, excursion and settled offset with a 0.005 mm tolerance; digest equality with R73 is a fact (INFO), not a verdict.
"""
import argparse, gzip, io, json, os, re, shutil, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402

TOL_MM = 0.005
R72_60S = {"excursion_mm": 82.00350294412522, "settled_offset_mm": 65.33203761816637}   # r73_00_id_baseline_60s, equal to R72 within the F-419 tolerance
R73_DIGEST_60S = "6cd2d754"
ID_ARM = "r74_own_id_baseline_60s"
DET_ARM, DET_ORIG = "r74_own_det_v1p0_ml2_n1600_dsoff_pfn3_fq", "r74_vox_v1p0_ml2_n1600_dsoff_pfn3_fq"
F1_ON_PREFIXES = ("r74_vox_", "r74_line_", "r74_own_smoke_", "r74_own_det_")   # arms whose arm_table f417 column is 1 (M.arms: f417 = 1); every other arm must carry no F-417 file
F410_ARM, F410_ABORT_SCAN = "r74_f410_meas_k8_mq_full", 1130


def rd(d, n):
    p = os.path.join(d, n)
    if os.path.exists(p):
        return gzip.open(p, "rt", errors="replace").read() if p.endswith(".gz") else open(p, errors="replace").read()
    if os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", "rt", errors="replace").read()
    return None


def csv(d, n):
    t = rd(d, n)
    return pd.read_csv(io.StringIO(t)) if t else None


def jsn(d, n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else None


def line(txt, tag):
    for l in (txt or "").splitlines():
        if tag in l:
            return l
    return None


class G:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))


def digest(runs, arm):
    return (rd(os.path.join(runs, arm), "state_trace_blank_runid.sha256") or "").strip() or None


def parse_arm(arm):
    """-> dict(voxel, ml, variant, kind) from the arm name."""
    m = re.match(r"r74_vox_v([0-9p]+)_ml(\d+)_", arm)
    if m:
        return dict(kind="vox", voxel=float(m.group(1).replace("p", ".")), ml=int(m.group(2)), variant=None)
    m = re.match(r"r74_line_(hardabs|hardrel|smooth)_v([0-9p]+)_", arm)
    if m:
        return dict(kind="line", voxel=float(m.group(2).replace("p", ".")), ml=2, variant=m.group(1))
    return None


def run_gates(runs):
    g = G()
    A = lambda a: os.path.join(runs, a)
    table = M.arms({})
    by = {t["id"]: t for t in table}
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    # ---- identity (F-419): the R74 tree at defaults reproduces the baseline
    S = jsn(A(ID_ARM), "arm_summary.json")
    for k, want in R72_60S.items():
        nm = "ID %s %s within %.3f mm of R72" % (ID_ARM, k, TOL_MM)
        got = None if S is None else S.get(k)
        g.add(nm, "MISSING" if got is None else ("PASS" if abs(got - want) <= TOL_MM else "FAIL"), "" if got is None else "got %.6f R72 %.6f diff %.6f" % (got, want, got - want))
    d0 = digest(runs, ID_ARM)
    g.add("ID %s digest equals the R73 reference (starts %s)" % (ID_ARM, R73_DIGEST_60S), "MISSING" if d0 is None else "INFO", "" if d0 is None else ("IDENTICAL " if d0.startswith(R73_DIGEST_60S) else "DIFFERENT ") + d0[:16])
    # ---- determinism control
    a, b = digest(runs, DET_ARM), digest(runs, DET_ORIG)
    miss = [n for n, x in ((DET_ARM, a), (DET_ORIG, b)) if x is None]
    g.add("DET %s digest == %s" % (DET_ARM, DET_ORIG), "MISSING" if a is None or b is None else ("PASS" if a == b else "FAIL"), ("digest file absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    # ---- engagement and structure of the F-428 / F-430 / smoke arms
    for t in table:
        arm = t["id"]
        pa = parse_arm(arm)
        if pa is None and arm not in ("r74_own_smoke_v1p0_60s", DET_ARM):
            continue
        if arm == "r74_own_smoke_v1p0_60s" or arm == DET_ARM:
            pa = dict(kind="vox", voxel=1.0, ml=2, variant=None)
        if not present(arm):
            g.add("VOX %s engaged" % arm, "MISSING"); continue
        log = rd(A(arm), "run.log") or ""
        mv, ml = re.search(r"map/voxel_size:\s+([0-9.eE+-]+)", log), re.search(r"map/max_layer:\s+(\d+)", log)
        ok = bool(mv and ml) and abs(float(mv.group(1)) - pa["voxel"]) < 1e-12 and int(ml.group(1)) == pa["ml"]
        g.add("VOX %s root voxel %s max_layer %d in the effective config" % (arm, pa["voxel"], pa["ml"]), "PASS" if ok else "FAIL", "log: voxel %s max_layer %s" % (mv and mv.group(1), ml and ml.group(1)))
        if pa["kind"] == "vox":
            C = jsn(A(arm), "r74_plane_check.json")
            if C is None:
                g.add("PLANE %s plane test values consistent (P1-P7)" % arm, "MISSING")
            else:
                g.add("PLANE %s plane test values consistent (P1-P7)" % arm, "PASS" if C.get("violations_total") == 0 and C.get("n_residuals", 0) > 0 else "FAIL",
                      "violations_total=%s n_residuals=%s" % (C.get("violations_total"), C.get("n_residuals")))
        else:
            v = pa["variant"]
            want = {"hardabs": ("hard", "absolute"), "hardrel": ("hard", "relative"), "smooth": ("smooth", None)}[v]
            l = re.search(r"plane/line_residual:\s+(\w+) / (\w+) / rel_sigmas ([0-9.eE+-]+)", log)
            ok = bool(l) and l.group(1) == want[0] and (want[1] is None or l.group(2) == want[1])
            g.add("LINE %s effective config line mode %s%s" % (arm, want[0], "/" + want[1] if want[1] else ""), "PASS" if ok else ("MISSING" if not log else "FAIL"), l.group(0) if l else "no line")
            D = csv(A(arm), "residual_dump_all.csv.gz")
            if D is None or "row_kind" not in D.columns:
                g.add("LINE %s dump has row_kind and consistent row counts" % arm, "MISSING"); continue
            n = D["row_kind"].value_counts().to_dict()
            c = lambda k: int(n.get(k, 0))
            if v == "smooth":
                good = c(3) == c(4) == c(5) and c(1) == c(2) == 0 and c(3) > 0
            else:
                good = c(2) <= c(1) and c(3) == c(4) == c(5) == 0   # R75 (F-434): the u1 row is gated separately in voxelplane.cpp, so kind 2 <= kind 1 (R74 required equality: wrong)
            g.add("LINE %s dump row kinds consistent (%s)" % (arm, "3=4=5>0" if v == "smooth" else "kind2 <= kind1, no smooth kinds"), "PASS" if good else "FAIL", "counts %s" % dict(sorted(n.items())))
            g.add("LINE %s rows of kind != 0 in frame 1 (fact)" % arm, "INFO", "%d of %d rows" % (int((D["row_kind"] != 0).sum()), len(D)))
    # ---- F-425: arms that did not ask for F-417 carry no F-417 file
    for t in table:
        arm = t["id"]
        if arm.startswith(F1_ON_PREFIXES) or t["batch"] == "refuse" or not present(arm):
            continue
        have = [f for f in os.listdir(A(arm)) if f.startswith("r73_f417_") or f.startswith("residual_dump_all") or f.startswith("shadow_residual")]
        g.add("F425 %s writes no F-417 / per-residual file" % arm, "PASS" if not have else "FAIL", ",".join(have))
    # ---- F-426: the switch engaged at the onset
    for t in table:
        if t["group"] != "SW":
            continue
        arm = t["id"]
        onset = M.SBS_ONSET if "sbs03" in arm else M.EEE_ONSET
        e = rd(A(arm), "engagement_imu.txt")
        if e is None:
            g.add("SWITCH %s engaged at scan %d" % (arm, onset), "MISSING"); continue
        l = line(e, "process_noise_switch") or ""
        mm = re.search(r"scan=(\d+) model=(\w+) applied=(\d) applied_at_frame=(-?\d+)", l)
        ok = bool(mm) and int(mm.group(1)) == onset and mm.group(3) == "1" and int(mm.group(4)) == onset
        g.add("SWITCH %s engaged at scan %d" % (arm, onset), "PASS" if ok else "FAIL", l.strip()[:150] or "no line")
    # ---- F-410: the full-bag arm ran past the R72 abort
    if present(F410_ARM):
        S = jsn(A(F410_ARM), "arm_summary.json") or {}
        raw = rd(A(F410_ARM), "raw_sha256.txt") or ""
        n = S.get("n_scans")
        g.add("F410 %s ran past scan %d without a failure dump" % (F410_ARM, F410_ABORT_SCAN), "MISSING" if n is None else ("PASS" if n > F410_ABORT_SCAN and "coupled_failure_dump.txt" not in raw else "FAIL"),
              "n_scans=%s failure_dump=%s" % (n, "coupled_failure_dump.txt" in raw))
    else:
        g.add("F410 %s ran past scan %d without a failure dump" % (F410_ARM, F410_ABORT_SCAN), "MISSING")
    # ---- F-418: every arm finished; the ZUPT detect-only arm equals the motion-Q baseline of its bag (it must not touch the state)
    notdone = [t["id"] for t in table if t["batch"] == "qa" and (not present(t["id"]) or (jsn(A(t["id"]), "arm.json") or {}).get("sentinel") != "DONE")]
    nqa = sum(1 for t in table if t["batch"] == "qa")
    g.add("QA all %d qa/qb arms finished (sentinel DONE)" % nqa, "MISSING" if len(notdone) == nqa and not any(present(t["id"]) for t in table if t["batch"] == "qa") else ("PASS" if not notdone else "FAIL"), "not done: %s" % ",".join(notdone[:6]) + (" ..." if len(notdone) > 6 else ""))
    for bag in M.BAGS:
        x, y = digest(runs, "r74_qa_%s_zupt_detect_mq" % bag), digest(runs, "r74_qb_%s_mq" % bag)
        g.add("QA %s zupt detect_only == the motion-Q baseline" % bag, "MISSING" if x is None or y is None else ("PASS" if x == y else "FAIL"), "" if x is None or y is None else "%s vs %s" % (x[:16], y[:16]))
    # ---- F-407: engagement_imu.txt for every arm that ran
    for t in table:
        arm = t["id"]
        if t["batch"] == "refuse" or not present(arm) or (jsn(A(arm), "arm.json") or {}).get("sentinel") == "FAILED":
            continue
        raw = rd(A(arm), "raw_sha256.txt") or ""
        g.add("ENGIMU %s wrote engagement_imu.txt" % arm, "PASS" if "engagement_imu.txt" in raw or os.path.exists(os.path.join(A(arm), "engagement_imu.txt")) else "FAIL")
    # ---- F-427: removed keys die at startup naming the key
    for t in table:
        if t["batch"] != "refuse":
            continue
        arm = t["id"]
        key = [k for k in M.REMOVED_KEYS if arm.endswith(k.split("process_noise.")[1].replace(".", "_"))]
        key = key[0].replace(".", "/") if key else "?"
        nm = "REFUSE %s dies at startup naming %s" % (arm, key)
        if not present(arm):
            g.add(nm, "MISSING"); continue
        meta = jsn(A(arm), "arm.json")
        tail = (rd(A(arm), "run_log_tail.txt") or "") + (rd(A(arm), "run.log") or "")
        ok = meta.get("sentinel") == "FAILED" and key in tail and "was removed in R74" in tail
        g.add(nm, "PASS" if ok else "FAIL", "sentinel=%s key_in_log=%s" % (meta.get("sentinel"), key in tail))
    return g


def report(g):
    out = ["%-8s %s%s" % (s, n, ("   [" + d + "]") if d else "") for n, s, d in g.rows]
    out += ["", "PASS %d  FAIL %d  MISSING %d  INFO %d" % tuple(sum(1 for r in g.rows if r[1] == k) for k in ("PASS", "FAIL", "MISSING", "INFO"))]
    return "\n".join(out)


def mkfake(root, bad=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)

    table = M.arms({})
    for t in table:
        arm = t["id"]
        refuse = t["batch"] == "refuse"
        w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if refuse else "DONE"}))
        if refuse:
            key = [k for k in M.REMOVED_KEYS if arm.endswith(k.split("process_noise.")[1].replace(".", "_"))][0].replace(".", "/")
            w(arm, "run_log_tail.txt", "terminate called: configuration key '%s' was removed in R74 (F-427) and has no replacement\n" % key)
            continue
        w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
        w(arm, "state_trace_blank_runid.sha256", "a" * 64 + "\n")
    # identity
    w(ID_ARM, "arm_summary.json", json.dumps(dict(R72_60S, **({"excursion_mm": R72_60S["excursion_mm"] + 0.006} if bad == "id_metric" else {}))))
    w(ID_ARM, "state_trace_blank_runid.sha256", R73_DIGEST_60S + "0" * 56 + "\n")
    # determinism
    w(DET_ARM, "state_trace_blank_runid.sha256", ("b" if bad == "det" else "a") * 64 + "\n")
    # vox / line / smoke arms
    for t in table:
        arm = t["id"]
        pa = parse_arm(arm)
        if pa is None and arm not in ("r74_own_smoke_v1p0_60s", DET_ARM):
            continue
        if pa is None:
            pa = dict(kind="vox", voxel=1.0, ml=2, variant=None)
        vox = pa["voxel"] if not (bad == "vox_value" and arm == "r74_vox_v1p0_ml3_n200_dsfirst_pfn1_fq") else 0.425
        modeline = {"hardabs": "hard / absolute", "hardrel": "hard / relative", "smooth": "smooth / absolute", None: "off / absolute"}[pa["variant"]]
        if bad == "line_mode" and arm == "r74_line_smooth_v1p0_n200_dsfirst_pfn1_fq":
            modeline = "off / absolute"
        w(arm, "run.log", "[params/voxel_map]\n  map/voxel_size:                  %s\n  map/max_layer:                   %d\n  plane/line_residual:             %s / rel_sigmas 3\n" % (vox, pa["ml"], modeline))
        if pa["kind"] == "vox":
            w(arm, "r74_plane_check.json", json.dumps({"violations_total": 3 if (bad == "plane" and arm == "r74_own_smoke_v1p0_60s") else 0, "n_residuals": 200}))
        else:
            v = pa["variant"]
            if v == "smooth":
                kinds = [3, 4, 5] * 40 + ([3] if bad == "kinds" and arm == "r74_line_smooth_v1p0_n200_dsfirst_pfn1_fq" else [])
            else:
                kinds = [0] * 50 + [1, 2] * 5 + [1] * 3 + ([2] * 4 if bad == "kinds" and arm == "r74_line_hardabs_v1p0_n200_dsfirst_pfn1_fq" else [])
            os.makedirs(os.path.join(root, arm), exist_ok=True)
            pd.DataFrame({"frame": 1, "row_kind": kinds}).to_csv(os.path.join(root, arm, "residual_dump_all.csv.gz"), index=False, compression="gzip")
    # F-425: an sw arm that wrote an F-417 file
    if bad == "f425":
        w("r74_sw_sbs03_a", "r73_f417_hth_frames.csv", "frame\n1\n")
    # switch arms
    for t in table:
        if t["group"] != "SW":
            continue
        onset = M.SBS_ONSET if "sbs03" in t["id"] else M.EEE_ONSET
        w(t["id"], "engagement_imu.txt", "[engagement/imu] propagations=100\n  process_noise_switch scan=%d model=isotropic applied=%d applied_at_frame=%d applied_at_propagation=900\n" %
          (onset, 0 if (bad == "switch" and t["id"] == "r74_sw_sbs03_a") else 1, onset))
    # F-410
    w(F410_ARM, "arm_summary.json", json.dumps({"n_scans": 500 if bad == "f410" else 3000}))
    # qa: zupt detect digests equal the mq baseline
    for bag in M.BAGS:
        w("r74_qb_%s_mq" % bag, "state_trace_blank_runid.sha256", "c" * 64 + "\n")
        w("r74_qa_%s_zupt_detect_mq" % bag, "state_trace_blank_runid.sha256", ("d" if (bad == "zupt_detect" and bag == "sbs_03") else "c") * 64 + "\n")
    if bad == "qa_failed":
        w("r74_qa_nya_01_nis_fq", "arm.json", json.dumps({"sentinel": "FAILED"}))
    if bad == "refuse_ran":
        w("r74_refuse_removed_motion_energy_source", "arm.json", json.dumps({"sentinel": "DONE"}))


CASES = {"id_metric": "ID r74_own_id_baseline_60s excursion_mm", "det": "DET r74_own_det", "vox_value": "VOX r74_vox_v1p0_ml3_n200_dsfirst_pfn1_fq root voxel", "plane": "PLANE r74_own_smoke_v1p0_60s",
         "line_mode": "LINE r74_line_smooth_v1p0_n200_dsfirst_pfn1_fq effective config", "kinds": "LINE r74_line_hardabs_v1p0_n200_dsfirst_pfn1_fq dump row kinds", "f425": "F425 r74_sw_sbs03_a",
         "switch": "SWITCH r74_sw_sbs03_a", "f410": "F410 r74_f410_meas_k8_mq_full", "zupt_detect": "QA sbs_03 zupt detect_only", "qa_failed": "QA all", "refuse_ran": "REFUSE r74_refuse_removed_motion_energy_source"}


def selftest():
    fails = []
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good)
        nb = [r for r in g.rows if r[1] not in ("PASS", "INFO")]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb[:10]])
        n_gates = len(g.rows)
        for bad, needle in CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            failed = [r[0] for r in run_gates(d).rows if r[1] == "FAIL"]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL a gate containing %r (failed: %s)" % (bad, needle, failed[:5]))
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] not in ("MISSING",) for r in run_gates(e).rows):
            fails.append("empty runs dir produced a non-MISSING gate: %s" % [r for r in run_gates(e).rows if r[1] != "MISSING"][:3])
    if fails:
        print("SELFTEST r74_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r74_gates PASS (%d gates PASS/INFO on good data; %d single breakages each flipped their gate to FAIL; empty dir all MISSING)" % (n_gates, len(CASES)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    txt = report(run_gates(a.runs))
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
