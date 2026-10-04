#!/usr/bin/env python3
"""R75 gates (F-434 fixes included): determinism digest, effective config of the two new line variants, row-kind consistency, engagement, completeness.

    python3 tools/r69/r75_gates.py --runs <runs_dir> [--out r75_gates.txt]     runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py)
    python3 tools/r69/r75_gates.py --selftest                                   fake arm dirs for every arm of the R75 manifest: good data all PASS, every single breakage FAILs its gate, empty dir all MISSING

PASS / FAIL / MISSING (arm or file absent: never a pass) / INFO. Exit 1 on any FAIL. Expected arms come from make_r75_manifest.arms(). No identity gate: R75 has no identity arm (the R74 identity result stands).
"""
import argparse, json, os, re, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r75_manifest as M  # noqa: E402
from r74_gates import rd, csv, jsn, digest, G, report  # noqa: E402

WANT = {"hardrelnoise": ("hard", "relative_noise"), "hardshape": ("hard", "shape")}
MIN_ROWS_FOR_ENGAGEMENT = 100


def parse_arm(arm):
    m = re.match(r"r75_line_(hardrelnoise|hardshape)_v([0-9p]+)_", arm)
    if m:
        return dict(variant=m.group(1), voxel=float(m.group(2).replace("p", ".")))
    if arm == M.DET_ID:
        return dict(variant="hardshape", voxel=1.0)
    return None


def run_gates(runs):
    g = G()
    A = lambda a: os.path.join(runs, a)
    table = M.arms()
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    # ---- determinism control (rule 3.7)
    a, b = digest(runs, M.DET_ID), digest(runs, M.DET_ORIG)
    miss = [n for n, x in ((M.DET_ID, a), (M.DET_ORIG, b)) if x is None]
    g.add("DET %s digest == %s" % (M.DET_ID, M.DET_ORIG), "MISSING" if miss else ("PASS" if a == b else "FAIL"),
          ("digest file absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    # ---- per arm
    notdone = []
    for t in table:
        arm = t["id"]; pa = parse_arm(arm)
        if not present(arm):
            g.add("LINE %s engaged" % arm, "MISSING"); notdone.append(arm); continue
        if (jsn(A(arm), "arm.json") or {}).get("sentinel") != "DONE":
            notdone.append(arm)
        log = rd(A(arm), "run.log") or ""
        mv, ml = re.search(r"map/voxel_size:\s+([0-9.eE+-]+)", log), re.search(r"map/max_layer:\s+(\d+)", log)
        ok = bool(mv and ml) and abs(float(mv.group(1)) - pa["voxel"]) < 1e-12 and int(ml.group(1)) == 2
        g.add("VOX %s root voxel %s max_layer 2 in the effective config" % (arm, pa["voxel"]), "PASS" if ok else "FAIL", "log: voxel %s max_layer %s" % (mv and mv.group(1), ml and ml.group(1)))
        want = WANT[pa["variant"]]
        l = re.search(r"plane/line_residual:\s+(\w+) / (\w+) / rel_sigmas ([0-9.eE+-]+) / shape_ratio ([0-9.eE+-]+)", log)
        ok = bool(l) and l.group(1) == want[0] and l.group(2) == want[1]
        if ok and want[1] == "relative_noise":
            ok = abs(float(l.group(3)) - 3.0) < 1e-12
        if ok and want[1] == "shape":
            ok = abs(float(l.group(4)) - 0.1) < 1e-12
        g.add("LINE %s effective config %s/%s%s" % (arm, want[0], want[1], " rel_sigmas 3" if want[1] == "relative_noise" else " shape_ratio 0.1"),
              "PASS" if ok else ("MISSING" if not log else "FAIL"), l.group(0) if l else "no line")
        sg = re.search(r"plane/weight_floor/sigma_r2:\s+([0-9.eE+-]+)", log)
        g.add("NOISE %s sensor noise variance (the floor of the relative_noise test) is 4e-4 in the effective config" % arm,
              "PASS" if sg and abs(float(sg.group(1)) - 4e-4) < 1e-12 else ("MISSING" if not log else "FAIL"), "log: sigma_r2 %s" % (sg and sg.group(1)))
        D = csv(A(arm), "residual_dump_all.csv.gz")
        if D is None or "row_kind" not in D.columns:
            g.add("LINE %s dump has row_kind and consistent row counts" % arm, "MISSING"); continue
        n = D["row_kind"].value_counts().to_dict()
        c = lambda k: int(n.get(k, 0))
        good = c(2) <= c(1) and c(3) == c(4) == c(5) == 0   # the u1 row is gated separately (voxelplane.cpp): kind 2 <= kind 1; smooth kinds never occur in the hard modes
        g.add("LINE %s dump row kinds consistent (kind2 <= kind1, no smooth kinds)" % arm, "PASS" if good else "FAIL", "counts %s" % dict(sorted(n.items())))
        nl = int((D["row_kind"] != 0).sum())
        g.add("LINE %s engagement: the variant labelled some rows as line (frame 1)" % arm, "FAIL" if (len(D) >= MIN_ROWS_FOR_ENGAGEMENT and nl == 0) else "PASS",
              "%d of %d rows (kind != 0), share %.3f" % (nl, len(D), nl / max(len(D), 1)))
    nq = len(table)
    g.add("LN all %d arms finished (sentinel DONE)" % nq, "MISSING" if len(notdone) == nq and not any(present(t["id"]) for t in table) else ("PASS" if not notdone else "FAIL"),
          "not done: %s" % ",".join(notdone[:6]) + (" ..." if len(notdone) > 6 else ""))
    for t in table:
        arm = t["id"]
        if not present(arm) or (jsn(A(arm), "arm.json") or {}).get("sentinel") == "FAILED":
            continue
        raw = rd(A(arm), "raw_sha256.txt") or ""
        g.add("ENGIMU %s wrote engagement_imu.txt" % arm, "PASS" if "engagement_imu.txt" in raw or os.path.exists(os.path.join(A(arm), "engagement_imu.txt")) else "FAIL")
    return g


def mkfake(root, bad=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)
    first = "r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq"; firstn = "r75_line_hardrelnoise_v0p425_n200_dsfirst_pfn1_fq"
    for t in M.arms():
        arm = t["id"]; pa = parse_arm(arm)
        w(arm, "arm.json", json.dumps({"sentinel": "FAILED" if (bad == "arm_failed" and arm == first) else "DONE"}))
        if not (bad == "no_engimu" and arm == first):
            w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
        w(arm, "state_trace_blank_runid.sha256", ("b" if (bad == "det" and arm == M.DET_ID) else "a") * 64 + "\n")
        vox = 0.425 if (bad == "vox_value" and arm == first) else pa["voxel"]
        thr = "relative_noise" if pa["variant"] == "hardrelnoise" else "shape"
        rs, sr = 3.0, 0.1
        if arm == first and bad == "threshold":
            thr = "absolute"
        if arm == first and bad == "shape_ratio":
            sr = 0.5
        if arm == firstn and bad == "rel_sigmas":
            rs = 1.0
        w(arm, "run.log", "[params/voxel_map]\n  map/voxel_size:                  %s\n  map/max_layer:                   2\n  plane/weight_floor/sigma_r2:     %s\n  plane/line_residual:             hard / %s / rel_sigmas %s / shape_ratio %s\n" % (vox, 0.0025 if (bad == "sigma" and arm == first) else 0.0004, thr, rs, sr))
        kinds = [0] * 50 + [1, 2] * 5 + [1] * 3        # kind 1 > kind 2: the good case of the R75 gate
        if bad == "kinds" and arm == first:
            kinds += [2] * 4
        if bad == "smooth_kinds" and arm == first:
            kinds += [3, 4, 5]
        if bad == "no_lines" and arm == first:
            kinds = [0] * 150
        pd.DataFrame({"frame": 1, "row_kind": kinds}).to_csv(os.path.join(root, arm, "residual_dump_all.csv.gz"), index=False, compression="gzip")


CASES = {"det": "DET r75_own_det", "vox_value": "VOX r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq root voxel", "threshold": "LINE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq effective config",
         "shape_ratio": "LINE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq effective config", "rel_sigmas": "LINE r75_line_hardrelnoise_v0p425_n200_dsfirst_pfn1_fq effective config",
         "kinds": "LINE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq dump row kinds", "smooth_kinds": "LINE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq dump row kinds",
         "sigma": "NOISE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq", "no_lines": "LINE r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq engagement", "arm_failed": "LN all", "no_engimu": "ENGIMU r75_line_hardshape_v1p0_n200_dsfirst_pfn1_fq"}


def selftest():
    fails = []
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good)
        nb = [r for r in g.rows if r[1] not in ("PASS", "INFO")]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb[:10]])
        if len(M.arms()) != 161:
            fails.append("manifest arm count %d != 161" % len(M.arms()))
        for bad, needle in CASES.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            failed = [r[0] for r in run_gates(d).rows if r[1] == "FAIL"]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL a gate containing %r (failed: %s)" % (bad, needle, failed[:5]))
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] != "MISSING" for r in run_gates(e).rows):
            fails.append("empty runs dir produced a non-MISSING gate: %s" % [r for r in run_gates(e).rows if r[1] != "MISSING"][:3])
    if fails:
        print("SELFTEST r75_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r75_gates PASS (%d gates PASS on good data; %d single breakages each flipped their gate to FAIL; empty dir all MISSING)" % (len(g.rows), len(CASES)))
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
