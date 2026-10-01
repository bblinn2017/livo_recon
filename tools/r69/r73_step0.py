#!/usr/bin/env python3
"""R73 STEP 0: retrieve and reduce data that R72 produced but did not return (F-396 raw files, F-398 raw matrices, F-403 step_cap.csv). NO simulation is run; Python only.

    python3 tools/r69/r73_step0.py --root <dir that contains the R72 listener archives> --out R73_RETURN/STEP0 [--arms-from r72_manifest/arm_table.tsv]

Discovery: every directory under --root that holds archive_index.json is an arm archive (archive/<name>.gz); the arm is the path component that equals an R72 arm name.
Per group
  EF (40 arms r72_ef_*):  r73_frame_errors.py (state_trace + state_reference are looked up in the archive and, for state_reference.txt, in the arm's run dir under --root) -> rows of
                          r73_f416_frame_errors.csv; r73_residual_report.py on residual_dump + lio_update_diag -> r73_f417_hth_frames.csv, r73_f417_classes.csv,
                          r73_f417_residuals.csv.gz (frames 1-5), r73_f417_consistency.csv (all with an `arm` column)
  RAW4 (r72_ef_n1600_dsoff_pfn{1,3}_{fq,mq}): the archived residual_dump.csv.gz, lio_update_diag.csv.gz, pose_control_first_frame_solve_matrices.txt.gz, residual_dump_state.csv.gz copied
                          byte for byte into STEP0/<arm>/ (they are already gzip; sha256 of each is listed in STEP0/raw_sha256.txt)
  CO (the 8 r72_co_* full-bag arms): coupled_record_f1..5.txt.gz copied; coupled_scan_status / joint_knot_all_scans / state_trace rows with scan <= 600 written as *_first600.csv.gz
  SC (r72_sc_fq_60s, r72_sc_fq_full, r72_sc_mq_full): step_cap.csv.gz copied
Every arm or file it cannot find is listed in STEP0/step0_missing.txt AND the exit code is 1: the return pack fails naming the follow-up (required_outputs.json).
"""
import argparse, glob, gzip, hashlib, json, os, shutil, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EF = ["r72_ef_n%d_ds%s_pfn%d_%s" % (n, ds, p, q) for n in (200, 400, 800, 1600, 3850) for ds in ("first", "off") for p in (1, 3) for q in ("fq", "mq")]
RAW4 = ["r72_ef_n1600_dsoff_pfn%d_%s" % (p, q) for p in (1, 3) for q in ("fq", "mq")]
CO = ["r72_co_%s_gateon_knots%d_%s" % (t, k, q) for t in ("tail", "meas") for k in (4, 8) for q in ("fq", "mq")]
SC = ["r72_sc_fq_60s", "r72_sc_fq_full", "r72_sc_mq_full"]


def discover(roots):
    arch = {}
    for root in roots:
        for dp, dn, fn in os.walk(root):
            if "archive_index.json" in fn:
                for part in dp.replace("\\", "/").split("/"):
                    arch.setdefault(part, dp)
    return arch


def ungz(src, dst):
    with gzip.open(src, "rb") as f, open(dst, "wb") as o:
        shutil.copyfileobj(f, o)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def find_ref(roots, arm):
    for root in roots:
        for dp, dn, fn in os.walk(root):
            if arm in dp.replace("\\", "/").split("/") and "state_reference.txt" in fn:
                return os.path.join(dp, "state_reference.txt")
    return None


def find_raw_dir(roots, arm):
    """R73 agent fix: a sentinel-FAILED job under on_failed_job:skip (every R72 EF arm) is never
    archived at all -- discover() finds nothing for it even though its raw output sits, untouched
    and uncompressed, directly in the job's own OUT_DIR. Fall back to locating that directory by
    name + the presence of a raw (non-.gz) file this tool actually needs, so real, present data
    is not reported missing just because it was never gzipped into an archive/."""
    for root in roots:
        for dp, dn, fn in os.walk(root):
            if os.path.basename(dp.rstrip("/\\")) == arm and ("residual_dump.csv" in fn or "state_trace.csv" in fn):
                return dp
    return None


def get_file(arc, raw, name, dst):
    """Copy `name` into `dst` (uncompressed), from an archived `<name>.gz` under `arc` if present,
    else from a plain, never-archived `<name>` under `raw`. Returns True if found either way."""
    if arc:
        g = os.path.join(arc, name + ".gz")
        if os.path.exists(g):
            ungz(g, dst)
            return True
    if raw:
        p = os.path.join(raw, name)
        if os.path.exists(p):
            shutil.copy(p, dst)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+", help="one or more directories (the R72 results root and the listener_out archive root)"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arch = discover(a.root)
    missing, tabs = [], {k: [] for k in ("f416", "hth", "cls", "res", "con")}
    raw_sha = []
    for arm in EF:
        ad = arch.get(arm)
        arc = os.path.join(ad, "archive") if ad and os.path.isdir(os.path.join(ad, "archive")) else ad
        raw = find_raw_dir(a.root, arm) if not arc else None
        if not arc and not raw:
            missing.append("%s: no archive and no raw output directory found under %s (F-416/F-417)" % (arm, a.root)); continue
        with tempfile.TemporaryDirectory() as t:
            ok = True
            for nm in ("state_trace.csv", "lio_gt_matched.csv", "residual_dump.csv", "lio_update_diag.csv"):
                if not get_file(arc, raw, nm, os.path.join(t, nm)) and nm in ("residual_dump.csv", "lio_update_diag.csv", "state_trace.csv"):
                    missing.append("%s: %s absent from both the archive and the raw output dir" % (arm, nm)); ok = False
            ref = find_ref(a.root, arm)
            if ref:
                shutil.copy(ref, os.path.join(t, "state_reference.txt"))
            else:
                missing.append("%s: state_reference.txt not found under --root (F-416 needs it)" % arm)
            if not ok:
                continue
            o = os.path.join(t, "o"); os.makedirs(o)
            if ref:
                rc = subprocess.call([sys.executable, os.path.join(HERE, "r73_frame_errors.py"), "--dir", t, "--out", o, "--arm", arm, "--first", "10"])
                if rc or not os.path.exists(os.path.join(o, "r73_f416_frame_errors.csv")):
                    missing.append("%s: r73_frame_errors.py failed rc=%s" % (arm, rc))
                else:
                    tabs["f416"].append(pd.read_csv(os.path.join(o, "r73_f416_frame_errors.csv")))
            rc = subprocess.call([sys.executable, os.path.join(HERE, "r73_residual_report.py"), "--dir", t, "--out", o, "--arm", arm, "--resid-frames", "1,2,3,4,5"])
            if rc:
                missing.append("%s: r73_residual_report.py failed rc=%s" % (arm, rc))
            for k, nm in (("hth", "r73_f417_hth_frames.csv"), ("cls", "r73_f417_classes.csv"), ("res", "r73_f417_residuals.csv.gz"), ("con", "r73_f417_consistency.csv")):
                p = os.path.join(o, nm)
                if os.path.exists(p):
                    tabs[k].append(pd.read_csv(p))
    names = {"f416": "r73_f416_frame_errors.csv", "hth": "r73_f417_hth_frames.csv", "cls": "r73_f417_classes.csv", "res": "r73_f417_residuals.csv.gz", "con": "r73_f417_consistency.csv"}
    for k, nm in names.items():
        df = pd.concat(tabs[k], ignore_index=True) if tabs[k] else pd.DataFrame()
        df.to_csv(os.path.join(a.out, nm), index=False, float_format="%.9g", compression="gzip" if nm.endswith(".gz") else None)
    # raw copies
    def copy_raw(arm, names_):
        ad = arch.get(arm)
        arc = os.path.join(ad, "archive") if ad and os.path.isdir(os.path.join(ad, "archive")) else ad
        raw = find_raw_dir(a.root, arm) if not arc else None
        if not arc and not raw:
            missing.append("%s: no archive and no raw output directory found (raw copy)" % arm); return
        d = os.path.join(a.out, arm); os.makedirs(d, exist_ok=True)
        for nm in names_:  # nm is the archived name, e.g. "residual_dump.csv.gz"
            s = os.path.join(arc, nm) if arc else None
            if s and os.path.exists(s):
                shutil.copy(s, d); raw_sha.append("%s  %s/%s" % (sha(os.path.join(d, nm)), arm, nm)); continue
            plain = nm[:-3] if nm.endswith(".gz") else nm  # R73 agent fix: fall back to the plain,
            rp = os.path.join(raw, plain) if raw else None  # never-archived raw file and gzip it
            if rp and os.path.exists(rp):                   # ourselves to the expected .gz name --
                dst = os.path.join(d, nm)                    # content is lossless, only the gzip
                with open(rp, "rb") as fi, gzip.open(dst, "wb") as fo:  # container is agent-produced,
                    shutil.copyfileobj(fi, fo)                           # not an R72 byte-for-byte copy.
                raw_sha.append("%s  %s/%s (gzipped by this tool from a never-archived raw file)" % (sha(dst), arm, nm))
                continue
            missing.append("%s: %s absent from both the archive and the raw output dir" % (arm, nm))
    for arm in RAW4:
        copy_raw(arm, ["residual_dump.csv.gz", "lio_update_diag.csv.gz", "pose_control_first_frame_solve_matrices.txt.gz", "residual_dump_state.csv.gz"])
    for arm in CO:
        copy_raw(arm, ["coupled_record_f%d.txt.gz" % i for i in range(1, 6)])
        ad = arch.get(arm)
        if ad:
            arc = os.path.join(ad, "archive") if os.path.isdir(os.path.join(ad, "archive")) else ad
            for nm, col in (("coupled_scan_status.csv", None), ("joint_knot_all_scans.csv", None), ("state_trace.csv", "scan_id")):
                g = os.path.join(arc, nm + ".gz")
                if not os.path.exists(g):
                    missing.append("%s: %s.gz absent from the archive" % (arm, nm)); continue
                df = pd.read_csv(g, dtype=str)
                c = col or next((x for x in ("scan_id", "scan", "frame", "scan_index") if x in df.columns), None)
                if c is None:
                    missing.append("%s: %s has no scan column" % (arm, nm)); continue
                df = df[pd.to_numeric(df[c], errors="coerce") <= 600]
                df.to_csv(os.path.join(a.out, arm, nm.replace(".csv", "_first600.csv.gz")), index=False, compression="gzip")
    for arm in SC:
        copy_raw(arm, ["step_cap.csv.gz"])
    open(os.path.join(a.out, "raw_sha256.txt"), "w").write("\n".join(raw_sha) + "\n")
    open(os.path.join(a.out, "step0_missing.txt"), "w").write("\n".join(missing) + ("\n" if missing else "NOTHING MISSING\n"))
    print("step 0: %d problems (STEP0/step0_missing.txt)" % len(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
