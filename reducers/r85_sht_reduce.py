#!/usr/bin/env python3
"""r85_sht_reduce: short (60 s) early-behaviour tier, host side. Python only (standard library); no node run.

  python3 r85_sht_reduce.py --ret <return dir> --manifest r85_manifest --out <return dir>
  python3 r85_sht_reduce.py --selftest

Inputs: r85_manifest/arm_table.tsv (group SHT, the 144 arms: arm, bag, edge, cell, pfn, nb), r85_manifest/onset_scans.json (`onset_scan_used` per bag) and, for every SHT arm,
<ret>/compact/<arm>/short_scan_all.csv.gz (written by tools/r69/r85_short_reduce.py: scan, t_s, residual_count, fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane, fs_iters,
fs_dx_pos_mm, fs_nis, fs_reduced_chi2, pos_err_mm). Only the 144 SHT arms enter: the smoke arms and the short determinism repeats carry the same file but are not read here.
60 s arms have no ATE (the ground-truth join needs at least 300 scans): nothing here uses an ATE or a ground-truth value; pos_err_mm is the file's drift reading
(distance of the arm's position from its own first-frame reference).
The eight cells (letter: point_filter_num, imu.ds.mode, neighborhood_size) are A 3 first 1 (the counterpart), B 1 first 1, C 3 off 1, D 1 off 1, E 3 first 2, F 1 first 2, G 3 off 2, H 1 off 2.
Cell labels relative to A: B (pfn), C (ds), E (nb) are single-factor cells; D (pfn+ds), F (pfn+nb), G (ds+nb) are two-factor cells; H (pfn+ds+nb) is the three-factor cell.

OUT r85_sht_checkpoints.csv, one row per SHT arm and checkpoint scan in 1, 2, 3, 5, 10, 20, 50, 100, 300, 480, 600:
  arm, bag, edge, cell, pfn, ds, nb, scan, residual_count, fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane, pos_err_mm   (all value cells blank when the arm's table has no such scan,
  and where the value in the file is empty or not finite)
OUT r85_sht_paired.csv, one row per (edge, cell other than A, metric, scan); metrics fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane, pos_err_mm; scan = each checkpoint scan and the
  text `stationary` (the arm's median of the metric over its scans 1 to onset-10, onset = onset_scan_used of the bag):
  edge, cell, factors_changed (names of pfn / ds / nb that differ from A, joined by +), factor_count (1 single-factor, 2 two-factor, 3 three-factor), pfn, ds, nb, metric, scan,
  n_pairs (bags where the cell arm and the A arm of the same bag and edge both have a value), median_diff (cell minus A, native units: counts, or mm; 3 decimals),
  n_lower (diff < 0), n_higher (diff > 0), n_equal (diff = 0 at 0.001), n_arms_ended_before (of the cell arms and A arms of that edge that have a file: arms whose table ends before the scan;
  for `stationary`: arms whose table ends before onset-10 of their bag). The file has a header even when there are no rows.
OUT r85_sht_missing.txt, one line per SHT arm whose short_scan_all.csv.gz is missing, unreadable or lacks a column (such an arm is left out of the tables, never silently), and one line per
  arm-table cell/pfn/nb value that disagrees with the cell letter. Exit 1 when the file is written; the tables are still written."""
import argparse
import gzip
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I drops the script dir from sys.path
import r85_common as C  # noqa: E402

CELLS = (("A", 3, "first", 1), ("B", 1, "first", 1), ("C", 3, "off", 1), ("D", 1, "off", 1), ("E", 3, "first", 2), ("F", 1, "first", 2), ("G", 3, "off", 2), ("H", 1, "off", 2))
CELL = {c[0]: c for c in CELLS}
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 50, 100, 300, 480, 600]
METRICS = ["fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "pos_err_mm"]
FILE_COLS = ["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"]
CK_HDR = ["arm", "bag", "edge", "cell", "pfn", "ds", "nb", "scan", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "pos_err_mm"]
PA_HDR = ["edge", "cell", "factors_changed", "factor_count", "pfn", "ds", "nb", "metric", "scan", "n_pairs", "median_diff", "n_lower", "n_higher", "n_equal", "n_arms_ended_before"]
STAT_BEFORE = 10   # stationary scans: 1 .. onset - 10


def factors_changed(letter):
    _, pfn, ds, nb = CELL[letter]
    _, p0, d0, n0 = CELL["A"]
    return [n for n, a, b in (("pfn", pfn, p0), ("ds", ds, d0), ("nb", nb, n0)) if a != b]


def vnum(x):
    return C.num(x)


def load_short(path):
    """-> (dict scan -> row dict of floats|None, None) or (None, reason)."""
    if not os.path.isfile(path):
        return None, "file missing"
    try:
        rows = C.read_rows(path)
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError) as e:
        return None, "unreadable (%s)" % str(e)[:60]
    if not rows:
        return None, "no data rows"
    miss = [c for c in FILE_COLS if c not in rows[0]]
    if miss:
        return None, "columns absent: " + ",".join(miss)
    d = {}
    for r in rows:
        s = vnum(r["scan"])
        if s is None:
            continue
        d[int(s)] = {k: vnum(r[k]) for k in FILE_COLS if k != "scan"}   # a repeated scan: the last row wins
    if not d:
        return None, "no row with a scan number"
    return d, None


def stat_median(tab, onset, metric):
    return C.median([v[metric] for s, v in tab.items() if s <= onset - STAT_BEFORE and v[metric] is not None])


def reduce(arms, onsets, loader):
    """arms: arm_table rows of group SHT. loader(arm) -> (table, reason). -> (checkpoint rows, paired rows, missing lines)."""
    missing, T, info = [], {}, {}
    for b in sorted({a["bag"] for a in arms} - set(onsets)):
        missing.append("onset_scans.json has no onset_scan_used for bag %s: the `stationary` rows of that bag have no values" % b)
    for a in arms:
        letter = a["cell"]
        if letter not in CELL:
            missing.append("arm table: %s has cell %r, not a letter A..H" % (a["arm"], letter))
            continue
        _, pfn, ds, nb = CELL[letter]
        if C.num(a.get("pfn")) != pfn or (a.get("nb", "") != "" and C.num(a.get("nb")) != nb):
            missing.append("arm table: %s has pfn %s nb %s, cell %s means pfn %d nb %d" % (a["arm"], a.get("pfn"), a.get("nb"), letter, pfn, nb))
        info[a["arm"]] = dict(bag=a["bag"], edge=C.edge_s(a["edge"]), cell=letter, pfn=pfn, ds=ds, nb=nb)
        tab, why = loader(a["arm"])
        if tab is None:
            missing.append("short_scan_all.csv.gz of %s: %s" % (a["arm"], why))
        else:
            T[a["arm"]] = tab
    ck = []
    for a in arms:
        i = info.get(a["arm"])
        if i is None:
            continue
        tab = T.get(a["arm"])
        for s in CHECKPOINTS:
            v = tab.get(s) if tab else None
            ck.append([a["arm"], i["bag"], i["edge"], i["cell"], i["pfn"], i["ds"], i["nb"], s] + [("%.9g" % v[m]) if (v and v[m] is not None) else ""
                                                                                                  for m in ("residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "pos_err_mm")])
    # pairing: cell arm minus the A arm of the same bag and edge
    byk = {}
    for arm, i in info.items():
        byk[(i["bag"], i["edge"], i["cell"])] = arm
    edges = sorted({i["edge"] for i in info.values()}, key=float)
    bags = sorted({i["bag"] for i in info.values()})
    pa = []
    for e in edges:
        for letter in [c[0] for c in CELLS if c[0] != "A"]:
            fc = factors_changed(letter)
            _, pfn, ds, nb = CELL[letter]
            for m in METRICS:
                for sc in CHECKPOINTS + ["stationary"]:
                    ds_, ended = [], 0
                    for b in bags:
                        ac, aa = byk.get((b, e, letter)), byk.get((b, e, "A"))
                        tabs = [T.get(x) for x in (ac, aa)]
                        for x, t in zip((ac, aa), tabs):
                            if x is None or t is None:
                                continue
                            if sc == "stationary":
                                ended += (b in onsets and max(t) < onsets[b] - STAT_BEFORE)
                            else:
                                ended += (max(t) < sc)
                        if None in tabs:
                            continue
                        if sc == "stationary":
                            v1, v0 = (stat_median(tabs[0], onsets[b], m), stat_median(tabs[1], onsets[b], m)) if b in onsets else (None, None)
                        else:
                            v1, v0 = (tabs[0].get(sc) or {}).get(m), (tabs[1].get(sc) or {}).get(m)
                        if v1 is None or v0 is None:
                            continue
                        ds_.append(round(v1 - v0, 3))
                    pa.append([e, letter, "+".join(fc), len(fc), pfn, ds, nb, m, sc, len(ds_), C.f3(C.median(ds_)), sum(d < 0 for d in ds_), sum(d > 0 for d in ds_), sum(d == 0 for d in ds_), ended])
    return ck, pa, missing


def run(ret, manifest, out):
    os.makedirs(out, exist_ok=True)
    arms = C.load_arm_table(manifest, "SHT")
    if not arms:
        print("no SHT arms in %s/arm_table.tsv" % manifest, file=sys.stderr)
        return 2
    on = json.load(open(os.path.join(manifest, "onset_scans.json")))["onset_scan_used"]
    ck, pa, missing = reduce(arms, {b: int(v) for b, v in on.items()}, lambda arm: load_short(os.path.join(ret, "compact", arm, "short_scan_all.csv.gz")))
    C.write_csv(os.path.join(out, "r85_sht_checkpoints.csv"), CK_HDR, ck)
    C.write_csv(os.path.join(out, "r85_sht_paired.csv"), PA_HDR, pa)
    print("SHT arms: %d, files read: %d, missing entries: %d" % (len(arms), len(arms) - sum(1 for m in missing if m.startswith("short_scan_all")), len(missing)))
    return C.write_missing(os.path.join(out, "r85_sht_missing.txt"), missing)


# ---------------------------------------------------------------- selftest (SYNTHETIC fixture built here; not data from any run)
def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    try:
        man, ret = os.path.join(d, "man"), os.path.join(d, "ret")
        os.makedirs(man)
        bags = ["eee_01", "nya_01"]
        json.dump({"onset_scan_used": {"eee_01": 485, "nya_01": 583}}, open(os.path.join(man, "onset_scans.json"), "w"))
        cols = ["arm", "group", "arm_kind", "bag", "edge", "cell", "pfn", "nb"]
        rows = []
        # offsets of each cell relative to A: residuals B +5 (higher), C -5 (lower), D 0 (equal); E..H +1 .. so that signs are known
        off_res = {"A": 0, "B": 5, "C": -5, "D": 0, "E": 1, "F": 2, "G": 3, "H": 4}
        short_tables = {}

        def make(arm, bag, letter, edge, n, base):
            _, pfn, ds, nb = CELL[letter]
            t = []
            for s in range(1, n + 1):
                res = base + s + off_res[letter] + (2 if bag == "nya_01" else 0) * 0
                vox = 100 + s
                t.append([s, round(0.1 * (s - 1), 4), res, res, vox, 40 + (3 if letter == "B" else 0), 2, 0.1, 1.0, 1.0, round(0.01 * s * (1 + (0.5 if letter == "C" else 0)), 6)])
            short_tables[arm] = t
            return t
        for bag in bags:
            for edge in ("0.2", "0.25"):
                for letter, pfn, ds, nb in CELLS:
                    arm = "r85_rt_sht_%s_v%s_%s" % (bag, edge.replace(".", "p"), letter)
                    rows.append([arm, "SHT", "sht", bag, edge, letter, pfn, nb])
        # decoys: smoke and determinism arms that must not enter
        rows.append(["r85_own_det_s_eee_01", "OWN", "det_sht", "eee_01", "", "A", 3, ""])
        rows.append(["r85_rt_sm_sht_H_60s", "OWN", "smoke", "eee_01", "", "H", 1, ""])
        with open(os.path.join(man, "arm_table.tsv"), "w") as f:
            f.write("\n".join("\t".join(map(str, r)) for r in [cols] + rows) + "\n")
        hdr = ",".join(FILE_COLS)
        for r in rows:
            arm, grp, kind, bag, edge, letter = r[:6]
            if arm.endswith("_v0p25_B") and bag == "nya_01":
                continue                                                    # missing file
            n = 700
            if arm.endswith("v0p2_G") and bag == "eee_01":
                n = 60                                                      # table ends at scan 60
            if arm.endswith("v0p25_C") and bag == "eee_01":
                n = 400                                                     # ends before 480 and 600
            t = make(arm, bag, letter, edge, n, 1000) if grp == "SHT" else make(arm, bag, letter, "0.25", 30, 99999)
            os.makedirs(os.path.join(ret, "compact", arm), exist_ok=True)
            with gzip.open(os.path.join(ret, "compact", arm, "short_scan_all.csv.gz"), "wt") as f:
                f.write(hdr + "\n" + "\n".join(",".join(map(str, x)) for x in t) + "\n")
        out = os.path.join(d, "out")
        rc = run(ret, man, out)
        CK = C.read_rows(os.path.join(out, "r85_sht_checkpoints.csv"))
        PA = C.read_rows(os.path.join(out, "r85_sht_paired.csv"))
        c("checkpoints header exact", open(os.path.join(out, "r85_sht_checkpoints.csv")).readline().strip() == ",".join(CK_HDR))
        c("paired header exact", open(os.path.join(out, "r85_sht_paired.csv")).readline().strip() == ",".join(PA_HDR))
        c("checkpoint rows: 32 arms x 11 checkpoints, decoys absent", len(CK) == 32 * 11 and not any(r["arm"] in ("r85_own_det_s_eee_01", "r85_rt_sm_sht_H_60s") for r in CK), str(len(CK)))
        r = [x for x in CK if x["arm"] == "r85_rt_sht_eee_01_v0p2_B" and x["scan"] == "10"][0]
        c("checkpoint values at scan 10 of cell B", (r["bag"], r["edge"], r["cell"], r["pfn"], r["ds"], r["nb"], r["residual_count"], r["fs_n_residuals"], r["fs_n_voxels"], r["fs_n_voxels_is_plane"], r["pos_err_mm"])
          == ("eee_01", "0.2", "B", "1", "first", "1", "1015", "1015", "110", "43", "0.1"), str(r))
        r = [x for x in CK if x["arm"] == "r85_rt_sht_eee_01_v0p2_G" and x["scan"] in ("100",)][0]
        c("checkpoint beyond the end of a short table is blank (G ends at 60)", r["fs_n_residuals"] == "" and r["pos_err_mm"] == "" and r["residual_count"] == "")
        r = [x for x in CK if x["arm"] == "r85_rt_sht_nya_01_v0p25_B" and x["scan"] == "1"][0]
        c("arm with no file: blank row", r["fs_n_residuals"] == "")
        mt = open(os.path.join(out, "r85_sht_missing.txt")).read().splitlines()
        c("missing file listed, exit 1", rc == 1 and len(mt) == 1 and "r85_rt_sht_nya_01_v0p25_B" in mt[0] and "file missing" in mt[0], str(mt))

        def P(e, cell, metric, scan):
            return [x for x in PA if x["edge"] == e and x["cell"] == cell and x["metric"] == metric and x["scan"] == str(scan)][0]
        p = P("0.2", "B", "fs_n_residuals", 10)
        c("sign: B minus A residuals +5 -> 2 pairs, 0 lower, 2 higher, median 5.000", (p["n_pairs"], p["n_lower"], p["n_higher"], p["n_equal"], p["median_diff"]) == ("2", "0", "2", "0", "5.000"), str(p))
        p = P("0.2", "C", "fs_n_residuals", 10)
        c("sign: C minus A residuals -5 -> 2 lower, median -5.000", (p["n_pairs"], p["n_lower"], p["n_higher"], p["median_diff"]) == ("2", "2", "0", "-5.000"), str(p))
        p = P("0.2", "D", "fs_n_residuals", 10)
        c("equal: D minus A = 0 -> 2 equal, median 0.000", (p["n_pairs"], p["n_equal"], p["median_diff"]) == ("2", "2", "0.000"), str(p))
        p = P("0.2", "B", "fs_n_voxels_is_plane", 10)
        c("plane voxels: B is +3 over A", (p["n_higher"], p["median_diff"]) == ("2", "3.000"), str(p))
        p = P("0.2", "C", "pos_err_mm", 10)
        c("pos_err_mm: C is 1.5x A's 0.1 -> diff +0.050 (cell minus A), higher", (p["n_higher"], p["median_diff"]) == ("2", "0.050"), str(p))
        c("labels: B single-factor pfn, C single-factor ds, E single-factor nb", [(P("0.2", x, "pos_err_mm", 1)["factor_count"], P("0.2", x, "pos_err_mm", 1)["factors_changed"]) for x in "BCE"] == [("1", "pfn"), ("1", "ds"), ("1", "nb")])
        c("labels: D pfn+ds, F pfn+nb, G ds+nb two-factor; H all three", [(P("0.2", x, "pos_err_mm", 1)["factor_count"], P("0.2", x, "pos_err_mm", 1)["factors_changed"]) for x in "DFGH"]
          == [("2", "pfn+ds"), ("2", "pfn+nb"), ("2", "ds+nb"), ("3", "pfn+ds+nb")])
        c("factor values of cell H: pfn 1, ds off, nb 2", (P("0.2", "H", "pos_err_mm", 1)["pfn"], P("0.2", "H", "pos_err_mm", 1)["ds"], P("0.2", "H", "pos_err_mm", 1)["nb"]) == ("1", "off", "2"))
        p = P("0.2", "G", "fs_n_residuals", 100)
        c("G (eee_01, table ends at 60) at scan 100: one pair only (nya_01), one arm ended before", (p["n_pairs"], p["n_arms_ended_before"]) == ("1", "1"), str(p))
        p = P("0.25", "C", "fs_n_residuals", 480)
        c("edge 0.25 C at scan 480: eee_01 C ends at 400 -> one pair, one arm ended before", (p["n_pairs"], p["n_arms_ended_before"]) == ("1", "1"), str(p))
        p = P("0.25", "B", "fs_n_residuals", 10)
        c("edge 0.25 B: the nya_01 B file is missing -> one pair only", p["n_pairs"] == "1", str(p))
        # stationary median: eee_01 onset 485 -> scans 1..475, residuals base+s+off: B minus A = 5 exactly; nya_01 onset 583 -> 1..573
        p = P("0.2", "B", "fs_n_residuals", "stationary")
        c("stationary median of the difference B minus A = 5.000 for both bags", (p["n_pairs"], p["median_diff"], p["n_higher"]) == ("2", "5.000", "2"), str(p))
        p = P("0.2", "G", "fs_n_residuals", "stationary")
        c("stationary: G of eee_01 ends at 60 (< 475): counted as ended before; its median uses scans 1..60", (p["n_arms_ended_before"], p["n_pairs"]) == ("1", "2"), str(p))
        c("paired rows: 2 edges x 7 cells x 4 metrics x 12 scans", len(PA) == 2 * 7 * 4 * 12, str(len(PA)))
        # empty case: no short files at all -> header-only paired content has rows with zero pairs, checkpoints blank; and an arm table without SHT arms exits 2
        out2 = os.path.join(d, "out2")
        rc2 = run(os.path.join(d, "nothing"), man, out2)
        pe = C.read_rows(os.path.join(out2, "r85_sht_paired.csv"))
        c("no file at all: exit 1, every pair count 0, median blank, missing lines = 32", rc2 == 1 and all(x["n_pairs"] == "0" and x["median_diff"] == "" for x in pe) and len(open(os.path.join(out2, "r85_sht_missing.txt")).read().splitlines()) == 32)
        # a file with a column absent is listed, not read
        bad = os.path.join(ret, "compact", "r85_rt_sht_eee_01_v0p2_H", "short_scan_all.csv.gz")
        with gzip.open(bad, "wt") as f:
            f.write("scan,t_s\n1,0.0\n")
        out3 = os.path.join(d, "out3")
        run(ret, man, out3)
        c("a file lacking columns is listed with the reason", any("r85_rt_sht_eee_01_v0p2_H" in x and "columns absent" in x for x in open(os.path.join(out3, "r85_sht_missing.txt")).read().splitlines()))
        man2 = os.path.join(d, "man2")
        os.makedirs(man2)
        with open(os.path.join(man2, "arm_table.tsv"), "w") as f:
            f.write("arm\tgroup\nx\tOWN\n")
        c("arm table without SHT arms: exit 2", run(ret, man2, os.path.join(d, "o4")) == 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return c.done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.ret and a.manifest and a.out):
        ap.error("--ret, --manifest and --out are required")
    return run(a.ret, a.manifest, a.out)


if __name__ == "__main__":
    sys.exit(main())
