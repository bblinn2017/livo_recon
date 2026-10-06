#!/usr/bin/env python3
"""R80C (F-484) gates for the 72 retime arms. Reads <root>/<arm>/lidar_msg_log_all.csv.gz (+ exit_status.txt if present) and r80c_grid.json; writes r80c_gates.txt; exit 1 on any FAIL.

    python3 tools/r69/r80c_check.py --root <dir holding one folder per arm> --grid r80c_grid.json --out r80c_gates.txt
    python3 tools/r69/r80c_check.py --selftest      # synthetic rows only: every FAIL path is shown to fire (rule 2.4)

Arm names: r80c_rt_<src> (retime), r80c_rtt_<src> (retime_trim); <src> = mg_<bag>_fq | b3_<bag>_mq | b1_<bag>_fq | b1_<bag>_mq.
Checks per arm: COLS (the 17 columns), ROWS (one row per bag message: grid json n), SHIFT (retime_shift_s <= 1e-6 on every row: the grid is never later than header + offset), GRID (scan_time within 2e-6 s of
t0 + (seq - seq0) * period on every row), NOTRIM (rt: n_trimmed all 0), TRIM0 (rtt: no verdict 1 row), ACCEPTOV (every accepted row has overlap_s <= 0), DROPOV (every verdict 1 row has overlap_s > 0),
TRIMDROP (rows with n_trimmed > 0 are verdict 0), EXIT (exit_status.txt has exit_code=0). INFO rows: drop share, trimmed share of points, shift mean / min.
Planning agent's script; run by me only on synthetic input.
"""
import argparse, csv, gzip, io, json, math, os, re, sys

COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict",
        "retime_shift_s", "n_trimmed"]


def num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def check_arm(arm, rows, header, grid, exit_text=None):
    out = []
    m = re.match(r"r80c_(rtt|rt)_(mg|b3|b1)_([a-z]+_\d\d)_(fq|mq)$", arm)
    if not m:
        return [("NAME", "FAIL", arm, "arm name does not match r80c_<rt|rtt>_<src>")]
    pol, bag = m.group(1), m.group(3)
    g = grid[bag]
    ok = lambda name, c, detail="": out.append((name, "PASS" if c else "FAIL", arm, detail))
    ok("COLS", header == COLS, "header %s" % (header if header != COLS else "ok"))
    ok("ROWS", len(rows) == g["n"], "%d rows, bag has %d messages" % (len(rows), g["n"]))
    sh = [num(r["retime_shift_s"]) for r in rows]; sc = [num(r["scan_time"]) for r in rows]; sq = [int(r["seq"]) for r in rows]
    ok("SHIFT", all(s <= 1e-6 for s in sh), "max shift %.9f s" % max(sh))
    dev = max(abs(sc[i] - (g["t0_s"] + (sq[i] - g["seq0"]) * g["period_s"])) for i in range(len(rows)))
    ok("GRID", dev < 2e-6, "max |scan_time - grid| = %.3e s" % dev)
    nt = [int(float(r["n_trimmed"])) for r in rows]; vd = [int(float(r["verdict"])) for r in rows]; ov = [num(r["overlap_s"]) for r in rows]
    if pol == "rt":
        ok("NOTRIM", sum(nt) == 0, "n_trimmed total %d" % sum(nt))
    else:
        ok("TRIM0", sum(1 for v in vd if v == 1) == 0, "verdict 1 rows %d" % sum(1 for v in vd if v == 1))
    ok("ACCEPTOV", all(not (v == 0 and o > 0) for v, o in zip(vd, ov)), "accepted rows with overlap_s > 0: %d" % sum(1 for v, o in zip(vd, ov) if v == 0 and o > 0))
    ok("DROPOV", all(o > 0 for v, o in zip(vd, ov) if v == 1), "dropped rows with overlap_s <= 0 or nan: %d" % sum(1 for v, o in zip(vd, ov) if v == 1 and not o > 0))
    ok("TRIMDROP", all(v == 0 for v, t in zip(vd, nt) if t > 0), "trimmed rows that were dropped: %d" % sum(1 for v, t in zip(vd, nt) if t > 0 and v != 0))
    if exit_text is not None:
        mm = re.search(r"exit_code=(\d+)", exit_text)
        ok("EXIT", bool(mm) and int(mm.group(1)) == 0, "exit_status: %s" % (mm.group(0) if mm else "no exit_code= found"))
    nk = sum(int(float(r["n_kept"])) for r in rows)
    out.append(("INFO", "INFO", arm, "verdict 1 rows %d of %d (%.2f%%); trimmed points %d (%.3f%% of kept + trimmed); mean shift %.3f ms, min shift %.3f ms"
                % (sum(1 for v in vd if v == 1), len(rows), 100.0 * sum(1 for v in vd if v == 1) / max(1, len(rows)), sum(nt), 100.0 * sum(nt) / max(1, nk + sum(nt)),
                   1e3 * sum(sh) / len(sh), 1e3 * min(sh))))
    return out


def read_arm(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return rd.fieldnames, rows


def selftest():
    grid = {"eee_01": {"n": 6, "seq0": 10, "t0_s": 1609059013.0, "period_s": 0.1}}
    def mk(pol, break_what=None):
        rows = []
        for k in range(6):
            r = {c: "0" for c in COLS}
            r.update(msg_index=str(k), seq=str(10 + k), scan_time="%.6f" % (1609059013.0 + 0.1 * k), retime_shift_s="-0.010000000", n_trimmed="0", verdict="0", overlap_s="-0.001")
            rows.append(r)
        if pol == "rtt":
            rows[2]["n_trimmed"] = "3"
        if break_what == "drop_rtt": rows[3]["verdict"] = "1"; rows[3]["overlap_s"] = "0.001"
        if break_what == "late": rows[1]["retime_shift_s"] = "0.002000000"
        if break_what == "grid": rows[4]["scan_time"] = "1609059013.400100"
        if break_what == "trim_rt": rows[2]["n_trimmed"] = "1"
        if break_what == "accept_ov": rows[5]["overlap_s"] = "0.0005"
        return rows
    good = check_arm("r80c_rtt_mg_eee_01_fq", mk("rtt"), COLS, grid, "exit_code=0") + check_arm("r80c_rt_mg_eee_01_fq", mk("rt"), COLS, grid, "exit_code=0")
    assert not [x for x in good if x[1] == "FAIL"], [x for x in good if x[1] == "FAIL"]
    cases = [("r80c_rtt_mg_eee_01_fq", "rtt", "drop_rtt", "TRIM0"), ("r80c_rt_mg_eee_01_fq", "rt", "late", "SHIFT"), ("r80c_rt_mg_eee_01_fq", "rt", "grid", "GRID"),
             ("r80c_rt_mg_eee_01_fq", "rt", "trim_rt", "NOTRIM"), ("r80c_rt_mg_eee_01_fq", "rt", "accept_ov", "ACCEPTOV")]
    for arm, pol, brk, expect in cases:
        res = check_arm(arm, mk(pol, brk), COLS, grid, "exit_code=0")
        assert any(x[0] == expect and x[1] == "FAIL" for x in res), (brk, res)
    assert any(x[0] == "EXIT" and x[1] == "FAIL" for x in check_arm("r80c_rt_mg_eee_01_fq", mk("rt"), COLS, grid, "exit_code=1"))
    assert any(x[0] == "COLS" and x[1] == "FAIL" for x in check_arm("r80c_rt_mg_eee_01_fq", mk("rt"), COLS[:-1], grid, "exit_code=0"))
    assert any(x[0] == "ROWS" and x[1] == "FAIL" for x in check_arm("r80c_rt_mg_eee_01_fq", mk("rt")[:5], COLS, grid, "exit_code=0"))
    print("selftest: good input passes; %d broken inputs each fail the named gate" % (len(cases) + 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--grid"); ap.add_argument("--out", default="r80c_gates.txt"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.root and a.grid):
        sys.exit("--root and --grid are required")
    grid = json.load(open(a.grid))
    arms = sorted(d for d in os.listdir(a.root) if d.startswith("r80c_") and os.path.isdir(os.path.join(a.root, d)))
    lines, nfail = [], 0
    for arm in arms:
        p = os.path.join(a.root, arm, "lidar_msg_log_all.csv.gz")
        if not os.path.exists(p):
            lines.append("FAIL MISSING %s: %s not found" % (arm, p)); nfail += 1; continue
        hdr, rows = read_arm(p)
        ep = os.path.join(a.root, arm, "exit_status.txt")
        et = open(ep).read() if os.path.exists(ep) else None
        for name, st, arm_, detail in check_arm(arm, rows, hdr, grid, et):
            lines.append("%s %s %s: %s" % (st, name, arm_, detail)); nfail += (st == "FAIL")
    lines.append("SUMMARY arms %d, FAIL %d" % (len(arms), nfail))
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(lines[-1]); sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
