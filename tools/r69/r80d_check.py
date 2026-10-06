#!/usr/bin/env python3
"""R80D (F-484, F-489, F-491) gates for the 170 arms. Reads <root>/<arm>/lidar_msg_log_all.csv.gz (+ exit_status.txt if present) and the two grid files; writes r80d_gates.txt; exit 1 on any FAIL.

    python3 tools/r69/r80d_check.py --root <dir holding one folder per arm> --grid-retime r80c_grid.json --grid-bag grid_check/r80d_grid_bag.json --out r80d_gates.txt
    python3 tools/r69/r80d_check.py --selftest      # synthetic rows only: every FAIL path is shown to fire (rule 2.4)

Arm names: r80d_rs_<src> (restamp) | r80d_off_<mgfq|b3mq>_<bag>_o<MMM> (retime_trim, time_offset = -MMM/1000) | r80d_e3_<rt|rtt>_<src> (eee_03 arms R80C did not run).
Grid used per arm: rs -> bag fit "restamp"; off and e3 -> shipped r80c_grid.json for the eight R80C bags, bag fit "retime" for eee_03 (exactly the manifest's choice).
Offset term of an arm = time_offset - (-0.1): 0 for rs, e3 and the o100 arms, so those are bit-identical to the grid; (-MMM/1000 + 0.1) for the other off arms.
Checks: COLS (17 columns), ROWS (grid n), SHIFT (rt, rtt, off: retime_shift_s <= 1e-6 on every row: the retime grid is never later than header + ref offset, whatever the run-time offset; NOT applied
to rs: a restamp line crosses the headers), GRID (scan_time within 2e-6 s of t0 + (seq - seq0) * period + offset term on every row), RESTAMP (rs: grid t0 = first header in the log - 0.1 and the line
reaches the last header - 0.1, both within 2e-6 s: the restamp.py rule), NOTRIM (rs, rt: n_trimmed all 0), TRIM0 (rtt, off: no verdict 1 row, since a trimmed scan cannot overlap), ACCEPTOV, DROPOV, TRIMDROP
(as R80C), EXIT. INFO rows: drop share, trimmed share, shift mean / min / max.
Planning agent's script; run by me only on synthetic input.
"""
import argparse, csv, gzip, json, os, re, sys

COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict",
        "retime_shift_s", "n_trimmed"]
REF_OFFSET = -0.1
BAGRE = r"[a-z]+_\d\d"


def num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse(arm):
    """-> (family, policy, bag, time_offset) or None"""
    m = re.match(r"r80d_rs_(mg|b3|b1)_(%s)_(fq|mq)$" % BAGRE, arm)
    if m:
        return "rs", "restamp", m.group(2), REF_OFFSET
    m = re.match(r"r80d_off_(mgfq|b3mq)_(%s)_o(\d{3})$" % BAGRE, arm)
    if m:
        return "off", "retime_trim", m.group(2), round(-int(m.group(3)) / 1000.0, 3)
    m = re.match(r"r80d_e3_(rtt|rt)_(mg|b3|b1)_(%s)_(fq|mq)$" % BAGRE, arm)
    if m:
        return "e3", "retime_trim" if m.group(1) == "rtt" else "retime", m.group(3), REF_OFFSET
    return None


def grid_for(fam, bag, grid_retime, grid_bag):
    if fam == "rs":
        g = grid_bag[bag]["restamp"]; return g["t0_s"], g["period_s"], g["seq0"], grid_bag[bag]["n"]
    if bag == "eee_03":
        g = grid_bag[bag]["retime"]; return g["t0_s"], g["period_s"], grid_bag[bag]["seq0"], grid_bag[bag]["n"]
    g = grid_retime[bag]; return g["t0_s"], g["period_s"], g["seq0"], g["n"]


def check_arm(arm, rows, header, grid_retime, grid_bag, exit_text=None):
    p = parse(arm)
    if not p:
        return [("NAME", "FAIL", arm, "arm name does not match r80d_rs_<src> | r80d_off_<tag>_<bag>_o<MMM> | r80d_e3_<rt|rtt>_<src>")]
    fam, pol, bag, to = p
    t0, per, seq0, n = grid_for(fam, bag, grid_retime, grid_bag)
    term = to - REF_OFFSET
    out = []
    ok = lambda name, c, detail="": out.append((name, "PASS" if c else "FAIL", arm, detail))
    ok("COLS", header == COLS, "header %s" % (header if header != COLS else "ok"))
    ok("ROWS", len(rows) == n, "%d rows, bag has %d messages" % (len(rows), n))
    sh = [num(r["retime_shift_s"]) for r in rows]; sc = [num(r["scan_time"]) for r in rows]; sq = [int(r["seq"]) for r in rows]; hd = [num(r["header_stamp"]) for r in rows]
    if fam != "rs":
        ok("SHIFT", all(s <= 1e-6 for s in sh), "max shift %.9f s" % max(sh))
    dev = max(abs(sc[i] - (t0 + (sq[i] - seq0) * per + term)) for i in range(len(rows)))
    ok("GRID", dev < 2e-6, "max |scan_time - (grid + offset term %.3f s)| = %.3e s" % (term, dev))
    if fam == "rs":
        d0 = abs(t0 - (hd[0] + REF_OFFSET)); d1 = abs(t0 + (sq[-1] - seq0) * per - (hd[-1] + REF_OFFSET))
        ok("RESTAMP", d0 < 2e-6 and d1 < 2e-6, "|t0 - (first header - 0.1)| = %.3e s, |line at last seq - (last header - 0.1)| = %.3e s" % (d0, d1))
    nt = [int(float(r["n_trimmed"])) for r in rows]; vd = [int(float(r["verdict"])) for r in rows]; ov = [num(r["overlap_s"]) for r in rows]
    if pol in ("restamp", "retime"):
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
    out.append(("INFO", "INFO", arm, "verdict 1 rows %d of %d (%.2f%%); trimmed points %d (%.3f%% of kept + trimmed); shift mean %.3f ms, min %.3f ms, max %.3f ms"
                % (sum(1 for v in vd if v == 1), len(rows), 100.0 * sum(1 for v in vd if v == 1) / max(1, len(rows)), sum(nt), 100.0 * sum(nt) / max(1, nk + sum(nt)),
                   1e3 * sum(sh) / len(sh), 1e3 * min(sh), 1e3 * max(sh))))
    return out


def read_arm(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return rd.fieldnames, rows


def selftest():
    N = 6
    gr = {"eee_01": {"n": N, "seq0": 10, "t0_s": 1609059013.0, "period_s": 0.1}}
    gb = {"eee_01": {"n": N, "seq0": 10, "retime": {"t0_s": 1609059013.0, "period_s": 0.1}, "restamp": {"t0_s": 1609059012.9, "period_s": 0.1002, "seq0": 10}},
          "eee_03": {"n": N, "seq0": 10, "retime": {"t0_s": 1609060000.0, "period_s": 0.1}, "restamp": {"t0_s": 1609060000.0, "period_s": 0.1, "seq0": 10}}}

    def mk(fam, brk=None, bag="eee_01", term=0.0):
        t0, per, seq0, _ = grid_for(fam if fam != "off" else "e3", bag, gr, gb)
        rows = []
        for k in range(N):
            r = {c: "0" for c in COLS}
            if fam == "rs":
                st = t0 + k * per
                hd = st + 0.1 if k in (0, N - 1) else st + 0.1 + 0.012
            else:
                st = t0 + k * per + term; hd = st + 0.1 + 0.01 * k - term
            r.update(msg_index=str(k), seq=str(10 + k), header_stamp="%.6f" % hd, scan_time="%.6f" % st, retime_shift_s="-0.010000000" if fam != "rs" else "0.003", n_trimmed="0", verdict="0", overlap_s="-0.001")
            rows.append(r)
        if fam == "off": rows[2]["n_trimmed"] = "3"
        if brk == "drop_trim": rows[3]["verdict"] = "1"; rows[3]["overlap_s"] = "0.001"
        if brk == "late": rows[1]["retime_shift_s"] = "0.002000000"
        if brk == "grid": rows[4]["scan_time"] = "1609059013.400100"
        if brk == "trim": rows[2]["n_trimmed"] = "1"
        if brk == "accept_ov": rows[5]["overlap_s"] = "0.0005"
        if brk == "restamp": rows[0]["header_stamp"] = "1609059013.500000"
        return rows
    good = (check_arm("r80d_rs_mg_eee_01_fq", mk("rs"), COLS, gr, gb, "exit_code=0") + check_arm("r80d_e3_rt_mg_eee_03_fq", mk("e3", bag="eee_03"), COLS, gr, gb, "exit_code=0")
            + check_arm("r80d_off_mgfq_eee_01_o115", mk("off", term=-0.015), COLS, gr, gb, "exit_code=0") + check_arm("r80d_off_mgfq_eee_01_o100", mk("off"), COLS, gr, gb, "exit_code=0")
            + check_arm("r80d_e3_rtt_mg_eee_03_fq", mk("off", bag="eee_03"), COLS, gr, gb, "exit_code=0"))
    bad = [x for x in good if x[1] == "FAIL"]
    assert not bad, bad
    cases = [("r80d_off_mgfq_eee_01_o115", mk("off", "drop_trim", term=-0.015), "TRIM0"), ("r80d_off_mgfq_eee_01_o085", mk("off", "late", term=0.015), "SHIFT"),
             ("r80d_e3_rt_mg_eee_03_fq", mk("e3", "grid", bag="eee_03"), "GRID"), ("r80d_e3_rt_mg_eee_03_fq", mk("e3", "trim", bag="eee_03"), "NOTRIM"),
             ("r80d_rs_mg_eee_01_fq", mk("rs", "trim"), "NOTRIM"), ("r80d_rs_mg_eee_01_fq", mk("rs", "accept_ov"), "ACCEPTOV"), ("r80d_rs_mg_eee_01_fq", mk("rs", "restamp"), "RESTAMP"),
             ("r80d_off_mgfq_eee_01_o115", mk("off", term=0.0), "GRID"),                       # offset term missing: scan_time at the centre grid although time_offset is -0.115
             ("r80d_off_mgfq_eee_01_o100", mk("off", "drop_trim"), "TRIM0")]
    for arm, rows, expect in cases:
        res = check_arm(arm, rows, COLS, gr, gb, "exit_code=0")
        assert any(x[0] == expect and x[1] == "FAIL" for x in res), (arm, expect, res)
    r0 = mk("rs")
    assert not any(x[0] == "SHIFT" for x in check_arm("r80d_rs_mg_eee_01_fq", r0, COLS, gr, gb))        # no SHIFT gate for restamp
    assert any(x[0] == "EXIT" and x[1] == "FAIL" for x in check_arm("r80d_rs_mg_eee_01_fq", r0, COLS, gr, gb, "exit_code=1"))
    assert any(x[0] == "COLS" and x[1] == "FAIL" for x in check_arm("r80d_rs_mg_eee_01_fq", r0, COLS[:-1], gr, gb))
    assert any(x[0] == "ROWS" and x[1] == "FAIL" for x in check_arm("r80d_rs_mg_eee_01_fq", r0[:5], COLS, gr, gb))
    assert check_arm("r80d_xx", r0, COLS, gr, gb)[0][0] == "NAME"
    assert parse("r80d_off_b3mq_nya_02_o085") == ("off", "retime_trim", "nya_02", -0.085)
    print("selftest: good input passes (5 arm kinds); %d broken inputs each fail the named gate; 4 more FAIL paths (EXIT, COLS, ROWS, NAME); restamp has no SHIFT gate" % len(cases))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--grid-retime"); ap.add_argument("--grid-bag"); ap.add_argument("--out", default="r80d_gates.txt"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.root and a.grid_retime and a.grid_bag):
        sys.exit("--root, --grid-retime and --grid-bag are required")
    gr = json.load(open(a.grid_retime)); gb = json.load(open(a.grid_bag))
    arms = sorted(d for d in os.listdir(a.root) if d.startswith("r80d_") and os.path.isdir(os.path.join(a.root, d)))
    lines, nfail = [], 0
    for arm in arms:
        p = os.path.join(a.root, arm, "lidar_msg_log_all.csv.gz")
        if not os.path.exists(p):
            lines.append("FAIL MISSING %s: %s not found" % (arm, p)); nfail += 1; continue
        hdr, rows = read_arm(p)
        ep = os.path.join(a.root, arm, "exit_status.txt")
        et = open(ep).read() if os.path.exists(ep) else None
        for name, st, arm_, detail in check_arm(arm, rows, hdr, gr, gb, et):
            lines.append("%s %s %s: %s" % (st, name, arm_, detail)); nfail += (st == "FAIL")
    fams = {}
    for arm in arms:
        pr = parse(arm); k = pr[0] if pr else "?"; fams[k] = fams.get(k, 0) + 1
    lines.append("SUMMARY arms %d (%s), FAIL %d" % (len(arms), ", ".join("%s %d" % kv for kv in sorted(fams.items())), nfail))
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(lines[-1]); sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
