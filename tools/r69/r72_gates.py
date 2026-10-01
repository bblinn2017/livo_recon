#!/usr/bin/env python3
"""R72 gates: identity (digest) gates, engagement gates for the new flags, refusal gates, the coupled-record presence gate. Replaces bootstrap_reduce.py --collect
and stage1b_checks.py for R72 (those two are written around R71 arm names; do not run them on R72 data).

    python3 tools/r69/r72_gates.py --runs <runs_dir> [--out r72_gates.txt]        runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py)
    python3 tools/r69/r72_gates.py --selftest                                       builds fake arm dirs, shows every gate PASS on good data and FAIL on a bad copy

A gate returns PASS / FAIL / MISSING (MISSING = the arm or file is not there; never counted as a pass). Exit code 1 if any gate FAILs; MISSING is listed and does not fail
the exit code, because a batch may not have been dispatched yet (the report says which).
"""
import argparse, json, os, re, shutil, sys, tempfile

R71 = {"r72_00_id_baseline_60s": "8cb066436baaedb45f2af20921ff56337a1b553e1aa2d8fafe9d4170a9043f19",
       "r72_00_id_baseline_full": "8bc89b6bb69aaf32d2283a969ad6cffad798836cf40ccf0def2fe9712d14e5c2",
       "r72_00_id_motionq_full": "98f3855cb2146bb06e9dc15b101a8699804a395b4807144323f99e2e509dbad1"}
IDENT = {"r72_id_stepcap_wide_60s": "r72_00_id_baseline_60s", "r72_id_zupt_detect_60s": "r72_00_id_baseline_60s",
         "r72_id_nisgate_dead_60s": "r72_00_id_baseline_60s", "r72_id_diag_on_60s": "r72_00_id_baseline_60s"}
REFUSE = {"r72_refuse_stepcap_suboption": "step_cap", "r72_refuse_stepcap_coupled": "step_cap", "r72_refuse_stepcap_zero": "step_cap",
          "r72_refuse_qadapt_suboption": "q_adapt", "r72_refuse_qadapt_inert_forgetting": "q_adapt", "r72_refuse_qadapt_inert_bounds": "q_adapt",
          "r72_refuse_qadapt_coupled": "q_adapt", "r72_refuse_zupt_suboption": "zupt", "r72_refuse_zupt_threshold": "zupt",
          "r72_refuse_local_poly_fixed": "energy_source", "r72_refuse_local_poly_order_instant": "local_poly",
          "r72_refuse_wnoa_zero": "continuous_prior", "r72_refuse_wnoa_suboption": "continuous_prior"}


def rd(d, n):
    p = os.path.join(d, n)
    return open(p, errors="replace").read() if os.path.exists(p) else None


def num(txt, key):
    m = re.search(r"\b" + re.escape(key) + r"=(-?[0-9.eE+-]+)", txt or "")
    return float(m.group(1)) if m else None


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


def run_gates(runs):
    g = G()
    A = lambda a: os.path.join(runs, a)
    # identity: R71 digests of the same jobs
    for arm, want in R71.items():
        got = digest(runs, arm)
        g.add("ID %s == R71 digest" % arm, "MISSING" if got is None else ("PASS" if got == want else "FAIL"), "" if got is None else "got %s want %s" % (got[:16], want[:16]))
    for arm, ref in IDENT.items():
        a, b = digest(runs, arm), digest(runs, ref)
        g.add("ID %s == %s" % (arm, ref), "MISSING" if a is None or b is None else ("PASS" if a == b else "FAIL"), "" if a is None or b is None else "%s vs %s" % (a[:16], b[:16]))
    # engagement of the identity arms (the code RAN, it just did not change anything)
    e = rd(A("r72_id_stepcap_wide_60s"), "engagement.txt")
    if e is None:
        g.add("ENG stepcap_wide ran and did not bind", "MISSING")
    else:
        it, bd = num(line(e, "lio/ekf/step_cap"), "iterations"), num(line(e, "lio/ekf/step_cap"), "bound")
        g.add("ENG stepcap_wide ran and did not bind", "PASS" if it and it > 0 and bd == 0 else "FAIL", "iterations=%s bound=%s" % (it, bd))
    e = rd(A("r72_id_zupt_detect_60s"), "engagement.txt")
    if e is None:
        g.add("ENG zupt detect_only ran, found rest, applied nothing", "MISSING")
    else:
        l = line(e, "lio/zupt")
        sc, st, ap = num(l, "scans"), num(l, "stationary"), num(l, "applied")
        g.add("ENG zupt detect_only ran, found rest, applied nothing", "PASS" if sc and sc > 0 and st and st > 0 and ap == 0 else "FAIL", "scans=%s stationary=%s applied=%s" % (sc, st, ap))
    e = rd(A("r72_id_nisgate_dead_60s"), "engagement.txt")
    ei = rd(A("r72_id_nisgate_dead_60s"), "engagement_imu.txt")
    if e is None:
        g.add("ENG nis_gate dead: evaluated every scan, gain never left 1", "MISSING")
    else:
        l = line(e, "lio/q_adapt")
        sc, ne = num(l, "scans"), num(l, "scans_with_gain_ne_1")
        imu_ne = num(line(ei, "q_gain"), "steps_with_gain_ne_1") if ei else None
        g.add("ENG nis_gate dead: evaluated every scan, gain never left 1", "PASS" if sc and sc > 0 and ne == 0 and imu_ne in (0, None) else "FAIL", "scans=%s gain_ne_1=%s imu_gain_ne_1=%s" % (sc, ne, imu_ne))
    # engagement of the smoke arms for each new flag
    def eng(arm, fn, cond, name, imu=False):
        t = rd(A(arm), "engagement_imu.txt" if imu else "engagement.txt")
        if t is None:
            g.add(name, "MISSING"); return
        ok, det = cond(t)
        g.add(name, "PASS" if ok else "FAIL", det)
    eng("r72_smoke_stepcap_fq_60s", None, lambda t: ("INERT" not in (line(t, "lio/ekf/step_cap") or "INERT"), (line(t, "lio/ekf/step_cap") or "no line")[-120:]), "SMOKE step cap not INERT")
    eng("r72_smoke_zupt_apply_60s", None, lambda t: ((num(line(t, "lio/zupt"), "applied") or 0) > 0, (line(t, "lio/zupt") or "no line")[-140:]), "SMOKE zupt apply applied > 0")
    eng("r72_smoke_innov_scale_60s", None, lambda t: ((num(line(t, "lio/q_adapt"), "scans_with_gain_ne_1") or 0) > 0, (line(t, "lio/q_adapt") or "no line")[:160]), "SMOKE innovation_scale moved the gain")
    eng("r72_smoke_nis_gate_60s", None, lambda t: ("INERT" not in (line(t, "lio/q_adapt") or "INERT"), (line(t, "lio/q_adapt") or "no line")[:160]), "SMOKE nis_gate not INERT")
    eng("r72_smoke_local_poly_60s", None, lambda t: ("INERT" not in (line(t, "energy_source=local_poly") or "INERT"), (line(t, "energy_source=local_poly") or "no line")[:160]), "SMOKE local_poly not INERT", imu=True)
    # refusals
    for arm, key in REFUSE.items():
        d = A(arm)
        if not os.path.isdir(d):
            g.add("REFUSE %s dies at startup naming %s" % (arm, key), "MISSING"); continue
        try:
            meta = json.load(open(os.path.join(d, "arm.json")))
        except Exception:
            g.add("REFUSE %s dies at startup naming %s" % (arm, key), "MISSING", "no arm.json"); continue
        tail = (rd(d, "run_log_tail.txt") or "") + (rd(d, "run.log") or "")
        ok = meta.get("sentinel") == "FAILED" and key in tail
        g.add("REFUSE %s dies at startup naming %s" % (arm, key), "PASS" if ok else "FAIL", "sentinel=%s key_in_log=%s" % (meta.get("sentinel"), key in tail))
    # coupled record smokes
    for t in ("tail", "measurement"):
        arm = "r72_smoke_co_%s_rec_60s" % t
        d = A(arm)
        names = (rd(d, "raw_sha256.txt") or "") if os.path.isdir(d) else None
        if names is None:
            g.add("RECORD %s wrote coupled_record_f1..f5 and no failure dump" % arm, "MISSING"); continue
        have = all(("coupled_record_f%d.txt" % i) in names for i in range(1, 6))
        g.add("RECORD %s wrote coupled_record_f1..f5 and no failure dump" % arm, "PASS" if have and "coupled_failure_dump.txt" not in names else "FAIL", "records=%s dump=%s" % (have, "coupled_failure_dump.txt" in names))
        s = None
        p = os.path.join(d, "r72_dump_summary.json")
        if os.path.exists(p):
            S = json.load(open(p))
            n_it = sum(len(r["iterations"]) for r in S["records"])
            allpd = all(it.get("A_pd_pass") for r in S["records"] for it in r["iterations"])
            g.add("RECORD %s dump reducer parsed all records, A_pd_pass in every iteration" % arm, "PASS" if S["records"] and n_it > 0 and allpd and not S["errors"] else "FAIL", "records=%d iterations=%d errors=%d" % (len(S["records"]), n_it, len(S["errors"])))
        else:
            g.add("RECORD %s dump reducer parsed all records, A_pd_pass in every iteration" % arm, "MISSING")
    return g


def report(g):
    out = ["%-6s %s%s" % (s, n, ("   [" + d + "]") if d else "") for n, s, d in g.rows]
    out.append("")
    out.append("PASS %d  FAIL %d  MISSING %d" % tuple(sum(1 for r in g.rows if r[1] == k) for k in ("PASS", "FAIL", "MISSING")))
    return "\n".join(out)


def selftest():
    """Fake arm dirs. Part 1: everything good -> every gate PASS. Part 2: break one thing per gate in a copy -> that gate FAILs (rule 2.4)."""
    def mk(root, bad=None):
        def w(arm, name, text):
            os.makedirs(os.path.join(root, arm), exist_ok=True)
            open(os.path.join(root, arm, name), "w").write(text)
        for arm, h in R71.items():
            w(arm, "state_trace_blank_runid.sha256", h + "\n")
        for arm in IDENT:
            w(arm, "state_trace_blank_runid.sha256", (R71["r72_00_id_baseline_60s"] if bad != "ident" or arm != "r72_id_zupt_detect_60s" else "0" * 64) + "\n")
        w("r72_id_stepcap_wide_60s", "engagement.txt", "  lio/ekf/step_cap mode=hard cap_rot=1e+06 rad cap_pos=1e+06 m iterations=%d bound=%d   <-- NOT ENGAGED\n" % (0 if bad == "cap_noiter" else 4000, 3 if bad == "cap_bound" else 0))
        w("r72_id_zupt_detect_60s", "engagement.txt", "  lio/zupt mode=detect_only scans=595 stationary=%d applied=%d\n" % (0 if bad == "zupt_nostat" else 470, 5 if bad == "zupt_applied" else 0))
        w("r72_id_nisgate_dead_60s", "engagement.txt", "  lio/q_adapt mode=nis_gate apply=both scans=595 scans_with_gain_ne_1=%d\n" % (2 if bad == "nis_moved" else 0))
        w("r72_id_nisgate_dead_60s", "engagement_imu.txt", "  q_gain steps=100 steps_with_gain_ne_1=0\n")
        w("r72_smoke_stepcap_fq_60s", "engagement.txt", "  lio/ekf/step_cap mode=hard iterations=100 bound=%d%s\n" % (0, "   <-- INERT: x" if bad == "cap_inert" else ""))
        w("r72_smoke_zupt_apply_60s", "engagement.txt", "  lio/zupt mode=apply scans=595 stationary=470 applied=%d\n" % (0 if bad == "zupt_apply0" else 460))
        w("r72_smoke_innov_scale_60s", "engagement.txt", "  lio/q_adapt mode=innovation_scale scans=595 scans_with_gain_ne_1=%d\n" % (0 if bad == "innov0" else 300))
        w("r72_smoke_nis_gate_60s", "engagement.txt", "  lio/q_adapt mode=nis_gate scans=595 scans_with_gain_ne_1=0   <-- NOT ENGAGED\n")
        w("r72_smoke_local_poly_60s", "engagement_imu.txt", "  motion/energy_source=local_poly order=2 scans_used=%d%s\n" % (0 if bad == "lp_inert" else 590, "   <-- INERT: x" if bad == "lp_inert" else ""))
        for arm, key in REFUSE.items():
            w(arm, "arm.json", json.dumps({"sentinel": "DONE" if bad == "refuse_ran" and arm == "r72_refuse_stepcap_zero" else "FAILED"}))
            w(arm, "run_log_tail.txt", "terminate called after throwing invalid_argument: lio/%s ...\n" % (key if not (bad == "refuse_key" and arm == "r72_refuse_zupt_threshold") else "other"))
        for t in ("tail", "measurement"):
            arm = "r72_smoke_co_%s_rec_60s" % t
            w(arm, "raw_sha256.txt", "".join("abc  coupled_record_f%d.txt\n" % i for i in range(1, 6 if bad != "rec_missing" else 4)) + ("abc  coupled_failure_dump.txt\n" if bad == "rec_dump" else ""))
            S = {"records": [{"iterations": [{"iter": 0, "A_pd_pass": bad != "rec_pd"}]}], "errors": []}
            w(arm, "r72_dump_summary.json", json.dumps(S))
    fails = []
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mk(good)
        g = run_gates(good)
        nb = [r for r in g.rows if r[1] != "PASS"]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb])
        n_gates = len(g.rows)
        # each breakage must flip exactly the named gate to FAIL
        cases = {"ident": "r72_id_zupt_detect_60s", "cap_noiter": "ENG stepcap_wide", "cap_bound": "ENG stepcap_wide", "zupt_nostat": "ENG zupt detect_only", "zupt_applied": "ENG zupt detect_only",
                 "nis_moved": "ENG nis_gate dead", "cap_inert": "SMOKE step cap", "zupt_apply0": "SMOKE zupt apply", "innov0": "SMOKE innovation_scale", "lp_inert": "SMOKE local_poly",
                 "refuse_ran": "REFUSE r72_refuse_stepcap_zero", "refuse_key": "REFUSE r72_refuse_zupt_threshold", "rec_missing": "RECORD r72_smoke_co_tail_rec_60s wrote", "rec_dump": "RECORD r72_smoke_co_tail_rec_60s wrote",
                 "rec_pd": "RECORD r72_smoke_co_tail_rec_60s dump reducer"}
        for bad, needle in cases.items():
            d = os.path.join(t, bad); mk(d, bad)
            gg = run_gates(d)
            failed = [r[0] for r in gg.rows if r[1] == "FAIL"]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not make a gate containing %r FAIL (failed: %s)" % (bad, needle, failed))
        # missing arm is MISSING, not PASS
        d = os.path.join(t, "empty"); os.makedirs(d)
        ge = run_gates(d)
        if any(r[1] != "MISSING" for r in ge.rows):
            fails.append("empty runs dir produced a non-MISSING gate")
    if fails:
        print("SELFTEST r72_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r72_gates PASS (%d gates all PASS on good data; %d single breakages each flipped their gate to FAIL; empty dir all MISSING)" % (n_gates, len(cases)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    g = run_gates(a.runs)
    txt = report(g)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    return 1 if any(r[1] == "FAIL" for r in g.rows) else 0


if __name__ == "__main__":
    sys.exit(main())
