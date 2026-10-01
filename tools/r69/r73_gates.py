#!/usr/bin/env python3
"""R73 gates: identity (metric tolerance + digest) gates, engagement gates, shadow / freeze / coupled-log gates, refusal gates.

    python3 tools/r69/r73_gates.py --runs <runs_dir> [--out r73_gates.txt]     runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py)
    python3 tools/r69/r73_gates.py --selftest                                    builds fake arm dirs: every gate PASS on good data, every single breakage FAILs its gate

PASS / FAIL / MISSING (MISSING = arm or file absent; never a pass). Exit 1 if any FAIL.
F-419 decision (Bryce, 2026-10-01): the three eee_01 reference arms must agree with the R72 values within 0.005 mm on ATE (full bag), excursion and settled offset; their NEW digests are
the R73 references (symmetrisation changed the arithmetic), every other identity arm must equal its reference digest.
"""
import argparse, gzip, io, json, os, re, shutil, sys, tempfile
import pandas as pd

TOL_MM = 0.005
R72 = {"r73_00_id_baseline_60s": {"excursion_mm": 82.00350294412522, "settled_offset_mm": 65.33203761816637},
       "r73_00_id_baseline_full": {"gt_ate_mm": 31.00227570852566, "excursion_mm": 82.00350294412522, "settled_offset_mm": 65.33203761816637},
       "r73_00_id_motionq_full": {"gt_ate_mm": 25.818652470553165, "excursion_mm": 3.7824168554633695, "settled_offset_mm": 2.496348347230532}}
REF60 = "r73_00_id_baseline_60s"
IDENT = {a: REF60 for a in ("r73_id_stepcap_wide_60s", "r73_id_zupt_detect_60s", "r73_id_nisgate_dead_60s", "r73_id_diag_on_60s", "r73_id_scanlog_60s", "r73_id_dump_lastiter_60s", "r73_id_shadow_same_60s")}
REFUSE = {"r73_refuse_stepcap_suboption": "step_cap", "r73_refuse_stepcap_coupled": "step_cap", "r73_refuse_stepcap_zero": "step_cap",
          "r73_refuse_gain_suboption": "process_noise/gain", "r73_refuse_gain_inert_forgetting": "process_noise/gain", "r73_refuse_gain_inert_bounds": "process_noise/gain",
          "r73_refuse_gain_coupled": "process_noise/gain", "r73_refuse_zupt_suboption": "zupt", "r73_refuse_zupt_threshold": "zupt",
          "r73_refuse_local_poly_fixed": "energy_source", "r73_refuse_local_poly_order_instant": "local_poly",
          "r73_refuse_wnoa_zero": "process_noise/disturbance", "r73_refuse_wnoa_suboption": "process_noise/disturbance",
          "r73_refuse_old_qadapt_name": "renamed", "r73_refuse_old_cprior_name": "renamed",
          "r73_refuse_shadow_window0": "shadow/window_start", "r73_refuse_shadow_window_order": "shadow/window_end", "r73_refuse_shadow_suboption": "shadow",
          "r73_refuse_shadow_motion_sub": "shadow/motion_beta", "r73_refuse_shadow_coupled": "shadow", "r73_refuse_scanlog_coupled": "scan_log",
          "r73_refuse_ipscale_iekf": "imu_prior_info_scale", "r73_refuse_ipscale_zero": "imu_prior_info_scale", "r73_refuse_dump_last_iter_alone": "residual_dump_last_iter"}
# the old-name refusals must also name the NEW key
REFUSE_ALSO = {"r73_refuse_old_qadapt_name": "imu/process_noise/gain/mode", "r73_refuse_old_cprior_name": "imu/process_noise/disturbance/mode"}


def rd(d, n):
    p = os.path.join(d, n)
    if os.path.exists(p):
        return open(p, errors="replace").read()
    if os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", "rt", errors="replace").read()
    return None


def num(txt, key):
    m = re.search(r"\b" + re.escape(key) + r"=(-?[0-9.eE+-]+)", txt or "")
    return float(m.group(1)) if m else None


def line(txt, tag):
    for l in (txt or "").splitlines():
        if tag in l:
            return l
    return None


def csv(d, n):
    t = rd(d, n)
    return pd.read_csv(io.StringIO(t), dtype=str) if t else None


class G:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))


def digest(runs, arm):
    return (rd(os.path.join(runs, arm), "state_trace_blank_runid.sha256") or "").strip() or None


def summ(runs, arm):
    p = os.path.join(runs, arm, "arm_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def run_gates(runs):
    g = G()
    A = lambda a: os.path.join(runs, a)
    # ---- F-419 identity: R72 metrics within 0.005 mm
    for arm, ref in R72.items():
        S = summ(runs, arm)
        for k, want in ref.items():
            nm = "ID %s %s within %.3f mm of R72" % (arm, k, TOL_MM)
            got = None if S is None else S.get(k)
            g.add(nm, "MISSING" if got is None else ("PASS" if abs(got - want) <= TOL_MM else "FAIL"), "" if got is None else "got %.6f R72 %.6f diff %.6f" % (got, want, got - want))
    d0 = digest(runs, REF60)
    g.add("ID %s has a digest (becomes the R73 reference)" % REF60, "MISSING" if d0 is None else "PASS", "" if d0 is None else d0[:16])
    for arm, ref in IDENT.items():
        a, b = digest(runs, arm), digest(runs, ref)
        g.add("ID %s == %s" % (arm, ref), "MISSING" if a is None or b is None else ("PASS" if a == b else "FAIL"), "" if a is None or b is None else "%s vs %s" % (a[:16], b[:16]))
    # ---- F-407: engagement_imu.txt exists for every arm that ran
    for arm in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        if arm.startswith("r73_refuse") or arm.endswith("broken_clone") or not os.path.exists(os.path.join(runs, arm, "arm.json")):
            continue
        if json.load(open(os.path.join(runs, arm, "arm.json"))).get("sentinel") == "FAILED":
            continue
        raw = rd(A(arm), "raw_sha256.txt") or ""
        g.add("ENGIMU %s wrote engagement_imu.txt" % arm, "PASS" if "engagement_imu.txt" in raw or os.path.exists(os.path.join(A(arm), "engagement_imu.txt")) else "FAIL")
    # ---- engagement of identity arms
    e = rd(A("r73_id_stepcap_wide_60s"), "engagement.txt")
    if e is None:
        g.add("ENG stepcap_wide ran and did not bind", "MISSING")
    else:
        it, bd = num(line(e, "lio/ekf/step_cap"), "iterations"), num(line(e, "lio/ekf/step_cap"), "bound")
        g.add("ENG stepcap_wide ran and did not bind", "PASS" if it and it > 0 and bd == 0 else "FAIL", "iterations=%s bound=%s" % (it, bd))
    e = rd(A("r73_id_zupt_detect_60s"), "engagement.txt")
    if e is None:
        g.add("ENG zupt detect_only ran, found rest, applied nothing", "MISSING")
    else:
        l = line(e, "lio/zupt")
        sc, st, ap = num(l, "scans"), num(l, "stationary"), num(l, "applied")
        g.add("ENG zupt detect_only ran, found rest, applied nothing", "PASS" if sc and sc > 0 and st and st > 0 and ap == 0 else "FAIL", "scans=%s stationary=%s applied=%s" % (sc, st, ap))
    e = rd(A("r73_id_nisgate_dead_60s"), "engagement.txt")
    ei = rd(A("r73_id_nisgate_dead_60s"), "engagement_imu.txt")
    if e is None:
        g.add("ENG nis_gate dead: evaluated every scan, gain never left 1", "MISSING")
    else:
        l = line(e, "imu/process_noise/gain")
        sc, ne = num(l, "scans"), num(l, "scans_with_gain_ne_1")
        imu_ne = num(line(ei, "q_gain"), "steps_with_gain_ne_1") if ei else None
        g.add("ENG nis_gate dead: evaluated every scan, gain never left 1", "PASS" if sc and sc > 0 and ne == 0 and imu_ne in (0, None) else "FAIL", "scans=%s gain_ne_1=%s imu_gain_ne_1=%s" % (sc, ne, imu_ne))

    def eng(arm, cond, name, imu=False):
        t = rd(A(arm), "engagement_imu.txt" if imu else "engagement.txt")
        if t is None:
            g.add(name, "MISSING"); return
        ok, det = cond(t)
        g.add(name, "PASS" if ok else "FAIL", det)
    eng("r73_smoke_stepcap_fq_60s", lambda t: ("INERT" not in (line(t, "lio/ekf/step_cap") or "INERT"), (line(t, "lio/ekf/step_cap") or "no line")[-120:]), "SMOKE step cap not INERT")
    eng("r73_smoke_zupt_apply_60s", lambda t: ((num(line(t, "lio/zupt"), "applied") or 0) > 0, (line(t, "lio/zupt") or "no line")[-140:]), "SMOKE zupt apply applied > 0")
    eng("r73_smoke_innov_scale_60s", lambda t: ((num(line(t, "imu/process_noise/gain"), "scans_with_gain_ne_1") or 0) > 0, (line(t, "imu/process_noise/gain") or "no line")[:160]), "SMOKE innovation_scale moved the gain")
    eng("r73_smoke_nis_gate_60s", lambda t: ("INERT" not in (line(t, "imu/process_noise/gain") or "INERT"), (line(t, "imu/process_noise/gain") or "no line")[:160]), "SMOKE nis_gate not INERT")
    eng("r73_smoke_local_poly_60s", lambda t: ("INERT" not in (line(t, "energy_source=local_poly") or "INERT"), (line(t, "energy_source=local_poly") or "no line")[:160]), "SMOKE local_poly not INERT", imu=True)
    eng("r73_smoke_wnoa_60s", lambda t: ("INERT" not in (line(t, "wnoa") or "INERT"), (line(t, "wnoa") or "no line")[:160]), "SMOKE wnoa not INERT", imu=True)
    # ---- shadow smokes and the shadow identity arm: engagement and table contents
    def shadow_ok(arm, lo, hi, tag):
        d = A(arm)
        S = csv(d, "shadow_scan_all.csv") if csv(d, "shadow_scan_all.csv") is not None else csv(d, "shadow_scan.csv")
        e = rd(d, "engagement.txt")
        if S is None or e is None:
            g.add("SHADOW %s shadow_scan table and engagement present" % tag, "MISSING"); return
        l = line(e, "lio/shadow")
        fr = pd.to_numeric(S["frame"])
        nd, ns = int(((S["role"] == "driver") & (fr >= lo) & (fr <= hi)).sum()), int(((S["role"] == "shadow") & (fr >= lo) & (fr <= hi)).sum())
        want = hi - lo + 1
        changed = num(l, "scans_where_the_map_digest_changed")
        sh = S[S["role"] == "shadow"]
        digest_ok = bool((sh["map_digest_before_pass"] == sh["map_digest_after_pass"]).all()) and len(sh) > 0
        runs_ = num(l, "scans_run")
        ok = nd == want and ns == want and changed == 0 and digest_ok and runs_ == want and "INERT" not in (l or "INERT") and "INVALID" not in (l or "")
        g.add("SHADOW %s: a driver and a shadow row for every window frame, map digest unchanged by every shadow pass" % tag, "PASS" if ok else "FAIL",
              "driver_rows=%d shadow_rows=%d want=%d scans_run=%s digest_changed=%s digests_equal=%s" % (nd, ns, want, runs_, changed, digest_ok))
    shadow_ok("r73_smoke_shadow_iso_60s", 20, 50, "smoke_iso")
    shadow_ok("r73_smoke_shadow_fixed_mq_60s", 20, 50, "smoke_fixed_mq")
    shadow_ok("r73_id_shadow_same_60s", 20, 50, "identity_same")
    # the shadow ImuProc is a separate instance: its engagement file exists and counts propagations
    t = rd(A("r73_smoke_shadow_iso_60s"), "engagement_imu_shadow.txt")
    g.add("SHADOW smoke_iso engagement_imu_shadow.txt written by the shadow ImuProc", "MISSING" if t is None else ("PASS" if t.strip() else "FAIL"))
    # ---- freeze smoke
    d = A("r73_smoke_freeze_60s")
    S = csv(d, "scan_log_all.csv")
    if S is None:
        g.add("FREEZE smoke: map digest constant after frame 1, frozen scans counted", "MISSING")
    else:
        S = S.assign(frame=pd.to_numeric(S["frame"])).sort_values("frame")
        n_after = S[S["frame"] > S["frame"].iloc[0]]["map_digest_exact"].nunique()
        fz = int(float(S["map_frozen_scans"].iloc[-1]))
        g.add("FREEZE smoke: map digest constant after frame 1, frozen scans counted", "PASS" if n_after <= 1 and fz > 0 else "FAIL", "distinct digests after frame 1=%d frozen_scans_last=%d" % (n_after, fz))
    # ---- coupled smokes: record, update log, no failure dump, info-scale engagement
    for t in ("tail", "measurement"):
        arm = "r73_smoke_co_%s_rec_60s" % t
        d = A(arm)
        raw = rd(d, "raw_sha256.txt") if os.path.isdir(d) else None
        if raw is None:
            g.add("RECORD %s wrote coupled_record_f1..f5, update logs and no failure dump" % arm, "MISSING"); continue
        have = all(("coupled_record_f%d.txt" % i) in raw for i in range(1, 6)) and "coupled_update_iter.csv" in raw and "coupled_update_scan.csv" in raw
        g.add("RECORD %s wrote coupled_record_f1..f5, update logs and no failure dump" % arm, "PASS" if have and "coupled_failure_dump.txt" not in raw else "FAIL", "have=%s dump=%s" % (have, "coupled_failure_dump.txt" in raw))
        p = os.path.join(d, "r73_dump_summary.json")
        if os.path.exists(p):
            S = json.load(open(p))
            n_it = sum(len(r["iterations"]) for r in S["records"])
            allpd = all(it.get("A_pd_pass") for r in S["records"] for it in r["iterations"])
            g.add("RECORD %s dump reducer parsed all records, A_pd_pass in every iteration" % arm, "PASS" if S["records"] and n_it > 0 and allpd and not S["errors"] else "FAIL", "records=%d iterations=%d errors=%d" % (len(S["records"]), n_it, len(S["errors"])))
        else:
            g.add("RECORD %s dump reducer parsed all records, A_pd_pass in every iteration" % arm, "MISSING")
    d = A("r73_smoke_co_ipscale_60s")
    e = rd(d, "engagement.txt")
    U = csv(d, "coupled_update_iter_all.csv")
    if e is None or U is None:
        g.add("IPSCALE smoke: engagement and every update-log row carry imu_prior_info_scale 0.1", "MISSING")
    else:
        ok = "imu_prior_info_scale=0.1" in e and (pd.to_numeric(U["imu_prior_info_scale"]) == 0.1).all()
        g.add("IPSCALE smoke: engagement and every update-log row carry imu_prior_info_scale 0.1", "PASS" if ok else "FAIL")
    # ---- refusals
    for arm, key in REFUSE.items():
        nm = "REFUSE %s dies at startup naming %s" % (arm, key)
        d = A(arm)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "arm.json")):
            g.add(nm, "MISSING"); continue
        meta = json.load(open(os.path.join(d, "arm.json")))
        tail = (rd(d, "run_log_tail.txt") or "") + (rd(d, "run.log") or "")
        ok = meta.get("sentinel") == "FAILED" and key in tail and (arm not in REFUSE_ALSO or REFUSE_ALSO[arm] in tail)
        g.add(nm, "PASS" if ok else "FAIL", "sentinel=%s key_in_log=%s" % (meta.get("sentinel"), key in tail))
    return g


def report(g):
    out = ["%-8s %s%s" % (s, n, ("   [" + d + "]") if d else "") for n, s, d in g.rows]
    out += ["", "PASS %d  FAIL %d  MISSING %d" % tuple(sum(1 for r in g.rows if r[1] == k) for k in ("PASS", "FAIL", "MISSING"))]
    return "\n".join(out)


def mkfake(root, bad=None):
    def w(arm, name, text):
        os.makedirs(os.path.join(root, arm), exist_ok=True)
        open(os.path.join(root, arm, name), "w").write(text)

    def tab(rows):
        return pd.DataFrame(rows).to_csv(index=False)
    for arm, ref in R72.items():
        S = dict(ref)
        if bad == "metric" and arm == "r73_00_id_baseline_full":
            S["excursion_mm"] += 0.006
        w(arm, "arm_summary.json", json.dumps(S))
        w(arm, "arm.json", json.dumps({"sentinel": "DONE"})); w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
        w(arm, "state_trace_blank_runid.sha256", "a" * 64 + "\n")
    for arm in IDENT:
        w(arm, "state_trace_blank_runid.sha256", ("b" * 64 if (bad == "ident" and arm == "r73_id_zupt_detect_60s") else "a" * 64) + "\n")
        w(arm, "arm.json", json.dumps({"sentinel": "DONE"})); w(arm, "raw_sha256.txt", "x  engagement_imu.txt\n")
    w("r73_00_id_baseline_60s", "state_trace_blank_runid.sha256", "a" * 64 + "\n")
    w("r73_id_stepcap_wide_60s", "engagement.txt", "  lio/ekf/step_cap iterations=%d bound=%d\n" % (0 if bad == "cap_noiter" else 4000, 3 if bad == "cap_bound" else 0))
    w("r73_id_zupt_detect_60s", "engagement.txt", "  lio/zupt mode=detect_only scans=595 stationary=%d applied=%d\n" % (0 if bad == "zupt_nostat" else 470, 5 if bad == "zupt_applied" else 0))
    w("r73_id_nisgate_dead_60s", "engagement.txt", "  imu/process_noise/gain mode=nis_gate scans=595 scans_with_gain_ne_1=%d\n" % (2 if bad == "nis_moved" else 0))
    w("r73_id_nisgate_dead_60s", "engagement_imu.txt", "  q_gain steps=100 steps_with_gain_ne_1=0\n")
    w("r73_smoke_stepcap_fq_60s", "engagement.txt", "  lio/ekf/step_cap iterations=100 bound=3%s\n" % ("   <-- INERT: x" if bad == "cap_inert" else ""))
    w("r73_smoke_zupt_apply_60s", "engagement.txt", "  lio/zupt mode=apply scans=595 stationary=470 applied=%d\n" % (0 if bad == "zupt_apply0" else 460))
    w("r73_smoke_innov_scale_60s", "engagement.txt", "  imu/process_noise/gain mode=innovation_scale scans_with_gain_ne_1=%d\n" % (0 if bad == "innov0" else 300))
    w("r73_smoke_nis_gate_60s", "engagement.txt", "  imu/process_noise/gain mode=nis_gate scans_with_gain_ne_1=4\n")
    w("r73_smoke_local_poly_60s", "engagement_imu.txt", "  motion/energy_source=local_poly order=2 scans_used=590%s\n" % ("   <-- INERT: x" if bad == "lp_inert" else ""))
    w("r73_smoke_wnoa_60s", "engagement_imu.txt", "  wnoa steps=100\n")
    for arm in ("r73_smoke_shadow_iso_60s", "r73_smoke_shadow_fixed_mq_60s", "r73_id_shadow_same_60s"):
        rows = []
        for f in range(20, 51 if not (bad == "shadow_rows" and arm == "r73_smoke_shadow_iso_60s") else 49):
            for role in ("driver", "shadow"):
                dg = "dead" if bad != "shadow_digest" or role == "driver" or arm != "r73_smoke_shadow_iso_60s" else "beef"
                rows.append(dict(frame=f, role=role, map_digest_before_pass="dead" if role == "shadow" else "", map_digest_after_pass=dg if role == "shadow" else ""))
        w(arm, "shadow_scan_all.csv", tab(rows))
        w(arm, "engagement.txt", "  lio/shadow mode=on window=[20,50] scans_run=31 scans_without_residuals=0 scans_where_the_map_digest_changed=%d shadow_propagations=31\n" % (1 if bad == "shadow_digest" and arm == "r73_smoke_shadow_iso_60s" else 0))
    w("r73_smoke_shadow_iso_60s", "engagement_imu_shadow.txt", "" if bad == "shadow_imu_empty" else "  shadow steps=31\n")
    rows = [dict(frame=f, map_digest_exact="d1" if (f == 1 or bad != "freeze_moves") else "d%d" % f, map_frozen_scans=f) for f in range(1, 30)]
    w("r73_smoke_freeze_60s", "scan_log_all.csv", tab(rows))
    for t in ("tail", "measurement"):
        arm = "r73_smoke_co_%s_rec_60s" % t
        w(arm, "raw_sha256.txt", "".join("abc  coupled_record_f%d.txt\n" % i for i in range(1, 6 if bad != "rec_missing" else 4)) + "abc  coupled_update_iter.csv\nabc  coupled_update_scan.csv\nabc  engagement_imu.txt\n" + ("abc  coupled_failure_dump.txt\n" if bad == "rec_dump" else ""))
        w(arm, "r73_dump_summary.json", json.dumps({"records": [{"iterations": [{"iter": 0, "A_pd_pass": bad != "rec_pd"}]}], "errors": []}))
        w(arm, "arm.json", json.dumps({"sentinel": "DONE"}))
    w("r73_smoke_co_ipscale_60s", "engagement.txt", "  estimator/mode=coupled imu_prior_info_scale=%s\n" % ("1" if bad == "ips_eng" else "0.1"))
    w("r73_smoke_co_ipscale_60s", "coupled_update_iter_all.csv", tab([dict(frame=1, imu_prior_info_scale=0.1), dict(frame=2, imu_prior_info_scale=0.1 if bad != "ips_rows" else 1.0)]))
    for arm, key in REFUSE.items():
        w(arm, "arm.json", json.dumps({"sentinel": "DONE" if bad == "refuse_ran" and arm == "r73_refuse_stepcap_zero" else "FAILED"}))
        k = "other" if (bad == "refuse_key" and arm == "r73_refuse_zupt_threshold") else key
        extra = REFUSE_ALSO.get(arm, "") if not (bad == "refuse_nonew" and arm == "r73_refuse_old_qadapt_name") else ""
        w(arm, "run_log_tail.txt", "terminate called after throwing invalid_argument: %s %s\n" % (k, extra))


def selftest():
    cases = {"metric": "ID r73_00_id_baseline_full excursion_mm", "ident": "ID r73_id_zupt_detect_60s", "cap_noiter": "ENG stepcap_wide", "cap_bound": "ENG stepcap_wide",
             "zupt_nostat": "ENG zupt detect_only", "zupt_applied": "ENG zupt detect_only", "nis_moved": "ENG nis_gate dead", "cap_inert": "SMOKE step cap", "zupt_apply0": "SMOKE zupt apply",
             "innov0": "SMOKE innovation_scale", "lp_inert": "SMOKE local_poly", "shadow_rows": "SHADOW smoke_iso", "shadow_digest": "SHADOW smoke_iso", "shadow_imu_empty": "SHADOW smoke_iso engagement_imu_shadow",
             "freeze_moves": "FREEZE smoke", "rec_missing": "RECORD r73_smoke_co_tail_rec_60s wrote", "rec_dump": "RECORD r73_smoke_co_tail_rec_60s wrote", "rec_pd": "RECORD r73_smoke_co_tail_rec_60s dump reducer",
             "ips_eng": "IPSCALE smoke", "ips_rows": "IPSCALE smoke", "refuse_ran": "REFUSE r73_refuse_stepcap_zero", "refuse_key": "REFUSE r73_refuse_zupt_threshold", "refuse_nonew": "REFUSE r73_refuse_old_qadapt_name"}
    fails = []
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good"); mkfake(good)
        g = run_gates(good)
        nb = [r for r in g.rows if r[1] != "PASS"]
        if nb:
            fails.append("good data: non-PASS gates: %s" % [(r[0], r[1]) for r in nb])
        n_gates = len(g.rows)
        for bad, needle in cases.items():
            d = os.path.join(t, bad); mkfake(d, bad)
            failed = [r[0] for r in run_gates(d).rows if r[1] == "FAIL"]
            if not any(needle in f for f in failed):
                fails.append("breakage %s did not FAIL a gate containing %r (failed: %s)" % (bad, needle, failed))
        e = os.path.join(t, "empty"); os.makedirs(e)
        if any(r[1] != "MISSING" for r in run_gates(e).rows):
            fails.append("empty runs dir produced a non-MISSING gate")
    if fails:
        print("SELFTEST r73_gates FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r73_gates PASS (%d gates PASS on good data; %d single breakages each flipped their gate to FAIL; empty dir all MISSING)" % (n_gates, len(cases)))
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
