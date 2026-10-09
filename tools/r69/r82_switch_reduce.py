#!/usr/bin/env python3
"""R81 (F-479 step 2): comparison of each process-noise SWITCH arm with the two plain controls, alignment-free, in each run's own first-frame coordinates.
Adapts r74_switch_reduce.py part 2 to the R81 arm ids and to the metric rule (Bryce 2026-10-05; see r82_alignfree.py: first-frame position and heading removed per run, first
frame = state_reference.txt when found, else the posterior of scan 1; no ATE; joined on scan_id).

    python3 tools/r69/r82_switch_reduce.py --runs <return compact dir> [--controls <dir> ...] --phase-summary <phase_summary.csv> [--phase-summary ...] \
        [--state-ref-root <dir>] [--onsets eee_02=481,sbs_03=340] [--thr-mm 50] [--out-dir <dir>]
    python3 tools/r69/r82_switch_reduce.py --selftest

ARM IDS (this is the naming of B1's make_r82_manifest.py as read from /home/claude/work/R81/pkg/r82_manifest/arm_table.tsv: swx = switch arms, swc = dc plain controls)
  switch arms   r82_<mode>_swx_<bag>_p<1|3>_<fq2mq|mq2fq>     mode dc|rt;  fq2mq: config A (fixed-Q) up to onset-1, config B (motion-Q) from the onset; mq2fq the reverse
  controls dc   r82_dc_swc_<bag>_fq   r82_dc_swc_<bag>_mq      (plain DISCARD controls, pfn 3, same config)
  controls rt   r82_rt_mg_<bag>_fq    r82_rt_b3_<bag>_mq       (the RR arms of the same bags; not duplicated)
  --arm-table (optional TSV): columns arm, ctl_a, ctl_b, onset (any may be empty) override the derived names and onset of that arm (B1's arm_table has none of them: not needed).
  Controls and arms are looked for as <dir>/<id>/pos_trace_all.csv.gz in --runs and every --controls dir.
  Onset (the scan of the switch): --onsets, else onset_scan_used of the phase_summary row of the arm, else of r80_b3_<bag>_mq (eee_02 481, sbs_03 340 in R80).

PER SCAN (r82_switch_frames.csv.gz, the name in B1's required_outputs; first-frame coordinates p'; every scan_id present in the arm AND both controls)
  dA_mm, dB_mm   |p'_switch - p'_A|, |p'_switch - p'_B| (position departure from control A / B, mm); dAB_mm = |p'_A - p'_B|
  stepdA_mm, stepdB_mm  |step_switch - step_A|, |step_switch - step_B| of the per-scan steps (the r74 F-426 part 2 quantity) in first-frame coordinates; blank at the first scan
  at_or_after_onset  1 from the onset scan on
PER ARM (r82_switch_pairs.csv)
  dA_pre_onset_max_mm   SANITY: the switch arm is config A before the onset, so this must be ~0 (any value above thr-mm is written to `note`; it would mean the switch changed something earlier)
  dB_at_onset_mm        departure from B at the onset scan (= how far apart A and B already are)
  leave_A_scan          first scan_id with dA >= thr (blank if never); leave_B_scan_from_onset: first scan_id >= onset with dB >= thr; leave_B_scan_any: first scan_id at all with dB >= thr
                        (A and B differ from scan 1 on, so leave_B_scan_any can lie before the onset and says nothing about the switch)
  dA_post_max_mm, dB_post_max_mm, dAB_post_max_mm   maxima over scans >= onset
  n_sep                 scans >= onset where the controls are separated by >= 2*thr (a run cannot be within thr of both there)
  frac_near_A, frac_near_B   over those scans: share with dA < thr, share with dB < thr
  verdict   (rule fixed here, planner to confirm; it needs n_sep >= 10)
              `follows B after switch`   frac_near_B >= 0.5 and frac_near_B > frac_near_A  (the arm left A and stays with B)
              `follows A`                frac_near_A >= 0.5 and frac_near_A >= frac_near_B (the history decides, config B did not move it)
              `neither`                  otherwise (it left A and does not sit with B either)
              `not separable`            n_sep < 10 (the two controls stay within 2*thr of each other: nothing to discriminate)
  dep_min_ctl_mm   max over scans of min(dA, dB) (how far from BOTH controls it ever is)
  gt_max_mm, collapse_flag  1 when dep_min_ctl_mm >= 500 AND gt_max_mm > 500 (rule 4.15c; the closest control is >= 500 mm away); collapse_vs_A / collapse_vs_B the same against one control
  first_frame_src  arm|A|B sources (state_reference / first_row)
r82_switch_summary.md : the table of the arms (one row each), with the ids, the onset and the verdict.
Planning agent's script. Run by me only on synthetic traces.
"""
import argparse, csv, gzip, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r82_alignfree as AF  # noqa: E402

BAG = AF.BAG
ARM_RE = re.compile(r"^r82_(dc|rt)_swx_(%s)_p([13])_(fq2mq|mq2fq)$" % BAG)
PAIR_COLS = ["arm", "mode", "bag", "direction", "ctl_a", "ctl_b", "onset", "onset_source", "n_joined", "dA_pre_onset_max_mm", "dB_at_onset_mm", "leave_A_scan", "leave_B_scan_from_onset", "leave_B_scan_any",
             "dA_post_max_mm", "dB_post_max_mm", "dAB_post_max_mm", "n_sep", "frac_near_A", "frac_near_B", "verdict", "dep_min_ctl_mm", "gt_max_mm", "collapse_flag", "collapse_vs_A", "collapse_vs_B",
             "first_frame_src", "note"]
FRAME_COLS = ["arm", "mode", "bag", "direction", "frame", "dA_mm", "dB_mm", "dAB_mm", "stepdA_mm", "stepdB_mm", "at_or_after_onset"]


def control_ids(mode, bag, direction, pfn="1"):
    """-> (A, B): the config before and after the onset. Controls are the F-503 downsampling arms of the same batch: point filter 1 -> 'off', 3 -> 'first'."""
    ds = "off" if str(pfn) == "1" else "first"
    fq = "r82_%s_ds_%s_%s_fq" % (mode, bag, ds)
    mq = "r82_%s_ds_%s_%s_mq" % (mode, bag, ds)
    return (fq, mq) if direction == "fq2mq" else (mq, fq)


def steps(T, frames):
    """per-scan steps (mm) of the first-frame positions on the given scan list; NaN at the first scan and where a scan is missing."""
    pos = np.full((len(frames), 3), np.nan)
    ok = np.isin(frames, T.ids)
    pos[ok] = T.pf[np.searchsorted(T.ids, frames[ok])]
    st = np.full((len(frames), 3), np.nan)
    st[1:] = (pos[1:] - pos[:-1]) * 1000.0
    return pos, st


def compare(arm, mode, bag, direction, S, A, B, onset, onset_source, ids_a, ids_b, gt_max, thr=50.0):
    """S, A, B: AF.Trace of the switch arm and controls A, B. -> (frames list of dict, pair row dict)."""
    frames = np.intersect1d(np.intersect1d(S.ids, A.ids), B.ids)
    pS, sS = steps(S, frames); pA, sA = steps(A, frames); pB, sB = steps(B, frames)
    dA, dB = 1000.0 * np.linalg.norm(pS - pA, axis=1), 1000.0 * np.linalg.norm(pS - pB, axis=1)
    dAB = 1000.0 * np.linalg.norm(pA - pB, axis=1)
    sdA, sdB = np.linalg.norm(sS - sA, axis=1), np.linalg.norm(sS - sB, axis=1)
    post = frames >= onset
    FR = [dict(arm=arm, mode=mode, bag=bag, direction=direction, frame=int(f), dA_mm=round(float(a), 4), dB_mm=round(float(b), 4), dAB_mm=round(float(c), 4),
               stepdA_mm=("" if not math.isfinite(x) else round(float(x), 4)), stepdB_mm=("" if not math.isfinite(y) else round(float(y), 4)), at_or_after_onset=int(f >= onset))
          for f, a, b, c, x, y in zip(frames, dA, dB, dAB, sdA, sdB)]
    notes = []
    pre = ~post
    r = dict(arm=arm, mode=mode, bag=bag, direction=direction, ctl_a=ids_a, ctl_b=ids_b, onset=int(onset), onset_source=onset_source, n_joined=int(len(frames)))
    r["dA_pre_onset_max_mm"] = round(float(dA[pre].max()), 4) if pre.any() else ""
    if pre.any() and dA[pre].max() >= thr:
        notes.append("departs from control A before the onset (max %.1f mm): the switch changed something earlier or A is not the same config" % dA[pre].max())
    i0 = int(np.searchsorted(frames, onset))
    r["dB_at_onset_mm"] = round(float(dB[i0]), 4) if i0 < len(frames) else ""
    first = lambda d, m: (int(frames[np.nonzero(m & (d >= thr))[0][0]]) if (m & (d >= thr)).any() else "")
    r["leave_A_scan"] = first(dA, np.ones(len(frames), bool))
    r["leave_B_scan_from_onset"] = first(dB, post)
    r["leave_B_scan_any"] = first(dB, np.ones(len(frames), bool))
    r["dA_post_max_mm"] = round(float(dA[post].max()), 4) if post.any() else ""
    r["dB_post_max_mm"] = round(float(dB[post].max()), 4) if post.any() else ""
    r["dAB_post_max_mm"] = round(float(dAB[post].max()), 4) if post.any() else ""
    sep = post & (dAB >= 2.0 * thr)
    r["n_sep"] = int(sep.sum())
    if sep.sum() >= 10:
        fa, fb = float((dA[sep] < thr).mean()), float((dB[sep] < thr).mean())
        r["frac_near_A"], r["frac_near_B"] = round(fa, 4), round(fb, 4)
        r["verdict"] = "follows B after switch" if (fb >= 0.5 and fb > fa) else ("follows A" if (fa >= 0.5 and fa >= fb) else "neither")
    else:
        r["frac_near_A"] = r["frac_near_B"] = ""
        r["verdict"] = "not separable"
    dmin = np.minimum(dA, dB)
    r["dep_min_ctl_mm"] = round(float(dmin.max()), 4) if len(frames) else ""
    r["gt_max_mm"] = "" if gt_max is None else round(gt_max, 3)
    if gt_max is None or not len(frames):
        r["collapse_flag"] = r["collapse_vs_A"] = r["collapse_vs_B"] = ""
    else:
        r["collapse_flag"] = int(dmin.max() >= 500.0 and gt_max > 500.0)
        r["collapse_vs_A"] = int(dA.max() >= 500.0 and gt_max > 500.0)
        r["collapse_vs_B"] = int(dB.max() >= 500.0 and gt_max > 500.0)
    r["first_frame_src"] = "%s|%s|%s" % (S.src, A.src, B.src)
    if len({S.src, A.src, B.src}) > 1:
        notes.append("first-frame sources differ")
    if S.src == "first_row":
        notes.append("first frame = posterior of scan 1 (no state_reference.txt)")
    r["note"] = "; ".join(notes)
    return FR, r


def md_table(rows):
    h = ["arm", "onset", "ctl A / ctl B", "leave A", "leave B (from onset)", "dB at onset", "dA post max", "dB post max", "frac near A / B", "verdict", "collapse"]
    L = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        fr = "%s / %s" % (r["frac_near_A"], r["frac_near_B"]) if r["frac_near_A"] != "" else "n_sep=%s" % r["n_sep"]
        L.append("| %s | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["arm"], r["onset"], r["ctl_a"], r["ctl_b"], r["leave_A_scan"] or "-", r["leave_B_scan_from_onset"] or "-",
                                                                                   r["dB_at_onset_mm"], r["dA_post_max_mm"], r["dB_post_max_mm"], fr, r["verdict"], r["collapse_flag"]))
    return "\n".join(L)


def run(a):
    phase = AF.load_phase(a.phase_summary)
    table = AF.load_table(a.arm_table)
    onsets = {}
    for kv in (a.onsets or "").split(","):
        if kv.strip():
            k, v = kv.split("="); onsets[k.strip()] = int(v)
    F = AF.Finder([a.runs] + list(a.controls), a.state_ref_root)
    arms = sorted(d for d in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, d)) and ARM_RE.match(d))
    FR, PR, bad = [], [], []
    for arm in arms:
        m = ARM_RE.match(arm); mode, bag, pfn, direction = m.group(1), m.group(2), m.group(3), m.group(4)
        row = table.get(arm, {})
        ca, cb = control_ids(mode, bag, direction, pfn)
        ca, cb = row.get("ctl_a") or ca, row.get("ctl_b") or cb
        S, A, B = F.get(arm), F.get(ca), F.get(cb)
        if S is None or A is None or B is None:
            bad.append("%s (missing %s)" % (arm, ", ".join(n for n, t in ((arm, S), (ca, A), (cb, B)) if t is None))); continue
        if row.get("onset"):
            on, osrc = int(row["onset"]), "arm-table"
        elif bag in onsets:
            on, osrc = onsets[bag], "--onsets"
        else:
            on, osrc = AF.pick_onset(arm, dict(src=arm, bag=bag, ctl=ca), phase)
            if on is None:
                bad.append("%s (no onset)" % arm); continue
        g = AF.num(phase.get(arm, {}).get("gt_max_mm"))
        f, r = compare(arm, mode, bag, direction, S, A, B, on, osrc, ca, cb, g, a.thr_mm)
        FR += f; PR.append(r)
    os.makedirs(a.out_dir, exist_ok=True)
    with gzip.open(os.path.join(a.out_dir, "r82_switch_frames.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, FRAME_COLS); w.writeheader(); w.writerows(FR)
    with open(os.path.join(a.out_dir, "r82_switch_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, PAIR_COLS); w.writeheader(); w.writerows(PR)
    with open(os.path.join(a.out_dir, "r82_switch_summary.md"), "w") as f:
        f.write("# R82 D5: switch arms against the two plain controls (alignment-free, own first-frame coordinates, threshold %g mm)\n\n" % a.thr_mm)
        for mode in ("dc", "rt"):
            rows = [r for r in PR if r["mode"] == mode]
            if rows:
                f.write("## mode %s\n\n%s\n\n" % (mode, md_table(rows)))
        f.write("Verdict rule: see the docstring of r82_switch_reduce.py. A = config before the onset, B = after. A 'not separable' row says the controls stay within %g mm of each other after the onset.\n" % (2 * a.thr_mm))
    exp = getattr(a, "expect", 8)
    print("switch arms %d (expected %d: D5 cells eee_01 p1 dc, eee_01 p1 rt, nya_01 p1 dc, sbs_02 p3 rt, each x 2 directions), problems %d %s" % (len(PR), exp, len(bad), bad[:6]))
    return 1 if (bad or len(PR) != exp) else 0


def selftest():
    import tempfile, shutil
    fails = []

    def chk(c, m):
        if not c: fails.append(m)
    t = tempfile.mkdtemp()
    try:
        n = 700; ids = np.arange(1, n + 1); on = 340
        Rz = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        FQ = np.zeros((n, 3)); FQ[:, 0] = 0.0005 * ids                              # plain fixed-Q control, healthy
        MQ = FQ.copy(); MQ[:, 1] += 0.06 * np.minimum(1.0, (ids - 1) / 20.0)        # plain motion-Q control: settles 60 mm apart in y within 20 scans (both start at the same first frame) ...
        MQ[on:, 0] += 0.02 * (ids[on:] - on)                                        # ... and departing at the onset, 20 mm per scan

        def mk(x, y0):
            """world trace of a run whose first-frame trajectory is x but that started with heading y0 at another position."""
            return (Rz(y0) @ x.T).T + np.array([3.0, -2.0, 0.0]), np.zeros(n) + y0
        behaviour = {"eee_02": "B", "sbs_03": "A", "nya_01": "neither", "sbs_01": "A"}
        for mode in ("dc", "rt"):
            for bag, beh in behaviour.items():
                for d in ("fq2mq", "mq2fq"):
                    nameA, nameB = control_ids(mode, bag, d)
                    Aarr, Barr = (FQ, MQ) if d == "fq2mq" else (MQ, FQ)
                    sw = Aarr.copy()
                    if beh == "B": sw[on:] = Barr[on:]                               # config A to the onset (scan_id on+1 is the first row of the slice), config B after
                    if beh == "neither": sw[on:, 2] += 0.03 * (ids[on:] - on)
                    for nm, arr, y0 in (("r82_%s_swx_%s_p1_%s" % (mode, bag, d), sw, 0.4), (nameA, Aarr, -1.3), (nameB, Barr, 2.2)):
                        xyz, yaw = mk(arr, y0)
                        AF._write_trace(os.path.join(t, "ret", nm, "pos_trace_all.csv.gz"), ids, xyz, yaw)
        ps = os.path.join(t, "ps.csv")
        with open(ps, "w") as f:
            f.write("arm,onset_scan_used,gt_max_mm\nr82_dc_ds_nya_01_off_mq,%d,100\nr82_dc_ds_sbs_01_off_mq,%d,100\nr82_rt_swx_eee_02_p1_fq2mq,481,900\n" % (on, on))
        ns = argparse.Namespace(expect=16, runs=os.path.join(t, "ret"), controls=[], phase_summary=[ps], arm_table=None, state_ref_root=[], onsets="eee_02=%d,sbs_03=%d,nya_01=%d,sbs_01=%d" % (on, on, on, on), thr_mm=50.0, out_dir=os.path.join(t, "out"))
        rc = run(ns)
        P = {r["arm"]: r for r in csv.DictReader(open(os.path.join(t, "out", "r82_switch_pairs.csv")))}
        chk(rc == 0 and len(P) == 16, "16 arms must exit 0: rc %r n %d" % (rc, len(P)))
        r = P["r82_dc_swx_eee_02_p1_fq2mq"]
        chk(float(r["dA_pre_onset_max_mm"]) < 1e-6 and r["leave_A_scan"] == str(on + 1), "pre-onset identical to A, leaves A at the first scan after the onset: %r" % {k: r[k] for k in ("leave_A_scan", "dA_pre_onset_max_mm")})
        chk(r["verdict"] == "follows B after switch" and float(r["frac_near_B"]) == 1.0, "verdict follow B: %r" % r["verdict"])
        chk(all(P["r82_%s_swx_%s_p1_%s" % (m_, b_, d_)]["verdict"] == "follows A" for m_ in ("dc", "rt") for b_ in ("sbs_03", "sbs_01") for d_ in ("fq2mq", "mq2fq")), "follows A")
        chk(P["r82_dc_swx_nya_01_p1_fq2mq"]["verdict"] == "neither", "neither: %r" % P["r82_dc_swx_nya_01_p1_fq2mq"]["verdict"])
        chk(P["r82_rt_swx_eee_02_p1_mq2fq"]["verdict"] == "follows B after switch", "mq2fq follows B (= fq): %r" % P["r82_rt_swx_eee_02_p1_mq2fq"]["verdict"])
        chk(P["r82_rt_swx_eee_02_p1_mq2fq"]["ctl_a"] == "r82_rt_ds_eee_02_off_mq" and P["r82_rt_swx_eee_02_p1_mq2fq"]["ctl_b"] == "r82_rt_ds_eee_02_off_fq", "rt control naming")
        chk(P["r82_dc_swx_eee_02_p1_fq2mq"]["ctl_a"] == "r82_dc_ds_eee_02_off_fq" and P["r82_dc_swx_eee_02_p1_fq2mq"]["ctl_b"] == "r82_dc_ds_eee_02_off_mq", "dc control naming")
        chk(P["r82_dc_swx_nya_01_p1_fq2mq"]["onset"] == str(on) and P["r82_dc_swx_nya_01_p1_fq2mq"]["onset_source"] == "--onsets", "onset from --onsets")
        # collapse: rt eee_02 fq2mq has gt_max 900 and follows B, which is ~7 m from A: min rule 0 (it sits on B), collapse_vs_A 1
        r = P["r82_rt_swx_eee_02_p1_fq2mq"]
        chk(r["collapse_flag"] == "0" and r["collapse_vs_A"] == "1" and r["collapse_vs_B"] == "0", "collapse flags: %r" % {k: r[k] for k in ("collapse_flag", "collapse_vs_A", "collapse_vs_B", "gt_max_mm")})
        chk(P["r82_dc_swx_nya_01_p1_fq2mq"]["collapse_flag"] == "", "gt_max unknown -> blank")
        Fr = list(csv.DictReader(gzip.open(os.path.join(t, "out", "r82_switch_frames.csv.gz"), "rt")))
        chk(len(Fr) == 16 * n and all(x["dA_mm"] != "" for x in Fr), "frames table size %d" % len(Fr))
        z = [x for x in Fr if x["arm"] == "r82_dc_swx_eee_02_p1_fq2mq" and int(x["frame"]) == on + 5][0]
        chk(float(z["dA_mm"]) > 100.0 and float(z["dB_mm"]) < 1e-6 and z["at_or_after_onset"] == "1", "frame row: %r" % z)
        md = open(os.path.join(t, "out", "r82_switch_summary.md")).read()
        chk("follows B after switch" in md and "## mode rt" in md, "markdown summary")
        shutil.rmtree(os.path.join(t, "ret", "r82_rt_swx_sbs_01_p1_mq2fq"))
        chk(run(ns) == 1, "15 arms must exit 1")
        # not separable: A and B identical
        sameA = os.path.join(t, "ret", control_ids("dc", "eee_02", "fq2mq")[0], "pos_trace_all.csv.gz")
        shutil.copy(sameA, os.path.join(t, "ret", control_ids("dc", "eee_02", "fq2mq")[1], "pos_trace_all.csv.gz"))
        run(ns)
        chk({r["arm"]: r for r in csv.DictReader(open(os.path.join(t, "out", "r82_switch_pairs.csv")))}["r82_dc_swx_eee_02_p1_fq2mq"]["verdict"] == "not separable", "not separable")
        # a switch arm that departs from A before the onset is noted
        xyz = FQ.copy(); xyz[100:, 0] += 0.2
        xy, yw = mk(xyz, 0.4); AF._write_trace(os.path.join(t, "ret", "r82_dc_swx_eee_02_p1_fq2mq", "pos_trace_all.csv.gz"), ids, xy, yw)
        AF._write_trace(os.path.join(t, "ret", control_ids("dc", "eee_02", "fq2mq")[0], "pos_trace_all.csv.gz"), ids, *mk(FQ, -1.3))
        run(ns)
        r = {r["arm"]: r for r in csv.DictReader(open(os.path.join(t, "out", "r82_switch_pairs.csv")))}["r82_dc_swx_eee_02_p1_fq2mq"]
        chk("before the onset" in r["note"], "pre-onset departure must be noted: %r" % r["note"])
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r82_switch_reduce", ("FAIL " + "; ".join(fails)) if fails else
          "PASS (follows B / follows A / neither / not separable on synthetic arms started with other headings and positions; control naming dc and rt; onset from the phase summary; "
          "collapse flags by the min rule and per control; pre-onset departure noted; a count other than 16 exits 1, checked by deleting one arm)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--controls", action="append", default=[]); ap.add_argument("--phase-summary", action="append", default=[])
    ap.add_argument("--state-ref-root", action="append", default=[]); ap.add_argument("--arm-table"); ap.add_argument("--onsets"); ap.add_argument("--thr-mm", type=float, default=50.0)
    ap.add_argument("--out-dir", default="."); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.runs and a.phase_summary):
        sys.exit("--runs and --phase-summary are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
