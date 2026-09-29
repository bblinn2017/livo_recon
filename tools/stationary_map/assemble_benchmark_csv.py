#!/usr/bin/env python3
"""Assemble the single return CSV. Every fact is parsed from a log/file, never typed.

  assemble_benchmark_csv.py --out R54_stationary_map_benchmark.csv --commit GITSHA
      --build-log build.log --ctest-log ctest_final.log [--ctest-log-pre ctest.log]
      [--cmake-cache build/CMakeCache.txt] [--mutation-log FILE ...]
      [--extra a.csv b.csv ...]

Columns: row_type,run_id,arm,family,metric,value,detail,git_commit,config_hash,command
--extra files are CSVs with (at least) row_type plus any of the other columns (missing ones
are blank); typical inputs: compare_backends outputs, benchmark summaries, S-2 / duplicate /
window-equivalence outputs, reference-surface stats. --mutation-log: text files each holding
the ctest output of one mutated build (detail = first 'FAILED'/'Failed' + CHECK line).
"""
import argparse, csv, os, re

FIELDS = ['row_type', 'run_id', 'arm', 'family', 'metric', 'value', 'detail',
          'git_commit', 'config_hash', 'command']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--build-log')
    ap.add_argument('--ctest-log')
    ap.add_argument('--ctest-log-pre')
    ap.add_argument('--cmake-cache')
    ap.add_argument('--mutation-log', nargs='*', default=[])
    ap.add_argument('--extra', nargs='*', default=[])
    a = ap.parse_args()
    rows = []

    def add(**kw):
        r = {k: '' for k in FIELDS}
        r['git_commit'] = a.commit
        r.update(kw)
        rows.append(r)

    flags = ''
    if a.cmake_cache:
        t = open(a.cmake_cache, errors='replace').read()
        bt = re.search(r'^CMAKE_BUILD_TYPE:\w+=(.*)$', t, re.M)
        fl = re.search(r'^CMAKE_CXX_FLAGS_RELEASE:\w+=(.*)$', t, re.M)
        flags = '%s,%s' % (bt.group(1) if bt else '?', fl.group(1) if fl else '?')
        add(row_type='build_check', metric='build_type_flags', value=flags, detail=a.cmake_cache)
    if a.build_log:
        t = open(a.build_log, errors='replace').read()
        errs = len(re.findall(r'\berror\b', t, re.I))
        warns = len(re.findall(r'warning:', t))
        add(row_type='build_check', metric='build_log', config_hash=flags,
            value='errors=%d' % errs, detail='warnings=%d; last line: %s'
            % (warns, t.strip().splitlines()[-1] if t.strip() else ''), command=os.path.basename(a.build_log))
    for path, tag in ((a.ctest_log_pre, 'ctest_pre'), (a.ctest_log, 'ctest_final')):
        if path:
            t = open(path, errors='replace').read()
            m = re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', t)
            add(row_type='test_check', arm=tag, metric='tests_passed',
                value=('%d/%s' % (int(m.group(3)) - int(m.group(2)), m.group(3))) if m else 'unparsed',
                detail=m.group(0) if m else '', command=os.path.basename(path))
    for path in a.mutation_log:
        t = open(path, errors='replace').read()
        chk = re.search(r'(tests/\S+:\d+:[^\n]*)', t)
        failed = re.search(r'(\d+) tests failed out of (\d+)', t)
        add(row_type='mutation_check', arm=os.path.splitext(os.path.basename(path))[0],
            metric='ctest_failures', value=failed.group(1) if failed else 'unparsed',
            detail=chk.group(1) if chk else '', command=os.path.basename(path))
    for path in a.extra:
        for r in csv.DictReader(open(path)):
            row = {k: r.get(k, '') for k in FIELDS}
            if not row['git_commit']:
                row['git_commit'] = a.commit
            if not row['command']:
                row['command'] = os.path.basename(path)
            rows.append(row)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print('wrote', len(rows), 'rows to', a.out)


if __name__ == '__main__':
    main()
