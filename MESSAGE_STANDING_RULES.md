# Standing rules for coding-agent messages and returns

1. The requesting agent makes the source edits, prepares the cumulative diff,
   and writes the coding-agent instructions. The coding agent builds, runs
   permanent tests, executes requested real-data campaigns, and may repair
   genuine compile/runtime defects while documenting every repair.
2. A coding-agent return must include the requested raw diagnostic evidence,
   resolved configurations, engagement evidence, build/test logs, final
   codebase, and cumulative diff in exactly one uniquely named zip.
3. The coding-agent report is limited to implementation status: patch
   application, compilation/tests, runtime completion, errors, warnings,
   repairs, deviations, missing jobs/files, and exact changes it made. It may
   inventory the evidence but must not substitute its own scientific analysis.
4. Do not aggregate away or silently omit raw evidence. If a real package-size
   limit prevents returning it in the single archive, stop and report the exact
   blocker before choosing an omission.
5. The requesting agent independently audits the source and raw evidence and
   performs all estimator analysis and user-facing reporting. A returned zip
   automatically triggers this audit; no additional request is required.
6. Behavioral validation uses real dataset runs. Synthetic tests are permitted
   only for isolated mathematical identities.
7. The requesting agent does not compile or run tests locally; those are
   coding-agent responsibilities.
