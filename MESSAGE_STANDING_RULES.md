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
8. When raw results risk exceeding the single-archive size limit, first use
   lossless per-file compression (prefer `.zst` for large CSV/text files) and
   place the compressed files inside the one required zip. Include a raw-file
   manifest with each run ID, original/compressed filename, original byte and
   row counts where applicable, and the SHA-256 of the uncompressed content.
   Compression must not round, truncate, resample, aggregate, or otherwise
   alter the evidence; retain round-trip-safe floating-point precision.
9. If lossless compression is insufficient, omit only demonstrably redundant
   representations such as summaries reproducible from retained raw files,
   duplicate logs, copied binaries, build intermediates, or identical campaign
   baseline artifacts. Record every omission and identify the retained source
   containing the equivalent data. Never reduce required component-wise state,
   correction, covariance, gating, process-noise, or per-iteration evidence to
   norms or summary statistics without explicit requester approval. If the
   archive still cannot fit, report the exact limit and projected size before
   making further omissions or splitting the return.
10. Campaign instructions must name the exact rows and columns required from
    every diagnostic source. When only a subset is needed, construct a compact
    lossless extraction and do not return the wider source file. The extraction
    must retain run/scan/iteration keys and enough metadata to reproduce its
    selection; the raw-file manifest records the omitted source columns and
    selection predicate. Do not include a large diagnostic file merely because
    it already exists when the requested analysis uses only a small subset.
