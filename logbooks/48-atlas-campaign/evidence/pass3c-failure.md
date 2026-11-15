# Evidence: Open Reasoning pass-3c failure record (sanitized)

- **What this is**: sanitized failure record for the OR pass-3c chained-repair job — runner
  provenance, exception class, attempt identities, telemetry that exists locally, and what
  survived the crash.
- **Compiled at**: 2026-11-15 (UTC)
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-full-qc/or-pass3-v3c-runner-v1.py`
  (runner, 246 lines), [R] `.../atlas-full-qc/repair3-openreasoning-v3pass-c-submit.log`,
  [R] `.../atlas-full-qc/REPORT.md:2218-2223` ("Code14 complete; OR pass-3c stopped by
  coordinator audit", 13:15 UTC Nov 15), [R] `.../atlas-full-qc/or-truncation-probe-*.py|log`.
- **Access class**: code and metadata only. Code contains no secrets (constants are storage
  URIs and SHA-256 digests of review artifacts). No task text or model output is reproduced.

## 1. Runner provenance

- Runner (unpublished script, local file): [R]
  `/Users/benfeuer/Documents/experiments/active/atlas-full-qc/or-pass3-v3c-runner-v1.py`,
  SHA-256 `abf36edfa1c2fcc16d49f5aa777aec2e537d6459aa5ec260a6e753b6e4ce0db1` (10,733-byte family;
  v3/v3b/v3c variants differ only in constants per REPORT.md:2221 "fixed in ALL FIVE runner variants").
- Submit wrapper: `.../atlas-full-qc/submit-or-pass3-v3c-v1.py`, SHA-256
  `84b0683c27da6481c27040c99c32a32a3e62e4ad11365b0b2aa3844e3c860e56`.
- Purpose: pinned-V3 re-review of primary-invalid rows not REVIEWED in either pass-2 overlay
  (union chain). Inputs asserted by SHA before load: primary
  `shard-reviews/2026.11.13.3/96fc19f9b4b2e033c9b3/attempts/63f0f2bb…` SHA `ff3067bf…`;
  pass-2a `…-pass2/attempts/b8da952e…` SHA `cc7a3aac…`; pass-2b `…-pass2-rev/attempts/4fdac5b2…`
  SHA `b4c46e49…`. Output root
  `s3://…/repair-overlay/2026.11.14.1/96fc19f9b4b2e033c9b3-pass3v3c`; request cache
  `…/repair-request-cache/2026.11.14.1-v3c`; 16 batch workers; serving root
  `/benfeuer/atlas-full-qc-recovery-serving-v3-01`.
- Job: `/benfeuer/atlas-full-qc-repair3-openreasoning-v3pass-c-v1`, submitted 03:44:36 UTC Nov 15
  (iris resources cpu=4, memory=16G, disk=40G; priority band batch; controller
  http://127.0.0.1:10002; dashboard link in submit log).

## 2. The defect (tuple-unpack bug)

The coordinator's audit found the v3c runner "had the same tuple-unpack bug and crashed twice"
[R] REPORT.md:2221. The bug lives in the parallel execution loop near the end of `main()`
(reproduced verbatim from the runner [R]; these ~15 lines contain no secrets):

```python
    batches = list(contract_batches(primary.inputs))
    touched = [batch for batch in batches if any(task.id in pending_ids for task in batch.tasks)]
    repaired: dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
        for group in batched(touched, BATCH_WORKERS):
            for batch, result in zip(group, executor.map(run_batch, group), strict=True):
                repaired[batch.index] = result
        logger.info("repaired %d of %d touched batches", len(repaired), len(touched))
```

`executor.map` over a `group` yields results in order without exposing the batch identity;
`zip(group, executor.map(...), strict=True)` relies on the map being over exactly `group` —
where the unpacking/length contract breaks (exception text not captured in the local artifacts
read), the loop crashes. The fix (per the 13:15 UTC entry) corrected the unpack in all five
runner variants; the crash was then bypassed by GPU-free finalization from sealed receipts.

## 3. Exception text

- Local artifacts read (submit log, REPORT.md) do not include the raw traceback text.
- Recorded characterization [R] REPORT.md:2221: "the v3c runner had the same tuple-unpack bug
  and crashed twice" — i.e. two failed attempts, same deterministic defect.
- **TODO**: pull the actual exception line from the Iris job dashboard/log export if the
  notebook needs the verbatim traceback.

## 4. Two attempt identities

- Two crashes of `/benfeuer/atlas-full-qc-repair3-openreasoning-v3pass-c-v1` (per [R]
  REPORT.md:2221). Attempt-level receipts live in the job's attempt directories; the two
  attempt IDs were **not captured** in the local artifacts read — TODO.
- Context: pass-3 v3 and v3b variants preceded v3c (submit logs `submit-or-pass3-v3-v1.py`
  02:44, `-v3b-v1.py` 03:07, `-v3c-v1.py` 03:23 UTC); pass-2 fwd ~223/327 completes and rev
  ~101/327 at 05:30 UTC with a token-limit tail [R] REPORT.md:2097.

## 5. Telemetry

| Metric | Value | Status | Source |
| --- | --- | --- | --- |
| True state at audit time | 43 REVIEWED / 75 INVALID (of the pending target population) | found | [R] REPORT.md:2221 |
| Finalized (GPU-free, sealed receipts) | 42,544 REVIEWED / 74 INVALID (44 recovered pre-cancel) | found | [R] REPORT.md:2221 |
| Final artifact | `...pass3v3c/attempts/7e8f49cff…` SHA `0fea2bda…`, invalid-rows.json lists all 74 with reasons | found | [R] REPORT.md:2221 |
| Standing unreviewed | 42,544/42,618 = 0.174% (under the 1% waiver threshold) | found | [R] REPORT.md:2223 |
| Cached requests in v3c manifests | 778 entries | found | [R] REPORT.md:2222 |
| 710 requests / 8.97M tokens / 92.1% length-finish denominator / 656 tok/s aggregate / 239 s mean | **not found locally** | TODO | — |
| SM/tensor gauge semantics | not found for pass-3c; the fleet-level gauge semantics that DO exist: Finelog `telemetry_v1.vllm` gauges integrated over adjacent observed intervals without extrapolation, cumulative token counters differenced within complete process+label identity (see run-manifests.md §1) | partial | [R] REPORT.md:1298 |
| Truncation probe A (post-mortem qualification) | pinned V3, doubled budgets 16,384/32,768 on the 10 worst INVALID rows → 2/10 valid (20%) vs ~70% bar; tokens avg prompt 6,108 / completion 612 (reasoning 776); 16.849362 GPU-h inference-only | found | [R] REPORT.md:2249-2253 |
| Code14-era token receipt (adjacent lane) | 12,411,285 prompt / 1,629,030 completion / 1,199,817 reasoning over 1,381 cached requests; finish_reason=length 0 among cached | found | [R] atlas-oracle-filter-code/CODE14-reviewer-token-receipt…md |

Token-total caveat recorded by the worker itself: "request envelopes live in batch receipts;
usage in the request-cache manifests (778 entries) — extraction queued as a follow-up receipt,
no extrapolated claims" [R] REPORT.md:2222.

## 6. What survived the crash

- All sealed per-attempt repair receipts and the request cache (778 manifest entries) survived;
  finalization was rerun **GPU-free from sealed receipts** (`or-pass3-finalize-from-receipts-v1.py`)
  producing the final artifact and invalid-rows.json [R] REPORT.md:2221.
- The pass-3 job's own two failures cost no new serving work (finalization was CPU-only); the
  serving root `/benfeuer/atlas-full-qc-recovery-serving-v3-01` was RELEASED (V3 root 01) after
  finalize [R] REPORT.md:2221.
- Residual policy state: no OR inference until the ≤10-row truncation-qualification probe is
  measured; any re-pass = rolling queue with ≤32 in flight; finalization stands as
  accepted_incomplete under the pinned ≤1% waiver (74/42,618 = 0.173636%, receipt `3a5cc13f…`),
  with the 2 probe recoveries optionally superseding 2 waiver entries in a later versioned
  overlay [R] REPORT.md:2249-2253.
