# Evidence: Open Reasoning pass-3c failure record (sanitized)

- **What this is**: sanitized failure record for the OR pass-3c chained-repair job — runner
  provenance, the unpacking defect with a minimal reproducible example, attempt identities,
  telemetry that exists locally, population transitions, and what survived the crash.
- **Compiled at**: 2026-11-15 (UTC); corrected by cold-read round 7.
- **Source cut**: the recorded 13:51 UTC Nov 15 source cut is branch `atlas-env-audit` head
  `96fc0736fc95d8f21d11e1c03dfca108d568b220` (pushed via atlas-push). **Evidence cutoff**: facts
  recorded under the 15:30 UTC entries (probe-A results, `accepted_incomplete` waiver
  acceptance) POSTDATE this cut; the fix/finalization facts are recorded under a "13:15 UTC"
  REPORT heading but were still absent from the 13:55 notebook catch-up bound to the cut (see
  §1 note). Everything else predates the cut.
- **Sources**: [R] `/Users/benfeuer/Documents/experiments/active/atlas-full-qc/or-pass3-v3c-runner-v1.py`
  (runner, 246 lines, pre-fix), [R] `.../atlas-full-qc/repair3-openreasoning-v3pass-c-submit.log`,
  [R] `.../claim-audits/fullqc-pass3c.report.md` (independent claim audit, receipt cutoff
  12:31:09 UTC), [R] `.../atlas-full-qc/REPORT.md:2218-2223` ("Code14 complete; OR pass-3c
  stopped by coordinator audit"; entry heading 13:15 UTC Nov 15) and `REPORT.md:2249-2253`
  ("Probe A measured"; entry heading 15:30 UTC Nov 15), [R] `.../atlas-full-qc/or-truncation-probe-v1.py|submit.log`.
- **Access class**: code and metadata only. Code contains no secrets (constants are storage
  URIs and SHA-256 digests of review artifacts). No task text or model output is reproduced.

## 1. Dated state table (crash → repair → recovery → finalization → waiver)

All times UTC, 2026-11-15. Event types are labeled. "Observation/code cut" = the artifact the
event's state is read from.

| # | Event (type) | Time (UTC) | Observation / code cut | Source | Postdates 13:51 cut? |
| --- | --- | --- | --- | --- | --- |
| 1 | Submit script saved | 03:23:16 | file mtime of `submit-or-pass3-v3c-v1.py` (see §4) | [R] file mtime | no |
| 2 | Job submitted + accepted | 03:44:32–03:44:36 | iris submit log (resources cpu=4, memory=16G, disk=40G; priority band batch; controller `http://127.0.0.1:10002`) | [R] submit log | no |
| 3 | Crash 1 (process attempt 0, exception) | 10:18:02.861 | hub log line, quoted by the claim audit | [R] fullqc-pass3c.report.md | no |
| 4 | Iris auto-retry scheduled (attempt 1 starts) | 10:18:26.702 | Iris task events | [R] claim audit | no |
| 5 | Crash 2 (attempt 1, same exception) | 11:47:03.566 | hub log line | [R] claim audit | no |
| 6 | Auto-retry scheduled (attempt 2 running) | 11:47:26.523; job describe 12:28:38 "State: running … failures=2" | Iris task events / describe | [R] claim audit | no |
| 7 | Receipt cutoff / independent claim audit | 12:31:09 | 43 REVIEWED / 75 INVALID across 104 latest batch pointers, 710 distinct responses, **no final artifact marker** (368 objects) | [R] claim audit | no |
| 8 | Coordinator audit refutes the status framing; pass-3c CANCELLED; unpack bug fixed in ALL FIVE runner variants; GPU-free finalization from sealed receipts → 42,544 REVIEWED / 74 INVALID (44 recovered pre-cancel); artifact `…pass3v3c/attempts/7e8f49cff…` SHA `0fea2bda…`; serving root V3-01 RELEASED | recorded under a "13:15 UTC" REPORT heading | [R] REPORT.md:2218–2223 | **not established at the cut** — see note below |
| 9 | Waiver acceptance: finalization stands `accepted_incomplete` under the pinned ≤1% waiver (74/42,618 = 0.173636%, receipt `3a5cc13f…`) | 15:30 UTC entry | [R] REPORT.md:2253 | **yes** |
| 10 | Probe A result (truncation-qualification probe): 2/10 valid (20%) | 15:30 UTC entry | [R] REPORT.md:2249–2253 | **yes** |

**Note on event 8 vs the cut.** The notebook catch-up bound to the 13:51 cut (`96fc0736…`)
records the pass-3c repair as **PENDING** (open item: "the pass-3c output-assembly bug … is
unfixed; 75 INVALID openreasoning rows need a repaired pass"; no final artifact). The full-qc
REPORT carries the fix + finalization under a **13:15 UTC** heading, which by its label predates
the cut by 36 minutes — but the REPORT's heading times are editorial labels, the entry's write
time is not independently pinned, and the cut-bound notebook did not reflect the entry. The
honest reading: at the recorded cut the repair was pending per the notebook; the fix,
finalization and root release are attested only by the REPORT entry, whose timing relative to
the cut is ambiguous; **no commit pins the runner fix** (the runner is an unpublished local
script). The waiver acceptance (event 9) and probe-A result (event 10) are unambiguously
post-cut.

## 2. The defect (tuple-unpack bug) — mechanism and minimal example

The coordinator's audit found the v3c runner "had the same tuple-unpack bug and crashed twice"
[R] REPORT.md:2221. The mechanism is established by the **pre-fix** runner file (the local
`or-pass3-v3c-runner-v1.py`, SHA-256 `abf36edfa1c2fcc16d49f5aa777aec2e537d6459aa5ec260a6e753b6e4ce0db1`,
still contains the defect — the fixed variants are not in the local read set) plus the hub-log
exception line quoted in the claim audit [R]:

- 10:18:02.861 UTC and 11:47:03.566 UTC: `AttributeError: 'tuple' object has no attribute 'repaired_task_ids'`,
  each immediately after `repaired 104 of 104 touched batches`.
- The claim audit's code reading: the runner's `run_batch` returns `repair_batch(...)` and
  `repair_batch` returns the pair `(batch, result)`; the loop then zips the original `group`
  with those tuples, so each `result` bound in the loop is itself the `(batch, result)` tuple,
  which is stored in `repaired[batch.index]` and later treated as a `RepairBatchResult`.

The failing loop (verbatim from the pre-fix runner [R]; ~15 lines, no secrets):

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

Minimal failing example, composed from the pre-fix runner file (runner line 135 calls
`repair_batch`, whose contract per the claim audit returns the pair; lines 141–143 are the
defect; line 163 is the failing attribute access). Runnable standalone; no task content:

```python
from concurrent.futures import ThreadPoolExecutor
from itertools import batched

BATCH_WORKERS = 16

class Batch:  # stand-in for ContractBatch
    def __init__(self, index: int) -> None:
        self.index = index

def repair_batch(batch):  # real contract: returns the (batch, result) pair
    return batch, "result"

def run_batch(batch):     # runner line 134-135 (verbatim shape)
    return repair_batch(batch)

touched = [Batch(i) for i in range(3)]
repaired = {}
with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
    for group in batched(touched, BATCH_WORKERS):
        # runner line 142 (verbatim): executor.map already yields (batch, result) tuples,
        # so zip pairs each batch with a TUPLE, not with the result object.
        for batch, result in zip(group, executor.map(run_batch, group), strict=True):
            repaired[batch.index] = result  # stores the (batch, result) tuple
print(f"repaired {len(repaired)} of {len(touched)} touched batches")
for result in repaired.values():
    _ = result.repaired_task_ids  # runner line 163: AttributeError: 'tuple' object has no attribute 'repaired_task_ids'
```

- **Snippet provenance**: PRE-FIX — the local runner file retains the defect; the fixed five
  variants are not in the local read set.
- **Status of the causal explanation**: the mechanism above is *reported*: it is composed from
  the pre-fix code and the hub-log exception line quoted in the claim audit [R]. The raw
  traceback in a durable job-log export is **not** in the local artifact set read — label the
  causal explanation as reported until that log (or a sanitized reproducer receipt) is
  archived. The earlier claim that "the unpacking/length contract breaks" without a mechanism
  is superseded by this section.

## 3. Exception text

- Verbatim line, quoted as hub-log output in the claim audit [R]
  `.../claim-audits/fullqc-pass3c.report.md`: `AttributeError: 'tuple' object has no attribute
  'repaired_task_ids'`, at 10:18:02.861 and 11:47:03.566 UTC, each immediately after
  `repaired 104 of 104 touched batches`.
- Recorded characterization [R] REPORT.md:2221: "the v3c runner had the same tuple-unpack bug
  and crashed twice" — two failed attempts, same deterministic defect.
- A durable Iris job-log export containing the full traceback is not in the local read set —
  TODO if the verbatim multi-line traceback is needed.

## 4. Attempt identities and submission-time reconciliation

- Iris attempts are **indexed 0/1/2** (one task, automatic retries): attempt 0 crashed at
  10:18:02.861 UTC, attempt 1 crashed at 11:47:03.566 UTC, attempt 2 was running at the
  12:28:38 describe ("failures=2") and was CANCELLED under the coordinator's order. **Opaque
  attempt UUID/receipt identifiers are not present in the claim-audit or full-qc artifacts
  read** — TODO; the hub-log query keys on the job/task prefix, not attempt IDs.
- **03:23 vs 03:44:36 reconciled**: 03:23:16 UTC is the file mtime of the submit wrapper
  `submit-or-pass3-v3c-v1.py` (script saved), and 02:44:08 / 03:07:07 are the mtimes of the
  v3 / v3b wrappers — these are code-cut times, not submission events. The actual job
  submission/acceptance is **03:44:32–03:44:36 UTC** per the Iris submit log
  (`repair3-openreasoning-v3pass-c-submit.log`). Earlier text implying a 03:23 submission is
  corrected.
- Context: pass-3 v3 and v3b variants preceded v3c (submit wrappers saved 02:44:08 and
  03:07:07 UTC); pass-2 fwd ~223/327 completes and rev ~101/327 at 05:30 UTC with a
  token-limit tail [R] REPORT.md:2097.

## 5. Telemetry

| Metric | Value | Status | Source |
| --- | --- | --- | --- |
| True state at audit cutoff (12:31:09) | 43 REVIEWED / 75 INVALID (of the 118-row pending target) | found | [R] fullqc-pass3c.report.md |
| Finalized (GPU-free, sealed receipts) | 42,544 REVIEWED / 74 INVALID (44 recovered pre-cancel) | found | [R] REPORT.md:2221 |
| Final artifact | `...pass3v3c/attempts/7e8f49cff…` SHA `0fea2bda…`, invalid-rows.json lists all 74 with reasons | found | [R] REPORT.md:2221 |
| Unreviewed fraction | **74/42,618 ≈ 0.174% (0.173636%) unreviewed; 42,544/42,618 ≈ 99.826% is the REVIEWED fraction** (corrected: an earlier draft divided 42,544 by 42,618 and mislabeled the quotient) | found | [R] REPORT.md:2223, 2253 |
| 710 inference requests | 710 **distinct model-response IDs** recorded in the immutable attempt receipts at the 12:31:09 cutoff (254/236/220 request observations across rounds 1/2/3; 0 cache hits) | found | [R] claim audit |
| 778 cache entries | 778 **entries in the v3c request-cache manifests** counted at the 13:15 entry (`repair-request-cache/2026.11.14.1-v3c`) | found | [R] REPORT.md:2222 |
| 710 vs 778 reconciliation | requests vs cache entries are different counters at different cutoffs: the 710 counts distinct response IDs deduplicated across attempt receipts as of 12:31:09; the 778 counts keyed request envelopes in the cache manifest as of the 13:15 entry (attempt 2 kept generating between the two cutoffs, and cache writes/receipt flushes are not the same unit). The exact delta is **not reconciled by a local artifact** — unexplained remainder ~68 | partial | [R] both, above |
| Token totals | 8,967,189 completion tokens of which 8,849,442 (98.7%) reasoning; 4,317,586 prompt tokens; 654/710 finish_reason=length (92.1%); 56 stop | found | [R] claim audit |
| Serving telemetry (job-scoped) | 723 completed requests 08:44:05–12:29:09 UTC; generation_tokens_total delta 9,149,245 (~677.5 tok/s); mean e2e latency 239.0 s; queue mean 0.563 s; **zero vLLM preemptions** (no counter resets); at 12:29:35 "Avg generation throughput: 656.3 tokens/s" | found | [R] claim audit |
| SM/tensor gauges | job-scoped: 100% utilization gauges with ~33% SM / ~1.5% tensor active, KV occupancy mean 5.08% | found | [R] claim audit |
| Fleet-level gauge semantics | Finelog `telemetry_v1.vllm` gauges integrated over adjacent observed intervals without extrapolation, cumulative token counters differenced within complete process+label identity (see run-manifests.md §1) | partial | [R] REPORT.md:1298 |
| Attribution | the 710 requests / 8.97M tokens / 92.1% length-finish / 656 tok/s / 239 s / zero-preemption figures originate from the **local claim-audit report** [R] — they are job-scoped audit measurements, distinct from the fleet-level Finelog windows in run-manifests.md §1 | — | — |

Token-total caveat recorded by the worker itself: "request envelopes live in batch receipts;
usage in the request-cache manifests (778 entries) — extraction queued as a follow-up receipt,
no extrapolated claims" [R] REPORT.md:2222.

## 6. Population transitions

| Population | Identity | Count | Source |
| --- | --- | ---: | --- |
| Whole OR source population | `tasktrove-math_openreasoning-qa` (QA release 42,618) / twin release `laion__nemotron-gym-math-openmathreasoning-v2` | 42,618 | [R] FINAL_STATUS-v2.json |
| Pass-3c repair subset | pending = primary-invalid AND not REVIEWED in either pass-2 overlay | 118 | [R] runner header + claim audit |
| Subset state at 12:31:09 cutoff | within the 118: REVIEWED 43 / INVALID 75 | 43 / 75 | [R] claim audit |
| Recovered pre-cancel | "44 recovered pre-cancel" — **membership not stated locally** (whether these 44 are within the 118 subset or the wider repair population is not said in the read artifacts) | 44 | [R] REPORT.md:2221 |
| Final whole-population state | 42,544 REVIEWED / 74 INVALID; 42,544 + 74 = 42,618 (arithmetic consistent) | 42,544 / 74 | [R] REPORT.md:2221 |

The overlap between the 74 final INVALID rows and the 118-row subset's 75 invalid rows at the
audit cutoff is **not stated by any local source** (e.g. whether some previously-valid rows
re-derived as invalid during finalization) — unknown. The counts above are disjoint within
each named population; cross-population transitions beyond what is quoted are marked unknown.

## 7. Probe A (truncation-qualification) — rule, result, consequence

- **Predeclared acceptance rule**: the FULL-QC worker reports re-passing is gated on a
  reviewer strategy achieving the "coordinator's **~70% qualification bar**" of valid reviews
  on the probe rows [R] REPORT.md:2251. The numeric predeclaration itself was **not located in
  the local artifacts read** (the 13:15 order recorded in REPORT.md:2221 defines the ≤10-row
  probe requirement but no numeric bar in the read set) — cite the rule as reported by the
  worker [R]; the probe runner's own docstring records the qualification intent without the
  number [R] `or-truncation-probe-v1.py:1-9`.
- **Observed result** (15:30 UTC entry, post-cut): pinned V3 with doubled output budgets
  (16,384/32,768) on the 10 worst INVALID rows → **2/10 valid (20%)**; 8/10 still "Incomplete
  or truncated structured response" after 3 attempts each [R] REPORT.md:2251.
- **Token-field semantics — corrected/caveated**: the worker's line reads "avg prompt 6,108 /
  completion 612 (reasoning 776)" [R] REPORT.md:2252. In the API usage schema used by the
  campaign's receipts, `reasoning_tokens` is a sub-field of `completion_tokens_details`, so
  per-response reasoning ≤ completion; a same-denominator pair of means (612 completion vs 776
  reasoning over the same 2 responses) is therefore impossible. The local carrier does not
  define the denominators (the probe cache manifest is not mirrored locally); the two figures
  likely average over different populations (e.g. completion over the 2 cached successes vs
  reasoning over a wider attempt set) — **unresolved**; do not quote them as same-denominator
  means. TODO: pull the probeA cache manifest.
- **What the result permits**: 20% is far below the ~70% bar → variant A is **not qualified**;
  no full OR re-pass is authorized; NO variant B per standing order [R] REPORT.md:2251. The
  finalization therefore stands `accepted_incomplete` under the pinned ≤1% waiver; the 2 probe
  recoveries may supersede 2 waiver entries in an optional later versioned overlay
  (pending-optional, zero-GPU) [R] REPORT.md:2253. Probe cost: 16.849362 GPU-h
  (inference-only, created→finished).

## 8. What survived the crash

- All sealed per-attempt repair receipts and the request cache (778 manifest entries) survived;
  finalization was rerun **GPU-free from sealed receipts** (`or-pass3-finalize-from-receipts-v1.py`,
  SHA-256 `2a2978ad2c2e32ae3d266290f0398eafc9089623bc2d302ec8204564a6978ea6`) producing the
  final artifact and invalid-rows.json [R] REPORT.md:2221.
- The pass-3 job's two failures cost no new serving work in finalization (CPU-only); the
  serving root `/benfeuer/atlas-full-qc-recovery-serving-v3-01` was RELEASED (V3 root 01) after
  finalize [R] REPORT.md:2221.
- Residual policy state (post-cut): no OR inference until the ≤10-row truncation-qualification
  probe is measured (measured 15:30, §7); any re-pass = rolling queue with ≤32 in flight;
  finalization stands as `accepted_incomplete` under the pinned ≤1% waiver (74/42,618 =
  0.173636%, receipt `3a5cc13f…`), with the 2 probe recoveries optionally superseding 2 waiver
  entries in a later versioned overlay [R] REPORT.md:2249-2253.
