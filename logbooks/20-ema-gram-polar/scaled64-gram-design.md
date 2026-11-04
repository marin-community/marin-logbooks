# Test one large Gram per shard on 64 devices

The next diagnostic compares `matrix` and `matrix_grouped` on 64 JAX v4 devices
(v4-128, sixteen hosts, an 8 by 8 data/model mesh). The working training baseline
remains `matrix` with the default eigensolver. The completed
[32-device trial](scaled-gram-completion-note.md) found lower grouped full
synthetic-update means on all eight processes, but mixed fixed-state results
and no established device-efficiency gain. It has finished; this is a distinct
trial with a new task identity.

The existing grouped implementation concatenates three banks of eleven full
4096-square Gram matrices before assigning whole matrices to devices. Increasing
from 32 to 64 devices reduces the maximum number of large matrices assigned to
one device from two to one. Independent-bank `matrix` assigns at most three
across those banks. These source assignments motivate a latency test; they do
not establish runtime kernel counts or speedup. The 64-device trial uses twice
the devices of the completed 32-device trial and four times the earlier
16-device baseline. Report latency and device count together, including the
local mean multiplied by device count as a cost proxy. Cross-trial local times
are unpaired and do not establish end-to-end cost or training throughput.

Reuse [scaled_gram_check.py](scaled_gram_check.py) byte for byte, with the same
[four source-file identities](scaled-gram-manifest.json). The original regional
checkpoint supplies seven weight groups, each with momentum, left Gram and right
Gram arrays (21 arrays total). Three Gram arrays each hold eleven 4096-square
matrices. Preserve full Gram
geometry, every-update refresh, source precision and the default solver.
The diagnostic computes directions from saved state and then from a synthetic
gradient with state refresh. It applies no direction to a model and restores no
full trainer. Each case runs `matrix` before `matrix_grouped`, separates the
first compiled call, and retains all three warmed calls. Timers end at local
output completion; host entry is not aligned. No calibration or barrier phase
is added. The [original design](scaled-gram-design.md) specifies the fixtures
and numerical checks.

The numerical sources are unchanged, so reuse the finished online logging,
CPU8 and CPU16 checks and the 17 optimizer regression tests. CPU16's full-update
direction tolerance failures remain recorded. The threshold is diagnostic and
alone does not establish training degradation. Fresh execution-host logging
checks and worker runs must initialize online W&B before numerical kernels.
Require all sixteen successful processes from the same attempt, exact source
and checkpoint identities, finished online runs, finite immutable inputs and
three calls with bitwise-identical outputs within each case, layout and process.
Verify replica hashes and statistics before deduplicating 448 direction records
(seven weight groups times 64 device slices) or 469 including the 21 state arrays
counted once. A record identifies one output leaf and its global array slice. Reuse the existing
64-device static summary validation with its unchanged source hashes.

[submit_scaled64_gram.py](submit_scaled64_gram.py) uses the new job name
`ema-gram-20-scaled64-gram-v4-128-20261104`, regional report prefix
`gs://marin-us-central2/ema-gram-20-scaled64-gram/2026.11.04.01`, and guard
`scratch/20261104-scaled64-selection.json`. The completed 32-device launch guard
is preserved. The new launcher verifies the published HEAD and existing source
checks, then refreshes capacity and accepts only an available ready v4-128 slice
in us-central2 with sixteen healthy, idle hosts. It submits nothing when none
exists. Availability is advisory and does not reserve a slice or promise a
start time. After acceptance, preserve the single job and one `iris job logs --follow`
process even if queued; do not cancel, resubmit or start a duplicate monitor.

Each host requests 120 CPU, 256 GiB RAM and 50 GiB disk at batch priority.
Default preemption and retry policy remain; no queue, task, training or wrapper
wall-clock limit is set. Large checkpoints and reports stay regional. No other
allocation is changed. TPU remains the default, and other suitable hardware
remains authorized for future interventions. No GPU job is prepared.
