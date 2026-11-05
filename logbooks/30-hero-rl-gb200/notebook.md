---
title: "Hero RL qualification on 48 GB200 GPUs"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/30"
related_issues: ["https://github.com/marin-community/marin/issues/10388", "https://github.com/marin-community/marin-logbooks/issues/2"]
research_branch: "codex/30-hero-rl-gb200"
marin_base_revision: "f880fca07a7dda6cb51e3bdbffcd407cb5dba457"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: complete
---

# Hero RL qualification on 48 GB200 GPUs

Hero passed fixed-weight numerical qualification and four applied RL updates at a 4,096-token request cap.
This archive preserves the experiment and failed alternatives. [Selection](selection.md) separates reusable
fixes from historical qualification code. Learning quality, optimizer restore and production adoption remain
unqualified.

## Question and decision criteria

Could the provisional step-64 SFT initializer generate with captured graphs, match captured-route trainer
scoring, and complete four applied updates with finite loss and gradient norm, router replay and weight publication within 48 GB200
GPUs?

Acceptance required at least 50,000 response-token comparisons, strictly fewer than 0.1% trainer/serving
probability ratios outside `[0.8, 1.2]`, actual graph capture and completed PIECEWISE execution on all twelve
serving ranks, then four applied updates with zero skips, finite loss/gradient norm/reward, replay hit
fraction 1, PPO clip fraction below 0.01, and four completed weight publications. A global-step counter or
successful initial publication alone was insufficient.

## Baseline and exact inputs

- Historical runtime pins: numerical run `3c4466634e4b657ea1219e021fab55cfbb5550e2` (tree
  `19453e1a84d966799af26257a53a9eefc187ebd5`); successful updates `899b04436748b22da9d1e8a245bc1bdbe7556593`
  (tree `54b620e8df50d103586d0123e2e20914156c72b5`). These are ancestors of the archival base. Later
  source/attestation refreshes have no new GPU qualification.
- Initializer: `models/hero-535b-rl-initializer@2026.10.22.64:f3bc87b5`, completed HF export version
  `2026.11.03.49`; export-manifest SHA256 `d6bfdb32b7f061ae6abc6af7c289522f3103ab1b9b772110aa451c3b757a6f25`.
  This is a provisional initializer from [the SFT
  investigation](https://github.com/marin-community/marin-logbooks/issues/2). Conversion source
  `9e7461971e6fa64d1bd28b31f3d34587038f41e5` preserved BF16 ordinary weights and FP32 effective router bias;
  metadata checks covered 193 shards, 56,504 tensors and 1,070,954,250,240 payload bytes, without a payload
  rehash.
- Architecture: executed 16K model config `4d97008b7e7626fe9eb517c196988741b48cd6a6`, QK multiplier 1.48,
  sliding window 2,048, capacity factor 1.15. Qualification used a 4,096-token request cap: 1,024 prompt plus
  3,072 generated tokens; physical recomputation length 4,224 does not enlarge served context.
- Tokenizer: `marin-community/marin-tokenizer@a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`, fingerprint
  `0366e934c6bd0d28e35023d3d2e447f276d0b65040e934db79c566b476800cec`; EOT 128009. GSM8K
  `openai/gsm8k@e53f048`, first 1,024 train and 128 test rows, explicit `#### number` answer contract; seed
  42.
- Hardware: twelve four-GPU GB200 trays, 36 learner GPUs at TP1/PP12/CP1/EP3/ETP1 and twelve separate serving
  ranks at TP1/PP1/DP12/EP12. Each tray requested 128 CPUs, 940 GiB host memory and 4 TB disk, interactive
  priority, zero automatic failure/preemption retries and a six-hour GPU-child deadline.
- Updates: 15 prompt groups × eight responses, microbatch one; full-parameter MuonH/AdamH/Adam route at
  `1e-6`, CPU momentum and activation offload, frozen query-balancing bias, captured router replay, NCCL
  broadcast/layerwise reload. Compilation remained enabled; eager mode was not used.

## Entries

### 2026-11-04 UTC — Separate staging and startup memory from numerical fit

The export contained model metadata and weights but no tokenizer files. Requiring bundled tokenizer files
failed before numerical qualification. Model staging was separated from independently pinned tokenizer staging
and exercised by the CPU prerequisite. This established metadata and tokenizer loadability without downloading the weight shards.

An early attempt then suffered container OOM during serving initialization. RunAI 0.16.1's unlimited
distributed-streaming buffers could collectively retain roughly the full 997.4 GiB model on a 940 GiB tray.
The experiment capped buffers at 16 GiB per serving rank with four loader threads. Ray's advertised heap was
also corrected from 940 to 844 GiB alongside a 96 GiB object store. The failed attempt lacked exact peak
telemetry, so this remains a plausible startup cause; Ray's scheduler budget does not limit actor RSS.

### 2026-11-04 UTC — Qualify ShortConv PIECEWISE support before the full model

The pinned vLLM ShortConv guard forced graph mode NONE despite compilation being configured. A launch-only
graph flag could not establish capture. An experimental PIECEWISE overlay instead registered ShortConv as a
split boundary and zeroed padded/null rows; full graph capture remained rejected. Capture sizes were `[1, 2,
4, 8, 16]`, with FLASH_ATTN, batch invariance and returned routes enabled.

The kernel/config gate passed seven cases, including four real FP32/BF16 × FX/Inductor capture/replay cases. A
tiny two-layer V2 engine recorded 54 captures and 108 replays with exact output equality. The matched-route
gate passed DP1 and DP4, including idle peers; DP4 recorded 54 captures per rank and 234/243/243/243 replays.
Its observed expert pairs were always `(1, 3)` and logprobs uniform (`-log(32)`), so this was mechanical
coverage of routing transport, not meaningful request-sensitive routing or full-Hero numerical evidence.

The exact ARM wheel was attested independently of uv 0.13.5's possibly empty `direct_url.json` archive hash.
Staging verified all 5,056 runtime files, seventeen native libraries and three Python preimages, then
atomically published a private package copy. The shared package cache was not patched. Exact source preimages,
payload hashes, wheel identity and dependency pins remain in the historical [overlay manifest at the qualified
source](https://github.com/marin-community/marin-logbooks/blob/3c4466634e4b657ea1219e021fab55cfbb5550e2/experiments/post_training/hero_rl/piecewise_payload/manifest.json).
The qualified manifest SHA256 was `266cb34f775746ac050b13f4dcce38a615ae1718b450c546fba4ca4c06dfca8d`, wheel
SHA256 `8a3ba6d09d43c779d67775ee9be0e54febcad724bce5192021b005407df0fdf8`, patch SHA256
`1b377716b4586cc8ad5e4c607048aa6bb39ce159893da847cf6514ba9aae1896` against vLLM
`39e62869693c46402b1a95fde4fc55ca1aab9ae1`. Torch was `2.13.0+cu132`, torchvision `0.28.0+cu132`, transformers
`5.16.1`.

### 2026-11-04 UTC — Full fixed-weight qualification, version 2026.11.04.33

At runtime `3c4466634e4b657ea1219e021fab55cfbb5550e2`, 128 prompts × eight samples produced 1,024 independent
responses and 109,519 qualified response tokens. Thirty per-token trainer/serving probability ratios fell
outside `[0.8, 1.2]`: 0.0273925%, below 0.1%. Source, starting step zero, token and route checks passed. All
twelve serving ranks captured graphs and recorded 3,905 completed native PIECEWISE dispatch observations in
total. NONE dispatches also occurred; these tables are not direct CUDA replay counters.

A report-only correction recognized the canonical model URI in the runtime archive's `checkpoint_path`; the
staged local metadata directory was not the model source. It retained the exact-source guard and all numerical
thresholds. The runtime and archived outputs were unchanged. The probe hash was
`f6cc736e7a62f6c5eccbd52a2b0602a2fb4d802c590ea7100ee8ffe4478a62eb`; FineStore archive commit
`46a47b2f53b2498f9ca9d36f23ff170f`, sequence 9.

### 2026-11-04 UTC — Update attempts .34 and .35 reject incomplete memory fixes

Version `2026.11.04.34` used the same `3c4466634e` runtime. It was OOMKilled inside optimizer offload after
the first optimizer call returned, before an applied-update metric or post-update publication was verified.
Worker RSS grew from 447 to 859.7 GiB; zero applied updates were qualified. Flushing cached blocks alone could
not remove two simultaneously live master/momentum banks.

A bounded same-size native allocator probe showed default rounding inflated two live banks from 72 to 128 MiB;
cache flushing retained that inflation. Setting `pinned_max_round_threshold_mb:1` reduced live bytes and
preserved values in that probe. Version `2026.11.04.35` at `fb306d54f76446a77ce18dddf4bb8f12b3e34b5c` then
failed in `torch.autograd.graph.save_on_cpu` during the first PPO forward pass with a CUDA invalid-argument
copy error. Initial publication had succeeded, but no update was applied. This failure was not a verified OOM.

### 2026-11-05 UTC — Heterogeneous activation-copy probe selects the cache cap

The narrow probe missed unequal exact-sized blocks sharing a power-of-two cache bucket. A managed GB200 probe
reproduced a 48 MiB `save_on_cpu` copy reusing a primed 36 MiB block under threshold-only configuration. The
original full-run failing sequence length/stride was not recorded; the probe establishes the mechanism without
asserting identical original inputs.

Baseline, paired threshold/cap, and cache-cap-only cases passed real MCore checkpoint/backward,
max-sequence/transposed/offset-slice gradients and live Muon momentum roundtrips. Cache-cap-only retained 72
MiB rather than 128 MiB for the two-bank test and avoided unsafe large-block reuse. Selected:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_max_cached_size_mb:1`. Torch git identity was
`cf30153c4c131c8164ee7798e5022d810682e2cb`; probe SHA256
`6b0fa391888b54ad39010907979f4b8e049f95560742d0b71e74f5711f4a968e`. Repeated large host allocation/free costs
still required full-run measurement.

### 2026-11-05 UTC — Four-update qualification, version 2026.11.05.1

At runtime `899b04436748b22da9d1e8a245bc1bdbe7556593`, all twelve GPU tasks succeeded with zero
failures/preemptions. Four steps each applied one update, with zero skips, finite loss/gradient norm/reward
and router replay hit fraction 1. PPO clip fractions were 0.00016173, 0.00016300, 0.00009259 and zero. Publications
1–4 completed in approximately 233/233/233/235 seconds, with all twelve receivers resuming under corresponding
lease labels 2–5. Every serving rank recorded capture completion and positive completed PIECEWISE execution
tables; total observations were 15,657, not direct replay counts. The largest cgroup peak in saved bounded
windows was 536.9 GiB against 940 GiB; this is not a guaranteed global lifetime maximum. Optimizer offload
took 14.1–14.6 seconds per update. The 128-row evaluation scores were 0.4140625 at step zero and 0.390625 at
step four; this small panel establishes no quality gain.

### 2026-11-05 UTC — Contribution review narrows the backend change

The draft's broader multiprocessing placement claim was not retained for the backend-API contribution. Keep
the existing node-local constraints there; archive the experimental packing code and unsupported
generalization. The exact EP12 qualification above remains a result for its historical runtime, without
establishing general multi-node support for the selected contribution.

## Final decision and reproduction boundary

Preserve the experimental runtime, one-off probes and failed alternatives here; promote only the five
independently useful groups in [selection.md](selection.md). The original contribution was [Marin
#10948](https://github.com/marin-community/marin/pull/10948). Optimizer restore, post-update tensor checksums,
longer context, H100 qualification, learning-quality improvement and production adoption remain unqualified.
The GPU work is complete; this archive authorizes no new run.

The retained [launcher](../../experiments/post_training/hero_rl/launch.py) and
[recipe](../../docs/recipes/hero-rl.md) contain the full command/configuration. For the historical numerical
command use source `3c4466634e`, stage `numerics`, version `2026.11.04.33`, `--hardware gb200-48 --priority
interactive --probe-prompts 128`; for updates use source `899b044367`, stage `updates`, version
`2026.11.05.1`, with the same initializer version/digest above. Commands require the completed model URI and
authorized storage access. Use full source pins when inspecting history; the archive head is not the measured
runtime.

The retained [ShortConv gate](../../scratch/hero_shortconv_validation/submit.py),
[matched-route gate](../../scratch/hero_shortconv_route_validation/submit.py), [same-size allocator
probe](../../scratch/hero_host_allocator_validation/probe.py), [heterogeneous activation-copy
probe](../../scratch/hero_save_on_cpu_validation/probe.py) and [overlay
patch](../../scratch/hero-shortconv-piecewise-39e-candidate.patch) carry the supporting implementations.
[support-source-hashes.json](support-source-hashes.json) records their frozen source identities. Their submit
wrappers require separate compute authorization; reading this archive is not a launch instruction.

Final saved evidence digests (SHA256; raw logs and restricted storage locations are excluded from this public
notebook):

| Receipt | SHA256 |
| --- | --- |
| Numerical report | `db7630e1390aff7f2a3c553dc19fbb72564235e46dbb2b2fc54c67c05bbf62a1` |
| Four-update report | `e5e174054e9438aff13c556d34df17423afaeba2a08f98323db290e296b4aa76` |
| All-rank graph/publication report | `0adbc03796880316086cc4ae52edc1e374d10a69c75d071685fa93beba721f5b` |
| Independent final audit | `19d28b6f8272aa6f72cdf52a8c9e5012b5c1bff7f56ff3fbe3b4884e14fff40d` |
| Heterogeneous-copy verification | `c46a0054b13e6466c105397831bf5d5dcff6a2b65f31c7b86b877505806f4e55` |
