---
issue: https://github.com/marin-community/marin-logbooks/issues/15
marin_base: 8577d258214dd0dcb992c04458ee4eb10496d4e5
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
---

# Dense optimizer scaling on Hero

Compare held-out Paloma macro loss and elapsed time for AdamW, Muon, AdamH, and MuonH on dense Grug models. Stop at 1e21 analytic model FLOPs per run. This ladder measures four recipes at shared shapes, rather than searching for each optimizer's compute-optimal architecture.

## 2026-10-26: Initial design and launch contract

User scope: “let's do the ladder through e21”; “dense is fine for this”. The parent estimate used nine rungs through 1e22; the selected experiment uses seven rungs through 1e21. No training results exist yet.

Use the original Harrier 2026.08.18 store and the Hero phased mixture, the Marin tokenizer (128256 vocabulary), 4096-token context, and seed 0 for both initialization and data. Disable simulated epoching so runs read the complete component datasets. Each optimizer at a given rung sees the same architecture, data order, and token count. Shape selection approximates 20 tokens per total parameter; budgets use the trainer's analytic forward/backward FLOPs including attention and the language-model head. FLOPs omit optimizer work, which is measured in elapsed time and throughput.

| Budget | Width | Layers | Batch | Steps | Tokens |
|---|---:|---:|---:|---:|---:|
| 1e+18 | 384 | 4 | 128 | 4963 | 2.602B |
| 3e+18 | 640 | 8 | 128 | 6540 | 3.429B |
| 1e+19 | 896 | 10 | 128 | 12539 | 6.574B |
| 3e+19 | 1280 | 14 | 128 | 18319 | 9.604B |
| 1e+20 | 1536 | 16 | 128 | 41956 | 21.997B |
| 3e+20 | 2048 | 24 | 128 | 56989 | 29.879B |
| 1e+21 | 2560 | 28 | 128 | 113219 | 59.359B |

The 28 selected runs total about 5.776e21 model FLOPs. Eight additional 1e18 calibration runs add 8e18 FLOPs, plus eight bounded 20-step smoke runs. Concurrency is capped at eight single-node H100x8 runs (64 GPUs), on cw-us-east-02a at batch priority. The four top-budget runs use only 32 GPUs together, so the aggregate ideal-utilization estimate is optimistic: 1e21 FLOPs on eight H100s takes about 4.8–7.2 training days at 200–300 TFLOP/s/GPU, before setup and overhead. Throughput gates will replace this projection with measured ETAs. This cluster is local to the Hero store. A read-only capacity check found eight free whole nodes; this is a dated observation, not a reservation.

The CPU coordinator first runs 20-step probes at the smallest and largest shapes for all four optimizers. Each probe compresses warmup/decay into its 20-step schedule to exercise the full learning rate; its data phase remains the initial Hero phase. Success gates twelve complete 1e18 calibration runs (three LR multipliers per optimizer). Select each optimizer's lowest final Paloma macro loss, then run its six remaining budgets, largest first so the critical path starts early. A failed smoke, failed calibration, missing final metric, or non-finite metric stops the coordinator before larger work. Rerunning the same command reuses successful artifacts and resumes failed training from its latest complete checkpoint.

Pilot LR centers: AdamW 0.001, Muon 0.02, AdamH 0.01, MuonH sqrt(0.02*0.1). Adam fallback LR is 0.001. Sweep common matrix/fallback multipliers 0.3, 1, 3 independently per optimizer. Across rungs, transfer the selected multiplier with sqrt(batch/reference_batch)*(reference_tokens/tokens)^0.3. All recipes use beta1=0.9, beta2=0.95, epsilon=1e-8, 1% warmup, cosine decay to 10%, and no clipping. AdamW and Muon use decoupled weight decay 0.1; AdamH and MuonH use zero decay. The H recipes preserve matrix norms and use AdamH on the head. The Muon recipe uses AdamW on the head. This is a three-point LR calibration, not proof of individually optimal recipes; endpoint winners and transfer across scale remain caveats.

All runs use W&B marin-community/marin_moe, group hero-dense-opt-20261026. IDs encode campaign, budget, optimizer, and LR multiplier; probes append -smoke. Durable outputs are under the caller-owned regional users/<username>/grug/<run-id>/2026.10.26 paths. Rolling checkpoints use the existing region-local 14-day temporary checkpoint helper, retain one, and save hourly. Keep only final durable checkpoints. Probes save no checkpoints. Allocation-free model shape inspection gives 106M, 205M, 326M, 600M, 842M, 1.711B, and 2.807B parameters. Conservative optimizer-inclusive checkpoint estimates at 12 bytes/parameter range from 1.28 GB to 33.69 GB. Four final checkpoints per rung total at most about 317 GB; one rolling checkpoint per run adds at most another 317 GB. The eight extra calibration finals add about 10.2 GB. These estimates omit small serialization overhead and conservatively price Muon momentum as Adam moments. No checkpoint source is inherited from another experiment.

Coordinator command after ordinary Iris bundle submission:

```bash
python -m experiments.grug.fast_track.dense_ladder --campaign hero-dense-opt-20261026 --version 2026.10.26 --max-concurrent 8 --run
```

Print the plan with the same command without --run. The checked-in plan.json records the derived recipes. The published source revision and actual job identity will be recorded after source publication and submission. No dirty training source is authorized. The user is the DRI; this chat owns monitoring through a 15-minute heartbeat until verified completion or an actionable failure.

Local validation: six numerical optimizer tests pass, including independent AdamW and AdamH references, per-layer norm preservation, and stacked Muon parity. File-scoped repository lint passes. Full affected-runner and integration review are in progress. Store metadata reports success and sampled cell ledgers and validation caches are complete; this does not guarantee every future data read succeeds.

## 2026-10-26: Source publication and submission

Published source: `4dff62de22d525417dcc56abb0516a20e8157a59`, with no uncommitted source changes. Iris bundled 16.0 MB of workspace source. The coordinator `/dlwh/hero-dense-opt-20261026-coord` was submitted at 16:55:48 America/Los_Angeles through the Marin hub to `cw-us-east-02a`. This is submission evidence, not training progress or completion.

The local affected runner passed its selected six optimizer tests and 990 isolated Harbor checks selected by the archive workflow removal. The three newly added budget-contract cases also passed separately. File-scoped required lint passes. Independent allocation-free probes built both endpoint models and all four optimizer state trees, and smoke/full artifact fingerprints and data wiring passed. The test runner's unrelated lockfile rewrite was reverted before source bundling.

The run keeps the original dataset, topology, seed, and optimizer choices from the plan. The temporary checkpoint helper's actual default is 14 days (corrected above); it retains one checkpoint. Top-rung optimizer-inclusive state is estimated at 33.69 GB each for AdamW/AdamH and 25.09 GB each for Muon/MuonH. These are shape estimates, not observed checkpoint bytes.

The 15-minute heartbeat `watch-dense-hero-optimizer-ladder` owns monitoring in the originating chat. It reports phase changes, failure, or required action and stays quiet on unchanged healthy/pending status. Final artifacts, loss comparisons, and throughput are still unobserved.

Scrubbed submission command (credentials inherited through Iris secret delivery):

```bash
iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --target-cluster cw-us-east-02a --priority batch \
  --cpu 2 --memory 8GB --disk 32GB --timeout 0 \
  --job-name hero-dense-opt-20261026-coord -- \
  python -m experiments.grug.fast_track.dense_ladder \
  --campaign hero-dense-opt-20261026 --version 2026.10.26 --max-concurrent 8 --run
```

## 2026-10-26: Startup rendezvous collision

The first probes failed before model training. Multiple sibling gangs resolved the same default `jax_coordinator-attempt-0` endpoint; JAX reported “unexpectedly tried to connect with a different incarnation.” Iris namespaces endpoints to the root job, so distinct child jobs need distinct coordinator endpoint names. The experiment tree was cancelled; no calibration or full ladder runs were started.

Use the existing `initialize_jax(endpoint_name=...)` API with the unique trainer ID before the ordinary trainer initialization. The latter is idempotent. This changes discovery isolation and preserves the recipe. A named port alone would not isolate the shared endpoint registry.

A separate startup observation showed CoreWeave's shared scratch default pointing at us-east-08a despite the us-east-02a compute placement. The retry explicitly sets `MARIN_TEMP_PREFIX=s3://marin-us-east-02a` and `JAX_COMPILATION_CACHE_DIR=s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache`. Parent job environment is inherited by children. The original Hero dataset stays in place; no data is copied.

The retry keeps campaign and artifact identities because there was no training progress or checkpoint state to fork, and uses a new coordinator job name ending in `-r2` to separate execution attempts. Successful artifacts would be reused; failed smoke artifacts are rerun by the executor. The revised source revision and job identity follow after publication.

## 2026-10-26: Retry submitted

Published retry source: `1d6cca120eeebfe2446bf690e8e8dbb6d4559422`. All nine focused numerical/budget tests pass after the rendezvous fix, and required lint passes. The clean-tree 16.0 MB source bundle was submitted as `/dlwh/hero-dense-opt-20261026-coord-r2` at 17:01:20 America/Los_Angeles. The first tree is terminal (failed/killed probes and a cancelled coordinator).

The retry uses the same command as the first submission with `--job-name hero-dense-opt-20261026-coord-r2` and these explicit environment arguments before `--`:

```text
-e MARIN_TEMP_PREFIX s3://marin-us-east-02a
-e JAX_COMPILATION_CACHE_DIR s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache
```

The existing 15-minute heartbeat now follows the retry coordinator. No training result or successful probe has yet been verified. The archive index PR is https://github.com/marin-community/marin-logbooks/pull/16; it remains a draft and is not merged.

## 2026-10-26: Endpoint gates passed; calibration started

At 17:09–17:10 America/Los_Angeles, Iris reports all eight 20-step endpoint probes succeeded. W&B independently reports all eight finished with final `_step=19` and finite training loss. This verifies the small and large shapes for all four optimizer state/update paths and clears the initial rendezvous failure. The probes have no evaluation or checkpoint contract and do not establish optimizer quality rankings.

The coordinator automatically started the first eight of twelve complete 1e18 LR calibration runs at 17:06:55–17:06:56. Live W&B summaries show training progress at steps 46–91, fresh timestamps, and finite losses. These runs are still warming up; no final Paloma scores or selected multipliers exist yet. The remaining four calibration jobs will start as slots free. All runs still use the published retry source `1d6cca120eeebfe2446bf690e8e8dbb6d4559422` and the original recipes. No resubmission or configuration change was made at this heartbeat.

The draft archive index PR remains open/draft with no feedback or checks. The next decision is the automatic minimum-final-Paloma-loss LR selection after all twelve calibrations succeed; the six larger budgets stay gated until then.
