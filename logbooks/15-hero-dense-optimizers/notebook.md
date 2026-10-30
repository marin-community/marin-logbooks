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

## 2026-10-26: Calibration preemptions and automatic recovery

At 17:57–17:59 America/Los_Angeles, two calibration runs are successful in Iris and finished in W&B: AdamW LR multiplier 0.3 and 3, each at final `_step=4962` (4963 updates). Their final Paloma macro losses are 4.3715143 and 3.9749632, respectively. This is a partial LR sweep, not a completed optimizer comparison or final selection.

Six initial calibration jobs were preempted by the scheduler. Iris reports zero application failures; preemption counts are one each for AdamW LR1, Muon LR0.3/LR1/LR3, and AdamH LR0.3, and two for AdamH LR1. Kueue diagnostics explicitly report eviction to accommodate another workload. Some retries are running; others wait behind scheduling gates. The coordinator has also dispatched AdamH LR3 and MuonH LR0.3 as slots freed. No manual resubmission, priority increase, cluster mutation, or recipe change was made.

The affected attempts had not reached their first hourly checkpoint. A live Muon LR1 retry (attempt 1) restarted from the beginning and reached update 1003 by 17:57:47 with finite losses. W&B refuses to log steps below the prior attempt's recorded step (4246), so its old summary and crashed status do not establish that the current retry is dead. Monitor actual attempt-aware Iris logs until the retry catches up. This lost work delays calibration; it is not evidence of an optimizer regression. The larger ladder remains gated until all twelve complete successfully.

## 2026-10-27: Calibration complete and larger ladder active

The saved monitoring state was last updated on October 26 at 19:31 America/Los_Angeles; the next live verification occurred on October 27 around 19:28. Runs continued under the coordinator during this observation gap. The heartbeat trigger carried an older scheduled timestamp; current state below comes from live Iris, W&B, and object metadata rather than that trigger time.

All twelve 1e18 calibration runs succeeded in Iris, finished in W&B at `_step=4962`, and have final checkpoint metadata at update 4963. The durable coordinator selection matches the independently read final summaries. The complete three-point sweep and run identities are in [calibration-results.json](calibration-results.json).

| Optimizer | Best tested LR multiplier | Final Paloma macro loss |
|---|---:|---:|
| AdamW | 3 | 3.9749632 |
| Muon | 3 | 3.9614902 |
| AdamH | 3 | 4.0136013 |
| MuonH | 1 | 4.1018953 |

Muon is 0.01347 loss lower than AdamW at this single small scale. The comparison has one seed, and the AdamW/Muon/AdamH winners hit the upper search boundary, so it does not establish statistically significant optimizer differences or a compute-optimal scaling ranking. The larger runs use the selected recipe transfer already specified in the launch contract; no new settings were introduced.

The coordinator automatically dispatched four 1e21 runs and four 3e20 runs at 2026-10-27 02:54:06 UTC (October 26 at 19:54 America/Los_Angeles), starting the larger budgets first. The remaining sixteen smaller-budget jobs have not been dispatched yet because these eight occupy the concurrency slots. None of the twenty-four larger-budget runs is complete.

At the current observation, Iris reports zero application failures on all eight larger jobs. Three have running GPU tasks (1e21 AdamW, 1e21 MuonH, 3e20 MuonH); five are behind scheduling gates. Preemption counts are 4,4,4,2 for the 1e21 AdamW/Muon/AdamH/MuonH jobs and 3,5,4,4 for the corresponding 3e20 jobs. Thus only 24 of the allowed 64 H100s are actually working on this campaign at this instant. No priority, allocation cap, or cluster change was made.

Fresh W&B and attempt-aware logs show finite losses and advancing steps on the three active tasks. The furthest 1e21 run, MuonH, is at about 15196/113219 updates (13.4%). Its instantaneous throughput is about 172k tokens/s. The 1e21 AdamW retry is around 3898 updates and the 3e20 MuonH retry around 3211. Other W&B summaries marked crashed are retained observations of preempted attempts, not current queued-task application failures. ETA remains dominated by variable GPU availability; no completion date is asserted.

The archive index PR remains open and draft with no feedback. Its twelve-hour monitoring window has expired; this heartbeat stops its PR arm while continuing the experiment monitor. The PR is not merged or marked ready.

## 2026-10-29: Shorter temporary checkpoint interval for the next launch

Change the dense optimizer ladder's temporary checkpoint interval from one hour
to 15 minutes for the next launch. Keep one rolling temporary checkpoint and the
permanent final export. The existing coordinator uses the original source
snapshot; this edit does not change its running or automatically retried tasks.

A replacement coordinator is authorized once all campaign GPU tasks are
preempted or waiting, with none running or initializing. Preserve the campaign
and artifact version, run IDs, checkpoint lineage, training recipes, batch
priority, and the eight-job/64-GPU cap. Verify that condition in live Iris and
Kubernetes state before cancelling the existing coordinator and submitting its
replacement from the updated source revision.

## 2026-10-29: Relaunch with 15-minute temporary saves

Relaunched the existing campaign from source
`5e46cfb93934c8c5e9a218f45117685887410d59` after all eight GPU tasks were
confirmed queued. Each retained checkpoint had metadata, both manifests, and
its data directory. The old coordinator and its task pods stopped before the
replacement was submitted.

The replacement reused the completed smoke and calibration artifacts and
dispatched the same eight selected large-budget runs. The campaign identifier
`hero-dense-opt-20261026` and artifact version `2026.10.26` remain fixed, as do
the W&B run IDs, checkpoint paths, training recipes, batch priority, and 64-GPU
cap. The replacement source configures temporary saves every 15 minutes. Its
GPU tasks are queued; checkpoint restore and the first saves at the new interval
remain unverified.

## 2026-10-30: First selected large-budget run complete

MuonH with LR multiplier 1 completed the 3e20-FLOP recipe: final Paloma macro
loss 3.0711474 and final training loss 2.0547531 after 56,989 updates
(29,878,648,832 tokens). Iris reports success and W&B reports finished at
zero-based step 56,988. The permanent checkpoint metadata records update
56,989 at 13:06:44 UTC with `is_temporary=false`; its data directory and both
manifests are present. This is the first completed larger-budget run, bringing
the selected ladder to 5/28 complete, including four selected calibrations.

The run used source `1d6cca120eeebfe2446bf690e8e8dbb6d4559422` before the
authorized checkpoint-cadence relaunch and source
`5e46cfb93934c8c5e9a218f45117685887410d59` afterward. The relaunch preserved
checkpoint lineage, campaign, artifact version, W&B IDs, and training recipes.
Temporary checkpoint pairs for all eight original 3e20/1e21 runs now verify the
15-minute setting at runtime: observed commit intervals were 902.0–917.1
seconds, with increasing update counts and complete metadata, manifests, and
data directories. Frozen pairs and the final result are in
[milestone-20261030.json](milestone-20261030.json).

The coordinator dispatched AdamW with the selected LR multiplier 3 at 1e20
FLOPs at 13:06:54 UTC as the slot became free. A later live check at 13:58 UTC
verified that job running with finite training loss and advancing W&B steps.
Seven other large-budget runs continue; fifteen lower-budget jobs still await
executor slots. Batch priority and the eight H100x8 job cap remain fixed.

Each recipe uses one seed and learning rates transferred from a three-point 1e18
sweep. For AdamW, Muon, and AdamH, the selected learning rate was the highest
tested. The other 3e20 recipes remain incomplete, so this result alone does not
rank the optimizers.
Scheduler waits and preemption recovery contribute to elapsed time; no
uninterrupted training-time or time-to-loss comparison is inferred from W&B
runtime. Continue verifying final loss, run state, and permanent checkpoint
metadata as each selected run finishes.

## 2026-10-30: AdamW completes the 3e20 budget

AdamW with LR multiplier 3 completed 56,989 updates (29,878,648,832 tokens) at
3e20 FLOPs. Final Paloma macro loss is 2.9459300 and training loss is
1.9284897. Iris reports success and W&B reports finished at zero-based step
56,988. The permanent checkpoint records update 56,989 at 17:56:41 UTC with
`is_temporary=false`; metadata, both manifests, and the data directory are
present. The selected 28-run experiment ladder is now 6/28 complete.

At this shared budget, AdamW's final Paloma macro loss is 0.1252174 lower than
MuonH's 3.0711474. Each recipe uses one seed and an LR multiplier transferred
from the three-point 1e18 sweep; AdamW selected the highest tested multiplier.
Muon and AdamH at 3e20 remain incomplete. This partial comparison does not
establish a four-optimizer ranking or a time-to-loss result.

The coordinator dispatched Muon with LR multiplier 3 at 1e20 FLOPs at
17:57:10 UTC. At the next live observation around 22:54 UTC, its W&B stream
showed finite loss and advancing steps. Fourteen jobs below 3e20 FLOPs still await
executor slots. The source, checkpoint lineage, training recipes, batch
priority, and eight-job/64-GPU cap remain fixed. The raw final metrics and
checkpoint metadata are preserved in
[adamw-3e20-20261030.json](adamw-3e20-20261030.json).
