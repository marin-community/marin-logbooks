# Private raw FP8 microbatch preparation

**Prepared, unexecuted, not released.** This is a research diagnostic, not a Marin contribution or a native-model quantization implementation. No imports, tests, probe execution, maintained-source edits, jobs, network, archive writes or commits were performed. All TPU/job/monitor windows stay paused.

## Reuse and source

The predecessor is `preparation-artifacts/native-jax-core-fp8-preparation-r3-freeze.json`, SHA256 `f2136668230bf4c537d9ef1bb2c98d4cbf49b4d3b87742aac782d275313a91e2`. Its runner/worker/config hashes were checked against that freeze. R3's original microbatch stage remains unexecuted and byte-identical at its original path on source054. This preparation changes neither its history nor its claimed results.

Reuse the existing runner byte-for-byte as `native-jax-fp8-microbatch-preparation-probe.py` (SHA256 `2c9e7640edf9991dda223daaa51e94a9e60e8e328290fa831bb225852f449f28`). Derive the sibling worker from R3, retaining its baseline/raw functions and dispatcher. The only worker changes are the existing microbatch stage and exact module-provenance checks. No parallel harness is introduced; only `microbatch` mode is proposed for future release. Do not rerun baseline/raw modes as part of this preparation.

Target clean LOCAL release source `<execution-source-root>`, commit `ba58e83e3310163e8d674ea9e95a85094d236a7a`, tree `223c7ff9b639ff2966941d8d9ea762ec157cb501`. It differs from R3's pinned054 in exactly shared quantization.py, _src/fp8.py and test_fp8.py. The derivative config retargets those hashes/commit/tree, adds the actually used raw-batch reshaper pipeline.py to source pins, and points to the new worker. Other source pins are inherited unchanged. This local release is not public evidence.

Before execution, root must verify a PUBLIC source commit/tree and published research runner/worker/config bytes after the outstanding PR/archive work. If public source identity differs, prepare a new explicit config/freeze and review it; do not silently rewrite this preparation. No public source pin or publication assertion is supplied here.

## Minimal new microbatch control

Keep R3's raw mean-square objective, one operation, output-first weight `(4,8)` filled with `.125`, stored at `P(model,data)` and gathered to `P(model,None)`, inputs at `P(data,None)`, and two equal microbatches with amplitudes32 and1. CPU1 uses data1/model1 and microbatch2; CPU4 uses data2/model2 and microbatch4. All mesh axes are Explicit. Pass the now-supported requested `out_sharding=P(data,model)` into the operation itself.

Use nonunit input/kernel/gradient scales2,.5,4 and nonempty histories:

| State | Initial history | Next scale |
| --- | --- | --- |
| input | 896,448,224,112 | 2 |
| kernel | 224,112,56,28 | .5 |
| output gradient | 229376,114688,57344,28672 | 4 |

The old-history maxima equal E4M3max448 times input/kernel scales and E5M2max57344 times gradient scale. Therefore the existing delayed-scaling formula retains those scales. Current fixture quotients, outputs and cotangents are exactly representable; this is a static fixture justification, not a successful runtime result.

Use independent NumPy matrix multiplication and the analytic mean-square cotangent `2*output/output.size` for each actual microbatch separately. The oracle computes loss, ordinary weight gradient, and all six state arrays: roll each seed history left then overwrite index0 with the appropriate absolute maximum; retain the known nonunit scale. Reduce the two independent history arrays by maximum and keep the latest scales. Average only ordinary losses/weight gradients. This oracle does not call shared FP8 metadata helpers or a shared reducer.

First check direct full-batch evaluation and the existing wrapper's no-split path (`microbatch_size=full_batch_size`) against their own full-batch oracle, including all six arrays. Then check each independently placed microbatch against its own oracle. The shared `accumulate_gradients`/overwrite partition control from R3 is retained and must itself match the independent max-history/latest-scale oracle. It is a corroborating control, not the numerical reference.

Only then run real `microbatched_arrays` on the two microbatches. Require exact independent loss and ordinary weight-gradient equality and all physical-placement checks before classifying metadata. Record all six observed/expected arrays and the separate arithmetic-average-of-all-leaves prediction. On this unchanged source, input new-history entry16.5 instead of32 and reduced output-gradient maxima are hypotheses until observed. If raw state differs, persist the diagnostic and raise so the command exits unsuccessfully. Exceptions before that point remain harness/provenance/control failures, not evidence of metadata averaging.

Check each loss, weight gradient and metadata array's actual sharding against its expected NamedSharding, full device set, and addressable shard shapes. Input and original storage weight placement are also checked. CPU4 is needed for nontrivial physical data/model partitioning; CPU1 alone cannot establish it. No gather-to-replicated workaround is allowed. This bounded control checks ordinary weight gradients; it does not add input differentiation or native-model conversion.

Do not compare full-batch state to microbatch state: the mean-loss output cotangent is smaller on the full batch. Do not assume arbitrary full-batch FP8 gradients/trajectories equal microbatch ones. No dtype/model/backend/tolerance changes are permitted to make a control succeed.

## Future commands, each separately gated

These are proposed commands only. They are not authorization. Fresh independent review must inspect the unexecuted derivative code/config/recipe and source/publication pins. Root must then release CPU1 explicitly. Use a clean exact PUBLIC source checkout and separately published research files. `FP8_OUTPUT_ROOT` must be an explicitly selected fresh private research-output directory, created by root only after release; each runner subdirectory must not exist. Do not overwrite old receipts. `gtimeout` was located at `<wall-timeout-command>` on this host; use that existing tool to impose180s wall time plus5s forced termination. Timeout/termination without a completed receipt is a separate bounded-command failure.

```sh
JAX_NUM_CPU_DEVICES=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled <wall-timeout-command> --signal=TERM --kill-after=5s 180s uv run --offline --directory "$FP8_SOURCE_ROOT" python "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-probe.py" microbatch --source-root "$FP8_SOURCE_ROOT" --research-root "$FP8_RESEARCH_ROOT" --config "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-config.json" --devices 1 --output "$FP8_OUTPUT_ROOT/microbatch-cpu1" > "$FP8_OUTPUT_ROOT/microbatch-cpu1.log" 2>&1
```

Stop manually on the first meaningful failure, including an expected metadata mismatch. Preserve receipt/log and classify the completed controls. Do not retry, edit the fixture, continue to CPU4, or execute a candidate fix automatically. CPU4 needs a separate root release after reviewing CPU1; an expected negative control is still a stop boundary.

```sh
JAX_NUM_CPU_DEVICES=4 UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled <wall-timeout-command> --signal=TERM --kill-after=5s 180s uv run --offline --directory "$FP8_SOURCE_ROOT" python "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-probe.py" microbatch --source-root "$FP8_SOURCE_ROOT" --research-root "$FP8_RESEARCH_ROOT" --config "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-config.json" --devices 4 --output "$FP8_OUTPUT_ROOT/microbatch-cpu4" > "$FP8_OUTPUT_ROOT/microbatch-cpu4.log" 2>&1
```

The reused runner validates clean exact source and source/worker hashes before loading its worker; imports and device checks happen in the same process as controls. New provenance entries cover the actual FP8, quantization, grad_accum and pipeline modules, not only package roots. Existing Iris package import/resolver checks are retained solely to reject an attached publishing runtime; no Iris client/job/login/monitor operation is called. Offline variables cover uv/HF paths; no tokenizer/model/data baseline mode is executed. Network denial remains an execution boundary, not something an environment variable alone proves.

## Review and readiness

Only text/data inspection and hash checks were performed. No AST compilation, project imports, hooks, tests or diagnostics ran. Fresh independent review and ordinary research-artifact hooks remain pending; these must not become a runtime-success claim. Required future evidence is the exact public source and research pins, each actual command/exit, staged receipt, all prerequisite control outcomes, all six state arrays and placement records, and manual stop disposition. Root may later archive this historical probe in the research archive only. It is not a proposed Marin repository file or source fix. Native integration, BF16, stacked state, optimizer/checkpoint continuation, accelerator kernels/performance and canonical acceptance remain outside this control.
