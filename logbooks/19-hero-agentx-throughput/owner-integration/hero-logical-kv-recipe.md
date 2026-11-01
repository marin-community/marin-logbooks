# Confirmed Hero AgentX recipe

BF16 Hero-535B on sixteen GB200 GPUs produces **811.36246 output tokens/s total, 50.71015 per GPU**, with 4,545 scored successes and zero scored errors. [The full-hour record](../results/guarded-retention-agentx-c128-r1-comparison.json) retains both native validation and artifact receipts. The send window is 3,600 seconds; the pinned InferenceX rate uses 3,628.39070 seconds from the first scored request start to the last scored completion. All 393 source sessions remain eligible. C128 limits concurrent conversation trees to 128 across the deployment; each tree can issue several HTTP subrequests.

[The frozen manifest](../experiments/logical-kv-retention0-16-agentx-c128-r1.json) and [selection record](hero-selected-recipe.json) specify the recipe. The executed research revision is `76c4802e66705b7c9390207366b1acdea8964f4a`. The manifest SHA256 is `ecb6a83d0ecfe6a491f4eb4f3704dfc6e6b58181dcbe05bfaf5c41a51327f24a`.

| Setting | Selected value |
| --- | --- |
| Model | Step-144000 BF16 split export; manifest `b20fdd303eb6ff8993ae5b5d7add14d65787e70981c056fea6de9dea62acfae1` |
| Parallelism | TP1 / PP1 / DP16 / EP16; four native tasks, four GPUs each |
| Native vLLM source | `a0be4d10f1c45dd44458c68080e5818253de66e4`, [draft #110](https://github.com/marin-community/vllm/pull/110) |
| Private wheel SHA256 | `23dd9d42018c75fcb89bfededd33af3984665917ffa3e4e7492da7409f1ab6ca`; immutable S3 input in the manifest |
| Precision and attention | BF16 weights/activations, native KV, SM100 FlashInfer TRTLLM prefill and trtllm-gen decode |
| Cache | Prefix caching on; `additional_config={"grug_logical_kv_cache":true}`; retention interval 0 |
| Scheduling | 2048 batched tokens, 64 sequences per rank, memory fraction 0.95 |
| CPU placement | Native GPU-local NUMA binding |
| Experts | Native Triton; FlashInfer autotune off |
| Compilation | Mode 3, native ShortConv CUDA graphs NONE; exact splitting operations and Inductor settings in the manifest |
| Router | Released 0.1.16, consistent hashing, DP size 16, stable `X-Session-ID`, retries disabled, 14,400-second timeout |
| Protocol | AgentX 1.0.12, seed 42, context 262144, native count-driven warmup 1,406 requests, 3,600-second send window and configured 30-second profiling grace |

The new cache stores each of six distinct global-attention KV heads once instead of repeating them to twelve stored heads. [Full-model qualification](../results/guarded-logical-cache-policies-r1-qualified.json) preserves token IDs and top20 logprobs exactly at matching internal batch shapes. Native decode and affected layer tests pass. The opt-in guard restricts precision, head geometry, parallelism and attention backends to the qualified path. Default behavior remains unchanged in the draft PR.

## Reproduce

Use a clean isolated research checkout containing these files. The required inputs are private S3 artifacts, the pinned Hugging Face corpus and the ARM serving wheel; access to those stores and the Iris cluster is required. Preserve dependency constraints and the original corpus. Do not substitute a current wheel or shorten the native warmup.

Before submission, follow [native accelerator placement](../../../.agents/skills/use-iris/references/dev-accelerators.md), refresh the manifest’s `placement` block from current Iris/provider inventory, and announce the work through the [resource-board procedure](/home/romain/.agents/skills/coordinate-resources/SKILL.md). Require the selected spare NVLink domain to contain no Hero training; check protected training priorities and quota headroom across the cluster. Historical node names are evidence, not a reservation. Commit and push the refreshed manifest, with a new `completion_retention_prefix` such as `s3://marin-us-east-08a/marin/users/romain/hero-agentx-reproduction-unique`, before executing. Keep the model, runtime and replay fields fixed. The manifest’s original `prepared_not_launched` field describes preparation; the linked result records the completed execution.

From that research checkout, submit a native four-task batch gang with the existing wrapper. Replace the three paths/names below with the committed refreshed manifest, a unique job name and a local receipt path:

```sh
marin-env uv run --no-sync python \
  logbooks/19-hero-agentx-throughput/submit_diagnostic.py \
  --agentx \
  --manifest path/to/refreshed-manifest.json \
  --name hero-agentx-confirm-unique \
  --output /absolute/local/path/submission.json
```

The wrapper derives four hosts × four GB200s, 32 CPU cores, 768 GB memory and 1 TB disk per task from the pinned series. It uses native batch retries and coscheduling. The measured confirmation used interactive dev-GPU holders with the same task resources and this same `agentx_candidate.py` entrypoint. An interactive holder is useful only when explicitly needed; do not allocate a second gang inside one.

After actual admission, verify four native task IDs, their attempt UIDs, sixteen distinct physical GPU UUIDs, spare-domain placement, the pinned task image and current priorities. Check startup promptly. The entrypoint records effective serving commands, native attempts, metrics and CPU observations; its node guard rejects placement outside the verified node list. Run no numerical hooks, GPU profiler or competing owned experiment during confirmation. Require native attempt/window validity, finished models and supervisors, full raw/entrypoint readbacks and final native/provider identity checks. A killed enclosing holder is distinct from a successful benchmark child. [Completion and scoring commands](hero-logical-kv-completion.md) locate the receipts and run the unchanged validators and processor.

## Evidence and remaining work

The earlier matched-compiler flag-off/on hours measure 552.49 versus 748.62 total output tokens/s, an observed 35.50% difference, with API cache reads rising from 33.49% to 65.74%. Neighboring activity differs, so the entire ratio is not isolated causally. The final 811.36 rate confirms the guarded recipe under another run’s conditions; it is not an additional algorithmic gain over 748.62.

Periodic checkpoints every 16,384 tokens qualify numerically and help a synthetic first interior branch. Their AgentX hour gives 747.62 total output tokens/s versus default 811.36. Larger C256 load gives 584.22 versus the earlier C128 748.62. Both alternatives remain unselected. These observed comparisons have differing warmup or neighbor conditions, retained in their records.

Each policy’s result links four raw archives, four entrypoint archives, a processed capsule and an endpoint-analysis capsule with matching full-readback hashes. [The default holder release](../results/logical-kv-retention0-16-agentx-c128-r1-released.json) and [periodic holder release](../results/logical-kv-retention16k-16-agentx-c128-r1-released.json) prove both model allocations terminal and all original provider pods gone.

[Paired expert timing](../results/expert-backend-local24-timing-r1-completed.json) passes both original FP32 checks on identical inputs and weights. CUTLASS is slower at tested row counts 3–64, faster at 1/2/256, and has a Triton/CUTLASS latency ratio of 1.049 at 2048. These component timings exclude full-model communication and shared experts. Keep Triton; the mixed operator result and unresolved whole-model source equivalence do not justify a full CUTLASS campaign. All GPU jobs are terminal and no owned GPU allocation or queue remains. The required High-tier review against the project goal remains open. No shared defaults, owner worktrees, dashboard code, releases or PR merges change. The separate [lossless server-metrics patch](README.md) remains available for the benchmark owner.
