# Repeat the selected Snowball and Mochi FP8 comparisons

The four profiles repeat the selected comparisons from the prior [FP8 campaign](https://github.com/marin-community/marin-logbooks/tree/652ad9a3f5f7da48d93223b033f43d05af43a2b5/logbooks/57-inference-perfmax/analysis/fp8-qualification-01a1f7c4). Snowball uses 8 H100s at 128 sessions and puts all 26 routed expert layers in block FP8. Mochi uses 16 GB200s at 64 sessions and puts layers 16–27 of 48 in block FP8. Both pairs use TP1, BF16 KV and the original checkpoints, goldens, numerical limits and AgentX replay. Each arm starts a fresh compiled server.

Both architectures package vLLM `e9dfdd2b94827b8f0f9ae4d21fff5f1a2a5dc796`, built from current main `06963d0b8bf361b1ed357dbe9b6c9fa44eea548f` with the exact #132, #133, #135 and combined #136 commits. The two build inputs pin one source archive and the two native base artifacts from release `47a8e415bff1`. The source diff from that release changes no native sources or dependencies. The packager verifies every native payload and each changed Python member. The only runtime source change from the retained Mochi wheel reuses the same expert-prefix string. The Snowball additions only select BF16 Triton for explicitly ignored FP8 experts. Neither Snowball arm ignores an expert layer.

The existing runner performs the storage and numerical gates followed by a three-minute Probe. A passing arm can proceed to the one-hour Confirm with the pinned AgentX warmup. Run from the recorded research checkout with the exact profile name.

```sh
inputs=logbooks/57-inference-perfmax/analysis/da6-fp8-confirm-01a1f7c4
uv run python -m experiments.evaluation.knotmeter.cli run \
  --stage probe --profile "$inputs/snowball-bf16-c128.yaml" \
  --campaign da6-fp8-confirm-01a1f7c4 --priority interactive
```

## Capture path

Both Snowball Probes finished with full checks and native archives on runner `4546acc39a32a528841bcdc22b20e744ada0265b`. Snowball Confirm uses the same runner. Both Mochi Probes also start on that revision. Mochi Confirm uses DA5's `8689b72dd46abe09b3be3c58cea155387870bf11`, which fixes coordinated follower shutdown and retains a leader snapshot before shutdown. It changes result capture and accounting after the timed window. The serving profiles, checker, goldens, limits and AgentX protocol are identical across these runner revisions.

The existing capture, trial and follower tests passed 67 cases on the Mochi launch checkout. A small CPU fixture used the exact capture helper to publish and read back two checksum-verified archive paths in the target S3 bucket. The normal path replaced the receipt with its final archive while keeping the earlier archive readable. The test without a final writer kept the before-shutdown receipt. These fixtures have no GPU measurements. [Storage receipts](s3://marin-us-east-02a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/capture-validation/8608046f605a/)

## Confirm results

Both FP8 recipes improved completed output throughput again. The gains were smaller than in the retained comparisons. All four fresh servers passed the original provisional numerical limits. Both pairs used the pinned AgentX 1.0.12 corpus, seed 42, cache reset, native warmup with ten additional requests per lane and a 3,600-second timing phase. All four deployments used TP1 and compilation with CUDA graphs. Mochi used piecewise graphs and a 2,048-token prefill cap.

| Model and weights | Output tokens/s/GPU | p90 TTFT (s) | p90 completion (s) | Timed successes / errors | Allocated GPU-hours | Native result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Snowball BF16 | 171.79464 | 72.93746 | 208.14216 | 6,625 / 0 | 10.6978 H100 | [59784e614e0d](s3://marin-us-east-08a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/trials/20261123T054812Z-59784e614e0d/result.json) |
| Snowball block FP8 | 178.03461 | 72.65030 | 189.36350 | 6,840 / 0 | 10.5156 H100 | [7e8d6f8aeaf3](s3://marin-us-east-08a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/trials/20261123T054812Z-7e8d6f8aeaf3/result.json) |
| Mochi BF16 | 27.32616 | 240.82713 | 249.07795 | 2,278 / 1 | 25.9633 GB200 | [4ec331659527](s3://marin-us-east-02a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/trials/20261123T055738Z-4ec331659527/result.json) |
| Mochi layers 16–27 block FP8 | 33.76948 | 206.06079 | 219.80529 | 2,624 / 1 | 24.1200 GB200 | [ac616c595a31](s3://marin-us-east-02a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/trials/20261123T055738Z-ac616c595a31/result.json) |

Snowball’s prior p90 TTFT was 84.22 s for BF16 and 63.60 s for FP8. Prior p90 completion was 216.11 s and 175.14 s. Mochi’s prior p90 TTFT was 252.47 s and 191.53 s. Prior p90 completion was 262.37 s and 199.39 s.

### Snowball on eight H100s at C128

The FP8 advantage was 3.63% in the repeat and 26.18% in the retained pair. The BF16 arm was 6.33% faster than its retained run. The FP8 arm was 12.67% slower than its retained run. Both pairs used H100 80 GB HBM3, driver 595.71.05, TP1/DP8/EP8, cache retention zero, the session router and the same AgentX workload. The mixed-precision fallback added to the Snowball wheel only applies to explicitly ignored experts. The two Snowball profiles exercise the same paths as before.

BF16 and FP8 stored 19,687,358,464 and 11,636,260,864 model parameter bytes per rank. FP8 saved 7.50 GiB per rank and packed all 26 routed expert layers. Usable KV capacity increased from 2,572,520 tokens per engine to 2,943,772–2,944,248. Both arms used BF16 activations and KV.

In 2,257 unique shared trace turns with equal prompt counts, the prompt cache served 51.98% of BF16 tokens and 68.78% of FP8 tokens. The corresponding retained shares were 50.52% and 67.11%. The matched sets also have equal output counts within each comparison. Over all successful timed requests with usage, the cache-share gap narrowed from 14.94 to 8.18 percentage points. These aggregate shares reflect which requests completed during each window. The per-turn cache advantage persisted, and the size of the whole-recipe throughput gain varied between windows. The request records do not identify a single cause for the performance change.

All reference, batch, cache and decode checks passed. Snowball's FP8 reference max and p99 were 0.226172 and 0.122289 against the provisional limits of 0.25 and 0.125. BF16 reached 0.196002 and 0.088755. Both had 19 exact same-split comparisons with zero error.

### Mochi on 16 GB200s at C64

The FP8 advantage was 23.58% in the repeat and 41.45% in the retained pair. BF16 was 8.15% faster than its retained run. FP8 was 5.52% slower than its retained run. Both repeat arms used driver 595.71.05, TP1/DP16/EP16, the session router and cache retention zero. The two runs used the same combined runtime and the same pinned capture revision. Rebuilding the prepared archives changed their packaging hashes. The native cache key, exact corpus and tokenizer file hashes, metadata and replay counts match the retained pair.

BF16 and FP8 stored 92,733,505,536 and 84,707,180,544 model parameter bytes per rank. FP8 saved 7.48 GiB per rank. All 16 ranks packed exactly layers 16–27 of 48 as block FP8 with Marlin. The remaining 36 routed layers ran in BF16 with Triton. Usable KV increased from 1,085,219 to 1,185,372 tokens per engine. Both retained BF16 activations, KV and ShortConv state. Across 439 shared trace turns with equal prompt and aggregate output counts, the prompt cache served 80.18% of BF16 tokens and 89.40% of FP8 tokens. The retained shares were 75.42% and 89.71%. The smaller cache-share gap can help explain why the whole-recipe advantage changed. This comparison varies expert storage, the expert backend and usable cache capacity together.

Mochi FP8 short-reference max and p99 were 0.606594 and 0.095197 against limits 0.65 and 0.125. BF16 reached 0.560200 and 0.089248. The FP8 long-reference max and p99 were 0.211190 and 0.055809 against 0.35 and 0.09. Both arms passed batch, cache and decode and had 19 exact same-split comparisons with zero error. Both native runs retained an early leader archive plus a final archive. Each had one timed error and completed within the AgentX error budget. BF16 and FP8 had two and one additional warmup errors, respectively.

### Reproduce the measurements

Use source commit `766f22517df47690adb131db889da841ccaff0ce` for Snowball and `624196047f02bc23b6ef095d10989f612b6cf031` for Mochi on the logbooks branch. Use the four profiles in this directory with `--stage confirm`, a new campaign name and interactive priority. The S3 result records include the frozen plans, native exports, the complete correctness reports and their hashes, all-rank storage, vLLM logs and every Iris attempt. Each bucket also contains the corresponding `summary.md` and selected `recommended.yaml`. [Snowball recipe](s3://marin-us-east-08a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/recommended.yaml) · [Mochi recipe](s3://marin-us-east-02a/marin/knotmeter/campaigns/da6-fp8-confirm-01a1f7c4/recommended.yaml)

The x86_64 wheel SHA-256 is `4c0fdfbbeb236c77f188701c7a9acfddffb8cb9b42ad98de37326b865c155602`. The aarch64 wheel SHA-256 is `dce5545688f759bba616afce3d32bd387c8a26d0a59d829ce26527ed6f8256b7`. Both install from checksummed manifests. The focused wheel suite passed 22 GPU cases on each architecture, including the five loader combinations, online quantization scope, padding and canonical Marlin ordering. The build inputs and focused test inputs in this directory name the exact source archive, native base, wheel manifests and result locations.

Across all attempts, the follow-up used 24.7639 H100-hours and 62.7253 GB200-hours. The campaigns contain 24.7311 and 62.6933 hours. The focused checks used the rest, 118 H100-seconds and 115 GB200-seconds. CPU packaging and capture checks used zero GPUs. Pod scheduling and container finish times covered every allocated attempt. All trials and focused jobs reached terminal success and the allocations were released.

One new window per arm gives a matched repeat for each recipe. Both pairs show consistent gains in usable KV and throughput direction. The size of the gains differed between windows. The original numerical limits remain provisional and cover serving.
