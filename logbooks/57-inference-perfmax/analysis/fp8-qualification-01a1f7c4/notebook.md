# FP8 qualification for Snowball and Mochi

The targets are Snowball v0.1.1 on H100 and Mochi step400407 on GB200. Reuse the pinned checkpoint and tokenizer fixtures, Levanter goldens, and AgentX workload from Marin #11320. Keep TP1 and BF16 KV. Compare to the qualified one-hour Snowball C128 and Mochi C64 and C128 deployments. The numerical contract includes reference, exact and changed cache splits, batch and decode.

## 2026-11-20

- The retained Mochi per-tensor Marlin deployment with vLLM #132, #133 and #134 passes all 19 exact cache checks. It fails the short reference maximum at book03 position 769. Both its upper and lower bounds are 0.672875 against a limit of 0.65, so the top-k cutoff does not explain that failure. The same position differs from the reference by 0.481496 and 0.554075 in two passing BF16 controls. The retained per-tensor Triton run reaches 0.762755 there.
- Assemble the three linked patches on vLLM main `c542f0c05fa` in an isolated branch. The experimental head `790f4cc3bd4db428e54bbf85bee5577b2337b4dd` also allows vLLM's existing 128 by 128 block scaling for GrugMoE expert weights. The first comparison uses Marlin's BF16 activations so it varies the weight scaling from the retained W8A16 run. Other projections and KV stay BF16, and the router stays FP32. Keep the original bounds.
- Build on CPU from the verified native payloads of the `47a8e415bff17b9376f02d9b92edcc67e962eaee` release. The build inputs name the source archive, base wheel hashes and targeted Python hashes. The diff from that base changes no native sources or dependencies. Use the ordinary correctness launch path before selecting any one-hour window.
- Both CPU packages passed after one retry per architecture. The first ARM build received the cross-platform option as two arguments and stopped in `build` argument parsing. The first x86 build packaged all original native payloads but the verifier looked for the legacy core extension name. The successful builds verify the MoE ELF architecture, byte equality for all native payloads, and the changed Python member hashes. All four jobs used zero GPUs.
- The raw scores and adjacent controls show two different cases. At book03:769, both tested BF16 vLLM runs favor token 279 over token 264 while Levanter favors 264. Both FP8 runs further raise 279. At book04:1016, both BF16 runs favor the same token as Levanter and both FP8 runs choose another. Adjacent positions have much smaller gaps. The retained response receipts verify input IDs, scorer indexing and the original saved log probabilities.
- Six down projections were sampled from the two fixed exports by byte range. Their approximate FP8 reconstruction error is usually about 2.65% in relative weight L2. Per-block scaling reduces the sampled errors only slightly. One early Mochi expert gains more from per-channel scaling, decreasing from 2.67% to 2.51%. These CPU values use synthetic inputs and BF16 scales, and they have no kernel or probability acceptance role.
- Snowball uses the same serving router as the BF16 baseline. Its fixed Levanter run used `router_dot_fp32=false`, `fp32_activations=false` and MoE capacity 1.0. The vLLM router uses an FP32 dot and inference has no MoE capacity drops. This is an existing difference in reference arithmetic to keep in view for failures. Marin #11776 records independent parity controls. Its different training settings do not establish a correction to this golden.

Sources

- [Fixed fixtures and shared checks](https://github.com/yonromai/marin/tree/3d04a90d36e393683a25ea01c1b3528f12ec3204/experiments/evaluation/inference_correctness)
- [Retained Mochi and Snowball results](../../handoff.md)
- [Blockwise FP8 runtime source](https://github.com/yonromai/vllm/commit/790f4cc3bd4db428e54bbf85bee5577b2337b4dd)

### First physical tests

Both models loaded FP8 in all routed expert layers. Marlin used BF16 activations. Other projections and KV/state used BF16, and the router stayed FP32. Both deployments passed 19 exact cache comparisons at TP1 with compilation and CUDA graphs on.

| Test | Reference max / p99 | Verdict | Allocated GPU-hours |
| --- | --- | --- | ---: |
| Snowball DP8, retention 128 | 0.226172 / 0.122289 | All required checks pass | 1.793333 H100 |
| Mochi DP4, retention 0 | Short 0.718176 / 0.124168. Long 0.233313 / 0.056864. | Short maximum fails 0.65. Batch, cache and decode pass. | 1.170000 GB200 |

Snowball passes all 1,770 fixed positions, four overlapping batch comparisons, 19 exact cache controls, 27 changed-step comparisons and 64 decode comparisons. The loaded footprint is 10.88 GiB per GPU versus 18.37 GiB for retained BF16 at DP8. Usable BF16 KV capacity rises from 2,575,809 to 2,943,772 tokens per engine, about 14.3%. The new C128 AgentX run measures the speed side of that tradeoff.

Mochi's blockwise run gives token 279 probability 0.945655 at book03:769 against 0.227479 in the golden. At book04:1016 it recovers the golden's top token. Finer scales can move different tokens in different directions. The next run keeps the first 24 expert layers in BF16 and quantizes the last 24. This preserves the early residual path and keeps half of routed weights in FP8. It uses the previously tested hybrid backend code on current main. The new engine precision check verifies the selected layers on every worker.

Twelve focused tests passed on each GPU type on the full-block wheel, including the new block loader case. They cost 115 H100-seconds and 107 GB200-seconds.

### Activation precision and layer selection

Snowball's block FP8 Triton run uses dynamically quantized FP8 expert activations. On the same DP8 topology and retention setting it passes batch, cache and decode, including all 19 exact cache cases. The fixed reference maximum is 0.302993 against 0.25. At `knowledge-longbench-02:15022`, the same token 5618 has probability 0.769889 in Levanter, 0.704756 in BF16 vLLM, 0.545471 in block Marlin and 0.466896 in block Triton. The scorer's lower and upper bounds coincide. The comparison varies the expert implementation as well as activation precision. [Aligned scores](snowball-w8a8-worst-token-scores.json)

Mochi's late-24 run uses BF16 Triton in layers 0–23 and block FP8 Marlin in layers 24–47. It loads 119.52 GiB per GPU at DP8, passes batch and all cache checks, and reaches a long-reference maximum of 0.15531. It still fails the short-reference maximum at book03:769 (0.69831 against 0.65) and decode p99 (0.170644 against 0.15). A short pair compares last-16 block FP8 with all-BF16 on the same current runtime, DP8, retention zero and eight fixed histories.

The first pair stopped during subset selection. The test passed a Pydantic model to a dataclass helper and stopped before scoring. These attempts used 3.05 GB200-hours. The corrected test was replayed through a local HTTP endpoint serving eight authentic retained responses, including deployment verification and 7,341 scored positions, before the second pair was submitted.

The routing-order microbenchmark compares alignment alone with alignment plus the exact #133 sort helper under CUDA graph replay. For 16 to 32,768 synthetic rows after dispatch, the added median time was 62–125 microseconds on H100 and 59–129 microseconds on GB200. Each sorting call handles one routed layer. The direct single-token path skips the sort. The measurement excludes serving, inter-device dispatch and cache effects. Costs were 65 H100-seconds and 66 GB200-seconds. [H100 timings](routing-timing-x86_64-results.json), [GB200 timings](routing-timing-aarch64-results.json)

### Snowball session routing

The retained C64 BF16 run eased KV pressure but still used only 24–28% of prefixes in live samples against 95–97% theoretical reuse. Early samples from the FP8 C128 run also show a large gap. Test the runner's existing consistent-hash router, which uses the AgentX session header to send later turns to the same native rank. Keep C128, DP8, retention 128 and the exact wheel. Run a fresh one-hour point, with the same checks on its worker before the timed client traffic goes through the router. [Earlier cache observations](../../snowball-tuning-01a1eabe.md)

```bash
marin-env .venv/bin/python -m experiments.evaluation.knotmeter.cli run \
  --profile logbooks/57-inference-perfmax/analysis/fp8-qualification-01a1f7c4/snowball-block.yaml \
  --set serving.router=true --campaign fp8-qualification-snowball-01a1f7c4 \
  --priority interactive --logbook https://github.com/marin-community/marin-logbooks/issues/57
```

Use `snowball-bf16-c128-control.yaml` for the same-wheel BF16 control with the router. It keeps the workload, C128, DP8, retention 128 and full gate. Kernel selection is the native BF16 default. This controls for the new runtime and route placement when estimating the cost of the FP8 recipe.

### Matched Mochi controls at DP8

The BF16 control passes the eight-history probe. At book03:769 the measured reference error is 0.581942 in BF16 and 0.732109 with the last 16 expert layers in block FP8. The short p99 values are 0.083103 and 0.113886. The long checks pass in both. Model-load memory falls from 149.42 to 129.49 GiB per GPU (13.3%) at matched DP8. This is a rejected precision setting. The retries cost 4,372 GB200-seconds for BF16 and 4,608 for late-16 FP8.

Reducing the late block selection from 24 to 16 layers made the worst token further from the golden. Test two separate changes from late-16 block at DP8. Move the same block FP8 coverage to layers 16–31, which leaves the final third in BF16. On the last 16 layers, try per-tensor scales, whose retained all-layer result was closer on the limiting token. Both use the same short probe and the same 16 of 48 expert layers. A short pass will select a candidate for the full suite.

The middle-16 block run passes all eight histories. Short maximum and p99 are 0.614273 and 0.101123. The late-16 tensor run reaches 0.655094 on the same worst token, down from 0.732109 for late-16 block. It still exceeds the 0.65 maximum and has a subset p99 of 0.125352. Middle-16 costs 4,416 GB200-seconds and late-16 tensor costs 4,424. Preserve the layer placement for the full DP16/C64 point with the same runtime and retention zero. The changed expert partition and the untested histories, batches and decode steps all still need validation.

The BF16 control already differs from the golden. The saved Levanter short run used BF16 from the fixed checkpoint, the dropless Sonic CuTe experts and zero reported routing overflow. Current vLLM's Mochi router also rounds its dot product to BF16 before FP32 bias and top-k. Levanter casts the resulting routing combine weights to BF16 while vLLM's router returns them in FP32. This and differences in expert and attention arithmetic are possible contributors. The retained responses cannot assign the error to one of them. [Aligned DP8 scores](matched-mochi-dp8-scores.json)

```bash
marin-env .venv/bin/python -m experiments.evaluation.knotmeter.cli run \
  --profile logbooks/57-inference-perfmax/analysis/fp8-qualification-01a1f7c4/mochi-mid16-block.yaml \
  --set gpus.count=16 --session-concurrency 64 --campaign fp8-qualification-mochi-01a1f7c4 \
  --priority interactive --logbook https://github.com/marin-community/marin-logbooks/issues/57
```

### Arithmetic with real expert weights

Take two experts from Mochi layer 31. The test feeds identical synthetic BF16 rows and fixed FP32 route weights to Marlin and a plain FP32 matmul reference with BF16 intermediate rounding. The reference dequantizes the exact native FP8 values and the same BF16 scales that Marlin receives.

Casting weights to FP8 changes the reference output by 4.55–4.77% in relative L2 across one, 41 and 257 rows. Marlin's output differs from the reference using the same quantized values by 0.48% for tensor scaling and 0.62–0.64% for block scaling. Removing the reference's intermediate BF16 rounding changes that reference by 0.38–0.40%. Weight approximation is the larger effect in this sample. The remaining difference includes arithmetic and rounding in the two implementations. Fixed synthetic inputs and routes limit what this says about token probabilities or routing changes. Cost was 87 GB200-seconds. [Native arithmetic and input hashes](component-real-experts-results.json)

The exact, separate block loader contribution passes 12 focused H100 tests (116 GPU-seconds). The combined hybrid wheel passes 14 focused GB200 tests (99 GPU-seconds). [Block contribution](component-block-contribution-results.json), [Hybrid wheel](component-mochi-hybrid-results.json)

### Full Mochi qualification gate

At DP16 the middle-16 block deployment passes the shared reference, batch, cache and decode checks before AgentX. The short maximum is 0.647211 at book03:769 against 0.65, with p99 0.092948 against 0.125. The long maximum is 0.274558 against 0.35. Decode p99 is 0.038357 against 0.15. Preserve the small short-reference margin when describing this result.

The same point loads 80.96 GiB per GPU versus 86.42 GiB in the retained BF16 DP16 run, with usable BF16 KV capacity of 1,218,836 versus 1,102,533 tokens per engine. The normal C64 replay is in progress. Run a fresh full gate and normal C128 comparison at DP32 to match the strongest retained BF16 throughput point. It uses the same profile with `--set gpus.count=32 --session-concurrency 128`. DP32 needs its own gate because the expert partition changes.

The exact, separate hybrid contribution also passes on GB200. It targets the existing selective FP8 draft and uses the ordinary BF16 loader test with Marlin configured globally. [Hybrid contribution regression](component-hybrid-contribution-results.json)

### Snowball cache retention under session affinity

The early C128 router samples show more actual prefix reuse than the unrouted trial. At 09:08:51 UTC the routed point shows 53.3% actual hits against 95.9% theoretical, with 91.4% KV occupancy. These are live samples. The fixed C128 retention-zero test keeps session affinity and skips periodic sliding-window checkpoints. It can show whether keeping only semantic checkpoints preserves more useful histories under load. Run the same full numerical gate and one-hour replay on `snowball-block-router-retention0.yaml`.

### First completed one-hour Snowball window

Without the router at C128, block FP8 delivers 122.41574 output tokens/s/GPU, 104.73873 s p90 TTFT and 301.64845 s p90 request completion. The previous qualified C128 BF16 run on a different vLLM source reached 124.39607, 103.33113 s and 297.61639 s. The replay recipe, saved corpus, seed, golden and numerical limits match. FP8 is 1.59% slower at this point with 7.49 GiB less loaded memory per GPU. FP8 measured 10.373% actual prefix hits. AgentX reports 5,107 profiling successes and one profiling error, with a valid and unchanged gang. The allocation cost is 42,872 H100-seconds, including load, gate, warmup, replay and drain. [Native C128 FP8 trial](s3://marin-us-east-08a/marin/knotmeter/campaigns/fp8-qualification-snowball-01a1f7c4/trials/20261120T074542Z-a2724beec7dc/result.json)

### Mochi DP32 gate and the distribution of DP8 changes

Middle-16 block FP8 also passes the full DP32 checks. The short maximum is 0.521529 and p99 is 0.097216. The long maximum is 0.283892 and decode p99 is 0.054338. At DP32 the model-load profiler shows 60.06 GiB per GPU versus retained BF16 at 56.05 GiB. Usable BF16 KV capacity is 1,507,772 versus 1,456,262 tokens per engine. The C128 replay will measure the performance side of those memory figures.

On the same eight DP8 histories, middle-16 block differs from matched BF16 by 0.08691 at p99 over 6,815 short positions. Late-16 block, late-16 tensor and late-24 block reach 0.09237, 0.09564 and 0.12587. Middle-16 changes the BF16 top token at 114 short positions, versus 89 for late-16 block. The shared gate uses probability differences against Levanter. [Matched DP8 probability changes](matched-mochi-dp8-probability-shifts.json)

### Snowball retention-zero gate and paired control

The routed retention-zero point passes the full gate, including 19 exact cache comparisons. Its reference maximum and p99 are 0.226172 and 0.122289, the same as retention 128 in the first run. The pinned warmup finishes all 1,406 requests in 220.43 seconds, compared with about 17 minutes in the routed retention-128 run. A normal window will measure throughput under sustained load. Use `snowball-bf16-c128-retention0.yaml` to pair the same cache and routing settings with BF16 on the same wheel.

### Mochi full decode graph preflight

The hybrid wheel includes vLLM #130. Its existing tests exercise graph normalization, padded rows and repeated full-graph decode with changing inputs and state slots. All nine pass on one GB200, including exact comparisons with eager decode. The allocation cost is 84 GB200-seconds. The prepared C64 profile changes only the graph mode to `FULL_AND_PIECEWISE` and preserves the ShortConv split. Use the full model gate before measuring it. [Native component results](component-mochi-graphs-results.json)

### Snowball C128 with routing and retention 128

The valid routed comparison reaches 138.08517 output tokens/s/GPU, p90 TTFT 100.91485 s and p90 request completion 272.88136 s. Session routing adds 12.8% throughput and reduces p90 completion by 9.5% relative to the same wheel without routing. The window has 5,538 profiling successes, zero errors, and a complete allocation of 43,432 H100-seconds.

AgentX reports a GPU prefix-hit rate of 30.77%, versus 10.37% without routing. Its exporter warns that some server counters decreased. The selected Iris attempt and GPU gang stayed the same, and Kubernetes recorded zero container restarts. Use the client completions for the performance comparison and treat the server-counter estimate as approximate. [Native routed trial](s3://marin-us-east-08a/marin/knotmeter/campaigns/fp8-qualification-snowball-01a1f7c4/trials/20261120T083024Z-16220c81d086/result.json)

### First completed Mochi window

Middle-16 block FP8 at DP16/C64 reaches 28.21292 output tokens/s/GPU, p90 TTFT 240.48749 s and p90 completion 257.17333 s. The retained C64 BF16 run on an earlier wheel reached 23.60952, 276.38157 s and 282.52929 s. The FP8 trial yields 19.5% higher throughput with 5.46 GiB less loaded memory per GPU. All 16 archived rank checks confirm the same FP8 layers and BF16 expert activations, attention cache and convolution state. The window is valid with 2,325 successes, zero errors and 86,064 allocated GB200-seconds. [Native Mochi C64 trial](s3://marin-us-east-02a/marin/knotmeter/campaigns/fp8-qualification-mochi-01a1f7c4/trials/20261120T084518Z-4921acafea29/result.json)

Run two follow-ups at the same DP16/C64. The same-wheel BF16 control uses `mochi-bf16-dp8-control.yaml` with `--set gpus.count=16 --session-concurrency 64`. The full-decode FP8 point uses `mochi-mid16-block-full-decode.yaml` with the same overrides. They measure the precision and graph-mode changes separately, with the existing full gate before warmup in both trials.

### Matched Snowball BF16 with retention 128

The same-wheel routed BF16 control reaches 134.44815 output tokens/s/GPU, p90 TTFT 103.93386 s and p90 completion 274.89422 s. Routed FP8 at 138.08517 is 2.7% faster in these two windows. Every rank reports a load footprint of 20.87 GiB in BF16 and 10.88 GiB in FP8. The stored parameter bytes are 19,687,358,464 and 11,636,260,864 per rank. Usable BF16 KV rises from 2,572,520 to 2,943,772 tokens per engine. The BF16 control cost 43,016 H100-seconds. [Matched BF16 trial](s3://marin-us-east-08a/marin/knotmeter/campaigns/fp8-qualification-snowball-01a1f7c4/trials/20261120T085808Z-e995203c42d8/result.json)

### Mochi DP16 follow-up gates

The same-wheel BF16 control and full-decode FP8 both pass all shared groups. Their short maxima are 0.564567 and 0.527668, their decode p99 values are 0.041541 and 0.053453, and both pass 19 exact cache cases. The full-decode worker logs confirm full CUDA graph capture. It loads 80.96 GiB per GPU against 90.92 GiB for the same-wheel BF16 point. The normal windows start with fresh caches after the gates.

### Isolate Marlin ordering cost

The alignment microbenchmark measures the sort in isolation. A matched serving run can test whether that cost moves client throughput and whether Snowball needs the sort for stable cache reuse. [H100 component timings](routing-timing-x86_64-results.json)

The experiment head `c96f9b5d27767a13c69961637c75dbd00622b010` removes the two canonical-ordering commits from the tested Snowball wheel. The source comparison changes only Marlin ordering and its isolated regression. The routed C128 retention-zero profile changes only the wheel. Run the full gate first. A mismatch in the fixed or exact cache comparisons stops the run before AgentX. A passing gate enables an uninstrumented normal hour for the one-setting throughput comparison.

### Completed Mochi DP32/C128 window

Middle-16 block FP8 reaches 31.01618 output tokens/s/GPU, p90 TTFT 270.17412 s and p90 completion 283.74354 s. The strongest retained C128 BF16 run on an earlier wheel reached 31.42858, 268.89855 s and 281.34879 s. FP8 is 1.3% slower in this comparison and has 51,510 more usable BF16 KV tokens per engine. Its model-load profile shows 4.01 GiB more per GPU. These are separate memory measurements.

The window is valid with 5,164 profiling successes and three HTTP 500 errors. The later Gloo peer-close exception starts after AgentX finalizes during server teardown. Keep it separate from the client errors, whose root cause remains open. The recorded cost is 184,894.612 GB200-seconds. Seven hosts use pod-container finish and the leader uses the Iris terminal timestamp. [Native DP32 trial](s3://marin-us-east-02a/marin/knotmeter/campaigns/fp8-qualification-mochi-01a1f7c4/trials/20261120T090555Z-3728f9354610/result.json)

### Snowball routed retention zero

The qualified retention-zero FP8 point reaches 203.87185 output tokens/s/GPU with p90 TTFT 63.59688 s and p90 completion 175.13810 s. At retention 128 the same wheel and placement policy reached 138.08517, 100.91485 s and 272.88136 s. Retention zero cuts the periodic sliding-window checkpoints and keeps replay boundaries and shared-prefix junctions. Both points load 10.88 GiB per rank and allocate a usable 2,943,772 BF16 KV tokens per engine. All eight rank receipts show the same 26 block FP8 Marlin layers. The retention-zero point cost 37,800 H100-seconds. [Native retention-zero trial](s3://marin-us-east-08a/marin/knotmeter/campaigns/fp8-qualification-snowball-01a1f7c4/trials/20261120T091052Z-f7c297ad3940/result.json)

Read `aiperf_artifacts/profile_export.jsonl` from each native leader archive and sum its request-local `usage_prompt_cache_read_tokens` over `usage_prompt_tokens` for successful profiling requests. Retention 128 returns 31.12% cached prompt tokens and zero returns 68.64%. Across 2,116 shared trace turns with equal prompt counts, the shares are 32.89% and 66.67% and mean TTFT is 29.75 s and 16.87 s. Match on conversation, source trace, source outer and inner indices, and turn index. Per-request usage corroborates the cache difference without using the cumulative counters. The matched subset has equal total prompt and completion counts. Retention zero profiles 7,638 successes with zero errors. The raw archive has 395 warmup-only missing-content errors, versus 413 at retention 128. [Request-local counts](snowball-retention-cache-usage.json)
