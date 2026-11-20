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

### Matched Mochi controls at DP8

The BF16 control passes the eight-history probe. At book03:769 the measured reference error is 0.581942 in BF16 and 0.732109 with the last 16 expert layers in block FP8. The short p99 values are 0.083103 and 0.113886. The long checks pass in both. Model-load memory falls from 149.42 to 129.49 GiB per GPU (13.3%) at matched DP8. This is a rejected precision setting. The retries cost 4,372 GB200-seconds for BF16 and 4,608 for late-16 FP8.

Reducing the late block selection from 24 to 16 layers made the worst token further from the golden. Test two separate changes from late-16 block at DP8. Move the same block FP8 coverage to layers 16–31, which leaves the final third in BF16. On the last 16 layers, try per-tensor scales, whose retained all-layer result was closer on the limiting token. Both use the same short probe and the same 16 of 48 expert layers. A short pass will select a candidate for the full suite.
