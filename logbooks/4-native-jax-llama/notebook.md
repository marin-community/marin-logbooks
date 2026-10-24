---
title: "Replacing Llama's named-array implementation with native JAX"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/4"
related_issues: []
research_branch: "codex/4-native-jax-llama"
marin_base_revision: "8aa8674d10894728ed0b455ae14e14f46457cf9f"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Replacing Llama's named-array implementation with native JAX

## Question and decision criteria

The goal is to replace the old Llama implementation and its existing entry points with native JAX arrays and explicit sharding, then remove the old implementation once correctness, checkpoint, and performance gates pass. `NativeLlama` is a transitional implementation name. Keeping two permanent Llama implementations is not the intended outcome.

The named-array implementation remains the migration reference. Both variants must use the same concrete initial weights, input batch, devices, attention/loss backend, optimizer, rematerialization policy, and physical partitions within a comparison. Auto versus Explicit mesh axis types are the intended control-mode difference. A matching parameter count or logical partition name is insufficient evidence of matched placement.

FP32 acceptance was explicitly revised during the campaign: first-step loss, gradients, and Adam state retain elementwise `atol=1e-5, rtol=1e-4`; **every** first-update and next-parameter tensor must have relative L2 error at most `0.005`; the vector of 20 measured losses must have relative L2 error at most `0.001`. BF16 retains all original elementwise checks. Exact within-pair initial parameter hashes and matching physical placement remain mandatory. The revision does not authorize further tolerance changes.

Performance evidence is a synchronized normal Trainer step, including backward and AdamW, with five warmups and 20 timed steps. Diagnostic hooks and the two optional post-timing profile steps are separate. A failed numerical comparison has a null accepted speed ratio even when raw timings exist. Short-run gates do not establish longer convergence or completion of the migration. No minimum speedup or allowable performance regression has been set for cutover; accepted full-step timings and profiles must inform that decision.

## Source and archive boundaries

This research tree starts at the published candidate `8aa8674d10894728ed0b455ae14e14f46457cf9f` from the `dlwh/marin` fork. The selected candidate includes a BF16-only barrier immediately before query scaling; it has not yet passed the full 1.53B Trainer gates. The archive retains the selected source tree and records historical measurements under their actual pins.

Two historical checkpoint measurement commits, `b8c47e8eea0d39412e3631404588b4f25ce4e45c` and `f659101960398fe9ad104e66600c0ccf3a2f34c8`, were outside the selected candidate's ancestry. Archive-only `ours` merges retain those histories without selecting their source trees as the current implementation. Those merges do **not** mean the checkpoint contribution has been integrated into the selected candidate. Historical runner snapshots are also retained for inspection. The production-entrypoint, checkpoint-runner and mask-boundary contribution histories are also retained through an archive-only `ours` merge. The [replacement gap audit](analysis/native-jax-llama-replacement-gap-audit.md) records their exact pins; retaining that history does not integrate those features into the selected `8aa` tree.

The artifact manifest records the selected files' source paths, archive paths, sizes, SHA256 hashes, and measurement pins. Original measurement reports remain unchanged alongside later acceptance evaluations. Machine-generated historical paths in frozen records are provenance, not portable instructions; [the reproduction guide](README.md) explains how to work from the archive. Raw profiles, checkpoints, weights, and routine logs remain outside Git. Small decisive errors, layouts, HLO, source, and analysis are retained here so the conclusions do not depend solely on seven-day artifact retention.

## Baseline workloads

| Workload | Parameters | Decoder | Vocabulary | Batch × sequence | Mesh |
| --- | ---: | --- | ---: | --- | --- |
| Small comparison | 106,972,160 | 8 layers, width 1024, MLP 2816, 16 query / 4 KV heads, head 64 | 8,192 | 8 × 512 | data 2 / model 2 |
| Large comparison | 1,531,037,696 | 28 layers, width 2048, MLP 5632, 16 query / 4 KV heads, head 128 | 65,536 | 8 × 2048 | data 2 / model 2 |
| Distributed checkpoint check | 1,531,037,696 | Same large architecture | 65,536 | 8 × 2048 | data 4 / model 2 |

The large parameter count was verified through the actual model's abstract construction. Parameters and Adam moments are FP32; compute is FP32 with highest precision or BF16 with default precision. The retained TPU reports use JAX/jaxlib 0.11.1 and `XLA_FLAGS=--xla_allow_excess_precision=false`. The large matched comparison uses AdamW with learning rate `0.001`, weight decay `0.01`, betas `0.9/0.999`, epsilon `1e-8`, scan and rematerialization.

Within each pair the initial weights and batch hashes match. Initial hashes differ across hardware runs; no result below claims identical weights across TPU generations and H100. The small vocabulary and shorter sequence make the 107M timing result a different workload from the large comparison.

## Entries

### 2026-10-23 — Native model, Trainer support, and the first numerical boundary

The initial contribution separated the native model, ordinary Trainer raw-array support, and a matched benchmark. This kept the production Trainer path in the experiment rather than comparing two standalone training loops. A later benchmark commit, `10f09d8ca27e1d454f83de59568d01106ce3bab3`, also retained first-step parameter and Adam snapshots from the normal timed executable. Host snapshots own their copies before later donation. The earlier v5p measurement at `9972d9f369ab4295ead7512d89ece98c9ed02b24` has diagnostic state capture only; its timing still uses the no-hook executable.

One concrete error came from native RoPE inverse-frequency construction. Evaluating the power while tracing on TPU differed from the reference's compile-time frequencies. The retained four-case actual-TPU probe covers FP32/BF16 and RoPE linear-scaling factors 1 and 8. Compile-time evaluation removes recorded forward and input-gradient differences; the old traced power reaches about `0.003655` FP32 and `0.03125` BF16 absolute error. This source change was retained. The source history contains the selected compile-time fix at `c9df38cddafd37b62040795be47b900ad950bdf7`.

Other lowering hypotheses did not justify a production patch. An attention-order experiment used base pin `55ce80535f5bcdd21797a9cfc2dcc025beb357dc` plus an explicit disposable runtime patch. A Q-only articulated compute-view experiment used base `a3d7f220c3a6f1fd150b6e53760c615a5dc08b72` plus its own runtime patch. The latter did not improve the FP32 gradient/update discrepancy, and both retained failed optimizer checks. Neither patch was adopted. The archive keeps their patch sources and failed FP32 reports; a clean source commit alone does not describe these runtime-patched experiments.

Evidence: [RoPE probe](evidence/rope/precision-probe.json), [attention-order provenance](evidence/rejected/attention-order/provenance.json), [Q-view provenance](evidence/rejected/q-compute-view/provenance.json), and their adjacent reports and patch sources.

### 2026-10-23 15:19–15:58 UTC — Re-evaluating the 107M FP32 checkpoint

The completed v5p, v4, and H100 comparisons initially failed strict elementwise update/parameter criteria. FP32 first-loss, gradient, and Adam checks passed; sparse near-zero updates still produced elementwise failures. H100 also failed the original elementwise loss trajectory. BF16 reference-backend comparisons failed additional gradient and trajectory gates.

After reviewing those discrepancies, the user approved the FP32 per-tensor and loss-vector relative-L2 limits stated above. Evaluator commit `2a4de13094a488af0b2df9325ef2fc50c2c60469` changes host-side acceptance only; it does not change the model, optimizer, compiled timed step, or old measurements. The three FP32 reports pass the approved criteria without a new hardware run. Original raw reports and failed job outcomes are retained rather than rewritten.

| Hardware | Measurement source | Legacy / native median step | Accepted FP32 ratio |
| --- | --- | ---: | ---: |
| v5p-8 | `9972d9f369ab4295ead7512d89ece98c9ed02b24` | 40.5603 / 27.0249 ms | 1.50085× |
| v4-8 | `10f09d8ca27e1d454f83de59568d01106ce3bab3` | 64.7087 / 48.3183 ms | 1.33922× |
| 4 H100 | `10f09d8ca27e1d454f83de59568d01106ce3bab3` | 40.5502 / 34.3289 ms | 1.18123× |

Worst FP32 update-tensor relative L2 is about `0.0187% / 0.0167% / 0.0953%` respectively, below the approved `0.5%`. H100's measured-loss-vector error is about `0.0111%`, below `0.1%`. These are accepted small-model FP32 measurements, not BF16 acceptance, large-model performance, or longer-convergence results.

Evidence: [original reports and approved evaluations](evidence/reference-107m/), with separate measurement and evaluator pins in every evaluation.

### 2026-10-23 16:27 UTC — Scaling the profiling question to 1.53B

The small model suggested embedding redistribution might explain the speedup: the named path had two embedding all-to-alls absent in native. The user requested a realistic approximately 1.5B profile rather than extrapolation from a vocabulary of 8,192.

Execution `levanter-native-jax-v5p-15b-profile-20261023`, source `ad2bc20438796c63415240ed1979e93d212fc6da`, completed both implementations on one v5p-8 slice. Raw synchronized medians were `1068.1201 / 1074.3019 ms`. Initial loss, all 255 gradients, and diagnostic/normal Adam state passed strict FP32 checks. Update/next-parameter tensor errors were within `0.5%`; the measured-loss-vector relative L2 was **0.67813%**, above `0.1%`. The accepted ratio remains null.

Two complete already-warmed profile steps per implementation were analyzed with `--warmup-steps 0 --breakdown-mode exclusive_global`. Decoder compute dominated; shared output projection/CE was about 133 ms per core per profiled step. The two named embedding all-to-alls totaled only about 0.79 ms per core per step. Their removal did not establish a net embedding speedup or explain a large end-to-end improvement. Overlapping copies, SparseCore work, and multiple TensorCores were not summed as wall time. This is evidence against carrying the 107M speedup over to this workload.

Evidence: [large reference report](evidence/large-reference/compare-float32.json). Large raw profile captures remain external; this archive keeps the later matched-Splash profile aggregates and the profile-analysis source. This earlier capture used the reference attention path and must not be conflated with the later common Splash comparison.

### 2026-10-23 — Common-state FP32 control

To distinguish forward/gradient disagreement at the same state from independent training drift, the diagnostic at `ad2bc20438796c63415240ed1979e93d212fc6da` compared both implementations from common states at steps 0 and 25. Losses were exact and gradients passed. At state 25, Adam and update comparisons passed strict elementwise checks. At state 0, near-zero embedding-gradient coordinates were amplified by Adam's normalization. A float64 reconstruction reproduced selected first-update differences to about `2.6e-9` absolute residual.

Those coordinates are a biased worst-case sample. They explain that selected local amplification, not the entire loss trajectory, and do not establish that the two independently trained state sequences are interchangeable. This negative boundary prevented repeatedly rerunning the same common-state experiment as though it could resolve the full trajectory.

Evidence: [common-state records](evidence/common-state/), [analysis](analysis/native-jax-commonstate-analysis.json), and the retained diagnostic runner.

### 2026-10-23 20:22–20:30 UTC — Two-host checkpoint and nested-rematerialization checks

A separate native-only migration gate exercised ordinary Trainer checkpoint/resume on two hosts and eight v5p chips, with data 4 / model 2. Both ranks exited successfully for the ordinary run at `b8c47e8eea0d39412e3631404588b4f25ce4e45c` and the nested-rematerialization run at `f659101960398fe9ad104e66600c0ccf3a2f34c8`. The latter uses `nested=2, save_carries=false, save_inputs=true`.

Each runner asserts exact local leaf structure, dtype, global shape, placement, local shard indices/shape, and byte SHA equality between the resumed and uninterrupted final states; all ranks must agree. Three completed checkpoints and equal step/counter values are required. Recovered metadata independently confirms three complete manifests of 42 arrays, including 12 model arrays, with `18,372,452,380` logical array bytes per checkpoint. The approximately 55 GB of checkpoint arrays per run stayed in GCP.

There is an artifact gap: the terminal attempt descriptions exposed no output archive, and concrete per-rank result/shard JSON was not recovered. The successful pinned runner assertions and both exit-zero outcomes support the bounded checkpoint result; manifests prove structure and model semantics. They do not independently expose concrete per-rank digests or physical device-kind mappings. OCDBT blob checksums are not logical tensor hashes. This is not legacy parity, throughput, or long-convergence evidence.

Evidence: [ordinary summary](evidence/checkpoint/checkpoint-summary.json), [nested summary](evidence/checkpoint/nested-checkpoint-summary.json), exact historical runner snapshots, and preserved historical Git commits.

### 2026-10-23 20:54–21:14 UTC — Matched Splash attention and the mask metadata fix

The next comparison selected the same Splash backend in both paths. Backend selection at `0acafd210b88c22011a5808b9c4b68d8eb088f25` alone exposed a native bounds failure. Commit `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` fixes shared mask-metadata sharding to local attention heads; it does not change kernel arithmetic.

Actual common-input 107M attention/RoPE replays found matching named/native results for the sampled head-64 layers. Separate materialized BF16 operator graphs could differ even within the named implementation; those controls did not establish a native primitive bug. Projection replay localized tiny FP32-master Q-gradient differences without producing a general fix.

The corrected full 107M Splash benchmark at `36e` passes **both** precisions and the independent canonical/raw/normal-state physical-layout audit. FP32 medians were `42.1078 / 28.6872 ms` (1.46783×); BF16 medians were `25.0174 / 11.9679 ms` (2.09038×). All 604 recorded BF16 error records have zero numerical difference. Aggregate zero-error reports do not prove signed-zero bit identity; that earlier wording was corrected. These measurements still cover a small head-64 model, not the large head-128 workload.

Evidence: [107M summary](evidence/splash-107m/summary.json), adjacent original reports, [attention replay analysis](analysis/native-jax-bf16-attention-replay-analysis.json), and [projection analysis](analysis/native-jax-bf16-projection-replay-analysis.json).

### 2026-10-23 21:41 UTC — Large matched-Splash comparison still fails

The large full Trainer experiment at clean `36e` uses the matched Splash backend, fixed large initial hash `d79fb8c383f9af209fd9e3e32e00798c4aeb460d779a1444d9ea20ba9f453f9a`, and batch hash `d65acfa301bad7db399ff9df86cbbc150c649317dd357b3a26c53c9eaea9caa4`. All canonical/raw physical placement categories and normal first-state placements match.

| Precision | Raw legacy / native median | Numerical result |
| --- | ---: | --- |
| FP32 highest | 1032.477849 / 1035.345410 ms | First loss, all 255 gradients, Adam, and approved update/parameter gates pass; 20-loss-vector L2 **0.366196%** fails `0.1%`. |
| BF16 default, FP32 masters | 286.208230 / 277.657764 ms | Diagnostic first loss exact; **26/255** gradient tensors fail, with embedding Adam-mu/update/parameter and trajectory failures; loss-vector L2 **3.64930%**. |

Accepted ratios remain null. The profiles again show decoder work dominating. Legacy embedding all-to-alls occupy about 0.786 ms FP32 / 0.399 ms BF16 per core per profiled step. CPU dispatch instrumentation suggests named-wrapper/Equinox-flatten overhead but perturbs timing; it is not a measured full-TPU causal attribution.

Evidence: [large matched reports](evidence/splash-15b/), [numerical readout](analysis/native-jax-splash-15b-numerical-analysis.md), and [profile readout](analysis/native-jax-splash-15b-profile-analysis.md). Timings were collected before profiling and remain separate from capture overhead.

### 2026-10-23 — Auditing the normal no-hook backward path

The BF16 failure is not solely an artifact of the diagnostic executable. Retained normal first-step Adam-mu has nonzero cross-variant error in 248/255 tensors, and the embedding tensor alone fails its elementwise gate (maximum about `0.000634766`, relative L2 about `1.26673%`). Normal Adam-nu has nonzero error in 248 tensors but passes its gate. Cross-variant counters match; concrete count values were not retained in that audit.

All nine last-block gradient tensors pass the diagnostic tolerance, while four—up, output projection, Q, and inputnorm—have nonzero errors. First-step FP32 mu is approximately `0.1*g` after rounding, so the same absolute tolerance is roughly ten times looser in gradient units. Passing moments cannot establish equal gradients. Cross-variant error-dictionary equality also cannot prove intravariant diagnostic/normal tensor identity. The first normal scalar loss was not separately retained, and exact normal gradients cannot be reconstructed from the moment aggregates.

A static normal-HLO scan found no literally repeated SSA operand inside a single all-reduce instruction. That is a narrow negative, not proof against semantic reduction issues or compiler defects.

Evidence: [normal backward records](analysis/native-jax-splash-15b-normal-backward-evidence.json), [depth summary](analysis/native-jax-splash-15b-gradient-depth-evidence.json), and [source-boundary audit](analysis/native-jax-normal-trainer-source-boundary-audit.json).

### 2026-10-23 22:34 UTC — A separate legacy graph-context failure

An initial block-capture guard failed because the combined diagnostic graph's legacy full-block output did not match the actual scan. The follow-up held the original source operands and hashes fixed and compared compilation contexts. Separately compiled full and manually composed blocks match the scan. In combined graphs, the full branch differs by relative L2 **16.7801%**, maximum **4.19921875**, while the manually composed branch remains exact. A forward-only combined graph reproduces the failure; cotangent branches are unnecessary.

StableHLO comparisons and optimized-HLO inspection identify context-dependent multi-output RMSNorm/reduction lowering, but no first differing runtime intermediate or incorrect arithmetic has been established. This legacy-only diagnostic failure is distinct from the native full-Trainer failure and must not be presented as its explanation. It justified capturing the full reference separately before attributing later VJP differences.

Evidence: [graph-context comparisons](evidence/legacy-graph-context/comparisons.json), [audit](analysis/native-jax-bf16-splash-15b-graph-context-artifact-audit.json), selected source/optimized HLO, and the frozen forward-context runner.

### 2026-10-23 23:10 UTC — Separate-reference large-block VJP localization

The corrected diagnostic compiles the actual full source block in a separate graph, then reuses the frozen large layer-27 operands and cotangents. Initial, batch, five captured operand hashes, finite values, and physical placements pass. All 33 within-variant full/composed output hashes match for named, native BF16, and the highest-FP32 control. This is fragment identity under its controls, not identity of the full Trainer backward graph.

The fixed-source MLP tail matches for all 11 output records. The attention prefix already differs: input VJP relative L2 **0.0588515%**, Q-master VJP **0.457671%**, and inputnorm-master VJP **0.565345%**. Maximum absolute errors are `3.73e-9 / 4.77e-7 / 9.54e-7` respectively, all below the original gradient absolute tolerance. Thus these fragment errors localize nonidentity; they are not themselves a strict gradient-gate failure. Full-block inputnorm context changes and tiny down/K/V differences also remain. The normal Trainer's last-block up/O pattern is not fully reproduced.

Both source StableHLO paths scale raw Splash dQ before inverse RoPE. Optimized native HLO instead rounds `scale*cos` and `scale*sin` before multiplying raw dQ; named first rounds `dQ*scale`. Both use the same BF16-rounded `1/sqrt(128) = 0.08837890625`. The shared private Splash function texts are byte-identical. Head 64 used the exact power-of-two scale `1/8`, so earlier small replay success did not exercise this head-128 rounding boundary.

These observations support a compiler-reassociation hypothesis. They do not quantify the effect from common captured raw dQ/trig values, establish an incorrect compiler transformation under its permitted arithmetic, or identify every full-model divergence. The fixed named residual/cotangent staged controls conflate boundary substitution, dtype, materialization, and fusion. The highest-FP32 control is a different compute dtype/precision control, not exact-real-arithmetic ground truth.

Evidence: [separate-reference records](evidence/separate-vjp/), [numerical analysis](analysis/native-jax-bf16-splash-15b-separate-vjp-packed-numerical-analysis.md), [independent audit](analysis/native-jax-bf16-splash-15b-separate-vjp-packed-independent-audit.json), and retained prefix HLO.

### 2026-10-23 23:46 UTC — BF16 query-scale barrier A/B

Candidate `8aa8674d10894728ed0b455ae14e14f46457cf9f` puts `optimization_barrier(q_)` **before** forward query scaling for BF16. Reverse AD then places the boundary after scaled dQ and before inverse RoPE. This is a bounded lowering constraint; FP32, tolerances, learning rate, model, and optimizer arithmetic remain unchanged.

The A/B ran baseline `36e` and candidate `8aa` sequentially on one physical v5p-8 slice with the actual 1.53B layer-27 operands. Each child imported the exact intended source bytes in a fresh process; all 11 source-file hashes, barrier-function presence, source guards, physical placements, and frozen assets were verified. Baseline reproduced the saved full/prefix forward, input-VJP, and Q-VJP hashes. Candidate restore was verified.

The prefix now matches named for all 11 output hashes. Full/composed remove input, Q, inputnorm, and V-master mismatches; the unchanged 48-element K and 680-element MLP-down differences remain. All 44 named and all 44 highest-FP32 result hashes, and all four native forward hashes, are unchanged; no new nonexact output appears. Optimized HLO now materializes BF16 `dQ*scale` before separate inverse-RoPE products, as intended.

This supports the rounding-boundary explanation for the removed fragment differences and warrants a matched full Trainer test. It does not establish full backward identity, BF16 trajectory acceptance, speed, or convergence. A common raw-dQ two-order control was not performed. Independent static inspection also confirms analogous reassociation in the **normal FP32 Trainer** HLO, but the BF16-only candidate does not test an FP32 barrier.

Evidence: [A/B controls](evidence/query-scale-ab/ab-summary.json), [independent numerical/HLO audit](analysis/native-jax-query-scale-ab-independent-audit.json), selected before/after HLO, and [normal FP32 static audit](analysis/native-jax-fp32-qscale-static-independent-review.json).

### Prepared next experiment — Full Trainer candidate validation

The next full 1.53B matched-Splash Trainer runner and material config are preserved **before execution**. At this notebook entry it is prepared, not submitted. It keeps the baseline weights/batch, all approved criteria, optimizer settings, five warmups and 20 timed steps. BF16 retains two post-timing profiled steps; FP32 captures no new profile because the candidate's branch is BF16-only. Both normal first-state and diagnostic checks remain enabled.

The V2 CPU proof checks actual fresh child module paths, all 11 source byte hashes, barrier-function presence, and fresh bytecode isolation. Its small CPU FP32 case passes; the vanilla-CPU BF16 strict failure and exit-one result remain intact. Actual large abstract graphs pass, and the archive fixture is bounded and complete. These checks establish runner/import/archive mechanics, not TPU numerical correctness. The exact prepared runner SHA is `dbcbc79aac5bde4e384521949bfa0b7193f1726d7a85a8a117374d9ee3b69361`.

A compact public-API RoPE/Splash gradient regression prototype is also retained. Its head-128 and head-64 forward/reverse abstract graphs pass on CPU for both sources; it has not run the Splash kernel on TPU or shown fail36e/pass8aa numerical behavior. It must reproduce the actual failure under unchanged tolerances before becoming a checked-in regression.

Evidence: [prepared config/proofs](prepared/), [V2 runner](runners/native_jax_query_scale_15b_trainer_v2.py). The next decision depends on the actual full Trainer result. No accepted 1.53B speed ratio is established here.

## Current decision and remaining work

The [replacement gap audit](analysis/native-jax-llama-replacement-gap-audit.md) maps canonical registries, production callers, existing Llama3/Yarn presets, paged decoding, loader boundaries and other model families that import the old Llama classes. Switching the benchmark or a single registry entry is insufficient. The audit distinguishes source already implemented on sibling branches from work still needed, and adds no new numerical or convergence threshold.

Retain the compile-time RoPE and Splash mask-metadata fixes. Do not adopt the rejected attention-order or Q-compute-view patches. The BF16 query-scale barrier is promising under a controlled large attention-prefix A/B and remains a candidate until full Trainer correctness, performance, and meaningful regression coverage pass. Do not replace the old Llama entry points or delete the reference implementation on the strength of small-model parity or one successful fragment.

The benchmark uses a shared fused-loss adapter for matched comparisons. Its results do not establish performance of the production `train_lm` entry point, which uses a different loss adapter on the native LM wrapper. The production wrapper and checkpoint contributions are separate branches and have not been integrated into the selected `8aa` source tree.

The large loss trajectory, strict BF16 full-model parity, full candidate timing, production-entry-point coverage, broader migration integration, and longer convergence remain open. The distributed native checkpoint result covers its exact bounded setup, with the artifact gap stated above. There is no claim of a completed migration or accepted large-model speedup.

Earlier fork contributions are [native model #2](https://github.com/dlwh/marin/pull/2), [Trainer #3](https://github.com/dlwh/marin/pull/3), and [benchmark #4](https://github.com/dlwh/marin/pull/4). Later contributions are [LM-boundary conversion #11](https://github.com/dlwh/marin/pull/11), [checkpoint runner #12](https://github.com/dlwh/marin/pull/12), [matched Splash backend #13](https://github.com/dlwh/marin/pull/13), and [mask metadata #14](https://github.com/dlwh/marin/pull/14). These are contribution links, not merge claims; current lifecycle belongs in the coordinating issue. Archive-only history merges, workflow removal, notebook, abandoned alternatives, and reproduction scripts stay out of production extraction.

The source, evidence, and next-run preparation should be committed and pushed before issue milestones. [The reproduction guide](README.md) and `manifest.json` define the retained byte-level inventory. Future findings should append entries that correct or refine this record rather than erase failed hypotheses.
