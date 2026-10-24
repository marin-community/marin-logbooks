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

## 2026-10-24 00:09 UTC — Full Trainer candidate submitted

The source, exact scientific runner, command/configuration and validation proofs were published at `61403c428985aa1aa63ff35a779eb4d5ac40de4e` before dispatch. After verifying the unique job and regional backup object were absent, the exact-once wrapper submitted `/dlwh/levanter-native-jax-v5p-15b-queryscale-8aa-20261023` at 00:09:51 UTC. Iris returned the matching canonical job ID and submission exited zero. The selected source remains `8aa8674d10894728ed0b455ae14e14f46457cf9f`.

The [launch record](evidence/query-scale-15b-launch.json) and [submission guard](runners/submit_native_jax_query_scale_15b_trainer_v2.py) preserve execution identity. At this entry no allocation, progress, numerical outcome or timing has been observed. Each precision compares native JAX with the Haliax Llama reference. The candidate preserves BF16 query-gradient scaling as a separate rounding boundary before inverse RoPE; its FP32 path is unchanged. Both retain the [prepared workload and acceptance criteria](prepared/query-scale-15b-trainer-v2-job-config.json). Monitoring uses the existing campaign heartbeat; a timeout or queue wait does not authorize a retry or cluster change.

## 2026-10-24 00:31 UTC — Compact public-API regression A/B prepared

The compact BF16 RoPE/Splash regression now has a frozen controller, API command/configuration, pure-build request proof, CPU proof and independent preparation review. Its immutable preparation manifest has SHA256 `7a06a2d772d2b3d9c740d52c8d09a083f1709e6e2549270c2f6bec0f6c6eb8e4`. This preparation was not submitted and supplies no TPU numerical result.

The controller compares baseline `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` with BF16 candidate `8aa8674d10894728ed0b455ae14e14f46457cf9f` in sequential fresh children. It freezes batch 4, sequence 512, 16 query / 4 KV heads, head dimensions 128 and 64, FP32 masters, BF16/default compute, seed 173, data2/model2 and the original elementwise `atol=1e-5, rtol=1e-4`. The unit-RMS cotangent sets this derivative diagnostic's scale; it does not change production tolerances. A compact regression claim requires an observed head-128 baseline failure and candidate pass together with head-64 passes for both sources. The controller preserves nonreproducing outcomes rather than searching seeds or thresholds.

The actual four-CPU proof covers abstract forward/reverse graphs, all 13 source-file hashes per variant, fresh imports, barrier presence, source restoration and bounded JSON archive mechanics. Its host input hashes reconstruct immutable seed generation; they do not intercept runtime device operands. No Splash numerical kernel or TPU compilation was exercised. The prototype is reused from the existing archive rather than copied again; generated CPU fixture archives and arrays remain outside this selection.

The prepared API request specifies one v5p-8 allocation, 16 CPUs, 64 GiB memory, 16 GiB disk, `us-central1/us-central1-a`, a 120-second scheduling limit, a 780-second execution limit and zero failure/preemption retries. The diagnostic archive permits at most 24 JSON files and 1 MiB per file; aggregate raw bytes and compressed archive bytes are each capped at 16 MiB. This is a reviewed preparation, not launch or recovery authority.

Evidence: [frozen preparation](prepared/splash-rotary-regression-ab-prepared.json), [API command/configuration](prepared/splash-rotary-regression-ab-command.json), [pure-build API request](prepared/splash-rotary-regression-ab-api-request.json), [CPU proof](prepared/splash-rotary-regression-ab-cpu-proof.json), [independent review](analysis/native-jax-splash-rotary-regression-ab-independent-review.json), and exact [controller](runners/native_jax_splash_rotary_regression_ab.py.source) / [API source](runners/submit_native_jax_rotary_regression_ab.py.source). The reproduction guide explains the frozen source suffix and historical paths.

## 2026-10-24 00:31 UTC — FP32 query-scale candidate prepared locally

Local candidate `f1e991207ceda752957938b1cfe8893027d3fed8` starts from baseline `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` and adds a four-line FP32-only query-scale barrier. Its BF16 path retains baseline behavior; this candidate does not combine the BF16 `8aa` change. An archive-only `ours` merge retains its exact history without changing the archive's selected `8aa` source tree. This is a local source commit, with no new fork PR or remote publication claimed by this entry.

The existing independent normal-FP32 HLO audit motivates testing the boundary: native materializes FP32 `scale*cos/sin` before raw Splash dQ multiplication, while named materializes `dQ*scale` first. That observation does not quantify a common raw-dQ/trig control or establish the cause of the failed full loss trajectory. The prepared design requires both an actual common-source FP32 prefix A/B and a same-raw-dQ/trig two-order diagnostic before considering another full Trainer comparison.

The preparation summary records 43 focused CPU test passes, two skipped real-TPU Splash cases, required mechanical checks and normal commit-hook success. Actual four-CPU abstract value/three-VJP graphs cover the large head-128 shapes: FP32 barrier primitive counts change from zero to two; BF16 jaxpr text hashes remain identical with zero barriers. These are source, tracing and CPU checks. No candidate TPU compilation, numerical A/B, optimized TPU HLO, full Trainer result, accepted speed ratio or convergence result exists in this preparation.

Evidence: [preparation summary](prepared/fp32-query-scale-prepared.json), [experiment design](analysis/native-jax-fp32-query-scale-experiment-design.md), [baseline](prepared/fp32-query-scale-abstract-baseline.json) / [candidate](prepared/fp32-query-scale-abstract-candidate.json) abstract proofs, exact [helper](runners/native_jax_fp32_query_scale_abstract.py.source) and [patch](prepared/fp32-query-scale-rounding.patch.json). The [static HLO audit](analysis/native-jax-fp32-qscale-static-independent-review.json) was already retained and is reused byte-for-byte.

## 2026-10-24 00:31 UTC — CPU-validated production-support integration retained

Local tree `be8093b10163be50b8f8e16135ccb4fccdc4efe5` combines the existing native LM training and checkpoint-support contributions with mask and checkpoint-metadata preservation. It is based on `36e`, leaves the old canonical registry unchanged, and does not contain the `8aa` BF16 barrier. Its exact source history is retained through another archive-only `ours` merge. Neither merge selects this integrated tree as the archive implementation, and this local tree is not a new fork PR or a canonical Llama cutover.

The retained validation summary records 64 focused CPU test passes, followed by four separate local Transformers consumer passes covering tied/untied weights and default/linear RoPE. The checkpoint CLI uses four layers, 8,848 parameters, one process and four CPU devices with data2/model2 and nested rematerialization. It records three checkpoints and 168 local shard comparisons; uninterrupted and resumed full-state SHA256 hashes both equal `2cae6fbc6dde736b6a30bc477321715d960917970a206448f88fa99f12f5482a`. Only the configuration, import/source proofs and result summary are retained here, not checkpoint arrays or raw logs.

This closes a bounded local integration check for public training, local HF loading/export and checkpoint paths. It does not establish multi-process or accelerator behavior, production performance, Llama3/YaRN support, paged decoding, remaining loader/caller coverage, legacy removal or migration acceptance. New rotary feature work is outside this checkpoint.

Evidence: [source integration proof](analysis/native-jax-replacement-source-integration-proof.json), [validation summary](analysis/native-jax-replacement-integration-validation.json), [checkpoint result summary](evidence/replacement-integration/smoke-validation.json), [exact smoke configuration](prepared/replacement-integration-smoke.yaml), and [actual import proof](prepared/replacement-integration-import-proof.json).

## 2026-10-24 00:31 UTC — Correction: execution timeout excludes queue time

Earlier campaign wording described the full Trainer's 1,800-second timeout as including queue time. That description was incorrect. At selected source `8aa`, Iris selects an attempt's `started_at_ms` for execution-timeout checks (`reads.py:1754–1777`) and compares `started_at_ms + timeout_ms` with the controller clock (`controller.py:1440–1453`). The CLI forwards `--timeout` as the execution timeout and does not set `scheduling_timeout`. Queue wait therefore does not consume that execution cap or gain a scheduling deadline from it.

The already submitted full Trainer job retains its original 1,800-second execution cap. This correction did not cancel or retry the queued job. No live job state was queried while preparing this archive checkpoint. Monitoring and any separately authorized action remain with the campaign owner.

The new compact API preparation explicitly separates 120 seconds for scheduling from 780 seconds for execution, with zero retries. Their configured sum is 900 seconds before reconciliation/finalization latency, not an exact wall-clock deadline. It remains prepared and unlaunched. Explicit scheduling deadlines are also checked for queued federation handoffs. The [pinned timeout source evidence](analysis/native-jax-iris-timeout-source-evidence.json) retains the exact relevant source excerpts; original preparation and launch records remain unchanged.

## Current decision and remaining work

The [replacement gap audit](analysis/native-jax-llama-replacement-gap-audit.md) maps canonical registries, production callers, existing Llama3/Yarn presets, paged decoding, loader boundaries and other model families that import the old Llama classes. Switching the benchmark or a single registry entry is insufficient. The audit distinguishes source already implemented on sibling branches from work still needed, and adds no new numerical or convergence threshold.

Retain the compile-time RoPE and Splash mask-metadata fixes. Do not adopt the rejected attention-order or Q-compute-view patches. The BF16 query-scale barrier is promising under a controlled large attention-prefix A/B and remains a candidate until full Trainer correctness, performance, and meaningful regression coverage pass. Do not replace the old Llama entry points or delete the reference implementation on the strength of small-model parity or one successful fragment.

The benchmark uses a shared fused-loss adapter for matched comparisons. Its results do not establish performance of the production `train_lm` entry point, which uses a different loss adapter on the native LM wrapper. The production wrapper and checkpoint contributions are separate branches and have not been integrated into the selected `8aa` source tree.

The new local FP32 candidate and CPU-validated integration tree are now retained as historical source choices. Their archive-only merges preserve source and evidence without adopting either into the selected `8aa` tree. The compact regression and FP32 numerical designs remain prepared-only; they add no accepted TPU result or speed ratio.

The large loss trajectory, strict BF16 full-model parity, full candidate timing, production-entry-point coverage, broader migration integration, and longer convergence remain open. The distributed native checkpoint result covers its exact bounded setup, with the artifact gap stated above. There is no claim of a completed migration or accepted large-model speedup.

Earlier fork contributions are [native model #2](https://github.com/dlwh/marin/pull/2), [Trainer #3](https://github.com/dlwh/marin/pull/3), and [benchmark #4](https://github.com/dlwh/marin/pull/4). Later contributions are [LM-boundary conversion #11](https://github.com/dlwh/marin/pull/11), [checkpoint runner #12](https://github.com/dlwh/marin/pull/12), [matched Splash backend #13](https://github.com/dlwh/marin/pull/13), and [mask metadata #14](https://github.com/dlwh/marin/pull/14). These are contribution links, not merge claims; current lifecycle belongs in the coordinating issue. Archive-only history merges, workflow removal, notebook, abandoned alternatives, and reproduction scripts stay out of production extraction.

The source, evidence, and next-run preparation should be committed and pushed before issue milestones. [The reproduction guide](README.md) and `manifest.json` define the retained byte-level inventory. Future findings should append entries that correct or refine this record rather than erase failed hypotheses.


## 2026-10-24 00:54 UTC — Scaled-RoPE contribution and exact native checkpoint continuity

The maintained native Llama contribution now supports Llama3 frequency scaling and YaRN's frequency ramp and attention multiplier. This closes a concrete replacement gap: existing `experiments/llama.py` presets select Llama3 RoPE, while the transitional native LM config previously rejected every rotary configuration except default/linear. Feature `598e1968819d3a2a85778841e4a3695ab6d89d8b` is published as [draft contribution PR16](https://github.com/dlwh/marin/pull/16), based on `7b1aab6e2e6837cac10595dd54d1b4d60b7d6361`. Its [publication payload](prepared/scaled-rotary/pr16-payload.json) records the exact title, body, source, base and labels at inspection. The [independent extraction review](analysis/scaled-rotary/maintained-extraction-review.json) classifies all eight changed files as maintained contribution code; it includes no historical experiment harness.

The independent HF review against Transformers 5.12.1 found three consequential semantic mismatches before publication. Missing `attention_factor` can make HF infer an `mscale/mscale_all_dim` ratio that differs from the native multiplier; `truncate:false` changes the correction ramp; and the inherited YaRN upper correction bound used half the head dimension where HF uses the full head dimension. The selected feature validates the effective HF scaling, rejects the unsupported nontruncated form, and corrects the shared named/native long-context range cap. Exported RoPE bases are consistent across HF metadata formats. The [initial findings](analysis/scaled-rotary/independent-review.json), [actual HF semantic probes](analysis/scaled-rotary/hf-semantics-review.json) and [targeted re-review](analysis/scaled-rotary/targeted-rereview.json) preserve the defects and their corrections. Numerical tolerances were unchanged.

The feature's selected CPU run passed 75 cases; the final HF boundary run passed nine, and a separate HF consumer run passed ten. These runs overlap and must not be added as unique cases. Their [compact result summary](evidence/scaled-rotary/cpu-validation-summary.json) retains input XML hashes and per-case outcomes while omitting host attributes and raw logs. The broader affected run remains **1828 passed, 197 skipped, 13 failed**. All 13 failed nodes reproduce on unchanged PR base `7b1aab` with the same nested `ModuleNotFoundError: No module named 'cloud'`. Only four exist on local main `8428b67fba80a1f8bf9dfeabc552f411cbb82be0`; all four also reproduce there. The [baseline audit](analysis/scaled-rotary/affected-baseline-audit.json) distinguishes those scopes. It does not claim a green affected suite. The [lint catalog](analysis/scaled-rotary/catalog-dispositions.md) completed six lanes plus composer: 21 inherited findings across 19 complete files, all unchanged from the actual PR base and outside the eight-file contribution. Every finding is answered; this was not a zero-finding catalog.

The actual local integration is `216279e6d8ea9a1c71242509b6b07e946c30d249`, with normal merge parents `be8093b10163be50b8f8e16135ccb4fccdc4efe5` and `598e196`. One benchmark conflict removed the obsolete scalar `rope_theta` constructor argument while preserving the matched Splash/reference selector. The [combined-tree source proof](analysis/replacement-scaled-rotary/source-proof.json) verifies the selected source and this resolution. All **111 selected cases passed on four CPU devices in 160.17 seconds**, using explicit root `pyproject.toml` safety defaults, including actual HF consumers. Required lint, format and type checks passed. The maintained attention source [36e2a9a97fab22b76dfa5a03964c6600ec5713ac](https://github.com/dlwh/marin/commit/36e2a9a97fab22b76dfa5a03964c6600ec5713ac), numerical acceptance criteria and canonical registry selection remained unchanged. The BF16 query-scale rounding candidate [8aa8674d10894728ed0b455ae14e14f46457cf9f](https://github.com/dlwh/marin/commit/8aa8674d10894728ed0b455ae14e14f46457cf9f) was not adopted in this integration.

A [disposable cross-revision probe](runners/probe_native_jax_cross_revision_checkpoint.py.source) loaded the retained tiny native checkpoint written by `be8093` through both the old source and the combined tree. All **168 local shards** matched the original full-state digest, including model, optimizer, step and RNG. Step and Adam counters remained two; hidden states and logits were bit-exact. Default and linear-factor8 architecture metadata matched exactly. The actual retained arrays use default factor1; factor8 was a schema/semantic check only. This is native-to-native restore evidence, not named-Haliax checkpoint conversion or a next-update test. No checkpoint data was rewritten. The [proof](analysis/replacement-scaled-rotary/cross-revision-proof.json) links [baseline](evidence/replacement-scaled-rotary/baseline-native-restore.json) and [combined](evidence/replacement-scaled-rotary/scaled-native-restore.json) hash records; arrays remain outside Git.

Archive-only history merge `b59358c2dc010bc2ece280e1b70020d9a0d5c539` retains both the maintained feature and actual integration history. It does not select their source trees. The archive's runnable source tree remains exact `8aa8674d10894728ed0b455ae14e14f46457cf9f`, apart from archive content and removed automatic workflows; the [tree proof](analysis/scaled-rotary/archive-source-tree-proof.json) records 6641 source blobs and their normalized tree hash. The original 154 selected artifacts remain byte-exact. Canonical Llama cutover, paged decode/cache and remaining caller/loader migration are unfinished.

### Accelerator state at this archive entry

The last retained observation of `/dlwh/levanter-native-jax-v5p-15b-queryscale-8aa-20261023` at **00:46:09 UTC** was PENDING, with no failures, preemptions or task logs. Its [launch record](evidence/query-scale-15b-launch.json) pins candidate source `8aa8674d10894728ed0b455ae14e14f46457cf9f`. The 00:40 queue diagnostic reported 256 GB host memory requested versus 208 GB available per worker. The 1800-second execution timeout excludes queue waiting. No full candidate numerical, trajectory or speed result follows from that state.

The compact public-API A/B was submitted once at **00:52:37.887092 UTC** as `/dlwh/levanter-native-jax-v5p-rotary-regression-ab-8aa-20261023`, after its exact source/config/controller were retained at `8dbe5ce2b0527989a3c4c91abe94a3b2cf5be28d`. The [launch record](evidence/compact-regression-submitted.json), [one-shot wrapper](runners/submit_native_jax_compact_once.py.source) and [budget proof](evidence/compact-regression-budget-proof.json) preserve that execution identity and bounded request: one v5p-8 slice exposing four physical chips/devices, CPU16, 64 GB RAM, 16 GB disk, scheduling timeout120 seconds plus execution timeout780 seconds, zero failure/preemption retries and a 16 MiB JSON-only artifact cap. The launch and terminal records below preserve the observed compact-job state; this archive preparation did not query the jobs. Direct artifact verification is recorded in the following entry before any numerical success claim.


### 2026-10-24 00:56 UTC — Compact public-API regression verified on TPU

Direct artifact inspection now establishes the compact regression. The job succeeded with task exit0; its 10,604-byte regional backup passed the GCS MD5 check and SHA256 `8fef5201a7cb82cfb48e8966bb322622dc39c582f1fe9c0c5785a54a58c9bb72`. All seven regular relative JSON files (65,820 bytes uncompressed) were inspected. Both 13-file source maps match their exact Git pins; actual fresh imports, the same four physical TPUv5 devices, result layouts, initial operands, source identity and unchanged-reference controls pass. The retained [public-API prototype](runners/native_jax_splash_rotary_gradient_regression.py) and unchanged strict `atol=1e-5, rtol=1e-4` were used.

For B4/S512/H16/KV4/head128, the baseline `36e` query-input VJP (the gradient with respect to the FP32 query input through its BF16 cast) fails the strict gate: max absolute error **0.03125**, relative L2 **0.0030482744 (0.30482744%)**, with **1,251,150 of 4,194,304** elements nonexact. Candidate `8aa` makes that VJP bit-exact with the independent Haliax named-axis RoPE-plus-Splash composition; forward and K/V VJPs are also exact. The scalar reduction loss is allclose but not exact, so this is not an all-output bit-identity claim. The head64 control passes both variants with exact Q/K/V VJPs. Across both head sizes, the checked outputs are the scalar reduction loss, attention forward output and Q/K/V input VJPs; the candidate introduces no additional gate failure.

The [actual reports](evidence/compact-regression/ab-summary.json), [direct audit](analysis/compact-regression/result-audit.json) and [terminal record](evidence/compact-regression/terminal-state.json) verify a focused regression fix for the BF16 query-scale/RoPE composition. The deterministic unit-RMS cotangent measures per-unit derivatives; it does not reproduce full-model mean-CE normalization. This result does not establish the queued full Trainer's gradients, trajectory, throughput or convergence, and it does not adopt the query-scale candidate into the maintained integration. No retry, resubmission, tolerance/seed search, checkpoint transfer or new source change followed this result. The campaign-wide conservative egress ledger, including its inherited 1 GB reserve, remains below the authorized 15 GB budget; the compact backup itself was only 10,604 bytes. The full migration goal remains incomplete.


### Prepared-only FP32 common-raw-dQ control

A separate [frozen preparation bundle](prepared/fp32-common-dq/manifest.json) preserves the FP32 query-scale candidate `f1e991207ceda752957938b1cfe8893027d3fed8` and a same-materialized-raw-dQ/trig diagnostic. Its thirteen original bundle files total 201,019 bytes. The [CPU4 controller summary](analysis/fp32-common-dq/cpu-controller-summary.json.gz) preserves all sixteen passing cross-child mechanics guards but **scientific_attribution_valid is false**: separately materialized scaled-Q forward differs by max `2.384185791015625e-7`, and named source versus replay full forward differs by max `7.152557373046875e-7`. These exact-identity failures remain negative controls; no tolerance was changed and no attribution or improvement follows.

The [actual-large abstract summary](analysis/fp32-common-dq/abstract-controller-summary.json) traces the 1.53B FP32 Splash graphs without TPU compilation or execution. This preparation is local-only and still requires independent review before hardware. The archive adds the frozen source and compact summaries, not the 7.5 MB of local HLO diagnostics. It grants no submission authority and does not change the selected runnable source tree. [Restoration instructions](README.md#restore-the-prepared-only-fp32-common-rawdq-bundle) explain the archive-only Python suffixes.


## 2026-10-24 01:05 UTC — Full query-scale candidate rejected despite partial BF16 improvement

This terminal result supersedes the preceding pending full-Trainer snapshots. `/dlwh/levanter-native-jax-v5p-15b-queryscale-8aa-20261023` completed both precision comparisons and failed its unchanged numerical gates, with task exit1 and zero preemptions. The [terminal record](evidence/query-scale-15b-trainer/terminal-record.json) is derived from saved direct-audit evidence, not a new job query. Both numerical reports have `comparison_valid:false`; all speed ratios remain null. No retry or source adoption followed.

The candidate is the BF16-only query-scale rounding barrier at `8aa8674d10894728ed0b455ae14e14f46457cf9f`, compared with maintained attention source `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`. The 1,531,037,696-parameter workload remained L28/D2048/MLP5632/H16/KV4/head128/V65536, global batch8 × sequence2048, mesh axis sizes data2/model2, AdamW, five warmups and twenty timed full Trainer steps on one v5p-8 slice exposing four physical TPUv5 devices. FP32 used highest matmul precision and BF16 the default. FP32 captured no extra profile steps; BF16 captured two after timing. For each dtype, the comparison pairs the existing Haliax Llama with NativeLlama on the same four concrete devices. Initial parameter and batch hashes, resolved configuration and versions matched the separately retained [36e matched-Splash baseline comparison](evidence/splash-15b/); separate baseline/candidate jobs matched hardware type, topology and canonical partition layouts. Local device IDs do not establish reuse of the same physical slice across jobs. The [independent audit](analysis/query-scale-15b-trainer/result-audit.md) verifies eleven runtime source hashes/imports in each actual benchmark child, all frozen matching guards and 2,300 canonical normal/diagnostic tensor-placement manifests per dtype. Placement checks compare physical tensor placement while allowing the intended legacy Auto versus native Explicit control metadata.

BF16 first-step gradient tensors failing the original elementwise `atol=1e-5, rtol=1e-4` decrease from **26 to 2 of 255**. The survivors are `model.embed_tokens.weight` (max absolute error `0.00048828125`, relative L2 `0.0002073133073`, or **0.0207313%**) and layer0 `o_proj.weight` (max `0.0000152587890625`, relative L2 `0.00003578135511`, or **0.00357814%**). Embedding Adam μ still fails in both diagnostic and ordinary no-hook execution; Adam ν and counts pass. Elementwise next-parameter failures fall from 248 to 73 of 255, and update failures from 248 to 74. The BF16 twenty-loss vector still fails: relative L2 **0.0362326987 (3.62327%)**, versus baseline **0.0364929885 (3.64930%)**; its maximum absolute gap is `0.7227973938`. BF16 retains every original elementwise gate; these L2 values describe errors and are not alternative BF16 acceptance criteria.

FP32 is the unchanged-path control: its entire per-tensor parity object matches baseline, and each implementation's twenty-loss trajectory exactly matches its own corresponding prior-run trajectory. The legacy and native trajectories are not equal to each other. First-step loss, gradients, Adam state and approved per-tensor update/parameter gates pass. The remaining FP32 trajectory relative L2 is **0.00366195687 (0.366196%)**, exceeding the approved **0.001 (0.1%)** limit, so that comparison also fails. The raw timing samples are retained, but numerical rejection prevents an accepted performance comparison.

The bounded regional backup was 7,953,898 bytes, SHA256 `dcef04f7557f65d95c1ed593acc43e6843d320160ff5c68f6c5e1d0209d81586`; all eighteen regular relative members matched the extracted bytes. Git retains both complete JSON reports, runtime/probe/guard records, all eight HLO/StableHLO files and the independent audit, with lossless gzip where needed. [Restoration instructions](README.md#inspect-the-rejected-full-trainer-candidate) and original hashes preserve exact bytes. The two BF16 XPlanes remain local/regional and are not in Git. Per-tensor gradient/optimizer errors are independently summarized from the reports; raw arrays were not retained, so coordinate errors cannot be recomputed from this archive. Loss-vector metrics were recomputed from all twenty saved losses.

This is a partial targeted gradient improvement, not accepted full-model parity, a proof of sufficient mechanism, convergence or migration completion. The [compact public RoPE-plus-Splash derivative regression](analysis/compact-regression/result-audit.json) remains valid within its narrower boundary. The next scientific question is the two surviving BF16 gradients and their full-graph context; the unchanged FP32 trajectory remains a separate unresolved control. No tolerance, LR, model or seed was changed to force acceptance. The archived benchmark runnable source pin remains `8aa8674d10894728ed0b455ae14e14f46457cf9f`, not an accepted maintained-model adoption. The maintained integration remains separate. All two hundred artifacts in the prior archive collection at `944004a3b5157ff618012a682953f3a7a3016400` remain byte-exact. The main migration goal is still incomplete.
