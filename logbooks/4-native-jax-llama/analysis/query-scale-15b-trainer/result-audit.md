# Independent full-Trainer query-scale audit

**Rejected: both numerical comparisons fail. No speed ratio or adoption is valid.**

Source `8aa8674d10894728ed0b455ae14e14f46457cf9f`; full Trainer, 1,531,037,696 parameters, Splash, batch 8 × sequence 2048, 28 layers, data 2 × model 2. Saved terminal evidence: failed, task exit 1, one failure, zero preemptions. Both dtype children completed with numerical exit 1; archive backup succeeded.

Archive independently verified: 7,953,898 bytes, SHA256 `dcef04f7557f65d95c1ed593acc43e6843d320160ff5c68f6c5e1d0209d81586`, 18 safe regular files, 44,676,180 raw bytes; each member byte-exact to extracted files. All five frozen preparation hashes, runner hash, encoded launcher payload, and 11 source hashes match. Same-child imports point to `/app` candidate modules, with distinct fresh pycache prefixes and bytecode writes disabled. Remote Git cleanliness/pin is declared; local pinned source and clean status were independently verified.

Requested v5p-8; actual probe records one process and four global/local TPU v5 devices (8 cores). Both dtypes independently pass all frozen matching guards and all 2,300 normal/diagnostic physical tensor placements across 12 groups; those canonical manifests also exactly match baseline. Auto → Explicit is the intended axis-control difference.

Baseline is `/private/tmp/native-jax-splash-15b-profile-results`, actually declaring source `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`. Parameter/native-import SHA256 `d79fb8c383f9af209fd9e3e32e00798c4aeb460d779a1444d9ea20ba9f453f9a` and batch SHA256 `d65acfa301bad7db399ff9df86cbbc150c649317dd357b3a26c53c9eaea9caa4` agree exactly. Resolved workload, optimizer, LR, versions, precision, device topology, and XLA flags match. Only source diff is the BF16-only barrier before query scaling. FP32 profile capture changed 2 → 0 steps after timing; BF16 stayed 2. Baseline lacks the new runtime-import hash record.

## Numerical gates

Original `atol=1e-5, rtol=1e-4` apply everywhere for BF16. FP32 retains original elementwise loss/gradient/Adam checks; every update/parameter tensor must have relative L2 ≤ 0.005 and the 20-loss vector ≤ 0.001. No threshold changed. Failed tensor counts below are elementwise counts, including FP32 groups accepted by their approved L2 criteria. L2 is the largest per-tensor value, not an aggregate model norm.

### bfloat16

| Comparison | Failed tensors baseline → candidate | Candidate max abs | Max tensor L2 baseline → candidate | Accepted |
|---|---:|---:|---:|:---:|
| adam_count | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| adam_mu | 1 → 1 / 255 | 4.88283113e-05 | 0.0171986353 → 0.000207317629 | no |
| adam_nu | 0 → 0 / 255 | 9.75132934e-08 | 0.0207457151 → 0.000163976423 | yes |
| execution_adam_count | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| execution_adam_mu | 1 → 1 / 255 | 4.88283113e-05 | 0.0171986353 → 0.000207317629 | no |
| execution_adam_nu | 0 → 0 / 255 | 9.75132934e-08 | 0.0207457151 → 0.000163976423 | yes |
| execution_parameters_after_one_step | 248 → 73 / 255 | 0.000999665819 | 0.0979078412 → 0.000638020458 | no |
| gradients | 26 → 2 / 255 | 0.00048828125 | 0.0171984304 → 0.000207313307 | no |
| loss | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| measured_step_losses | 1 → 1 / 1 | 0.722797394 | 0.0364929885 → 0.0362326987 | no |
| parameters_after_one_step | 248 → 73 / 255 | 0.000999665819 | 0.0979078412 → 0.000638020458 | no |
| updates | 248 → 74 / 255 | 0.000999665819 | 0.13960743 → 0.000909757568 | no |

### float32

| Comparison | Failed tensors baseline → candidate | Candidate max abs | Max tensor L2 baseline → candidate | Accepted |
|---|---:|---:|---:|:---:|
| adam_count | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| adam_mu | 0 → 0 / 255 | 6.14672899e-08 | 1.54850886e-06 → 1.54850886e-06 | yes |
| adam_nu | 0 → 0 / 255 | 3.78349796e-10 | 1.79819688e-06 → 1.79819688e-06 | yes |
| execution_adam_count | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| execution_adam_mu | 0 → 0 / 255 | 6.14672899e-08 | 1.54850886e-06 → 1.54850886e-06 | yes |
| execution_adam_nu | 0 → 0 / 255 | 3.78349796e-10 | 1.79819688e-06 → 1.79819688e-06 | yes |
| execution_parameters_after_one_step | 63 → 63 / 255 | 0.000845920469 | 0.000205737189 → 0.000205737189 | yes |
| gradients | 0 → 0 / 255 | 6.10947609e-07 | 1.54807378e-06 → 1.54807378e-06 | yes |
| loss | 0 → 0 / 1 | 0 | 0 → 0 | yes |
| measured_step_losses | 1 → 1 / 1 | 0.0926728249 | 0.00366195687 → 0.00366195687 | no |
| parameters_after_one_step | 63 → 63 / 255 | 0.000845920469 | 0.000205737189 → 0.000205737189 | yes |
| updates | 77 → 77 / 255 | 0.000845920469 | 0.000293347053 → 0.000293347053 | yes |

## Remaining BF16 gradients

| Tensor | Max abs | Mean abs | Relative L2 |
|---|---:|---:|---:|
| `model.embed_tokens.weight` | 0.00048828125 | 5.63195445835e-09 | 0.000207313307328 |
| `model.layers.0.self_attn.o_proj.weight` | 1.52587890625e-05 | 6.9268146774e-11 | 3.57813551091e-05 |

BF16 Adam μ fails only on embeddings: max abs 4.88283112645e-5, L2 0.000207317629247. Adam ν and counts pass. No-hook execution-state metrics exactly equal diagnostic counterparts in both dtypes.

BF16 20-loss vector: max gap 0.722797393799 at measured step 18; 15/20 values fail elementwise; relative L2 0.0362326987088 (baseline 0.0364929884672). Final legacy/native losses are 7.94388532639 / 7.61148977280. Legacy baseline/candidate loss trajectories are exactly equal; native trajectories differ. The full vectors and signed gaps are retained in the JSON audit.

FP32 is an unchanged-path control: the entire per-tensor parity object, acceptance object, initial comparison and both 20-step trajectories are exactly equal to baseline. Its only rejected gate remains trajectory L2 0.00366195687093 > 0.001; max gap 0.0926728248596 at measured step 19. First-step update/parameter L2 gates pass.

Raw synchronized mean step times in seconds (legacy/native): FP32 1.04216422240 / 1.04678600645; BF16 0.288179574250 / 0.281409721650. **All speed ratios remain null** because both numerical comparisons are invalid.

## Evidence boundary and next decision

Per-tensor gradient/optimizer/parameter errors come from retained benchmark reports; raw arrays were not retained, so this audit cannot independently recompute their elementwise errors or coordinate mismatch counts. It independently recomputes aggregate summaries, gate logic, physical manifests, hashes and the 20-loss vector metrics. Profiles were unnecessary and were not parsed. Observed baseline-to-candidate improvements support further localization; they do not establish mechanism or convergence.

Keep full-Trainer candidate rejected. Localize the two surviving BF16 gradient discrepancies (embedding and block-0 o_proj) using exact matched common operands and full-graph/compiler context before authorizing another bounded test. Treat unchanged FP32 trajectory failure as a separate unresolved control. No retry or adoption is justified by this result alone.

No downloads, remote queries, jobs, retries, tolerance changes or source edits occurred during this audit. Audit egress: zero; supplied conservative campaign ledger including reserve: 1,078,401,174 bytes / 15 GB.
