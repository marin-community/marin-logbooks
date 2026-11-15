# Exp2 SFT data accounting — CPU accounting receipt content

- **What it is:** the realized-sample accounting for exp2 SFT (artifact `2026.11.15.19`, CPU continuation of the sealed round-2 loop): totals, preservation/reconstruction split, hashes, declarations and measured rates.
- **Compiled at:** 2026-11-15T15:14Z; round-7 cold-read corrections applied 2026-11-15.
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md` (L17, 21, 23, 25, 27); [R] `.../artifacts/experiment2/accounting19-final-verification.json`; [R] `.../artifacts/experiment2/completion19-receipt.json`; [R] `.../artifacts/experiment2/smoke4-validation/resolved-plan.json` (component declaration); [R] `.../artifacts/experiment2/full1{0,1}-consumed-audit.json` (leaf declarations via STATE.md L1563/1575); [R] `.../artifacts/experiment2/population-262k-final/truncation-census.json`; [R] `recovery13-postprocess{17,18}-accounting-prefix-backup.log`, `recovery13-postprocess18-prefix-parity-second.log`; [M] W&B SFT run referenced below (URL requires authentication).
- **Access class:** aggregate token/row counts, hashes, rates and configuration counts only. No dataset rows, task text, prompts or per-row token payloads are reproduced. The **full 375-row per-source aggregate table is published** in [per-source-accounting.md](per-source-accounting.md) (it is aggregate metadata, not data); the private S3 accounting root holds the per-row payloads.
- **Artifact naming (round-7 correction):** the **70,562,321-byte `realized-samples.jsonl` is the accounting payload**, whose SHA-256 is `58713a89dd097645ef82a5b8307cb52c9a08cde7836d44824e9ebfa06deed20d`. The **accounting receipt hash is `379ef811b955d287d2b3fafb53b9a9dfa61fe7445d643f11d4f8e4f23542c989`** (the S3 `receipt.json`). The payload is never a "manifest"; the earlier "realized-samples manifest" labels in this file and in the verification script's field name (`sample_manifest_sha256`) refer to the payload file's hash, not to a separate manifest document.

## Headline receipt (verified 2026-11-15 09:44 UTC) [R: accounting19-final-verification.json; STATE.md L23]

| Field | Value |
|---|---|
| Accounting root (private) | `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round2/sft-data-accounting/2026.11.15.19` |
| Accounting receipt SHA-256 | `379ef811b955d287d2b3fafb53b9a9dfa61fe7445d643f11d4f8e4f23542c989` (hash of the accounting receipt `receipt.json`) |
| Realized-samples payload SHA-256 | `58713a89dd097645ef82a5b8307cb52c9a08cde7836d44824e9ebfa06deed20d` (SHA-256 of the `realized-samples.jsonl` **payload**, 70,562,321 bytes) |
| Realized-samples payload bytes | **70,562,321** |
| Realized samples | **52,224** global indices, all contiguous (`global_index_coverage_exact: true`), = 204 updates × batch 256 |
| Committed updates / batch / context | 204 / 256 / 262,144 |
| Global sampled positions | **13,690,208,256** |
| Real tokens | **12,849,743,650** |
| Padding tokens | **840,464,606** (**6.139%**) |
| Assistant (chat) loss tokens | **7,751,383,148** |
| Causal replay loss tokens | **2,821,707,252** |
| Observed SFT trace occurrences | 871,429 (with 46 clipped occurrences) — not unique-corpus counts |
| Verification status | Metadata/config pins, per-row token/loss bounds and report totals pass; independent verification file SHA `fb35b83d366768598e93d8a3833cf95ce28ff9784600befa7bc796948a96fdbb` (bound in completion19-receipt) |

## Preservation / reconstruction split (deliberate cancellations, durable prefixes)

| Field | Value |
|---|---|
| Preserved prefix rows | **19,264 contiguous rows / 25,923,115 bytes** (preserved before deliberate `.18` cancellation), SHA `4ca1daea05b20c7b46d805fe67a6dacdc2a734d5ff5c1fb14a623b68ef669c1c`, bound to consumed-config SHA `1286fb00…` and checkpoint metadata SHA `e0df1d10…` [R: STATE.md L25] |
| Earlier .17 prefix | 11,707 uncommitted rows / 15,789,770 bytes, SHA `beffe2963a579269deb3fe2d786f943d31473c4f2f44cda85cdbaee0fb48f15c`, preserved before deliberate `.17` cancellation [R: STATE.md L27] |
| Reconstructed remaining rows | **32,960** |
| Byte identity | the 19,264-row prefix is byte-identical in the final manifest (`resumed_prefix_byte_identity_verified: true`); the complete 11,707-row `.17/.18` parity check passed [R: STATE.md L25] |
| Resume semantics | `.19` reused the 19,264-row verified prefix (including a non-aligned 37-row prefix + remaining native loader records) and reconstructed only the remaining indices; fresh reconstruction matched over 320 samples across packed/unpacked/concatenated cases [R: STATE.md L25] |

## Rate accounting (estimates — derived, not measured)

- `.15.17` (pre-fix) CPU accounting advanced only **~14–19 rows/second** across the 52,224 actual samples (batch 256 × 204, context 262,144). Profiling found repeated per-sample metadata reads and linear concatenation lookup; segment-ID list conversion alone was **45.8%** of sampled CPU [R: STATE.md L25/27].
- Fix: commit `c768cf0b30e1211f2cc897efcb8f45b9b9c1deaa` batches the native remapping and reuses native cumulative offsets without changing order, token hashes or masks (48 relevant tests, lint, baseline Pyrefly pass) [R: STATE.md L27].
- Post-fix reconstruction rate: **not separately recorded in STATE.md or the verification summary read** — the `.19` accounting simply succeeded inside its 1,800-second deadline [R: STATE.md L23].
- **All per-source token-rate fields are estimates, not measurements:** `estimated_real_tokens_per_second` / `positions_per_second` figures in the verification file (e.g., 168.10, 612.34, 5.02, 48.38) and the aggregate rates (e.g., 249,689.22 real tokens/s) are **counterfactual modeling estimates tied to the pinned 252.269-second batch-256 step-time model** (itself extrapolated from batch-8 timing, `s3://…/evidence/20261113-exp2-full11/full11-sizing/batch256-step-time-model.json`); the same applies to the derived `greedy_packed_*`/`ideal_packed_*` update-equivalent and packing fields. **MFU = model FLOP utilization**: the receipt's padding-compute/MFU shares assume cost proportional to positions and are **not hardware MFU measurements** [R: STATE.md L23; verification `limitations`].

## Mixture declarations and the 377-vs-375 key-set comparison

- **377 components** declared: `datasets/sft/data/components` in the validated resolved plan contains exactly **377** component keys [R: `smoke4-validation/resolved-plan.json`; components include e.g. `sft/source/agenttrove`, `sft/source/nemotron_sft_v3/agentic_v1/interactive_agent`, `sft/source/nemotron_sft_v3/agentic_v1/tool_calling`]. The accounting verification's `per_source` block covers **375** per-source entries.
- **Key-set comparison (computed for round 7):** the declared set minus the accounting set is exactly **`pretrain/c08q1`** and **`sft/source/penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-web-search-mcqa`**; no accounting key falls outside the declared set. So exactly two declared components realized **no sampled rows** among the 52,224 realized indices; **why** they sampled zero rows (mixture weighting/sampling over the consumed caches) is not recorded in the verification file — the cause is **unknown**; only the counts (377 declared / 375 realized) and the two key identities are established. Full detail: [per-source-accounting.md](per-source-accounting.md).
- **216 serialized-chat leaves (pack 256)** + **1,711 text replay leaves (pack=None)** in the consumed SFT config (context 262,144, batch 256, CP32/EP8) [R: STATE.md L1563 (full10 audit), L1575 (full11 consumed audit)]. The truncation census ran across **216 unique pinned SFT caches** after excluding deterministic validation rows [R: STATE.md L1517; `population-262k-final/truncation-census.json`].

## Final SFT losses — evaluation-step, population and token-weighting context

The final losses **0.690054 (train) / 0.676990 (validation)** are the trainer's final metrics reported at the **last optimizer update, native step 203** (of 204 updates numbered 0–203), from the SFT W&B run [M: https://wandb.ai/nyu-dice-lab/marin-long-loop/runs/sft-2026.11.13.11-289c5e517585, FINISHED at 203; R: STATE.md L21, L85]. Population: the realized SFT mixture accounted in this file — 52,224 realized sample indices, 13,690,208,256 positions, 12,849,743,650 real tokens. Token weighting: the training loss is borne by the **loss-bearing tokens** accounted above — 7,751,383,148 assistant (serialized-chat) tokens + 2,821,707,252 causal replay tokens = 10,573,090,400 loss tokens (padding carries no loss). Per the source, these are **not micro policy scores** [R: STATE.md L85]. The exact per-batch loss aggregation of the trainer is not restated here beyond the accounting above.

## Reconstructed-row validation over 320 samples

The `.19` accounting's reconstruction was validated against fresh reconstruction **over 320 samples spanning packed, unpacked and concatenated cases**: a resumed, non-aligned 37-row prefix plus the remaining native loader records matched fresh reconstruction in all three case families [R: STATE.md L25]. The 320-sample selection recipe as recorded in the source is this coverage across the three packing cases; no finer per-sample selection rule is published in STATE.md or the verification file read for this ledger.

## Per-source totals availability

- **Full per-source aggregate table published:** the 375-entry `per_source` mapping (per-component positions, real tokens, pad tokens, assistant/replay loss tokens, sequences and `sft_traces_seen`) is transcribed and re-summed in [per-source-accounting.md](per-source-accounting.md); its column totals equal the verification file's `overall` block exactly. The verification file additionally carries per-source fields not tabulated there (unpacked positions/sequences, per-source pad fractions, document real-length quantiles, counterfactual packing/update-equivalent and token-rate estimates) [R — present locally].
- Population-level source census: 21,245,048 traces / 234,419,368,922 source tokens eligible; left-32K truncation would discard 45,350,330,774 tokens (19.3458%) [R: STATE.md L1517]. Chat-cap-256 census: 971,833 chat sequences, 254,760,189,952 positions, 234,383,016,035 real, 20,377,173,917 pad (7.99857%) [R: STATE.md L1537].

## Limitations recorded in the receipt [R: accounting19-final-verification.json `limitations`; STATE.md L23]

- Padding-compute/MFU shares assume cost proportional to positions (MFU = model FLOP utilization); counterfactual timing/MFU estimates use the pinned **252.269 s/update** batch-256 step-time model (`s3://…/evidence/20261113-exp2-full11/full11-sizing/batch256-step-time-model.json`) — they are **not** hardware MFU measurements.
- Native multiworker SFT phase timing remains uncertified; none of the 32 successful tasks has a recorded `output_archive` reference (evidence-availability limit, not proof of destruction) [R: STATE.md L17].
- SFT identity is supplemented by the separately deep-verified 10-object `.15.20` archive (manifest SHA `2469051c8673d8a9e7cef3e85c76550c017b51f0e335711df928f7148279c7b0`); all 204 global training steps 0..203 occur in its unsampled history; the provider (W&B) refuses complete native history [M: https://wandb.ai/nyu-dice-lab/marin-long-loop/runs/sft-2026.11.13.11-289c5e517585 (requires authentication), FINISHED at 203].
