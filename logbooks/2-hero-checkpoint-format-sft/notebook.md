---
title: "Hero adjusted averaging and compact chat SFT"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/2"
related_issues: ["https://github.com/marin-community/marin/issues/10428"]
research_branch: "codex/2-hero-checkpoint-format-sft"
marin_base_revision: "c39c6f2f5b1b1c5aaa29d51a39304b40c7612dca"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: completed-provisional-handoff
---

# Hero adjusted averaging and compact chat SFT

This notebook retrospectively consolidates the October 19–22, 2026 experiment. The research branch preserves the full reviewed source and tests at Marin revision `c39c6f2f5b1b1c5aaa29d51a39304b40c7612dca`, including its history and earlier experiment revisions. Its inherited automatic workflows are removed. No archive CI or new compute was launched.

The source tree contains both the executed exploratory campaign and unqualified alternatives. The formal A–E pilot gates, 65K YaRN extension, serving proofs and single-execution qualification are archived proposals. They are not prerequisites satisfied by the completed resumed campaign. The [operator guide](../../docs/references/hero-format-sft.md#reproduce-the-exploratory-campaign) describes the public reproduction bindings. A fresh run must use fresh output roots and record its own source, request and artifact identities. These later bindings were CPU-tested; the historical executions used the exact earlier revisions below.

## Chronological decision record

| Date (UTC) | Attempt and reason | Measurement or failure | Decision |
| --- | --- | --- | --- |
| Oct 19–21 | Approximate a cooldown by LR-adjusted WSM-inspired merging of three saved checkpoints | One matched 59-row C4 panel improved 1.01465%; all row deltas lower | Keep B as parent for format learning; do not claim retrained-cooldown equivalence |
| Oct 21 | Seed five output chat-control columns from related donor tokens | Construction/ancestry checks passed; no donor ablation | Use C as frozen compact-SFT base |
| Oct 21–22 | Prepare genuine shared chat data/cache and bounded eight-update smoke | Storage exhaustion first prevented preparation; fresh donor construction succeeded after cleanup. Runtime setup repairs preceded actual gradient execution | Preserve failed artifacts; retain verified permanent parent/cache |
| Oct 22 | Smoke and stateful recovery | Early segments saved updates 1 and 2 before NCCL invalid-argument errors while alternating gradient and shorter acquisition executables | Defer acquisition execution until gradients finish; preserve full Adam, counters, deterministic position and queued diagnostics |
| Oct 22 | Continue verified update 8 to 64 total | 56 finite new updates; EOT-token NLL 3.38140 at 8 to 0.42875 at 64 on 16 training-source prefixes | Export merged checkpoint for response review; no extra training |
| Oct 22 | Generate and inspect 100 fresh responses | All EOT; no 2048-cap hits; content errors and two raw special-token anomalies | Provisional capacity-planning proxy; intended-backend measurement still needed |
| Oct 23 | Fresh independent contribution selection under workflow #9467 | Campaign modules have no maintained callers outside this new cluster | Archive entire harness; promote only independent existing export/kernel/resource fixes |

The detailed methods, source pins, measurements, checkpoint hashes, failures and limitations follow. Private runtime identities and credentials remain in internal immutable evidence; public checkpoint locations below were explicitly authorized for the handoff.

## TL;DR

We tried to turn an in-progress Hero pretraining checkpoint into a usable model for **provisional RL capacity planning**, without paying for a separate full-model cooldown or a full-model SFT run. The sequence was **learning-rate-adjusted WSM-style checkpoint merging → chat-token donor initialization → short LoRA plus vocabulary SFT → 100 reviewed responses**.

The merge lowered loss by 1.01465% on one matched C4 panel. After 64 compact SFT updates, all 100 sampled responses ended their assistant turn at EOT without hitting the generation cap. The reviewed outputs show basic reasoning and instruction following, alongside content errors and two special-token anomalies. This is enough to hand off for intended-backend throughput, latency and memory experiments. It does not establish cooldown equivalence, broad model quality or serving parity.

## Why we tried this

The immediate need was a checkpoint that can produce plausible assistant responses, do some basic thinking and follow turn boundaries so other teams can estimate RL rollout capacity. Waiting for an expensive cooldown and a fully qualified post-training campaign would delay those systems measurements.

Hero here is a roughly 535B-parameter MoE (535,477,106,688 parameter-array elements in the checkpoint manifest): 48 layers, hidden width 6,144, 384 routed experts with eight selected per token, two shared experts per layer, and latent expert width 3,072. The saved configuration, rather than the model name alone, defines the artifact for backend sizing.

We separated three problems. Checkpoint merging might cheaply improve the pretraining endpoint. Donor initialization might give the chat-control output tokens a useful starting point. Compact SFT might teach the shared Marin serialization and stopping behavior while changing only a small parameter subset. Each stage produced a separate artifact; production pretraining was untouched.

| Candidate | What it contains | Question it addresses |
| --- | --- | --- |
| A | Original pretraining checkpoint at step 294000 | What is the latest-checkpoint baseline? |
| B | LR-adjusted weighted merge of steps 282000, 288000 and 294000 | Can offline merging improve a fixed pretraining-loss panel without more pretraining? |
| C | B with five output chat-token columns initialized from donors | Can we give the control tokens a better starting point for format learning? |
| D | C after LoRA plus five-token input/output vocabulary SFT, merged into ordinary BF16 weights | Does a short format-training run yield usable assistant responses and turn termination? |

The executed campaign retained Hero's 16,384-token architecture and used no YaRN context extension. Optional longer-context code preserved in this archive was not qualified by these results.

## 1. LR-adjusted WSM-style merge: A → B

The reference is [Tian et al., WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](https://arxiv.org/abs/2507.17634), especially [§3.1–3.2](https://arxiv.org/html/2507.17634v1#S3). WSM means **Warmup-Stable and Merge**. A weighted checkpoint average retains different fractions of the updates between snapshots, connecting checkpoint weights to a synthetic LR-decay profile. We used this as motivation for an offline approximation; we did not rerun the optimizer along the counterfactual cooldown trajectory.

Our adjustment accounts for Hero's actual learning rate changing over the saved window. We reconstructed the saved scheduler at completed-update count minus one and checked it against logged Muon and Adam learning rates. For each checkpoint interval, we divided the integrated target linear-cooldown LR by the integrated actual LR. The target starts at the window's initial LR and decreases to zero at step 294000. Muon and Adam had matching normalized schedules, so one coefficient set applied to the trainable weights.

For the two intervals, the retained-update ratios were `r1=0.767876358` and `r2=0.268771198`. Rewriting `θ282000 + r1·(θ288000−θ282000) + r2·(θ294000−θ288000)` gives these snapshot weights:

| Checkpoint step | Merge coefficient |
| ---: | ---: |
| 282000 | 0.232123642 |
| 288000 | 0.499105160 |
| 294000 | 0.268771198 |

This corrects for the LR mass represented by each saved interval; it cannot reconstruct gradients or optimizer states from a new training trajectory. The implementation streams FP32 accumulation and writes BF16 weights. It preserves the latest effective routing-control bias, including its pending update, instead of averaging that control state. See the pinned [coefficient and construction implementation](https://github.com/marin-community/marin/blob/2416e9d797b0fbbb23327105618c56f823d42844/experiments/grug/moe_hero_ep/averaging.py). This was one selected merge, with no sweep over windows or merge rules.

### How we measured B

We compared A and B on **the exact same 59 C4 sequences**, each 16,384 tokens: identical tokens, next-token weights, segment IDs, tokenizer, forward configuration and evaluator implementation. There were 966,597 weighted next-token targets and 966,656 valid input tokens. Aggregate loss is summed cross-entropy divided by summed loss weight, in nats per target token. Evaluation used BF16 compute, FP32 routing and dropless expert1/data64 execution; filler tokens were excluded. These are mesh-axis sizes: expert1 keeps each model's experts local, while data64 distributes sequences over 64 ranks. The training mesh instead shards experts over 64 ranks; context1 and replica1 mean no additional context or replica partition.

| Measurement | A: latest checkpoint | B: adjusted merge |
| --- | ---: | ---: |
| Aggregate next-token loss | 2.130703034 | 2.109083774 |
| Routing entropy | 5.647986889 | 5.579939842 |
| Router z-loss | 13.127799988 | 13.315979004 |

B−A loss was −0.021619259, or **−1.01465%**. All 59 per-sequence losses decreased; deltas ranged from −0.036113082 to −0.009727833. A's primary three repeats spanned 0.000039287; six observations across two attempts spanned 0.000093145. B had one forward-pass measurement. Routing assignment totals were valid, but the distribution changed; lower C4 loss does not establish unchanged routing behavior or broader quality.

This is a small fixed-panel sanity comparison, not evidence across domains or against an actually retrained cooldown. A's separate full native format/content suite did not finish before preemption retries were exhausted. Its completed 35/35 structural checks covered 32 simple-chat and three 16K retrieval cases; they are not a full quality result.

## 2. Chat-token donor initialization: B → C

We wanted the model to learn explicit role headers, thinking delimiters and an assistant end-of-turn token in the shared Marin template. The initialization hypothesis was to seed those output logits from existing donor tokens with related text or stopping meaning, then let SFT learn their use in context. These were existing vocabulary IDs; we did not expand the tokenizer.

| Donor text | Target chat token | Target ID |
| --- | --- | ---: |
| ` role` | `start_header_id` | 128006 |
| ` message` | `end_header_id` | 128007 |
| ` think` | `start_think` | 128002 |
| ` answer` | `end_think` | 128003 |
| `end_of_text` | `eot_id` | 128009 |

(you know what we mean with the token ids. Can't be bothered to get them to render properly here)

Leading spaces in the first four donor strings matter. Only the five **output projection columns** changed at this stage. Input embeddings and routing-control state remained unchanged. Construction checks bound C to B's metadata and manifest, verified the five donor columns and retained bias, and found finite summaries for all 36 parameter arrays in the 37-array manifest.

These checks establish construction and ancestry. We did **not** run a donor-versus-no-donor ablation, so we cannot isolate donor initialization's contribution to the final behavior.

## 3. Compact LoRA plus vocabulary SFT: C → D

The method follows [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): freeze base weights and train low-rank updates. Here we also trained compact deltas for the five selected input-embedding rows and output columns. LoRA covered attention and shared dense projections; base parameters and routing-control biases stayed frozen. The intended benefit was format acquisition with compact gradients and optimizer state, without full-model SFT.

Data came from the first 6,000 scanned training rows of the existing [HuggingFaceH4/no_robots adapter, pinned source revision](https://huggingface.co/datasets/HuggingFaceH4/no_robots/tree/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b). Content deduplication and a 20% held-out split with seed 20261019 produced 4,806 training and 1,186 held-out conversations. The training cache contained 1,385,205 non-padding tokens.

The 65,975,980 processed non-padding tokens therefore represent about **47.6 training-corpus token-volume equivalents**, not that many unique tokens. The seeded packed-data mixture wraps finite source indices when exhausted; continuation resumes the same deterministic batch position. This substantial reuse matters when interpreting training-source EOT improvement. Loss masking supervised assistant targets using the shared Marin chat recipe. The held-out split was prepared, but a complete held-out evaluation was not performed in this exploratory campaign.

| Setting | Executed value |
| --- | --- |
| Trainable updates | Rank-16 LoRA, alpha 16; five input/output vocabulary deltas |
| Peak learning rates | Adapters 1e-4; vocabulary 1e-5 |
| Batch / source context | 64 sequences × 16,384 positions |
| Training hardware / mesh | 64 GB200 GPUs, one process per GPU; 16 tasks × four processes; expert64/context1/replica1 |
| Initial smoke | 8 updates, 2 warmup updates, seed 20261021 |
| Longer test | Restore step 8, then 56 additional updates to 64 total |
| Checkpointing | Every update; compact parameters, full Adam, counters and deterministic data position |

The longer test extended the finite linear schedule horizon from 8 to 64 while retaining Adam and schedule count 8. Its first new LR rose from the earlier short cooldown tail; warmup and counters were not reset. Thus this segmented history differs from a fresh 64-update run. Completed state and unfinished diagnostic queues survived continuation.

### How we measured SFT learning

We required actual distributed startup, finite per-update supervised loss, gradients and routing measurements, and audits that returned gradients and Adam moments contained only the intended compact trainable subset. Numerical abort bounds were loss 20, gradient norm 1000 and router z-loss 100; they were broad failure bounds, not quality thresholds.

The focused acquisition diagnostic reused **16 fixed training-source prefixes**, scoring one expected EOT token per prefix under teacher forcing. NLL is `−log p(EOT | prefix)`; top-1 counts how often EOT is the highest-probability next token. This probes learning to end a turn in those contexts. It is distinct from average sequence NLL and does not demonstrate held-out stopping behavior.

| Total update | Mean EOT-token NLL | Top-1 EOT matches |
| ---: | ---: | ---: |
| 8 | 3.38140 | 1/16 |
| 16 | 1.35890 | 12/16 |
| 32 | 0.65719 | 13/16 |
| 48 | 0.46985 | 14/16 |
| 64 | 0.42875 | 14/16 |

Assistant-supervised training loss was 1.70861 at step 8 and 1.63851 at step 64 on **different packed batches**, so this is not a matched loss comparison. All 56 new updates were finite.

Mean capacity-overflow rate was 0.17389 at update 8 and 0.19403 at update 64, measured on different packed batches. These endpoints do not isolate a matched-input routing trend. This is the mean across layers of dropped routed token-to-expert assignments divided by valid routed assignments (eight per valid token), not the fraction of whole examples discarded. Assignments exceeding dispatch capacity contribute no routed expert output; residual and shared-expert paths remain. Its quality impact is unassessed.

The campaign processed 67,108,864 token positions, including 65,975,980 non-padding and 42,664,754 assistant-supervised tokens. The 56-update continuation took 29m53s; that excludes prior preparation, failed attempts and the initial smoke. [Detailed SFT measurements](https://github.com/marin-community/marin/issues/10428#issuecomment-6348169425).

We exported D as ordinary BF16 Hero weights with the LoRA and vocabulary deltas baked in, retaining the separate compact checkpoint for continuation. Final metadata/handoff hashes and compact optimizer structure were verified. Independent numerical adapter-versus-merged parity remains unmeasured. There was no LoRA-only versus vocabulary-only ablation or full-model-SFT control.

## 4. Does the result generate usable responses?

Teacher-forced EOT acquisition was insufficient for the handoff decision, so we generated **100 fresh-prompt responses**: 40 math tasks, 40 Python tasks and 20 follow-ups supplied with fixed conversation history. Each case generated one actual assistant response at temperature 0.7 with a 2,048-token cap and EOT stopping. Prompts were exact-normalized-disjoint from the prepared SFT data; this does not establish semantic novelty or absence from pretraining.

We verified exactly 100 ordered unique cases, seeds, checkpoint/panel identities, two generation batches of 64 and 36 cases, raw token IDs, stop reasons and report hashes, then inspected every response for basic reasoning, code plausibility, history use, role boundaries, repetition and truncation. Generated code was not executed. Capped responses would have been treated as unfinished/censored separately from broken stopping; none hit the cap.

| Review dimension | Observed result |
| --- | --- |
| Termination | 100/100 ended at EOT; zero cap hits |
| Length | Median 113.5 tokens; mean 127.39; maximum 496 |
| Math, 40 cases | 33 correct final answers, 4 wrong, 2 incomplete, 1 ambiguous reference; one correct answer had contradictory reasoning |
| Python, 40 cases | 32 statically plausible implementations; 8 with functional defects |
| Fixed-history follow-ups, 20 cases | 18 clear passes; one possession-tracking error; one borderline item-count interpretation |
| Serialization / repetition | No literal role continuation or pathological repetition; two raw special-token anomalies |

The anomalies were a stray closing thinking marker and an end/start-of-text pair before EOT. All responses stopping at EOT therefore does not imply perfectly clean formatting. The fixed-history cases also do not establish reliable live extended conversation.

Read [all 100 prompts, raw responses and per-case judgments](https://gist.github.com/dlwh/408759a8eb09796d253b8b53a43769d4), or the [rollout-review handoff](https://github.com/marin-community/marin/issues/10428#issuecomment-6351113178).

The sampler used native BF16 dropless expert1/data64 evaluation, full-prefix recomputation without a KV cache, and physical padding to 4,096 positions. The model retained its 16K architecture; these short responses do not test the full context. This differs from capacity-limited expert64 training and from a serving/RL generation backend. Its timing is not an RL-throughput estimate.

## Failures and changes retained in the record

Preparation first failed for lack of storage; a fresh C completed after cleanup, and the verified shared cache was retained. Initial GPU attempts exposed CPU-backend and canonical-runtime setup problems. After updates could run, alternating gradient and shorter acquisition executables failed with an NCCL invalid-argument error. The recovery deferred acquisition model executions until the gradient sequence completed and preserved pending diagnostics across restarts. Full compact state was actually restored across smoke recovery segments; the successful 56-update continuation had no preemptions. We do not claim it demonstrated recovery from a preemption that did not occur.

The earlier comments preserve the [loader failure](https://github.com/marin-community/marin/issues/10428#issuecomment-6333608741), [runtime setup diagnosis](https://github.com/marin-community/marin/issues/10428#issuecomment-6334252384), and subsequent recovery evidence.

## Reproduction and references

- [Archived experimental operator guide](../../docs/references/hero-format-sft.md): construction, CPU data/cache preparation, compact SFT continuation, export and the frozen rollout panel.
- [WSM paper](https://arxiv.org/abs/2507.17634) and [LoRA paper](https://arxiv.org/abs/2106.09685) supply the method references; our actual modifications and evidence are described above.
- [Pinned no_robots dataset card](https://huggingface.co/datasets/HuggingFaceH4/no_robots/blob/e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/README.md) and [shared Marin chat template](https://github.com/marin-community/marin/blob/baeb4706a00c03dcbd2b0dd915f76c6384fbdf71/lib/marin/src/marin/datakit/chat_template.py).
- Historical construction source: `2416e9d797b0fbbb23327105618c56f823d42844`; matched A/B evaluator: `47c6a090cfb6c5d875079264fa25cbf3bf20d737`; completed SFT/rollout model implementation: `4d97008b7e7626fe9eb517c196988741b48cd6a6`. New public reproduction bindings have CPU coverage; a run at a different revision must record its own identity.
- The fixed C4 panel's raw SHA256 was `e1c77e834b636f2889595751b10b3061c0d4583824aa5144808cba7a0bd4b880`. Raw reports, row deltas, telemetry and immutable execution segments remain archived; W&B was disabled.

## Checkpoint artifacts

The merged step-64 native checkpoint contains the LoRA and vocabulary updates baked into ordinary BF16 weights. It needs no runtime adapter:

```text
s3://marin-us-east-02a/marin/grug/hero-format-sft/2026.10.21/sft-phase-2416e9d7/SFT-long-test-v1-20261022/attempts/c77824dfce148a77/export/step-64
```

Canonical metadata SHA256 digest: `aeec732ddaa4d47d146b0056b5b52891e5062a24a61857c35bb48f7e72cdfe57`.

The separate compact checkpoint retains adapters, five vocabulary deltas, full Adam state, counters and deterministic data position for training continuation:

```text
s3://marin-us-east-02a/marin/grug/hero-format-sft/2026.10.21/sft-phase-2416e9d7/SFT-long-test-v1-20261022/attempts/c77824dfce148a77/adapters/step-64
```

Canonical metadata SHA256 digest: `ce7a80fe9c7621e53d3f7745189a523473d7e228eb671a43039396924ae25fb6`.

Each digest hashes the parsed `metadata.json` object using UTF-8 `json.dumps(value, sort_keys=True, separators=(",", ":"))`. The [checkpoint restore implementation](../../experiments/grug/moe_hero_ep/weights.py) checks that digest and requires permanent metadata before loading arrays.

Use `marin-community/marin-tokenizer` at revision `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`, the shared MARIN chat template, and EOT token `128009`. Intended-backend loading and parity still need measurement.

## Conclusion and next measurement

The combined procedure produced a checkpoint that can generate enough coherent, bounded assistant behavior for a provisional systems handoff. The evidence is: a matched C4 improvement for B, measured training-source EOT acquisition during SFT, and direct inspection of 100 generated responses from D. These measure different properties and do not isolate each stage's contribution.

Next, load the merged checkpoint on the intended RL backend and measure throughput, latency and memory at representative prompt/response lengths and concurrency, checking tokenizer/template/EOT behavior and native-versus-backend agreement. Full held-out/native format/content evaluation, post-SFT matched pretraining-loss regression, independent merge parity and long-context behavior remain open. Track those future measurements in the coordinating logbooks issue; Marin contribution review is separate. No further training or production adoption is implied.


## Contribution selection and reproduction

[Selection review](selection.md) records every reviewed component and the forward-maintenance criterion. Keep the archived source dependency graph together. Run CPU regressions with `uv run pytest experiments/grug/moe_hero_ep/ lib/levanter/tests/kernels/test_short_conv.py`; repository markers exclude live cluster work by default. Kernel accelerator checks require their declared hardware and were not newly run for this archive. The [archived guide](../../docs/references/hero-format-sft.md) contains metadata construction, cache preparation, campaign continuation and rollout commands. Follow its exploratory-campaign section for the executed path; later formal qualification/context-extension sections describe unmeasured alternatives. Checkpoint access requires separately provisioned credentials and same-region placement. No model arrays are checked into this branch.

### 2026-10-23 - Research publication and extracted contribution

The full source was first published as archive commit `5f5aaebeb1ec1c12052962639f43055835f568a9`; exact notebook readback and source preservation were verified before the public issue handoff. [Marin #10556](https://github.com/marin-community/marin/pull/10556) contains only the selected existing-interface fixes on current main. [Selection and extracted-tree validation](selection.md#published-contribution) records its exact source and limits. The earlier broad PR #10539 is superseded. Both contribution review and future intended-backend measurements remain open; no merge or new compute is implied by this archival handoff.
