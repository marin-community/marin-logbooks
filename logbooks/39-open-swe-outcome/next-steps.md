# Next steps: repeat the outcome-filtering test on Qwen3.6-35B-A3B

Handoff for [marin-logbooks#39](https://github.com/marin-community/marin-logbooks/issues/39). The
Qwen3-8B results and the Qwen3.6 model work are in [notebook.md](notebook.md). Read its entries before you
start. This file lists what to run next, what to measure, and what would change the decision.

Issue #39 closed with the Qwen3-8B result. Before launching compute for the repeat, open a new logbooks issue
through `file-issue` and link #39 from it. This branch's notebook remains the record of both the Qwen3-8B work
and the Levanter model.

## Status (2026-11-17)

| Phase 0 item | State |
|---|---|
| 1. Training path | **Implemented in Levanter, verified on CPU at tiny scale.** Real-weight and accelerator checks are still open (below). |
| 2. Cost at 65,536 tokens | Not started. |
| 3. Base-model noise `s` | Not started. |
| 4. Serving at 65,536 tokens | Not started. |

### What exists now

- `lib/levanter/src/levanter/models/qwen3_5_moe.py`: `Qwen35MoeConfig` (`type: qwen3_5_moe`) and
  `Qwen35MoeLMHeadModel`. The config defaults are Qwen3.6-35B-A3B's.
  - It loads `Qwen/Qwen3.6-35B-A3B` directly. The loader keeps the text decoder and drops the vision tower and
    the MTP head.
  - It exports a text-only `Qwen3_5MoeForCausalLM` checkpoint.
- `lib/levanter/tests/test_qwen3_5_moe.py`: logits, loss and per-parameter gradients match HF within 1e-4. The
  tests also check export, reload in HF, and loading from a vision-language checkpoint. Run:

  ```bash
  uv run --package marin-levanter --group test --extra torch_test \
    pytest lib/levanter/tests/test_qwen3_5_moe.py -m "torch or not torch"
  ```

- [qwen36_hub_shapes.py](qwen36_hub_shapes.py) compares Levanter's tensor names and shapes against the Hub
  checkpoint's safetensors headers, without downloading weights. All 693 tensors and 34,660,610,688 parameters
  match.

## Do next, in order

1. **Promote the model to Marin.** Extract the three source files and the test into a clean Marin branch from
   current `main`, through `manage-pr`, with the line
   `Research logbook: https://github.com/marin-community/marin-logbooks/issues/39`
   (or the new issue). Leave the notebook, the shape script and the workflow removals in logbooks. The
   extracted files are:
   - `lib/levanter/src/levanter/models/qwen3_5_moe.py`
   - `lib/levanter/tests/test_qwen3_5_moe.py`
   - the `partial` RoPE classes in `lib/levanter/src/levanter/layers/rotary.py`
   - the one-line `GatedRmsNorm.axis` static-field change in `lib/levanter/src/levanter/layers/gated_deltanet.py`

2. **Real-weight parity on one node.** This is the first accelerator job.
   - Load `Qwen/Qwen3.6-35B-A3B` in Levanter and in HF transformers in bf16.
   - Score a few held-out trajectories from `val-loss/v1` with each, and compare per-token loss.
   - Expect agreement to roughly bf16 precision. A systematic gap points at a conversion bug that the tiny test
     cannot see.
   - HF's eager path falls back to the slow torch delta rule unless `flash-linear-attention` is installed. That
     is acceptable for a few rows.

3. **Check sequence packing before any training.** This is a correctness trap.
   - The Gated DeltaNet layers (30 of the 40) carry a recurrent state along the sequence. They ignore the
     segment IDs that keep packed documents apart in the full-attention layers.
   - If the SFT data loader packs several trajectories into one 65,536-token sequence, state leaks from one
     trajectory into the next.
   - Check whether the Qwen3-8B runs packed; their config is in the unpublished `trial.py`.
   - Then pick one fix:
     - one trajectory per sequence, padded at the end (padding after the last real token is harmless for a
       causal model);
     - make `Qwen35MoeGatedDeltaNet` reset its conv and delta-rule state at segment boundaries, and add a test
       that packed and unpacked logits agree.

4. **Cost (Phase 0 step 2).** Measure step time and peak memory at 65,536 tokens on one 8-GPU node (or the
   equivalent TPU slice), then estimate four runs of about 412M tokens each. Things that may dominate:
   - The delta rule is Levanter's pure-JAX chunked implementation, `chunk_gated_delta_rule`. It uses a
     sequential forward substitution inside each 64-token chunk, and no fused kernel is wired in.
     - Profile it first (`profile-training`).
     - `gdn_chunk_size` changes cost but not results.
     - The Pallas kernel in `levanter.kernels.pallas.gdn2` implements a different recurrence (Gated DeltaNet-2),
       so it is not a drop-in replacement.
   - **Experts.** There are 256 experts of width 512, routed through the Qwen3 MoE block's `shard_map` path.
     Check how the mesh shards the `experts` axis, and that a 35B model plus Adam state fits with FSDP.
   - **Checkpointing.** Gradient checkpointing currently covers a whole group of four layers. If activations at
     65,536 tokens do not fit, checkpoint each layer instead.
   - **Load-balancing loss.** It is on (coefficient 0.001), summed over layers. Decide whether to keep it during
     SFT, and record the choice.
   - If four runs cost too much, run one seed per arm first and add the second only if the result lands in the
     in-between band.

5. **Serving the fine-tunes (Phase 0 step 4).**
   - Exports are text-only `Qwen3_5MoeForCausalLM` checkpoints (`model_type: qwen3_5_moe_text`) without the MTP
     head. Confirm that Marin's vLLM build serves that architecture.
   - If it does not, add an export step that copies `model.visual.*` and `mtp.*` back from the base checkpoint
     and writes the vision-language config. The MTP weights will be stale after fine-tuning, so do not enable
     MTP speculative decoding on them.
   - Serve at `max_model_len` 65,536 or more, with compilation and CUDA graphs on; never eager mode. The
     current serve config, `experiments/evaluation/serve/models/Qwen/Qwen3.6-35B-A3B.yaml`, sets 32,768.

6. **Base-model noise (Phase 0 step 3).** Evaluate the base `Qwen/Qwen3.6-35B-A3B` twice on the 200-task set
   with the same agent, limits and serving settings that the fine-tunes will use. This measures `s` and gives
   the reference score. The dashboard's 50–64% comes from 500-task runs whose agent settings must be checked
   before comparing.

## Question and decision

**Question:** at a model that solves about half of SWE-bench Verified, does fine-tuning on solved trajectories
beat fine-tuning on the matched random arm?

**Why Qwen3.6:** Qwen3-8B solved 2–5% of Verified, so only a large effect would show. Qwen3.6-35B-A3B scores
50–64% on full SWE-bench Verified on Marin's eval dashboard (12 runs, most at 54–59%). At that level a
difference of a few points between arms would be measurable task by task.

**Decision rule.** Fix this before training. It is issue #39's rule.

- Let `s` be the standard deviation of the solve-count difference between two evaluations of the same
  checkpoint on the same 200 tasks.
- **Adopt the filter** if the solved arm beats the random arm by more than `3s`, pooled over seeds.
- **Keep all data** if the difference is within `1.5s`.
- **Anything in between:** run a larger rung (31,600 rows per arm, already built as `solved-31600` and
  `random-31600`).

## Phase 1: training

- **Data:** reuse the existing arms unchanged: `solved-10000_5e1e1b9c` and `random-10000_d5a9e952` under
  `s3://marin-us-east-08a/marin/open-swe-outcome/`.
- **Tokenization:** retokenize with Qwen3.6's chat template.
  - The Qwen3-8B template carried `{% generation %}` tags for the assistant-token loss mask. The Qwen3.6
    template will need the same.
  - Check that assistant turns, tool calls and thinking blocks are masked as intended. Print the mask for a
    few rows before training.
- **Model config:** `type: qwen3_5_moe` with `initialize_from_hf: Qwen/Qwen3.6-35B-A3B` and
  `max_seq_len: 65536`. Levanter uses `max_seq_len` as the training sequence length, and RoPE does not depend
  on it. Settle packing (step 3 above) first.
- **Training config:** match the Qwen3-8B runs except where the model forces a change: 65,536-token window,
  assistant-token loss, about one epoch, seeds 0 and 1. Record every difference in the notebook.
- **Watch the base model's skill.** The teacher models (MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B,
  Qwen3.8-27B, DeepSeek-V4-Flash) are about as strong as the base model, so fine-tuning on their traces could
  lower the base score. That outcome still answers the filtering question, but report it.

## Phase 2: readouts

Run these in order of cost. Each uses fixed inputs that already exist, so results compare directly with
Qwen3-8B.

| Readout | Inputs | What it answers | Cost |
|---|---|---|---|
| Held-out loss | `val-loss/v1`: 299 resolved and 299 unresolved held-out trajectories, plus 100 training rows per arm | Does the solved arm predict held-out successes better? Does each arm memorize its own rows? | ~1 GPU-hour per checkpoint at 8B; re-measure |
| Paired win rate | `val-loss/v2-paired`: 299 tasks, each with one resolved and one unresolved attempt from the same teacher | On a fixed task, does the solved arm prefer the successful attempt? | as above |
| SWE-bench Verified | Harbor `swebench-verified-200-c9`, 200 tasks, terminus-2, 50-minute agent limit | Does the solved arm solve more tasks? This is the decision metric. | ~14 h per checkpoint at 9–12 trials at a time |

The two loss readouts:

- **Scoring:** use `experiments/open_swe_outcome/val_loss.py score` with `--scores-dir turn-scores`, so
  per-turn losses are kept. Then run `logbooks/39-open-swe-outcome/paired.py` and `classify.py`.
- **Chat template:** `TOKENIZER` and `TOKENIZED_ARTIFACT` in `val_loss.py` point at Qwen3-8B. Point them at the
  Qwen3.6 tokenizer and its tokenized cache. The sets themselves are stored as messages and need no rebuild.
- **Scorer:** `score` loads the model through HF transformers in bf16.
  - Install `flash-linear-attention` and `causal-conv1d` for HF's fast delta-rule path. Without them, HF falls
    back to a slow per-chunk torch loop, which is impractical at 65,536 tokens.
  - Alternatively, score with the Levanter model.
  - Record which scorer and GPU count were used.

SWE-bench:

- **Comparison:** compare task by task, over tasks graded in every run.
- **Grades:** read per-trial grades from the FineStore `samples` table. One incomplete archived attempt fails
  Harbor's finalize step and leaves the run record without metrics. That happened to three of the four Qwen3-8B
  runs.
- **Failure modes to report:** agent timeouts and infrastructure failures.

## Known traps

- **Packing and linear attention.** See step 3 above.
- **Arm format.** The arms differ from raw traces in two ways:
  - Tool-call IDs are renumbered.
  - A 1,222-character "Solution Originality" block is inserted into every v1_2 mini-SWE-agent trace.

  `val_loss.py` reproduces both and checks them against training rows. Any new data built from raw traces must
  apply them too.
- **Unpublished code.** The code that built the arms and trained the Qwen3-8B runs (`arms.py`, `trial.py`,
  the Harbor policy) is on the unpublished branch `worktree-open-swe-outcome-pilot` (`d97646fe35`). Ask the
  owner to push it before reusing it.
- **Text-only exports.** See step 5 above.
- **Iris:**
  - `--priority production` is allowed only on `cw-rno2a`.
  - Jobs at default priority can sit in the Kueue queue for hours.
  - The data lives in `cw-us-east-08a`.
- **Scoring with SDPA:** disable cuDNN SDPA (`torch.backends.cuda.enable_cudnn_sdp(False)`). It found no
  execution plan at these lengths on H100.
- **Daytona snapshot quota (45):** a full 500-task Verified run creates one snapshot per task image and hits the
  quota. Stay on the 200-task policy, or confirm that snapshot cleanup works first.

## Deliverables

1. The Marin PR for the Levanter model (step 1).
2. Notebook entries for each remaining Phase 0 item, training, and each readout. Use the format of the
   existing entries: source revision, command, artifacts, result, interpretation.
3. A decision under the rule above, recorded in the notebook and in the coordinating issue body.
4. If the filter is adopted, a Marin PR that adds outcome filtering to
   `lib/marin/src/marin/datakit/download/open_swe_traces.py`, linking the issue.
