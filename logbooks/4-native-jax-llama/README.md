# Reproducing the native-JAX Llama campaign

Start with [the notebook](notebook.md). `manifest.json` maps selected evidence to exact source pins, records each retained file's SHA256 and size, and distinguishes measured results from prepared controls. The research branch preserves Marin history; it is not intended to merge into the logbooks index branch.

## Evidence available without accelerators

The JSON under `evidence/` contains numerical errors, all acceptance decisions, original timing samples, and placement records. It contains no model or checkpoint tensors. `evidence/campaign-snapshot.json` gives a compact derived table; original reports remain the authoritative measurement records. `analysis/` retains the interpretation and exact inference limits. `hlo/` retains selected normal-Trainer, graph-context, separate-VJP, and before/after barrier graphs.

The original 107M reference reports correctly say `comparison_valid=false` under their original strict evaluator. Their adjacent `.approved.json` files are later evaluations under `2a4de13094a488af0b2df9325ef2fc50c2c60469`. Do not rewrite historical exit codes or confuse evaluator revision with measurement revision. Large failed comparisons retain null accepted ratios.

Compare retained file bytes against the manifest before analysis:

```python
import hashlib
import json
from pathlib import Path

root = Path("logbooks/4-native-jax-llama")
manifest = json.loads((root / "manifest.json").read_text())
for item in manifest["selected_files"]:
    data = (root / item["archive_relative"]).read_bytes()
    assert len(data) == item["bytes"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
```

The manifest's `source_local` paths describe collection history only. They are not paths a reader should expect on another machine. Frozen scripts and reports retain their original bytes, including historical path strings. Re-running the older analysis scripts requires explicitly setting their declared input/output roots to the corresponding archived data; preserve the original script and hash the relocated copy. Never silently edit a frozen file and then report its old SHA as the executed source.

## Source revisions and runtime

Use a fresh disposable checkout at the **measurement** pin for a historical reproduction. The research branch's selected source is `8aa8674d10894728ed0b455ae14e14f46457cf9f`. The normal matched-Splash baseline is `36e2a9a97fab22b76dfa5a03964c6600ec5713ac`. The first large reference/common-state runs use `ad2bc20438796c63415240ed1979e93d212fc6da`; earlier small measurements use the pins in their reports.

The checkpoint runs use `b8c47e8eea0d39412e3631404588b4f25ce4e45c` and `f659101960398fe9ad104e66600c0ccf3a2f34c8`. Their histories are retained through archive-only `ours` merges. Check out those commits to reproduce their source; do not assume the selected `8aa` tree integrates them. Likewise, the rejected attention-order and Q-compute-view experiments require their retained runtime patch **and** the report's base revision.

Use `AGENTS.md`, `lib/levanter/AGENTS.md`, `pyproject.toml` and `uv.lock` at the selected measurement revision for dependency setup; use frozen dependency resolution so replay does not rewrite the lock. Historical TPU records used Python 3.12, JAX/jaxlib 0.11.1, Equinox 0.13.2, Optax 0.2.6, and a TPU extra. Verify actual imported `levanter`, `haliax`, attention-core, and benchmark module paths point into the intended checkout. An existing editable environment pointing at another worktree is insufficient. The prepared V2 runner records those import checks in fresh child processes.

## Repeat a matched full Trainer measurement

From the intended source checkout, the benchmark entry point is `lib/levanter/scripts/bench/bench_llama_native.py`. Its defaults are small; the large workload requires explicit flags:

```bash
XLA_FLAGS=--xla_allow_excess_precision=false \
JAX_DEFAULT_MATMUL_PRECISION=highest \
uv run --frozen --package marin-levanter --extra tpu \
  python lib/levanter/scripts/bench/bench_llama_native.py \
  --mode compare --attention-backend splash --compute-dtype float32 \
  --vocab-size 65536 --hidden-dim 2048 --intermediate-dim 5632 \
  --num-layers 28 --num-heads 16 --num-kv-heads 4 \
  --batch-size 8 --seq-len 2048 --data-size 2 --model-size 2 \
  --learning-rate 0.001 --weight-decay 0.01 \
  --beta1 0.9 --beta2 0.999 --epsilon 1e-8 --seed 42 \
  --warmup-steps 5 --measured-steps 20 --gradient-checkpointing \
  --profile-steps 0 --parity-atol 1e-5 --parity-rtol 1e-4 \
  --update-parameter-relative-l2 0.005 --loss-trajectory-relative-l2 0.001 \
  --output-dir /tmp/native-jax-15b-reproduction/float32
```

Check the exact argparse spelling at the measurement pin before executing; the prepared V2 JSON is the authoritative reviewed next-run argument list. BF16 changes compute dtype to `bfloat16` and matmul precision to `default`, and retains every original elementwise gate. The relative-L2 options remain recorded configuration fields but affect acceptance only for FP32; the evaluator ignores them for BF16. Profiles, when requested, run after timing and measured loss collection. The V2 runner is designed to verify both reports, exact source/import identities, frozen large hashes, normal-executable placement, XPlane counts, and bounded archive membership. It expects one JSON configuration argument; the retained prepared command/config/proofs describe that interface. It was **not** a completed TPU measurement at archival entry time.

A repeat needs four suitable TPU devices on the same host for the data2/model2 setup. Do not treat a local CPU run, abstract graph, or a job exiting zero with missing guarded reports as that measurement. Read the original exact weight/batch hash and physical layout checks, not only `comparison_valid`.

## Repeat the controlled block A/B

The frozen readable helpers in `runners/` are retained for the separate source-reference, forward-context, and VJP controls. The A/B controller is self-contained and embeds exact frozen assets. It requires a disposable checkout at candidate `8aa`, restores the exact baseline attention-core bytes before its first fresh child, then restores candidate bytes before its second fresh child and in cleanup. It must not run in a shared or dirty development worktree.

The controller's interface is:

```bash
XLA_FLAGS=--xla_allow_excess_precision=false \
python logbooks/4-native-jax-llama/runners/native_jax_query_scale_ab.py \
  --worktree /path/to/disposable-8aa-checkout \
  --output-dir /tmp/native-jax-query-scale-ab-reproduction \
  --expected-backend tpu
```

Run this with the intended checkout's TPU dependency environment and module paths. The historical wrapper wrote the frozen assets to `/tmp`; the controller expects that writable execution scratch location. Its parent must remain free of JAX imports before spawning children. Do not disable the source/import/initial-weight/batch/operand/placement/hash guards to get a result. `--small --expected-backend cpu` and `--abstract-only` exercise different, explicitly marked proofs and do not establish large TPU numerical behavior.

The BF16 fragment's residual K/down differences, the fixed named residual/cotangent controls, and its distinction from full Trainer backward all remain material. The common raw-dQ two-rounding-order control was not performed. The controller tests a barrier, not every possible causal explanation.

## Reproduce the newly retained preparations

The compact A/B [preparation manifest](prepared/splash-rotary-regression-ab-prepared.json) and [independent review](analysis/native-jax-splash-rotary-regression-ab-independent-review.json) describe a prepared experiment, with no TPU numerical result. Its exact controller is [native_jax_splash_rotary_regression_ab.py.source](runners/native_jax_splash_rotary_regression_ab.py.source); the exact API builder is [submit_native_jax_rotary_regression_ab.py.source](runners/submit_native_jax_rotary_regression_ab.py.source). The `.py.source` suffix marks immutable research source retained as evidence, not adopted production code. The executable bytes and SHA256 are unchanged, and their Python ASTs were checked separately. Python can execute these files directly regardless of suffix.

The compact API builder has historical hardcoded preparation/command paths. For an exact historical rerun, restore the manifest-selected controller, API builder, preparation and command files to their recorded `/private/tmp` basenames and verify every hash. Its source workspace must be a clean disposable checkout at exact `8aa`; the research archive HEAD is a different commit and must not be passed as that workspace. Use the [command/configuration](prepared/splash-rotary-regression-ab-command.json), [pure-build request](prepared/splash-rotary-regression-ab-api-request.json) and [CPU proof command](prepared/splash-rotary-regression-ab-cpu-command.json) as the retained entrypoints. Alternatively, retain explicitly relocated copies and their new hashes before executing them. Restoring preparation files or constructing a request does not authorize submission.

The pure-build API request includes expected public GitHub-account and task-local source-path provenance. It was built without a client, credential resolution, RPC or submission. Its scheduling timeout is 120 seconds and execution timeout is 780 seconds, with zero failure/preemption retries; reconciliation/finalization latency is outside their configured sum. The [timeout correction](analysis/native-jax-iris-timeout-source-evidence.json) explains why the prior full-Trainer CLI `--timeout 1800` bounds execution without bounding queue wait.

Reproduce the FP32 source preparation in a separate checkout at `f1e991207ceda752957938b1cfe8893027d3fed8`, using its [manifest](prepared/fp32-query-scale-prepared.json), [design](analysis/native-jax-fp32-query-scale-experiment-design.md), exact [patch](prepared/fp32-query-scale-rounding.patch.json), [abstract helper](runners/native_jax_fp32_query_scale_abstract.py.source) and adjacent abstract proofs. Its base is `36e`, and its BF16 path has no barrier. The proposed common-source prefix and same-raw-dQ/trig numerical controls have not run. Referenced routine test/commit logs remain outside this archive; the frozen preparation summary records their bounded outcomes.

The patch is a lossless JSON container so mandatory single-space blank unified-diff context lines remain intact. Encode its `patch_utf8` field as UTF-8 to recover the original 747 bytes, then check `original_sha256` (`25c3f29f9e20780e4f5f3867a2069eb55136aa2b34a096c3ebc386cdb990d779`). The selected-file hash covers the JSON container; the embedded hash covers the extracted patch.

The CPU integration source is retained at `be8093b10163be50b8f8e16135ccb4fccdc4efe5`. Check out that commit separately to use the integrated public training/checkpoint paths. The [source proof](analysis/native-jax-replacement-source-integration-proof.json), [validation summary](analysis/native-jax-replacement-integration-validation.json), [import proof](prepared/replacement-integration-import-proof.json), [checkpoint configuration](prepared/replacement-integration-smoke.yaml) and [result summary](evidence/replacement-integration/smoke-validation.json) define its bounded one-process/four-CPU validation. Choose fresh local checkpoint/output paths in an explicitly derived configuration when repeating the smoke. No checkpoint arrays are retained. Environment scripts and raw logs referenced in summaries are intentionally omitted; configure the checkout's dependencies and verify imports rather than assuming another machine's environment paths.

Both new source histories are reachable through archive-only `ours` merges. Their presence does not integrate either into the selected `8aa` tree or switch the canonical Llama registry. The original 134 selected artifacts remain unchanged; the prototype and static FP32 audit are reused without duplicate copies. Unrelated rotary feature work is outside this archive checkpoint.

## Checkpoint and profile boundaries

The historical native checkpoint runner lives in Git at `lib/levanter/scripts/bench/smoke_llama_native_checkpoint.py`; exact measured snapshots are under `runners/historical/`. Use their original config/runtime requirements from the preserved revisions. The recorded distributed test uses two processes, four local TPU devices per process, and data4/model2. Checkpoint arrays stayed region-local. This archive retains assertions and recovered metadata, not the roughly 55 GB of arrays per run or missing concrete per-rank result JSON.

Dense XPlane/Perfetto data is deliberately not in Git. The manifest records regional TTL archive identities, sizes, and checksums when available; access may require the original project permissions and the data may expire. The archive retains compact profile analysis, exact raw timing samples, analyzer source, and decisive HLO. A fresh raw profile analysis after expiry requires another authorized capture; it cannot be recreated from aggregates alone.

When raw profiles are available, run `lib/marin/tools/profile_summary.py summarize` separately for each profile with `--warmup-steps 0 --breakdown-mode exclusive_global`. Only two already-warmed steps were captured. Inspect actual train-module markers because automatic step estimates can mix conversion modules with training. Never sum overlapping async copies or multiple core durations as wall time.

## Publishing another milestone

Append new hypothesis, exact execution/source/config, guards, outcome, interpretation, and next decision to the notebook. Keep failed alternatives. Preserve published Git history and remove inherited automatic workflows on research branches. Archive source/evidence commits first, then post commit-pinned issue milestones. Extract only code and tests that Marin will maintain into contribution PRs; leave this record, discarded patches, reproduction scripts, and archive-only history/workflow changes here.
