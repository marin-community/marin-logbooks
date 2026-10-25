# Prepared core FP8 diagnostic R2 — not executed

This is preparation only. Root must review and publish the exact probe and config on the existing research branch before an experiment. The offline CPU controls are in the authorized campaign; root reviews and publishes this recipe before execution. No implementation or maintained helper branch was created.

Source is clean detached public PR30 `054413ea8563f7216a33ee8dd34fc001c13b37ff`, tree `649b0eb013ee87bab6748cccde8d819f842d0704`, in `<execution-source-root>`; origin push is disabled. The config pins 24 source blobs and SHA256 hashes: all original 20 audited files plus tiny-corpus/tokenizer/tracker/data dependencies are byte-exact public069. Only the clean pinned054 checkout is an execution-source dependency; no separate trace contribution clone is required.

## Reproduction records and installed environment

Execution source is the [dlwh/marin tree pinned to054413ea8563f7216a33ee8dd34fc001c13b37ff](https://github.com/dlwh/marin/tree/054413ea8563f7216a33ee8dd34fc001c13b37ff); the compared base is [06941a509aabad0830c32ce1689daecaccfe3cdb](https://github.com/dlwh/marin/tree/06941a509aabad0830c32ce1689daecaccfe3cdb). Research inputs reside on the [codex/4-native-jax-llama logbooks branch](https://github.com/marin-community/marin-logbooks/tree/codex/4-native-jax-llama), at logbooks/4-native-jax-llama/prepared/core-fp8-prelaunch. Its inherited project source is not the execution checkout.

The sibling [runner](native-jax-core-fp8-preparation-r2-probe.py), [worker](native-jax-core-fp8-preparation-r2-worker.py) and [config](native-jax-core-fp8-preparation-r2-config.json) must stay together with their exact names and bytes. The [manifest](manifest.json) identifies all copies. The [source-proof record](prelaunch-freeze-and-source-proof.json), under key native-jax-core-fp8-preparation-r2-freeze.json, preserves the exact R2 freeze as a sanitized projection, including commands_after_gate: six explicit commands ordered baseline/raw/microbatch on CPU1 then CPU4. Read its stop conditions before progressing.

The [environment record](environment.json) records Python3.12.11 on macOS26.6.2arm64, uv0.13.5 and selected installed distribution versions read from .dist-info metadata without project imports. This environment was reused for earlier CPU trace evaluation; FP8 runtime compatibility is unqualified until this probe produces receipts. The recorded subset is not a full environment export and is not asserted equal to the source lock. Its source054uv.lock SHA256 is971039e63603be5d53506edf5e5573aeebd3b45006825ff28b59a2c48c54bd01, blob401eb969b4c29df06cdbcd628939757d406adf82.

For a new environment, start from the normal uv setup in the [installation guide pinned to054](https://github.com/dlwh/marin/blob/054413ea8563f7216a33ee8dd34fc001c13b37ff/docs/tutorials/installation.md) before enabling the offline/no-sync variables below. No setup or installation was executed in preparing this record; these links do not guarantee reproduction of the complete installed environment. Compare actual installed metadata and record differences. This diagnostic uses local tokenizer/corpus assets and requires no live accounts, live data or model download during execution. A private environment script and any separate trace clone are not inputs.

## Three bounded questions

1. Does the existing canonical public FP8 route complete its two-step finite-loss smoke with a wholly local tokenizer/data source, and does the identical native configuration fail at the actual quantization guard? Expected from source: named smoke succeeds; native rejects at `main/train_lm.py:230-231`, before the later `Trainer` guard at `trainer.py:334-335`. This is a prediction, not a runtime result. The baseline copies the existing `test_train_lm_fp8` recipe: two layers, hidden32, heads/KV2, sequence64, default intermediate/model fields, default F32 policy, two steps, batch=device count, one eval batch, FP8 config/history defaults, local checkpointer and JSON tracker. It reuses `tiny_corpus.construct_small_data_cache` with its original token/data sizes and `stage_gpt2_tokenizer` for checked-in GPT2 files, binding both model tokenizer/reference to that local directory. Data indices remain within the larger local tokenizer vocabulary. No HF model weights/download are required. A guard with a different reason is recorded distinctly. A finite loss is smoke, not FP8/native numerical or checkpoint-continuation parity.
2. Can the existing raw `Fp8DotGeneralOp` run under an Explicit mesh, preserving values, input/weight gradients, and all six overwritten state arrays? The raw case reuses the exact representable operand/cotangent tables and nonunit scales from `haliax/tests/test_fp8.py:194-230`, with output-first weights matching NativeLinear. It runs the same operation first on a one-device Auto mesh and then on all selected devices in Explicit mode; CPU4 uses data2/model2. Stored weights are `P(model,data)` and gathered to `P(model,None)`, matching the existing native column layout. Inputs/output use data/model roles. Independent NumPy products give exact representable forward/dgrad/wgrad expectations; histories must acquire the actual global operand/cotangent maxima and existing scales remain unchanged with zero historical amax. The existing `apply_updates` overwrite path supplies persisted state. Expected outcome is deliberately unknown: custom VJP/dot sharding may reject before any parity claim. This does not wire NativeLinear or copy a quantizer.
3. If the raw operation succeeds, does the existing raw microbatch accumulator preserve the shared overwrite reduction? The microbatch case mirrors the unequal-amplitude state test in `test_grad_accum.py:89-129`: two microbatches with input amplitude32 and1, same initial operation/weights for each. It compares actual `microbatched_arrays` with direct per-microbatch gradients combined by existing `accumulate_gradients` and `partition_for_grad_overwrite`, averaging only additive gradients. It requires exact loss and host-array additive weight-gradient equality before recording all six states and a state-match result. A gradient mismatch fails its own tagged control and cannot be classified as state-reduction success. Shared-reference input amax is32. Source suggests raw sum/average may instead produce16.5; that is an unobserved risk. Placement/compile exceptions are separate outcomes. No fake state/engine or implementation dispatch assertion is used.

## Execution sequence after the publication gate

Use a fresh output directory for every command; the probe refuses overwrite. Run baselineCPU1, then rawCPU1, then microbatchCPU1. Stop after an unexpected baseline failure or raw-placement error and review its recorded exception before proceeding. If those controls are understood, repeat on CPU4. Do not blindly run a loop after an error. The source-derived microbatch mismatch is a valid diagnostic outcome, not a previously observed failure; the JSON distinguishes it from exceptions. The probe raises exceptions after recording them.

Published research runner/worker/config and clean pinned execution source are separate checkouts/locations. Publish the sibling worker `native-jax-core-fp8-preparation-r2-worker.py` alongside the runner and config; its exact hash is validated before imports. Pass both locations, the config and a fresh output explicitly. The config contains immutable production commit/tree/file identities and question definitions; it contains no author-specific source path or immutable preparation/authorization flags. Preparation status belongs to the freeze/history, not future runtime configuration.

The public recipe below requires an already installed compatible venv and uv; it does not depend on the author's private environment script. Set these four absolute paths for the executing host. Keep outputs outside the clean source checkout. Run one MODE/COUNT command at a time in the sequence above, retaining its log and receipt before proceeding.

```bash
export FP8_SOURCE_ROOT=/absolute/path/to/clean-pinned-054-source
export FP8_RESEARCH_ROOT=/absolute/path/to/published-fp8-research-files
export FP8_OUTPUT_ROOT=/absolute/path/to/new-fp8-results
export FP8_VENV=/absolute/path/to/already-installed-compatible-venv
export UV_PROJECT_ENVIRONMENT="$FP8_VENV"
export UV_NO_SYNC=1 UV_OFFLINE=1 UV_FROZEN=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_allow_excess_precision=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=
for fp8_package_source in "$FP8_SOURCE_ROOT"/lib/*/src; do
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$fp8_package_source"
done

export JAX_NUM_CPU_DEVICES=COUNT
uv run --directory "$FP8_SOURCE_ROOT" python \
  "$FP8_RESEARCH_ROOT/native-jax-core-fp8-preparation-r2-probe.py" MODE \
  --source-root "$FP8_SOURCE_ROOT" \
  --research-root "$FP8_RESEARCH_ROOT" \
  --config "$FP8_RESEARCH_ROOT/native-jax-core-fp8-preparation-r2-config.json" \
  --devices COUNT \
  --output "$FP8_OUTPUT_ROOT/MODE-cpuCOUNT" \
  > "$FP8_OUTPUT_ROOT/MODE-cpuCOUNT.log" 2>&1
```

Create the fresh FP8_OUTPUT_ROOT directory before the first command; each case's MODE-cpuCOUNT directory must not already exist. The probe refuses overwrite rather than altering an earlier receipt. Failure to parse the CLI or create its fresh output directory is an invocation failure reported to stderr; no FP8 inference is valid from it. Once that directory exists, stage-tagged `receipt.json` is written before source/config/import/device checks. Preflight failures are classified `invalid_harness_or_provenance`, preserving observed source/module/device details and exception/traceback rather than treating missing dependencies or wrong devices as unsupported FP8. The stdlib-only runner validates the sibling worker filename/hash from config, records that worker hash, writes preflight.project_imports, and loads the worker with importlib.util inside the failure envelope. The worker has all project imports at top level and exposes execute(args, receipt, checkpoint), with no CLI or main. Missing project dependencies therefore retain the same preflight receipt without local imports or lint suppression.

Future source/device/import checks execute in the same process as the diagnostics. The raw Auto result is persisted under completed_controls before Explicit placement starts; a later failure retains that Auto evidence and tags its kind/stage (placement, forward, gradient, value/gradient assertion or state assertion). Baseline canonical smoke evidence is likewise retained before the native control. Microbatch exact loss and additive-gradient controls gate state classification. Diagnostic exceptions are recorded with their original traceback and re-raised, with no retry or precision/placement change.

The R2 freeze supplies six exact commands using these explicit environment parameters. The standalone diagnostic invokes existing public training/primitives directly; it is not a pytest selection and changes no default markers. No probe/config edits are needed when the clean source root, published research location, venv or output location differ across hosts.

## Unknowns and boundaries

- Auto success cannot establish Explicit correctness; a failing exact-value/state assertion is retained, not relaxed. Syntax validation does not verify project APIs or runtime placement.
- Raw output `reshard` cannot rescue a dot/VJP that already rejects its Explicit types. The existing FP8 op ignores extra sharding kwargs, so this is an actual feasibility question. Amax global reductions under DP/TP must match the independent full-array reference.
- Only one operation and two microbatches are in scope. Per-layer stacked histories, remat/nested scan, BF16 computation/state dtypes, optimizer exclusion, real full-checkpoint continuation, LoRA combinations, NativeLinear/converter wiring, and arbitrary QuantizationConfig target selection remain for a later bounded maintained response.
- Canonical platform attention dispatch remains unchanged: None chooses JAX_FLASH on CPU (GPU NVTE/TPU Splash in source), whereas native None maps to reference and rejects JAX_FLASH/NVTE/blocksize/scanFalse. The native negative control therefore cannot establish attention parity. No settings are silently rewritten to vanilla.
- CPU float8 values and exact synthetic controls cannot prove accelerator FP8 GEMM use, BF16-default acceptance, model-level parity, speed, larger-model stability or scientific/canonical cutover. No hardware/network/GCP/jobs/CI/automation are part of this preparation.

AST parsing, compilation to an unexecuted code object and the applicable ordinary static hooks passed; no runtime check was performed. No project modules were imported and no assets, trained weights or runtime results exist from this probe. The R2 freeze pins runner/worker/config/plan bytes and source comparison; the original private environment remains preserved but is not a required published input; earlier feasibility report remains unchanged, including its omission of the earlier public-entrypoint guard, corrected here explicitly.

## R1 review response

The original freeze/artifacts and three-P2 review remain immutable. F1 is addressed by explicit source/research/config/output arguments, production-only config identities and the self-contained public recipe. F2 is addressed by stage-tagged persistent partial receipts, import/provenance/device preflight classification, retained Auto and canonical completions, and re-raised original exceptions. F3 is addressed by exact host-array additive gradient equality before state classification; no tolerance or fixture math changed. The stale public guard reference is corrected to230-231. This response has only AST/code-object compilation qualification; no project import, runtime API, smoke, raw or state-reducer outcome has been observed.

## R2 ordinary-hook response

Fresh R1 independent review found zero remaining F1-F3 issues; that receipt remains historical and immutable. The subsequent required ordinary preview rejected local project imports (PLC0415) and applied formatting/license fixes only to preview copies. R2 separates a stdlib-only runner from a sibling worker with top-level project imports. Config pins the worker filename and final formatted hash; the receipt retains the checked worker hash before importlib loading. The worker retains the R1 execute body, nested diagnostics, numerical fixtures, stages, source/device checks and exact assertions. The loop cotangent is explicitly bound as a lambda default to satisfy B023 while preserving the immediately used value. No noqa/SKIP or precision/tolerance change is used. These are static hook phases, not observed research outcomes.

Required ordinary checks run on copies in an isolated preview at the intended research evidence path, using ./infra/pre-commit.py --fix for runner, worker, config and this plan. Only the final formatted bytes are frozen and must be published together. AST parsing/code-object compilation and ordinary static hooks do not import or execute the runner/worker diagnostics. All runtime outcomes remain unobserved, pending root review of the published recipe.
