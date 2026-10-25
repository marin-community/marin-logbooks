# Raw FP8 microbatch state control

This control is prepared and unexecuted. It asks whether `microbatched_arrays` preserves FP8 state while averaging ordinary loss and weight gradients.

The FP8 custom gradient carries six state arrays alongside the ordinary weight gradient: an input scale and maximum-magnitude history, a weight scale and history, and an output-gradient scale and history. These arrays overwrite quantization state rather than enter the optimizer as additive gradients. The existing shared reducer, `haliax.quantization.accumulate_gradients`, combines histories by elementwise maximum and retains the latest scales. The named-array microbatch path uses that reducer. The raw-array path currently adds and divides every gradient leaf. Whether that produces incorrect state in a real FP8 call is the question being tested.

## Source and references

Use clean public [source ba58e83](https://github.com/dlwh/marin/tree/ba58e83e3310163e8d674ea9e95a85094d236a7a), tree `223c7ff9b639ff2966941d8d9ea762ec157cb501`, published as [draft PR32](https://github.com/dlwh/marin/pull/32). It includes the shared backward-placement correction and requested forward-output placement. The [publication snapshot](public-source-proof.json) records the exact head/base at `2026-10-25T03:57:55Z`; it does not establish merge or completed CI. Execute this source checkout separately from the research checkout supplying this recipe.

Keep the sibling [runner](native-jax-fp8-microbatch-preparation-probe.py), [worker](native-jax-fp8-microbatch-preparation-worker.py) and [config](native-jax-fp8-microbatch-preparation-config.json) together. The runner is unchanged from [revision R3 of the earlier core-FP8 control](https://github.com/marin-community/marin-logbooks/tree/4fc43a3c19c89f7894d415629426fe21f122d220/logbooks/4-native-jax-llama/prepared/core-fp8-auto-control-r3). It provides source-integrity checks and staged execution receipts. This worker derives R3's unexecuted microbatch stage by adding independent numerical/state references and prerequisite controls, and requests the now-supported FP8 output layout. The original R3 microbatch stage remains unexecuted. The [delta](r3-preparation-delta.patch) preserves those research-code changes.

The config pins 25 repository-relative source files and the worker basename/hash. The [initial freeze](initial-preparation-freeze.json) preserves its historical local-only preparation status and original SHA-256 prefix `56034ade`. [Readiness](readiness.json) records the later public-source, [independent static review](independent-review.json), and [ordinary-check](ordinary-checks.json) phases separately. The original runner, worker, config and preparation plan passed ordinary checks without changed hashes. The [historical plan](historical-preparation-plan.md) and [manifest](manifest.json) document public projections and original hashes. No project imports or scientific controls ran during those checks.

## Fixture and independent oracle

The objective is mean squared output. An output-first weight of shape `(4, 8)` contains `0.125`. Two equal-size microbatches have constant input amplitudes `32` and `1`. The initial FP8 state is:

| State pair | Scale | Initial maximum-magnitude history |
| --- | ---: | --- |
| Input | 2 | 896, 448, 224, 112 |
| Weight | 0.5 | 224, 112, 56, 28 |
| Output gradient | 4 | 229376, 114688, 57344, 28672 |

Inputs and weights use E4M3FN (`float8_e4m3fn`); output gradients use E5M2 (`float8_e5m2`). The old history maxima equal the format maximum times the scale: `448 * 2`, `448 * 0.5`, and `57344 * 4`. Delayed scaling therefore retains scales `2`, `0.5`, and `4` for both independent microbatch calls. The fixture's outputs and cotangents are exactly representable in the selected formats; this is a fixture justification, not an observed result.

Independent NumPy matrix products and the analytic mean-loss cotangent `2 * output / output.size` determine each microbatch's loss and ordinary weight gradient. For each state pair, the oracle rolls the initial history left and replaces entry zero with the corresponding absolute maximum; the scale retains its stated value. It combines the resulting histories by maximum and keeps the second call's scales. Only the two ordinary gradients and losses are averaged. No shared FP8 reference or reduction helper supplies this oracle.

The input history's new entries are `32` and `1`: maximum reduction gives `32`, whereas averaging gives `16.5`. Output-gradient histories also distinguish the reductions. Identical retained scales intentionally leave latest-scale selection indistinguishable from averaging scales in this fixture. This control cannot qualify varied-scale selection behavior.

The following prerequisites must pass before classifying raw state:

1. Direct full-batch evaluation and the wrapper's no-split path each match their own full-batch NumPy loss, weight-gradient and six-state reference.
2. Each independently placed microbatch matches its own NumPy loss, weight-gradient and six-state reference.
3. The shared reducer combines those two real gradient results and matches the independent maximum-history/latest-scale reference. This checks the existing named-path reducer; it is not the oracle.
4. Raw accumulation matches the independent ordinary loss and weight gradient and passes all placement checks. Only then record all six actual/expected state arrays and a separate arithmetic-average prediction.

Numerical comparisons use exact equality. Full-batch state has its own mean-loss cotangent and is checked independently; it is not substituted for microbatch state. Arbitrary full-batch and microbatch FP8 trajectories need not be equal. A raw-state mismatch is recorded before raising. Earlier provenance, placement, loss or gradient failures must not be reported as metadata-averaging evidence.

## Logical devices and placement

`CPU1` and `CPU4` mean one or four logical JAX CPU devices in one process. They do not describe physical cores or hosts. The Explicit mesh has `data` and `model` axes: `data` partitions examples and stored input-feature weights; `model` partitions output features.

CPU1 uses mesh `data=1, model=1`, two examples per microbatch, and four examples total. CPU4 uses `data=2, model=2`, four examples per microbatch, and eight examples total. Both cases have exactly two microbatches, with amplitudes `32` and `1` respectively.

`P(...)` denotes a JAX `PartitionSpec`. Inputs use `P(data, None)`, stored weights `P(model, data)`, compute weights `P(model, None)`, and requested outputs `P(data, model)`. State is replicated. Check equivalent actual shardings, full device sets and addressable shard shapes. CPU1 alone cannot qualify nontrivial physical partitioning.

## Environment and proposed commands

Reuse the installed compatible environment and modern uv `0.13.5` described by the [earlier environment subset](https://github.com/marin-community/marin-logbooks/blob/5488bb8d96f74ab35088ec8ed0f1be457405cc85/logbooks/4-native-jax-llama/prepared/core-fp8-prelaunch/environment.json) and [setup recipe](https://github.com/marin-community/marin-logbooks/blob/5488bb8d96f74ab35088ec8ed0f1be457405cc85/logbooks/4-native-jax-llama/prepared/core-fp8-prelaunch/recipe.md). Those records describe a reused environment, not a full lock-matched installation or a runtime qualification of this worker. The [ba58 installation guide](https://github.com/dlwh/marin/blob/ba58e83e3310163e8d674ea9e95a85094d236a7a/docs/tutorials/installation.md) and [lockfile](https://github.com/dlwh/marin/blob/ba58e83e3310163e8d674ea9e95a85094d236a7a/uv.lock) are source references. Do not install or sync dependencies during the control.

The execution owner is the campaign's coordinating agent. Before execution, that owner records the exact public source and research commit, selected CPU count, command, fresh output path and sandbox boundary. This is an internal release record under the campaign's existing local-CPU authorization; publication alone does not release a command. CPU1 and CPU4 require separate records. All TPU/job/monitor windows remain paused.

Set `FP8_SOURCE_ROOT` to clean exact public ba58 source, `FP8_RESEARCH_ROOT` to this published directory, `FP8_OUTPUT_ROOT` to a fresh private output parent and `FP8_VENV` to the installed environment. Set `FP8_UV` to the existing modern uv executable and `FP8_TIMEOUT` to the existing GNU timeout executable (`gtimeout` on the preparation host). Create the output parent only after release; each case directory must be absent. Use the task execution tool's default sandbox with networking denied and record that boundary. Offline flags below add client controls; they do not establish OS network denial.

```sh
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
```

After recording CPU1 release:

```sh
JAX_NUM_CPU_DEVICES=1 "$FP8_TIMEOUT" --signal=TERM --kill-after=5s 180s "$FP8_UV" run --no-sync --offline --directory "$FP8_SOURCE_ROOT" python "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-probe.py" microbatch --source-root "$FP8_SOURCE_ROOT" --research-root "$FP8_RESEARCH_ROOT" --config "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-config.json" --devices 1 --output "$FP8_OUTPUT_ROOT/microbatch-cpu1" > "$FP8_OUTPUT_ROOT/microbatch-cpu1.log" 2>&1
```

Stop on the first meaningful failure, including an expected metadata mismatch. Preserve the staged receipt/log and classify completed prerequisites. Do not retry, alter the fixture or execute a candidate fix as automatic recovery. A timeout without a completed receipt is a separate bounded-command failure. Review CPU1 before separately releasing CPU4:

```sh
JAX_NUM_CPU_DEVICES=4 "$FP8_TIMEOUT" --signal=TERM --kill-after=5s 180s "$FP8_UV" run --no-sync --offline --directory "$FP8_SOURCE_ROOT" python "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-probe.py" microbatch --source-root "$FP8_SOURCE_ROOT" --research-root "$FP8_RESEARCH_ROOT" --config "$FP8_RESEARCH_ROOT/native-jax-fp8-microbatch-preparation-config.json" --devices 4 --output "$FP8_OUTPUT_ROOT/microbatch-cpu4" > "$FP8_OUTPUT_ROOT/microbatch-cpu4.log" 2>&1
```

The runner verifies clean source/file/worker hashes before loading the worker. That same process checks own-source module provenance, CPU count/platform and absence of an attached Iris publishing runtime. Only the runner's `microbatch` command is proposed. It performs no Iris job/client/login/monitor or accelerator operation. Native model quantization, BF16, stacked histories, optimizer/checkpoint continuation, varied-scale selection, input differentiation, accelerator FP8 kernels/performance and canonical scientific acceptance remain outside this control.
