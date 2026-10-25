# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8 diagnostic worker; imported only inside the runner's recorded preflight."""

import dataclasses
import json
import math
from pathlib import Path

import equinox as eqx
import haliax
import haliax._src.fp8 as fp8_source
import haliax.quantization as quantization_source
import iris
import jax
import jax.numpy as jnp
import levanter
import levanter.grad_accum as grad_accum_source
import levanter.pipeline as pipeline_source
import marin
import numpy as np
from haliax.quantization import (
    Fp8DotGeneralOp,
    QuantizationConfig,
    accumulate_gradients,
    apply_updates,
    partition_for_grad_overwrite,
)
from iris.runtime.telemetry import resolve
from jax.sharding import AxisType, Mesh, NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.grad_accum import microbatched_arrays
from levanter.main import train_lm
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.testing import tiny_corpus
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig


def execute(args, receipt, checkpoint):
    STATE_FIELDS = (
        "input_scale",
        "kernel_scale",
        "output_grad_scale",
        "input_amax_history",
        "kernel_amax_history",
        "output_grad_amax_history",
    )

    def host_state(operation):
        return {name: np.asarray(getattr(operation, name)).tolist() for name in STATE_FIELDS}

    def max_difference(left, right):
        return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))

    def placed(value, mesh, spec):
        return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))

    def operator(mesh, *, history_length, nonunit):
        operation = Fp8DotGeneralOp.init(amax_history_length=history_length)
        if nonunit:
            operation = dataclasses.replace(
                operation,
                input_scale=jnp.array([2.0], jnp.float32),
                kernel_scale=jnp.array([0.5], jnp.float32),
                output_grad_scale=jnp.array([4.0], jnp.float32),
            )
        return jax.tree.map(lambda value: placed(value, mesh, P()), operation)

    def baseline(root, output):
        checkpoint("baseline.local_assets")
        tokenizer = output / "tokenizer"
        tokenizer.mkdir()
        stage_gpt2_tokenizer(root / "lib/levanter/tests", tokenizer)
        data, _ = tiny_corpus.construct_small_data_cache(str(output / "data"), tokenizer=str(tokenizer))
        model = LlamaConfig(
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=64,
            hidden_dim=32,
            attn_backend=None,
            reference_checkpoint=str(tokenizer),
            tokenizer=str(tokenizer),
        )

        def config(model_config, name):
            directory = output / name
            directory.mkdir()
            return train_lm.TrainLmConfig(
                data=data,
                model=model_config,
                trainer=TrainerConfig(
                    quantization=QuantizationConfig(fp8=True),
                    num_train_steps=2,
                    train_batch_size=jax.device_count(),
                    max_eval_batches=1,
                    tracker=JsonFileTrackerConfig(output_path=str(directory)),
                    checkpointer=CheckpointerConfig(base_path=str(directory / "checkpoints")),
                    require_accelerator=False,
                    distributed=DistributedConfig(initialize_jax_distributed=False),
                    shutdown_at_exit=False,
                    log_dir=directory / "logs",
                    id=name,
                ),
            )

        checkpoint("baseline.canonical.train")
        train_lm.main(config(model, "canonical"))
        checkpoint("baseline.canonical.metrics")
        metrics = json.loads((output / "canonical/eval_results.json").read_text())
        assert metrics["parameter_count"] > 0
        assert "train/loss" in metrics and math.isfinite(metrics["train/loss"])
        receipt["completed_controls"]["canonical"] = {"outcome": "finite_loss_smoke", "metrics": metrics}
        checkpoint("baseline.native.configuration")
        native_model = NativeLlamaLmConfig(**{f.name: getattr(model, f.name) for f in dataclasses.fields(model)})
        checkpoint("baseline.native.public_call")
        try:
            train_lm.main(config(native_model, "native"))
        except ValueError as error:
            native_result = {
                "outcome": "rejected",
                "exception": type(error).__name__,
                "message": str(error),
                "matches_expected_quantization_guard": (
                    str(error) == "llama_native training does not support Haliax quantization"
                ),
            }
        else:
            native_result = {"outcome": "accepted_unexpected_for_pinned_source"}
        receipt["completed_controls"]["native"] = native_result
        checkpoint("baseline.completed")
        return {
            "canonical": {"outcome": "finite_loss_smoke", "metrics": metrics},
            "native": native_result,
            "boundary": "Two-step public FP8 smoke; no numerical/gradient/resume or hardware parity claim.",
        }

    def raw_control_placement(value, mesh, spec, axis_type):
        if axis_type == AxisType.Auto:
            return jax.lax.with_sharding_constraint(value, NamedSharding(mesh, spec))
        return reshard(value, spec)

    def raw_forward_and_state(devices):
        tensor_parallel = 2 if len(devices) == 4 else 1
        data_parallel = len(devices) // tensor_parallel
        shape = (data_parallel, tensor_parallel)
        table = np.array([-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3], np.float32)
        cot_table = np.array([-2, -1, -0.5, 0.5, 1, 2], np.float32)
        x = np.resize(table, (8 * data_parallel, 16))
        weight = np.resize(table[::-1], (8 * tensor_parallel, 16))
        cot = np.resize(cot_table, (x.shape[0], weight.shape[0]))
        expected = (x @ weight.T, cot @ weight, cot.T @ x)
        outcomes = receipt["completed_controls"]
        states = {}
        for kind in ("auto", "explicit"):
            checkpoint(f"raw.{kind}.placement", control_kind=kind)
            selected = devices[:1] if kind == "auto" else devices
            sizes = (1, 1) if kind == "auto" else shape
            axis_type = AxisType.Auto if kind == "auto" else AxisType.Explicit
            mesh = Mesh(np.array(selected).reshape(sizes), ("data", "model"), axis_types=(axis_type, axis_type))
            with jax.set_mesh(mesh):
                op = operator(mesh, history_length=16, nonunit=True)
                inputs = placed(x, mesh, P("data", None))
                stored_weight = placed(weight, mesh, P("model", "data"))
                cotangent = placed(cot, mesh, P("data", "model"))

                def forward(state, *, mesh=mesh, axis_type=axis_type):
                    operation, input_values, parameter = state
                    parameter = raw_control_placement(parameter, mesh, P("model", None), axis_type)
                    values = operation(input_values, parameter, (((1,), (1,)), ((), ())))
                    return raw_control_placement(values, mesh, P("data", "model"), axis_type)

                state = (op, inputs, stored_weight)
                checkpoint(f"raw.{kind}.forward", control_kind=kind)
                values = eqx.filter_jit(forward)(state)
                checkpoint(f"raw.{kind}.gradients", control_kind=kind)
                gradients = eqx.filter_jit(
                    eqx.filter_grad(lambda s, cotangent=cotangent: jnp.sum(forward(s) * cotangent))
                )(state)
                checkpoint(f"raw.{kind}.value_gradient_assertions", control_kind=kind)
                op_gradient, input_gradient, weight_gradient = gradients
                for actual, want in zip((values, input_gradient, weight_gradient), expected, strict=True):
                    np.testing.assert_array_equal(np.asarray(actual), want)
                checkpoint(f"raw.{kind}.state_assertions", control_kind=kind)
                for name, maximum in (
                    ("input", np.max(np.abs(x))),
                    ("kernel", np.max(np.abs(weight))),
                    ("output_grad", np.max(np.abs(cot))),
                ):
                    history = np.zeros(16, np.float32)
                    history[0] = maximum
                    np.testing.assert_array_equal(np.asarray(getattr(op_gradient, name + "_amax_history")), history)
                    np.testing.assert_array_equal(getattr(op_gradient, name + "_scale"), getattr(op, name + "_scale"))
                overwritten = apply_updates(op, None, op_gradient)
                states[kind] = host_state(overwritten)
                outcomes[kind] = {
                    "shape": list(sizes),
                    "axis_type": str(axis_type),
                    "max_absolute_errors": [
                        max_difference(a, b)
                        for a, b in zip((values, input_gradient, weight_gradient), expected, strict=True)
                    ],
                    "state_after_overwrite": states[kind],
                    "output_sharding": str(values.sharding),
                    "weight_gradient_sharding": str(weight_gradient.sharding),
                }
                checkpoint(f"raw.{kind}.completed", control_kind=kind)
        checkpoint("raw.auto_explicit_state_identity", control_kind="both")
        assert states["auto"] == states["explicit"]
        return {
            "outcome": "exact_representable_values_gradients_and_state",
            "controls": outcomes,
            "boundary": "Raw existing FP8 operation only; no NativeLinear/converter integration or hardware claim.",
        }

    def microbatch_state(devices):
        checkpoint("microbatch.explicit.placement", control_kind="explicit")
        tensor_parallel = 2 if len(devices) == 4 else 1
        data_parallel = len(devices) // tensor_parallel
        mesh = Mesh(
            np.array(devices).reshape(data_parallel, tensor_parallel),
            ("data", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        microbatch_size = 2 * data_parallel
        with jax.set_mesh(mesh):
            op = operator(mesh, history_length=4, nonunit=True)
            seed_histories = {
                "input": np.array([896, 448, 224, 112], np.float32),
                "kernel": np.array([224, 112, 56, 28], np.float32),
                "output_grad": np.array([229376, 114688, 57344, 28672], np.float32),
            }
            op = dataclasses.replace(
                op,
                **{name + "_amax_history": placed(history, mesh, P()) for name, history in seed_histories.items()},
            )
            weight_host = np.full((4, 8), 0.125, np.float32)
            weight = placed(weight_host, mesh, P("model", "data"))
            params = (op, weight)
            values = np.concatenate(
                [np.full((microbatch_size, 8), 32, np.float32), np.ones((microbatch_size, 8), np.float32)]
            )
            inputs = placed(values, mesh, P("data", None))

            def evaluate(batch):
                def loss(parameters):
                    operation, parameter = parameters
                    output = operation(
                        batch,
                        reshard(parameter, P("model", None)),
                        (((1,), (1,)), ((), ())),
                        out_sharding=P("data", "model"),
                    )
                    return jnp.mean(output**2), {}

                return eqx.filter_value_and_grad(loss, has_aux=True)(params)

            def independent_reference(batch):
                output = batch @ weight_host.T
                cotangent = np.float32(2) * output / np.float32(output.size)
                state = {}
                for name, maximum, retained_scale in (
                    ("input", np.max(np.abs(batch)), 2),
                    ("kernel", np.max(np.abs(weight_host)), 0.5),
                    ("output_grad", np.max(np.abs(cotangent)), 4),
                ):
                    # History maxima are FP8 maxima * retained scales: 448*2, 448*.5, 57344*4.
                    # Thus the delayed scale remains the nonunit seed for each independent call.
                    history = np.roll(seed_histories[name], -1)
                    history[0] = maximum
                    state[name + "_amax_history"] = history
                    state[name + "_scale"] = np.array([retained_scale], np.float32)
                return np.mean(output**2), cotangent.T @ batch, state

            def check_placement(value, spec, label):
                expected = NamedSharding(mesh, spec)
                observed = value.sharding
                receipt.setdefault("placements", {})[label] = {
                    "observed": str(observed),
                    "expected": str(expected),
                    "shape": list(value.shape),
                    "local_shard_shapes": [list(shard.data.shape) for shard in value.addressable_shards],
                }
                assert observed.is_equivalent_to(expected, value.ndim), label
                assert set(observed.device_set) == set(devices), label
                assert all(shard.data.shape == expected.shard_shape(value.shape) for shard in value.addressable_shards)

            def check_result(result, reference, label):
                (loss, _), (operation_gradient, weight_gradient) = result
                expected_loss, expected_weight_gradient, expected_state = reference
                checkpoint(f"microbatch.explicit.{label}.ordinary_assertions", control_kind="explicit")
                np.testing.assert_array_equal(np.asarray(loss), expected_loss)
                np.testing.assert_array_equal(np.asarray(weight_gradient), expected_weight_gradient)
                check_placement(loss, P(), label + ".loss")
                check_placement(weight_gradient, P("model", "data"), label + ".weight_gradient")
                for name in STATE_FIELDS:
                    actual = getattr(operation_gradient, name)
                    check_placement(actual, P(), label + "." + name)
                    np.testing.assert_array_equal(np.asarray(actual), expected_state[name])
                receipt["completed_controls"][label] = {
                    "outcome": "independent_exact_loss_gradient_state_and_placement",
                    "loss": float(loss),
                    "state": host_state(operation_gradient),
                }
                checkpoint(f"microbatch.explicit.{label}.completed", control_kind="explicit")

            check_placement(inputs, P("data", None), "inputs")
            check_placement(weight, P("model", "data"), "stored_weight")
            checkpoint("microbatch.explicit.no_microbatch.direct", control_kind="explicit")
            full_reference = independent_reference(values)
            direct_full = eqx.filter_jit(evaluate)(inputs)
            check_result(direct_full, full_reference, "direct_full")
            checkpoint("microbatch.explicit.no_microbatch.wrapper", control_kind="explicit")
            unsplit = eqx.filter_jit(microbatched_arrays(evaluate, values.shape[0]))(inputs)
            check_result(unsplit, full_reference, "no_microbatch_wrapper")

            per_microbatch = []
            per_reference = []
            for index, start in enumerate((0, microbatch_size)):
                checkpoint(f"microbatch.explicit.independent_call_{index}", control_kind="explicit")
                batch_host = values[start : start + microbatch_size]
                batch = placed(batch_host, mesh, P("data", None))
                result = eqx.filter_jit(evaluate)(batch)
                reference = independent_reference(batch_host)
                check_result(result, reference, f"per_microbatch_{index}")
                per_microbatch.append(result)
                per_reference.append(reference)

            independent_loss = (per_reference[0][0] + per_reference[1][0]) / np.float32(2)
            independent_gradient = (per_reference[0][1] + per_reference[1][1]) / np.float32(2)
            independent_state = {
                name: (
                    np.maximum(per_reference[0][2][name], per_reference[1][2][name])
                    if name.endswith("_amax_history")
                    else per_reference[1][2][name]
                )
                for name in STATE_FIELDS
            }
            checkpoint("microbatch.explicit.shared_reference", control_kind="explicit")
            reduced = accumulate_gradients(per_microbatch[0][1], per_microbatch[1][1])
            overwrites, additive = partition_for_grad_overwrite(reduced)
            reference_gradients = eqx.combine(overwrites, jax.tree.map(lambda leaf: leaf / 2, additive))
            reference_loss = (per_microbatch[0][0][0] + per_microbatch[1][0][0]) / 2
            check_result(
                ((reference_loss, {}), reference_gradients),
                (independent_loss, independent_gradient, independent_state),
                "shared_reducer",
            )

            checkpoint("microbatch.explicit.actual_reducer", control_kind="explicit")
            (actual_loss, _), actual_gradients = eqx.filter_jit(microbatched_arrays(evaluate, microbatch_size))(inputs)
            checkpoint("microbatch.explicit.actual_ordinary_assertions", control_kind="explicit")
            np.testing.assert_array_equal(np.asarray(actual_loss), independent_loss)
            np.testing.assert_array_equal(np.asarray(actual_gradients[1]), independent_gradient)
            check_placement(actual_loss, P(), "raw.loss")
            check_placement(actual_gradients[1], P("model", "data"), "raw.weight_gradient")
            for name in STATE_FIELDS:
                check_placement(getattr(actual_gradients[0], name), P(), "raw." + name)
            observed = host_state(actual_gradients[0])
            expected = {name: value.tolist() for name, value in independent_state.items()}
            averaging_prediction = {
                name: ((per_reference[0][2][name] + per_reference[1][2][name]) / np.float32(2)).tolist()
                for name in STATE_FIELDS
            }
            matches = observed == expected
            receipt["microbatch_diagnostic"] = {
                "outcome": "state_reduction_matches" if matches else "state_reduction_mismatch_observed",
                "ordinary_loss_and_additive_gradient_controls": "independent_exact_match",
                "microbatch_size": microbatch_size,
                "actual_loss": float(actual_loss),
                "actual_state": observed,
                "independent_max_history_retained_scale_reference": expected,
                "independent_average_all_leaves_prediction": averaging_prediction,
                "matches_averaging_prediction": observed == averaging_prediction,
                "boundary": "Same-recipe per-microbatch state only; no full-batch FP8 trajectory equivalence assumed.",
            }
            checkpoint("microbatch.explicit.state_classification", control_kind="explicit")
            assert matches, "Raw FP8 metadata differs from independent max-history/retained-scale reference"
            return receipt["microbatch_diagnostic"]

    checkpoint("preflight.module_provenance")
    modules = {
        module.__name__: module.__file__
        for module in (
            haliax,
            levanter,
            marin,
            iris,
            quantization_source,
            fp8_source,
            grad_accum_source,
            pipeline_source,
        )
    }
    receipt["source_modules"] = modules
    checkpoint("preflight.module_provenance")
    assert all(Path(path).resolve().is_relative_to(args.source_root) for path in modules.values()), modules
    checkpoint("preflight.cpu_devices")
    devices = jax.devices()
    receipt["actual_device_count"] = jax.device_count()
    receipt["actual_devices"] = [str(device) for device in devices]
    receipt["device_platforms"] = [device.platform for device in devices]
    checkpoint("preflight.cpu_devices")
    assert jax.device_count() == args.devices and all(device.platform == "cpu" for device in devices)
    checkpoint("preflight.no_publishing_runtime")
    receipt["iris_publishing_runtime"] = resolve() is not None
    checkpoint("preflight.no_publishing_runtime")
    assert not receipt["iris_publishing_runtime"], "No Iris publishing runtime permitted"
    receipt["preflight_complete"] = True
    checkpoint("diagnostic.dispatch")
    if args.mode == "baseline":
        return baseline(args.source_root, args.output)
    if args.mode == "raw":
        return raw_forward_and_state(devices)
    return microbatch_state(devices)
