# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8 diagnostic worker; imported only inside the runner's recorded preflight."""

import dataclasses
import json
import math
from pathlib import Path

import equinox as eqx
import haliax
import iris
import jax
import jax.numpy as jnp
import levanter
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

                def forward(state):
                    operation, input_values, parameter = state
                    parameter = reshard(parameter, P("model", None))
                    values = operation(input_values, parameter, (((1,), (1,)), ((), ())))
                    return reshard(values, P("data", "model"))

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
            op = operator(mesh, history_length=4, nonunit=False)
            weight = placed(np.full((4, 8), 0.125, np.float32), mesh, P("model", "data"))
            params = (op, weight)
            values = np.concatenate(
                [np.full((microbatch_size, 8), 32, np.float32), np.ones((microbatch_size, 8), np.float32)]
            )
            inputs = placed(values, mesh, P("data", None))

            def evaluate(batch):
                def loss(parameters):
                    operation, parameter = parameters
                    output = operation(batch, reshard(parameter, P("model", None)), (((1,), (1,)), ((), ())))
                    output = reshard(output, P("data", "model"))
                    return jnp.mean(output**2), {}

                return eqx.filter_value_and_grad(loss, has_aux=True)(params)

            checkpoint("microbatch.explicit.actual_reducer", control_kind="explicit")
            (actual_loss, _), actual_gradients = eqx.filter_jit(microbatched_arrays(evaluate, microbatch_size))(inputs)
            checkpoint("microbatch.explicit.shared_reference", control_kind="explicit")
            per_microbatch = [evaluate(inputs[start : start + microbatch_size]) for start in (0, microbatch_size)]
            reduced = accumulate_gradients(per_microbatch[0][1], per_microbatch[1][1])
            overwrites, additive = partition_for_grad_overwrite(reduced)
            averaged = jax.tree.map(lambda leaf: leaf / 2, additive)
            reference = eqx.combine(overwrites, averaged)
            checkpoint("microbatch.explicit.reference_assertion", control_kind="explicit")
            np.testing.assert_array_equal(reference[0].input_amax_history, np.array([32, 0, 0, 0], np.float32))
            expected_loss = (per_microbatch[0][0][0] + per_microbatch[1][0][0]) / 2
            checkpoint("microbatch.explicit.loss_assertion", control_kind="explicit")
            np.testing.assert_array_equal(actual_loss, expected_loss)
            receipt["completed_controls"]["microbatch_loss"] = {"outcome": "exact_match", "value": float(actual_loss)}
            checkpoint("microbatch.explicit.additive_gradient_assertion", control_kind="explicit")
            receipt["additive_weight_gradient_max_absolute_difference"] = max_difference(
                actual_gradients[1], reference[1]
            )
            checkpoint("microbatch.explicit.additive_gradient_assertion", control_kind="explicit")
            np.testing.assert_array_equal(np.asarray(actual_gradients[1]), np.asarray(reference[1]))
            receipt["completed_controls"]["microbatch_additive_gradient"] = {"outcome": "exact_match"}
            checkpoint("microbatch.explicit.state_classification", control_kind="explicit")
            observed = host_state(actual_gradients[0])
            expected = host_state(reference[0])
            return {
                "outcome": "state_reduction_matches" if observed == expected else "state_reduction_mismatch_observed",
                "ordinary_loss_and_additive_gradient_controls": "exact_match",
                "microbatch_size": microbatch_size,
                "actual_loss": float(actual_loss),
                "actual_state": observed,
                "shared_custom_reduction_reference": expected,
                "weight_gradient_max_absolute_difference": max_difference(actual_gradients[1], reference[1]),
                "boundary": (
                    "Mismatch is only observed if this prepared case is later executed successfully; "
                    "exceptions remain separate."
                ),
            }

    checkpoint("preflight.module_provenance")
    modules = {module.__name__: module.__file__ for module in (haliax, levanter, marin, iris)}
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
