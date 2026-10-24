# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deliberate CPU-only failures of actual tap identity and callback consistency."""

import argparse
import json
from pathlib import Path

import capture
import child
import jax
import jax.numpy as jnp
import numpy as np
from diagnostic import compare, retention, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, choices=("gradient-perturbation", "duplicate-disagreement", "nonfinite")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"case": args.case, "completed": False, "deliberate_cpu_mechanical_fault": True}
    try:
        if args.case == "duplicate-disagreement":
            receive = capture.Collector.receive

            def disagree(self, label, partitioned, value, index, device_ids):
                identity = int(np.asarray(device_ids).reshape(-1)[0]) if partitioned else "global"
                if (label, identity) in self.values:
                    value = np.asarray(value).copy()
                    value.view(np.uint16).flat[0] ^= 1
                receive(self, label, partitioned, value, index, device_ids)

            capture.Collector.receive = disagree
        else:
            make_tap = capture.make_tap

            def faulty_tap(kind, mesh):
                tap = make_tap(kind, mesh)
                if kind != "native":
                    return tap

                @jax.custom_vjp
                def fault(y):
                    return y

                def forward(y):
                    return y, ()

                def backward(_residual, g):
                    value = jnp.nan if args.case == "nonfinite" else 1.0
                    return (g + jnp.asarray(value, dtype=g.dtype),)

                fault.defvjp(forward, backward)
                return lambda y, x, index: fault(tap(y, x, index))

            capture.make_tap = faulty_tap
        namespace = argparse.Namespace(output=args.output, config=args.config, mode="cpu", phase="instrumented")
        child.run(namespace)
        compare(json.loads(args.anchor.read_text()), json.loads((args.output / "report.json").read_text()))
        raise AssertionError("Deliberately wrong real computation/capture was accepted")
    except Exception as error:
        receipt["observed_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "probe-receipt.json", receipt)
        retention(args.output)


if __name__ == "__main__":
    main()
