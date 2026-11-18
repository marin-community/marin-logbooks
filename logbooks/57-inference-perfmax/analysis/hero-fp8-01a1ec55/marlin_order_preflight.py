# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU readback and serialization checks before the one-GPU order probe."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def main():
    directory = Path(__file__).parent
    source = directory / "marlin_history_bootstrap.py"
    tree = ast.parse(source.read_text())
    compile(tree, str(source), "exec")
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"maximum_difference", "_canonicalize_marlin_moe_token_order"}
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["maximum_difference"](torch.tensor([1, -2]), torch.tensor([1, -1])) == 1
    canonical = namespace["_canonicalize_marlin_moe_token_order"](
        torch.tensor([8, 3, 9, 1, 7, 4, 9, 6, 5, 2, 9, 0, -999, -2, -3, -4], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        torch.tensor([12], dtype=torch.int32),
        4,
        9,
    )
    expected = [1, 3, 4, 6, 7, 8, 9, 9, 0, 2, 5, 9]
    assert canonical[:12].tolist() == expected
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    local_names = {
        node.id for node in ast.walk(main_node) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    shadowed = sorted(local_names & functions)
    assert not shadowed, shadowed

    # Execute the actual order-row expression, with the old shadowing trigger
    # present in the local namespace, then round-trip the whole record as JSON.
    row_expression = next(
        node
        for node in ast.walk(main_node)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "weight_mode" for key in node.value.keys)
    )
    namespace.update(
        mode="random-bf16-source",
        route="hot-local-four-mixed",
        count=115,
        block=8,
        used=8,
        regions=1,
        maximum=2048,
        ascending=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        descending=torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]),
        first=(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        same=(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        different=(torch.tensor([1.0, 2.25]), torch.tensor([3.0, 4.5])),
        canonical_first=(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        canonical_second=(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
    )
    exec(compile(ast.Module(body=[row_expression], type_ignores=[]), str(source), "exec"), namespace)
    serialized = json.loads(json.dumps({"layout_order": [namespace["row"]]}))
    assert serialized["layout_order"][0]["reversed_order_w13_max"] == 0.25
    assert serialized["layout_order"][0]["reversed_order_output_max"] == 0.5
    assert serialized["layout_order"][0]["same_order_output_max"] == 0

    old_source = subprocess.check_output(
        ["git", "show", f"1b533f51280474e6cda81ca74243c6ca209d0157:{source.resolve().relative_to(Path.cwd())}"]
    )
    # The native-quantized weights, input RNG and routing probabilities were
    # constructed before the history loops. Their AST is unchanged by reuse.
    old_main = next(
        node for node in ast.parse(old_source).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def input_control(main_function):
        initialization = next(
            node
            for node in main_function.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Tuple) and len(target.elts) == 6 for target in node.targets)
        )
        context = next(node for node in main_function.body if isinstance(node, ast.With))
        statements = []
        for node in context.body:
            if not isinstance(node, ast.Assign):
                break
            if any(isinstance(target, ast.Name) and target.id == "reused" for target in node.targets):
                break
            statements.append(node)
        return ast.dump(ast.Module(body=[initialization, *statements], type_ignores=[]), include_attributes=False)

    controls = input_control(main_node)
    assert controls == input_control(old_main)
    record = {
        "scope": (
            "CPU syntax, unchanged input controls, actual readback-row and JSON serialization; no GPU numerical claim"
        ),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "previous_source": "1b533f51280474e6cda81ca74243c6ca209d0157",
        "input_control_ast_sha256": hashlib.sha256(controls.encode()).hexdigest(),
        "input_controls_unchanged": True,
        "canonical_active_order": expected,
        "global_function_shadowing_in_main": shadowed,
        "serialized_order_row": serialized["layout_order"][0],
        "status": "passed",
    }
    (directory / "marlin-order-a3-cpu-preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
