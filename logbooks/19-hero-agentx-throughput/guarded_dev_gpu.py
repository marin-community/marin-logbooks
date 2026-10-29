# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the native dev-GPU CLI with a checked node list and one holder attempt."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-cli", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    args, native_args = parser.parse_known_args()
    manifest = json.loads(args.placement.read_text())
    placement = manifest.get("placement", manifest)
    nodes = placement["nodes"]
    assert nodes and placement["hero_training_gpus"] == 0
    spec = importlib.util.spec_from_file_location("hero_native_dev_gpu", args.native_cli)
    assert spec is not None and spec.loader is not None
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    command = (
        "import os; assert os.environ.get('IRIS_NODE_NAME') in "
        + repr(nodes)
        + ", 'Holder outside verified spare nodes'; "
        + native.HOLDER_COMMAND
    )
    compile(command, "guarded-holder", "exec")
    native.HOLDER_COMMAND = command
    submit = native.IrisClient.submit

    def single_attempt_submit(client, *positional, **kwargs):
        assert kwargs["name"].startswith("dev-gpu-")
        kwargs["max_retries_failure"] = 0
        kwargs["max_retries_preemption"] = 0
        return submit(client, *positional, **kwargs)

    native.IrisClient.submit = single_attempt_submit
    sys.argv = [str(args.native_cli), *native_args]
    native.main()


if __name__ == "__main__":
    main()
