# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One A3 identity132 CPU owner; no training or output repair."""

import argparse
import json
from functools import partial

from experiments.post_training.identity_path_b.checkpoints import CheckpointEvent
from experiments.post_training.identity_path_b.identity_scope import owner, run
from experiments.post_training.identity_path_b.prefill_scope import serve
from experiments.post_training.identity_path_b.prefill_scope.data import template
from experiments.post_training.identity_path_b.prefill_scope.run import PrefillSession


def submission(*, name: str, output: str, credentials: dict) -> dict:
    request = owner.screen_submission(name=name, arm="A3", output=output, credentials=credentials)
    old = "experiments.post_training.identity_path_b.identity_scope.owner "
    command = request["entrypoint"].command[-1]
    if command.count(old) != 1:
        raise ValueError("Unexpected native identity owner entrypoint")
    request["entrypoint"].command[-1] = command.replace(
        old, "experiments.post_training.identity_path_b.prefill_scope.owner "
    )
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A3",), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    template()
    if args.inspect_only:
        with run.payload() as (root, _):
            event = CheckpointEvent(**json.loads((root / "A3.json").read_bytes()))
            config = serve.service_config(event)
            print(
                json.dumps(
                    {"arm": "A3", "model": config.model.weights, "output": args.output, "prefill": True, "calls": 0}
                )
            )
        return
    owner.screen_worker(
        arm="A3",
        output=args.output,
        generator=partial(run.generate, session_factory=PrefillSession),
        serve_submission=serve.submission,
    )


if __name__ == "__main__":
    main()
