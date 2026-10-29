# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Existing9e3857 full dev/unseen capture; endpoint capability is env/secret only."""

import argparse
import json
from pathlib import Path

from rigging.secrets import require_secret

from experiments.interp.snowball_identity import identity_eval as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--unseen-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "candidate-binding.json").write_text(
        json.dumps({"candidate_weights": args.model, "label": args.label}) + "\n"
    )
    endpoint = require_secret("IDENTITY_ENDPOINT").rstrip("/")
    for name, prompts in [
        ("generate", args.prompt_dir),
        ("unseen-third-person", args.unseen_dir),
    ]:
        shared.generate.callback(
            base_url=endpoint,
            model=None,
            label=args.label,
            out=args.out / name,
            prompt_dir=prompts,
            split="dev",
            workers=16,
            conditions=("no_system", "system_baseline"),
        )


if __name__ == "__main__":
    main()
