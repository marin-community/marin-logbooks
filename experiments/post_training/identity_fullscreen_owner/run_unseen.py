# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the frozen unseen screen only; never issue dev-generation requests."""

import argparse
import hashlib
import json
from pathlib import Path

from rigging.secrets import require_secret

from experiments.interp.snowball_identity import identity_eval as shared

UNSEEN_DIR = Path(__file__).resolve().parent / "data/unseen-third-person"
UNSEEN_SHA256 = {
    "identity.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "controls.jsonl": "042ecdb5883e671bf5db582c3a57efac637dcee118299292dde17703971e9511",
}


def checked_unseen_rows():
    for name, expected in UNSEEN_SHA256.items():
        if hashlib.sha256((UNSEEN_DIR / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Committed unseen inputs differ from the historical screen")
    rows = shared.load_rows(UNSEEN_DIR, "dev")
    if len(rows) != 30:
        raise ValueError("Unseen continuation requires the exact historical30 controls")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checked_unseen_rows()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "candidate-binding.json").write_text(
        json.dumps({"candidate_weights": args.model, "label": args.label, "dev_regeneration": False}) + "\n"
    )
    callback = shared.generate.callback
    if callback is None:
        raise RuntimeError("Frozen screen has no generation callback")
    callback(
        base_url=require_secret("IDENTITY_ENDPOINT").rstrip("/"),
        model=None,
        label=args.label,
        out=args.out / "unseen-third-person",
        prompt_dir=UNSEEN_DIR,
        split="dev",
        workers=16,
        conditions=("no_system", "system_baseline"),
    )


if __name__ == "__main__":
    main()
