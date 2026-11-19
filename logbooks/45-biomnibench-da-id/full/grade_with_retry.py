# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive each judge response and apply the selected frozen-submission retry policy."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from llm_judge import JudgeConfig, JudgeResponseError, evaluate
from openai import OpenAI


def grade(rubric: Path, submission: Path, root: Path, config: JudgeConfig, client: OpenAI, *, frozen_retries: int = 0):
    # A failed directory inspection must not masquerade as a confirmed empty answer.
    entries = sorted(path.name for path in submission.iterdir())
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission-inspection.json").write_text(json.dumps({"status": "inspected", "files": entries}))
    if frozen_retries not in (0, 2):
        raise ValueError("Expected zero or two frozen-submission retries")
    for attempt in range(2 + frozen_retries):
        output = root / "judge-attempts" / str(attempt)
        try:
            source = submission if attempt < 2 else root / "judge-attempts" / "0"
            evaluate(rubric, source, output, config, client)
        except JudgeResponseError:
            if attempt == 1 + frozen_retries:
                raise
        else:
            evaluation = json.loads((output / "evaluation.json").read_text())
            evaluation["judge_attempts"] = attempt + 1
            (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
            shutil.copyfile(output / "reward.json", root / "reward.json")
            return
        finally:
            # Root copies support independent regrading even when both attempts fail.
            for name in ("trace.md", "answer.txt"):
                if (output / name).exists():
                    shutil.copyfile(output / name, root / name)


def main():
    config = JudgeConfig(**json.loads(Path("/tests/judge-config.json").read_text()))
    config = replace(config, model=os.environ["MODEL_NAME"], base_url=os.environ["OPENAI_BASE_URL"])
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=config.base_url, timeout=config.timeout_sec, max_retries=0
    ) as client:
        grade(
            Path("/tests/rubric.txt"),
            Path("/app"),
            Path("/logs/verifier"),
            config,
            client,
            frozen_retries=int(os.environ.get("BIOMNI_FROZEN_JUDGE_RETRIES", "0")),
        )


if __name__ == "__main__":
    main()
