# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise every selected source's real grader before allocating study GPUs."""

import argparse
from dataclasses import asdict

import pyarrow.parquet as pq
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import require_secret
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, grades_with_judge
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.task_grading import grade_task
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.sc_default_data import SOURCE_FAMILIES, SOURCE_URI, write_json


def check_graders(output_path: str) -> None:
    filesystem, path = filesystem_for(SOURCE_URI + "/train.parquet")
    selected = {}
    with filesystem.open(path, "rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                task = TaskSpec.model_validate_json(row["task_spec"])
                source = task.id.rsplit("-", 1)[0]
                selected.setdefault(source, task)
            if selected.keys() == SOURCE_FAMILIES.keys():
                break
    if selected.keys() != SOURCE_FAMILIES.keys():
        raise ValueError("The grader gate did not cover every selected source")
    machines = LocalGraderMachines()
    connection = JudgeConnection("https://api.together.xyz/v1", require_secret("TOGETHER_API_KEY"))
    results = []
    for source, task in selected.items():
        reference = reference_reply(task)
        event = (
            reference.event
            if reference is not None
            else TextMessage(role="assistant", content="SC study invalid-answer grading control.")
        )
        factory, specification = None, None
        if task.grader.environment is not None and not grades_with_judge(task.grader):
            factory, specification = machines.machine(task.grader.environment, 4096)
        grade = grade_task(
            task,
            GradingAttempt(ConversationTrace(events=(*task.context.events, event))),
            machine_factory=factory,
            machine_spec=specification,
            judge_connection=connection,
        )
        record = {"source": source, "task_id": task.id, "reference_control": reference is not None, **asdict(grade)}
        results.append(record)
        write_json(output_path.rstrip("/") + "/grader-readiness.json", results)
        print(source, grade.status, grade.reward, grade.error, flush=True)
        if grade.status is not Outcome.GRADED or grade.reward is None:
            raise RuntimeError(f"The canonical grader is unavailable for {source}: {grade.error}")
        if reference is not None and grade.reward != 1:
            raise RuntimeError(f"The reference control failed for {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    check_graders(parser.parse_args().output_path)
