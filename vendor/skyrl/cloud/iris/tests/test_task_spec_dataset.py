"""Load canonical task rows into the rollout worker's execution contract."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.validation import validate_spec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec
from skyrl_gym.rollout_specs import is_shell_harness
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from taskcompendium.models import CommandSemantics, ConversationInput, EnvironmentRequirements, ScriptGrader, TextMessage
from tests.cpu.task_specs import text_task
from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_train.evidence import TrainingEvidence
from skyrl_train.telemetry import ProcessTelemetry, TelemetryConfig, TRAINER_ROLE
from skyrl_train.rollouts.task_projections import verification_result
from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results
from transformers import PreTrainedTokenizerFast


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


def test_canonical_rows_lower_at_load_and_admit_image_and_package_graders(tmp_path: Path):
    image = text_task("image-task").model_copy(
        update={
            "grader": ScriptGrader(
                argv=("sh", "/tests/grade.sh"),
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    docker_image="fixture@sha256:" + "0" * 64,
                ),
            )
        }
    )
    prefix = text_task("prefix-task", source="structure-closure").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="Complete"), TextMessage(role="assistant", content='{"a":['))
            ),
            "grader": ScriptGrader(
                argv=("python", "/tests/grade.py"),
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    packages_lock="s3://fixture/grader.lock",
                ),
            ),
        }
    )
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in (image, prefix)]), path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    config = get_default_config()
    dataset = SourceTaskDataset(
        [str(path)],
        tokenizer,
        128,
        num_workers=1,
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
    )
    loaded = [row["input"].spec for row in dataset.collate_fn(list(dataset))]
    assert [spec.task.id for spec in loaded] == ["image-task", "prefix-task"]
    assert is_shell_harness(loaded[0].harness)
    assert loaded[1].harness == ASSISTANT_PREFIX_HARNESS
    factories = {
        "daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created")),
        "local": LocalAdmissionFactory(),
    }
    for spec in loaded:
        validate_spec(spec, factories)


def test_filtered_task_identity_and_nullable_verdict_survive_persisted_evaluation(
    tmp_path: Path,
):
    tasks = [
        text_task("kept-first", prompt="short Ω"),
        text_task("filtered", prompt="long " * 200),
        text_task("kept-last", prompt="last"),
    ]
    path = tmp_path / "identity-input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]),
        path,
    )
    native = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native,
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    config = get_default_config()
    root = tmp_path / "evidence"
    with ProcessTelemetry(
        TelemetryConfig(),
        TRAINER_ROLE,
        TrainingEvidence(str(root), expected_steps=0, declarations={}),
    ):
        dataset = SourceTaskDataset(
            [str(path)],
            tokenizer,
            50,
            num_workers=1,
            environment_configs=OmegaConf.to_container(
                config.environment.task_sessions, resolve=True
            ),
        )
    rows = dataset.collate_fn(list(dataset))
    identities = [row["input"].metadata.task_identity for row in rows]
    assert [identity.task_id for identity in identities] == ["kept-first", "kept-last"]
    assert [identity.source_row_index for identity in identities] == [0, 2]
    first_hash = hashlib.sha256(
        json.dumps(
            json.loads(tasks[0].model_dump_json()),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert identities[0].task_spec_sha256 == first_hash
    manifest = json.loads((root / "manifest.json").read_bytes())
    retained = [
        json.loads(line)
        for chunk in manifest["chunks"]
        for line in (root / chunk["path"]).read_bytes().splitlines()
    ]
    observed = next(row["fields"] for row in retained if row["name"] == "input_dataset")
    assert (observed["before_rows"], observed["after_rows"]) == (3, 2)
    assert (
        observed["before_task_identity_sha256"]
        != observed["after_task_identity_sha256"]
    )
    output = tmp_path / "eval"
    output.mkdir()
    batch = {
        "prompt_token_ids": [[0], [0]],
        "response_ids": [[0], [0]],
        "rewards": [1.0, 1.0],
        "verification_results": [
            verification_result(GradeResult(Outcome.GRADED, 1.0, passed=True)),
            verification_result(GradeResult(Outcome.GRADED, 1.0)),
        ],
    }
    dump_per_dataset_eval_results(
        str(output),
        tokenizer,
        batch,
        ["fixture", "fixture"],
        ["taskcompendium"] * 2,
        {},
        [row["uid"] for row in rows],
        identities,
    )
    persisted = [
        json.loads(line) for line in (output / "fixture.jsonl").read_text().splitlines()
    ]
    assert [row["task_identity"]["source_row_index"] for row in persisted] == [0, 2]
    assert [row["verification"]["passed"] for row in persisted] == [True, None]
    assert (
        persisted[1]["verification"]["passed_unavailable_reason"]
        == "verifier_did_not_supply_boolean"
    )
    assert (output / "fixture.jsonl").stat().st_mode & 0o777 == 0o600
