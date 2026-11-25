"""Run the task 013 Claude Code pilot inside an Iris job."""

import argparse
import asyncio
import json
import logging
import signal
import time
import uuid
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from evalrecords.records import CuratedSettings, TaskEvidence, TaskImageBuild, TaskMachineSettings
from evalrecords.rollout_results import logical_samples, terminal_attempts
from finestore.reader import ReadView
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from huggingface_hub import snapshot_download
from levanter.compat.hf_checkpoints import load_tokenizer
from marin.evaluation.dataset_evaluator import runtime_config
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.evaluation.task_worker import daytona_connection, execution_config, machine_factories
from marin.inference.iris import remote_inference
from marin.rollouts.evaluation import run_evaluation
from openai import AsyncOpenAI
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from rolloutengine.agents.claude_code import CONTEXT, VERSION, claude_harness, message_digest
from rolloutengine.agents.model_relay import VllmTokenizer
from rolloutengine.chat import ChatModel
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.evaluation.spec import ClaudeCodeAgent
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks import swebench_science as benchmark
from experiments.evaluation.benchmarks.harbor import harbor_item
from experiments.evaluation.cli import resolve_model_config
from experiments.post_training.task_curation.harbor_packages import write_package_archives
from experiments.post_training.task_curation.pipeline import source_recipe

PLATFORM_SHA512 = "f469fe57b0ed27c3359f3b4ff42783565f60dd1b6b7a923bace153c8f4519d97bc0facde6c81ca982ecebc16aeb9cb803f45f493ce520cdfcb3d56728ec19ecd"
INSTALL = f"""RUN set -eu; \\
    if ! command -v curl >/dev/null; then apt-get update && apt-get install -y --no-install-recommends curl ca-certificates; fi; \\
    curl -fsSL https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-{VERSION}.tgz -o /tmp/claude.tgz; \\
    echo '{PLATFORM_SHA512}  /tmp/claude.tgz' | sha512sum -c -; \\
    tar -xzOf /tmp/claude.tgz package/claude > /usr/local/bin/claude; chmod 755 /usr/local/bin/claude; \\
    rm /tmp/claude.tgz; claude --version | grep -F '{VERSION} '
"""


def corpus(root: StoragePath, ids: tuple[str, ...]) -> str:
    [source] = benchmark.benchmarks()[0].resolve("pi-xhigh")
    output = root / "corpus"
    if not (output / "evaluation_manifest.json").exists():
        package_names = [f"task_{i}" for i in ids]
        with TemporaryDirectory() as directory:
            raw = Path(
                snapshot_download(
                    benchmark.DATASET,
                    repo_type="dataset",
                    revision=benchmark.REVISION,
                    local_dir=directory,
                    allow_patterns=[f"tasks/{name}/*" for name in package_names],
                )
            )
            write_package_archives([raw / "tasks" / name for name in package_names], root / "input")
        recipe = replace(
            source.config.recipe,
            name="swebench-science-claude-code-" + "-".join(ids),
            convert=partial(benchmark.convert, install=INSTALL),
        )
        protocol = source.config.protocol.model_copy(
            update={
                "expected_population": len(ids),
                "sampling": source.config.protocol.sampling.model_copy(update={"max_tokens": None}),
            }
        )
        agent = source.config.agent.model_copy(update={"harness": ClaudeCodeAgent(), "max_input_tokens": CONTEXT})
        client = LocalClient()
        try:
            with (
                TemporaryDirectory() as tmp,
                set_current_client(client),
                ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=tmp) as z,
            ):
                publish_evaluation_corpus(
                    source_recipe(recipe, {}, None),
                    protocol,
                    partial(harbor_item, agent=agent),
                    z,
                    str(root / "input"),
                    str(output),
                    protocol_name="claude-code-xhigh",
                )
        finally:
            client.shutdown()
    actual = {c.task.source.row.split("task_")[-1][:3] for c in evaluation_cases(str(output))}
    assert actual == set(ids), actual
    return str(output)


async def evaluate(session, model, dataset, root, ids, tokenizer=None):
    settings = CuratedSettings(
        concurrency=len(ids),
        max_tokens=CONTEXT - 1,
        task_evidence=TaskEvidence.REFERENCE,
        task_machine=TaskMachineSettings(image_build=TaskImageBuild.DECLARATIVE),
    )
    config = execution_config(runtime_config(model, settings, session))
    assert config.model.context_window == CONTEXT
    endpoint = session.model.endpoint

    async def check_wire(request):
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            assert body["reasoning_effort"] == "xhigh" and body["chat_template_kwargs"]["enable_thinking"]
            cap = body["max_tokens"]
            assert 0 < cap < CONTEXT
            (root / "wire" / f"{uuid.uuid4().hex}.json").write_text(
                json.dumps(
                    {
                        "model": body["model"],
                        "max_tokens": cap,
                        "reasoning_effort": body["reasoning_effort"],
                        "enable_thinking": True,
                        "messages_sha256": message_digest(body["messages"]),
                    }
                ),
                auto_mkdir=True,
            )

    def record(run_id, call, data):
        (root / "requests" / run_id / f"{call:05d}.json").write_text(json.dumps(data), auto_mkdir=True)

    output = str(root / "results")
    async with httpx.AsyncClient(event_hooks={"request": [check_wire]}, timeout=1800) as http:
        async with AsyncOpenAI(
            base_url=endpoint.base_url, api_key=endpoint.api_key or "local", http_client=http, timeout=1800
        ) as client:
            await run_evaluation(
                (dataset,),
                output,
                config,
                ChatModel(client, config.model.name, max_tokens=None),
                transaction_bytes=1024 * 1024 * 1024,
                machine_factories=machine_factories(config, daytona_connection(require_secret("DAYTONA_API_KEY"))),
                harnesses={
                    "claude-code": lambda _agent, _budget: claude_harness(
                        VllmTokenizer(client, config.model.name), record
                    )
                },
                tokenizer=tokenizer if tokenizer is not None else load_tokenizer(
                    config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False
                ),
            )
    reader = ReadView(output)
    attempts = terminal_attempts(reader)
    assert len(attempts) == len(ids) and all(a["outcome"] == "graded" for a in attempts.values())
    grades = {s["doc_id"]: dict(s["metrics"])["reward"] for s in logical_samples(reader)}
    assert len(grades) == len(ids)
    (root / "graded.json").write_text(json.dumps(grades), auto_mkdir=True)
    if ids == ("013",):
        records = [json.loads(p.read_text()) for p in (root / "requests" / "*" / "*.json").glob()]
        assert any("Edit" in r.get("tool_calls", []) or "Bash" in r.get("tool_calls", []) for r in records)
        assert any(r.get("calls", 0) > 0 for r in records)
    print("SWE-BENCH SCIENCE NATIVE GRADES", list(ids), list(grades.values()), flush=True)


def launch_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", required=True, type=int)
    parser.add_argument("--cluster", required=True, choices=("cw-us-east-02a", "cw-rno2a"))
    return parser


def configure_deadline(args):
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()

    def expired(*_):
        raise TimeoutError("GPU authorization expired")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, args.deadline - int(time.time()) - 100))


def inference_service(model, cluster):
    hardware = AcceleratorChoice(Platform.GPU, gpu_type="H100", gpu_count=4, target_cluster=cluster)
    service = inference_config_for_model(model, hardware, env_vars={}, priority=0, instances=1)
    return replace(
        service,
        broker=None,
        iris=replace(service.iris, cache_ttl_days=0, max_retries_failure=0, max_retries_preemption=1),
    )


def launch(args, ids: tuple[str, ...], phase: str):
    configure_deadline(args)
    model = resolve_model_config(None, Path(__file__).with_name("model.yaml"))
    root = StoragePath(args.output) / phase
    dataset = corpus(root, ids)
    with remote_inference(inference_service(model, args.cluster)) as session:
        asyncio.run(evaluate(session, model, dataset, root, ids))


if __name__ == "__main__":
    launch(launch_parser().parse_args(), ("013",), "pilot")
