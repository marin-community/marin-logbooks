# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay two SWE-bench prefixes and compare equal-budget continuation choices."""

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import torch
from datasets import load_dataset
from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, DaytonaConflictError, DaytonaError, Image
from huggingface_hub import snapshot_download
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.post_training.goal_value.data import Episode, stable_hash
from experiments.post_training.goal_value.model import MODEL_ID, MODEL_REVISION, TurnEncoder, ValueHead, encode

DATASET = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
ACTOR = "Qwen/Qwen3-14B"
ACTOR_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
WORKDIR = "/testbed"
CUT = 4
OUTPUT_CHARS = 6000
MAX_INPUT_TOKENS = 32000
MAX_ACTION_TOKENS = 1024
MAX_PATCH_BYTES = 200000
GATEWAY_ERRORS = {502, 503, 504}
SANDBOX_CREATION_DELAYS = (4, 8, 16)
ACTIVATE_TESTBED = "source /opt/miniconda3/bin/activate testbed && "
ACTOR_INSTRUCTIONS = (
    "You are taking over a software-maintenance task. Inspect the current code or diff and make a focused fix. "
    "Use the available actions to edit the repository and check your fix. Use a short private analysis for each step. "
    'After that, return {"command":"shell command"} to act or {"submit":true} after confirming the fix. '
    "Commands run in a fresh shell in /testbed with the testbed Python environment active. "
    "If a command fails, read the output and change your approach."
)


class ReplayMismatch(RuntimeError):
    """A prefix could not be restored to the recorded action boundary."""


@dataclass(frozen=True)
class Prefix:
    episode: Episode
    commands: tuple[str, ...]
    archived_codes: tuple[int, ...]


@dataclass(frozen=True)
class Replayed:
    messages: list[dict]
    exit_codes: tuple[int, ...]
    diff_sha256: str


def create_workspace(client: Daytona, params: CreateSandboxFromImageParams):
    """Use the same workspace name on gateway retries to avoid duplicate sandboxes."""
    name = params.name
    if name is None:
        raise ValueError("workspace creation retries require a name")
    delays = iter(SANDBOX_CREATION_DELAYS)
    while True:
        try:
            try:
                return client.create(params, timeout=240)
            except DaytonaConflictError:
                workspace = client.get(name, request_timeout=30)
                workspace.wait_for_sandbox_start(timeout=240)
                return workspace
        except DaytonaError as error:
            delay = next(delays, None)
            if error.status_code not in GATEWAY_ERRORS or delay is None:
                raise
            print(f"Workspace gateway returned HTTP {error.status_code}. Retrying in {delay}s.", flush=True)
            time.sleep(delay)


def scenario_prefix(record: dict) -> Prefix:
    """Load a private prefix manifest whose placeholder outcome carries no grader data."""
    episode = Episode(**record["episode"])
    if episode.split != "test" or len(episode.predict_before) != CUT or episode.outcome != 0:
        raise ValueError("expected four actions on an issue in the test split")
    commands = []
    for point in episode.predict_before:
        step = episode.messages[point]
        calls = step.get("tool_calls", ())
        if step["role"] != "assistant" or len(calls) != 1 or calls[0]["function_name"] != "bash":
            raise ValueError("expected exactly one bash command per archived action")
        commands.append(calls[0]["arguments"]["command"])
    archived_codes = tuple(record["exit_codes"])
    if len(archived_codes) != CUT or not all(isinstance(code, int) for code in archived_codes):
        raise ValueError("expected a recorded exit code for each replayed action")
    return Prefix(episode, tuple(commands), archived_codes)


@contextmanager
def sandbox(client: Daytona, image: str, base_commit: str):
    """Allocate a private, network-blocked workspace at the benchmark base commit."""
    params = CreateSandboxFromImageParams(
        image=Image.base(image),
        name=f"value-resampling-{uuid4().hex}",
        public=False,
        ephemeral=True,
        network_block_all=True,
        auto_stop_interval=10,
        ttl_minutes=30,
        labels={"purpose": "swebench-prefix-resampling"},
    )
    workspace = create_workspace(client, params)
    try:
        base = shlex.quote(base_commit)
        checkout = workspace.process.exec(
            f"git checkout --quiet --detach {base} && git diff --quiet HEAD -- . && git rev-parse HEAD",
            cwd=WORKDIR,
            timeout=30,
        )
        if checkout.exit_code or checkout.result.strip() != base_commit:
            raise ReplayMismatch("sandbox source tree differs from the benchmark base or checkout failed")
        yield workspace
    finally:
        client.delete(workspace, timeout=60)


def tool_action(workspace, command: str):
    result = workspace.process.exec("bash -c " + shlex.quote(command), cwd=WORKDIR, timeout=45)
    output = result.result
    if len(output) > OUTPUT_CHARS:
        half = OUTPUT_CHARS // 2
        output = output[:half] + "\n[Output omitted]\n" + output[-half:]
    return result.exit_code, output


def patch(workspace) -> str:
    result = workspace.process.exec("git add -N -A -- . && git diff --binary HEAD -- .", cwd=WORKDIR, timeout=30)
    if result.exit_code:
        raise RuntimeError("could not extract a patch")
    return result.result


def replay(workspace, prefix: Prefix) -> Replayed:
    episode = prefix.episode
    messages = list(episode.messages[: episode.predict_before[0]])
    exit_codes = []
    for command, point in zip(prefix.commands, episode.predict_before, strict=True):
        code, output = tool_action(workspace, command)
        exit_codes.append(code)
        messages.append(episode.messages[point])
        messages.append({"role": "tool", "content": json.dumps({"returncode": code, "output": output})})
    if tuple(exit_codes) != prefix.archived_codes:
        raise ReplayMismatch(f"archive and replay exit codes differ: {prefix.archived_codes} vs {tuple(exit_codes)}")
    digest = hashlib.sha256(patch(workspace).encode()).hexdigest()
    return Replayed(messages, tuple(exit_codes), digest)


class Value:
    """Score a replayed cutoff with the trained, frozen pre-action readout."""

    def __init__(self, head_path: Path, model_dir: str):
        self.encoder = TurnEncoder(model_dir, torch.device("cuda"))
        state = torch.load(head_path, weights_only=True)
        if (state["model"], state["revision"], state["max_tokens"]) != (MODEL_ID, MODEL_REVISION, 65536):
            raise ValueError("the value checkpoint must be the 65k Qwen3-4B head")
        self.head = ValueHead(state["state_dict"]["weight"].shape[-1]).eval()
        self.head.load_state_dict(state["state_dict"])

    @torch.inference_mode()
    def score(self, prefix: Prefix, state: Replayed) -> float:
        points = [index for index, row in enumerate(state.messages) if row["role"] == "assistant"]
        episode = replace(
            prefix.episode,
            messages=[*state.messages, {"role": "assistant", "content": ""}],
            predict_before=[*points, len(state.messages)],
        )
        tokens = encode(episode, self.encoder.tokenizer, 65536)
        if len(tokens.positions) != CUT + 1:
            raise ValueError("replayed state exceeds context cap")
        _, hidden = self.encoder.hidden([tokens])[0]
        return float(self.head(hidden[-1].float()).sigmoid())


class Actor:
    """Continue a restored workspace using the same frozen actor and budgets in both arms."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(ACTOR, revision=ACTOR_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            ACTOR,
            revision=ACTOR_REVISION,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def next_action(self, goal: str, history: list[dict], seed: int, tokens: int) -> tuple[str, int, int]:
        for start in range(len(history) + 1):
            observed = "\n".join(json.dumps(row, ensure_ascii=False) for row in history[start:])
            messages = [
                {"role": "system", "content": ACTOR_INSTRUCTIONS},
                {"role": "user", "content": goal + "\n\nObserved history:\n" + observed.replace("<|", "<\\u007c")},
            ]
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(ids) <= MAX_INPUT_TOKENS:
                break
        else:
            raise ValueError("task exceeds actor context")
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
        torch.manual_seed(seed)
        generated = self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0, len(ids) :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        _, separator, response = raw.partition("</think>")
        return (response if separator else ""), len(generated), len(ids)


def parse_action(text: str) -> dict:
    """Extract one action object from the actor response."""
    try:
        action, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        if not isinstance(action, dict):
            return {}
        if action == {"submit": True} or (action.keys() == {"command"} and isinstance(action["command"], str)):
            return action
    except (ValueError, TypeError):
        pass
    return {}


def continue_branch(workspace, actor, prefix, state, seed, actions, token_budget):
    history = state.messages[prefix.episode.predict_before[0] :]
    tokens = 0
    commands = 0
    prompt_tokens = 0
    attempted_actions = 0
    malformed_actions = 0
    failed_commands = 0
    submitted = False
    for action_index in range(actions):
        if tokens >= token_budget:
            break
        text, produced, consumed = actor.next_action(
            prefix.episode.goal, history, seed + action_index, min(MAX_ACTION_TOKENS, token_budget - tokens)
        )
        tokens += produced
        prompt_tokens += consumed
        attempted_actions += 1
        action = parse_action(text)
        if action == {"submit": True}:
            submitted = True
            break
        history.append({"role": "assistant", "content": json.dumps(action) if action else text[-500:]})
        if "command" in action:
            commands += 1
            code, output = tool_action(workspace, ACTIVATE_TESTBED + action["command"])
            failed_commands += int(code != 0)
        else:
            malformed_actions += 1
            code, output = 2, "Expected a command or submit JSON object"
        history.append({"role": "tool", "returncode": code, "output": output})
    return patch(workspace), {
        "generated_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "commands": commands,
        "actor_calls": attempted_actions,
        "malformed_actions": malformed_actions,
        "failed_commands": failed_commands,
        "explicit_submit": submitted,
    }


def verify(workspace, spec, proposed_patch: str) -> int:
    """Grade the proposed patch in a fresh workspace after the actor has stopped."""
    if len(proposed_patch.encode()) > MAX_PATCH_BYTES:
        return 0
    if proposed_patch.strip():
        workspace.fs.upload_file(proposed_patch.encode(), "/tmp/proposed.patch")
        applied = workspace.process.exec("git apply --whitespace=nowarn /tmp/proposed.patch", cwd=WORKDIR, timeout=30)
        if applied.exit_code:
            return 0
    workspace.fs.upload_file(spec.eval_script.encode(), "/tmp/eval.sh")
    evaluated = workspace.process.exec("bash /tmp/eval.sh", cwd=WORKDIR, timeout=480)
    if ">>>>> Start Test Output" not in evaluated.result or ">>>>> End Test Output" not in evaluated.result:
        raise RuntimeError("verifier did not complete a test run")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "test.txt"
        output.write_text(evaluated.result)
        report = get_eval_report(
            spec,
            {"instance_id": spec.instance_id, "model_patch": proposed_patch},
            str(output),
            False,
        )
    return int(report[spec.instance_id]["resolved"])


def run_case(client, instance, prefixes, scorer, args, actor):
    spec = make_test_spec(instance, namespace="swebench")
    image = spec.instance_image_key
    outputs, values = [], []
    scoring_seconds = 0.0
    for prefix in prefixes:
        with sandbox(client, image, instance["base_commit"]) as workspace:
            observed = replay(workspace, prefix)
        outputs.append(observed)
        start = time.monotonic()
        values.append(scorer.score(prefix, observed))
        scoring_seconds += time.monotonic() - start

    issue_seed = stable_hash("resampling-seed", spec.instance_id) % (2**31)
    selected = int(values[1] > values[0]) if abs(values[0] - values[1]) > 1e-6 else None
    print(f"Case replay complete with two live scores and {args.replicates} paired seeds", flush=True)
    results = {}
    for offset in range(args.replicates):
        seed = issue_seed + offset * (args.actions + 1)
        by_branch = []
        for branch, prefix in enumerate(prefixes):
            with sandbox(client, image, instance["base_commit"]) as workspace:
                restored = replay(workspace, prefix)
                reference = outputs[branch]
                if (restored.exit_codes, restored.diff_sha256) != (reference.exit_codes, reference.diff_sha256):
                    raise ReplayMismatch("replaying a branch changed its workspace patch or tool exit codes")
                proposed, usage = continue_branch(
                    workspace,
                    actor,
                    prefix,
                    reference,
                    seed,
                    args.actions,
                    args.tokens,
                )
            with sandbox(client, image, instance["base_commit"]) as workspace:
                outcome = verify(workspace, spec, proposed)
            by_branch.append({"pass": outcome, "patch_bytes": len(proposed.encode()), **usage})
        results[str(seed)] = by_branch
        print(f"Both branches completed paired continuations for seed {offset + 1}", flush=True)
    return {
        "issue": spec.instance_id,
        "values": values,
        "scoring_seconds": scoring_seconds,
        "guided_branch": selected,
        "continuations": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=6144)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    if min(args.actions, args.tokens, args.replicates) < 1:
        parser.error("continuation limits must be positive")
    cases = json.loads(args.manifest.read_text())
    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split="test")
    instances = {row["instance_id"]: row for row in dataset}
    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    actor = Actor()
    scorer = Value(args.head, model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        for case in cases:
            instance = instances[case["instance_id"]]
            prefixes = tuple(map(scenario_prefix, case["prefixes"]))
            if (
                len(prefixes) != 2
                or any(prefix.episode.task_id != f"swebench-verified:{instance['instance_id']}" for prefix in prefixes)
                or prefixes[0].episode.goal != prefixes[1].episode.goal
            ):
                raise ValueError("expected two equivalent goals for the selected instance")
            try:
                result = run_case(client, instance, prefixes, scorer, args, actor)
            except ReplayMismatch as error:
                result = {"issue": instance["instance_id"], "replay_error": str(error)}
            except DaytonaError as error:
                if error.status_code not in GATEWAY_ERRORS:
                    raise
                result = {"issue": instance["instance_id"], "environment_error": f"Daytona HTTP {error.status_code}"}
            payload = json.dumps(result)
            handle.write(payload + "\n")
            handle.flush()
            print("RESAMPLING_RESULT " + payload, flush=True)


if __name__ == "__main__":
    main()
