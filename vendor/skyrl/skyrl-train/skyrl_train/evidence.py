"""Bounded, acknowledged training evidence beside the durable run artifacts."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path

from marinskyrl.resource_locator import join_resource_path
from marinskyrl.environment_contract import TRAINING_TYPE_ENV
from skyrl_train.io import io

MAX_RECORD_BYTES = 64 * 1024
CHUNK_BYTES = 256 * 1024
MAX_RUN_BYTES = 16 * 1024 * 1024
MAX_QUEUED_BYTES = 1024 * 1024
MAX_QUEUED_CHUNKS = 64
CLOSE_TIMEOUT = 30.0
EVENTS = frozenset(
    {
        "training_metrics",
        "selected_token_staleness",
        "optimizer_window",
        "checkpoint_published",
        "training_schedule",
        "policy_update",
        "consumed_staleness",
        "async_phase_window",
        "rollout_call",
        "rollout_group_disposition",
        "generated_work",
        "consumed_work",
        "rollout_closed",
        "shutdown_failure",
        "input_dataset",
    }
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("evidence count is not an integer")
    return value


def runtime_identity() -> dict:
    """Observe this interpreter and small module metadata, without reading weights or secrets."""
    packages: dict[str, str | None] = {}
    for name in ("torch", "ray", "transformers", "vllm", "megatron-core", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    raw = os.environ.get("MARIN_PROVENANCE")
    declared_source = json.loads(raw) if raw else {}
    loaded: dict[str, dict[str, str] | None] = {}
    for name in (
        "skyrl_train.trainer",
        "skyrl_train.telemetry",
        "skyrl_train.dataset.tasks",
    ):
        module = sys.modules.get(name)
        if module is not None and module.__file__ is not None:
            path = Path(module.__file__)
            loaded[name] = {
                "path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        else:
            loaded[name] = None
    return {
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "packages": packages,
        "evidence_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inherited_source_declaration": {
            key: declared_source.get(key) for key in ("base_commit", "tree_hash", "dirty")
        },
        "run_id": os.environ.get("SKYRL_RUN_ID"),
        "execution_uid": os.environ.get("SKYRL_EXECUTION_UID"),
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "training_type": os.environ.get(TRAINING_TYPE_ENV),
        "observed_driver_module_files": loaded,
        "limits": [
            "package metadata is not loaded GPU module/weight attestation",
            "inherited source is a declaration",
            "served judge revision, external assets and optimizer resume are not observed here",
        ],
    }


class TrainingEvidence:
    """Persist ordered bounded chunks, then publish completeness only after their writes return."""

    def __init__(
        self,
        output_path: str,
        *,
        expected_steps: int | None,
        declarations: Mapping[str, object],
        close_timeout: float = CLOSE_TIMEOUT,
    ):
        self.output_path = output_path
        self.expected_steps = expected_steps
        self.starting_step = 0
        self.declarations = dict(declarations)
        self.chunks: list[dict] = []
        self.pending = bytearray()
        self.record_count = 0
        self.total_bytes = 0
        self.failure: str | None = None
        self.updates: dict[int, dict] = {}
        self.windows: dict[int, list[dict]] = {}
        self.consumed: Counter[int] = Counter()
        self.ages: dict[int, Counter[int]] = {}
        self.token_summaries: dict[int, dict] = {}
        self.intervals: Counter[str] = Counter()
        self.dispositions: Counter[str] = Counter()
        self.disposition_tokens: Counter[str] = Counter()
        self.generated_tokens = 0
        self.closed_boundary: dict | None = None
        self.started_unix_ns = time.time_ns()
        self._condition = threading.Condition()
        self._queue: deque[tuple[str, bytes]] = deque()
        self._queued_bytes = 0
        self._next_chunk = 0
        self._closing = False
        self._finished = False
        self._training_failed = False
        self._error: Exception | None = None
        self._close_timeout = close_timeout
        self._rejected_records = 0
        # A stuck storage call cannot keep the driver process alive after the close deadline.
        self._publisher = threading.Thread(target=self._run_publisher, name="training-evidence", daemon=True)

    def start(self) -> None:
        self.identity = runtime_identity()
        self._publisher.start()

    def record(self, name: str, fields: Mapping[str, object], attributes: Mapping[str, str]) -> None:
        with self._condition:
            if name not in EVENTS:
                return
            if self.failure is not None or self._closing:
                self._rejected_records += 1
                return
            recorded = False
            try:
                nonfinite = [
                    key for key, value in fields.items() if isinstance(value, float) and not math.isfinite(value)
                ]
                values = {key: None if key in nonfinite else value for key, value in fields.items()}
                row = {
                    "sequence": self.record_count,
                    "unix_ns": time.time_ns(),
                    "name": name,
                    "fields": values,
                    "attributes": dict(attributes),
                    "nonfinite_fields": nonfinite,
                }
                payload = _json(row) + b"\n"
                if len(payload) > MAX_RECORD_BYTES or self.total_bytes + len(payload) > MAX_RUN_BYTES:
                    raise ValueError("training evidence byte cap exceeded")
                self.pending.extend(payload)
                self.total_bytes += len(payload)
                self.record_count += 1
                recorded = True
                if name == "training_schedule":
                    self.expected_steps = _integer(fields["expected_steps"])
                    self.starting_step = _integer(fields["starting_step"])
                elif name == "selected_token_staleness":
                    self.token_summaries[int(attributes["step"])] = dict(values)
                elif name == "optimizer_window":
                    self.windows.setdefault(int(attributes["step"]), []).append(
                        {"metrics": values, "nonfinite_fields": nonfinite}
                    )
                elif name == "policy_update":
                    step = int(attributes["step"])
                    if step in self.updates:
                        raise ValueError("duplicate policy update evidence")
                    self.updates[step] = {"metrics": values, "nonfinite_fields": nonfinite}
                elif name == "consumed_staleness":
                    step = int(attributes["step"])
                    self.consumed[step] += _integer(fields["sequences"])
                    self.ages.setdefault(step, Counter())[_integer(fields["staleness"])] += _integer(
                        fields["response_tokens"]
                    )
                elif name in {"async_phase_window", "rollout_call"}:
                    self.intervals[name] += 1
                elif name == "rollout_group_disposition":
                    disposition = attributes["disposition"]
                    self.dispositions[disposition] += 1
                    self.disposition_tokens[disposition] += _integer(fields["tokens"])
                elif name == "generated_work":
                    self.generated_tokens += _integer(fields["generated_tokens"])
                elif name == "rollout_closed":
                    self.closed_boundary = dict(values)
                if name == "policy_update" or len(self.pending) >= CHUNK_BYTES:
                    self._enqueue()
            except Exception as error:
                # Retention failure must not replace a training exception. The terminal manifest
                # stays ineligible, even if a later final write succeeds.
                if not recorded:
                    self._rejected_records += 1
                self._fail(error)

    def _fail(self, error: Exception) -> None:
        if self._error is None:
            self._error = error
            self.failure = type(error).__name__

    def _enqueue(self) -> None:
        # Called under the condition; immutable payloads leave the producer before any I/O.
        if not self.pending:
            return
        if self._queued_bytes + len(self.pending) > MAX_QUEUED_BYTES or len(self._queue) >= MAX_QUEUED_CHUNKS:
            raise BufferError("training evidence publisher queue exhausted")
        relative = f"chunks/{self._next_chunk:06d}.jsonl"
        payload = bytes(self.pending)
        self.pending.clear()
        self._next_chunk += 1
        self._queued_bytes += len(payload)
        self._queue.append((relative, payload))
        self._condition.notify()

    def flush(self) -> None:
        """Queue pending records without waiting for storage on the producer thread."""
        with self._condition:
            try:
                self._enqueue()
            except Exception as error:
                self._fail(error)

    def _run_publisher(self) -> None:
        try:
            self._publish("recording")
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._queue or self._closing)
                    if not self._queue and self.pending:
                        # Only close reaches this branch, after all bounded queued writes ACK.
                        self._enqueue()
                    if not self._queue:
                        break
                    relative, payload = self._queue.popleft()
                io.write_bytes_atomic(join_resource_path(self.output_path, relative), payload)
                with self._condition:
                    self._queued_bytes -= len(payload)
                    self.chunks.append(
                        {
                            "path": relative,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                self._publish("recording")
            status = self._publish("failed" if self._training_failed or self.failure is not None else "completed")
            with self._condition:
                if status == "failed" or self.failure is None:
                    self._finished = True
                    return
            # A close deadline may expire while the final atomic write is awaiting ACK.
            # Serialize the failure projection after that write; artifact success remains
            # a separate eligibility requirement even if storage never returns its ACK.
            self._publish("failed")
        except Exception as error:
            with self._condition:
                self._fail(error)
            # No retry of a failed object write. The last ACKed manifest remains ineligible.
        finally:
            with self._condition:
                self._finished = True

    def _publish(self, status: str) -> str:
        with self._condition:
            if self.failure is not None:
                status = "failed"
            payload = self._manifest(status)
        io.write_bytes_atomic(join_resource_path(self.output_path, "manifest.json"), payload)
        return status

    def _manifest(self, status: str) -> bytes:
        observed = sorted(step for step in self.updates if self.starting_step < step <= (self.expected_steps or 0))
        missing_count = (
            None if self.expected_steps is None else max(0, self.expected_steps - self.starting_step - len(observed))
        )
        joined = all(self.consumed[step] > 0 for step in observed)
        update_fields_present = all(
            {"policy_update_steps", "skipped_steps", "raw_grad_norm", "policy_loss"} <= row["metrics"].keys()
            for row in self.updates.values()
        )
        window_counts_match = bool(observed) and all(
            len(self.windows.get(step, [])) > 0
            and sum(row["metrics"]["applied"] for row in self.windows[step])
            == self.updates[step]["metrics"].get("policy_update_steps")
            and sum(not row["metrics"]["applied"] for row in self.windows[step])
            == self.updates[step]["metrics"].get("skipped_steps")
            for step in observed
        )
        windows_finite_and_applied = window_counts_match and all(
            not row["nonfinite_fields"]
            and row["metrics"]["applied"] is True
            and row["metrics"].get("raw_grad_norm") is not None
            and row["metrics"].get("loss_finite") is True
            for windows in self.windows.values()
            for row in windows
        )
        complete = (
            status == "completed"
            and self.failure is None
            and not self.pending
            and not self._queued_bytes
            and missing_count == 0
        )
        complete = (
            complete
            and self.closed_boundary is not None
            and self.closed_boundary.get("driver_dispatch_stopped") is True
            and self.closed_boundary.get("accounting_complete") is True
            and joined
            and update_fields_present
            and window_counts_match
        )
        manifest = {
            "status": status,
            "complete": complete,
            "failure_class": self.failure,
            "started_unix_ns": self.started_unix_ns,
            "published_unix_ns": time.time_ns(),
            "identity": self.identity,
            "declarations": self.declarations,
            "expected_outer_steps": self.expected_steps,
            "starting_step": self.starting_step,
            "observed_outer_steps": observed,
            "missing_outer_step_count": missing_count,
            "records": self.record_count,
            "bytes": self.total_bytes,
            "chunks": self.chunks,
            "pending_bytes": len(self.pending),
            "unacknowledged_bytes": self._queued_bytes + len(self.pending),
            "rejected_records": self._rejected_records,
            "close_ack_limit": (
                "Completeness describes retained producer records; successful caller close and artifact completion are separate required lifecycle ACKs."
            ),
            "policy_updates": self.updates,
            "optimizer_windows": self.windows,
            "optimizer_window_counts_match": window_counts_match,
            "selected_sequences_by_outer_step": dict(self.consumed),
            "selected_age_to_update_join_complete": joined,
            "selected_group_response_tokens_by_step_and_group_age": {
                step: dict(ages) for step, ages in self.ages.items()
            },
            "selected_loss_token_age_metrics_by_outer_step": self.token_summaries,
            "selected_token_age_limit": (
                "Pre-objective loss-mask metrics, using engine stamps where present and lease fallback otherwise; no per-window token membership attestation."
            ),
            "operational_interval_counts": dict(self.intervals),
            "buffer_consumed_limit": (
                "Buffer consumed means batch selected; actual applied optimizer windows are joined by outer step separately, not exact per-window token membership."
            ),
            "all_updates_finite_and_applied": (
                bool(observed)
                and missing_count == 0
                and update_fields_present
                and windows_finite_and_applied
                and all(
                    not row["nonfinite_fields"]
                    and row["metrics"]["policy_update_steps"] > 0
                    and row["metrics"]["skipped_steps"] == 0
                    for row in self.updates.values()
                )
            ),
            "generated_tokens": self.generated_tokens,
            "disposition_groups": dict(self.dispositions),
            "disposition_tokens": dict(self.disposition_tokens),
            "committed_token_accounting_balanced": (
                self.closed_boundary is not None and self.generated_tokens == sum(self.disposition_tokens.values())
            ),
            "closed_boundary": self.closed_boundary,
            "uncommitted_generation_tokens": None,
            "accounting_limit": (
                "Cancelled/uncommitted remote generation may have produced unknown tokens; never zero by absence."
            ),
            "clock_limit": "Driver wall/monotonic operational intervals, not GPU kernel overlap or utilization.",
            "upload_ack": (
                "All listed chunk writes returned before this manifest write; no content readback/resume attestation."
            ),
        }
        return _json(manifest)

    def close(self, *, failed: bool) -> None:
        """Wait a finite time for ordered publication, preserving a primary training failure."""
        with self._condition:
            self._training_failed = self._training_failed or failed
            self._closing = True
            self._condition.notify()
        self._publisher.join(self._close_timeout)
        with self._condition:
            if not self._finished:
                self._fail(TimeoutError("training evidence publication did not ACK before close deadline"))
            error = self._error
        if error is not None and not failed:
            raise error
