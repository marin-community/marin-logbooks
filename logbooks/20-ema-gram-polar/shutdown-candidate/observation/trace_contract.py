# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local trace instrumentation design; no runtime imports, jobs or numerical kernels.

Wrappers must be explicitly connected by a future authorized qualification
entrypoint. Auditing these traces alone never certifies runtime qualification.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Identity:
    campaign: str
    scenario: str
    candidate_commit: str
    execution_id: str
    attempt_uid: str
    task_uid: str
    worker_id: str
    rank: int
    ranks: tuple[int, ...]
    online_run_id: str


class Trace:
    def __init__(self, path: Path, identity: Identity):
        self.identity = identity
        self._stream = path.open("x")
        self._seq = 0
        self.emit("identity")

    def emit(self, event: str, **fields):
        row = {
            "identity": asdict(self.identity),
            "seq": self._seq,
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "fields": fields,
        }
        self._stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._stream.flush()
        self._seq += 1

    def observe(self, event: str, fields: dict):
        self.emit(event, **fields)

    def close(self):
        self._stream.close()


def traced_call(trace: Trace, event: str, call, *, entered=None, returned=None):
    """Trace callback failures without replacing the original error or exit code."""
    trace.emit(event + "_enter", **(entered or {}))
    try:
        result = call()
    except BaseException as original:
        try:
            trace.emit(event + "_error", error_type=type(original).__name__)
        except BaseException as trace_error:
            original.add_note(f"Error trace could not be persisted: {type(trace_error).__name__}")
        raise
    trace.emit(event + "_return", **(returned(result) if returned else {}))
    return result


class TracedCoordinator:
    """Delegate the existing protocol without extra service reads or barriers."""

    def __init__(self, coordinator, trace: Trace):
        self._coordinator = coordinator
        self._trace = trace

    def key_value_set(self, key, value, allow_overwrite=False):
        return traced_call(
            self._trace,
            "receipt_write",
            lambda: self._coordinator.key_value_set(key, value, allow_overwrite=allow_overwrite),
            entered={"key": key},
            returned=lambda _: {"key": key, "value_sha256": hashlib.sha256(value.encode()).hexdigest()},
        )

    def wait_at_barrier(self, key, timeout, process_ids):
        phase = key.rsplit("/", 1)[-1]
        return traced_call(
            self._trace,
            "barrier",
            lambda: self._coordinator.wait_at_barrier(key, timeout, process_ids),
            entered={"key": key, "phase": phase, "ranks": list(process_ids), "health_timeout_ms": timeout},
            returned=lambda _: {"key": key, "phase": phase},
        )

    def key_value_dir_get(self, prefix):
        return traced_call(
            self._trace,
            "receipt_read",
            lambda: self._coordinator.key_value_dir_get(prefix),
            entered={"prefix": prefix},
            returned=lambda pairs: {
                "prefix": prefix,
                "keys": [key for key, _ in pairs],
                "values_sha256": [hashlib.sha256(value.encode()).hexdigest() for _, value in pairs],
            },
        )


def traced_shutdown(shutdown, trace: Trace):
    def call():
        return traced_call(trace, "shutdown", shutdown)

    return call


def audit_clean_trace(paths: list[Path], expected: dict) -> dict:
    """Check local ordering and identity against a supplied immutable manifest.

    Monotonic timestamps are checked only within each process. Cross-host
    ordering follows the recorded two-barrier protocol, not clock comparison.
    Online run IDs are bindings, not independent proof of ONLINE state.
    """
    errors = []
    ranks_seen = []
    tasks_seen = []
    bindings = ("campaign", "scenario", "candidate_commit", "execution_id")
    trace_hashes = {}
    for path in paths:
        raw = path.read_bytes()
        trace_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        try:
            rows = [json.loads(line) for line in raw.splitlines()]
            if not rows:
                raise ValueError("empty trace")
            identity = rows[0]["identity"]
            rank = identity["rank"]
            ranks_seen.append(rank)
            tasks_seen.append(identity["task_uid"])
            for key in bindings:
                if identity[key] != expected[key]:
                    raise ValueError(f"{key} differs from manifest")
            if identity["ranks"] != expected["ranks"] or rank not in expected["ranks"]:
                raise ValueError("rank membership differs from manifest")
            bound = expected["rank_bindings"][str(rank)]
            for key in ("task_uid", "attempt_uid", "worker_id", "online_run_id"):
                if identity[key] != bound[key]:
                    raise ValueError(f"{key} differs from rank binding")
            if [row["seq"] for row in rows] != list(range(len(rows))):
                raise ValueError("sequence gap or duplication")
            if any(row["identity"] != identity for row in rows):
                raise ValueError("identity changed within trace")
            clocks = [row["monotonic_ns"] for row in rows]
            if clocks != sorted(clocks):
                raise ValueError("local monotonic clock went backward")
            required = [
                "receipt_write_return",
                "ready_return",
                "receipt_read_return",
                "read_return",
                "shutdown_enter",
                "shutdown_return",
            ]
            if expected.get("require_hook_lifecycle", False):
                required = ["hook_install_return", "hook_enter", *required, "hook_return"]
            observed = []
            for row in rows:
                event = row["event"]
                if event.endswith("_error"):
                    raise ValueError("operation failure recorded in clean trace")
                fields = row["fields"]
                if event == "barrier_return":
                    event = fields["phase"] + "_return"
                    if fields["key"] != expected["receipt_prefix"] + fields["phase"]:
                        raise ValueError("barrier identity differs from manifest")
                if event in ("ready_return", "read_return"):
                    if fields["key"] != expected["receipt_prefix"] + event.removesuffix("_return"):
                        raise ValueError("barrier identity differs from manifest")
                if event == "receipt_read_return":
                    expected_keys = {expected["receipt_prefix"] + str(r) for r in expected["ranks"]}
                    if len(fields["keys"]) != len(expected_keys) or set(fields["keys"]) != expected_keys:
                        raise ValueError("receipt read has missing, duplicate or foreign ranks")
                if event in required:
                    observed.append(event)
            if observed != required:
                raise ValueError(f"clean finalization order differs: {observed}")
        except (KeyError, ValueError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if sorted(ranks_seen) != sorted(expected["ranks"]):
        errors.append({"error": "trace rank coverage is missing or duplicated"})
    if len(tasks_seen) != len(set(tasks_seen)):
        errors.append({"error": "task UID reused across ranks"})
    return {
        "local_clean_ordering_pass": not errors,
        "errors": errors,
        "trace_sha256": trace_hashes,
        "runtime_qualified": False,
        "unverified": [
            "native ONLINE state and full history",
            "controller and natural task outcomes",
            "actual coordinator protocol semantics",
            "source/runtime attestation",
            "resource retention",
        ],
    }
