# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequential rank-pinned pilot with pacing, repeat checks, and latency stops."""

import io
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from client import ScoringClient
from run import run_panel
from targets import Target, targets


@dataclass
class PacedClient(ScoringClient):
    sentinels: list[Target] = field(default_factory=list)
    guard_log: TextIO | None = None
    baseline: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    last_completion: float | None = None
    scored_targets: int = 0

    def request(self, route: str, payload: dict | None) -> dict:
        if route == "v1/completions" and self.last_completion is not None:
            time.sleep(max(0, 1 - (time.monotonic() - self.last_completion)))
        started = time.monotonic()
        result = super().request(route, payload)
        if route == "v1/completions":
            self.last_completion = time.monotonic()
            elapsed = self.last_completion - started
            self.latencies.append(elapsed)
            if elapsed > 5:
                raise RuntimeError("Completion latency exceeded five-second stop threshold")
            if len(self.latencies) >= 13:
                initial = statistics.median(self.latencies[:8])
                recent = statistics.median(self.latencies[-5:])
                if recent > max(1.5, 3 * initial):
                    raise RuntimeError("Sustained completion latency exceeded stop threshold")
        return result

    def check_sentinels(self) -> None:
        assert self.guard_log is not None
        for index, target in enumerate(self.sentinels):
            result = super().score(target.prefix, target.text, target.single_token)
            value = result["logprob"]
            if len(self.baseline) <= index:
                self.baseline.append(value)
            delta = abs(value - self.baseline[index])
            self.guard_log.write(
                json.dumps({"after_targets": self.scored_targets, "sentinel": index, "logprob": value, "delta": delta})
                + "\n"
            )
            self.guard_log.flush()
            if delta > 1e-6:
                raise RuntimeError(f"Sentinel {index} drifted by {delta}; stopping")

    def score(self, prefix: str, target: str, single_token: bool) -> dict:
        if self.scored_targets and self.scored_targets % 48 == 0:
            self.check_sentinels()
        result = super().score(prefix, target, single_token)
        self.scored_targets += 1
        return result


def run_paced_panel(client: PacedClient, inputs: Path, panel: str, variant_count: int, output: TextIO) -> None:
    """Run the selected panel; release completion only after final checks."""
    panels = json.loads((inputs / "panels.json").read_text())["panels"]
    client.session.headers["X-data-parallel-rank"] = "0"
    client.sentinels = [
        target
        for target in targets(panels["pilot"][:1], panels["demonstrations"], set())
        if target.strand == "+"
        and ((target.mapping == "none" and target.candidate == "reference") or target.mapping == "primary")
    ]
    assert len(client.sentinels) == 6
    buffer = io.StringIO()
    try:
        client.check_sentinels()
        run_panel(
            [client], inputs / "panels.json", inputs / "ensembl-reference.summary.json", panel, variant_count, buffer
        )
        client.check_sentinels()
    except BaseException:
        output.write(buffer.getvalue())
        output.write(json.dumps({"type": "invalid", "reason": "Paced trial did not pass all guards"}) + "\n")
        output.flush()
        raise
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    records[0].update({"data_parallel_rank": 0, "completion_gap_seconds": 1, "sentinel_tolerance": 1e-6})
    records[-1]["requests_sent"] = client.requests_sent
    for record in records:
        output.write(json.dumps(record) + "\n")
    output.flush()
