# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fault: first execution duplicates G/3; second execution omits G/3."""

import argparse
import json
from pathlib import Path

import callback_compatibility as worker
import capture
import jax
import numpy as np
from jax.sharding import AxisType

COLLECTORS = []


class MissingSecondExecutionRecord(capture.Collector):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.execution = len(COLLECTORS)
        COLLECTORS.append(self)

    def receive(self, label, partitioned, value, index, device_ids):
        selected = label == "G" and partitioned and int(np.asarray(device_ids).reshape(-1)[0]) == 3
        if self.execution == 2 and selected:
            return
        super().receive(label, partitioned, value, index, device_ids)
        if self.execution == 1 and selected:
            super().receive(label, partitioned, value, index, device_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "deliberate_cpu_fault": True, "executions": []}
    with jax.default_matmul_precision("default"):
        report["provenance"] = worker.provenance(json.loads(args.config.read_text()), "cpu")
        capture.Collector = MissingSecondExecutionRecord
        try:
            worker.run_mode(AxisType.Auto, args.output, report["executions"])
        except KeyError as error:
            if error.args != (("G", 3),) or len(COLLECTORS) != 3:
                raise
            first, second = COLLECTORS[1:]
            if first.counts[("G", 3)] != 2 or ("G", 3) in second.values:
                raise AssertionError("Fault did not reproduce the aggregate-count loophole") from error
            if len(first.values) != 10 or len(second.values) != 9 or first.errors or second.errors:
                raise AssertionError("Unexpected fault coverage") from error
            report["observed_error"] = repr(error)
            report["first_execution_G_ordinal3_callbacks"] = first.counts[("G", 3)]
            report["second_execution_G_ordinal3_callbacks"] = 0
            report["first_execution_required_identities"] = len(first.values)
            report["second_execution_required_identities"] = len(second.values)
            report["missing_second_execution_rejected_despite_first_duplicates"] = True
            report["completed"] = True
        else:
            raise AssertionError("A missing second-execution per-device record was accepted")
        finally:
            worker.write(args.output / "negative-report.json", json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
