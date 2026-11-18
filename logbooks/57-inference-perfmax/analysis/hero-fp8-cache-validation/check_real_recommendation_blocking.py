# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise recommendation rejection with an unchanged retained failed trial."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.evaluation.knotmeter.report import accepted

source, campaign, receipt = map(Path, sys.argv[1:])
raw = source.read_bytes()
result = json.loads(raw)
measurement = result["measurement"]
assert result["job_succeeded"] and result["cost_complete"]
assert result["manifest"]["series"]["duration"] == 3600
assert not result["manifest"]["diagnostic"]
assert all(measurement["validity"][key] for key in ("protocol_valid", "submission_valid", "gang_valid"))
checks = measurement["checks"]
required = result["manifest"]["series"]["profile"]["checks"]
assert sorted((row["name"], row["kind"]) for row in checks) == sorted((row["name"], row["kind"]) for row in required)
numerical = [row for row in checks if row["kind"] == "numerical"]
assert numerical and any(row["status"] == "failed" for row in numerical)
assert all(row["status"] == "passed" for row in checks if row["kind"] != "numerical")
assert all(
    row["checkpoint_revision"] == result["manifest"]["series"]["profile"]["model"]["revision"]
    and row["runtime_source"] == result["manifest"]["vllm_release"]["source_commit"]
    for row in numerical
)
assert not accepted(result)
campaign.mkdir(parents=True, exist_ok=False)
trial = campaign / "trials" / result["trial_id"]
trial.mkdir(parents=True)
shutil.copy2(source, trial / "result.json")
command = [
    sys.executable,
    "-m",
    "experiments.evaluation.knotmeter.cli",
    "report",
    "--campaign-uri",
    str(campaign),
]
rendered = subprocess.run(command, capture_output=True, text=True, check=True)
report = rendered.stdout
assert (campaign / "summary.md").read_text() == report
rejected = subprocess.run([*command, "--recommend", result["trial_id"]], capture_output=True, text=True)
assert rejected.returncode == 1
rejection = rejected.stderr.strip()
assert not (campaign / "recommended.yaml").exists()
assert not (campaign / "decision.json").exists()
assert source.read_bytes() == raw
record = {
    "scope": "Existing report API with a byte-identical real result; no check status or metric was edited",
    "trial_uri": result["trial_uri"],
    "result_sha256": hashlib.sha256(raw).hexdigest(),
    "report_module_path": str(Path(accepted.__code__.co_filename).resolve()),
    "report_module_sha256": hashlib.sha256(Path(accepted.__code__.co_filename).read_bytes()).hexdigest(),
    "public_report_exit_code": rendered.returncode,
    "public_recommendation_exit_code": rejected.returncode,
    "validity": measurement["validity"],
    "cost_complete": result["cost_complete"],
    "failed_checks": [{key: row[key] for key in ("name", "kind", "status")} for row in numerical],
    "accepted": False,
    "rejection": rejection,
    "accepted_profile_written": False,
    "accepted_decision_written": False,
    "terminal_and_saved_report_match": True,
}
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
