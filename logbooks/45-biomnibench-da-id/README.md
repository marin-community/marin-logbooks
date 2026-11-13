# BiomniBench-DA pilot archive

Start with [notebook.md](notebook.md) for decisions, evidence and limits, and [protocol.md](protocol.md) for the proposed evaluation protocol. This branch starts from Marin `246f795749166285a864e891e56facfbcf9127e7`; inherited GitHub workflows are removed solely for archive publication. It must not be merged into either repository's main branch.

## What is preserved

`pilot/` contains the scorer, four task bundles, source metadata, model and harness configurations, streaming image preparation, sandbox validation and the three-task launcher. The control's criterion grades, offline scoring controls and successful runtime report are small evidence files. `source-provenance.json` hashes the original workspace files before publication edits. The original execution occurred before this archive commit.

Private registry addresses are replaced by `registry.example.invalid`. Full serving logs, private runtime records and input data remain outside Git. An authorized operator must supply registry access, immutable image references and credentials. The model and dataset pins are public. Do not interpret placeholder image references as downloadable artifacts.

## Reproduction prerequisites

Run commands from the repository root with Python 3.12 and the workspace dependencies installed. Supply CoreWeave credentials through the existing supported environment and Iris secret mechanisms. No credential values or VM-specific credential paths are stored here. Set `DAYTONA_EVAL_SECRET_SOURCE` to the operator's approved Iris secret reference, and choose unique `BIOMNI_IMAGE_JOB_NAME` and `BIOMNI_RUNTIME_JOB_NAME` values. The launcher currently selects the same `marin` submission cluster, `cw-us-east-02a` peer and `us` Daytona target used by the pilot; confirm availability before submitting.

For existing images, replace the placeholder registry prefix in the control/remaining image manifests and task configurations with an accessible registry holding the recorded digests. The digest pins remain authoritative. For rebuilding, stage the source control data using the URL and hash in `source-data-manifest.json` under `pilot/build-contexts/da-4-6/data/mmc5.csv`, and build the preserved Dockerfile remotely through TaskForge's image builder. Record the resulting base/control image manifest. The historical `publish_input_layer.py` and `submit_input_layer.py` preserve the control's missing-input repair path; they are unnecessary when the base build contains the verified control data.

`submit_remaining_images.py` builds the remaining images from the base and the pinned public inputs. It downloads inputs remotely, verifies known hashes and file sizes, records the complete resulting hashes, and pushes immutable image references. This allocates a bounded CPU job. Save its output `remaining-images.json` beside the scripts. Image regeneration can change digests and must be followed by fresh validation.

## Validation and submission

These commands may allocate remote resources; inspect configuration and budgets before running them. The four-task pilot scope does not authorize a full benchmark.

```bash
export PYTHONPATH=lib/rolloutengine/src:lib/taskforge/src:lib/taskcompendium/src:lib/shellbox/src
.venv/bin/python logbooks/45-biomnibench-da-id/pilot/submit_remaining_runtime_check.py
```

This submits three sequential sandbox checks, with no model calls and a 2,100-second CPU-job limit. Read back its `remaining-runtime-report.json` through the documented Iris output archive and fsutil workflow. Preserve old reports under separate attempt names before replacing working manifests or receipts. Verify all three statuses and cleanup records, then run:

```bash
.venv/bin/python logbooks/45-biomnibench-da-id/pilot/finalize_remaining_tasks.py
.venv/bin/python logbooks/45-biomnibench-da-id/pilot/launch_remaining.py
```

The final command only runs preflight. Adding `--submit` requests four H100s for Qwen and eight for MiniMax, two concurrent trials, zero trial retries, a 30-minute scheduling timeout and a four-hour parent cap. It excludes the completed control. Do not run it while the recorded continuation is active. Inspect the exact Iris job before replacing its submission receipt; a missing local receipt does not prove no remote job exists.

The public copy has been checked mechanically, but has not been used for a fresh paid evaluation. The original runtime checks and control grade apply to the preserved executed inputs, not automatically to future rebuilt images or publication edits.

## Follow-up ownership

The current continuation's exact run identity is in the notebook. Verify durable grades and EvalDash ingestion before changing its status to scored. Check child-job and sandbox cleanup. No scheduled host follow-up tool was available when this archive was prepared; a persistent monitoring heartbeat has not been installed. The next operator must resume monitoring the existing job, not launch a duplicate.
