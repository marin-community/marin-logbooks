# Qwen public-50 pass

This is one fresh Qwen3.6-35B-A3B attempt on each of the 50 public tasks,
including the five disclosed input variants. It is evidence for an ID
proposal; it is not an adopted ID protocol or a reproduction of the paper.
The pilot and its judge-only recovery remain separate.

The policy uses Pi 0.87.0, 16 concurrent trials, a 3600-second agent limit,
a 900-second verifier limit, no solver retries, and at most one immediate
retry of malformed judge output. Both judge attempts and the frozen reports
remain in the verifier archive. The original grading prompt and rubrics are
unchanged. A failed second attempt remains ungraded. Qwen uses H100x4 and
one shared MiniMax judge uses H100x8; the parent has an eight-hour cap.

`fetch_metadata.py` and `plan_inputs.py` select the pinned release without
fetching dataset payloads locally. `submit_images.py` runs the input-layer
preparation remotely, reusing four validated pilot images. Supply its plan
and pilot directories, a unique job name, and the approved credential
sources in the calling process. It creates no model services.

After reading back the image manifests, `prepare_tasks.py` writes public task
recipes under `tasks/` and private registry bindings under ignored
`runtime-tasks/`. Pass `--source-dir`, `--plan-dir` and `--images`; the last
argument must contain all 44 resolved image manifests and their input hashes.
The public recipes retain image digests with a placeholder registry hostname.
`task-manifest.json` records the 50 task identities, input hashes, instruction
and rubric hashes, resource requests, and five variant flags.

Run `validate_runtime.py` remotely in the pinned Harbor environment with the
authorized Daytona eval secret. It resolves snapshot requests through the
actual Harbor policy, ensures the 45 resource-specific snapshots with four
concurrent preparations, and checks hashes in all five modified-input tasks
plus the two largest distinct unchanged-input images. Each validation sandbox
is closed. Read back its `validation.json` and `runtime-report.json` before
running `launch_full.py --submit`; without that flag the launcher only
preflights. These commands allocate resources and require run authorization.
On this shared VM, submit an Iris CPU launcher running
`launch_full.py --submit-in-cluster` instead: full launcher preflight peaked
at 543 MiB locally, above the node's 500 MiB limit. The remote launcher calls
the same `launch_group` entry point and waits for its child evaluation, so
the parent remains alive. Supply the approved named storage and Daytona
secrets, `MARIN_GIT_SHA`, and `MARIN_EVAL_USER`; retain the eight-hour parent
cap. The submission receipt is printed and archived in `IRIS_OUTPUT_DIR`.
`submit_remote.py --validation <verified-validation.json> --job-name <unique-name>`
performs that submission with one CPU, 2 GiB RAM, and a 16 GiB disk. Both
failure and preemption retries are disabled to prevent automatic duplicate
full passes. It refuses to submit without the seven-check validation result.

Offline grading tests use the real OpenAI client with a local mock HTTP
transport and make no network requests:

```bash
PYTHONPATH=logbooks/45-biomnibench-da-id/pilot:logbooks/45-biomnibench-da-id/full \
  .venv/bin/python -m pytest logbooks/45-biomnibench-da-id/full/test_grading.py -n 0
```

The shared runner's invalid-reward regression is covered by
`tests/evaluation/test_harbor_runner.py`. The root VM resource rules apply to
local checks. Full-project type checks need remote resources; an observed
local run used 1.86 GiB and must not be repeated on this shared VM.
