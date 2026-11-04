# Panel-60 baselines: runner pin and arm bindings

Protocol: Ben's gist `penfever/ff6565deebbd68700685807f8e081be1` at history version e35e3eb5 (files copied byte-for-byte to `t310/panel60/ben-gist/`). The run is thinking off, temperature 0, at most 8,192 tokens, 60 items, concurrency 1.

## Runner pin

| item | value |
|---|---|
| runtime commit | `f41b27c558222fac6404546280339ce37a0dfa72` (Ben's `provenance.json runtime_source_commit`), worktree `~/oa/marin-panel60-f41b27c5` |
| runtime patch | Ben's `dashboard-prefetch-workaround.patch` c723de59, applied unchanged (one line in `dashboard_server.py`) |
| runner | `panel60_runner.py` 239c61ad = Ben's `runner.py` 5fcce77b with **one change**: the inference child's priority `PRIORITY_BAND_INTERACTIVE` → `PRIORITY_BAND_PRODUCTION` (orders: production) |
| orchestrator job | as Ben's `/benfeuer/cd16-native60-r2-20261026`: 1 CPU, 2 GiB, 5 GiB disk, cw-rno2a. It starts one H100x8 vLLM serve (the inference child) and releases it when done |
| serve | Ben's `serving.json` unchanged except name, location and revision: TP1, DP8, expert parallel, max_model_len 12288, reasoning parser grug, tool parser hermes, `--override-generation-config {"temperature":0}`, bfloat16 |

## Arms

Files: `t310/panel60/arms/<arm>/cd16_protocol.json` and `cd16_serving.json`, built by `t310/panel60/build_arms.py` 337b7c62 from `ben-gist/` (rerunning it reproduces all four files byte for byte).

| arm | model name (served_model) | location | revision | protocol / serving sha256 |
|---|---|---|---|---|
| v0p1 | `snowball-v0p1-panel60` | `open-athena/Marin-Snowball-67B-A2B-v0.1` (public; anonymous model_info OK) | `adc0078a19a4383bc998fcdb43867b3360baf5f3` | e9b3fcdb / 53cdb189 |
| v0p1-sysdefault | `snowball-v0p1-sysdefault-panel60` | `s3://marin-rno2a/marin/checkpoints/snowball-identity-sysdefault/v0.1-sysdefault/hf` (v0.1 weights; chat_template.jinja 2b0cee0e, the published sysdefault template) | none (S3) | ccccbe77 / 2ebf6fb7 |

- **Rows:** the 60 `identity_rows` are byte-equal to Ben's protocol. The canonical sha256 0fafe46f is reproduced and equals his `rows_canonical_sha256`. Only `served_model` and `run` differ in each protocol file.
- **sysdefault template:** it injects "You are Marin Snowball, an AI model created by Open Athena." only when the first message is not a system message. Ben's 3 rows that carry a date-only system message (`…-033-000/002/003`) therefore get **no** identity prompt in this arm. That is the template's own behaviour, and it is reported as such.
- **Grading:** Ben's rubric (`review.json` and `record_review.py`). Each row records whether it was graded by a model (an agent) or by hand.
