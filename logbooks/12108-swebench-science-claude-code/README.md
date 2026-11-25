# Claude Code on SWE-bench Science

The experiment compares the existing Pi results with Claude Code 2.1.330 on Marin, using the pinned Qwen3.8-27B checkpoint. The benchmark PR uses Pi. This experimental branch runs task 013 before the five remaining tasks in the previously selected panel (001, 003, 012, 015, 028). Each selected task keeps its original instruction, solver image base, native grader, patch collection and timeout.

The published leaderboard identifies Claude Code at xhigh. The paper and released Claude profile provide no numeric limit for this Qwen run. Across the 119 public traces, task 014 reports 30,890,624 cached input tokens in 186 calls. At least one call used 166,079 cached tokens. Task 015's output counters imply a lower bound of 7,015 tokens on an individual reply. Neither lower bound identifies a configured limit. Run `python reference_limits.py` to reproduce. [Paper](https://arxiv.org/abs/2608.19799) · [Task 014](https://github.com/swescience/swescience.github.io/blob/f16153e5c47c46b2f6f2ebf0aaefae34bdf8c8bd/public/traces/qwen3-8-27b-xhigh/task-014.json) · [Released profile](https://huggingface.co/datasets/OpenMOSS-Team/SWE-bench-Science/blob/997408298e8e4d09978b1120b95c4540d1a71bcb/profiles/claude.env.example)

The local protocol sets both the serving window and Claude Code's declared window to 262,144, the checkpoint's native capacity. Claude Code picks its reply limit on each call. The adapter records its requested limit, Qwen's outgoing wire limit, tool names and token usage in the private run output. Claude Code uses Bash, Edit and Read as in the authors' task 013 trace. The harness version matches the earlier Pier pilot. The authors' exact CLI version remains unknown. [Pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md)

In a CPU control, the pinned Claude Code binary read a file, changed `one` to `two` with Edit, and returned a final answer. A scripted model supplied the three responses. On every request Claude Code asked for 32,000 output tokens and `xhigh`. Its final usage record reported a 262,144-token context. During a benchmark run we record the requested limit afresh on every call, including calls that ask for a different limit.

On the first live tool turn, Claude Code included its environment instructions after the user message. The pinned Qwen template rejects a system message in that position. The adapter collects system text into the single leading system message, preserving the order of system blocks and the order of user, assistant and tool messages. Rendering a synthetic Bash call and its result with the pinned template verified this for both string and block representations of the environment instructions. The authors' provider-side handling of these messages is unknown.

The Iris CPU orchestrator needs the Daytona evaluation credential and regional storage credentials as mounted secrets. Launch the module from the repository root with `python -m logbooks.12108-swebench-science-claude-code.run --cluster <iris-peer> --output <private-regional-prefix> --deadline <UTC-epoch-seconds>`. It acquires a single four-H100 inference allocation. This entry point submits only task 013. The remaining five require a separate launch after inspection of the graded pilot, tool results and wire metadata.

**Task 013 passed** at experimental commit `e50c6ebf70`. The authors also solved it. Claude Code made 22 requests, all with 32,000 requested and forwarded output tokens and `xhigh`. The largest prompt was 49,582 tokens and the largest reply was 3,038. The saved tool results include 23 Read, nine Bash and three Edit calls. This case fit inside the earlier Pi limits. Run `python -m logbooks.12108-swebench-science-claude-code.check_pilot <pilot-directory>` to check the grade and the request records. The `panel` module accepts the same launch arguments as `run` plus `--graded-pilot <pilot-directory>` and checks those records before requesting GPUs for 001, 003, 012, 015 and 028.

## Matched six-task result

All six tasks graded. Marin Claude Code solved three, Marin Pi solved three, and the authors’ Claude Code solved four. The Marin Claude Code grades agreed with the authors on five of six IDs. We selected these tasks using earlier model outcomes. The rates describe this panel only. [Authors’ task matrix](https://github.com/swescience/swescience.github.io/blob/f16153e5c47c46b2f6f2ebf0aaefae34bdf8c8bd/data/task-matrix.json) · [Marin Pi comparison](https://github.com/marin-community/marin/issues/12108)

| Task | Marin Claude Code private tests | Marin Pi solved | Authors solved | Claude Code requests | Largest prompt | Largest reply |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 001 | 1/3 | No | No | 85 | 175,757 | 12,142 |
| 003 | 4/6 | No | No | 38 | 107,222 | 9,499 |
| 012 | 8/10 | Yes | Yes | 26 | 65,990 | 12,324 |
| 013 | 5/5 | Yes | Yes | 22 | 49,582 | 3,038 |
| 015 | 31/31 | No | Yes | 124 | 227,683 | 18,820 |
| 028 | 10/10 | Yes | Yes | 27 | 72,335 | 4,823 |

Every graded attempt passed its public test. The six graded attempts made 322 model requests. Claude Code requested 32,000 output tokens at `xhigh` on each, and the Qwen server received those settings. Five tasks exceeded Pi’s 65,536-token window. One task, 015, also exceeded Pi’s 16,384-token reply cap. The comparison changes both harness and token budget, so it cannot identify which caused the difference on 015.

The original job timer interrupted the first attempts on 001 and 015. Marin preserved the other grades and started fresh attempts on these two IDs under the extended allocation. Run `python -m logbooks.12108-swebench-science-claude-code.check_panel <panel-directory>` to reproduce the grade and request audit. Across the five companion tasks, all 447 requests, including the interrupted attempts, matched the outgoing message digest, requested reply ceiling and reasoning effort. The task 013 pilot stored checks on limits and usage separately.

## Expanded paired comparison

The expansion covers the other 112 tasks with existing Pi and author grades. Task 019 remains on hold. For the checkpoint, rank eligible three-digit IDs by the SHA-256 digest of `swebench-science-claude-code-evidence-v1/<ID>` and take the first 32. The checkpoint and remaining 80 have separate coverage records. Each task has its own request, wire, and grade archive. The driver serves at most 16 concurrent tasks on one four-H100 server and reuses completed grades on restart. Run the `expansion` module with the same cluster, output and deadline arguments as the pilot.

The first expanded launch used commit `0585a5310f`. On task 050 the native attempt reached its atomic result write and exceeded FineStore’s 100 MiB default transaction size. On task 008 the CLI exited after retries of an already outstanding request. For a recovery launch the experiment supplies a 1 GiB transaction ceiling and serializes model completions within each CLI session. Previously committed grades carry forward. A CPU control verified that a per-transaction size override can preserve identical record bytes after the default limit rejects them.
