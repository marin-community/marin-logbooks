# Shared inference correctness handoff

Required checks now block accepted recommendations when evidence is pending,
failed or scoped to another checkpoint/runtime. Diagnostic measurements and
reports remain available. Neither selected profile is qualified. No accepted
recommendation or tuning campaign was produced.

[Draft Marin PR #11320](https://github.com/marin-community/marin/pull/11320)
contains source `2cd493de0bcedf598a61bd2974b44bd65f15857a`. The
[82-line skill](../../.agents/skills/inference-perfmax/SKILL.md) guides campaign
choices; the [runner README](../../experiments/evaluation/knotmeter/README.md)
gives commands. CI on this source passed: 17 successful checks and 20 skipped.
The 42 focused CPU/HTTP/CLI cases, lint and type checks pass. One High-tier
review used independent, read-only Claude, Codex and AGY readers on source
`7db04440dcd9873c517d034a585d67e69e1e8f8c`. Source `2cd493de0b` fixes its
findings; those fixes received targeted tests and replay of retained observations.
The [review resolutions](analysis/goal-review-resolution.json) map each finding
to its change, validation and remaining limits.

The targets are Snowball v0.1.1 at HF revision
`59852d6f07261722bbec1edb6ab148317156b58e`, on eight H100s, and Hero/Mochi SFT
step400407, on sixteen GB200s. Hero and Mochi name the same model. Hero's export
manifest is `1a7886b05df25338a12642b25f25525c0eb6c5551384934a2f089d7ec5b4a88d`.
Both use BF16 and reviewed vLLM source
`0ac9dcb157ee86ccb447bc717259159c41ad456a`, with compilation and CUDA graphs.
Deployment checks verify weights and export/config hashes, tokenizer revision,
runtime version, GPU family, BF16, quantization and KV-cache policy, EOS, and
enabled CUDA graphs and prefix caching. Reports retain the exact scheduler,
parallel, cache and compilation settings. The resolved plan records the launch
configuration; a pass applies to that recorded deployment.

| Check | Hero/Mochi | Snowball |
| --- | --- | --- |
| Deployment identity | Passed | Passed |
| Saved reference comparisons | Passed under provisional bounds: 37 histories, 15,855 positions | Pending: 13/64 histories and 194/1,770 positions; observed subset is within bounds |
| Engine batching | Passed: four requests observed running on engine 0; four next-token comparisons | Passed with the same limited coverage |
| Decode versus fresh recomputation | Passed: 64 emitted-token comparisons | Passed: 64 emitted-token comparisons |
| Required cache/state suite | Failed coverage; incomplete qualification | Failed coverage; incomplete qualification |
| Qualified normal AgentX window | Not run | Not run |
| Accepted recommendation | Blocked | Blocked |

Pinned Levanter runs produced the golden next-token log probabilities from
matching BF16 checkpoints and exact token histories. Each manifest records the
reference revisions and export lineage. At each selected position, the checker
compares the two top-50 lists. Shared tokens have an exact probability difference.
A token missing from one list has probability between zero and that list's
50th probability. The checker takes the largest possible error over the union
of returned tokens, then reports its maximum, p99, p90 and fraction above 0.075
across positions. This does not measure full-vocabulary KL or trainer ratios.

The parser now excludes an extra observed prompt token below the top 50 when
finding that cutoff. Including it could lower the cutoff and understate the
bound. It remains available for decode rescoring. An independent, normalized
80-token CPU example reproduced the bug. The corrected comparator replayed
saved observations without GPUs; input archives and output report hashes were
verified. Original observations and failed reports remain intact.

| Reference group | Positions | Maximum | p99 | p90 | Fraction above 0.075 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hero short | 11,370 | 0.470978 | 0.093855 | 0.026770 | 0.014864 |
| Hero long | 4,485 | 0.278653 | 0.054485 | 0.024352 | 0.004682 |
| Snowball observed subset | 194 | 0.078634 | 0.064165 | 0.025497 | 0.005155 |

The [Hero replay pointer](analysis/hero-reference-replay.json) and
[Snowball replay pointer](analysis/snowball-reference-replay.json) contain the
full report locations and hashes. The reports retain the original runtime,
model, policy and other stage outcomes. Only reference comparisons were
recomputed. Replay source is `50c2cc5fcc70f56ac6f294a4a5dc62601b4ae098`.

For cache checks, a *split* is the number of prompt tokens reused before the
remaining tokens are computed. Fresh versus warmed execution changes that
split. Two warmed repeats with equal hit counts use the same split and require
exact agreement. Changed splits use separate provisional limits. These checks
compare next-token distributions after the same history, including alternate
branches. Planned boundaries are 127/128/129, 255/256/257 and 2047/2048/2049.

Both live checks reached boundary 127. A 144-token branch had zero cached tokens
when fresh and 112 after seeding its 127-token prefix. The decode probe then
used a 127-token prompt, generated 257 tokens and appended three branch tokens:
127 + 257 + 3 = 387. Coverage requires a hit beyond token 127, proving reuse of
generated tokens. Both profiles still hit only 112. The old checker stopped
there, before the other eight boundaries and final matched-repeat control.
Complete same-split and changed-split groups remain missing; raw prefill
responses exist at boundary 127. The corrected checker retains those comparisons
and continues after missing decode reuse. Real HTTP fixtures validate this
behavior. The corrected full suite has not run on either GPU profile.

The retained settings use prefix-cache retention interval zero. The
[pinned runtime documentation](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/vllm/config/cache.py#L148)
says this retains replay boundaries and shared-prefix junctions rather than
periodic checkpoints. It may explain the missing reuse; that cause remains an
inference. No retention setting, numerical limit or required check was changed
to make a profile pass. Snowball's final transport retry stayed
unadmitted for over half an hour and was canceled. Its remaining 51 reference
histories are pending. No more GPU work followed.

The policy files separate fixed goldens from provisional error limits. The
limits below are engineering screens; broad calibration and RL learning
qualification were outside this pass.

| Comparison | Maximum | p99 | p90 | Fraction above 0.075 |
| --- | ---: | ---: | ---: | ---: |
| Hero short reference | 0.65 | 0.125 | 0.04 | 0.035 |
| Hero long reference | 0.35 | 0.09 | 0.04 | 0.02 |
| Snowball reference | 0.25 | 0.125 | 0.05 | 0.05 |
| Equal cached split | 0 | 0 | 0 | 0 |
| Changed prefill split | 0.20 | 0.12 | 0.06 | 0.15 |
| Batch | 0.20 | 0.18 | 0.15 | 0.50 |
| Decode | 0.25 | 0.15 | 0.075 | 0.15 |

Hero's reference limits use the matching BF16 controls in
[#11430](https://github.com/marin-community/marin/issues/11430). The other path
limits remain initial screens. No bound changed during final validation.
The existing step42150 gate passed 64 prompts each at pipeline sizes
one and two on this vLLM runtime, with maxima 0.124960 and 0.124768 under its
original 0.125 bound, per merged
[#11241](https://github.com/marin-community/marin/pull/11241). This is regression
evidence for another checkpoint. Its pass does not qualify either preset.

Goldens live at `s3://marin-us-east-02a/marin/inference/goldens/`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Finference%2Fgoldens%2F)).
The [Snowball fixture](../../lib/marin/src/marin/testing/inference/resources/endpoint/snowball-v0.1.1.json)
and [Hero fixture](../../lib/marin/src/marin/testing/inference/resources/endpoint/hero-sft-step400407.json)
pin each manifest hash, checkpoint and tokenizer. The manifests hash the token
histories and scores and record reference code and export lineage. Routine
checks reuse these files across candidate runtimes. They do not regenerate
references.

Look up references without allocating GPUs:

```sh
uv run python -m marin.testing.inference.check_endpoint --fixture snowball-v0.1.1 --lookup
uv run python -m marin.testing.inference.check_endpoint --fixture hero-sft-step400407 --lookup
```

The pytest launcher owns deployment and writes an execution receipt with the
durable run URI. These commands repeat the suite with unchanged serving settings.
They can collect the remaining prefill comparisons but may still fail the
decode-reuse requirement. A rerun alone does not establish qualification:

```sh
uv run python -m experiments.evaluation.inference_correctness.launch \
  --profile snowball-h100 --set memory=384GB --name snowball-correctness --output snowball-execution.json
uv run python -m experiments.evaluation.inference_correctness.launch \
  --profile hero-gb200 --name hero-correctness --output hero-execution.json
```

For a dedicated existing endpoint with `VLLM_SERVER_DEV_MODE=1`, the
[endpoint CLI and pytest commands](../../experiments/evaluation/inference_correctness/README.md)
call the same functions. Perfmax retains the CLI report, requires both its exit
status and result to pass, then resets the cache before AgentX warms the server
for its trace replay. A normal performance window uses the pinned one-hour
AgentX protocol and its submission-validity checks. Recommendation acceptance
also requires all configured checks, matching numerical evidence, the same
serving hosts throughout measurement and complete cost records. An ordinary
run stops before AgentX on a failed check. An explicit diagnostic may continue
and remains ineligible for recommendation. Reports show the latest serving
attempt's check outcome even when no performance window started.

The new shared-check perfmax GPU path has not run end to end after integration.
The pinned-image CPU CLI preflight, real local HTTP tests and GPU pytest runs
cover its components. No fresh-context tuning child ran because neither profile
passed every required check. The planning journey did expose a false message
that fixture checks lacked commands; it was fixed and checked with a real CPU
`plan` invocation.

Recorded cost for this shared-correctness goal is **4.802 H100-hours and
9.338 GB200-hours**, including reference generation, setup, failed attempts and
drain. Earlier manual AgentX diagnostic trials are outside this cost total.
Two unadmitted Snowball
attempts retain incomplete allocation receipts; no allocation was observed.
Their uncertainty remains visible in [the cost ledger](analysis/correctness-costs.json).
Observed concurrent peaks were eight H100s and sixteen GB200s, below the agreed
48/128 ceilings. All owned GPU jobs are terminal and all resource-board rows are
closed. No child work remains running.

Experimental source, negative findings and receipts are preserved on the
[research branch](https://github.com/marin-community/marin-logbooks/tree/codex/57-inference-perfmax)
through the [research-engineering workflow](../../.agents/skills/research-engineering/SKILL.md).
The contribution and index PRs remain draft and unmerged. Trainer/rollout
log-probability alignment, sampling transformations and RL learning quality
remain unqualified; no new investigation was started for them.
