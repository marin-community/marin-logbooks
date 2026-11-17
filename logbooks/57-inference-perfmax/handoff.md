# Inference validation and tuning

Both selected BF16 deployments pass the corrected shared suite under unchanged
provisional numerical policies. The fresh-context Snowball campaign completed
two valid one-hour windows and retained C128 for its higher throughput. The
required review of goal completion remains pending. [PR #11320](https://github.com/marin-community/marin/pull/11320)
is draft and unmerged.

Contribution [e929e4528e](https://github.com/yonromai/marin/commit/e929e4528e399e8f8d1dd6f5f4bf6909532b020f)
contains the [82-line skill](../../.agents/skills/inference-perfmax/SKILL.md),
[manual runner](../../experiments/evaluation/knotmeter/README.md) and
[shared checks](../../experiments/evaluation/inference_correctness/README.md).
Forty-four focused CPU/HTTP/CLI cases passed; the 20 affected report cases passed
again after the reporting-only scope fix. Required lint and type checks pass.
CI is terminal on `e929e4528e`: 17 successful checks and 20 skipped. Earlier
[review resolutions](analysis/goal-review-resolution.json) apply to unchanged
code; they do not replace the new goal's final review.

The targets are [Snowball v0.1.1](../../lib/marin/src/marin/testing/inference/resources/endpoint/snowball-v0.1.1.json)
at HF revision `59852d6f07261722bbec1edb6ab148317156b58e`, on eight H100s,
and [Hero/Mochi SFT step400407](../../lib/marin/src/marin/testing/inference/resources/endpoint/hero-sft-step400407.json),
on sixteen GB200s. Hero and Mochi name the same model. The fixtures pin the
weights, export/config hashes, tokenizer, EOS, reference manifests and policies.
Both use reviewed vLLM source `0ac9dcb157ee86ccb447bc717259159c41ad456a`, BF16,
compilation and CUDA graphs. Plans and reports retain the actual settings.

| Check | Snowball/H100 | Hero/Mochi/GB200 |
| --- | --- | --- |
| Identity and serving configuration | Passed | Passed |
| Fixed Levanter reference | 64 histories, 1,770 positions passed | 37 histories, 15,855 positions passed |
| Batch dependence | Four overlapping requests on engine 0; four comparisons passed | Same coverage, passed |
| Cache reuse | Nine prefill, prepared-branch and decode-created reuse boundaries passed | Same coverage, passed |
| Exact cache controls | Nine branches and one repeat: zero error | Same coverage, zero error |
| Decode versus recomputation | 64 generated-token comparisons passed | Same coverage, passed |

The [Hero pytest report](analysis/hero-qualified-a2-summary.json) and
[Snowball CLI audit](analysis/snowball-c64-attempt0-root-audit.json) identify the
native archives and report hashes. The coordinating agent verified their bytes
and outcomes.
Snowball completed its checks before Kueue interrupted native warmup. Its
[derived receipt](analysis/snowball-c64-attempt0-derived-receipt.json) and
[recovery record](analysis/snowball-c64-attempt0-receipt-recovery.json) preserve
the missing-receipt fact and original failed worker record. That attempt has
no completed performance window. Both interfaces call the same checking code.

A cache miss can be legitimate. In the pinned runtime, retention interval zero
keeps selected replay boundaries and shared-prefix junctions. It does not promise
arbitrary decode-created checkpoints. The presets now retain checkpoints every
128 tokens. For example, after a 127-token prompt generates 257 tokens and gains
a three-token suffix, Hero reuses 256 tokens and Snowball 368. Both hits reach
generated history. Every required boundary still has to demonstrate actual reuse.

Equal hit counts also do not prove equal prefill steps. The
[pinned scheduler counterexample](analysis/hero-scheduler-split-reproduction.json)
shows a 144-token history with a 112-token joint hit can compute its suffix as
32 tokens or as 16 plus 16. A private live trace failed during startup and
produced no split records; the earlier live difference's cause remains unproven.
The corrected checker retains initial seed/branch comparisons under the existing
changed-step bound. It also fully prepares each branch's usable prefix, runs a
sibling that differs inside that region, then requires the original branch to
reuse the full prefix and return identical scores. The
[injected-state regression](analysis/branch-checker-regression.txt) fails the
new exact check while the old checker misses it. No numerical bound was relaxed.

The [numerical policy explanation](numerical-policies.md) gives each limit's
basis, observed statistics and remaining weakness. Hero's reference margins
have prior BF16 control evidence, with different top-k and launch settings.
Snowball's reference cutoffs and the nonzero cache, batch and decode cutoffs
remain initial screens. The metric bounds absolute probability error over
returned top-50 lists; it does not establish full-vocabulary KL, acceptable
trainer probability ratios or RL learning quality. Validated checking behavior
and suitable acceptance limits are separate claims. Selected reports and saved
configurations keep that RL scope limit visible. Goldens stayed fixed; this
phase generated no new reference scores. The older step42150
[release gate](analysis/runtime-adoption-11241.json) remains regression evidence
for another checkpoint, with its original coverage and 0.125 bound.

| Snowball point | Reported completion tokens/s/GPU | p90 TTFT | p90 per-request completion | Profiling successes/errors | H100-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| C128 baseline | 124.39607 | 103.33113 s | 297.61639 s | 5,165 / 1 | 11.368889 |
| C64 candidate | 92.30190 | 60.39601 s | 154.10088 s | 3,392 / 0 | 12.738633, including preemption |

The [baseline audit](analysis/snowball-baseline-root-audit.json) and
[C64 audit](analysis/snowball-c64-root-audit.json) verify the native exports,
check and archive hashes, phase counts and allocations. The one-hour recipe
reports successful-request spans of 3,629.77401 and 3,629.10885 seconds under its
native drain convention. Throughput counts reported completion tokens from
successful profiling requests, divided by that span and eight physical GPUs.
Completion latency is per request. These metrics do not measure a whole session
or RL step. Warmup has 989 nonempty responses and 417 empty-content errors at
C128, versus 510 and 192 at C64. Native deadline credit cancellations are 59 and
25. The runner's 1% error limit counts profiling requests; warmup errors are
reported separately. Native validity, duration coverage and unchanged-host
checks pass for both windows.

The [fresh-context tuning agent](snowball-tuning-01a1eabe.md) found prior results and
upstream guidance, and kept old-runtime measurements separate from its baseline.
Live cache pressure led it to replace the planned prefill-budget diagnostic with
a normal C64 run, avoiding another startup and warmup for a short screen. It
reused the same verified CPU-prepared workload. The
[submitted-plan audit](analysis/snowball-c64-plan-root-audit.json) confirms
concurrency is the only profile difference. The baseline checker source is
`548b9fd14d`; C64 source `06b9a63682` includes the corrected branch checks.
The runtime, serving settings, workload, numerical policy and AgentX protocol
match. C64 loses 25.80% throughput while reducing p90 TTFT by 41.55% and completion
latency by 48.22%. The tuning agent retained C128 for the primary throughput
objective. Its 5% similarity band was a predeclared heuristic, not a numerical
target supplied by Romain. Only two loads were tested; the campaign does not
establish a global optimum or a universally acceptable latency.

Inspection of the tuning agent's commands and exports also found one avoidable shell
wrapper error: editing the active Bash script made it exit 127 after a successful
CLI and native run. Its last command now uses `exec`; C64's local wait exited zero.
Transient port-forward errors and delayed result publication did not cause extra
GPU runs. Reproducible [commands](analysis/snowball-tuning-01a1eabe.sh), native
identities, frozen plans and exports remain in the campaign records. The saved
[decision](analysis/final-campaign-decision.json) selects C128. The
[report refresh audit](analysis/report-refresh-audit.json) confirms that the final
terminal report and saved summary match, the recommendation reloads with the
tested settings and exact runtime, and the reporting update preserves that choice.

The [cost ledger](analysis/complete-validation-costs.json) records 25.363078
H100-hours and 13.576868 GB200-hours for this validation phase, including both
C64 attempts. The two-point campaign accounts for 23.964314–24.107522 H100-hours;
the range reflects the preempted pod's last confirmed running time versus its
Iris terminal time. The recorded total across earlier phases is 32.003078
H100-hours and 27.947243 GB200-hours, using the upper estimate. Two older
unadmitted Snowball attempts retain incomplete allocation receipts; no allocation
was observed. Setup, checks, warmup, drain and failures count. Queueing is
excluded. Owned peaks were eight H100s and sixteen GB200s, below the 48/128
ceilings. Interactive priority was requested; Iris's normal fallback was allowed.
All owned jobs and [campaign pods](analysis/snowball-c64-cleanup-root.json) are
terminal, and all resource-board entries are closed.
