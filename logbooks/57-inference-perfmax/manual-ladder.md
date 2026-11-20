# Manual inference ladder validation

Real BF16 [Snowball](https://github.com/yonromai/marin/blob/4546acc39a32a528841bcdc22b20e744ada0265b/experiments/evaluation/knotmeter/profiles/snowball-h100.yaml) Probe and Screen trials and a [Hero](https://github.com/yonromai/marin/blob/4546acc39a32a528841bcdc22b20e744ada0265b/experiments/evaluation/knotmeter/profiles/hero-gb200.yaml) Probe across four hosts completed
at concurrency eight, using the full pinned 393-session corpus. Hero and Mochi
name the same model. The Screen passed native benchmark admission and local
metric coverage. The actual CLI rejected it as a recommendation because it was
not a Confirm. No tuning campaign or full-hour Confirm was run.

[Marin #11320](https://github.com/marin-community/marin/pull/11320) depends on the
shared checker in [#11794](https://github.com/marin-community/marin/pull/11794).
Live source was [`4ace869cd6`](https://github.com/marin-community/marin-logbooks/tree/4ace869cd643b5682df99e5c434eeebe4f2503ce); review source `600aa99a90` put the same feature patch
on main `346d446d46ad` and checker `55b8183e57df`. Final source
`4546acc39a32` adds report fixes and guidance from that review. The measurement,
serving and shared checker paths remain unchanged. The
[earlier research](https://github.com/marin-community/marin-logbooks/blob/a858ad5f339e9d3a9d0550c12c614e7d3cbb82c0/logbooks/57-inference-perfmax/handoff.md)
and its older-runtime results remain preserved.

All three trials use vLLM `47a8e415bff17b9376f02d9b92edcc67e962eaee` and AgentX
1.0.12, build `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`. Probe measured 180 seconds
with one additional warmup request per lane and the unsafe override. Screen
measured 900 seconds with ten additional warmup requests per lane and no unsafe
override. Startup, checks, warmup and drain fall outside those windows.
The unsafe override admits Probe's short window below the normal 900-second
minimum. Probe remains diagnostic.

A recommendation requires a valid Confirm of at least 3,600 measured seconds,
at least ten additional warmup requests per lane and no unsafe override.
Required numerical checks must pass for the measured checkpoint and runtime.
GPU attempts must stay unchanged during profiling, and allocation cost must be
complete.

The full corpus is available to each trial; a timed window can finish only a
subset of requests. The table counts profiling completions. Correctness checks
use separate numerical fixtures.

| Trial and retained result | Allocation | Profiling requests | Allocated GPU-hours |
| --- | --- | --- | ---: |
| Snowball Probe [20261120T102502Z-ff378b058bee](s3://marin-us-east-02a/marin/knotmeter/campaigns/manual-ladder-snowball-20261120-01a1d379/trials/20261120T102502Z-ff378b058bee/result.json) · [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fmanual-ladder-snowball-20261120-01a1d379%2Ftrials%2F20261120T102502Z-ff378b058bee%2Fresult.json) | 8 H100 | 7 successes, 0 errors | 1.669 |
| Hero Probe [20261120T100206Z-ace8b9c7e7f6](s3://marin-us-east-02a/marin/knotmeter/campaigns/manual-ladder-hero-20261120-01a1d379/trials/20261120T100206Z-ace8b9c7e7f6/result.json) · [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fmanual-ladder-hero-20261120-01a1d379%2Ftrials%2F20261120T100206Z-ace8b9c7e7f6%2Fresult.json) | 16 GB200, four hosts | 4 successes, 0 errors | 6.181 |
| Snowball Screen [20261120T105318Z-e8d90c5ee70d](s3://marin-us-east-02a/marin/knotmeter/campaigns/manual-ladder-snowball-20261120-01a1d379/trials/20261120T105318Z-e8d90c5ee70d/result.json) · [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fcampaigns%2Fmanual-ladder-snowball-20261120-01a1d379%2Ftrials%2F20261120T105318Z-e8d90c5ee70d%2Fresult.json) | 8 H100 | 388 successes, 0 errors | 3.613 |

All five shared check groups passed under the pinned provisional
[Snowball policy](https://github.com/yonromai/marin/blob/55b8183e57dfc2295711a845e6b34b2b665709c2/experiments/evaluation/inference_correctness/fixtures/snowball-v0.1.1.json) and
[Hero policy](https://github.com/yonromai/marin/blob/55b8183e57dfc2295711a845e6b34b2b665709c2/experiments/evaluation/inference_correctness/fixtures/hero-sft-step400407.json):
deployment, saved references, batching, cache reuse and decoding. Checks finished
before measurement. Native exports, raw check reports, GPU identities and
complete allocation costs were retained. Both Probes remain diagnostic. Serving
checks do not establish trainer log-probability alignment or RL learning quality.

The Screen reported 37.4 output tokens/s/GPU, p90 TTFT 6.0 seconds and p90 request
completion 13.9 seconds. One window provides no measured repeatability. Default
loads of C128 for Snowball and C64 for Hero were not validated on this source.

Two earlier Snowball attempts lost their CPU coordinator and stopped before a
retained window. They cost 0.104 and 1.655 H100-hours. The second attempt's observed
check pass has no raw report, so it remains pending. An earlier Hero attempt
failed at the shared gang-launch interface and cost 0.092 GB200-hours; the fix
has a regression through the real gang launcher. Total validation cost,
including these failures, is 7.042 H100-hours and 6.273 GB200-hours.
Totals use unrounded allocation receipts. All owned
GPU jobs are terminal and all owned resource rows are closed. Interactive
priority was requested; normal Iris Batch fallback was accepted.

The final focused suite passed 83 CPU/HTTP tests. One isolated replay passed
through the real pinned upstream tools, streaming HTTP, cache restoration and
physical-GPU normalization; three earlier stage variants also passed. Lint and
types passed. The final affected suite recorded 652 passes and five failures. All five
reproduce on main `346d446d46ad` with the same empty fixture hashes. An earlier broader run reached its
600-second budget after 1,972 passes and nine failures; all nine reproduced on
its base, `f6484db03743`. That broader run remains incomplete.

Three isolated skill readers all started with checks and Probe. All confused
trial IDs with storage URIs in report commands; help and skill now distinguish
them. The High review found a saved recommendation could lose its measured
concurrency override. CPU regressions reproduced and fixed that bug and a report
view that hid the saved selection. Review responses and model provenance limits
are retained with the evidence.

The runner reruns the full checker for each trial. Prior evidence helps choose
experiments; it does not skip those checks. Keep `run` attached for complete
allocation tracking: lost Pod timestamps can block acceptance after detached
work. Repeats use separate runner invocations and manual row comparisons. Native
multi-run and confidence aggregation are not exposed. InferenceX's aggregate
`tp` is its physical-GPU normalization input; the resolved profile and deployment
checks record actual TP, DP and host layout.

The [evidence bundle](s3://marin-us-east-02a/marin/knotmeter/validation/manual-ladder-20261120-01a1d379/evidence-a3d10be1edff37e198218833c0b1ab2fd3e7a512091f257471938e6c99776a12.tar.gz) · [Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fknotmeter%2Fvalidation%2Fmanual-ladder-20261120-01a1d379%2Fevidence-a3d10be1edff37e198218833c0b1ab2fd3e7a512091f257471938e6c99776a12.tar.gz)
contains test logs, live metadata, reader packets and responses, reviews and their
resolution. Its uploaded bytes were read back and matched SHA-256
`a3d10be1edff37e198218833c0b1ab2fd3e7a512091f257471938e6c99776a12`. Raw native AgentX archives remain at each trial's receipt URL.
Filer requires Marina login. Current PR CI is pending.
