# A11 BFCL

A11 averages **31.2195%** across five complete evaluations of the frozen 123-task BFCL tool-calling parity slice. Its registered difference from plain v0.1 is +3.5772 percentage points, with a 95% run-bootstrap interval of [−0.4878, +7.4797]. The interval spans zero. “Parity” means the same pre-existing 123 task IDs and evaluator configuration for every model; no new task selection or random-sample claim is made. Each run counts native verifier reward-1 outcomes divided by 123. These percentages describe this restricted slice, not the full Berkeley Function Calling Leaderboard.

| Runs, logical slot order | Complete/planned | Mean | Sample SD | Min–max | Mean 95% CI |
|---|---:|---:|---:|---:|---:|
| 41, 35, 38, 45, 33 correct /123 | 5/5 | 31.2195% | 3.8821 pp | 26.8293–36.5854% | [28.2927, 34.3089]% |

| Registered comparison | Difference, pp | Unadjusted 95% CI, pp |
|---|---:|---:|
| Primary: A11 − fresh v0.1 | +3.5772 | [−0.4878, +7.4797] |
| Secondary: A11 − A3 | +0.1626 | [−4.0650, +4.2276] |
| Secondary: A11 − A9 | Pending | A9's final evaluation chain is still active |

A11 and A3 train independently from the same Marin-Snowball-67B-A2B-v0.1 weights with fresh optimizer states, seed 310, and the same learning-rate/freezing recipe. A11 does not resume A3's trained checkpoint. Training combines general chat/code/maths/tool replay with identity answers naming Marin's Latest MoE and the Marin Community as developer/trainer. Every identity example is rendered without reasoning. A11 extends the same ordered training ledger to 72 updates; A3 uses 36. A11's first 36 packed updates match A3's data exactly, but cumulative exposure increases; no byte-identical early weight trajectory is claimed. A9 is the separately registered 72-update recipe retaining a deterministic 10% subset of eligible identity reasoning traces; its pending contrast will be appended after all its chains finish or exhaust. No partial A9 mean is used here.

The plain-v0.1 reference reuses five fresh evaluations of the published starting checkpoint, without historical-score pooling. We resample complete runs independently within each condition, sharing comparator draws across registered contrasts: Python seed 297, 10,000 iterations, percentile order statistics 249/9749. Earlier results and the registered A9/A10/A11 random-stream block starts remain unchanged. A11 uses its original five-run draw block. No task bootstrap, repeat pairing or extra contrast was added.

All eight physical workflows remain in the record: slot 1 has one failed original and one successful replacement; slot 4 has two failed attempts and one successful final replacement; slots 2, 3 and 5 each succeed on their original attempt. Unscored infrastructure outcomes remain absent scores; native-scored agent failures retain their verifier reward, including zero. Valid complete scores were never rerun. Five actual native result records for the admitted logical runs and all 615 scored sample rows were checked locally against the reports and reward arrays, including grading flags and native error semantics. Model/evaluator configurations match the frozen profile except model name/location. Full typed and JSON sample-reader projections agree; they are not original Parquet bytes. Restricted source records retain capability URLs that are omitted here.

One trained checkpoint per condition and five stochastic evaluation clusters limit inference. The near-zero A11–A3 point difference does not establish equivalence or a general absence of additional-training effects. Cumulative doses, packing and replay exposure complicate comparisons with A9; native identity and paired maths remain separate evidence. No release acceptance follows from this result.

[Final analysis](A11-run-bootstrap-extension-marten-r1.json) (`b42e9fd2fec0fbc896a82c2fae89623b2951e5dbf7bdfdd31c2a51d5b4ddba71`), [actual five-run native readback](../A11-BFCL-five-r1/final-five-FULL123-native-record-and-full-samples-readback-margay-r1.json), [registration](A11-run-comparison-preregistration-marten-r1.json) and [exhaustion amendment](BFCL-infra-exhaustion-analysis-addendum-marten-r1.json) retain the evidence. The original slot-1 transcript diagnostic remains unknown because that original attempt produced no sessions; its successful replacement does not substitute for it.
