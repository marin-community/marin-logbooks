# A12 BFCL

A12 misses the owner’s point-estimate BFCL loss limit of 5 pp against fresh v0.1. Its five full runs score 16, 20, 19, 14 and 22 of 123, for a mean 14.797% and sample standard deviation 2.597 percentage points. The run-bootstrap mean interval is [12.683%, 16.748%].

| Registered comparison | Difference, pp | Unadjusted 95% interval, pp |
|---|---:|---:|
| A12 − fresh v0.1 | −12.846 | [−16.423,−9.919] |
| A12 − A8a | −11.382 | [−14.146,−8.455] |

The fresh-v0.1 comparison is primary; A8a is secondary. Both intervals exclude zero below, and both lie below −5 pp. No lower-bound noninferiority criterion is added. These are five evaluations of one checkpoint. They do not estimate variation across independent training runs or establish why capability fell.

A12 trained fresh from v0.1 using the exact first 18 data updates of A8a, then recovered by exporting its existing native checkpoint without retraining. Shortening the horizon changes cumulative identity/replay exposure. These comparisons do not isolate a reasoning or identity mechanism. The new evaluations used production priority; historical comparators used interactive priority. Scheduling was not randomized, and no score effect is attributed to priority or latency.

The five logical slots required nine physical workflows. Four original workflows had seven infrastructure-unscored items in total; each was retained with NULL outcomes and replaced once as authorized. No original partial score enters the mean. Original slot4’s AgentTimeoutError on bfcl-irrelevance-15 is a native-scored reward 0, retained in its 14/123 result. All 18 owned jobs/attempts are closed with zero missing resource windows. The original slot1 transcript diagnostic remains bound to the original workflow, not its replacement.

Independent local qualification checks untouched native records from all nine physical workflows and all 1,107 sample rows in matching full JSON/23-column typed-reader projections. All five admitted runs have 123 fixed unique task IDs and 615 scored binary rewards in total. Launch/model/HF/readiness/source bindings and full native evaluator/model configurations match the frozen profile except model name/location. Reader projections are not original Parquet bytes or an independent verifier recomputation; restricted capability URLs were not copied into this report. Global child-status mirrors do not attest original peer execution policy.

The registered bootstrap exactly replays the frozen earlier analyses through A11, then draws 10,000 A12 run-cluster means from lexically ordered actual admitted run IDs using Python Random with seed 297. Each A12 draw is shared across the two contrasts against their frozen comparator draw. Intervals use sorted endpoints 249/9749; there is no repeat-number pairing or item bootstrap. A12 starts at state hash 0b389af4 and leaves state hash cde9ce3c reserved for A13. No sibling outcome, new comparison or missing-as-zero repair is used.

Evidence: [final analysis](A12-run-bootstrap-extension-marten-r1.json), [pre-registration](A12-A13-science-and-analysis-preregistration-marten-r1.json), [nine-physical/five-logical index](../A12-BFCL-production-five-r1/authorized-nine-physical-and-admitted-five-logical-BFCL-index-margay-r1.json). Earlier BFCL/maths/SWE files remain unchanged. This result authorizes no new retry, model run, cancellation or release.
