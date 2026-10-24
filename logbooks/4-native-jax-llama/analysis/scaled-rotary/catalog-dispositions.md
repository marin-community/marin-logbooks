# Scaled-RoPE lint-catalog dispositions

The catalog completed **six review lanes plus the composer**, all exit0, without timeout (333.36seconds). It emitted21 findings across19 files. Every finding’s matching `infra/lint` rule was read, and every complete cited file is **byte-for-byte identical** between feature `598e1968819d3a2a85778841e4a3695ab6d89d8b` and actual PRbase `7b1aab6e2e6837cac10595dd54d1b4d60b7d6361` (Git blob plus SHA256 and length). Current working bytes also match those blobs.

The catalog used main `8428b67fba80a1f8bf9dfeabc552f411cbb82be0` as its merge base, reviewing2472files. The actual stacked PR changes8Levanterfiles. All21 findings concern inherited evaluation/JAXPP/Iris/Taskforge code outside that diff. Each is answered as inherited scope; none is presented as globally fixed or automatically harmless. Main presence/identity is separately recorded per file, including missing/different main versions.

Honest label evidence: catalog completed; **21 inherited findings answered, zero findings introduced in the8-file PR**. Do not describe this as a zero-finding catalog. Saved model-refresh/telemetry warnings did not change the recorded exit0/no-timeout completion statuses.

|#|Rule|Location|Disposition|
|---:|---|---|---|
|1|`ml-naive-path-join`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/olympiadbench_judge/regrade.py`:194|Identical complete blob to PRbase; inherited, outside8-file PR|
|2|`ml-naive-path-join`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/shared_grader_replay/regrade_deterministic.py`:145|Identical complete blob to PRbase; inherited, outside8-file PR|
|3|`ml-naive-path-join`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/critical_difference/plot.py`:82|Identical complete blob to PRbase; inherited, outside8-file PR|
|4|`ml-naive-path-join`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/estimated_swebench/estimate.py`:43|Identical complete blob to PRbase; inherited, outside8-file PR|
|5|`ml-naive-path-join`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/harbor_verifier_audit/review_terminal_bench.py`:217|Identical complete blob to PRbase; inherited, outside8-file PR|
|6|`ml-tuple-return-shape`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/snowball_tournament/pairwise.py`:41|Identical complete blob to PRbase; inherited, outside8-file PR|
|7|`ml-bare-any`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/snowball_tournament/pairwise.py`:41|Identical complete blob to PRbase; inherited, outside8-file PR|
|8|`ml-tuple-return-shape`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/snowball_tournament/pairwise.py`:60|Identical complete blob to PRbase; inherited, outside8-file PR|
|9|`ml-tuple-return-shape`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/structure_closure_tournament/refresh.py`:34|Identical complete blob to PRbase; inherited, outside8-file PR|
|10|`ml-stale-inline-comment`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/tau3/laion-snowball-67b-a2b-relay-sft-acont-step999.yaml`:4|Identical complete blob to PRbase; inherited, outside8-file PR|
|11|`ml-stale-inline-comment`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/tau3/laion-snowball-67b-a2b-relay-sft-allkimi-step1203.yaml`:4|Identical complete blob to PRbase; inherited, outside8-file PR|
|12|`ml-stale-inline-comment`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/tau3/open-athena-Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02.yaml`:3|Identical complete blob to PRbase; inherited, outside8-file PR|
|13|`ml-stale-inline-comment`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/tau3/open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom.yaml`:3|Identical complete blob to PRbase; inherited, outside8-file PR|
|14|`ml-stale-inline-comment`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/tau3/open-athena-Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92.yaml`:5|Identical complete blob to PRbase; inherited, outside8-file PR|
|15|`ml-slop-test`|`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/test_structure_closure_tournament.py`:23|Identical complete blob to PRbase; inherited, outside8-file PR|
|16|`ml-naive-path-join`|`experiments/evaluation/prewarm_uploaded_model.py`:36|Identical complete blob to PRbase; inherited, outside8-file PR|
|17|`ml-module-globals`|`experiments/grug/moe_hero_pipeline/runtime/jaxpp_host_startup.patch`:230|Identical complete blob to PRbase; inherited, outside8-file PR|
|18|`ml-stale-inline-comment`|`lib/iris/config/examples/coreweave.yaml`:14|Identical complete blob to PRbase; inherited, outside8-file PR|
|19|`ml-raw-dict-vs-dataclass`|`lib/taskforge/src/taskforge/agentic/tools.py`:469|Identical complete blob to PRbase; inherited, outside8-file PR|
|20|`ml-tuple-return-shape`|`lib/taskforge/tests/llm/test_agent_live.py`:256|Identical complete blob to PRbase; inherited, outside8-file PR|
|21|`ml-tuple-return-shape`|`lib/taskforge/tests/llm/test_client.py`:385|Identical complete blob to PRbase; inherited, outside8-file PR|

For the Tau3 comments, server context capacity and a request suite cap can be distinct; the audit records the reported apparent inconsistency without changing evaluation policy. Other findings remain inherited advisory design/maintenance points. Full source excerpts, rule text, per-pin blob/SHA256 identities and lane output hashes are retained in `/private/tmp/native-jax-scaled-rotary-catalog-dispositions.json`. Catalog completion metadata: `/tmp/marin-linter/codex-native-jax-scaled-rotary/20261024T003512-2u6attih/summary.md`.

No source edits, catalog rerun, push or remote job/PR actions were performed. The feature worktree remains clean.
