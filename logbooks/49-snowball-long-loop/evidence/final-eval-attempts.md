# Final Micro210 evaluation — attempt/benchmark-level ledger (exp1 and exp2)

- **What it is:** final-eval coverage ledger: attempts, scored, unscored (UID + reason), mean rewards where available, the verifier-only recovery rows, and exactly what remains private.
- **Compiled at:** 2026-11-15T15:14Z
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md`; [R] `.../artifacts/experiment2/completion19-receipt.json`, `recovery13-finaleval-complete-audit-20261115T0834Z.json`, `recovery13-finaleval-*-20261115*.json`; [R] `/Users/benfeuer/Documents/experiments/complete/long-loop-prototype/STATUS.md` lines 1040–1082; [P] HF model pages and recipe ledger referenced below.
- **Access class:** aggregate counts, UIDs, exception types, hashes and URIs only. Attempt-level score rows, task text and model outputs are data and remain in the private archives.

## Experiment 2 — final evaluation of selected native step 20 (sealed 2026-11-15 10:49 UTC)

Benchmark-level ledger [R: `completion19-receipt.json` `evaluation` block]:

| Quantity | Value |
|---|---|
| Unique tasks | 200 (from pinned Micro210 source; frozen selection excludes Tau3 — reduced Micro210 coverage must be disclosed) |
| Attempts | 440 expanded attempts across 20 supported paths |
| Paths | 20 |
| Scored (supplemental view) | **437** |
| Infra-unscored | **3** — TB2 (`data_source: tb2-recovery`) UIDs **225, 268, 269**, exception `VerifierTimeoutError`; exact original workspaces and mutable dependency state could not be certified |
| Native result | 436 scored + 4 VerifierTimeoutError rows (native aggregate immutable) |
| Verifier-only recovery | BixBench **UID 1** (`bix-51-q6` / trial `0db0bb8ff86f45bcbc96d0e15c9ed1ff`, session `eval-None-0--1_0`): frozen `llm_eval` on the original retained answer completed in **2.97 s, reward 0.0**; no resampling, no native-row mutation |
| Mean reward | Not present in the local benchmark-level receipts read; per-benchmark means live in the report/aggregate inside the private archive |
| EvalDash status | `infra_failed`; excludes the three unscored attempts from means |

Progress milestones (bounded, read-only samples) [R: STATE.md L35–45]: 352/440 at 06:29:34, 377/440 at 06:57:42, 389/440 at 07:11:42, 401/440 at 07:19:42, 402/440 at 07:27:42 (2026-11-15). Partial retention audit at 07:26 independently read **402 ZIP archives / 402 unique records / 7,982,988 bytes**, all inner manifests/records verified, 401 with verified native scores; the only unscored record was the known BixBench UID1 verifier timeout.

What remains private: per-attempt reward rows, per-benchmark scored/mean breakdowns and the three TB2 unscored attempt records are inside the sealed archive `s3://marin-us-east-08a/.../round2/archive/2026.11.15.19` — **2703 objects / 418,227,235 bytes, manifest SHA `bec5323a9e7efd19ff4f251759092a1f0146a70866b41ae000922f297277aaff`**, plus the separately deep-verified 10-object SFT supplement (manifest SHA `2469051c8673d8a9e7cef3e85c76550c017b51f0e335711df928f7148279c7b0`). The published model is public at [P] https://huggingface.co/open-athena/Snowball-LongLoop-Round2-2026.11.14.13, revision `6546ba2d5a1d44992d6376d4c51c1a69f16b1d10` (49 files / 39 weight shards, all manifest-matched).

## Experiment 1 — final Micro210 evaluation of best16 (sealed 2026-11-13 06:40 UTC)

Original evaluation [R: STATUS.md L1040]:

| Quantity | Value |
|---|---|
| Tasks / attempts | 200 tasks / 440 exact selected attempts |
| Scored | 432 valid scored results |
| Unscored | **8** — RuntimeServiceError rows: UIDs **47, 48** (ARC); **182, 183, 184** (ACL); **399, 400, 401** (DNA); task/environment hashes and repetitions unchanged |
| Scored-attempt mean reward | 0.4529531 |
| Benchmark mean (incomplete macro-available) | 0.5186441 |

Recovery (post5, 2026-11-13 02:38 UTC) [R: STATUS.md L1040–1046]:

- Recovery selection frozen and readback-verified at `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-micro-eval9-relay-recovery/manifest.json`, SHA `8e4a16e24a59a14c54b0dc13048dc8d4237e18919a0dc4723f09fdce6dfbb47e`; 8-row CPU recovery prep passed.
- Post5 recovery eval child SUCCEEDED; 7 recoveries graded; strict result combination FAILED for original UID 399 / recovery UID 5 (DNA): `AddTestsDirError` / no verifier at the 1,800 s deadline (`SANDBOX_NOT_RUNNING`, fail-closed cleanup then verifier upload failed); no workspace remains, so no verifier-only recovery was possible.
- **Combined ledger: 439 scored / 1 unscored** (all 440 identities; 432 original scored rows preserved); scored-attempt mean **0.45256431274521797**; unscored row = **UID 399** (DNA), `AddTestsDirError`, with the explicit finalization limitation approved by the coordinator (no further GPU retry).
- EvalDash: `infra_failed` at **439/440**.
- Durable diagnosis manifest: `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261113-micro-recovery-post5/manifest.json`, SHA `4f648a481326c13d2f348b0c726f0d2738734e3a48c5c04ecdd4b1d2934e2dd0`.

What remains private: attempt-level reward rows and per-benchmark tables for exp1 are bound inside the sealed round-1 archive — seal SHA `5a67530dacefa5af0523a2010671c86097e3d7f666582616b339adeaa98bd585`, **2926 objects / 1,351,769,694 bytes**, inventory sizes matched [R: STATUS.md L1082]. The round-1 model/recipe summary is public via the recipe ledger entry [P] (recipe tar SHA `16d487dd31e9af86033aaf55fefe3e994f2631beba03cb4b7836f88972475612`, GitHub #11337, "Micro439/440 scores").

## Unavailable (explicit gaps)

- **Per-benchmark scored/mean table for either experiment:** not extractable from the local files read; the benchmark-level report/aggregate JSONs live only in the sealed private S3 archives (paths above). Both STATUS.md and the exp2 completion receipt publish only overall scored/unscored counts and means.
- **Attempt-level rows (reward per UID):** withheld by rule (task-level outcome data); the UID lists above are coverage metadata only.
- **Exp2 per-benchmark coverage split** (which of the 20 paths contributed which attempts): requires the private archive report; the selection fact that 60 selected agentic task identities run on native Harbor and Tau3 is excluded is published [R: STATE.md L67].
- **The exact per-attempt reason codes for the 4 native VerifierTimeoutError rows** beyond the two identified (Bix UID1 recovered; TB2 225/268/269 infra-unscored): the remaining native timeout rows are subsumed by the recovery; any residual row identities live in the private native result.
