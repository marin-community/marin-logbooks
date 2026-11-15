# Final-eval coverage summary

- **What it is:** a **coverage summary** of the final Micro210 evaluations of exp1 (best16) and exp2 (best20): scored/unscored attempt counts, unscored attempt identities, benchmark-level means, and what stays private. It is **not an attempt-level score ledger** — per-attempt reward rows are not reproduced here (they are private data and are also unnecessary for coverage); per-benchmark counts/means are public via the recipe ledger (linked below).
- **Compiled at:** 2026-11-15T15:14Z; round-7 cold-read corrections applied 2026-11-15.
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md`; [R] `.../artifacts/experiment2/completion19-receipt.json`, `recovery13-finaleval-complete-audit-20261115T0834Z.json`, `recovery13-finaleval-*-20261115*.json`; [R] `/Users/benfeuer/Documents/experiments/complete/long-loop-prototype/STATUS.md` lines 1040–1082; [P] marin#11337 round-1 body and round-2 comment (per-benchmark tables), HF model pages; [M] EvalDash (Marina login required).
- **Access class:** aggregate counts, UIDs, exception types, hashes and URIs are publishable coverage metadata. Attempt-level reward rows, task text and model outputs are data and remain in the private archives.

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
| Mean scored-attempt reward | **0.4048171547** (public) |
| Mean of 20 benchmark means | **0.4828958533** (public) |
| EvalDash status | `infra_failed`; excludes the three unscored attempts from means [M: EvalDash run `long-loop-round2-2026.11.14.13` — Marina login] |

**Per-benchmark counts and means are PUBLIC** — the exact 20-benchmark table (scored/attempted and mean reward per benchmark, with the round-1 comparison) is published in the round-2 recipe entry: [marin#11337 comment 6895748469, "Micro eval policy and scores" section](https://github.com/marin-community/marin/issues/11337#issuecomment-6895748469). The overall means above and the coverage counts come from the same comment and agree with the local completion receipt's counts.

Progress milestones (bounded, read-only samples) [R: STATE.md L35–45]: 352/440 at 06:29:34, 377/440 at 06:57:42, 389/440 at 07:11:42, 401/440 at 07:19:42, 402/440 at 07:27:42 (2026-11-15). Partial retention audit at 07:26 independently read **402 ZIP archives / 402 unique records / 7,982,988 bytes**, all inner manifests/records verified, 401 with verified native scores; the only unscored record was the known BixBench UID1 verifier timeout.

**Native VerifierTimeoutError rows — exactly four, all identified:** the native result's four timeout attempts are BixBench **UID 1** (recovered via the verifier-only regrade above) and TB2 UIDs **225, 268, 269** (explicitly infra-unscored). There are **no residual unidentified timeout rows beyond these four** [R: `completion19-receipt.json` `evaluation`; STATE.md L29]. The per-attempt diagnostic fields that remain unknown for those four attempts are exactly: for the three TB2 attempts, the original workspace contents and mutable dependency state (the stated reason they cannot be scored); for BixBench UID1, the cause of the pre-request verifier stall (STATE.md L45: the exact stall is unproven; judge logs show no request in the original verifier window). Attempt-level rows (reward per UID) stay private as data. (Publication-scope note, round-14 re-review: scored/unscored status, UIDs and recovery lineage are published metadata; single aggregate recovery-policy outcomes such as the BixBench UID1 regrade reward 0.0 are likewise published metadata; per-attempt reward rows beyond that remain withheld data.)

What remains private: per-attempt reward rows and per-attempt diagnostics live inside the sealed archive `s3://marin-us-east-08a/.../round2/archive/2026.11.15.19` — **2703 objects / 418,227,235 bytes, manifest SHA `bec5323a9e7efd19ff4f251759092a1f0146a70866b41ae000922f297277aaff`**, plus the separately deep-verified 10-object SFT supplement (manifest SHA `2469051c8673d8a9e7cef3e85c76550c017b51f0e335711df928f7148279c7b0`). The published model is public at [P] https://huggingface.co/open-athena/Snowball-LongLoop-Round2-2026.11.14.13, revision `6546ba2d5a1d44992d6376d4c51c1a69f16b1d10` (49 files / 39 weight shards, manifest-matched; see build-provenance.md for the hash-verification scope).

## Experiment 1 — final Micro210 evaluation of best16 (sealed 2026-11-13 06:40 UTC)

Original evaluation [R: STATUS.md L1040; tracker round-1 body]:

| Quantity | Value |
|---|---|
| Tasks / attempts | 200 tasks / 440 exact selected attempts |
| Scored (original) | 432 valid scored results |
| Unscored (original) | **8** — RuntimeServiceError rows: UIDs **47, 48** (ARC); **182, 183, 184** (ACL); **399, 400, 401** (DNA); task/environment hashes and repetitions unchanged |
| Scored-attempt mean reward (original) | 0.4529530863 (public) |
| Mean of 20 benchmark means (original) | 0.5186440986 (public) |

Recovery (post5, 2026-11-13 02:38 UTC) [R: STATUS.md L1040–1046]:

- Recovery selection frozen and readback-verified at `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-micro-eval9-relay-recovery/manifest.json`, SHA `8e4a16e24a59a14c54b0dc13048dc8d4237e18919a0dc4723f09fdce6dfbb47e`; 8-row CPU recovery prep passed.
- Post5 recovery eval child SUCCEEDED; 7 recoveries graded; strict result combination FAILED for original UID 399 / recovery UID 5 (DNA): `AddTestsDirError` / no verifier at the 1,800 s deadline (`SANDBOX_NOT_RUNNING`, fail-closed cleanup then verifier upload failed); no workspace remains, so no verifier-only recovery was possible.
- **Combined ledger: 439 scored / 1 unscored** (all 440 identities; 432 original scored rows preserved); scored-attempt mean **0.45256431274521797**; mean of 20 benchmark means **0.5216066358**; unscored row = **UID 399** (DNA), `AddTestsDirError`, with the explicit finalization limitation approved by the coordinator (no further GPU retry).
- EvalDash: `infra_failed` at **439/440**.
- Durable diagnosis manifest: `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261113-micro-recovery-post5/manifest.json`, SHA `4f648a481326c13d2f348b0c726f0d2738734e3a48c5c04ecdd4b1d2934e2dd0`.

**Per-benchmark counts and means are PUBLIC** — the exact 20-benchmark table for round 1 (scored/attempted and mean scored reward per benchmark, post-recovery) is published in the round-1 recipe entry: [marin#11337 issue body, "Micro eval policy and scores" section](https://github.com/marin-community/marin/issues/11337#issue-6370294160) (same table mirrored in comment 6895748469's round-1 comparison columns).

What remains private: attempt-level reward rows for exp1 are bound inside the sealed round-1 archive — seal SHA `5a67530dacefa5af0523a2010671c86097e3d7f666582616b339adeaa98bd585`, **2926 objects / 1,351,769,694 bytes**, inventory sizes matched [R: STATUS.md L1082]. The round-1 model/recipe summary is public via the recipe ledger entry [P] (recipe tar SHA `16d487dd31e9af86033aaf55fefe3e994f2631beba03cb4b7836f88972475612`, GitHub #11337, "Micro439/440 scores").

## Private (explicit gaps)

- **Attempt-level reward rows (reward per UID) for either experiment:** withheld by rule (task-level outcome data); the UID lists above are coverage metadata only. All per-benchmark counts/means are public via the linked #11337 tables.
- **Per-attempt diagnostics for the identified unscored/recovered attempts:** for exp2's four native timeout attempts (Bix UID1; TB2 UIDs 225/268/269) the unknown fields are exactly those listed above (TB2 original workspaces/dependency state; BixBench pre-request stall cause); per-attempt diagnostic rows otherwise live in the private archives.
- **Exp2 per-benchmark coverage split beyond the public table** (which of the 20 paths contributed which attempts at UID granularity): requires the private archive; the published fact that 60 selected agentic task identities run on native Harbor and Tau3 is excluded is public [R: STATE.md L67].
