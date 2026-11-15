# Build provenance — measured builds of the Snowball long-loop experiments

- **What it is:** what is known about the measured exp2 builds (base commits, recorded dirty tree hashes, branch, seal commit) and whether the dirty-tree contents are recoverable from local sources.
- **Compiled at:** 2026-11-15T15:14Z; round-7 cold-read corrections applied 2026-11-15.
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md`; [R] `.../artifacts/experiment2/exp1-vs-exp2-audit/full11-audit-wandb-history.json` (W&B run config record); [R] `.../artifacts/experiment2/recovery13-finalization-checkpoint-status-20261115T0{737,805}Z.json` and `recovery13-export20-terminal-receipts-20261115T0538Z.json` (MARIN_PROVENANCE records); [R] `/Users/benfeuer/Documents/marin-worktrees/exp2-batch256` (local git repo — `git rev-parse`, `git cat-file`, `git ls-tree -r` diffs run read-only for this file); [P] marin#11337 round-2 recipe entry (public base-commit links), HF model and seal tag references.
- **Access class:** commit/tree hashes, branch names, flags and produced-artifact identities only. No source file contents are reproduced; runtime-input file contents (local-only config/spec bundles) are referenced by path, not reproduced.

## Build 1 — exp2 SFT (full11)

| Field | Value | Source |
|---|---|---|
| Base commit | `442bf9d8780786e49fee8408ad4b84fcecb8191e` (public: [marin-community/marin commit 442bf9d878](https://github.com/marin-community/marin/commit/442bf9d8780786e49fee8408ad4b84fcecb8191e), linked from the round-2 recipe entry [P]) | [R: STATE.md L1569/1572] |
| Branch | `benfeuer/exp2-batch256` (pushed; no PR/main change; PR #11120 on hold) | [R: STATE.md] |
| W&B-recorded git_commit | `442bf9d878` | [R: full11-audit-wandb-history.json] |
| W&B-recorded git_tree_hash | `00a8f56ce2` (with `git_dirty: true` in the run config) — a **git tree SHA**, full value `00a8f56ce2bc11e555a68b0f40911970785ba03e` | [R: full11-audit-wandb-history.json L~31994] |
| What it produced | SFT run `/benfeuer/long-loop-exp2-full-2026.11.13.11`: 204/204 updates, native step203, verified HF handoff; final train loss 0.690054 / validation 0.676990; 2,994.3104 H100-h | [R: STATE.md L1793] |

## Build 2 — exp2 RL / sampling (recovery `.14.13`)

| Field | Value | Source |
|---|---|---|
| Base commit | `038ad7f188a889dfed07cac9a29fe34eabd7e5cd` (live RL runtime; public: [marin-community/marin commit 038ad7f188](https://github.com/marin-community/marin/commit/038ad7f188a889dfed07cac9a29fe34eabd7e5cd), linked from the round-2 recipe entry [P]) | [R: STATE.md L13, eval18 milestone `runtime` field] |
| Branch | `benfeuer/exp2-batch256` | [R: STATE.md L1973] |
| MARIN_PROVENANCE tree_hash | `266e04d2a1`, `dirty: true`, built_by `benfeuer` — a **git tree SHA**, full value `266e04d2a1378d7f870deb12563c3ce51c7e4084` | [R: `recovery13-finalization-checkpoint-status-*.json`; `recovery13-export20-terminal-receipts-20261115T0538Z.json`] |
| What it produced | RL run `/benfeuer/long-loop-exp2-recovery-2026.11.14.13`: 20 optimizer updates, selected native best 20, checkpoints 16/18/20 + protected best, final eval, step20 HF export (manifest SHA `3b64003b0a6374d3e180f4163c8d7998905eb917ffdd5031791e213d6a8083e1`; model identity `sha256:07f0cee1f1226da67786366d611e6cb8e6310d463acd2226d6c32ae5d03bf32e`) | [R: STATE.md L57] |

STATE.md notes: "Sampling remains tied to `038ad7f188` and recorded build tree `266e04d2a1`; its native provenance retains its original dirty flag and frozen launch evidence." [R: STATE.md L13]

## Build 3 — exp2 postprocessing / seal

| Field | Value | Source |
|---|---|---|
| Seal commit | `4913edc4884291fc54d7023eda66edc38b9a9e2e` (committed/pushed on `benfeuer/exp2-batch256`) | [R: STATE.md L13, L25] |
| What it produced | postprocessing artifact `2026.11.15.19`: recovered view/report, SFT consumption accounting, CPU archive/HF publication, completion receipt, seal tag `experiments/exp2-rl20-2026.11.15` (remotely verified) | [R: STATE.md L7–13; completion19-receipt.json] |

Preceding postprocessing commits on the same branch (also local/pushed): `c768cf0b30` (batched accounting remapping), `1dc806fc1b`, `94fc271031`, `eece9a6bde`, `80d9f12aac`, `3e032af2a6` [R: STATE.md + worktree `git log`].

## Are the dirty-tree patches recoverable? — verified answer, publicly cross-checkable

Checked in `/Users/benfeuer/Documents/marin-worktrees/exp2-batch256` (read-only git inspection; no guessing). The narrowly established result: **the recorded dirty-tree hashes equal the base commits' own git trees**. Because both base commits are pushed to the public `marin-community/marin` repository (both are linked from the round-2 recipe entry in marin#11337 [P]), any reader can verify this without access to private evidence:

1. **Object type:** the two recorded hashes are **git tree SHAs** (verified locally with `git cat-file -t` → `tree`). Full values:
   - SFT build: `00a8f56ce2bc11e555a68b0f40911970785ba03e`
   - RL/sampling build: `266e04d2a1378d7f870deb12563c3ce51c7e4084`
2. **They are exactly the base commits' own trees:** `git rev-parse 442bf9d878^{tree}` → `00a8f56ce2bc11e555a68b0f40911970785ba03e`; `git rev-parse 038ad7f188^{tree}` → `266e04d2a1378d7f870deb12563c3ce51c7e4084`. A full recursive `git ls-tree -r` diff of each dirty tree against its base commit tree shows **zero differences**. A public cross-check is `git cat-file -t <hash>` and `git rev-parse <commit>^{tree}` on a clone of `marin-community/marin`.
3. **Consequence:** the tracked-code content of both measured builds is byte-identical to the pushed base commits; the recorded "dirty" flag reflects **untracked working-tree files** (e.g., runtime-inputs bundles), which do not participate in the git tree hash. There is no hidden tracked-code patch inside either tree hash.
4. **Recoverability verdict (tracked sources): RESOLVED — trivially.** Any tracked-code rebuild of either measured build is exactly `git checkout 442bf9d878…` / `git checkout 038ad7f188…` on `benfeuer/exp2-batch256` (both commits are ancestors of the current branch, which sits at seal commit `4913edc488`). Nothing about the tracked trees is unrecoverable.
5. **What this does NOT establish:** untracked runtime-input bundles (the worktree currently holds three untracked runtime-input directories — `experiments/post_training/long_loop/runtime-inputs/20261113-exp2-full11/`, `20261114-exp2-recovery12/`, `20261114-exp2-recovery13/` — the frozen local spec/plan bundles for the measured runs; present locally [R], not committed) and the frozen spec/config directories remain **private**. This tree equality is **not a complete runnable recipe**; the round-2 recipe entry itself states no self-contained, run-ready bundle exists for round 2 [P: #11337 comment 6895748469].
6. For completeness, the superseded `.14.12` build recorded `MARIN_PROVENANCE` with base_commit `87ee29f84f`, tree_hash `6f70dab0b8`, `dirty: true`, same branch [R: `recovery12-quota-20261114T160306Z.log`]. No recoverability check was run for that superseded build (out of scope; its run was cancelled and its runtime superseded).

## Hash-verification scope (what was and was not checked)

The producer/receipt wording "verified" and "manifest-matched" must be read at the following granularity, stage by stage [R: STATE.md L57, L79, L1799; `recovery13-hf20-verification-20261115T053324Z.json`; `hf19-publication-verification.json` → `completion19-receipt.json`]:

- **SFT handoff (step203 → `.13` adoption):** weight-body SHAs were **never computed**. What was checked: the actual 1,056-file inventory/handoff verification, exact payload-inventory checks (object sets, types, sizes), metadata/config/lineage pins and byte-size/metadata verification [R: STATE.md L79; L1799 "Weight payload SHAs were never computed for the SFT/HF handoff (byte-size/metadata verification only)"]. No tensor payload was rehashed.
- **Checkpoint-20 conversion/export (native DCP → HF safetensors):** the producer export manifest was validated — manifest SHA-256 `3b64003b0a6374d3e180f4163c8d7998905eb917ffdd5031791e213d6a8083e1`, model identity `sha256:07f0cee1f1226da67786366d611e6cb8e6310d463acd2226d6c32ae5d03bf32e`, no inherited stale export manifest, and selected native20 / export request20 / HF path20 identity match [R: STATE.md L57].
- **HF upload (step20 publication):** the 05:33 independent verification covered **metadata/headers/manifest hashes, not full body rehashes** — all remote shard headers, shape/dtype byte counts, contiguous offsets, tensor bounds and exact index mapping passed; all producer-manifest object sizes matched; **six non-weight file SHAs were independently read and matched**; "remote weight bodies were not independently rehashed" [R: STATE.md L57]. The 10:08 completion-stage hub check reports all 49 files / 39 weight shards passing "Hub SHA/size checks" with every producer export-manifest SHA matching the streamed/uploaded file — that compares **Hub-reported SHAs/sizes against the manifest**; the verifier still performed no independent full rehash of the weight-tensor payloads. Earlier "verified"/"manifest-matched" wording should be understood at exactly this scope.
