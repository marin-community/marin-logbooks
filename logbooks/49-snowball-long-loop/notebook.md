---
title: "Snowball long-loop campaign (exp1 and exp2)"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/49"
related_issues:
  - "https://github.com/marin-community/marin-logbooks/issues/48"
research_branch: "codex/49-snowball-long-loop"
marin_base_revision: "6e09d9043eb71d7bd487b5e3d79e275b95c454fd"
archive_template_revision: "3744203cdaa19ab4b4943e9836e143e7a4c339c4"
status: active
---

# Snowball long-loop campaign (exp1 and exp2)

## Question and decision criteria

- Goal and why it matters: the Snowball long-loop campaign builds a repeatable **SFT → asynchronous RLOO-N RL → frozen Micro210 eval** loop on RNO2A within ≤256 H100s total (~50B SFT positions, 102,400-token RL context in exp1, 1,800-s agent deadline, end-to-end target <36 h) as the engineering testbed for a future 535B MoE loop ("Mochi"). This record keeps a chronological, evidence-backed account of exp1 (long-loop prototype) and exp2 (new SFT mix + longer loop), plus related context (SFT new-mix data work, Snowball IsoFLOP ablations).
- Baseline and alternatives: record-keeping task; no decision between arms.
- Evidence needed to choose: n/a (record). Every claim carries primary evidence (job IDs, seal/archive receipt SHAs, HF revision SHAs, git tags, W&B references); anything unverified is labeled as a hypothesis.
- Constraints and unresolved questions: public record — no secrets, credentials, private user info, or private dataset contents. Production Atlas publication hold (v76) belongs to the Atlas record; marin#11120 remains held and unchanged.

## Baseline

- Date: 2026-11-15 13:55 UTC (08:55 EST)
- Published source revision: exp2 seal tag `experiments/exp2-rl20-2026.11.15` at commit `4913edc488…` (pushed and verified); marin_base_revision of this archive `6e09d9043eb71d7bd487b5e3d79e275b95c454fd`.
- Configuration, command, and external inputs: exp1 records in /experiments/complete/long-loop-prototype/{DESIGN,STATUS,POLICY}.md; exp2 records in /experiments/active/sft-new-mix-candidate/STATE.md (authoritative) and artifacts/experiment2/; tracker drafting in /experiments/active/exp2-tracker-entry/.
- Environment and hardware when relevant: RNO2A cluster; exp1/exp2 H100 budgets inline below.
- Result and evidence: see entries.
- Limitations: catch-up compiled from experiment directories, the claim-audit report and git tags; W&B series were not re-extracted.

## Entries

### 2026-11-15 13:55 UTC - Campaign catch-up record (2026-11-08 through 2026-11-15)

- Question or hypothesis and reason for trying it: build the chronological catch-up record of the long-loop campaign so far, per the research-secretary brief.
- Published source revision: exp2 seal tag `experiments/exp2-rl20-2026.11.15` @ `4913edc488…`.
- Command or material configuration: sources read: /experiments/complete/long-loop-prototype/*, /experiments/active/sft-new-mix-candidate/{STATE.md,POLICY.md}, /experiments/active/claim-audits/snowball-d-kills.report.md, /experiments/active/snowball-isoflop-ablations/{PLAN,HANDOFF}.md, `git tag -l` on the experiment repo.
- External inputs and artifact references: inline below.
- Environment and hardware when relevant: inline below.
- Result, including failure or restart: chronology below.

**Exp1 — long-loop prototype (2026-11-08 → 2026-11-13)**

- 2026-11-08: design revised; 256-GPU standing authorization (DESIGN.md). Exp1 = 746-update SFT (50.063B positions) → RL rounds → best-checkpoint Micro210 eval (200 tasks / 440 attempts).
- 2026-11-09: full SFT `/benfeuer/long-loop-full-sft-2026.11.09.11` submitted (pin `d7795c77097a…`, 746 steps / 50.063B positions); **failed at the first checkpoint gate** — `checkpoint_samples` device-mesh ValueError (experiment-code bug). Archive smoke sealed (manifest `d7f805e3721b…`); public HF `open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1` @ `638e48507118…` verified.
- 2026-11-10 → 11-12: RL admission fixes, snapshots rebuilt; RL continuations `/benfeuer/long-loop-full-2026.11.11.4` (source `3943467007…`, 80 learner + 176 rollout H100); step-16 validation 120/120, avg verifier 0.29175.
- 2026-11-12: best16 policy-only export; Micro210 eval on 128 H100 + 8 judge (job `…micro-full-2026.11.12.9-1f6c45826bfb`).
- **2026-11-13 06:40 UTC: Exp1 SEALED.** Archive seal SHA `5a67530dacef…`, 2,926 objects / 1.35 GB; end-to-end **80.57 h vs the <36 h target** (overrun 44.57 h); RL FAILED at round 17, best16 adopted; EvalDash **infra_failed 439/440**.
- 2026-11-13 11:38 UTC: round-1 entry published in Marin recipe tracker [marin#11337](https://github.com/marin-community/marin/issues/11337) (recipe tar SHA `16d487dd31e9af86…`); 173,721 objects / 6.44 TB checkpoint cleanup verified; marin#11120 merged at `2e11e34c7d…`.

**Exp2 — new SFT mix + longer loop (2026-11-13 → 2026-11-15)**

- Design: SFT mix + longer loop — 204 updates / 13,690,208,256 positions, batch 256, CP32/EP8, 262,144-token context.
- 2026-11-13 16:19–19:07 UTC: SFT smokes `.2/.3/.4/.6/.8` (256 H100, CP32/EP8, 262K ctx, batch 8); full-run attempt `.9` failed pre-allocation (absolute `--spec` path absent on the remote worker, 0 H100h); `.11` `/benfeuer/long-loop-exp2-full-2026.11.13.11` launched and RUNNING (source `442bf9d878…`).
- 2026-11-14 08:14 UTC: SFT handoff completed at native step 203 (handoff receipt SHA `855aaeb7d96a…`); RL child auto-submitted but Kueue-gated (248 GPUs short).
- 2026-11-14 15:48 UTC: recovery `.12` RL child (runtime `87ee29f84f…`) — **deliberately cancelled 17:06:33 UTC** for ignoring canonical Harbor harness routing; 167.31 H100-hours spent.
- 2026-11-14 17:15 UTC: recovery `.13` `/benfeuer/long-loop-exp2-recovery-2026.11.14.13` launched (runtime `038ad7f188…`, branch `benfeuer/exp2-batch256`; W&B run `o40fk9b6`); first optimizer update 18:16:58 UTC.
- 2026-11-14 evening: checkpoints 10/16/18 committed (retention "Rolling 3 + best", a user decision); Daytona snapshot quota cleared (40/45); eval steps 2 and 4 scored 100/120 pre-capacity and left unrecovered per the no-relabel policy; standing quota-purge rule authorized.
- 2026-11-15 04:24 UTC: checkpoint 20 committed (552.18-s save). 04:59 UTC eval20: 120/120, raw mean 0.26758; **selected best = native step 20** (trailing-5 EMA 0.25262); W&B FINISHED at 20; budget callback stopped at 38,575 s vs the 36,000-s budget.
- 05:33 UTC: step-20 HF export verified — 39 shards / 134,157,778,944 tensor bytes, manifest `3b64003b0a63…`.
- 05:59:42 UTC: final Micro210 eval on checkpoint 20 (16×8=128 H100 + MiniMax judge, 65,536 ctx, Harbor/Daytona declarative provisioning); 401/440 by 07:19 UTC; BixBench UID1 VerifierTimeoutError recovered verifier-only with actual reward 0.0 (no resampling) → **437/440 scored by 08:35 UTC**; TB2 UIDs 225/268/269 infra-unscored.
- 08:52 → 09:44 UTC: CPU accounting too slow (~14–19 rows/s); fixes `c768cf0b…` then `.15.19` (source `4913edc488…`); final accounting: **13,690,208,256 positions, 12,849,743,650 real tokens, 6.139% padding**, manifest `58713a89dd09…`.
- 10:08 UTC: HF model **public and verified: `open-athena/Snowball-LongLoop-Round2-2026.11.14.13`** @ `6546ba2d5a1d44992d6376d4c51c1a69f16b1d10` (49 files / 39 shards); SFT provenance supplement `.15.20` (W&B run `sft-2026.11.13.11-289c5e517585` in the marin-long-loop project, 203 steps).
- **10:49 UTC (05:49 EST): Exp2 SEALED — "completed with limitations."** Archive 2,703 objects / 418,227,235 bytes, manifest `bec5323a9e7e…`; completion receipt `s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261114-exp2-recovery13/postprocessing/2026.11.15.19/completion.json` SHA `839a0308fca1…`; seal tag `experiments/exp2-rl20-2026.11.15` pushed and verified at `4913edc488…`. (The coordinator's brief cites 05:58 EST; STATE.md's 10:49 UTC timestamp is the primary-evidence figure.)

**IsoFLOP D-run kill audit — "preemption waves" refuted (2026-11-15 07:25–12:30 UTC)**

- Report: /experiments/active/claim-audits/snowball-d-kills.report.md; cluster cw-rno2a; jobs `/benfeuer/snowball-isoflop-D-r{2..5}-v10`.
- **REFUTED:** every audited task has attempt 0 and **zero PREEMPTED attempts**. Verified causes: D-r2 and D-r3 roots **self-cancelled** after a child watchdog exit-124 ("No progress after process_started", 1,800 s; all 8 ranks exit 124) via the experiment's own `budget.py:107` handler; **D-r4 was cancelled by the experiment's separate phase-budget guard** at its 96-H100-hour ceiling (reconstructed usage 95.80 of 96); D-r5 took the same self-cancel path. No OOM recorded in failed children (0 OOMKilled / RESOURCE_EXHAUSTED lines across 19,280-row log queries). Real Kueue preemptions in the window belonged to other users' jobs, not this tree.
- Misleading counters traced to: `attempt_counts.py:40` counting TASK_STATE_KILLED as preemption, and `jobs.py:1108` using the generic "Terminated by user" label. Plus a real experiment defect: `run_probes.py:56` misses watchdog exit-124 codes in aggregate GetJobStatus. Underlying training-stall hang under investigation in marin issue #11440.

**Related context**

- **sft-new-mix-candidate** (exp2's data arm): GLM two-stage cleaning/repair of the 216-source cold-start catalog, new demonstrations, long-trace preservation to 262,144 tokens; authorized "mild premise repairs"; tracks marin#10788/#10799; mixture quantities/weights not yet decided (POLICY.md; STATE.md header).
- **snowball-isoflop-ablations**: small-scale IsoFLOP/ablation cells on a separate 64-GPU interactive allocation; as of PLAN.md (2026-11-13) valid 20-update cells: d768 full, d1024 full/assistant; blocked: d512 full/assistant, d768 assistant; ~131.6 H100-hours used; explicitly forbidden from touching exp2 (HANDOFF.md).

**Current state**

- Exp2 sealed 2026-11-15 10:49 UTC: RL finished 20 updates (final policy loss −0.0047245670), best = step 20, final eval **437/440 scored** (3 TB2 UIDs infra-unscored, no imputation), model public on HF, seal tag `experiments/exp2-rl20-2026.11.15` @ `4913edc488…`; no training/eval jobs active; observer exited.
- Exp1 archived and entered in marin#11337 as round 1 (Micro 439/440 — infra-failed).

**Open items**

1. Round-2 tracker entry for marin#11337 is being drafted by the worker in /experiments/active/exp2-tracker-entry/ (must compute round-2 Micro headline metrics and check for a recipe bundle — recipe-bundle existence unverified).
2. Checkpoint cleanup blocked: archive proof completeness false (provider refuses complete W&B history; native SFT phase timing uncertified).
3. marin#11120 held and unchanged.
4. IsoFLOP ablations: blocked cells (d512 full/assistant, d768 assistant) pending; training-stall hang open as marin#11440; budget guard active (r5 ceiling raised to 143.80 in guard config).
5. SFT-mix data work: cleaning/repair pipeline ongoing; mixture decisions outstanding.
- Evidence: per-item references above; compiled by the research secretary from primary files, 2026-11-15 13:55 UTC (08:55 EST).
- Interpretation and limitations: numbers are as recorded in the cited primary artifacts (STATE.md, audit reports, receipts); the catch-up spans 2026-11-08 onward.
- Decision and next action: continue hourly catch-up entries on this branch.
