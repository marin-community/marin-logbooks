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

> [!NOTE]
> Read the newest entries first: the 13:55 UTC catch-up entry below is superseded **in part** by the later revision entries at the end of this file — especially **revision 4 (14:52 UTC)**, which contains the consolidated, corrected glossary and stage table, and the current synopsis in **revision 3 (14:42 UTC)**. Corrections are appended, never rewritten.

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

### 2026-11-15 14:30 UTC - Revision 1 after cold-read round 1: corrections, glossary, stage table, results, timing

- Question or hypothesis and reason for trying it: an independent cold-read review of the 13:55 entry returned NOTES (35 items). This entry appends corrections and context; the 13:55 entry is preserved unedited (corrections are new entries).
- Published source revision: exp2 seal tag `experiments/exp2-rl20-2026.11.15` @ commit https://github.com/marin-community/marin/commit/4913edc488 (post-processing revision, not the training revision — see stage table).
- Command or material configuration: review process: a context-free reviewer read only public GitHub/HF material and returned a local-only notes report [R]; no private material was read by that reviewer; no secrets found in reviewed text.
- External inputs and artifact references: marin#11337 (recipe tracker: round-1 entry + [round-2 entry posted 2026-11-15 14:12:05 UTC](https://github.com/marin-community/marin/issues/11337#issuecomment-6895748469)), marin#11120 (open, held), marin#11440, marin#10788/#10799, HF cards for the models named below, Micro210 dataset card https://huggingface.co/datasets/open-athena/Marin-Micro-Eval-210.
- Environment and hardware when relevant: RNO2A cluster; see budget scope below.
- Result, including failure or restart: corrections and supplements below.

**Evidence classes used in this record**

- [P] = publicly verifiable (marin/marin-logbooks/HF public). [M] = authenticated access required (private open-athena releases, Marin login, W&B). [R] = reported from local/private records; not publicly inspectable. The 13:55 entry's "every claim carries primary evidence" and "REFUTED" wording are qualified: conclusions of audits [R] whose public corroboration is limited; the D-run cancellation *mechanism* is supported by the audit's log evidence [R], while the *cause of the underlying training stall* remains an open investigation (marin#11440) [P].

**Corrections superseding the 13:55 entry**

1. **marin#11120 is OPEN, not merged.** The 13:55 entry's "PR11120 merged at `2e11e34c7d…`" is withdrawn: that commit (https://github.com/marin-community/marin/commit/2e11e34c7dc6661a935f5b33157b58584723a0f7, "Merge main and retain long-loop Daytona lifecycle fixes") is a branch merge retaining Daytona lifecycle fixes, cited in the exp1 sealing record — it is not evidence of PR #11120 merging. #11120 remains open and held [P].
2. **Exp1 final eval**: the EvalDash run status is `infra_failed`, but the scored coverage is **439 scored, 1 unscored (UID 399)** — "439/440" counts scored attempts, not 439 failures [P via marin#11337 round-1 entry].
3. **Exp1 RL outcome detail**: 17 optimizer updates **completed**; update-18 **admission** failed with `GroupAdmissionStalledError`; RL16 (from run `/benfeuer/long-loop-full-2026.11.12.5`, RL source `0c1e81ca93635ac157db90ff786a8f8fba65b13c` [M]) was selected as best. The failed `.11.4`/`3943467007…` was an earlier material continuation, abandoned after admission fixes [M].
4. **Exp1 SFT lineage**: the failed `.09.11` run (746 steps was its *plan*, not achieved work) was replaced by a **successful** full SFT `/benfeuer/long-loop-full-sft-2026.11.09.12` (source `9662fdba02896ffb03fd21066c1750b363e78cb6` [M], 746 updates, 50,063,212,544 packed positions); export `.10.1`. The `.11` run had used fixed mixture weights instead of the intended schedule — that correction motivated `.09.12` [P via marin#11337/#11120 records]. Note: the exp1 record's 50.063B figure in the 13:55 entry referred to the plan; the achieved figure is the same nominal 50,063,212,544 packed positions [M].
5. **Round-1 model identity**: the HF `open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1` artifact is a **plumbing smoke** (15-update SFT smoke + 3-update RL), which established handoffs/runtime coverage, not the full recipe. The round-1 final model is **open-athena/Snowball-LongLoop-Round1-2026.11.12.7** @ `475ba62d4e9498b075d52ec7168449b19a4ab683` [P] (best16 policy from the successful RL continuation). Its HF card cites archive source `9b672ba…` (export identity) while the tracker's training source is `0c1e81ca…` — export vs training identities [P/M].
6. **Update counting**: exp2 executed **204 optimizer updates, numbered 0–203**; "native step 203", "203 steps" and "204 updates" in the 13:55 entry all refer to this. W&B step labels are 0-based update indices; exp1 checkpoints similarly: native checkpoint 746 = 746 completed updates (W&B steps 0–745).
7. **`58713a89dd09…` is the SHA-256 of `realized-samples.jsonl` (70,562,321 bytes), not a manifest**; the CPU-accounting receipt hash is `379ef811…` [M]. Public counts: 52,224 realized samples; 840,464,606 padding positions; 7,751,383,148 assistant + 2,821,707,252 causal-replay loss tokens; 12,849,743,650 real tokens of 13,690,208,256 sampled positions (6.139% padding, denominator = sampled positions); reconstruction preserved a 19,264-row prefix and reconstructed 32,960 rows (validation described in the round-2 tracker). These are sampled/packed counts, not a unique-corpus token count.
8. **Exp2 "new SFT mix"**: the round-2 consumed recipe kept the **same 377-component declaration and 19-entry weight-schedule hashes as round 1** (`d372def…`, `e954bba…` [M]); the actual changes were consumed formats/packing, loss treatment, optimizer (MuonH vs round-1 Adam) and exposure. **No GLM-cleaned demonstrations were consumed in exp2** — the GLM cleaning/repair work is the *planned* data arm; mixture weights remain undecided.
9. **"Longer loop" defined**: exp2's design changes were the longer **SFT context (262,144 tokens; exp1 SFT was shorter)** and a longer training-recipe tail; RL context is 102,400 tokens in both rounds, and exp2's RL budget (10 h) was shorter than round 1's 12 h; the achieved comparison is 20 vs 17 completed RL updates. The label does not mean a longer RL context.
10. **IsoFLOP status was stale**: https://github.com/marin-community/marin/issues/11440 reports 2026-11-14 04:25 UTC completion of 20-update controls for **both d512 cells and both d768 masks** after adding ALLCOLLECTIVES to the joint workaround. Those workaround timings are **excluded from source-runtime throughput comparisons**; the "blocked" state applies to original-runtime qualification. The 13:55 entry's Nov-13 plan snapshot is retained as historical. `143.80` = cumulative H100-hours for r5 after the ceiling raise.
11. **RL budget overrun semantics**: the 36,000-s RL budget clock **starts at the first optimizer completion and stops at a completed-step boundary**; the observed 38,575 s includes the final update/save phases within the stopping semantics. The 1,800-s agent deadline and the 36-h end-to-end target are separate quantities.
12. **Seal timing**: the completion receipt timestamp is 2026-11-15 **10:49:53.982839 UTC** (per the round-2 tracker, [P]); the coordinator's earlier "05:58 EST" reference is approximate. Exp1's end-to-end window: submission 2026-11-09 21:45:42.016 UTC → coordinator completion 2026-11-13 06:19:50.069 UTC = 80.57 h (the 06:40 UTC event is the separate archive seal) [P]. Exp2 end-to-end elapsed was **not totaled**; native SFT-phase timing is unavailable and some estimates extrapolate batch-8 smoke timing.
13. **Exp2 completion states**: the original coordinator **failed its report-coverage gate**; CPU post-processing produced the replacement completion receipt (`completion.json` SHA `839a0308fca1…` [M]); the final EvalDash dashboard retains `infra_failed` status. "Sealed" certifies the copied archive bytes (2,703 objects), not complete telemetry or an unattended successful loop. Round-2 checkpoint cleanup remains blocked (archive proof completeness false: incomplete W&B provider history; unverified native SFT-phase timing; 14-day storage lifecycle with unverified expiry).
14. **No-active-jobs scope**: "no training/eval jobs active" in the 13:55 entry refers to the **exp2 job tree only**, checked at ~13:55 UTC; IsoFLOP ablation cells are governed separately (budget guard active).
15. **No-relabel policy (operational)**: original eval outputs/workspaces are never re-generated or re-labeled; only verifier-only recovery is allowed when a verifier times out (reward recorded as actually measured — BixBench UID1: 0.0); infra-unscored slots stay unscored (TB2 UIDs 225/268/269; exp1 UID 399). Resampling is not permitted. Early 100/120 validations (steps 2/4) were left unrecovered because pre-capacity infra failures are not re-run under this policy.
16. **GPU budget scope**: the ≤256 H100 ceiling covers the long-loop training/eval loop (learner + rollout, with judges via the hosted judge service); the IsoFLOP ablations run on a **separate 64-GPU interactive allocation** outside that ceiling. Round-1 audited requested cost: **12,624.162622 H100-hours** [P]; a complete round-2 allocation-hours total is unavailable [M].
17. **Build provenance**: both measured round-2 builds were **dirty**: SFT tree `00a8f56ce2…`, RL/eval tree `266e04d2a1…` [M]; their base commits alone do not reconstruct the builds, so exact public reproduction is currently unresolved; the `4913edc488…` tag is the post-processing/seal revision, not the training revision. marin_base_revision/archive_template_revision in this file's frontmatter are archive scaffolding, not training identities.

**Glossary (compact)**

- **Snowball**: the 67B-total/~2B-active MoE initializer model used for these rounds (distinct from the future 535B "Mochi" target). **SFT**: supervised fine-tuning. **RL**: asynchronous RLOO-N (REINFORCE leave-one-out with N samples) with group admission. **RNO2A / cw-rno2a**: the compute cluster / queue. **CP32/EP8**: context-parallel 32 / expert-parallel 8. **Harbor**: agent harness; **Pi**: agent CLI; **Daytona**: sandbox provider; **Kueue**: Kubernetes job queueing; **EvalDash**: evaluation dashboard. **Micro210**: frozen eval set = 200 tasks / 20 benchmarks / 440 attempt slots (Tau3 excluded; repetitions vary by benchmark; validation = disjoint 20 SWE + 100 non-agentic tasks). **TB2**: a Micro210 benchmark. **GLM**: the SFT-mix data-cleaning teacher model family. **IsoFLOP**: equal-compute scaling ablation; **D cells**: width/depth variants at batch-256 (d512/d768/d1024; "full" vs "assistant" = loss-mask scope). **EMA**: trailing-5 validation EMA (span 5, alpha 1/3, all-120-tasks-scored eligibility, two-update cadence). **MuonH**: round-2 optimizer.

**Results summary (public tracker metrics [P])**

- Round 1 (exp1): scored-attempt mean reward 0.4525643127; mean of 20 benchmark means 0.5216066358; 439/440 scored; status infra_failed. Round 2 (exp2, sealed): 0.4048171547; 0.4828958533; 437/440 scored (3 TB2 UIDs infra-unscored, no imputation); final EvalDash status infra_failed; original coordinator failed its coverage gate (replacement CPU post-processing receipt). These heterogeneous partial-credit means are **not pass@1**; the round-2 decline is **descriptive only** — no matched baseline (no SFT-only or pre-RL Micro eval), no repeated-seed uncertainty, and the recipes differ in multiple factors; per-benchmark samples are tiny (10 distinct tasks each, repeated slots are not independent).
- Checkpoint selection (exp2): eval20 raw mean 0.26758, trailing-5 EMA 0.25262 → native step 20 selected; steps 2/4 (100/120, incomplete coverage) ineligible under the all-scored rule; policy loss −0.0047245670 is an optimization diagnostic, not a quality score.

**Limitations (added)**

- No matched RL-vs-SFT-only evaluation; two-round comparisons change multiple factors; not causal.
- Data lineage: seven historical upstream HF revisions unknown; exact overlap ≠ semantic decontamination; SWE dependency drift possible on rerun. Full component weights and per-source accounting are [M].
- Dirty builds (see correction 17) mean exact reproduction is unresolved; round 2 has no self-contained recipe bundle (tracker finding [P]); round-1 recipe tar is [M].
- Retention: round-2 retains RL16/18/20, SFT203, corrected handoff, under a 14-day lifecycle with unverified expiry; public HF weights do not preserve optimizer state.
- The D-run audit conclusions rest on [R] log evidence (19,280-row queries, per-job tables in the local report); the training-stall root cause is open (marin#11440 [P]).

**Next steps (concrete conditions)**

1. Publish round-2 tracker carry-overs (recipe/build provenance, telemetry/timing gaps, three unscored slots) — blocked items listed in marin#11337 round-2 entry [P].
2. marin#11440 hang diagnosis; IsoFLOP original-runtime throughput qualification (workaround timings excluded).
3. SFT-mix data arm: GLM cleaning/repair decisions (marin#10788/#10799); whether a future round consumes cleaned data is undecided.
4. marin#11120 remains open/held pending user decision; another long-loop round is not currently authorized — undecided.
- Evidence: report-path note: the cold-read notes report is local-only [R]; this entry is the durable response.
- Decision and next action: fresh cold read on the updated branch next.

### 2026-11-15 14:33 UTC - Revision 2 after cold-read round 2: optimizer and context corrections, timing bound, per-round recovery policy, stage/difference tables

- Question or hypothesis and reason for trying it: a fresh cold-read round 2 returned NOTES (23/27 items), including two factual errors in revision 1. This entry corrects them with public sources.
- Published source revision: round-1 SFT source https://github.com/marin-community/marin/commit/9662fdba02896ffb03fd21066c1750b363e78cb6; round-2 tracker entry https://github.com/marin-community/marin/issues/11337#issuecomment-6895748469; exp2 launch report https://github.com/marin-community/marin/pull/11120#issuecomment-6863473319; exp2 deadline controls https://github.com/marin-community/marin/pull/11120#issuecomment-6858592893.
- Command or material configuration: see difference table below.
- External inputs and artifact references: Micro210 dataset card at frozen revision https://huggingface.co/datasets/open-athena/Marin-Micro-Eval-210/blob/4564ed081c316bc3231ae5c3a53a46b39248c819/README.md; exp1 eval YAML https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/experiments/post_training/long_loop/specs/evaluation.yaml (`judge_gpu_count: 8`).
- Environment and hardware when relevant: judge allocation is **local co-allocated GPUs, 8 H100s per eval** (per the pinned eval YAML), not a remote hosted service for the exp1 Micro eval; these judge GPUs count within the campaign's per-stage allocations; a complete round-2 allocation-hours total remains unmeasured [M]. The 64-GPU IsoFLOP allocation is outside the ≤256 ceiling.
- Result, including failure or restart:

**CORRECTIONS superseding revision 1 (and the 13:55 entry)**

1. **SFT optimizer direction was wrong in revision 1.** Round 1 SFT used **MuonH** (`grug_moe_muonh_v1`, pinned round-1 SFT YAML at `9662fdba…` [P]); round 2 consumed **Adam** (LR 5e-5, betas 0.9/0.95, cosine schedule) per the round-2 tracker [P]. The RL optimizer (AdamW) is separate from both. "MuonH: round-2 optimizer" in the revision-1 glossary is withdrawn.
2. **SFT context was wrong in revision 1.** Both full rounds used **262,144-token SFT context** (round-1 YAML and round-2 tracker [P]); no 262K-vs-shorter comparison exists. Smaller-context settings belonged to smokes/the 32k plumbing smoke only.
3. **"Longer loop" is a historical plan label.** Executed comparison: SFT exposure fell 746 → 204 updates; RL budget fell 12 h → 10 h; RL context 102,400 in both; achieved RL updates rose 17 → 20. No dimension of the executed round 2 was longer than round 1 except the achieved update count.
4. **Recovery policy is per-round.** Round 1: eight startup-failed slots were **restarted with the same identities/seed/settings**, yielding seven grades; 432 originally scored rows preserved; UID 399 unscored. Round 2: no generation resampling; only verifier-only regrade of BixBench UID1 (actual reward 0.0); TB2 UIDs 225/268/269 infra-unscored. Restarting a slot that never produced output ≠ rerunning a completed answer.
5. **Round-2 elapsed time vs the 36-h target (bound from public reporting).** `.13.11` was publicly reported running 2026-11-13 20:00:54 UTC [P]; replacement-completion receipt 2026-11-15 10:49:53.982839 UTC [P] → **lower bound ≈38 h 49 min**, exceeding the 36-h end-to-end target under a submission-to-replacement-completion definition. The exp2 deadline controls additionally configured non-eval task deadlines totaling 36 h, a 72-h final-eval exception, and a 109-h coordinator deadline — configured safety deadlines, not the measured end-to-end elapsed time. **Assessment: the measured bound shows the end-to-end target was not demonstrably met in round 2** (missing native SFT-phase timing prevents an exact total; this does not make the target unknowable).
6. **Timestamp discipline**: the revision-1 entry was drafted at 14:30 UTC and published at 14:24:13 UTC (commit `8fc7be25e7…`); observation/drafting/publication times were conflated. Going forward, entry headings use the publication time and observation times are stated separately. The 13:55 entry's heading is its publication label (actual commit 14:2x UTC); noted as a drafting-time label, not event time.
7. **Micro210 vs validation vs dataset card**: the dataset card describes 210 tasks / 21 benchmarks / 450 repeated slots; the campaign's frozen eval excludes **Tau3** → 200 tasks / 20 benchmarks / 440 slots. The 120-task checkpoint-selection set (20 SWE + 100 non-agentic) is a **separate training holdout**, disjoint from training data; its identity is [M]. The card's profile is a compact/simplified uniform profile (e.g., generated-choice TruthfulQA rather than original MC2) and is not a full release-policy reproduction [P].
8. **D-run cancellation audit**: relabeled as **findings reported by an unpublished (local) audit** [R]; marin#11440 concerns the underlying training stall and is related context only, not corroboration of the cancellation ledger. The per-job attempt/exit/audit tables remain [R]; code identifiers (`budget.py:107` etc.) remain without public permalinks (open work).
9. **Exp1 failure causal history**: `.09.11` terminated on the runtime checkpoint-gate `checkpoint_samples` device-mesh ValueError [R]; separately, its recipe was invalidated because it used **fixed mixture weights instead of the intended schedule** [R] — the `.09.12` replacement fixed the recipe (and ran to completion). These are distinct: one terminated the run, the other invalidated the recipe and motivated replacement.
10. **Exp2 launch history completed**: `.13.9` planned batch 8 / 6,410 updates (abandoned sizing plan); an earlier 32-H100 Adam attempt failed allocation; `.13.10` allocated workers and was cancelled for the batch change (66.3749 H100-hours); `.13.11` (batch 256 / 204 updates) reported running 2026-11-13 20:00:54 UTC [P]. The 13:55 entry's 16:19–19:07 range covered smokes only.
11. **Operational shorthands**: "40/45" = Daytona snapshot quota used/limit at that moment [R]; "~14–19 rows/s" = CPU-accounting processing rate (rows = realized samples) [R]; checkpoints 10/16/18 committed are selected milestones at a two-update validation cadence, not the full save history [R]; "pre-capacity" = scored before full eval capacity was available; "group admission" = RLOO-N admission gating over sample groups; TB2 = Terminal-Bench 2 (version as pinned in the consumed eval config [M]); Atlas v76 hold is unrelated to this campaign and is cross-referenced in logbooks #48 only.

**Stage lineage and difference table (public parts [P]; private consumed configs [M])**

| Stage | Round 1 (exp1) | Round 2 (exp2) |
|---|---|---|
| Initializer | Snowball base (pins in tracker) | same base |
| SFT updates | 746 | 204 (numbered 0–203) |
| SFT positions | 50,063,212,544 packed | 13,690,208,256 sampled |
| SFT context | 262,144 | 262,144 |
| SFT optimizer | MuonH (`grug_moe_muonh_v1`) | Adam (LR 5e-5, betas 0.9/0.95, cosine) |
| CP/EP geometry | CP8 within nodes | CP32 across nodes |
| Microbatch | 32 | 8 |
| MoE dispatch | local Sonic | EP ring |
| Optimizer numerics | BF16 / offloaded | FP32 / on-device Adam |
| Loss mask | assistant-only supervision; round-1 audited batch covered 94.97% of intended last-conversation assistant tokens (recorder repairs not applied to the scored model) | assistant-only (harness/recorder repairs present in round 2 [M]) |
| Scheduling | — | cosine |
| RL context | 102,400 | 102,400 |
| RL budget | 12 h | 10 h (38,575 s consumed) |
| Achieved RL updates | 17 completed; update-18 admission failed (`GroupAdmissionStalledError`) | 20 completed |
| Selected checkpoint | RL16 | native step 20 |
| Final model | open-athena/Snowball-LongLoop-Round1-2026.11.12.7 @ `475ba62d…` | open-athena/Snowball-LongLoop-Round2-2026.11.14.13 @ `6546ba2d…` |
| Final eval | 439/440 scored, UID 399 unscored, run status infra_failed | 437/440 scored (TB2 UIDs 225/268/269 unscored), status infra_failed |
| Scores (scored-attempt mean reward / mean of benchmark means) | 0.4525643127 / 0.5216066358 | 0.4048171547 / 0.4828958533 |

- Active expert-capacity dropping was unmeasured and dropless equivalence unestablished (exp1-vs-exp2 audit summary [P]); the two rounds differ in many factors simultaneously — no single-factor attribution is supported.
- Best-checkpoint selection: the per-checkpoint validation ledger (all candidates, eligibility, EMA sequence) is [M]/[R] and not yet published; the record cites only the selected point's values; selection maximality is **reported, not publicly re-verified**.
- Final scores are tracker-reported aggregates [P]; attempt-level rows and recovery lineage are private; public recomputation is not yet possible.
- 377 component declarations expand into 216 chat leaves + 1,711 replay leaves [P]; exact weights/per-source accounting [M]; seven unknown upstream revisions affect specific replay sources [M].
- "Mild premise repairs" (data arm): planned/ongoing work governed by marin#10788's public contract (grounded repairs, no invented execution results, drop unverifiable continuations); nothing from this arm was consumed by exp2; teacher/prompt pins and counts not yet published.
- Retention/resumability: retained artifacts (RL16/18/20, SFT203, corrected handoff) exist as of the round-2 tracker's snapshot; exact 14-day lifecycle expiry deadlines unverified; resumability **unverified**; assigned as a verification next action.
- Next-step statuses: round-2 tracker entry **published** (14:12:05 UTC) — the carry-over work is the *follow-up*, not a pending publication; three unscored eval slots are accepted permanent limitations under the no-resampling policy (a new eval version would need separate authorization); no new round authorized — undecided pending user.
- Evidence: all [P] items above link public commits/issues/cards; [M]/[R] items remain as labeled.
- Interpretation and limitations: unchanged from revision 1's limitations section, plus: the 38 h 49 min figure is a public-reporting bound, not a measured phase-total.
- Decision and next action: fresh cold read round 3 next.

### 2026-11-15 14:42 UTC - Revision 3 after cold-read round 3: authoritative synopsis, RLOO-Neutral correction, separated loss rows, complete stage table, verdict wording

- Question or hypothesis and reason for trying it: fresh cold-read round 3 returned NOTES on all targets (21/23 items), including one glossary error introduced by revision 1. This entry corrects it and consolidates the current state. Entry headings are editorial publication labels; exact commit times are in git metadata (this commit's time is authoritative for publication).
- Published source revision: advantage estimator definition https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/skyrl-train/skyrl_train/utils/advantage_estimators.py#L153; round-1 SFT YAML https://github.com/marin-community/marin/blob/9662fdba02896ffb03fd21066c1750b363e78cb6/experiments/post_training/long_loop/specs/sft.yaml; RL YAML https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/experiments/post_training/long_loop/specs/rl.yaml; round-1-vs-2 audit summary https://github.com/marin-community/marin/pull/11120#issuecomment-6864384835; round-1 cleanup exception https://github.com/marin-community/marin/issues/11337#issuecomment-6851731143.
- Command or material configuration: see corrected stage table below.
- External inputs and artifact references: as above.
- Environment and hardware when relevant: as previously recorded.

**CURRENT SYNOPSIS (authoritative as of this entry; supersedes all earlier "Current state"/"Open items" blocks, which remain as history)**

- Engineering question: can the SFT → asynchronous RLOO-N RL → frozen Micro210 eval loop complete end-to-end within <36 h on ≤256 H100s? **Verdict: no, for both rounds under the submission-to-replacement-completion endpoint.** Exp1: 80.57 h (exceeded by 44.57 h). Exp2: public-reporting lower bound ≈38 h 49 min — **exceeded by at least ≈2 h 49 min**; exact elapsed unknown (native SFT-phase timing unavailable). A successful unattended repeatable loop is not established: exp2 required manual recovery (a cancelled RL child replaced by recovery `.13`; coordinator report-gate failure replaced by CPU post-processing) and manual sealing.
- Quality question (did RL improve Micro210 scores?): **unanswered** — no matched SFT-only/pre-RL evaluation, no repeated-seed uncertainty, and the two rounds differ in many factors; the round-1→2 metric decline (0.4525643127 → 0.4048171547 scored-attempt mean reward; 0.5216066358 → 0.4828958533 mean of benchmark means) is descriptive only.
- Coverage: exp1 439/440 scored (UID 399 unscored); exp2 437/440 (TB2 UIDs 225/268/269 permanently unscored under the round-2 no-resampling policy — accepted limitations). Archive integrity: both archives sealed (byte-verified), which does not certify complete telemetry.
- Provenance: exact public reproduction unavailable — round-2 dirty builds (`00a8f56ce2…`, `266e04d2a1…`), private frozen inputs, no self-contained round-2 recipe bundle; round-1 recipe tar requires authenticated access. Public source/config commits are [P]; consumed artifacts [M]; local audits [R].
- State: marin#11120 OPEN and held; no new round authorized (undecided pending user); round-2 tracker entry published 14:12:05 UTC (marin#11337).

**Corrections and completions**

1. **RLOO-N = RLOO-Neutral**, not "N samples" (revision-1/2 glossary wrong). Per the pinned advantage-estimator code [P]: infrastructure-failed samples are excluded from the group baseline and receive zero advantage; model failures remain eligible with zero reward. Group size **N=8**. "Group admission" = admission gating over sample groups (a group whose admission stalls halts training — exp1's update-18 `GroupAdmissionStalledError`).
2. **Loss rows separated** (replacing the revision-2 conflated row): (i) SFT chat masking: assistant-only supervision in both rounds; (ii) SFT causal replay: round 2 consumed 2,821,707,252 causal-replay loss tokens (replayed text also carries loss), in addition to 7,751,383,148 assistant loss tokens; (iii) RL token coverage: the **94.97%** figure is the audited coverage of intended last-conversation assistant tokens in the **first consumed RL SWE batch** of round 1 only (recorder repairs were not applied to the scored round-1 model) — not an SFT measurement and not run-wide.
3. **Stage table completed**: CP/EP: round 1 **CP8/EP1**, round 2 **CP32/EP8**; microbatch 8 vs 32 as per-device microbatch within global batch 256; precision: round 1 BF16 parameters/compute with offloaded optimizer state; round 2 FP32 parameters with on-device Adam state; schedules: round 1 linear (trainer horizon 4,750 updates, 475-step decay) **stopped at 746**; round 2 cosine. Round 2 additionally: BF16→FP32 parameter numerics change accompanies the Adam switch.
4. **Timestamps**: git commit times: initial catch-up `437b2e41dc…` 14:07:59 UTC; revision 1 `8fc7be25e7…` 14:24:13; revision 2 `8d30fa34e5…` 14:33:07; navigation note `44f481ead3…` 14:34:36. The revision-2 "drafted at 14:30, published 14:24" wording was wrong as written (an editorial label confused with commit time); heading times hereafter are labels, and git metadata is the publication evidence.
5. **Evidence classes apply to objects, not whole claims**: round-1 SFT/RL source commits (`9662fdba…`, `0c1e81ca…`) and their YAMLs are **[P]** (public code/config); the *runs* they produced are [M]/[R]. Tracker-reported aggregates are public **reports** of private measurements — labeled "publicly reported", not publicly inspectable.
6. **Exp2 pre-`.13.11` attempts**: the earlier 32-H100 smoke failed before its first update during **Adam second-moment allocation** (optimizer-memory failure, not scheduler GPU admission); `.13.9` failed on remote-spec preallocation; `.13.10` allocated and was cancelled for the batch change (66.3749 H100-hours).
7. **Round-1 cleanup exception** (explaining "cleanup verified" alongside blocked evidence gates): a later explicit user request authorized manual deletion of **12 superseded checkpoints**; SFT746 and RL16 were retained; incomplete-evidence flags were unchanged (https://github.com/marin-community/marin/issues/11337#issuecomment-6851731143 [P]).
8. **RL objective summary** (from pinned base code/configs [P]; round 2's dirty build may deviate — delimitation [M]): 32 SWE + 32 non-agentic groups of 8; staleness 2; neutral failure masking; zero-variance filtering; clipped policy updates with TIS; enabled reasoning-span reward shaping with a configured 0.1 penalty. Training reward (shaped, filtered) is distinct from the raw verifier reward reported in validation.
9. **Checkpoint selection**: eligibility requires a **complete trailing five-observation window** (per the selection-callback code [P]); the full per-candidate ledger (run ID, step, scored counts, raw metric, EMA, eligibility, selection) is not published — the cited selected-point values are reported, and selection maximality is not publicly re-verified. Exp1's 0.29175 belongs to the successful continuation's step-16 validation, not the abandoned `.11.4`.
10. **IsoFLOP scope**: this record tracks the ablation lane only as a runtime-status dependency; no equal-compute scientific result is public yet. d512/d768/d1024 denote width/depth variants at batch 256; "full" vs "assistant" is loss-mask scope; the #11440 workaround completions (2026-11-14 04:25 UTC) qualify finite-update execution under the workaround, and exclude those timings from source-runtime throughput comparison.
11. **Data arm execution state**: the two-stage GLM cleaning is **planned/ongoing**, governed by marin#10788's public contract; no executed-cleaning counts have been published; nothing was consumed by exp2.
12. **Retention**: preserved artifacts' existence is observed as of the round-2 tracker snapshot; automatic 14-day lifecycle expiry is **unverified**, so preservation beyond that observation time is not established; resumability unverified; verification assigned as an actionable item (owner: campaign operator role).
13. **HF model-card revisions** linking tracker/notebook and labeling source roles/dirty-build limits are an action for the card owners — outside this record's write scope.
14. **Privacy**: the initial catch-up commit (`437b2e41dc…`) and PR #51's commit expose a personal workstation email in author metadata. Neutral noreply identity is used for all commits after `437b2e41dc…`; remediation of already-published metadata requires repository-owner handling (flagged to the coordinator; no history rewrite performed here).
- Evidence: [P] links inline; reported-private items labeled.
- Interpretation and limitations: previous limitations stand; the synopsis adds the explicit missed-target verdicts.
- Decision and next action: fresh cold read round 4 next.

### 2026-11-15 14:52 UTC - Revision 4 after cold-read round 4: consolidated corrected glossary and stage table, microbatch correction, selection-rule precision, consolidated open-work table

- Question or hypothesis and reason for trying it: fresh cold-read round 4 returned NOTES (20 items); one concrete factual error remains in revision 3 (microbatch units). This entry consolidates the fully corrected operational detail in one place so readers need not apply per-entry patches.
- Published source revision: round-1 SFT YAML https://github.com/marin-community/marin/blob/9662fdba02896ffb03fd21066c1750b363e78cb6/experiments/post_training/long_loop/specs/sft.yaml (global batch 256, accumulation 8); exp2 launch report https://github.com/marin-community/marin/pull/11120#issuecomment-6863473319 (round-2 global microbatch 8, 32 accumulation iterations); selection code https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/marinskyrl/checkpoint_paths.py#L33 and https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/marinskyrl/callbacks/builtin.py#L260; RL YAML (reasoning_span shaping) https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/experiments/post_training/long_loop/specs/rl.yaml.
- Command or material configuration: see consolidated table below.
- External inputs and artifact references: as above.
- Environment and hardware when relevant: configured stage allocations: SFT 256 H100; RL 80 learner + 176 rollout; final eval 128 candidate + 8 judge (judge GPUs co-allocated locally). These are configured per-stage counts; no public concurrent-peak allocation timeline exists; requested-allocation totals: round 1 12,624.162622 H100-hours [P]; round 2 incomplete [M].

**CURRENT GLOSSARY (consolidated; supersedes every earlier glossary text in the 13:55 and revision-1/2 entries)**

- **RLOO-N** = RLOO-Neutral advantage estimator: infrastructure-failed samples are excluded from the group baseline and receive zero advantage; model failures remain eligible with zero reward; group size N=8. **Group admission**: a sample group is admitted via the group-admission rule; a stalled group admission halts training (exp1: `GroupAdmissionStalledError` at update 18).
- **MuonH** = the round-1 SFT optimizer (`grug_moe_muonh_v1`); round 2 consumed Adam (LR 5e-5, betas 0.9/0.95, cosine). RL optimizer: AdamW (separate).
- **Snowball** = the 67B-total / ~2B-active MoE initializer. **SFT** = supervised fine-tuning; **RL** = asynchronous RLOO-N RL; **Micro210** = frozen eval set at dataset revision `4564ed08…` — card total 210 tasks / 21 benchmarks / 450 slots, campaign-frozen cut excludes Tau3 → 200 tasks / 20 benchmarks / 440 attempt slots; the 120-task checkpoint-selection set (20 SWE + 100 non-agentic) is a separate training holdout; whether it overlaps final Micro210 membership is not publicly established.
- **TB2** = Terminal-Bench 2 (version pinned in the consumed eval config [M]); **SWE** = software-engineering (agentic coding) tasks; **UID** = attempt identifier in the eval dashboard; **CP/EP** = context/expert parallelism; **Sonic / EP ring** = MoE token-dispatch implementations (round 1 local Sonic, round 2 EP ring); **dropless equivalence** = whether MoE expert-capacity dropping behaves identically without dropping (unmeasured).
- **Harbor** (agent harness), **Pi** (agent CLI), **Daytona** (sandbox provider), **Kueue** (job queueing), **EvalDash** (eval dashboard), **Filer** (internal file store) — infrastructure services.
- **IsoFLOP D cells**: width/depth variants of the small ablation model at global batch 256 (d512/d768/d1024; full vs assistant = loss-mask scope); #11440's completed workaround controls include separate calibration probes at global batch 8 or 2 with different geometry — those are diagnostics, not the D campaign cells.
- **TITO / TIS / router replay**: token-in-token-out trajectory format, truncated importance sampling, and router-replay mechanics — defined in the linked tracker; not redefined here.

**CORRECTED STAGE TABLE (consolidated; supersedes all earlier tables; [P] links for config claims)**

| Field | Round 1 (exp1) | Round 2 (exp2) |
|---|---|---|
| SFT updates | 746 (linear schedule, horizon 4,750, 475-step decay, stopped at 746) | 204 (numbered 0–203; cosine) |
| SFT global batch | 256 (global microbatch 32, accumulation 8) | 256 (global microbatch 8, accumulation 32) |
| SFT context / positions | 262,144 / 50,063,212,544 packed | 262,144 / 13,690,208,256 sampled |
| SFT optimizer | MuonH (`grug_moe_muonh_v1`) | Adam (LR 5e-5, betas 0.9/0.95, cosine) |
| Precision / optimizer state | BF16 params/compute; offloaded optimizer state | FP32 params; on-device Adam state |
| CP / EP / dispatch | CP8 / EP1 / local Sonic | CP32 / EP8 / EP ring |
| Loss: SFT chat mask | assistant-only | assistant-only; plus 2,821,707,252 causal-replay loss tokens |
| Loss: RL token coverage | 94.97% audited on the FIRST consumed RL SWE batch only (recorder repairs not in the scored model) | recorder/harness repairs present [M]; no public coverage audit |
| RL context / budget / updates | 102,400 / 12 h / 17 completed (update-18 admission failed) | 102,400 / 10 h / 20 completed |
| Judge allocation | 8 H100 co-allocated (eval YAML) | MiniMax judge via harness [M] |
| Selected checkpoint / model | RL16 / Round1-2026.11.12.7 @ `475ba62d…` | native step 20 / Round2-2026.11.14.13 @ `6546ba2d…` |
| Final eval scored | 439/440 (UID 399 unscored — a task in the tb2-recovery benchmark; tracker's "DNA assembly" is that task's name) | 437/440 (TB2 UIDs 225/268/269 unscored) |
| Timing vs <36 h | 80.57 h — missed | ≥38 h 49 min bound — missed by ≥2 h 49 m |
| Scores (scored-attempt mean / benchmark-mean avg) | 0.4525643127 / 0.5216066358 | 0.4048171547 / 0.4828958533 (round-1/2 tracker recomputation from published benchmark rows agrees to rounding) |

- RL reward shaping: `reasoning_span.target: marker` in the pinned RL YAML [P] applies a configured 0.1 penalty keyed on a span marker; the exact trigger/unit semantics are defined by the base implementation, and round-2 equivalence (dirty build) is unverified. Shaped training reward ≠ raw validation reward.
- Selection rule (pinned code [P]): selection uses an **adjusted finite trailing window of the five scheduled validations at two-update intervals**; every scheduled observation in the window must be complete (scored); ties keep the earliest checkpoint. Round-2 equivalence to this code is unverified (dirty build). Candidate ledgers remain unpublished.
- Remaining unavailable evidence (explicit): consumed dirty-build sources/configs (no public reconstruction), SFT mixture source weights/realized round-1 exposure (unmeasured publicly), seven unknown upstream revisions (sources not named publicly), attempt-level score/recovery ledger, D-run audit tables, round-2 allocation-hours. These are acknowledged gaps, not broken links.
- Open work (consolidated, dated 2026-11-15): (1) publish round-2 recipe/build provenance bundle — owner: campaign operator; blocked on private inputs; (2) retention/resumability verification (dated inventory + expiry check) — owner: campaign operator, untracked issue; (3) marin#11440 stall diagnosis — open; (4) IsoFLOP original-runtime throughput qualification — blocked (workaround timings excluded); (5) SFT-mix data arm — design publicly evidenced; no executed cleaning counts published; teacher/prompt pins undecided; (6) HF card revisions linking tracker/notebook — owner action, outstanding; (7) commit-metadata remediation (personal workstation email in the initial notebook/index commits) — repository-owner decision, flagged to the coordinator; (8) no new long-loop round authorized — user decision pending.
- Evidence: [P] links inline; [M]/[R] labeled.
- Interpretation and limitations: as previously recorded.
- Decision and next action: fresh cold read round 5 next.

### 2026-11-15 14:58 UTC - Revision 5 after cold-read round 5: broken-link fix, shaping semantics, loss-row split, timing-bound wording

- Question or hypothesis and reason for trying it: fresh cold-read round 5 returned NOTES (21 items); concrete fixes below.
- Published source revision: corrected selection-callback link https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/skyrl-train/skyrl_train/callbacks/builtin.py#L260 (the revision-4 link's directory path was wrong — no `marinskyrl/callbacks` directory exists at that revision; the correct package root is `skyrl-train/skyrl_train`). Shaping implementation: https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/skyrl-train/skyrl_train/trajectory_runners/trajectory_reward_shaping.py#L507.
- Command or material configuration: n/a.
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:

**Corrections**

1. **Selection-callback link fixed** (revision 4 cited a 404 path). The revision-4 description of the selection rule (finite trailing window of five scheduled two-update validations; all scheduled observations must be scored; ties keep the earliest checkpoint) is supported by the corrected base code at the link above.
2. **Reasoning-span shaping semantics** (base implementation [P]; round-2 dirty-build equivalence unverified): the pinned function subtracts **0.1 from each located malformed-span marker's token advantage** when detected marker locations match the violation count; otherwise it applies a **trajectory-level reward penalty of −0.1**. A "malformed span" is a reasoning span whose marker pattern fails the violation detector's check. So the penalty is per-marker-token (advantage-level) in the matched case and once per trajectory in the fallback case.
3. **SFT loss rows split** (revision 4's single row conflated them): SFT chat masking = assistant-only supervision in both rounds; SFT causal replay = replay text also carries loss — round 2 consumed **2,821,707,252 causal-replay loss tokens** (round-1 realized replay-loss total not published [M]). They are separate rows in any later table.
4. **Elapsed-time wording** (revision 2 correction 5 over-attributed): the ≥38 h 49 min bound requires only the submission and completion timestamps — both are public reports; the **exact initial submission timestamp is missing from the public record**, and native SFT-phase timing is separately unavailable. What is established: under the public-reporting endpoints, the target was missed; exact elapsed and phase attribution remain unmeasured.
5. **Overlap scope**: the 120-task selection holdout's disjointness from **training** is the established claim; its overlap with final Micro210 membership is **not established** and is not claimed. A sanitized three-way membership/overlap report remains unpublished.
6. **Issue #49 navigation**: a milestone comment with the revision-4+ notebook link and current verdicts accompanies this entry; earlier comments remain as dated history.
- Evidence: [P] links inline.
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 6 next.
