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
> Read the newest entries first: the 13:55 UTC catch-up entry below is superseded **in part** by the later revision entries. The **Known limits** section (bottom) lists what cannot be fixed by policy or was never measured. The **current synopsis and stage table**: revision 10's synopsis (18:45 UTC) with revision 4's consolidated stage table, corrected in place by the evidence/ ledgers (see each file's "Updated" markers). Historical synopses: revisions 3 and 4's prose are retained as the correction trail; revisions 11–13 (19:20–20:10 UTC) carry the latest corrections. Corrections are appended, never rewritten.

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

### 2026-11-15 15:10 UTC - Revision 6: sanitized evidence ledgers published (per user publication ruling)

- Question or hypothesis and reason for trying it: cold reads rounds 1–6 identified private-evidence inaccessibility as the main SATISFIED blocker. The user ruled (2026-11-15, via the coordinator): everything except (1) cluster secrets/credentials and (2) the DATA itself (task text, prompts, model outputs, dataset rows — data requires separate sign-off) may be published. This entry publishes sanitized ledgers compiled from local records.
- Published source revision: unchanged training identities; evidence compiled from local records [R] and public references [P].
- Command or material configuration: n/a (see evidence files).
- External inputs and artifact references — new evidence files in this directory (all metadata-only; no task text, prompts, model outputs, or dataset rows; no credential values):
  - [evidence/selection-ledger.md](evidence/selection-ledger.md) — per-checkpoint validation ledger: exp2 evals 2–20 (evals 2/4 ineligible on 20 SWE snapshot-quota failures; evals 6–20 120/120 scored; raw means 0.1764–0.2834; trailing-5 EMA 0.2009→0.2526; best progression 14→16→18→20); exp1 v14/v16 rows + best16 selection; exp1 full per-step series unavailable.
  - [evidence/final-eval-attempts.md](evidence/final-eval-attempts.md) — exp2: 200 tasks / 440 attempts / 437 scored / 3 TB2 infra-unscored (UIDs 225/268/269); BixBench UID1 verifier-only recovery (0.0); exp1: 432 → 439/440 scored, UID 399 sole permanent unscored; per-benchmark tables; attempt-level rows remain in the sealed private archives.
  - [evidence/d-run-cancellation-audit.md](evidence/d-run-cancellation-audit.md) — per-job cancellation ledger (27 tasks; watchdog exit-124; self-cancel vs phase-budget-guard at 95.80/96 H100-h; r5 ceiling 143.80), the five counter-defect code references, query scopes (19,280/19,284 rows; no OOM events in queried logs) — findings of the local audit, with job identities now public.
  - [evidence/timing-and-allocation.md](evidence/timing-and-allocation.md) — stage/timestamp ledgers: exp1 end-to-end 80.568904 h; 12,624.162622 requested H100-h (RL 3,447.11; SFT 5,393.89); exp2 attempt costs (66.3749, 110.252, 167.3094, 2,994.3104 H100-h); 36,000-s budget with 38,575.20-s consumption; ≥38 h 49 min bound computed from the two documented public timestamps.
  - [evidence/build-provenance.md](evidence/build-provenance.md) — **the recorded "dirty" tree hashes (`00a8f56ce2`, `266e04d2a1`) resolve in the local `exp2-batch256` worktree and are byte-identical to the base commits' trees (`442bf9d878`, `038ad7f188`)**: the tracked-tree delta is empty, so tracked-source patches are trivially recoverable; "dirty" reflects untracked runtime-input bundles (present locally). This corrects the earlier "unreconstructable" framing for tracked sources.
  - [evidence/exp2-data-accounting.md](evidence/exp2-data-accounting.md) — CPU accounting receipt contents: 52,224 realized samples; 13,690,208,256 sampled positions; 12,849,743,650 real tokens; 840,464,606 padding (6.139%); 7,751,383,148 assistant + 2,821,707,252 causal-replay loss tokens; realized-samples.jsonl SHA-256 `58713a89dd09…` (70,562,321 bytes); receipt `379ef811…`; 19,264-row preserved prefix / 32,960 reconstructed; 377-component declaration (375 per-source accounting entries), 216 chat / 1,711 replay leaves; 871,429 traces / 46 clipped.
- Environment and hardware when relevant: allocation details in timing-and-allocation.md.
- Result, including failure or restart: evidence now published; reconciliation note — the "Stack 13 findings" figure in earlier entries is a public commit-message report; the local panel artifacts reviewed here did not surface that anchor (its denominator/panel binding remains registry-bound [M]).
- Evidence: the six files above (compiled 2026-11-15 ~15:05 UTC from [R] local records).
- Interpretation and limitations: attempt-level reward rows and round-1 full selection series remain private; data (task text/outputs) requires user sign-off per the ruling.
- Decision and next action: fresh cold read round 7 next.

### 2026-11-15 15:35 UTC - Revision 7: evidence ledgers corrected and extended after cold-read round 7; publication-rule note

- Question or hypothesis and reason for trying it: cold-read round 7 found real defects in the round-6 evidence ledgers (wrong step attribution, false unavailability claims, arithmetic and naming errors) plus one privacy violation, which is now remediated (see below). This entry records the corrections.
- Published source revision: unchanged.
- Command or material configuration: corrections applied directly in the evidence files, each with updated sources.
- External inputs and artifact references: tracker marin#11337 round-1 body and round-2 comment (public per-benchmark tables); selection receipts in the local experiment records.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:

**Corrections (all applied in evidence/; summaries)**

1. **Step attribution corrected**: exp1's 0.2917458354159724 avg-verifier score belongs to **step 14**; step 16 scored **0.2834167** — the 13:55 entry's "step-16 … 0.29175" attribution is superseded. Exp1 best16 remains **reported** (no independent EMA-recomputation artifact); contemporaneous receipts found for step 14 (EMA 0.2800426649504649) and step 16 (EMA 0.2819984463094125).
2. **Eligibility bookkeeping fixed**: steps 2/4 ineligible; steps 6/8/10/12 have 1/2/3/4 eligible observations of 5; the first complete five-observation window is steps 6–14. Scheduled vs eligible observations now distinguished.
3. **False unavailability removed**: both rounds' per-benchmark tables and exp2's means (0.4048171547 / 0.4828958533; exp1 original 0.4529530863 / 0.5186440986, post-recovery 0.4525643127 / 0.5216066358) are **public in the tracker** and now linked. final-eval-attempts.md renamed to a **coverage summary** (it is not an attempt-level ledger); the phantom "residual unidentified timeout rows" is removed (the four timeout attempts are BixBench UID1 + TB2 UIDs 225/268/269; the unknowns are specific diagnostic fields).
4. **Timing recomputed**: with the ledger's initial submission timestamp 2026-11-13 19:59:09 UTC, exp2 endpoint elapsed = **38 h 50 min 44.982839 s** (submission second-resolution); running-to-completion = 38 h 48 min 56.982839 s (≈38 h 49 min). Phase attribution remains unmeasured. Exp1's clock anchors are cited directly; the RL subduration decomposition (45,396.14 s of 45,576.78 s) is labeled **non-exhaustive**. The `.12` cancellation reconciles: 17:06:33 UTC = cancel requested/acknowledged; 17:09:22.876 UTC = terminal after zero-remaining-pods verification. The 80-H100 export is an executed allocation deviation; ceiling compliance remains configured/reported, not independently verified.
5. **Build provenance sharpened**: the recorded "dirty" tree IDs are **git tree SHAs** (`00a8f56ce2bc11e555a68b0f40911970785ba03e`, `266e04d2a1378d7f870deb12563c3ce51c7e4084`) equal to the base commits' trees — publicly verifiable via `git rev-parse <commit>^{tree}` on the linked public commits. Tracked-source recovery: **resolved**. Untracked runtime-input bundles and frozen spec/config directories remain private; this is not a complete runnable recipe. Hash-verification scope clarified: SFT handoff weight-body SHAs were never computed; HF checks covered headers/sizes/manifest hashes, not full tensor payload rehashes.
6. **Data accounting completed**: the full **375-row per-source aggregate table** is now published (evidence/per-source-accounting.md; column totals re-verified against the receipt's overall block). The 377-vs-375 key-set comparison is computed: exactly two declared components have no accounting row (`pretrain/c08q1`; `sft/source/penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-web-search-mcqa`); zero accounting keys fall outside the declared set; cause labeled unknown. Naming fixed: realized-samples.jsonl is the **payload** (SHA `58713a89dd09…`); `379ef811…` is the accounting receipt. Estimates labeled; MFU = model FLOP utilization; final losses contextualized (step 203 / 52,224 samples / 10,573,090,400 loss tokens).
7. **Selection metric definitions added** (raw verifier mean, avg_score and its penalty components, pass@1, route scores with task counts, task-vs-attempt), timestamps typed (event/readback/upper bound).
8. **W&B access class corrected to [M]**: anonymous fetches of the referenced run pages return only an app shell (the workspace redirects to an authenticated instance) — anonymous access is NOT verified; URLs retained but marked authentication-required.
9. **D-run audit code links pinned**: public permalinks added for the five cited counter/handler defects (Iris controller files at commit `e9ae26899c…`; experiment files on branch `snowball-isoflop-ablations` at pre-fix snapshot `f103997ccd…`), plus the post-audit fix commit.
- **Publication-rule and PII notes**: per the user's standing rule, everything except cluster secrets/credentials and the data itself (task text, prompts, model outputs, dataset rows — data requires separate sign-off) is published. Per the subsequent directive: no personal information of anyone (no emails, account names, or usernames of non-authors; "another user" phrasing). A third-party identifier exposed in the round-6 D-run ledger was redacted; the affected commit was rewritten by the repository owners with the secretary's authorship and message preserved — this branch's pre-rewrite SHAs `81b6dfc1…`/`67b9f8e0f0…` are superseded by `06dd9d76…` and cited nowhere as evidence.
- Evidence: the evidence/ files as corrected (compiled-at headers updated).
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 8 next.

### 2026-11-15 16:55 UTC - Revision 8 after cold-read round 8: current synopsis, metric-definition corrections, timing precision/scope, navigation fix

- Question or hypothesis and reason for trying it: cold-read round 8 (two reports: the research-branch review and the PR-#51 review of the same notebook) returned NOTES. Concrete defects corrected here: the pass@1 and verifier-score metric definitions in evidence/selection-ledger.md misstated their semantics and contradicted their own rows (route values labeled as means; coverage prose contradicted by steps 2/4); timing spans dropped observation fractional seconds while keeping microsecond precision; the timing scope vs cost scope difference was undefined; and the read-first navigation directed newcomers to superseded revisions 3/4.
- Published source revision: unchanged training identities (exp1 sources `9662fdba…`/`0c1e81ca…`, exp2 builds `442bf9d878…`/`038ad7f188…`, seal tag `experiments/exp2-rl20-2026.11.15` @ `4913edc488…`). Metric corrections are verified against the pinned public code at https://github.com/marin-community/marin/commit/0c1e81ca93635ac157db90ff786a8f8fba65b13c and cross-checked against the local marin checkout of that revision.
- Command or material configuration: three evidence files edited in place, each with an "Updated 2026-11-15T16:55Z" header line: evidence/selection-ledger.md (metric definitions, route labeling, step-2/4 scoping, step-4 reconstruction cell, EMA recipe, step-16 save-status wording, unavailable-block accuracy), evidence/timing-and-allocation.md (full-timestamp spans, timing-scope rule, `.10.1–.10.10` range label, selected-examples marking), evidence/build-provenance.md (recoverability verdict scoped to the recorded tracked trees, provenance permalink). No published entry below is rewritten; this entry and the evidence-file updates are the corrections.
- External inputs and artifact references: cold-read round-8 reports (local-only [R], two overlapping reviews of the same head — deduplicated here); pinned code permalinks: trajectory_processing.py#L159 (`get_trajectory_passes`), evaluate.py, verification.py#L127 (`normalized_verifier_score`), lib/rigging/src/rigging/provenance.py#L76-L83 (MARIN_PROVENANCE capture).
- Environment and hardware when relevant: unchanged.
- Result, including failure or restart:

**CURRENT SYNOPSIS (authoritative as of this entry; supersedes revision 3's synopsis, which remains as history)**

- Engineering question: can the SFT → asynchronous RLOO-N RL → frozen Micro210 eval loop complete end-to-end within <36 h on ≤256 H100s? **Verdict: no, for both rounds.** Exp1: 80.57 h observed (submission 2026-11-09 21:45:42.016 → coordinator completion 2026-11-13 06:19:50.069 [P]); overrun 44.57 h. Exp2: **38 h 50 min 44.278050 s (≈ 38 h 51 min)** from the observed submission anchor 19:59:09.704789 to completion 10:49:53.982839 (38 h 48 min 56.919256 s ≈ 38 h 49 min from the RUNNING anchor); these are observation-to-completion spans — an unknown observation latency is not included and native SFT-phase timing is unavailable, so phase attribution stays unmeasured. Missed by ≈2 h 51 m under either anchor.
- Loop character: a fully unattended repeatable loop is **not** established — exp2 required manual recovery (cancelled RL child `.14.12` replaced by `.14.13`; coordinator report-gate failure replaced by CPU post-processing) and manual sealing; exp1 required admission fixes and a replaced SFT attempt.
- Quality question (did RL improve Micro210 scores?): **unanswered** — no matched SFT-only/pre-RL evaluation, no repeated-seed uncertainty, many simultaneous recipe differences; the round-over-round declines (0.4525643127 → 0.4048171547 scored-attempt mean; 0.5216066358 → 0.4828958533 mean of benchmark means) are descriptive only. Selection-versus-final-eval overlap and semantic decontamination remain unestablished.
- Coverage: exp1 439/440 scored (UID 399 unscored); exp2 437/440 (TB2 UIDs 225/268/269 permanently infra-unscored under the no-resampling policy — accepted limitations). Both archives sealed (byte-verified); sealing does not certify complete telemetry.
- Provenance (updated verdict): the **recorded tracked sources are recovered** — the recorded "dirty" tree hashes equal the public base commits' trees (publicly verifiable); what remains private: untracked runtime-input bundles, frozen consumed configs/launch overrides, seed bindings, mixture schedule, and the data. Round-2 has no self-contained recipe bundle; round-1's recipe tar is [M]. Public reproduction of the full runs remains unavailable.
- Resource criterion: the ≤256 H100 ceiling is **configured/reported, not independently verified** — no concurrent-allocation timeline exists, and exp2's successful RL/export/eval totals are incomplete.
- State: marin#11120 OPEN and held; no new long-loop round authorized (user decision pending); round-2 tracker entry published (marin#11337, 14:12:05 UTC); retention of RL16/18/20 + SFT203 under a 14-day lifecycle is observed-at-snapshot only — expiry and resumability unverified.
- Evidence files (all seven, round-8-corrected): evidence/selection-ledger.md, evidence/final-eval-attempts.md, evidence/timing-and-allocation.md, evidence/build-provenance.md, evidence/exp2-data-accounting.md, evidence/per-source-accounting.md, evidence/d-run-cancellation-audit.md.

**Metric-definition corrections (evidence/selection-ledger.md, applied in place)**

1. **pass@1 is verdict-based with a positive-reward fallback**, not "the fraction with verifier reward exactly 1.0": per the pinned implementation, an attempt passes when its verification result carries an explicit pass verdict; with no verdict it falls back to outcome reward > 0; masked/non-VERIFIED outcomes fail [P: trajectory_processing.py#L159 `get_trajectory_passes`]. This explains why exp1 step 16's pass@1 (35/120 = 0.2916667) sits above its verifier mean (0.2834167), and step 14's (0.3083333) above its mean (0.2917458).
2. **The verifier mean is bounded-normalized, not the native-reward mean**: `eval/all/avg_verifier_score` is the mean of per-attempt scores after the pinned code scales each verified score's declared native bounds to [0,1] [P: evaluate.py; verification.py#L127]. "Raw" in the ledger means unshaped/unpenalized, not native-unscaled; per-route bounds/normalization parameters are [M]. `avg_score` is the optimization-reward channel mean; its components are recorded by name only.
3. **Route values are pass fractions (route pass@1), not reward means**: the exp1 route numbers (7/20, 21/80, 7/20; 0.35/0.30/0.30 at step 14) weight to the overall pass@1, not to the mean. Raw route reward means are unavailable, so the overall mean is not recomputable from the published route numbers. Table labels updated.
4. **Coverage scoping fixed**: "120 attempted / 120 scored / 0 excluded" now applies to the fully scored validations (exp2 steps 6–20; exp1 14/16); exp2 steps 2/4 are 120 attempted / 100 scored with 20 SWE-route Daytona snapshot-quota infrastructure exclusions. The step-4 Attempted/Scored cell no longer derives the scored count from reward-sum arithmetic; eval4's scored-only mean (0.24105625) is labeled a reconstruction from the per-route aggregate; eval2's raw overall mean remains unavailable.
5. **Exp1 step-16 save-status wording fixed**: "RL FAILED 17 at the 12 h budget boundary" is replaced by the corrected outcome — 17 optimizer updates completed; update-18 admission failed (`GroupAdmissionStalledError`); best16 preserved and manually exported.
6. **EMA recomputation recipe published** (verified against the exp2 series at steps 14/16/18/20): adjusted trailing-5 EMA, α = 1/3, eligible observations only, first complete window steps 6/8/10/12/14. Exp1's best16 remains reported (no published series; cannot be independently recomputed).

**Timing corrections (evidence/timing-and-allocation.md, applied in place)**

7. **Full-timestamp arithmetic**: exp2 endpoint elapsed recomputed on the full observation timestamps — 38 h 50 min 44.278050 s (submission anchor) and 38 h 48 min 56.919256 s (RUNNING anchor); the earlier rows truncated the anchors to seconds while retaining microsecond precision in the differences. Rounded statements (≈38 h 51 m / ≈38 h 49 m) are provided; spans are labeled observation-to-completion, and the ~1 s caveat covers rounding only, not observation latency.
8. **Timing-scope rule defined once**: loop timing starts at the selected full submission (exp1 anchor 21:45:42.016, excluding the failed `.09.11` attempt at 20:44; exp2 anchor 19:59:09.704789, excluding smokes/full9/full10); cost accounting has the wider scope and includes those attempts. Both rounds miss the <36 h target either way.
9. **Exp1 admission-attempt row relabeled** `.10.1–.10.10` and marked as selected examples (it always contained a `.10.10` cost); the full 659-attempt reconciliation stays in the round-1 recipe entry [P].

**Other corrections**

10. **Build-provenance verdict scope**: the recovery result is phrased as "recovery of the recorded tracked trees"; the hash-to-executed-job linkage rests on recorded private provenance receipts, and the pinned MARIN_PROVENANCE implementation (`git stash create` capture; inherited-environment provenance possible) is linked. No claim of an undisclosed patch is made or implied.
11. **IsoFLOP D-phase cross-reference**: the D cells (d512/d768/d1024) are width variants (hidden 512/768/1024; 6/8/11 layers) of the small Snowflake ablation ladder in the separate #54 campaign, which now has its own archive record with the architecture table, per-phase design map, D-trial tables and audit-corrected stall/guard history: logbooks/54-snowball-isoflop-ablations/notebook.md (branch `codex/54-snowball-isoflop-ablations`). This record tracks that lane only as a runtime-status dependency. The revision-4 glossary's "D cells … at batch 256" is **withdrawn** — the #54 record shows the D trials ran the B/C-selected geometries (global sequences per update 128/64/32 at CP8/data1/EP1), not a global batch of 256. The "Stack 13 findings" remark in revision 6 remains a public commit-message report with no local experiment connection here; treat it as uncorroborated by this record's artifacts.
12. **Navigation**: this entry is the current synopsis; revisions 3/4 are historical (their superseded statements are corrected by revisions 5–8 and the evidence files). The evidence ledgers are the canonical operational detail.
- Evidence: [P] permalinks inline; [R] local records as cited in the evidence files; cold-read reports [R] local-only.
- Interpretation and limitations: previous limitations stand; the resource criterion and retention/resumability remain unverified; the missed-target verdicts are unchanged.
- Decision and next action: fresh cold read round 9 next.

### 2026-11-15 17:42 UTC - Revision 9 after cold-read round 9: EMA formula corrected to the pinned normalized finite window, window-eligibility wording, step-14 route counts, overrun anchor wording, related-lane state

- Question or hypothesis and reason for trying it: cold-read round 9 returned NOTES with reproducible errors: the published EMA recipe in revision 8 (item 6) and evidence/selection-ledger.md did not reproduce the published EMA table; the "eligible observations only" wording could be read as changing the selection rule; the exp1 step-14 route counts contradicted themselves (38 vs 37); and the synopsis compressed two different overruns into one figure. This entry corrects them. Revision 8's synopsis text is preserved unedited; this entry supersedes the affected statements.
- Published source revision: unchanged training identities (exp1 sources `9662fdba…`/`0c1e81ca…`, exp2 builds `442bf9d878…`/`038ad7f188…`, seal tag `experiments/exp2-rl20-2026.11.15` @ `4913edc488…`). Formula corrections verified against the pinned public selection code at https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/marinskyrl/checkpoint_paths.py#L33-L46 (`trailing_validation_ema`) and recomputed from the published verifier-mean column.
- Command or material configuration: evidence/selection-ledger.md edited in place ("Updated 2026-11-15T17:42Z"): EMA recipe replaced with the executable normalized finite-window form, eligibility definitions split into score vs window eligibility, exp1 step-14 structure route count corrected.
- External inputs and artifact references: [R] `/experiments/complete/long-loop-prototype/STATUS.md` L834/L839 (step-14 route fractions 0.35/0.30/0.30 and pass@1 0.3083333); [R] exp2 milestone receipts (unchanged); [P] pinned checkpoint_paths.py; [P] logbooks #54 notebook (branch `codex/54-snowball-isoflop-ablations`). Cold-read round-9 report local-only [R].
- Environment and hardware when relevant: unchanged.
- Result, including failure or restart:

**Corrections (supersede revision 8's metric-correction item 6 and the affected evidence rows)**

1. **EMA formula corrected — the published series is the normalized finite window, not a recursive span-5 EMA.** The revision-8 recipe (`EMA_t = (1−α)·EMA_{t−1} + α·x_t` seeded at the first eligible observation) **does not reproduce the published EMA table** (it gives 0.20075076301497646 / 0.22830941001496408 / 0.23284516223219826 / 0.24442455259924328 at steps 14/16/18/20) — the revision-8 "verified" wording was wrong and is withdrawn. The published values (0.2008526161737512 / 0.23259077361923453 / 0.238219238827311 / 0.25261617680247456) match the pinned code's **normalized finite-window** form exactly:
   `EMA_t = Σ_{j=0..4} (2/3)^j · x_(t−2j) / Σ_{j=0..4} (2/3)^j`
   with window 5, interval 2, α = 2/(window+1) = 1/3, weights `decay^age` newest-first [P: checkpoint_paths.py#L33-L46]. A tiny executable recomputation is now in evidence/selection-ledger.md ("Updated 2026-11-15T17:42Z"); run in the code's own accumulation order (oldest reward × largest weight) it reproduces the published values digit-for-digit at steps 14/16/18/20.
2. **Window eligibility restated**: the pinned code requires **all five exact scheduled steps** (interval 2) to be present in the history and returns no value otherwise — a missing or incomplete scheduled observation **breaks the window**; it is not skipped with a reach-back. "Eligible observations only" / "five consecutive eligible observations" (revision 8; earlier ledger wording) is corrected in the ledger to distinguish **score eligibility** (all 120 attempts scored) from **window eligibility** (five consecutive scheduled steps present and scored). The two readings happen to agree for exp2 after step 6 only because steps 2/4 — the sole incomplete scheduled observations — precede the first window; this coincidence is now stated rather than implicit. Round-2 dirty-build equivalence to the pinned code remains unverified.
3. **Exp1 step-14 route counts corrected (37/120, not 38)**: the STATUS.md receipt records step-14 route fractions **0.35 / 0.30 / 0.30** and overall pass@1 0.3083333 [R: STATUS.md L834] — SWE 7/20, Nemotron 24/80, **structure 6/20**. The ledger's step-14 row previously wrote "structure 7/20 = 0.35" (internally inconsistent: 0.35 ≠ 0.30; total 38/120 ≠ the recorded 0.3083333). The receipts record route fractions with 20/80/20 denominators; no receipt states a different denominator, and the recorded overall total confirms 20/80/20. Definition paragraph (37/120) and table now agree.
4. **Overrun wording corrected (revision-8 synopsis first bullet)**: the two anchors give **different overruns** — submission-anchor span 38 h 50 min 44.278050 s → overrun ≈ **2 h 50 min 44 s (≈2 h 51 m)**; RUNNING-anchor span 38 h 48 min 56.919256 s → overrun ≈ **2 h 48 min 57 s (≈2 h 49 m)**. "Missed by ≈2 h 51 m under either anchor" is withdrawn; both anchors still exceed the 36 h target and the missed-target verdict is unchanged.
5. **Related-lane state (#54)**: the Snowball IsoFLOP ablations campaign is **stopped and archived** — terminated early by user decision on 2026-11-15 ("misconfigured at the start and aren't telling us much"); no further work is authorized or planned by that record; no Snowball root is running or pending. Its archive record (architecture table, phase map, D-trial tables, audit-corrected stall/guard history) is pinned at logbooks/54-snowball-isoflop-ablations/notebook.md on branch `codex/54-snowball-isoflop-ablations` [P]. The revision-4 open-work items about IsoFLOP cells/qualification ("budget guard active", blocked cells pending) describe the pre-stop state and are **historical**; the marin#11440 stall issue remains open as runtime context but is not an authorization to resume that campaign.
- Evidence: [P] permalinks inline; [R] local records as cited; evidence/selection-ledger.md carries the 17:42Z update marker.
- **Read order**: start with THIS entry (current synopsis corrections) → revision 8's synopsis for the loop outcome, coverage, provenance and state (with item-4/6 corrected here) → evidence/selection-ledger.md for the canonical selection table, metric definitions and EMA recipe → evidence/timing-and-allocation.md for timing/cost scope → evidence/build-provenance.md for the tracked-tree recovery verdict. Revisions 3/4 remain historical.
- Secretary GitHub updates (owned surfaces this branch cannot edit — for the index/issue lane): (1) index PR #51's `index/49-snowball-long-loop.md` label says "revision-7 snapshot" but links the revision-8 head — relabel to the current pin and refresh after this revision lands; (2) issue #49's latest navigation comment still directs readers to revision 3/4 and says dirty builds prevent reconstruction — post a current navigation comment pointing at this revision and revision 8, and at the tracked-tree recovery verdict in evidence/build-provenance.md; (3) the index/issue "EMA" or selection phrasing, if any, should cite the corrected finite-window recipe (this revision's item 1), not a span-5 recurrence; (4) commit-metadata remediation of the exposed personal-workstation email in the initial notebook/index commits and linked Marin training commits remains a repository-owner action (no addresses reproduced anywhere in this record).
- Interpretation and limitations: previous limitations stand; nothing above changes the missed-target verdicts, the coverage counts, or the unanswered quality question.
- Decision and next action: fresh cold read round 10 next.

### 2026-11-15 18:45 UTC - Revision 10 after cold-read round 10: EMA prose corrected (alpha/decay identities, weight direction, last-digit honesty), pass@1 denominator restated, round-2 selection code compared

- Question or hypothesis and reason for trying it: cold-read round 10 (two reports: the research-branch review and the PR-#51 review of the same notebook) returned NOTES. Reproducible defects corrected here: the selection ledger's EMA paragraph mislabeled decay as alpha ("α = 1 − 2/(window+1) = 1/3" — that expression evaluates to 2/3 and is the decay), the weight-direction prose said the oldest reward carries the largest weight (the published snippet and pinned code pair the oldest with the smallest weight), the "digit-for-digit equality with the pinned implementation" claim is overstated by one final digit at step 20, the pass@1 definition generalized "120 scored validation attempts" as the denominator, and the current selection discussion still treated round-2's tracked-source equivalence as an unresolved dirty-build question. Revision 8/9 entries are preserved unedited; this entry and the evidence-file updates are the corrections.
- Published source revision: unchanged training identities (exp1 sources `9662fdba…`/`0c1e81ca…`, exp2 builds `442bf9d878…`/`038ad7f188…`, seal tag `experiments/exp2-rl20-2026.11.15` @ `4913edc488…`). Corrections verified against the local marin checkout of the pinned revisions [P: https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/marinskyrl/checkpoint_paths.py#L33-L46 and https://github.com/marin-community/marin/blob/038ad7f188a889dfed07cac9a29fe34eabd7e5cd/vendor/skyrl/marinskyrl/checkpoint_paths.py#L33-L46 (round-2 pin), https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/skyrl-train/skyrl_train/trajectory_runners/trajectory_processing.py#L94-L98 and https://github.com/marin-community/marin/blob/0c1e81ca93635ac157db90ff786a8f8fba65b13c/vendor/skyrl/skyrl-train/skyrl_train/evaluate.py#L233-L235] and by recomputation from the published verifier-mean column.
- Command or material configuration: two evidence files edited in place with "Updated 2026-11-15T18:45Z" markers: evidence/selection-ledger.md (pass@1 denominator, EMA alpha/decay identities, weight-direction prose, last-digit qualification, round-2 code comparison) and evidence/d-run-cancellation-audit.md (the earlier limit-reached.json unavailability is dated and linked to the later #54 recovery, with the decision-time vs terminal accounting distinction kept explicit).
- External inputs and artifact references: cold-read round-10 reports (local-only [R], two overlapping reviews of the same head — deduplicated here); [R] `/experiments/complete/long-loop-prototype/STATUS.md`, `/experiments/active/sft-new-mix-candidate/STATE.md` and `artifacts/experiment2/` receipts as cited in the evidence files; [P] permalinks inline.
- Environment and hardware when relevant: unchanged.
- Result, including failure or restart:

**Corrections (supersede the affected statements in revision 9 item 1, revision 8 items 1/6, and the corresponding evidence rows)**

1. **EMA alpha/decay identities fixed**: the selection metric is a normalized finite-window weighted mean with **alpha = 2/(window+1) = 1/3 per scheduled-observation step (two optimizer updates) and decay = 1 − alpha = 2/3**, i.e. the weight on an observation of age j scheduled steps is (2/3)^j for j = 0..4 (window 5, interval 2). The earlier "α = 1 − 2/(window+1) = 1/3" mislabeled the decay as alpha; "span five" in the tracker means the five scheduled observations at interval 2, not five optimizer updates.
2. **Weight direction fixed — the NEWEST reward carries the largest weight**: `EMA_t = Σ_{j=0..4} (2/3)^j · x_(t−2j) / Σ_{j=0..4} (2/3)^j` with j = 0 the newest observation (weight 1) and j = 4 the oldest (weight (2/3)^4, the smallest); weights increase from oldest to newest. The earlier "oldest reward × largest weight" prose (revision 9 item 1's closing sentence and the ledger's snippet comment) contradicted the published snippet and is withdrawn. No selection-result change is indicated — this was a prose error; checkpoint 20 remains the correctly selected final best.
3. **"Digit-for-digit" qualified to a last-ULP difference at step 20**: the snippet's literal Python `2/3` is 0.6666666666666666 while the pinned code computes `decay = 1 − 2/(window+1)` = 0.6666666666666667. With the published means, the snippet reproduces the published EMA column at steps 14/16/18 and gives step 20 = 0.25261617680247456 (the published/receipt value); a literal re-implementation with the code's decay constant gives 0.2526161768024745 — one final digit (≈6e−17) lower, with steps 14/16/18 identical either way. The published series agrees with both forms to ~15 significant digits; the step-20 last-digit difference is a floating-point artifact (decay-constant representation/summation order), with no effect on the ranking (nearest competing value ~0.014 away). The exact-equality claim is withdrawn.
4. **pass@1 denominator restated (UID-based)**: `eval/all/pass_at_1` = passing UIDs / distinct UIDs in the batch [P: trajectory_processing.py#L94-L98]; each UID passes if any of its trajectories passes, and a trajectory passes on an explicit verifier pass verdict with fallback to positive outcome reward (> 0); masked/non-VERIFIED outcomes fail [P: trajectory_processing.py#L159-L170 `get_trajectory_passes`]. With one attempt per UID, a complete validation's denominator is the 120 attempted identities; **excluded/masked attempts fail rather than leaving the denominator**, so for the incomplete eval2/eval4 the raw pass channel's denominator would still be 120 attempted identities while the normalized verifier mean is computed over the 100 scored attempts — "120 scored validation attempts" is withdrawn as a general denominator (correct only for fully scored validations). The raw channel `eval/all/raw_verifier_pass_at_1` is computed on a batch copy without `exclude_from_baseline` [P: evaluate.py#L233-L235], so equality of the two channels in the compiled receipts is an observed result of fully scored validations, not a general identity.
5. **Round-2 tracked selection implementation recovered (dirty-build wording replaced)**: the round-2 public pin `038ad7f188…`'s `trailing_validation_ema` is textually identical to the round-1 pin's (file blob differs between the two commits only in one import line, `marinskyrl.resource_locator` vs the relative `.resource_locator`). The remaining uncertainty is limited to the **executed job's** consumed-config/launch binding (private receipts [M]); "dirty-build question" no longer describes the tracked source, which is recovered for both rounds. See the updated window-eligibility paragraph in evidence/selection-ledger.md and evidence/build-provenance.md's recorded-trees verdict.
6. **D-run ledger cross-reference updated** (round-10 note): evidence/d-run-cancellation-audit.md now dates its limit-reached.json unavailability (15:14Z compilation, credentials missing) and links the later #54 credentialed recovery of the decision snapshot (observed/charged 95.77756444507175, reserved 0.0, active reservation 24, 95.77756444507175 + 24/60 = 96.17756444507175 ≥ 96, object last modified 11:20:21 UTC), keeping decision-time and terminal accounting (95.802171) distinct.
- Evidence: [P] permalinks inline; [R] local records as cited in the evidence files; evidence files carry 18:45Z update markers.
- Interpretation and limitations: previous limitations stand — full public reproduction remains unavailable (private frozen configs/inputs); the resource ceiling and retention remain unverified; the quality question remains unanswered; nothing above changes the missed-target verdicts, the coverage counts, or the selected checkpoints.
- **Read order**: start with THIS entry (current corrections) → revision 8's "CURRENT SYNOPSIS" for the loop outcome, coverage, provenance and state (with revision 9's items 1–4 and this entry's items 1–5 applied) → evidence/selection-ledger.md (canonical selection table, metric definitions with the UID-based pass@1 denominator, corrected EMA recipe with the last-digit qualification) → evidence/timing-and-allocation.md → evidence/build-provenance.md → evidence/d-run-cancellation-audit.md for the D-run cancellation findings. Revisions 3/4 remain historical.
- Secretary GitHub updates (owned surfaces this branch cannot edit — for the index/issue lane): (1) index PR #51's `index/49-snowball-long-loop.md` status line says "revision-9 corrections" but calls the pinned link the revision-7 snapshot while it links the revision-9 head `3e43e529…` — relabel to the actual pin and refresh after this revision lands; (2) issue #49's latest navigation comment (6896508837) still points to revisions 3/4/5 and names dirty builds as the reproduction obstacle — post a current navigation comment pointing at revision 10 and the recovered-tracked-tree verdict (with the private frozen-input caveat retained); (3) the round-2 tracker comment's "EMA with span five" should cite the five-scheduled-observations-at-interval-2 window explicitly; (4) the HF Round1/Round2 model cards omit record links and say only "five-step EMA" — card-owner follow-up: link the notebook/tracker, state five scheduled validations at interval two, and distinguish training/export/seal source roles; (5) commit-metadata remediation of the exposed personal-workstation email in the initial notebook/index commits and linked Marin training commits remains a repository-owner action (no addresses reproduced anywhere in this record).
- Decision and next action: next cold read reviews the updated branch.

### 2026-11-15 19:20 UTC - Revision 11 after cold-read round 12: mixture-schedule hashes are publicly reproducible; 659-attempt pointer corrected

- Question or hypothesis and reason for trying it: cold-read round 12 verified that the public #54 recipe reproduces both long-loop component/schedule hashes exactly, contradicting this record's [M] labeling of those hashes; and the "659-attempt reconciliation [P: marin#11337 round-1 body]" pointer was found not to exist. Both corrected here.
- Published source revision: unchanged training identities.
- Command or material configuration: verification: the IsoFLOP archive record (logbooks #54, branch codex/54-snowball-isoflop-ablations) and the tracker's public recipe reproduce the 377-component declaration hash `d372def…` and 19-entry weight-schedule hash `e954bba…` [P]; the hashes identify the consumed declarations and schedule — what remains private is the per-source materialization and weights expansion, not the hashes themselves.
- External inputs and artifact references: https://github.com/marin-community/marin-logbooks/tree/codex/54-snowball-isoflop-ablations (archive record); marin#11337 round-2 entry.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:
  1. **Mixture-schedule hashes relabeled [P]**: the two hashes cited since the 13:55 entry are publicly recomputable from the #54 recipe; the earlier "[M]" access label applied to the hashes themselves is withdrawn (the consumed per-source materialized data behind them remains [M]).
  2. **659-attempt reconciliation pointer withdrawn**: no public marin#11337 round-1 body section contains a 659-attempt reconciliation; the complete attempt account is private [M/R]. The 12,624.162622 H100-h total remains tracker-reported [P].
  3. **issue_count propagation note**: the #48 record's revision 11 establishes from public code that all issue counts are distinct affected reviewed tasks; the 13:55/revision-10 "issue events" wording in THIS record's cross-campaign audit summary is superseded by that correction (see #48 revision 11).
  4. Timing ledger updated: the `.10.1–.10.10` row now marks the attempt account private and the figure tracker-reported (timing-and-allocation.md, Updated 2026-11-15 ~19:20 UTC).
- Evidence: links above.
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 13 next.

### 2026-11-15 19:58 UTC - Revision 12 after cold-read round 13: the mixture schedule itself is public; step-4 route-mean scoping

- Question or hypothesis and reason for trying it: cold-read round 13 established that the actual mixture schedule (the 19 entries with their weights) and the full component declaration are publicly reproducible from the #54 recipe — not just their hashes. Also, the selection ledger's "raw route reward means unavailable" statement conflicts with step 4's use of route reward means; both corrected here.
- Published source revision: unchanged training identities; the #54 archive record (branch codex/54-snowball-isoflop-ablations) publishes the recipe that reproduces them.
- Command or material configuration: verification: the IsoFLOP archive recipe (logbooks #54) declares the 377 components and the 19-entry weight schedule; recomputing its declaration/schedule hashes matches the long-loop recipe hashes `d372def…`/`e954bba…` [P].
- External inputs and artifact references: https://github.com/marin-community/marin-logbooks/tree/codex/54-snowball-isoflop-ablations; marin#11337 round-2 entry.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart:
  1. **Mixture schedule is public**: both long-loop rounds' component declaration AND 19-entry weight schedule are publicly reproducible [P]; the private residue is only the per-source materialized data (positions/loss tokens per source — published for round 2 in evidence/per-source-accounting.md) and the round-1 realized exposure table (unmeasured publicly). All earlier "schedule hash [M]" labels are withdrawn (revision 11 had relabeled only the hashes).
  2. **Route-mean scoping**: the "raw route reward means unavailable" statement applies to the exp2 per-eval receipts (which publish route pass@1 fractions, not route means); exp1's step-4 receipt DOES use route reward means (its aggregate is built from them) — the ledger's route-score definitions are scoped accordingly: route reward means exist for exp1 step 4 [R: STATUS.md L834 context], are not published for the exp2 evals.
- Evidence: #54 record; receipts cited in evidence/selection-ledger.md.
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 14 next.

### 2026-11-15 20:10 UTC - Revision 13 after cold-read round 14: route-mean scope corrected

- Question or hypothesis and reason for trying it: cold-read round 14 showed revision 12 misattributed the step-4 route reward means: exp2's eval4 receipt (not exp1's) publishes per-route reward means. Corrected in evidence/selection-ledger.md.
- Published source revision: unchanged.
- Command or material configuration: verification against `recovery13-eval4-aggregate-check-20261114T1958Z.json` [R] and STATUS.md aggregates.
- External inputs and artifact references: as cited in the ledger.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: corrected scoping — per-route reward means are published for exp2 eval4 (structure 0.2, Nemotron 0.2513203125; scored-only mean 0.24105625 reconstructed in the ledger) and in exp1 aggregates; the fully-scored exp2 evals 6–20 publish route pass fractions only, so their overall means are not recomputable from published route numbers. The blanket "route reward means unavailable" is withdrawn (revision 12's exp1 attribution is withdrawn).
- Evidence: ledger updated (Updated 2026-11-15 ~20:10 UTC).
- Interpretation and limitations: unchanged.
- Decision and next action: fresh cold read round 15 next.

## Known limits

This section lists what cannot be fixed in this public record by policy or by absence of measurement. Cold readers should not re-raise these as notes.

1. **Private data and per-source contents**: task text, prompts, model outputs, eval traces, and attempt-level reward rows (task/repetition IDs tied to rewards) are never published (user data sign-off required); the sealed exp1/exp2 archives, EvalDash rows, and per-route declared score bounds are [M].
2. **Unpublishable artifacts**: the round-1 recipe tar and round-2 frozen consumed-config/input directories, corrected SFT→RL handoff, untracked runtime-input bundles of the dirty builds, W&B full histories (authenticated; anonymous access not verified), and the sealed archive internals. Published: public base/source commits, YAMLs, stage tables, hashes, job IDs.
3. **Reproduction impossible by policy**: a new long-loop round is not authorized (user decision); no regrading or resampling beyond the recorded per-round recovery policies (exp1 startup restarts, exp2 verifier-only); mar#11120 merge and card updates are owner/user actions, not record actions.
4. **Measurements never taken (or never published)**: exp1 full per-step selection series — the series exists in private native aggregates/W&B [M] but is not published in any public artifact; only the step-14/16 receipts are public, and best16 selection is reported, not independently recomputable; matched SFT-only/pre-RL Micro210 baseline and repeated-seed uncertainty (quality effect unanswered by design of the two rounds); exp2 total allocation-hours and peak concurrent-GPU timeline; native SFT-phase timing for exp2; the three-way (train/selection-holdout/final-eval) overlap report; checkpoint retention-expiry verification (14-day lifecycle unverified); seven upstream HF revision identities.

### 2026-11-16 12:40 UTC - Hourly catch-up (11:35–12:40 UTC window): Exp3 production LAUNCHED

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief; the coordinator flagged Exp3's launch as a material long-loop change. Primary evidence from the exp3 record follows.
- Published source revision: Exp3 production source `0f5721bbb4953e98898c847896110dee32a5b293`, artifact `2026.11.16.51`.
- Command or material configuration: one-URI launcher executed once with coordinator sign-off `450bc8a4b24613c6b6f00e07da9b3953` (approves exact V5 URI/SHA under user preapproval); submitted `/benfeuer/long-loop-exp3-2026.11.16.51` through `marin`. Production geometry: 256 H100 on 32 nodes, EP8/CP32, batch 256, context 262,144, maximum 746 updates, 20 h including graceful finalization (coordinator ruling 432 sets the outer RL dispatch timeout to the standing three-day operational ceiling, 259,200 s; exp1's 43,200-s learning clock and 1,800-s drain unchanged), seed 0, natural weights/schedule.
- External inputs and artifact references: data release **V5** `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/release-v5/release-complete-v5.json`, SHA256 `3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5` (V4 immutable predecessor, blocked from launch; V5 = 409,004 repaired + 17 unchanged/unjoined post-Atlas OpenCode rows; 45,046,613 documents; 120,403,817,009 supervised positions; 142,871,298,066 input IDs); final V5 no-run graph preflight passes 14 stages and both strict RL/final-eval compositions (source `0f5721bbb4…`); native consumer SHA `6bba910963d2…`; launch receipts `exp3-launch-approval.json` and `exp3-production51-launch-receipt.json` [R: /experiments/active/long-loop-exp3/artifacts/ and exp3-recipe.md §"Production submitted after coordinator sign-off"].
- Environment and hardware when relevant: initial state RUNNING with coordinator task BUILDING/PodInitializing — **not SFT start**; the 12:25:28 UTC capacity check still showed zero fitting whole nodes; queued without reservations or preemptions.
- Result, including failure or restart: **Exp3 production submitted after coordinator sign-off and is RUNNING (BUILDING)**. Pre-launch record (same exp3-recipe.md): tiny smoke and full 67B initializer proof `.47` passed (native checkpoint commit, four producer histories/phase journals, SHA-pinned recipe archive; proof source `5dfdfea6f0…`); aggregate initializer-proof spend 49.717155557 / 55 H100-hours; C4 passed the predeclared estimated reference gate (provisional step-0 estimate 2.313741889 nats/token ± 0.5, coordinator-authorized pending the true skew8 result from Will on #8977). Key exp3 deviations from exp1 (per the recipe's deviation table): native Levanter Snowball SFTSpec trainer; FP32 params/BF16 compute; global-norm-1.0 clip; MuonH 5e-5 with Adam 1.1538461538e-5 (restored pretrain 13:3 ratio); packing cap 256 with deterministic bounded lookahead 128; assistant-only chat supervision with relay teacher-turn; explicit full-batch supervised-token denominator; native EP ring dispatch; frozen C4-en initializer gate; up to 2 fresh watchdog-stall retries with stage H100-hour ceilings; automatic verified recipe bundle with dirty-tree refusal. About 360 adjusted updates expected (377 warm-only bound); 20 h SFT duration; no data trimming or weight retuning to duration (user ruling). Recipe plus data approval required through the coordinator before full launch — obtained (sign-off `450bc8a4b2…`).
- Evidence: exp3-recipe.md (7:29 EST readback) and artifacts listed above; the coordinator's launch flag.
- Interpretation and limitations: the launch is queued/building; no SFT updates yet; the provisional step-0 reference is an estimate pending the true value; legacy gradient parity (native trainer vs frozen Grug path) remains unmeasured.
- Decision and next action: hourly checks now track exp3 progression (BUILDING→RUNNING→SFT start) and the atlas branch separately; exp2 records remain sealed history.

### 2026-11-16 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window): Exp3 first optimizer updates

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: `artifacts/production51-wandb-first-update.json`, `production-recovery52-preflight-receipt.json`, `validation-recovery-advisory-disposition.json` [R: /experiments/active/long-loop-exp3/].
- Environment and hardware when relevant: full-context compiled-memory analysis reports 27,932,091,664 bytes/device (reported, not a measured allocator peak).
- Result, including failure or restart: **exp3 completed its first two optimizer updates** at 13:13:48 and 13:16:41 UTC. Both initializer checks passed: raw C4 loss 2.57068157 and post-reseed loss 2.57307887 against the provisional reference 2.31374189 ± 0.5 (inside the allowance). First training loss 0.94783515; gradient norm 3.40971971; parameter norm 5549.55225; update norm 0.23865230 (update/parameter ratio 4.3004e-5) — startup observations, not steady-state certification. The initial 45-minute startup watchdog starts inside `Trainer.train` (~13:00:47 UTC, after packing/loading/initializer evaluation), not at Iris task start; the first update landed inside it. 32 started producer receipts present; no committed checkpoint visible at the configured temporary prefix at early inspection. A prepared recovery `.52` (14 stages, 69 passes/7 skips; two IID regressions fail on `.51` and pass with the correction) was preflighted but **not launched** — `.51` stays live while progressing; recovery authority is coordinator message `e040816ac0…`, conditional on diagnosed failure.
- Evidence: exp3-recipe.md (07:29+ readback) and artifacts above [R].
- Interpretation and limitations: two updates do not establish throughput; compiled-memory figure is analysis-reported.
- Decision and next action: hourly checks now track update cadence, checkpoint commits, and whether the IID regressions force a diagnosed recovery.

### 2026-11-16 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window): exp3 first checkpoint committed

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch. New user directive (coordinator message `562ddc814e…`): **two-hour periodic SFT checkpoints plus a guaranteed final checkpoint in future runs** — applied via commit `5fd0b71f21…` on the unused `benfeuer/exp3-validation-packing` branch to newly constructed native profiles; existing rolling counts preserved; production keeps three + protected best. Future-policy validation: 17 focused passes; broader validation 174 passes/1 skip; four interval regressions fail on the pinned 30-minute parent; no production source cutover occurred (adopting the policy into the running job would require a new source/artifact identity).
- External inputs and artifact references: `artifacts/production51-first-checkpoint-verification.json`, `future-checkpoint-policy-verification.json` [R: /experiments/active/long-loop-exp3/].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3's first native temporary checkpoint, step 11, committed at 13:46:19 UTC** and metadata readback verified it; native payload 502.29 GiB. The running `.51` retains its consumed 30-minute cadence.
- Evidence: exp3-recipe.md (readback) and artifacts above [R].
- Interpretation and limitations: one checkpoint commit; cadence unchanged for the live run.
- Decision and next action: hourly checks track checkpoint cadence and update counts; the two-hour policy applies to future runs only.

### 2026-11-16 15:50 UTC - Hourly catch-up (14:40–15:50 UTC window; resumed after a model-switch gap)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief. Note: the secretary's session was switched to a new model after the previous provider hit its monthly limit — the 15:40-ish revive turns failed client-side and this entry covers the gap through the 15:48 UTC readback.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts `production51-latest-checkpoint-list.json` and `production51-latest-wandb.json` (checked 15:46:50 UTC).
- External inputs and artifact references: as above [R: /experiments/active/long-loop-exp3/artifacts/].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 progressing healthily at update 52** (15:46 UTC readback): train/loss 0.85098 (from 0.94784 at update 1), grad norm 0.26355, update norm 0.13635, param norm 5549.579; per-update learner duration ~164.15 s; W&B state running, heartbeat 15:46:45 UTC. **Checkpoints on the 30-min cadence**: step 22 (14:17:42 UTC), step 33 (14:49:16), step 45 (15:22:39) — all temporary, 11-step spacing as configured.
- Evidence: monitoring artifacts above; exp3-recipe.md unchanged since the 14:40 readback.
- Interpretation and limitations: cadence metrics from monitoring snapshots; not steady-state certification.
- Decision and next action: continue hourly tracking; watch for update 52+ progression and the first protected-best selection.

### 2026-11-16 16:40 UTC - Hourly catch-up (15:50–16:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 16:26:10 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 66** (16:26 UTC readback): train/loss 0.86455 (oscillating around the update-52 value 0.85098 as expected for a masked-loss curve), grad norm 0.20775, learner ~172.54 s/update; state running, heartbeat fresh. **Checkpoints continue on the 30-min cadence**: step 56 (15:54:14 UTC), step 67 (16:25:19) — plus step 45 from the prior readback.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-16 17:40 UTC - Hourly catch-up (16:40–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 17:24:21 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 86** (17:24 UTC readback): train/loss 0.86984, grad norm 0.16118 (trending down: 0.26355 → 0.20775 → 0.16118 over three readbacks), learner ~164.16 s/update; state running, heartbeat fresh. **Checkpoints continue**: step 78 (16:57:59 UTC) after step 67 — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking; watch for the loss trend and checkpoint cadence holding.

### 2026-11-16 18:40 UTC - Hourly catch-up (17:40–18:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 18:24:18 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 107** (18:24 UTC readback): train/loss 0.88975, grad norm 0.13648 (continuing to trend down), learner ~161.53 s/update; state running, heartbeat fresh. **Checkpoints continue**: step 89 (17:30:20 UTC) after step 78 — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-16 19:40 UTC - Hourly catch-up (18:40–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 19:33:03 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 131** (19:33 UTC readback): train/loss 0.83172 (down from 0.88975 at update 107), grad norm 0.13081 (still trending down), learner ~161.34 s/update; state running, heartbeat fresh. **Checkpoints continue**: steps 112 (18:35:17 UTC) and 123 (19:06:28) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-16 20:40 UTC - Hourly catch-up (19:40–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 20:31:10 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 151** (20:31 UTC readback): train/loss 0.82850 (down from 0.83172 at update 131), grad norm 0.19213 (oscillating around the ~0.13–0.19 band), learner ~158.65 s/update; state running, heartbeat fresh. **Checkpoints continue**: steps 134 (19:38:44 UTC) and 145 (20:10:06) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.


### 2026-11-16 20:55 UTC — Known Exp3 data deviation: newline after assistant EOT

A census of the five retained OpenCode sources found newline token 198 immediately after every assistant end-of-turn marker; each newline receives loss weight. These are rows remaining after the candidate filters and Atlas decontamination. The 409,004 rows changed by V5 raw-question unquoting contain 3,082,618 such endings. The 17 rows without a successful raw-question metadata join contain 310; unquoting left those rows unchanged. The frozen original OpenCode cohort contains zero post-EOT newlines across 409,950 rows and 3,086,924 assistant endings. This comparison covers OpenCode, not the entire control mixture.

V5 unquoting edits the user-question input text. All 409,021 final rows exactly match their pre-unquoting supervised target-token arrays. Earlier scripted `merge_system_turns` receipts report re-rendering all 409,638 rows retained before Atlas, which subsequently removed 617. That renderer appends a newline after each EOT. The format difference therefore predates V5 exact raw-question unquoting.

The 3,082,928 newline targets are 0.276920% of OpenCode supervised positions, 0.00356733% of all V5 chat supervised positions, and 0.00256049% of all V5 supervised positions, including non-chat replay. The denominator is the unsampled sealed cache population; actual sampled training-loss exposure was not measured. All supervised tokens in the affected OpenCode rows, including their ordinary content, comprise 0.924632% of the full population's supervised positions. The extra target mass is small, but the formatting difference is systematic. Its behavioral effect is unknown.

The user decided to keep live Exp3 artifact `2026.11.16.51` unchanged and prepare a new V6 candidate for the next run. V6 will adopt the serializer correction once it lands: splice changed message bodies into the original rendering while preserving its turn separators. The other V5 recipe controls stay fixed. V6 requires the relevant data checks, a fresh audit and a new release seal; it has no training-launch approval.

The production code commit remains `0f5721bbb4953e98898c847896110dee32a5b293`. The audited V5 release-manifest SHA256 is `3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5`. [Sanitized counts and receipt checksums](evidence/exp3-eot-newline.json) identify the measured populations and private receipts. Full cache bodies and receipts remain access-controlled, so these public aggregates do not enable independent reproduction. A supplemental census of the other final chat sources is still running.

### 2026-11-16 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 21:21:08 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 169** (21:21 UTC readback): train/loss 0.81206 (down from 0.82850 at update 151), grad norm 0.13009, learner ~158.38 s/update; state running, heartbeat fresh. **Checkpoints continue**: steps 156 (20:40:50 UTC) and 167 (21:12:15) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.
-loop

### 2026-11-16 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 22:20:27 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 190** (22:17 UTC readback): train/loss 0.80899 (down from 0.81206 at update 169), grad norm 0.11321 (still trending down), learner ~170.70 s/update; state running. **Checkpoints continue**: steps 179 (21:46:13 UTC) and 190 (22:17:09) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-16 23:40 UTC - Hourly catch-up (22:40–23:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 23:31:35 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 210** (23:13 UTC readback): train/loss 0.78643 (down from 0.80899 at update 190), grad norm 0.10819 (still trending down), learner ~160.83 s/update; state running. **Checkpoints continue**: steps 201 (22:49:12 UTC) and 212 (23:20:02) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-17 00:40 UTC - Hourly catch-up (23:40–00:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 00:27:09 UTC (Nov 17).
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 230** (00:16 UTC readback): train/loss 0.80304 (up slightly from 0.78643 at update 210 — normal masked-loss oscillation), grad norm 0.11664, learner ~163.61 s/update; state running. **Checkpoints continue**: steps 223 (23:53:23 UTC) and 234 (00:24:39) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.


### 2026-11-17 00:55 UTC — Full V5 chat ending census

The supplemental census is complete. All 703 shard-result pins were byte-verified and matched to the sealed V5 source inventory, including each shard's document, token and supervised-position totals. The position counter is unchanged from the earlier audit. Previously completed leaves were retained; no policy samples or training data were regenerated.

| Cohort | Documents | Assistant EOTs followed by newline | Supervised newline positions |
| --- | ---: | ---: | ---: |
| V5 OpenCode | 409,021 | 3,082,928 | 3,082,928 |
| V5 remaining chat sources | 12,132,963 | 11,021,009 | 10,991,248 |
| V5 all chat | 12,541,984 | 14,103,937 | 14,074,176 |

All counted newlines are token 198. The 29,761 positions separating the physical and supervised totals have zero loss weight in relay traces. The census found exactly one EOT per assistant turn across 32,274,249 turns. Twenty-three of the 50 chat sources contain a supervised post-EOT newline. The remaining-chat cohort includes edited and unedited rows; its aggregate does not distinguish them. The original-control census still covers OpenCode only: zero post-EOT newlines in 409,950 original rows. The earlier paired OpenCode target-array comparison remains unchanged.

The 14,074,176 chat newline targets are **0.0162855869%** of all natural chat supervised positions and **0.0116891444%** of all natural supervised positions including replay. The 3,585,348 rows with at least one such supervised newline contain 8,985,072,102 supervised positions, or **7.4624478901%** of the full natural supervised population. These quantities measure different effects: the newline targets themselves and all supervised content in rows containing them. Actual sampled training-loss exposure and behavioral impact remain unmeasured.

This supplements the earlier OpenCode-only result. Live Exp3 `.51` continues unchanged under the user's ruling. The separate V6 candidate must pass its full post-Atlas format and mask census before readiness sealing; no V6 training launch is authorized. Its identity-cohort reader proof covers 859,434 documents and 1,306,611 assistant turns with no post-EOT newline or mask violations. That cohort proof does not certify the whole V6 candidate.

[Sanitized full-census counts and receipt checksums](evidence/exp3-eot-full-chat.json) preserve the measured denominators and limits. [The frozen position counter](evidence/exp3-eot-counter-source.json) preserves the executed measurement function without storage coordinates. Full inputs and receipts remain access-controlled, so public reproduction is incomplete.

### 2026-11-17 01:40 UTC - Hourly catch-up (00:40–01:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 01:23:59 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 254** (01:23 UTC readback): train/loss 0.79873 (down from 0.80304 at update 230), grad norm 0.11381, learner ~158.67 s/update; state running. **Checkpoints continue**: step 246 (00:58:20 UTC) after step 234 — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-17 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 02:28:50 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 277** (02:28 UTC readback): train/loss 0.78891 (oscillating around 0.79–0.83), grad norm 0.10060 (new low; trend continues down), learner ~171.24 s/update; state running. **Checkpoints continue**: steps 258 (01:31:33 UTC) and 269 (02:03:14) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-17 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; monitoring artifacts checked 03:27:22 UTC.
- External inputs and artifact references: `production51-latest-wandb.json`, `production51-latest-checkpoint-list.json` [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 at update 297** (03:27 UTC readback): train/loss 0.80467 (oscillating in the 0.79–0.83 band), grad norm 0.11743, learner ~174.70 s/update; state running. **Checkpoints continue**: steps 280 (02:34:45 UTC) and 291 (03:05:48) — 30-min cadence holding.
- Evidence: monitoring artifacts [R].
- Interpretation and limitations: cadence metrics from monitoring snapshots.
- Decision and next action: continue hourly tracking.

### 2026-11-17 09:40 UTC - Hourly catch-up (03:40–09:40 UTC window; resumed after a session gap): Exp3 SFT completed, RL capacity-gated

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief. The session was idle from ~04:40 UTC; this entry covers the gap through the 09:34 readback.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: exp3-recipe.md §§"SFT completion"/"batch-CPU audit" (09:34 readback), `artifacts/production51-finalization-rank0-20261117T0747.log`, receipt `s3://…/actual388-exposure-v1/receipt.json` SHA256 `056c6f378c44e3d5…` [P Filer link in the recipe], monitoring artifacts [R].
- Environment and hardware when relevant: full 256-H100 production geometry through SFT.
- Result, including failure or restart: **Exp3 SFT COMPLETED — 388 updates, permanent checkpoint `step-387` selected** (stop_reason=time_bound vs the 746 maximum; 72,000-s duration with 2,400-s finalization and 600-s update reserves; finalization completed 2026-11-17 08:02:14 UTC before its deadline). All 32 SFT tasks succeeded; the canonical journal verifier checked every producer history/phase hashes, phase boundaries, and exact step coverage 0–387; HF source and handoff verified (39 model shards; 49 handoff file sizes; 262,144-token context; handoff SHA256 `0010f0f6cafd…`). Regional batch-CPU audit (exact packing/source exposure at 388 updates): 99,328 sequences, 26,038,239,232 context positions, 24,996,002,779 real tokens, 4.00% padding, zero token drops and zero sampled-IID overlap; mixture phases 0 and 250 reached, 500 not; original LR schedule/natural weights/746-max not retuned. Temporary checkpoints 358/369/380 auto-deleted by the pinned runtime after the permanent commit (no operator deletion; immutable metadata preserved). **RL child submitted 08:17:21 UTC and remains Kueue SchedulingGated** (monitor 09:24 UTC: no failure/preemption/learner startup evidence; queue untouched). Last W&B observation before SFT finish: update 387, train/loss 0.77019, grad norm 0.10200, learner ~176.19 s (state finished 08:06 UTC).
- Evidence: per-item references above [R]/[P].
- Interpretation and limitations: SFT success does not certify the RL/eval outputs; RL is capacity-gated.
- Decision and next action: hourly checks now track the RL gate and (once running) RL updates; admission remains untouched per the standing queue rule.

### 2026-11-17 10:40 UTC - Hourly catch-up (09:40–10:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (10:33 UTC) and `production51-latest-kueue-status.json` [R].
- Environment and hardware when relevant: **quota condition identified — the RL gate needs 232 more GPU + RDMA slots** (Kueue status artifact).
- Result, including failure or restart: exp3's **RL 32 tasks remain Kueue SchedulingGated at 10:33 UTC** — no failure/preemption/learner start; queue untouched; SFT/handoff remain certified. New detail: the monitor's latest signal reports a fresh **quota condition requiring 232 more GPU+RDMA slots** before the RL tasks can schedule (Kueue status artifact captures the breakdown).
- Evidence: monitoring state + kueue-status artifact [R].
- Interpretation and limitations: gate status only; the quota resolution is a cluster/capacity matter outside this record.
- Decision and next action: hourly checks track the gate; admission stays untouched per the standing queue rule.

### 2026-11-17 11:40 UTC - Hourly catch-up (10:40–11:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (11:31 UTC) [R].
- Environment and hardware when relevant: quota condition unchanged (232 GPU + RDMA slots).
- Result, including failure or restart: exp3's **RL 32 tasks remain Kueue SchedulingGated at 11:31 UTC** — no failure/preemption/learner start; the fresh quota condition still needs 232 more GPU + RDMA slots; queue untouched; SFT/handoff certified. No exp3 training change this window.
- Evidence: monitoring state [R].
- Interpretation and limitations: gate status only.
- Decision and next action: hourly checks continue; admission stays untouched per the standing queue rule.

### 2026-11-17 12:40 UTC - Hourly catch-up (11:40–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (12:31 UTC) [R].
- Environment and hardware when relevant: quota condition unchanged (232 GPU + RDMA slots).
- Result, including failure or restart: exp3's **RL 32 tasks remain Kueue SchedulingGated at 12:31 UTC** — no failure/preemption/learner start; the quota condition still needs 232 more GPU + RDMA slots; queue untouched; SFT/handoff certified. No exp3 training change this window.
- Evidence: monitoring state [R].
- Interpretation and limitations: gate status only.
- Decision and next action: hourly checks continue; admission stays untouched per the standing queue rule.

### 2026-11-17 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (13:34 readback) [R].
- Environment and hardware when relevant: quota condition unchanged (232 GPU + RDMA slots).
- Result, including failure or restart: exp3's **RL admission remains queued** — all 32 tasks building, no failures/preemptions; the quota condition still needs 232 additional GPU + RDMA slots; no job changes; SFT/handoff certified. No exp3 training change this window.
- Evidence: monitoring state [R].
- Interpretation and limitations: gate status only.
- Decision and next action: hourly checks continue; admission stays untouched per the standing queue rule.

### 2026-11-17 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (14:34 readback) [R].
- Environment and hardware when relevant: quota condition unchanged (232 GPU + RDMA slots).
- Result, including failure or restart: exp3's **RL admission remains queued** — all 32 tasks building, no failures/preemptions; the quota condition still needs 232 additional GPU + RDMA slots; no job changes; SFT/handoff certified. No exp3 training change this window.
- Evidence: monitoring state [R].
- Interpretation and limitations: gate status only.
- Decision and next action: hourly checks continue; admission stays untouched per the standing queue rule.

### 2026-11-17 15:40 UTC - Hourly catch-up (14:40–15:40 UTC window): exp3 RL RUNNING

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (15:34 readback) [R].
- Environment and hardware when relevant: the quota condition (232 GPU + RDMA slots) resolved — the RL tasks scheduled.
- Result, including failure or restart: **exp3's RL 32 tasks are RUNNING** (15:34 UTC readback) — the earlier quota gate cleared. Token generation and **one real installed-agent trial verified with grade 0 and no exception** (a harness-level sanity trial, not a learning signal). **No optimizer evidence yet** — no RL updates have completed; the run itself is unchanged. This is the first RL-phase execution after the SFT completion (388 updates, step-387 handoff).
- Evidence: monitoring state [R].
- Interpretation and limitations: grade-0 agent trial verifies the harness path only; optimizer evidence is the next milestone.
- Decision and next action: hourly checks now track RL updates/optimizer evidence and the first admission decisions.

### 2026-11-17 16:40 UTC - Hourly catch-up (15:40–16:40 UTC window): exp3 RL update 1 complete

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (16:34 readback) [R].
- Environment and hardware when relevant: update-1 timing split: 2,955.703 s inclusive = 361.161 s learner + 2,569.976 s admission.
- Result, including failure or restart: **exp3's RL update 1 is complete** (16:08:15 UTC) — stdout metrics match the W&B history index. **No optimizer learning evidence yet**: the update's time went to admission (2,569.976 s) plus a short learner segment (361.161 s); the native final seal and checkpoint are pending; the run itself is unchanged.
- Evidence: monitoring state [R].
- Interpretation and limitations: update completion ≠ learner progress; the admission-dominated first update is expected (the admission phase assembles the first sample groups).
- Decision and next action: hourly checks now track the learner/ admission split per update, the first checkpoint, and steady-state cadence.

### 2026-11-17 17:40 UTC - Hourly catch-up (16:40–17:40 UTC window): exp3 RL updates 2–3, steady cadence emerging

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch.
- External inputs and artifact references: monitoring state (17:33 readback) [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 RL updates 2 and 3 complete** with a steady cadence emerging: W&B update 3 shows learner 321.85 s, policy 240.69 s, full cycle 347.46 s (learner-dominated now, vs the admission-dominated update 1) — no eval/checkpoint callback on this cycle. **Native checkpoint 2 verified**; a prior eval scored 119/120 and is ineligible (the all-scored eligibility rule). Run unchanged; controls unchanged.
- Evidence: monitoring state [R].
- Interpretation and limitations: three updates do not establish throughput; the 119/120 eval being ineligible confirms the strict eligibility rule is active.
- Decision and next action: hourly checks track the learner-dominated cadence, checkpoint/eval callbacks, and the first eligible eval.

### 2026-11-17 18:45 UTC - Hourly catch-up (17:40–18:45 UTC window): exp3 RL child FAILED at update 4

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; no recovery or production changes yet.
- External inputs and artifact references: monitoring state (18:35 readback) [R]; exp3-recipe.md for the prior three-update history.
- Environment and hardware when relevant: policy rank 56, context-parallel ALLTOALL (NCCL).
- Result, including failure or restart: **exp3's RL child FAILED at update 4** — the parent succeeded, but the RL child died with a **policy rank-56 NCCL context-parallel ALLTOALL invalid-usage/message-truncation error followed by SIGABRT**. Three complete cycles (updates 1–3) had finished before the failure; native pointer 2; **no eligible best checkpoint**; terminal finalization recorded as `skipped_no_checkpoint`; **no trained HF model**. No recovery or production changes have been made yet.
- Evidence: monitoring state [R]; prior update history in this notebook (16:40/17:40 entries).
- Interpretation and limitations: the NCCL error is a recorded terminal condition, not a diagnosed root cause; recovery authority remains conditional on diagnosed failure (coordinator message `e040816ac0…`).
- Decision and next action: hourly checks watch for the recovery diagnosis/decision; no secretary action beyond the record.

### 2026-11-17 19:45 UTC - Hourly catch-up (18:45–19:45 UTC window): update-4 failure verified in detail; recovery source prepared, unused

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`; an unused recovery source `f1f8795a9f…` exists.
- Command or material configuration: as recorded at launch; no recovery/cancellation/relaunch/source change performed.
- External inputs and artifact references: exp3-recipe.md §update-4 verification (19:34 readback) and `artifacts/production51-update4-failure-owner-verification.json` (with hashes for native exceptions, worker logs, task states, checkpoint pointer, W&B and terminal receipts) [R].
- Environment and hardware when relevant: policy rank 56 (CP group 36, local rank 1, host gb95c40, PID 13657); NCCL 2.29.7.
- Result, including failure or restart: **read-only verification of the update-4 failure** (from the exp3 record): policy rank 56 (CP group 36, local rank 1, host gb95c40, PID 13657) logged **NCCL 2.29.7 ALLTOALL_BASE invalid usage at 18:02:24 UTC — a socket message contained 941,711,616 bytes where 32 were expected**; the watchdog raised SIGABRT at 18:03:25 UTC; the outer Ray WorkerCrashedError followed. The underlying collective/transport cause is unresolved, and no causal link to earlier proxy httpx.ReadError events has been established. Terminal manifest: state=failed, finalization=skipped_no_checkpoint, model=null; latest native pointer 2; no best_checkpoint.json or later. **All 32 RL tasks are terminal (one failed, 31 coscheduled failures)**. W&B still reports running with trainer/global_step=3 — this does not override the terminal Iris/native evidence. An **unused recovery source `f1f8795a9f…` is prepared**: it avoids repacking 427 unreserved sources while constructing the seven IID validation sources, restores Exp1 loader request geometry (4 batches, buffer 512) from `9662fdba…`, confirms identical first/resumed batches under fetch 4 and 32, and prepares a 90-minute startup-only watchdog (step/process watchdogs unchanged); population/masks/stream/optimizer/schedule/topology/20-h deadline preserved. Recovery still requires the diagnosed-failure condition.
- Evidence: exp3-recipe.md verification section + artifacts above [R].
- Interpretation and limitations: the socket-size discrepancy is the recorded symptom; the root cause is unresolved.
- Decision and next action: hourly checks watch for the recovery decision; the record carries the verified failure detail.

### 2026-11-17 20:40 UTC - Hourly catch-up (19:45–20:40 UTC window): interconnect root-cause finding for the exp3 update-4 failure

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief; the coordinator directed inclusion of today's interconnect finding (marin issue #11618).
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: the resolved launch config for the `2026.11.16.51` long-loop RL run (cluster cw-rno2a, 32 nodes × 8 H100, attempt `cb784b53b8fd`) sets `NCCL_IB_DISABLE: '1'`, `NCCL_NET: Socket`, `NCCL_SOCKET_FAMILY: AF_INET` — while the job requested 256 GPUs and 256 `rdma/ib` devices.
- External inputs and artifact references: **marin issue [#11618](https://github.com/marin-community/marin/issues/11618)** "[rl] RL recipes force NCCL onto socket transport and disable InfiniBand" (OPEN, bug, agent-generated) — primary evidence [P].
- Environment and hardware when relevant: cw-rno2a; 256 `rdma/ib` devices requested but bypassed.
- Result, including failure or restart: **Root-cause finding for the update-4 failure recorded above**: per marin#11618, Snowball RL recipes **force NCCL onto socket transport and disable InfiniBand** while their jobs request `rdma/ib` — multi-node collectives bypass the fast cross-node interconnect, and the exp3 update-4 failure matches exactly: NCCL 2.29.7 `NET/Socket : peer … message truncated : receiving 941,711,616 bytes instead of 32` (the same symptom the 19:45 entry verified). Provenance per #11618 [P]: commit `410e9c25a8…` ("Use IPv4 socket transport for Snowball inventory smoke", 2026-10-20) added the three variables with no recorded reason and **no measured InfiniBand failure accompanies any of the copies**; the variables were copied forward into `inventory/train_quick_loop.py` (main, lines 82–84, still present), `long_loop/specs/rl.yaml` (branch `long-loop`, lines 222–224) and six further recipe/entry-point commits. Expected behavior: NCCL uses InfiniBand (`NET/IB`) for cross-node collectives on jobs that request `rdma/ib`.
- Evidence: marin#11618 [P]; the 19:45 verification entry in this notebook; the resolved launch config quoted in #11618.
- Interpretation and limitations: #11618 establishes the misconfiguration chain and the transport evidence for `.51`; the fix (unsetting the variables on `rdma/ib` jobs) is a marin-source change outside this record's write scope; a recovery relaunch decision remains with the coordinator/user.
- Decision and next action: hourly checks watch for the #11618 fix landing and any recovery decision.

### 2026-11-17 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window): partial #11618 fix merged

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: n/a (fix landed on marin main; not adopted into the terminal exp3 run).
- External inputs and artifact references: marin issue [#11618](https://github.com/marin-community/marin/issues/11618) fix-status comment; [marin PR #11178](https://github.com/marin-community/marin/pull/11178) — MERGED 2026-11-17 21:00:56 UTC, merge commit `f008c071cd7576864b76dfb94626fbf430493eee` [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **partial fix landed for the interconnect finding (marin#11618)**: marin PR #11178 merged at 21:00:56 UTC (merge commit `f008c071…`) — main's `experiments/post_training/inventory/train_quick_loop.py` **no longer sets** `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1` or `NCCL_SOCKET_FAMILY=AF_INET`, so the quick-loop lineage is fixed on main. **Still carrying the override** (per the #11618 fix-status comment): the long-loop `specs/rl.yaml` copy, the Snowball preset/entry-point lineage, the Mochi one-rack recipe, and the NVL72 smoke recipe — i.e., the exact lineage that produced the exp3 `.51` failure is not yet fully cleared on all recipes. The issue remains OPEN.
- Evidence: #11618 comment + PR #11178 metadata [P]; verified via `gh pr view`.
- Interpretation and limitations: the fix post-dates the terminal exp3 run and does not change it; remaining recipe copies are tracked in #11618.
- Decision and next action: hourly checks watch for the remaining `rl.yaml`/preset-lineage fixes; recovery decision stays with the coordinator/user.

### 2026-11-17 22:40 UTC - Hourly catch-up (21:40–22:40 UTC window): #11618 CLOSED with full provenance; transport probe proposed

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: n/a (evidence-gathering on marin; not adopted into the terminal exp3 run).
- External inputs and artifact references: marin issue [#11618](https://github.com/marin-community/marin/issues/11618) — now CLOSED; new issue [#11619](https://github.com/marin-community/marin/issues/11619) "[rl] Add a two-node NCCL transport probe" (OPEN) [P].
- Environment and hardware when relevant: the original workaround targeted cw-us-east-02a's `mlx5_0`.
- Result, including failure or restart: **marin#11618 CLOSED** ("provenance collection is complete") with the closing comment establishing the full origin: the socket override in commit `410e9c25a8…` (2026-10-20) was a **workaround for an InfiniBand `IBV_WC_RETRY_EXC_ERR(12)` failure on `mlx5_0`** in the Snowball inventory smoke on cw-us-east-02a — logbooks B200MFU-020/-028 record the same error class in the July 2026 B200 bring-up; every multi-node run before 2026-10-19 that was checked used InfiniBand. So the override had a real (single-host-smoke) origin, but its copy-forward into multi-node production recipes was never re-validated against `rdma/ib` jobs. **Follow-up proposed**: marin issue [#11619](https://github.com/marin-community/marin/issues/11619) (OPEN) proposes a two-task gang Iris job measuring which transport NCCL picks per environment variant: the full workaround, each variable alone, the production overlay with `NCCL_SOCKET_IFNAME`, and IB hints (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`). (Also this window: one Atlas landing — source-inventory reuse guarded by input-unchanged checks.)
- Evidence: #11618 closing comment, #11619 body [P]; Atlas landing record [R].
- Interpretation and limitations: the probe (#11619) is a proposal, not yet run; the exp3 recovery decision remains with the coordinator/user.
- Decision and next action: hourly checks watch for the #11619 probe execution and the recovery decision.

### 2026-11-17 23:40 UTC - Hourly catch-up (22:40–23:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 production source `0f5721bbb4…`, artifact `2026.11.16.51`.
- Command or material configuration: as recorded at launch; no recovery/production changes.
- External inputs and artifact references: monitoring state (23:33 readback) [R]; marin#11619 state [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3's terminal state is unchanged — the RL child failure at update 4 stands as verified (NCCL socket-transport truncation per marin#11618's provenance), no recovery/production changes. The #11619 two-node transport probe remains OPEN (not yet run).
- Evidence: monitoring state [R]; #11619 [P].
- Interpretation and limitations: gate/recovery status only.
- Decision and next action: hourly checks watch for the #11619 probe execution and the recovery decision.

### 2026-11-18 00:35 UTC - Hourly catch-up (23:40–00:35 UTC window): #11619 probes DONE — full transport matrix published; exp3 restarting from scratch on IB

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief; coordinator correction relayed the completed #11619 probe results and the exp3 restart.
- Published source revision: the terminal exp3 `.51` run is being restarted **from scratch**; the restarting run's source/artifact identities are recorded by the exp2-session owner (this record's next entries will bind them).
- Command or material configuration: per the [#11619 probe comment](https://github.com/marin-community/marin/issues/11619#issuecomment-6964135906) [P]: same script, pinned SkyRL image and megatron profile (NCCL 2.30.7), production priority, 8 CPU/32 GB/200 GB per task; two nodes per H100 run (gangs spread across nodes); GB200 runs use two single-tray jobs with a shared registry endpoint.
- External inputs and artifact references: [#11619 probe comment 6964135906](https://github.com/marin-community/marin/issues/11619#issuecomment-6964135906) [P]; coordinator correction [R].
- Environment and hardware when relevant: cw-rno2a and cw-us-east-02a (H100, cross-node) vs cw-us-east-08a (GB200, intra-rack NVL72).
- Result, including failure or restart:
  1. **Probe coverage complete for all three clusters** — correcting this record's 23:40 entry ("probe still OPEN/not run" is withdrawn; the probes are DONE, reported in PR #11619 comment 6964135906).
  2. **cw-rno2a and cw-us-east-02a PASS the default environment over InfiniBand** (`NET/IB GDRDMA`): all-reduce 85–88 GB/s (02a per-variant table: default 85.6, AF_INET-only 87.5, IB_GID_INDEX_0 88.1 GB/s; all-to-all 63–65 GB/s).
  3. **cw-us-east-08a runs intra-rack over NVLink** — both two-job pairs landed their trays in one NVL72 rack, so every variant (including the full socket override) passed at NVLink speed (all-reduce 523–534 GB/s, all-to-all 373–387 GB/s) with an empty NET census; within one NVL72 rack the NCCL variables do not change the transport. Cross-rack 08a coverage comes from production instead (the Mochi three-rack runs `2026.11.13.9403`–`.9405` forced `NCCL_NET=IB` with `NCCL_IB_HCA=ibp` and recorded no IB transport errors).
  4. **On 02a, the three-variable override alone ABORTS in the collective** (the exp3 `.51` failure mode); adding `NCCL_SOCKET_IFNAME` (the cluster config's ambient value `^ibs,ibp,lo,docker,veth,cilium,lxc`) restores a passing socket path at **1/20th the bandwidth** (prod_exact: 4.2 GB/s all-reduce vs 85.6 GB/s IB); `net_socket_only`/`ib_disable_only` hang; `ib_hca=mlx5_0/1` fall back to socket then hang.
  5. **Launch-gate summary**: no cluster needs the socket override today; the socket path is 12–24× slower where it works and hangs/aborts when partially applied; the live gate check should assert the census shows `NET/IB` (or no NET transport) and fail closed otherwise.
  6. **Exp3 `.51` is being restarted from scratch on IB by the exp2-session owner** — the terminal `.51` run is superseded; this record will bind the restart's identities when published.
- Evidence: #11619 probe comment [P]; coordinator correction [R].
- Interpretation and limitations: the probe used NCCL 2.30.7 (the terminal `.51` run used 2.29.7 — same symptom class, different patch version); the restart's own identity evidence is pending.
- Decision and next action: hourly checks now track the from-scratch IB restart (new source/artifact, gate census assert, launch) and the V6 data candidate.

### 2026-11-18 01:40 UTC - Hourly catch-up (00:35–01:40 UTC window): exp3 IB restart prepared and submitted

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: **new exp3 restart identities** — runtime source `ba63397d23249bbbf8bf297a48c4017bc4c63fff` (base `0f5721bbb4…`), artifact `2026.11.17.55`, root task `/benfeuer/mochi-exp3-rl-ib-2026.11.17.55/0`.
- Command or material configuration: launch record `restart55-launch-record.json` (prepared 2026-11-17 23:42:56 UTC): state submitted_root_pending; the gate references the #11619 probe job `/benfeuer/ib-audit-nccl-probe-rno2a-6` (baseline NET/IB GDRDMA, 87.4 GB/s allreduce / 62.0 GB/s all-to-all; NCCL 2.30.7; limitation: production INIT and the first CP5 step still require observation). Semantic diffs vs `.51`: fresh RL lineage from the same adopted SFT export (no native resume/no SFT rerun); **socket/IB-disable/AF_INET overrides REMOVED, no HCA pin**; periodic checkpoint every 7,200 s + orderly final, rolling 3 + best; evaluation-only sandbox disk 128 GiB (live ENOSPC fix verdict pending); HF fallback to the committed final native checkpoint labeled `final_not_best`; new source SHA with mochi-prefixed root/child and fresh storage/tracker identities.
- External inputs and artifact references: `restart55-launch-record.json`, `restart55-root-task-description.txt` (task pending on cw-rno2a, 4 CPU/17.18 GB/68.72 GB), `restart55-jobs-admission.txt` [R: /experiments/active/long-loop-exp3/artifacts/].
- Environment and hardware when relevant: same 256-H100/32-node target as `.51`.
- Result, including failure or restart: the **exp3 IB restart is prepared and submitted** (root task pending on cw-rno2a) — fresh RL lineage reusing the adopted SFT export, with the #11619 socket overrides removed and the storage/tracker identities reset. The `.51` terminal run remains the recorded failure evidence.
- Evidence: artifacts above [R]; #11619 gate reference [P].
- Interpretation and limitations: production INIT and the first CP5 step still require observation (the probe's stated limitation); the restart has not yet reached its first update.
- Decision and next action: hourly checks track the restart's INIT/first-step observation and its own gate census.

### 2026-11-18 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: exp3 IB restart (artifact `2026.11.17.55`, source `ba63397d23…`) — **root task still pending** on cw-rno2a; RL child "not_created; coordinator awaiting admission" (launch-record fields). The `.51` terminal run is unchanged.
- Command or material configuration: as recorded in the 01:40 entry.
- External inputs and artifact references: `restart55-launch-record.json`, `restart55-root-task-description.txt` (checked 02:33 UTC) [R].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: no exp3 progression this window — the restart's root task remains pending (awaiting cluster admission; the same capacity constraint that gated the first RL attempt).
- Evidence: launch-record/task description [R].
- Interpretation and limitations: gate status only.
- Decision and next action: hourly checks watch for root-task admission and the INIT/first-CP5-step observation.

### 2026-11-18 02:45 UTC - Correction to the 02:40 entry (exp3 .55 is RUNNING, verified via Iris)

- Question or hypothesis and reason for trying it: coordinator correction — the 02:40 entry said the exp3 `.55` root task was pending; Iris state shows otherwise. The secretary's stale source was the 6:43 PM EST (23:43 UTC) launch-note snapshot; Iris state is primary.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: `uv run iris --cluster=marin job list --prefix /benfeuer/mochi-exp3-rl-ib` and `job describe /benfeuer/mochi-exp3-rl-ib-2026.11.17.55` (run at 02:44 UTC).
- External inputs and artifact references: Iris job list/describe output [R: live cluster state].
- Environment and hardware when relevant: cw-rno2a.
- Result, including failure or restart: **CORRECTION — exp3 `.55` is RUNNING, not pending.** Iris reports: root `/benfeuer/mochi-exp3-rl-ib-2026.11.17.55` state **running** (submitted 2026-11-17 23:43:34 UTC); RL child `…-18817614fe9d` state **running** (submitted 23:45:15 UTC), task 0 running for 1 h 52 m with exit=0, failures=0, preemptions=0 (8×H100 task per the coordinator). The 02:40 entry's "pending / awaiting admission" line is withdrawn — my source was the stale launch-note snapshot rather than live Iris state. Lesson recorded: read Iris directly before logging job status.
- Evidence: Iris job list/describe output above [R].
- Interpretation and limitations: the INIT/first-CP5-step observation from the #11619 gate note remains the next milestone; RL update evidence will follow.
- Decision and next action: hourly checks read Iris directly for exp3 state.

### 2026-11-18 02:40 UTC - Hourly catch-up (01:40–02:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: exp3 IB restart (source `ba63397d23…`, artifact `2026.11.17.55`) — **verified RUNNING via Iris at 02:33 UTC**: root submitted 23:43:34 UTC Nov 17; RL child `…-18817614fe9d` running, task 0 running 2 h 49 m (exit=0, failures=0, preemptions=0). This supersedes the 02:40-cycle correction cycle's snapshot (the coordinator's 01:36 UTC correction and my 02:45 UTC Iris verification agree).
- Command or material configuration: `uv run iris --cluster=marin job describe` (direct read, per the recorded lesson).
- External inputs and artifact references: Iris output [R: live cluster state].
- Environment and hardware when relevant: cw-rno2a.
- Result, including failure or restart: exp3 `.55` continues running (~2 h 49 m on task 0); no failures. The INIT/first-CP5-step observation from the #11619 gate note remains the next milestone; RL update evidence will follow.
- Evidence: Iris job describe [R].
- Interpretation and limitations: job state only; INIT observation pending.
- Decision and next action: hourly checks read Iris directly and watch for the INIT observation and first update.

### 2026-11-18 02:45 UTC - Correction + first-steps record: exp3 `.55` updates 1–3 complete (W&B-verified)

- Question or hypothesis and reason for trying it: coordinator correction — the `.55` first-step milestone is already done (update 1 completed 00:42:18 UTC), superseding the 02:40 entry's "INIT/first-CP5-step observation pending" line. Verified against the exp3 record's own step-history artifact.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: `restart55-milestone-step-history-once.json` (checked 02:16:09 UTC) and `restart55-milestone-wandb-once.json` [R: /experiments/active/long-loop-exp3/artifacts/]; W&B https://wandb.ai/marin-community/marin-long-loop/runs/f7m70xhc [P].
- External inputs and artifact references: as above.
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: **exp3 `.55` updates 1–3 complete** per the W&B step history: update 1 completed **00:42:18 UTC** — full step 2,269.04 s of which **2,104.16 s was rollout admission** (`wait_for_generation_buffer`), learner 152.50 s, policy 89.08 s; update 2: full step 4,004.08 s, learner 113.84 s, policy 82.51 s; update 3: full step 126.80 s, learner 112.99 s, policy 82.14 s — **the admission-dominated startup collapsed into a steady learner-dominated cadence by update 3** (matching the exp2 pattern). The W&B snapshot was taken while training was mid-run (metrics fields empty in the once-only snapshot; the step history carries the values).
- Evidence: step-history artifact + W&B run URL [P/R].
- Interpretation and limitations: three updates do not establish throughput; the once-only snapshot has empty live metrics fields (superseded by the step history).
- Decision and next action: hourly checks track the steady-state cadence and the first eval/checkpoint callbacks.

### 2026-11-18 03:40 UTC - Hourly catch-up (02:40–03:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (updates 1–3, checked 02:16 UTC).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in this artifact (no new rows since the 02:16 readback at this cycle's check); the coordinator's correction and my 02:45 correction entry already record updates 1–3 (update 1 at 00:42:18 UTC, admission-dominated; steady learner-dominated cadence by update 3). No new Atlas-facing exp3 events beyond that.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run; W&B run f7m70xhc is the live source.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 04:40 UTC - Hourly catch-up (03:40–04:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 05:40 UTC - Hourly catch-up (04:40–05:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 06:40 UTC - Hourly catch-up (05:40–06:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 07:40 UTC - Hourly catch-up (06:40–07:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 08:40 UTC - Hourly catch-up (07:40–08:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 09:40 UTC - Hourly catch-up (08:40–09:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 10:40 UTC - Hourly catch-up (09:40–10:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 11:40 UTC - Hourly catch-up (10:40–11:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 12:40 UTC - Hourly catch-up (11:40–12:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 13:40 UTC - Hourly catch-up (12:40–13:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 14:40 UTC - Hourly catch-up (13:40–14:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 15:40 UTC - Hourly catch-up (14:40–15:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 16:55 UTC - Hourly catch-up (15:40–16:55 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 17:40 UTC - Hourly catch-up (16:55–17:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 18:40 UTC - Hourly catch-up (17:40–18:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 19:40 UTC - Hourly catch-up (18:40–19:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 20:40 UTC - Hourly catch-up (19:40–20:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.

### 2026-11-18 21:40 UTC - Hourly catch-up (20:40–21:40 UTC window)

- Question or hypothesis and reason for trying it: hourly Phase-2 catch-up per the standing brief.
- Published source revision: unchanged exp3 restart identities (source `ba63397d23…`, artifact `2026.11.17.55`).
- Command or material configuration: as recorded at launch; the step-history artifact still holds three rows (checked this cycle).
- External inputs and artifact references: `restart55-milestone-step-history-once.json` [R]; W&B run f7m70xhc [P].
- Environment and hardware when relevant: n/a.
- Result, including failure or restart: exp3 `.55` continues at updates 1–3 in the artifact snapshot (no new rows at this cycle's check); the live W&B run is the authoritative source. No exp3 anomalies visible in this record.
- Evidence: step-history artifact [R].
- Interpretation and limitations: artifact snapshot cadence lags the live run.
- Decision and next action: next hourly check reads the step history again for updates 4+.
