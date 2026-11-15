# Selection ledger — Snowball long-loop exp1 (best16) and exp2 (best20)

- **What it is:** per-checkpoint validation ledger (native step, timestamp, attempted/scored counts, raw verifier mean, trailing-5 adjusted EMA, eligibility, selected-best flag, save status) for Experiment 2 (round 2) and the findable Experiment 1 (round 1) equivalents.
- **Compiled at:** 2026-11-15T15:14Z
- **Sources:** [R] `/Users/benfeuer/Documents/experiments/active/sft-new-mix-candidate/STATE.md`; [R] `.../sft-new-mix-candidate/artifacts/experiment2/recovery13-eval{8,10,12,14,16,18}-*milestone*.json`, `recovery13-eval4-ineligible-decision-20261114.json`, `recovery13-eval4-aggregate-check-20261114T1958Z.json`, `recovery13-eval2-native-error-census.json`, `recovery13-rl-compute-complete-best20-milestone-20261115.json`, `recovery13-selection-after-eval6-20261114T2102Z.json`; [R] `/Users/benfeuer/Documents/experiments/complete/long-loop-prototype/STATUS.md`; [P] W&B runs referenced below (public dashboard URLs).
- **Access class:** aggregate metrics, step numbers, timestamps and URIs only. No task text, prompts, model outputs or dataset rows. Verifier/task state lives in the sealed private archives (see final-eval-attempts.md).

## Selection rule (exp2, consumed config)

Selection metric `eval/all/avg_verifier_score`; **adjusted trailing-5 EMA, alpha 1/3, evaluation interval 2; highest eligible metric; earliest checkpoint on ties** [R: `recovery13-rl-compute-complete-best20-milestone-20261115.json`, `recovery13-checkpoint14-writing-selection-review-20261115.json`]. First complete eligible window is steps 6/8/10/12/14 [R: checkpoint14-writing-selection-review].

## Exp2 RL validation ledger (run `/benfeuer/long-loop-exp2-recovery-2026.11.14.13`, W&B `o40fk9b6` [P])

| Native step | Validation finished (UTC) | Attempted/Scored | Raw verifier mean | Trailing-5 adj. EMA | Eligibility | Selected best | Save status |
|---|---|---|---|---|---|---|---|
| 2 | 2026-11-14 ~19:47 (quota failures 19:47:15–16) | 120 / 100 | not captured locally (native aggregate in private archive) | n/a | **INELIGIBLE** — all 20 SWE attempts failed Daytona snapshot quota ("Maximum allowed: 45") before capacity cleared [R: `recovery13-eval2-native-error-census.json`, STATE.md L49] | no | interval ckpt later rotated out |
| 4 | 2026-11-14 ~19:57 | 120 / 100 (computed from per-route aggregate: structure 0.2×20 + Nemotron 0.2513203125×80) | 0.24105625 over scored [R-computed: `recovery13-eval4-aggregate-check-20261114T1958Z.json`] | n/a | **INELIGIBLE** — same 20 SWE snapshot-quota failures; results unchanged, no rerun/EMA injection [R: `recovery13-eval4-ineligible-decision-20261114.json`] | no | interval ckpt later rotated out |
| 6 | by 2026-11-14 21:02 | 120 / 120 | 0.20007916874930573 [R: `recovery13-selection-after-eval6-20261114T2102Z.json`] | n/a (2 of 5 obs) | eligible score, insufficient window | no | interval ckpt later rotated out |
| 8 | 2026-11-14 21:39:22 | 120 / 120 | 0.1920711965509095 [R: `recovery13-eval8-completed-milestone-20261114.json`] | n/a (2 of 5 obs) | eligible score, insufficient window | no | interval ckpt later rotated out |
| 10 | 2026-11-14 22:59:40 | 120 / 120 | 0.17641340216121468 [R: `recovery13-eval10-completed-milestone-20261114.json`] | n/a (3 of 5 obs) | eligible score, insufficient window | no | interval ckpt later rotated out |
| 12 | 2026-11-14 23:46:03 | 120 / 120 | 0.1951813725490196 [R: `recovery13-eval12-completed-milestone-20261114.json`] | n/a (4 of 5 obs) | eligible score, insufficient window | no | interval ckpt later rotated out |
| 14 | 2026-11-15 01:12:03 | 120 / 120 | 0.21824999999999997 | **0.2008526161737512** [R: `recovery13-eval14-first-best-milestone-20261115.json`] | eligible (first complete 5-obs window) | **yes** (first best) | interval ckpt; later rotated out |
| 16 | 2026-11-15 02:46:22 | 120 / 120 | 0.2834267040149393 | **0.23259077361923453** (recomputed = recorded; validator PASS) [R: `recovery13-eval16-best16-milestone-20261115.json`] | eligible | **yes** | **retained native** (rolling 3) |
| 18 | 2026-11-15 03:40:29 | 120 / 120 | 0.24191666666666664 (avg_score 0.24108333; pass@1 0.25833333) | **0.238219238827311** [R: `recovery13-eval18-best18-step19-milestone-20261115.json`] | eligible | **yes** | **retained native** (rolling 3) |
| 20 | 2026-11-15 04:59:53 | 120 / 120 | 0.26758333333333334 (penalized mean 0.2659166666666667) | **0.25261617680247456** [R: `recovery13-rl-compute-complete-best20-milestone-20261115.json`] | eligible | **yes — final best 20** | **retained native + protected best**; exported to HF [P] |

Notes:
- Rolling retention is "rolling 3 + protected best" (user decision, 2026-11-14): trainer `max_ckpts_to_keep=3`, protected best; predecessor SFT step203 and its HF handoff preserved [R: STATE.md L73, completion19-receipt `retention`]. Retained native checkpoints: 16/18/20 [R: completion19-receipt].
- Eval2/eval4 were never repaired, relabeled or re-run; no EMA injection occurred [R: STATE.md L49].
- Eval16 training-tail diagnosis: 3 sandbox ENOSPC errors + 1 agent timeout were training outcomes, **not** eval16 exclusions [R: `recovery13-eval16-best16-milestone-20261115.json`].
- Independent EMA recomputation passes at 14/16/18/20; W&B `o40fk9b6` FINISHED at step 20 with independently delivered history [P: https://wandb.ai/marin-community/marin-long-loop/runs/o40fk9b6].

## Exp1 validation evidence (round 1, findable subset from STATUS.md)

Cadence: validation every 2 updates, 120 attempts per validation.

| Native step | Validation finished (UTC) | Attempted/Scored | Mean verifier reward | Best-EMA context | Selected best | Save status |
|---|---|---|---|---|---|---|
| 14 | 2026-11-12 18:51:25 | 120 / 120 | avg_verifier_score 0.2917458354159724 (pass@1 0.30833; SWE 0.35, Nemotron 0.30, structure 0.30) | best was step 12 before this; best 14 EMA 0.2800426649504649 | 14 (at 19:11 read) | save14 committed 19:11:43 in 564.66 s |
| 16 | 2026-11-12 20:10:14 | 120 / 120 | 0.2834167 (pass@1 35/120 = 0.2916667; SWE 7/20 = 0.35, Nemotron 21/80 = 0.2625, structure 7/20 = 0.35) | aggregate readback SHA256 9383d9fd7aa79462e8cedd1ac2a3a1ec546363f27a21cc1b3861bc4dfeddaa5d | best metadata still 14 at 20:12 read; later "latest committed native/best16" | best16 selected; RL FAILED 17 at the 12 h budget boundary (receipt after step17: 41,558.937 s of 43,200 s, 1,800 s drain margin); best16 preserved and manually exported |

Sources: [R] `/Users/benfeuer/Documents/experiments/complete/long-loop-prototype/STATUS.md` lines 825–898.

## Unavailable (explicit gaps)

- **Exp2 eval2/eval4 raw overall verifier means:** not present in local STATE/artifact summaries read for this ledger; attempt-level scores live only in the sealed private archive (2703 objects, manifest SHA `bec5323a9e7efd19ff4f251759092a1f0146a70866b41ae000922f297277aaff`). Eval4's scored-only mean above is computed from the per-route aggregate receipt.
- **Exp1 validation rows for steps other than 14 and 16:** per-step raw means/EMAs for other checkpoints are not recorded in STATUS.md; they live in the round-1 native aggregate and W&B history, which are outside the sources compiled here.
- **Exact per-step timestamps for eval6/eval18 aggregate emission:** STATE.md and milestone receipts give read-back and finished times as above; the eval6 aggregate read time (21:02 log) is the closest recorded timestamp.
- **Per-attempt validation scores:** withheld by rule (model outputs / task-level data); only aggregates are published.
