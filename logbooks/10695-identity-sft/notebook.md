---
title: "Snowball identity SFT, Path B"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/23"
related_issues: ["https://github.com/marin-community/marin/issues/10695"]
research_branch: "codex/10695-identity-sft"
marin_base_revision: "fca974b90bea247cf640e9cb17b315ed3a89e89d"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: active
---

# Snowball identity SFT, Path B

## Question and decision criteria

- Goal and why it matters: Marin Snowball 67B-A2B v0.1 calls itself ChatGPT/OpenAI with thinking ON and Nemotron/NVIDIA with thinking OFF. The goal is a short SFT from v0.1 after which the model itself says it is Marin's MoE from the Marin Community, without a system prompt.
- Baseline and alternatives: v0.1 with no system prompt. Alternatives were context distillation, on-policy distillation (#10491) and SFT on [marin-community/identity-data](https://huggingface.co/datasets/marin-community/identity-data) @665ac6e mixed with replay (this notebook, "Path B").
- Evidence needed to choose: the native-boundary identity screen (52 core prompts and 14 third-person controls, thinking OFF and ON, special tokens kept), BFCL parity (5 runs, run-level bootstrap), paired GSM8K and MATH500, SWE100, and Panel-60.
- Constraints and unresolved questions: tool calling (BFCL) must not regress. Panel-60 is a development set, not held out. Mini840 (#10804) is still a pilot.
- Held-out identity split: the identity prompts and targets that were once kept back as a held-out split (`experiments/post_training/identity_path_b/identity_scope/payload.zip` in the pinned commits) are now public and no longer held out. Ben Feuer's public Panel-60 now provides a separate development screen; it is not held out either.

## Source

Experiment code from eleven Marin draft PRs is kept here. Each PR head is preserved unchanged on its own branch; the only added commit on each removes the inherited `.github/workflows`. Reusable parts are promoted to Marin in separate, smaller PRs (see Final decision); the rest stays only here. The Path B training and evaluation source used by the runs is listed in [REPRODUCE.md §1](REPRODUCE.md) and [PINS.md](PINS.md).

| Marin PR | contents | archive branch | commit |
|---|---|---|---|
| marin-community/marin#10689 | Snowball identity interpretability harness | `codex/10695-identity-sft-pr10689` | [`a03fee01d8`](https://github.com/marin-community/marin-logbooks/tree/a03fee01d8a944876776561430bf2b99ecf5eb74) |
| marin-community/marin#10690 | identity SFT recipe and its precision audits | `codex/10695-identity-sft-pr10690` | [`35b5f17fb0`](https://github.com/marin-community/marin-logbooks/tree/35b5f17fb0f8d06d932ced4c8127e9a1b6956b88) |
| marin-community/marin#10691 | identity precision gate and route checks | `codex/10695-identity-sft-pr10691` | [`8196a8a0df`](https://github.com/marin-community/marin-logbooks/tree/8196a8a0dfa9ee23198f57c68491ff43f64ba1b7) |
| marin-community/marin#10693 | identity LR0 precision probe job | `codex/10695-identity-sft-pr10693` | [`1095e146e0`](https://github.com/marin-community/marin-logbooks/tree/1095e146e084c0087a56a3138e6153579a3227df) |
| marin-community/marin#10694 | identity MoE-drop and torch route-floor diagnostics | `codex/10695-identity-sft-pr10694` | [`7aebfceaa8`](https://github.com/marin-community/marin-logbooks/tree/7aebfceaa893fa545a9e57463538423c78c478fe) |
| marin-community/marin#10701 | first identity scorer (superseded by marin-community/marin#10884) | `codex/10695-identity-sft-pr10701` | [`c9addf3973`](https://github.com/marin-community/marin-logbooks/tree/c9addf3973c949518afb5847af013b4efeb8585b) |
| marin-community/marin#10702 | identity SFT milestone-16 spending-gate eval | `codex/10695-identity-sft-pr10702` | [`4adba32ffe`](https://github.com/marin-community/marin-logbooks/tree/4adba32ffe067359601a28714f48af07e1f653ec) |
| marin-community/marin#10708 | audited 16-update identity milestone run | `codex/10695-identity-sft-pr10708` | [`8dd7358cff`](https://github.com/marin-community/marin-logbooks/tree/8dd7358cff8ba9b2100f8561d35b137c600a9eea) |
| marin-community/marin#10730 | SkyRL teacher-prompt conditioning without changing student tokens | `codex/10695-identity-sft-pr10730` | [`4a7edc5aa1`](https://github.com/marin-community/marin-logbooks/tree/4a7edc5aa1d020fc521e8a4eb33b5c21ba41324b) |
| marin-community/marin#10731 | SkyRL Iris job bounds and rollout request plumbing | `codex/10695-identity-sft-pr10731` | [`a70aebc1fb`](https://github.com/marin-community/marin-logbooks/tree/a70aebc1fb9ccde6306847ed9266bc9358671772) |
| marin-community/marin#10732 | identity OPD from the context-distillation checkpoint | `codex/10695-identity-sft-pr10732` | [`9eb868b391`](https://github.com/marin-community/marin-logbooks/tree/9eb868b391843db05200e65dc1a64fb80fd8cdf6) |

## Baseline

- Date: the fresh v0.1 reference runs in `evidence/notes/path-b/bfcl-noise-r1/` (2026 October-November; see each file).
- Published source revision: see [REPRODUCE.md §1](REPRODUCE.md) and [PINS.md](PINS.md).
- Configuration, command, and external inputs: Snowball v0.1 served with no system prompt. The eval commands are in [REPRODUCE.md §5](REPRODUCE.md).
- Environment and hardware when relevant: CoreWeave H100 through Iris.
- Result and evidence: BFCL 27.6%. With thinking ON, v0.1 claims another lab in 46 of 52 core prompts. Each number links to its file in [RESULTS-INDEX.md](RESULTS-INDEX.md).
- Limitations: single BFCL runs of the same model vary a lot. Five fresh v0.1 runs ranged from 25.2% to 33.3% (`fresh-v01-reference-extension-marten-r1.json`, `reference.min_max_percent`). That is why each model gets five runs and a run-level bootstrap. Runs are not paired across models.

## Entries

The arm-by-arm history (A0 to A13, merges M25/M49/M51/M75) is in #10695 and in the byte-exact working notes under [`evidence/`](evidence/MANIFEST.json). [REPRODUCE.md](REPRODUCE.md) gives one row per arm.

### 2026-10-30 00:26 UTC - First r5 checkpoints fail BFCL

- Question or hypothesis and reason for trying it: the first Path B training round (r5) mixed identity rows into replay at 10% and 25%, with identity rows either plain (B-base) or carrying reasoning traces (B-traces). Before training further, does tool calling survive at the first checkpoint?
- Command or material configuration: one BFCL parity run (123 tasks) per first checkpoint, paired per task against one v0.1 reference run with the operator system prompt (34 of 123), 10,000-draw paired bootstrap.
- Result (measured): every arm lost tool calling. B-base10 (update 35) 21 of 123, −10.6 points [−19.5, −1.6]; B-base25 (update 51) 19, −12.2 [−22.0, −2.4]; B-traces10 (update 36) 11, −18.7 [−27.6, −9.8]; B-traces25 (update 54) 10, −19.5 [−27.6, −11.4].
- Evidence: [evidence/notes/path-b/r5-checkpoint-bfcl-r1/independent-first-four-paired-reading-marten-r2.json](evidence/notes/path-b/r5-checkpoint-bfcl-r1/independent-first-four-paired-reading-marten-r2.json).
- Interpretation and limitations: one run per checkpoint, against a reference run that had a system prompt. Later work used five runs per model and a fresh v0.1 reference without a system prompt (27.6%), so these numbers are not directly comparable with the tables below. Hypotheses at this point: the special-token row freeze, the thinking-ON identity rows, or the replay mix itself.
- Decision and next action: stop r5. Run a controlled 36-update ablation from v0.1 (A0–A4) that changes one factor per arm.

### 2026-10-30 01:13 UTC - A0–A4 ablation: thinking-ON identity rows break tool calling

- Question or hypothesis and reason for trying it: which r5 factor causes the BFCL loss?
- Command or material configuration: packet `evidence/notes/path-b/bfcl-ablation-r1`. A0 is the r5 B-base10 prefix (control). A1 removes the partial special-row freezes. A2 is replay only. A3 replaces each thinking-ON identity row with the same-source thinking-OFF row. A4 bridges to the earlier CD16-T3 recipe (different data mix; see [REPRODUCE.md §3](REPRODUCE.md)). BFCL: 5 full runs per model, run-level bootstrap against a fresh v0.1 reference.
- Result (measured): BFCL v0.1 27.6%. A0 15.6% (−12.1 [−16.5, −7.6]); A1 14.8% (−12.8 [−16.3, −9.9]); A2 24.6% (−3.1 [−6.2, −0.7]); A3 31.1% (+3.4 [−0.5, +7.2]); A4 36.3% (+8.6 [+4.4, +11.9]). In one BFCL run each, A0 answered once without a tool call on 60 of 123 tasks; A3 did so on 12.
- Evidence: [fresh-v01-reference-extension-marten-r1.json](evidence/notes/path-b/bfcl-noise-r1/fresh-v01-reference-extension-marten-r1.json), [mechanical-failure-counts-marten-r1.json](evidence/notes/path-b/bfcl-failure-diagnostic-r1/mechanical-failure-counts-marten-r1.json).
- Interpretation and limitations: A1 ≈ A0: removing the partial special-row freezes alone did not restore BFCL. This does not show the freeze has no effect. In these arms, swapping the thinking-ON identity rows for thinking-OFF rows (A3) removed the loss. Hypothesis (not directly tested): thinking-ON identity rows, whose answer follows an empty reasoning block, teach the model to answer and stop instead of calling a tool. A4 uses a different data mix, so it does not isolate the identity rows.
- Decision and next action: A3 becomes the first candidate. Test whether thinking-ON identity rows with real reasoning avoid the loss (A5b, A6).

### 2026-10-30 05:22 UTC - Context-distillation teacher pilot (A5b) yields no usable traces

- Question or hypothesis and reason for trying it: can v0.1 with an identity system prompt write thinking-ON identity reasoning that we could train on?
- Command or material configuration: packet `evidence/notes/path-b/ablation-a5b-pilot-r1`; 200 pilot generations, then a QC prescreen.
- Result (measured): 200 of 200 requests completed; 0 of 200 were kept by the prescreen.
- Evidence: [ablation-a5b-pilot-reading-r1/report.json](evidence/notes/path-b/ablation-a5b-pilot-reading-r1/report.json) (`gate.prescreen_kept`).
- Interpretation and limitations: the reading stopped at the prescreen; no full role-aware reading was run.
- Decision and next action: use the QC-passed GLM rationales that already exist for identity rows (A6).

### 2026-10-30 07:46 UTC - A6: GLM reasoning on thinking-ON identity rows does not fix BFCL

- Question or hypothesis and reason for trying it: if thinking-ON identity rows carry real (GLM-written) reasoning instead of an empty block, is tool calling preserved?
- Command or material configuration: packet `evidence/notes/path-b/ablation-a6-r1`. A0's schedule with GLM reasoning on every eligible thinking-ON identity row; ineligible ones become thinking-OFF rows.
- Result (measured): BFCL 15.4% (−12.2 [−15.6, −9.3]). In one run, A6 answered once without a tool call on 66 of 123 tasks and recited its identity on 29.
- Evidence: [A6-run-bootstrap-extension-marten-r1.json](evidence/notes/path-b/bfcl-noise-r1/A6-run-bootstrap-extension-marten-r1.json), [mechanical-failure-counts-marten-r1.json](evidence/notes/path-b/bfcl-failure-diagnostic-r1/mechanical-failure-counts-marten-r1.json).
- Interpretation and limitations: adding these GLM rationales did not restore BFCL in this mix. This does not show that reasoning content never matters; the next test lowers the share of thinking-ON identity rows.
- Decision and next action: keep only a small share of the GLM-reasoned rows (A8a about 10%, A8b about 30%).

### 2026-10-30 09:55 UTC - A8a and A8b: the 10% arm is near the reference; the 30% arm loses BFCL

- Question or hypothesis and reason for trying it: how many thinking-ON identity rows can the mix hold before tool calling breaks?
- Command or material configuration: packet `evidence/notes/path-b/ablation-a8-r1`. Nested subsets of A6: A8a keeps 179 of 1,791 GLM-reasoned rows, A8b keeps 537; the rest become thinking-OFF rows. Same 36 updates.
- Result (measured): A8a 26.2% (−1.5 [−4.9, +1.5]); A8b 17.2% (−10.4 [−14.0, −7.3]). Answered once without a tool call (one run): A8a 17, A8b 43 of 123.
- Evidence: [A8-joint-run-bootstrap-extension-marten-r1.json](evidence/notes/path-b/bfcl-noise-r1/A8-joint-run-bootstrap-extension-marten-r1.json).
- Interpretation and limitations: the A8a interval includes 0; no non-inferiority margin was registered.
- Decision and next action: A8a becomes the second candidate. Test training length (A9, A11) and a small Tülu-style set (A10).

### 2026-10-30 13:54 UTC - Training length (A9, A11) and a Tülu-style set (A10)

- Question or hypothesis and reason for trying it: does more training improve identity without hurting tools, and does a much smaller identity set work?
- Command or material configuration: A9 = A8a's recipe at 72 updates (13:54 UTC launch); A10 = 440 identity rows from 44 sources, interleaved into replay, 36 updates (14:47); A11 = A3's recipe at 72 updates (17:13). Packets `ablation-a9-r1`, `ablation-a10-r1`, `ablation-a11-r1`.
- Result (measured): BFCL A9 24.6% (−3.1 [−6.7, +0.3]); A10 24.0% (−3.7 [−8.0, +0.0], 4 runs); A11 31.2% (+3.6 [−0.5, +7.5]). Native identity screen, thinking ON, of 52 core prompts: A9 claims another lab in 1, A10 in 9, A11 in 2. A9 and A11 each rewrite 1 of 14 third-person controls in both modes.
- Evidence: [RESULTS-INDEX.md](RESULTS-INDEX.md) rows for A9, A10 and A11.
- Interpretation and limitations: 72 updates did not improve identity over 36 and added a few other-lab answers. A10 is too weak.
- Decision and next action: stay at 36 updates. Drop the Tülu-style set.

### 2026-10-30 16:18 UTC - Native-boundary identity screen of A3 and A8a

- Question or hypothesis and reason for trying it: with special tokens kept and no system prompt, what do the two candidates say about themselves?
- Command or material configuration: packet `evidence/notes/path-b/native-boundary-identity132-r1`; 52 core identity prompts and 14 third-person controls, thinking OFF and ON, read with the role-aware `creator_MC_final` flag.
- Result (measured): thinking ON, A3 gives the name in 31 and the creator in 44 of 52, A8a in 34 and 33; neither claims another lab. A3's reasoning invents a "system message" in 17 of 52; A8a's in 1.
- Evidence: [v01-A3-native-role-aware-ledger-marten-r1.json](evidence/notes/path-b/native-boundary-identity132-r1/v01-A3-native-role-aware-ledger-marten-r1.json), [A8a-native-role-aware-ledger-marten-r1.json](evidence/notes/path-b/native-boundary-A8a-replacement1-r1/A8a-native-role-aware-ledger-marten-r1.json).
- Interpretation and limitations: only 31 of the 52 core prompts ask for the model's identity, and those 31 include yes/no checks on other models, so not all of them require the exact name; 17 ask only about the creator, origin or contact; 4 are open descriptions where the name is optional. A3 has the better BFCL; A8a has the cleaner reasoning.
- Decision and next action: look for one model with both: add tool replay to A8a (A13), shorten A8a (A12), and merge A3 with A8a.

### 2026-10-30 23:15 UTC - SWE100 top-up for v0.1, A3 and A8a

- Question or hypothesis and reason for trying it: do the candidates keep agentic coding ability?
- Command or material configuration: mini-swe-agent 2.1, full 100-of-100 runs only; production top-up packet `evidence/notes/path-b/second-agentic-swe100-production-topup-execution-r1`.
- Result (measured): v0.1 28, 33, 28 (mean 29.7%); A3 28, 29, 28 (28.3%); A8a 25, 27, 28, 29 (27.3%). A3 vs v0.1 −1.3 [−4.7, +0.7]; A8a vs v0.1 −2.4 [−5.6, +0.2].
- Evidence: [SWE100-production-topup-joint-bootstrap-extension-marten-r1.json](evidence/notes/path-b/bfcl-noise-r1/SWE100-production-topup-joint-bootstrap-extension-marten-r1.json).
- Interpretation and limitations: both intervals include 0, with 3–4 runs per model. This archive has no SWE100 result for A13.
- Decision and next action: SWE100 does not separate the candidates.

### 2026-10-30 23:21 UTC - A12 (shorter) and A13 (more thinking-ON tool replay)

- Question or hypothesis and reason for trying it: can A8a keep its clean reasoning and gain tool calling, either by training less (A12, 18 updates) or by doubling the thinking-ON tool-calling replay rows (A13)?
- Command or material configuration: packet `evidence/notes/path-b/ablation-a12-a13-r1`; A13 was relaunched at 23:57 UTC after a preemption (`A13-preemption-replacement1-r2`).
- Result (measured): BFCL A12 14.8% (−12.8 [−16.4, −9.9]); A13 28.3% (+0.7 [−3.4, +4.4]); A13 − A8a +2.1 [−1.5, +5.7]. Answered once without a tool call (one run): A12 45, A13 16. A13 identity, thinking ON: name 35, creator 32, other lab 0 of 52 (thinking OFF: other lab 1); reasoning invents a system message in 3. Paired maths: A12 GSM8K +0.2 [−0.7, +1.1]; A13 GSM8K −1.1 [−1.9, −0.3], MATH500 interval includes 0 (no equivalence test). For comparison, A3 GSM8K −1.7 [−2.6, −0.9] and A8a −1.1 [−1.9, −0.3].
- Evidence: [A12-run-bootstrap-extension-marten-r1.json](evidence/notes/path-b/bfcl-noise-r1/A12-run-bootstrap-extension-marten-r1.json), [A13-run-bootstrap-extension-shoveler-r1.json](evidence/notes/path-b/bfcl-noise-r1/A13-run-bootstrap-extension-shoveler-r1.json), [A13-native-role-aware-ledger-marten-r1.json](evidence/notes/path-b/native-boundary-A13-r1/A13-native-role-aware-ledger-marten-r1.json), [A13 paired maths](evidence/notes/path-b/A13-paired-math-r1/paired-math-analysis-marten-r1.json).
- Interpretation and limitations: the observed arm scores are A12 (18 updates) 14.8%, A8a (36) 26.2% and A9 (72) 24.6%. These are separate arms, not one controlled trajectory: A9's realized dose and packing differ from A8a's. The A13 − A8a gain is within noise. The measured GSM8K point losses for A3, A8a and A13 are 1.1–1.7 points; A12's interval includes 0 but does not establish zero loss, and A12 loses tool calling.
- Decision and next action: A13 becomes the lead candidate on 2026-10-31, with A3 as the fallback.

### 2026-10-31 00:07 UTC - A3/A8a merge sweep

- Question or hypothesis and reason for trying it: can a linear merge of A3 and A8a get A3's BFCL and A8a's clean reasoning?
- Command or material configuration: W = (1−t)·A3 + t·A8a with t = 0.25, 0.49, 0.51, 0.75 (M25, M49, M51, M75); packet `evidence/notes/path-b/merge-sweep-r3`.
- Result (measured): BFCL M25 30.4%, M49 29.5%, M51 23.8%, M75 23.9%; M49 and M51 have 4 of the planned 5 runs. Reasoning invents a system message in 18, 13, 1 and 1 of 52. GSM8K: M25 −1.5, M49 −1.6, M51 −1.2, M75 −1.2 points.
- Evidence: [RESULTS-INDEX.md](RESULTS-INDEX.md) rows for M25, M49, M51 and M75.
- Interpretation and limitations: BFCL and the invented system message switch together between t = 0.49 and 0.51; no tested blend gets both.
- Decision and next action: no merge candidate.

### 2026-10-31 08:30 UTC - Panel-60 on the three candidates

- Question or hypothesis and reason for trying it: how do the candidates do on Ben Feuer's public 60-question identity panel?
- Command or material configuration: Panel-60 protocol from the [public gist](https://gist.github.com/penfever/ff6565deebbd68700685807f8e081be1); all 300 answers graded. Ship line: clean ≥ 20, name ≥ 15, other lab ≤ 3.
- Result (measured), clean / name / creator / other lab / invents: v0.1 10 / 5 / 4 / 17 / 22; v0.1 with system prompt 33 / 26 / 40 / 4 / 13; A13 15 / 35 / 54 / 1 / 3; A8a 14 / 35 / 47 / 0 / 3; A3 15 / 34 / 56 / 0 / 1.
- Evidence: [panel60-unblinded-ship-line-report-margay-r1.json](evidence/notes/t310/panel60/pathb-exec-r1/panel60-unblinded-ship-line-report-margay-r1.json).
- Interpretation and limitations: all three candidates meet the name and other-lab lines; none meets the clean line. Panel-60 is a development set, not held out.
- Decision and next action: keep A13 as the lead. The release choice is Ben Feuer's and is still open.

### 2026-11-03 - Public release of the candidates and preservation of A13's inputs

- Question or hypothesis and reason for trying it: the work must be reproducible after temporary storage expires.
- Command or material configuration: A3, A8a and A13 were made public as open-athena HF repos. A13's training inputs were copied to [open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30](https://huggingface.co/datasets/open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30) and read back byte for byte.
- Result (measured): see the weights table in [REPRODUCE.md](REPRODUCE.md) and [captured-inputs.json](captured-inputs.json).
- Evidence: [preserve-A13-rebuild-inputs-r1/](evidence/notes/path-b/preserve-A13-rebuild-inputs-r1/INPUTS.md), [KEEP-HF.md](evidence/notes/path-b/KEEP-HF.md).
- Decision and next action: verify that the A13 data step rebuilds from these inputs (next entry).

### 2026-11-03 22:50 UTC - A13 rebuild and launch compose check

- Question or hypothesis and reason for trying it: can someone with only the public inputs rebuild A13's packed training file, and does the A13 launch config still compose?
- Published source revision: the A13 training pin in [REPRODUCE.md §1](REPRODUCE.md).
- Command or material configuration: `rebuild_a13.py --input-root DIR --output DIR`, and `evidence/compose_a13.py` against both A13 launch packets.
- External inputs and artifact references: [open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30 @df171cb4](https://huggingface.co/datasets/open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30/tree/df171cb4a351470f374f7605c30ab853f66cc5f5), listed in [captured-inputs.json](captured-inputs.json).
- Environment and hardware when relevant: CPU only. Python 3.12, pyarrow 23.0.1, numpy 2.3.5, transformers 5.12.1, tokenizers 0.22.2.
- Result, including failure or restart: the packed file matches SHA256 `f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba` (288 rows). Both packets compose: the original packet to job `pathb-bfcl-a13-r1-69ddc88fb2ff`, the replacement packet to `pathb-bfcl-a13-r1-5a3bd99de0a6`.
- Evidence: [evidence/a13-rebuild-receipt.json](evidence/a13-rebuild-receipt.json).
- Interpretation and limitations: this covers A13 only. The other arms' data steps are documented in REPRODUCE.md but were not rerun.
- Decision and next action: the A13 reproduction path is verified.

## Final decision

- Selected approach and evidence: A13 is the lead candidate. Its recipe is thinking-OFF identity rows, plus 10% thinking-ON identity rows with GLM reasoning, plus thinking-ON tool-calling replay rows doubled. Its BFCL is 28.3% (+0.7 [−3.4, +4.4] vs v0.1). With thinking ON, 0 of 52 native final answers claim another lab (thinking OFF: 1 of 52). GSM8K is −1.1 points [−1.9, −0.3]. MATH500 is +0.2 points [−1.07, +1.47]; the interval includes 0, but no equivalence test was run. A3 (thinking-OFF identity rows only) is the fallback.
- Rejected alternatives and why:
  - A0 and A1, trained with thinking-ON identity rows, scored about 15% on BFCL (v0.1: 27.6%). This fits the model answering and stopping instead of calling a tool.
  - Context distillation from v0.1 with an identity system prompt passed QC on 0 of 200 pilot traces.
  - A9 and A11 (72 updates) did not improve the identity screen over the 36-update arms.
  - A12 (18 updates) scored 14.8% on BFCL.
  - A10 (tiny Tülu-style set) still claimed another lab in 9 of 52 thinking-ON answers.
  - Linear merges of A3 and A8a (M25, M49, M51, M75) did not get both A3's BFCL and A8a's reasoning.
- Marin contribution PRs: #10683, #10684, #10685, #10687, #10705, #10782 (library fixes); #10707 (export audit), #10884 (native identity scorer), #10885 (BFCL parity eval); #10686, #10688, #10706 (uncertain, still open).
- Reproduction instructions: [REPRODUCE.md](REPRODUCE.md)
- Remaining uncertainty: the 36-update arms screened on paired maths lost about 1–2 GSM8K points; halving training (A12) removed that cost but collapsed BFCL. Panel-60 is a development set. Mini840 results are still pending.
