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
