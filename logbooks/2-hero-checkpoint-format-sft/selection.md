# Contribution selection, 2026-10-23

Two fresh reviewers independently inspected the complete proposed contribution at Marin source `c39c6f2f5b1b1c5aaa29d51a39304b40c7612dca` before extraction. Each applied the forward-maintenance criterion from [research-engineering at workflow PR #9467](https://github.com/marin-community/marin/blob/ee916beaf7fb7d243ff8a949335089be983b8e55/.agents/skills/research-engineering/SKILL.md): will Marin use and maintain this piece, or does it only reproduce this trial?

## Selected existing interfaces

| Files | Decision and reason |
| --- | --- |
| `lib/levanter/src/levanter/kernels/triton/short_conv/{api,triton_gpu}.py` and its existing kernel tests | Promote the activation-only VJP for frozen convolution weights. Ordinary callers benefit without adopting Hero SFT. Retain ordinary joint and weight-only gradient coverage because the custom VJP also serves those modes. |
| `experiments/grug/moe_hero_ep/weights.py`, `ops/export_vllm.py`, `ops/test_export_vllm.py` | Promote preservation of authoritative FP32 router bias while casting learned weights to BF16. The existing native-to-HF exporter is a maintained consumer; its regression uses bias values that BF16 would round. |
| `lib/fray/src/fray/types.py` | Promote only the GB200 device literal: the existing native sampler already constructs that resource. Archive the concrete collection annotation change, because its only real JSON decoder is the new campaign launcher. No new test dependency is justified to preserve that archival interface. |

## Archived components

| Files or family | Why these remain in the research branch |
| --- | --- |
| `averaging.py`, `test_averaging.py` | LR-adjusted averaging, A/B/C provenance, exact donor assignments, optimizer reconstruction and physical-region probes form the experiment's preparation protocol. Successfully producing B/C does not select this whole protocol as a maintained API. |
| `sft.py`, `test_sft.py`, `sft_launch.py`, `sft_campaign.py`, `test_sft_campaign.py` | Compact training, recovery ancestry, telemetry and deferred acquisition are a campaign-specific backend with no maintained caller outside the new cluster. Preserve the complete implementation and numerical/resume tests here. |
| `model.py` additions | LoRA/vocabulary hooks and YaRN/HF rotary changes support only this research cluster. The extracted contribution must not retain them merely to keep archived SFT imports working. |
| `sft_hlo.py`, `test_sft_hlo.py`, compiled-gradient audit | Compiler-sensitive exploratory audit. The actual campaign did not require a general HLO absence certificate; direct kernel regressions cover the retained VJP boundary. |
| `format_data.py`, `test_format_data.py` | Campaign-specific catalog prefixes, donor validation and manifests. Shared SFT data recipes and tokenization already exist in Marin. |
| `format_eval.py`, `test_format_eval.py`, `format_handoff.py`, `test_format_handoff.py` | A–E policy, 500-case quality gates and a single-execution/fresh-optimizer assembler. The resumed campaign did not qualify through that design. |
| `format_pipeline.py`, `test_format_pipeline.py` | The formal quality-qualified pilot path was not the selected execution. |
| `format_inventory.py` | Historical source-window and tokenizer evidence discovery. Existing checkpoint discovery remains in Marin. |
| `format_job.py`, `test_format_job.py` | Campaign request and launch protocol has no independent maintained caller after archiving. Standard Iris/Fray launch APIs remain available. |
| `format_measure.py`, `test_format_measure.py` | The panel and integration served this fixed A/B comparison and the new qualification harness. No selected future consumer. |
| `format_sample.py`, `format_suite.py`, `test_format_suite.py` | Campaign orchestration around existing native sampling/checkpoint primitives, including the unfinished full suite. |
| `rollout_sanity.py`, `test_rollout_sanity.py`, `rollout_sanity_data/` | Frozen 100-case panel, seeds, references, two waves and physical padding reproduce this trial. They are not an adopted benchmark. |
| `serving_parity.py`, `test_serving_parity.py`, `sft_merge_parity.py`, `test_sft_merge_parity.py` | Unqualified follow-up protocols. Existing shared backend-parity primitives remain in Marin. |
| `context_extension.py`, `yarn.py`, `test_context_extension.py` | Separate 65K proposal was not selected or measured. |
| `conftest.py`, `format_test_support.py` | Fixtures solely for the archived campaign. Keeping them in Marin would leave dependencies on removed modules. |
| `docs/references/hero-format-sft.md`, README addition, mkdocs entry, `.agents/projects/hero-checkpoint-averaging-format-sft.md` | Historical commands, proposed paths and research status belong with the archived source. They should not advertise an adopted maintained interface. |

A future adoption of compact Hero SFT can promote a narrower backend with an actual maintained consumer and a base-checkpoint contract. Historical reproducibility is preserved by this branch regardless of that later decision.
