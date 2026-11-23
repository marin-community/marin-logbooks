# Repeat the selected Snowball and Mochi FP8 comparisons

The four profiles repeat the selected comparisons from the prior [FP8 campaign](https://github.com/marin-community/marin-logbooks/tree/652ad9a3f5f7da48d93223b033f43d05af43a2b5/logbooks/57-inference-perfmax/analysis/fp8-qualification-01a1f7c4). Snowball uses 8 H100s at 128 sessions and puts all 26 routed expert layers in block FP8. Mochi uses 16 GB200s at 64 sessions and puts layers 16–27 of 48 in block FP8. Both pairs use TP1, BF16 KV and the original checkpoints, goldens, numerical limits and AgentX replay. Each arm starts a fresh compiled server.

Both architectures package vLLM `e9dfdd2b94827b8f0f9ae4d21fff5f1a2a5dc796`, built from current main `06963d0b8bf361b1ed357dbe9b6c9fa44eea548f` with the exact #132, #133, #135 and combined #136 commits. The two build inputs pin one source archive and the two native base artifacts from release `47a8e415bff1`. The source diff from that release changes no native sources or dependencies. The packager verifies every native payload and each changed Python member. The only runtime source change from the retained Mochi wheel reuses the same expert-prefix string. The Snowball additions only select BF16 Triton for explicitly ignored FP8 experts. Neither Snowball arm ignores an expert layer.

The existing runner performs the storage and numerical gates followed by a three-minute Probe. A passing arm can proceed to the one-hour Confirm with the pinned AgentX warmup. Run from the recorded research checkout with the exact profile name.

```sh
inputs=logbooks/57-inference-perfmax/analysis/da6-fp8-confirm-01a1f7c4
uv run python -m experiments.evaluation.knotmeter.cli run \
  --stage probe --profile "$inputs/snowball-bf16-c128.yaml" \
  --campaign da6-fp8-confirm-01a1f7c4 --priority interactive
```
