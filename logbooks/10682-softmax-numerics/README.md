# Softmax numerical measurements

Programs supporting [Marin PR #10682](https://github.com/marin-community/marin/pull/10682).
The source tree is pinned to Marin1545fb4898912be01051e82e1796906d667b78ab, with the experiment programs
in this directory. This is an archive of the executed measurement program, not a replacement trainer.

- `softmax_numerics.py` compares stable token logprobs/entropy with Transformer Engine2.19 and bf16 entropy,
  including configured1024-token scoring/entropy chunks, mixed-gradient production training passes,
  ten microbatch shapes, compiled binaries, warm CUDA/wall time and allocated memory.
- `checkpoint_softmax.py` samples identical Qwen/Snowball checkpoint logits and records per-token values.
- `rollout_replay.py` runs one production Megatron PPO step on a saved rollout, recording parameter/logits
  fingerprints, masks, policy loss, gradient norm and token logprob/entropy tensors.

Run from `vendor/skyrl` in the existing frozen Megatron development runtime on H100, with model
weights and saved rollout/config paths supplied explicitly. For example:

```sh
python ../../logbooks/10682-softmax-numerics/softmax_numerics.py --commit 1545fb4898912be01051e82e1796906d667b78ab --seed 17 --output /tmp/softmax.json
python ../../logbooks/10682-softmax-numerics/softmax_numerics.py --commit 1545fb4898912be01051e82e1796906d667b78ab --seed 17 --varied-shapes --output /tmp/varied.json
python ../../logbooks/10682-softmax-numerics/checkpoint_softmax.py --help
python ../../logbooks/10682-softmax-numerics/rollout_replay.py --help
```

Large checkpoint, rollout and output artifacts require the original storage access. Their command,
commit, job, immutable revision and output receipts remain in the softmax-numerics workstream records;
no private artifact payloads or access links are published in this archive.
