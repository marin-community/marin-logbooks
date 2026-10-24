# Reproduce the finite same-source FP32 control

The frozen CPU run failed its exact S1 comparison and exited 1. This archive retains reports/graphs and hashes, not model/input/gradient arrays. The independent review checked source and retained metadata without rerunning numerical code. No TPU or full-Trainer conclusion follows.

`frozen-manifest.json` pins all 24 original payloads. Their gzip copies are under `../../evidence/fp32-source-identity/`; decompress each to the path obtained by removing `.gz`, then verify original lengths/SHA256s. For Python restore the original `.py` basename after removing `.source`. Direct source copies also exist under `../../runners/fp32-source-identity/`. Keep the first-attempt runner/report and repaired runner/report separate. The repair changes only scalar-safe result recording; numerical source functions and exact comparison are unchanged.

The runner imports the already archived 19-file v2 bundle at `../fp32-common-dq-v2/`, whose original manifest hash is `2f91c521b719d855acee45272489578d3ee48de574a34c94f398468cb3d495d3`. Restore each `.py.source` to its original `.py` basename, with all remaining files unchanged. `dependency-proof.json` verifies every restored helper byte, all 13 configured scientific blobs at each source pin, and ancestry of both pins in the public archive. Use baseline `36e2a9a97fab22b76dfa5a03964c6600ec5713ac` for execution; candidate `f1e991207ceda752957938b1cfe8893027d3fed8` was protected but was not executed by S1.

The frozen `config.json` retains absolute paths to the v2 bundle, baseline source worktree and prior output provenance. Recreate those paths in an isolated environment where they do not already exist; do not overwrite the original frozen directories. A local worktree of the baseline commit supplies the source fixture. Restore `prior-baseline-provenance.json.gz` as `provenance.json` at the configured retained-output directory, verifying SHA256 `e613c142f25bc01c8438e7aaed8dacd48f42dd5bd8affd49c8eed18c0fceda5e`. Its hashes/placements are scientific guard inputs.

This is the minimal input set the runner reads. It does not recreate the original 79-file protection enumeration: the historical run protected all files then present in the old bundle/output directories, while only the required old provenance is newly copied here. The complete retained report and independent review record verification of that original 79-file set. A minimal rerun records its actual smaller set; do not claim those protection inventories are identical. Additional old output files are unnecessary to its source/operand guard inputs.

Use the recorded interpreter/dependency versions and source paths from the reports. Preserve four CPU devices, source layout and highest precision; a version/path/placement guard failure is a failed reproduction. The recorded command, after restoring the original runner and config in a separate directory, is:

```bash
env -u LIBTPU_INIT_ARGS \
  JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 \
  XLA_FLAGS=--xla_allow_excess_precision=false \
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /private/tmp/marin-native-jax-query-scale-rounding/.venv/bin/python \
  /restored/source_identity.py --config /restored/config.json \
  --output /fresh/nonexistent-output
```

Replace only the restored runner/config and fresh output paths. Preserve the frozen config's source/input paths and contents. Expected scientific outcome is exit 1 with `completed:false`, `s1_exact:false` and the retained 460/1024 mismatch. This archival command was not executed here. Stop after S1 failure; do not run canonical-import/raw-dQ controls or launch hardware from this record.
