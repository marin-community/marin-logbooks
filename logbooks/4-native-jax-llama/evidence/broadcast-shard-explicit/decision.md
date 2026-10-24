# CPU broadcast transport decision

Select the requested source payload before placing it on an all-Explicit JAX mesh, and transport floating values as bytes so collective reductions preserve signed zero and NaN payloads. Keep the existing Auto path. This is the implementation in [draft PR22](https://github.com/dlwh/marin/pull/22), source [91793b80543533ed7a93894ce169dda2b3db9796](https://github.com/dlwh/marin/commit/91793b80543533ed7a93894ce169dda2b3db9796), parent `919c1cc2fa0b512a1d7e2e4065d2dcda8f5e7bb3`. The supplied publication receipt records the verified public draft/head and a completed required lint review (six lanes plus composer, zero findings) at 20:23 UTC on 2026-10-24. This archival step made no GitHub query.

Auto and Explicit are JAX mesh-axis modes: compiler-managed placement versus placement carried by array types.

## Decision and rejected alternatives

The existing broadcast helper stages an array with a synthetic leading source/device dimension. Direct scalar indexing of that sharded dimension failed on an Explicit mesh before output placement could occur. Replicating the complete staged array made indexing work, but CPU compilation retained N payload-sized slots. For a 512x512 FP32 payload (1,048,576 bytes), estimated per-device temporary bytes grew from 4,194,308 on four devices to 8,388,612 on eight, with an all-gather of `[N,512,512]`. The original Auto branch had the same N-payload compiled pattern and remains unchanged. These are compiler estimates, not measured peak memory or throughput. [Scaling records](scaling-comparison.json), [four-device HLO](full-replication-4.hlo), [eight-device HLO](full-replication-8.hlo).

A length-one array index made JAX gather only the selected source payload. Its ordinary floating-point collective retained bounded temporary size but changed FP32 negative zero to positive zero; the [saved bit-pattern report](selected-float-payload-bits.json) records `0x80000000` becoming `0x00000000`. The final two-rank regression also rejects this [historical floating-transport patch](rejected-floating-transport.patch) on BF16 signed-zero/NaN bytes. A generic complex `.view()` round trip was also rejected because reconstruction through arithmetic changed complex NaN/sign components; it is not part of the implementation.

The selected Explicit path bitcasts floating components to uint8, gathers one selected source, reverses the bitcast and applies the requested output sharding. Bool and integer arrays use direct gather. Complex values transport real and imaginary components separately and reconstruct with `lax.complex`. This follows the existing checkpoint byte-transport primitive pattern, without a new dtype table or typed-key API. The literal Auto/mixed expression and source-process index calculation remain unchanged. [Preservation proof](preservation-proof.json), [fresh independent static response review](maintained-response-review.md).

## CPU evidence and limits

The [bit-preserving probe](bit-preserving-probe.py.source) records 128 compiled-kernel dtype/source-index cases across four/eight CPUs and x64 off/on: 28 cases per device count without x64 and 36 with x64. They cover FP16/BF16/FP32/complex64/bool/integers, plus enabled FP64/complex128/64-bit integers. Exact global and shard-local bytes and requested physical layouts match for selected source-slot indices 0/1. Its selection helper is AST-identical to the maintained helper after captured-variable renaming. These are isolated kernel tests, not 128 full-helper or multi-process runs. [Four-device results](dtype-source-4-x640.json), [eight-device results](dtype-source-8-x640.json), [x64 four-device results](dtype-source-4-x641.json), [x64 eight-device results](dtype-source-8-x641.json).

For the 1 MiB real payload, the selected-byte kernel's estimated temporary bytes stay 2,097,156 at both device counts; for the 2 MiB complex payload they stay 4,194,384. The HLO contains one selected payload, including byte dimensions, rather than N staged payloads: [real4](selected-bytes-real-4.hlo), [real8](selected-bytes-real-8.hlo), [complex4](selected-bytes-complex-4.hlo), [complex8](selected-bytes-complex-8.hlo).

An actual unchanged-parent Auto baseline canonicalized BF16 NaN payload `0x7fc5` to `0x7fc0`, while exact numerical real/imaginary values and local layouts matched for both source processes 0/1. No Auto negative-zero change is asserted. The final regression therefore checks Auto values with explicit NaN equality and checks new Explicit transport by strict local bytes. Both retain dtype, shape, static LmExample metadata and every requested local layout assertion. [Baseline evidence](auto-baseline.json).

The final existing utility file passed 19 tests/skipped 2 on four CPU devices and passed 17/skipped 4 on one CPU device. Its broadcast cases use two actual CPU processes on one machine, differing non-source inputs and source processes 0/1 in both mesh modes. Forcing source0 makes both mode cases fail when source1 is requested; the old floating transport passes Auto and fails the new Explicit byte assertion. Normal required precommit passed. [Selected validation receipts](validation.json).

The campaign's broad run used the preceding 8,798-byte snapshot with the same production helper and an earlier Auto byte oracle:2,068 passed,37 skipped,14 failed. Thirteen failures matched actual unchanged `2b65a9e3dbbe6dc28d3093cf2c59052922ca2b2d` baseline nodes with a missing optional `cloud` import; the remaining failure was the superseded Auto test. Final focused tests validate the corrected oracle. The broad run is not relabeled as a final-fixture green suite. [Broad proof](broad-affected-proof.json).

There is no physical multi-host, accelerator, performance or peak-memory qualification. Mixed Auto/Explicit feature placement remains unsupported generally; its old expression is preserved, without claiming all historical mixed cases succeed. Typed PRNG-key host conversion is not added. The archive preparation reran no tests or compiler probes, initialized no model/client and changed no scientific source, numerical criteria or canonical defaults. This shared helper contribution does not resolve the campaign's rejected large-Trainer numerical/adoption gate.

## Reproduction

Use a separate checkout and the recorded CPU Python3.12/JAX0.11.1 environment. Public source and frozen patch provenance are in `validation.json`; these tests ran the frozen working tree before the equivalent 91793 commit was created. Do not apply the historical patch to the research archive's scientific source tree.

For the selected maintained behavior, check out 91793 and run its existing utility file with the repository default safety markers. Select the parent CPU count through XLA_FLAGS and leave JAX_NUM_CPU_DEVICES unset so existing subprocess fixtures can choose their own count:

```bash
unset JAX_NUM_CPU_DEVICES
JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4 --xla_allow_excess_precision=false' python -m pytest lib/levanter/tests/test_jax_utils.py -q
```

The retained standalone probes can run from a scratch directory after restoring their `.py` suffixes and creating `output/`. Run the scaling probe with CPU device counts 4 and 8. Run the bit-preserving probe with both counts and JAX_ENABLE_X64 set to 0 and 1. The archived copies change only the historical output prefix to `output/`; `manifest.json` retains the original executed script length/SHA256 and exact normalization. Compiler filenames can differ; compare values, physical layouts and memory/collective records, not HLO metadata byte identity across environments.

To reproduce the selected-floating rejection in a disposable checkout, start at parent 919c, apply `rejected-floating-transport.patch`, and run `payload-bits-probe.py` there. For its final two-rank negative control, use the 91793 `test_jax_utils.py` fixture with that historical helper. To reproduce the wrong-source control, change only the fixture's `source=source` call to `source=0`; expected source1 values must fail. These are local CPU controls and require local loopback sockets, without remote services or accelerator access.
