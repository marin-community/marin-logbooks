# TPU callback host-CPU backend review

## Verdict

**The installed JAX 0.11.1 source supports a separately controlled, fresh-process `JAX_PLATFORMS=tpu,cpu` candidate.** It explicitly initializes both backends, gives the first listed backend the default priority, and requires both to initialize successfully. The callback implementation requires a local CPU device even when its compiled caller runs on TPU. This can address the new reported missing-CPU failure while preserving a TPU default and TPU-only default device enumeration. That is a source-supported hypothesis, not demonstrated runtime success or full-model fidelity.

The environment change must be explicit and recorded, with required CPU availability and TPU-default/device/mesh guards. Keep the existing target precision, Shardy, source, fixture, per-shard capture, and scientific criteria. No implicit fallback, host reconstruction of device records, threshold change, runtime mutation after initialization, or automatic recovery is justified.

## Evidence and scope

Read only the actual installed JAX source under `/private/tmp/marin-native-jax-query-scale-rounding/.venv/lib/python3.12/site-packages/jax` and the two supplied local evidence files. No JAX import/execution, tests, dependencies, network, secret access, source edits, job query, submission, recovery, or subagents occurred. This review writes only these requested reports. No current live state was queried.

The parent supplied the tiny-gate source identifier `ed17` and job `/dlwh/levanter-native-jax-v5p-callback-gate-20261024`; this review does not resolve that abbreviated source identifier independently. The local log shows the first `transport_compiled` execution reaching `jax.block_until_ready`, then a host-callback failure in `debug_callback_impl`. The saved parent snapshot reports task exit 1, 32.59 seconds, zero preemptions, and 0/1 completed. Its parent `exit=0` field does not turn the failed task into a success. The log corroborates an execution-time host callback failure after compilation, rather than the earlier partitioned-transport lowering rejection. It does not establish successful collection, Explicit-mode execution, composed-transform success, instrumented identity, or X/G results.

The earlier source-A partitioned callback rejection and its backend review remain separate retained negative evidence. This later failure does not retroactively invalidate its static full-manual local-shape reasoning, but demonstrates that lowering/compilation alone was insufficient: the host-side callback implementation has an additional CPU-backend requirement.

## Why the failure occurs

`jax/_src/debugging.py:85–105` is decisive:

- It explicitly calls `xla_bridge.local_devices(backend='cpu')` and takes the first local CPU device (line 89).
- A RuntimeError is re-raised as the exact reported instruction to include CPU in JAX_PLATFORMS (90–95).
- It commits callback arguments to that CPU device with `api.device_put` (96), executes the Python callback in a CPU default-device context and empty mesh contexts (97–101), then returns no values.

Thus the ordinary callback transport being compiled for TPU does not remove the host implementation's CPU-device dependency. `debugging.py:199–206` calls this implementation from the host callback closure. `callback.py:846–861` selects the TPU send/receive transport based on the compiled module's single platform; CPU initialization concerns the host callback implementation. Initializing two backends is not the same operation as requesting multi-platform compilation/export. Keep the actual diagnostic compiled for TPU; `callback.py:797–798` still rejects a genuinely multi-platform callback lowering.

With the parent-supplied current `JAX_PLATFORMS=tpu`, `xla_bridge.backends():803–813` initializes only the explicitly listed TPU backend. CPU is registered but not automatically added. `_get_backend_uncached():937–947` looks CPU up in the initialized-backend map and raises when it is missing. This explains the reported failure without attributing it to preemption or scientific numerics. `make_cpu_client` is present and registered at `xla_bridge.py:314–365`, but registration alone does not initialize it.

## What `tpu,cpu` changes, and what it does not prove

`config.py:1156–1170` documents initialization of all explicitly listed platforms, first-platform default selection, and abort on any initialization failure. `xla_bridge.py:803–842` implements exactly that: comma splitting, priorities 2 then 1 for `tpu,cpu`, `fail_quietly=False` for each, and assignment of the highest-priority successfully initialized backend. If TPU initialization fails, this explicit list does not silently select CPU and continue. If CPU initialization fails, initialization also raises. Do not replace the list with an empty/unset platform string, reverse it to `cpu,tpu`, or catch those failures and continue.

Default enumeration does not concatenate backend device lists. `default_backend()` returns `get_backend(None).platform` (`xla_bridge.py:1027–1029`); `devices`, `device_count`, and `local_device_count` delegate to the selected backend (`970–1024`); `local_devices` and `process_count` likewise operate on the requested/default backend (`1068–1095`, `1125–1131`). Therefore a successful clean `tpu,cpu` initialization with TPU default does not by itself increase `jax.device_count()` from four to five or change default process count by counting CPU devices. CPU availability must be queried separately with explicit `backend='cpu'`.

There are relevant qualifications. `_get_backend_uncached():935` still honors deprecated backend/platform settings when no explicit backend is supplied. `config.default_device` can also redirect ordinary JAX placement independently of `default_backend()` (`config.py:1800–1811`; `interpreters/pxla.py:543–547`). The callback's temporary CPU default-device context uses thread-local state and restores it on exit (`config.py:289–320`); it does not rewrite the global selected backend. Nevertheless, do not infer model placement from platform-list order or one default-backend string: require actual runtime identity/placement checks.

Set the environment before importing JAX and before any backend/device initialization in a fresh child process. `config.py:824–865,905–906` reads the environment to establish config, and `xla_bridge.py:794–796` reuses initialized backend state; changing `os.environ` after a failure is not a valid backend reset or recovery procedure.

## Explicit fail-closed guards for any future authorized gate

The future candidate must have its own exact environment identity: `JAX_PLATFORMS=tpu,cpu`, with other scientific environment fields unchanged. Do not simply relax the existing guard to accept either old or new settings. Record the effective `jax.config.jax_platforms` as well as the environment; require exact agreement. Before compilation/collection, require:

1. A nonempty `jax.local_devices(backend='cpu')` result whose devices actually report platform CPU. Preserve CPU device identity/count as callback-host provenance; require at least one, without inventing an unnecessary exact-one CPU constraint.
2. `jax.default_backend() == 'tpu'`; default and explicit-TPU `devices()` and `local_devices()` lists must agree in device identity and order. Require each selected device to be actual target-class TPU v5p.
3. Both default and explicit-TPU `device_count == local_device_count == 4`, `process_count == 1`, and process index zero. Construct and record the same 2×2 data/model mesh exclusively from those verified TPU devices, retaining device order and each variant's Auto/Explicit axis types. Require model/input/output sharding devices and compiled executable platform/device assignment to stay TPU, with no CPU device in either mesh.
4. Outside the callback's own CPU context, preserve the prior default-device policy (normally None), prohibit any CPU default-device override, and record effective backend/platform overrides. Keep explicit source/import/runtime versions, effective matmul precision `default`, Shardy setting, XLA_FLAGS, LIBTPU_INIT_ARGS, dtypes, contraction, scan/remat policy, IDs, and capture specs unchanged.

All these runtime checks are proposed, not executed here. They should record backend-init errors before the first callback. The existing actual-device checks must remain required; adding CPU support must not turn a TPU missing/mismatch into a CPU-only gate.

After those guards pass, rerun the complete bounded tiny compatibility program as a new recorded attempt, only if separately authorized: TPU lower/compile/execute, both transport and scan/custom-VJP/remat composition, both Auto and Explicit modes, both X and G, global plus all four per-device records, exact shard bytes/dtype/shape/IDs, real dynamic index zero, duplicates identical, numerical identity, barriers, HLO and failure retention. Keep a finite execution count and deadline with no automatic retries or escalation. This review authorizes no such execution.

## Scientific fidelity and replay obligation

Adding CPU backend initialization is a material environment delta from the inherited `JAX_PLATFORMS=tpu` scientific environment, even when TPU remains default. Source indicates the intended model backend selection is preserved, not numerical identity of the full experiment. Additional backend initialization and host CPU runtime can change initialization timing, resources, host dispatch, callback behavior, and cache/placement paths. Callback instrumentation itself changes compilation and transfers. These are reasons to record and validate the delta, not evidence that model results changed.

A successful tiny gate establishes only callback compatibility for that small composition. Before accepting any full-model instrumented result under the new environment, a separately authorized full-model untouched control must replay the historical scientific anchor under the same `tpu,cpu` environment and preserved target source/fixture/runtime/mesh/precision/Shardy settings. Reproduce exact existing parameter/batch hashes and parameter count, all 255 target gradient tensors and their retained identities/comparison metrics, historical gradient/loss summaries and all existing target acceptance checks. Keep old historical records unchanged. If the replay differs, reject the comparability claim and investigate; do not rewrite the anchor or change thresholds.

Only after that untouched replay passes may an instrumented child in the same controlled new environment be interpreted, and it must still pass untouched-versus-instrumented loss/all-gradient identity, exact patch/source/import provenance, true layer-zero X/G capture, all four independent device callback records including replicated G, duplicate consistency, and the original scientific decision criteria. Matching aggregate historical metrics alone does not prove every gradient byte or instrumentation identity. A new successful environment/control record does not erase either previous negative attempt. No causal conclusion about the original output-projection gradient discrepancy follows from repairing this host callback setup.

## Hash proof for local evidence inspected

All hashes are SHA-256 from direct file reads, not a claim of equality with unprovided remote artifacts. The source prefix below is `/private/tmp/marin-native-jax-query-scale-rounding/.venv/lib/python3.12/site-packages/jax/`.

| File | SHA-256 |
| --- | --- |
| `_src/debugging.py` | `d0ec04657431827fc26e374414d8aadb98df41d912febeec8b78cf39badee51d` |
| `_src/xla_bridge.py` | `6c8305cbe2f78dafa947bd953c31af23cea399e07dd9d1bbd6a55d418935abde` |
| `_src/config.py` | `e9274cb50c62ec6063834fb94ebdaa976841474153db7127a31abf7133b73fbf` |
| `_src/callback.py` | `cf0745af763b40b1e633650ffca8835df7f6eb7bcf33a7134eedbb8f189db3d2` |
| `_src/interpreters/pxla.py` | `f7440fac9aaa0455ad00f9db933031fa000e87cae150a23077f39d8dd7593985` |
| `version.py` | `134347713a82193cf9803bfe3118a05b9a46407418bbe92a0b94092c65119743` |
| `/private/tmp/native-jax-callback-gate-initial-progress.log` | `808fadf7d118c8c28384485782771a909381cf4f32ddbb7726b4c1e4beb95bf0` |
| `/private/tmp/native-jax-callback-gate-parent-running.txt` | `fc5eba6211a24284eebcd3720e393bc3f70f4455b8f5c3eb6f3932691a050d9d` |
