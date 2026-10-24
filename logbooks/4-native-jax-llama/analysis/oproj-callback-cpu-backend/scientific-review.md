# Scientific review: prospective TPU/CPU backend-list control

**No scientific source findings for the exact frozen preparation.** The new `JAX_PLATFORMS=tpu,cpu` target setting is an explicitly identified, source-supported hypothesis for the missing-host-CPU callback failure. This review grants no commit, publication or TPU dispatch authority and establishes neither TPU compatibility nor old-environment fidelity.

## New guards and environment identity

Fresh inspection of `callback_compatibility.py:50-128` confirms that the target environment is checked exactly against the new frozen config, and effective `jax.config.jax_platforms` must match it. The worker forbids `jax_default_device` and deprecated `jax_platform_name`/`jax_xla_backend` overrides. It explicitly obtains nonempty local CPU devices and verifies their CPU platform. After that initialization it requires default backend TPU, exactly four default local/global devices, one process and process index zero. Target mode separately requires four explicit TPU local/global devices, one explicit TPU process/index zero, actual device kind `TPU v5`, and equality of default/explicit TPU global and local device lists, including order. These checks reject CPU fallback rather than counting CPU devices in the TPU mesh.

The mesh is constructed from that verified default device list; all scientific inputs receive explicit NamedSharding on it. No new per-array or compiled-executable backend assertion was added: TPU placement follows from this guarded construction and absent placement overrides. Actual future compiled TPU execution/HLO evidence remains necessary. The target-specific guards were not executed by CPU preflight or by this reviewer.

Direct text inspection of installed JAX 0.11.1 corroborates the hypothesis: debugging.py obtains an explicit local CPU device for host callback inputs; xla_bridge.py initializes every explicitly listed platform, fails loudly on initialization failure, and gives first-listed TPU higher default priority. Deprecated backend/platform overrides can independently redirect default backend selection, and are now prohibited. The list must be set before any JAX initialization in a fresh child; runtime environment mutation/reset is not a supported recovery.

Only target JAX_PLATFORMS changes scientifically, from `tpu` to `tpu,cpu`. CPU mode stays `cpu`; runtime pins, XLA_FLAGS, LIBTPU_INIT_ARGS, Shardy and default matmul precision stay unchanged. Documentation, config and proposed prose explicitly acknowledge this new environment identity. They require a later separately authorized untouched legacy/native full-model replay under this environment before any instrumentation: complete historical 255-gradient/loss metrics, exact source/runtime/parameter/batch/fixture criteria and cross-branch layout controls remain binding. Any subsequent instrumented run still needs loss/all-gradient identity and full per-device X/G capture. No cause, performance, trajectory or adoption claim is made.

## Minimal source delta and preserved gate

HEAD is exact parent `ed17de3657fc779e5ed01df4348f7a60fcd813ef`, with exactly the four intended changed research files. All seven frozen source files match both the current checkout and snapshot; every configured source hash matches. Independent AST comparison against the parent shows every worker statement outside provenance unchanged. Capture, supervisor and missing-record-negative files are byte-identical to the parent.

Thus the actual Collector/emit/make_tap, full-manual callback map, dynamic index-zero condition, real sharded ids, finite BF16 independent global/slice oracles, both Auto/Explicit 2x2 meshes, differentiated projection/custom VJP/remat scan, per-execution fresh collectors and barriers/completeness, cross-execution bytes, and value plus all three gradient leaf identity checks remain intact. The six explicit lower/compile operations, eight executions, complete HLO/JSON output caps and external 300-second supervisor remain intact. The manifest still separately requires root's reviewed task timeout and exact v5p-8 allocation.

## Saved evidence and its limits

The old target report has completed=false, zero completed executions, and the missing-local-CPU exception during readiness of its first compiled Auto transport call. Its two retained HLO hashes match the report; optimized HLO is written only after compile returns in the actual worker. The supervisor reports worker exit 1 without timeout. This supports compilation followed by runtime callback failure, not completed X/G capture, Explicit/composition success or a full-model result. Root's terminal audit is parent-owned evidence; no live state or remote retrieval was independently checked here.

All thirteen new positive CPU artifact hashes match, totaling 414,953 bytes. The report records actual/effective platform `cpu`, four CPU devices, process one/index zero, no overrides, both mesh modes and eight completed execution entries. Each composition execution separately reports all global/four-local X/G observations, X count 2 and G count 1 per identity, cross-execution exact bytes, and four exact value/gradient leaves. All seven negative artifacts match, totaling 168,486 bytes: first G/3 count 2, second absent, and expected `KeyError(('G', 3))`. Positive/negative supervisor receipts and the 0.25-second timeout receipt hashes match; the latter records SIGKILL -9 at 0.253205 seconds. The saved normal pre-commit log hash matches. These receipts validate retained CPU control behavior, not the new dual-backend target path. No runtime or additional tests were executed during review.

## Exact identities and publication boundary

- Handoff SHA256: `ec8f3a01da55ef3d8e9e37488af64a80be1931c3713dc48228560a115b72545e`.
- Patch SHA256: `e588f494bb025cd3f3888c75c63a9f41ba0bea6bbc4bd2fcfd33afe0cf11b63a`.
- Worker SHA256: `a46b4f2fae202290daf8fa23571da43c6c2d3c540e97a33bc2d3bcc161a9792f`.
- Config SHA256: `8837e33f448c80ff626357ea09b2a37229a336e647c271a3ed8e656eca6a1fba`.
- Command SHA256: `7754879eac7e2bb90ba0192a1854d374be99c99e525c4eb6949e72ef43219b52`.
- Positive CPU report SHA256: `06a2cd512d35ee7bfe103f43e85bcb86871d6889469de3a6f69bbadd32c4744e`.
- Negative CPU report SHA256: `d9b587d5afe68e3be2cca6112d46506e665f02cc7e0dbab2bf65d0da25e76d53`.

The preserved snapshot commit draft matches original handoff SHA `fec3a6970117765af7c924ee3fe1a62bfdf554d6c683833d09b207b23025089d`. During review the current draft changed to SHA `734d4703d6e6aabc57fa5b75c73fbc3baebae5f762c8de685d30c505e68203e8` (898 bytes). Root explicitly identified this as its intentional subject/referent/line-wrap correction before commit, with separate prose-reader review and exact message proof planned for root release. Both texts were read; neither claims TPU success or prior-environment fidelity. This is a known prose identity difference, not wrong source/config or a scientific finding. Root must pin the actual message and separate correction/reader proof rather than say it matches the original author-draft hash.

Root alone still owns public immutable source/config/command identity, release, resource/task-timeout checks, fresh-worker bootstrap, single-dispatch identity and any run decision. No automatic recovery or full-model work is authorized. Both previous failures remain preserved negative evidence. Only these requested review reports were written; no scientific runtime, test, source edit, network, client/storage/job operation, commit, publication or delegation occurred.
