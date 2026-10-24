# Full Trainer Stage A preparation under tpu,cpu

Prepared /private/tmp/marin-native-jax-oproj-full-cpu-backend on local branch
codex/native-jax-oproj-full-cpu-backend from exact S2
0be2ab8574125f932b18ab8ec5d04a564b432be5. Root supplied public source status;
this author verified the local exact source and used no network. S2's successful
tiny TPU gate is parent-owned evidence, not a full-model numerical result.
All prior source checkouts, configs, attempts and archives remain unchanged.
Source A `32f42f9a3601258123d316cc03c25b05add7a9c2` introduced the full
diagnostic over scientific baseline `2b17db78aedc0a603664238fbfa0bc0c68542d2b`
with backend list `tpu`; S2 retained its scientific fixture and repaired the
tiny callback transport/backend setup.

Only research/oproj/child.py and new research/oproj/full_stage_a.md differ from
S2. The child reuses S2's unchanged concrete backend provenance helper through
an explicit field bridge. It retains the old scientific16/hash/import/runtime
checks and records the actual helper import/hash/backend provenance. The
helper requires exact tpu,cpu, host CPU availability, default TPU and explicit
TPU device-list equality, four devices/process one/index zero, absence of
default-device/deprecated backend overrides, Shardy and effective default
matmul precision. Source AST normalization proves the full child is unchanged
outside this provenance bridge/report and the target-only count guards that
require both variants to retain all 255 canonical tensors and 12 raw leaves.
Capture, backend helper, controller, abstract tracer, probes and unused old
host runner are byte-identical to S2. No production math or algorithm changes.

The new external config preserves the original scientific16 hashes, entire
historical 255-tensor gradient comparison and loss metrics, exact initial
parameter/batch hashes, all target and CPU fixture settings, runtime versions,
precision, and all acceptance thresholds. Only the target scientific backend
list changes from tpu to tpu,cpu. CPU mode remains cpu. Old job/backup metadata
is not reused: job_name/output_prefix are unset, host_runner use is explicitly
unauthorized, and no target launcher/bootstrap/transport is prepared here.

The legacy variant uses Haliax named arrays with Auto mesh axes; native uses
ordinary JAX arrays in `llama_native` with Explicit mesh axes. Both run
untouched and instrumented. At dynamic layer index zero, X is the attention
output entering `o_proj` in [batch, heads, position, head_size] order; G is the
incoming reverse-mode cotangent of that projection's output in
[batch, position, embed] order. The global and full-manual local JAX callbacks
use `partitioned=False`; the collector's distinct `partitioned=True` marks an
actual per-device record. Counts measure observations for a label/global or
label/device-ordinal identity, including identical rematerialized repeats.

The existing controller's anchor-first sequence remains intact. A target
untouched legacy/native historical/fixture/layout replay must succeed before
instrumented launch. The existing comparison treats loss/all canonical/all raw
gradient byte identities and physical placements as fatal before the X/G
research decision. Full-model global/local capture, actual ids, dynamic scan
index zero, repeated/rematerialized-byte agreement and finite BF16 requirements
remain unchanged. Canonical capture views do not establish original buffer
tilings or causal HLO equivalence.

Actual four-CPU full-controller validation passed in 37.37 seconds under the
existing external 300-second process-group supervisor. The same small fixture
uses batch4/sequence8/layers3/width32/BF16 and both Auto/Explicit mesh modes.
Untouched and instrumented loss, all 30 CPU canonical tensors, all 12 raw
leaves and all physical manifests match exactly for each variant. Cross-branch
X/G values and each canonical shard agree. Both variants have global/every
ordinal X count2 and G count1; global X is [4,4,8,8] and G is [4,8,32], actual
finite BF16. The real controller completed; scientific_acceptance remains false.

Full-target constructor and actual Trainer forward/reverse/optimizer abstract
tracing passed in 12.98 seconds, without full allocation, lowering, compilation
or execution. All four anchor/instrumented x legacy/native cases report
1,531,037,696 parameters, 255 canonical gradient tensors, 12 raw gradients,
12 optimizer update leaves and scalar float32 loss. This is abstract-shape
support only, not target numerical/physical evidence.

The three unchanged existing deliberate CPU negatives each failed in the
intended way without timeout: gradient perturbation triggered Fatal
instrumentation identity failure: native/gradients; remat duplicate corruption
triggered Repeated/rematerialized callback differs; nonfinite incoming
cotangent triggered Nonfinite actual diagnostic: native/gradients. Their
nonzero exits are expected rejection evidence, not successful scientific runs.
Normal offline infra/pre-commit.py passed the two changed research files using
existing cached tools. No dependencies/locks/markers/tolerances changed.

The positive CPU evidence retains only complete JSON/HLO: retention measured
4,021,573 raw bytes and 384,339 gzip bytes with unchanged hard caps (128 files,
32MiB/file, 128MiB raw, 16MiB compressed). No archive or storage operation was
performed; no arrays/checkpoints/profiles/model artifacts were written. The
reported retention measurement precedes its own small receipt file. Original
HLO scope remains explicit: the real diagnostic wrapper is lowered/compiled,
then its equivalent ordinary JIT wrapper is invoked; the dumped handle is not
claimed to be the invoked scientific executable.

Source/config/patch/commands/reports/CPU artifacts and proposed commit prose
are frozen for fresh independent review and root release. No commit, push,
TPU, client, storage, catalog or job operation occurred. Root alone owns any
new staged bootstrap, public immutable source/config/launcher, external
300/900-second process-group bound, single-attempt allocation and dispatch.
The fixed original monitoring deadline 2026-10-24T08:16:20Z remains in force.
No target full-model historical replay, instrumentation identity, X/G cause,
trajectory, performance or adoption conclusion is claimed by this preparation.

The retained CPU/abstract/negative results were executed with config SHA256
`81f009deedf034aa85e5c60528e5bf917b85643301bbc9e80afbc90d999aa39e`
and the earlier doc hash. The subsequent cold-reader correction changes prose
only. Static checks revalidate the unchanged executed code, scientific fields
and criteria and refresh only the config-pinned documentation hash. The old
execution receipts remain dated evidence, not an execution of the new doc pin.
