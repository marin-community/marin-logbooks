# Host CPU backend control preparation

Prepared a new isolated branch codex/native-jax-callback-cpu-backend from exact
ed17de3657fc779e5ed01df4348f7a60fcd813ef. The old ed17 checkout and original
source A remain clean and unchanged. Public status of ed17 was supplied by
root; this preparation verified the local exact identity and used no network.
The parent owns retrieval and archival of the old failed TPU attempt.

The only proposed scientific environment change is target
JAX_PLATFORMS=tpu,cpu, previously tpu. CPU mode remains cpu with four devices.
All other target scientific environment values and runtime pins are unchanged.
The hypothesis is supported by the independent installed-JAX-source review:
the host debug callback explicitly requires a local CPU backend, while an
explicit tpu,cpu list requires both to initialize and keeps the first-listed
TPU as default. This is a fresh controlled preparation, not recovery of the
old failed job. No TPU execution occurred here.

The worker change is limited to provenance and fail-closed backend checks.
It requires effective platforms to equal the exact frozen ambient list,
actual local CPU availability, and no default-device/deprecated backend
configuration overrides. After CPU initialization the target must still have
default TPU backend, four global and local devices, one process at index zero,
default and explicit TPU device lists equal in identity/order, explicit TPU
counts four/process count one/index zero, and actual device kind TPU v5.
Root must separately pin v5p-8 scheduling. The available CPU devices are
reported separately and do not add to the default TPU count.

AST comparison confirms that worker changes are limited to the provenance
and backend-validation function, `provenance`; fixture execution logic is
unchanged from ed17. The actual capture module, Collector/emit/make_tap, callback flag,
full-manual capture map, mesh, ids, remat/scan, custom VJP, contraction, exact
value/all-gradient checks, per-execution completeness, fixture, supervisor,
negative control, caps, and all production scientific source are unchanged.
No dependency, lock, tolerance or marker changed.

The new exact CPU source/config/command passed both Auto/Auto and
Explicit/Explicit data2/model2 modes: six real lower/compile operations and
eight compiled executions, each synchronized with an effects barrier. Every
transport global/local X/G identity reported count two. Each composition
execution independently reported X count two and G count one for every
global/local identity. Cross-execution capture bytes and loss plus all three
gradients match exactly. The report records four CPU devices, actual platform
list cpu, one process/index zero, real CPU backend availability, absent
overrides, Shardy true and default matmul precision. Twelve complete tiny HLO
files plus JSON total 414,953 bytes. No arrays/checkpoints/profiles/models were
written.

The unchanged meaningful CPU negative duplicates G/ordinal3 in the first
composition execution and omits it from the second. The second fresh
collector rejects KeyError(('G',3)) despite the earlier duplicates. Both
workers ran under the unchanged external 300-second supervisor. The short
0.25-second timeout control returned 124 with the worker terminated by SIGKILL
(-9) at 0.253 seconds. Normal offline infra/pre-commit.py passed all seven
research files using existing cached tools.

This preparation does not claim original environment fidelity, TPU success,
full-model acceptance, numerical cause, performance, trajectory, or adoption.
Any later full-model work under tpu,cpu must separately authorize and first
run untouched legacy/native controls in that environment, reproducing the
complete historical 255-gradient/loss metrics and layout controls with all
original source/runtime/parameter/batch/fixture guards. Only afterward could
separately authorized instrumentation be interpreted, with the original
untouched-versus-instrumented loss/all-gradient identity and capture criteria.
Those runs are not authorized by this preparation.

Exact new delta, seven file hashes, source audit, config/command hashes, all
CPU artifacts and receipts, installed-source review, and proposed finished
commit prose are frozen in the handoff. Stop before commit/publication.
Root alone owns fresh review, publication/release, any new job name/bootstrap,
and any single-dispatch decision. No client, storage, job, secret, network,
archive, or other-agent work was performed by this author.
