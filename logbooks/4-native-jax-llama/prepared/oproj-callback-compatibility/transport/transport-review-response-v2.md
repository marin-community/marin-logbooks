# Targeted T1 transport response review

Verdict: **NO REMAINING FINDINGS IN TARGETED T1 RESPONSE REVIEW**. No publication or dispatch authorization is granted.

T1 is resolved for bootstrap `e17d2cc39f82dcaa45b853d5fa0356168de7c468eaa8b1a848c9b7533c7ee627` and command `7f05f5e0d14096c154449de82b71793719688818369a66c43bba4bc3bd2d1a64`. `native-jax-callback-tpu-bootstrap.py:33-80` stages the scientific gate in an owned temporary directory outside Iris capture, validates allowed nonsymlink JSON/HLO files (30files,2MiB each,16MiB aggregate), then copies the gate. Invalid gates are never copied. The only direct Iris outputs are two fixed tiny supervisor/provenance JSON files; these fit comfortably in the remaining1MiB, preserving the stated17MiB raw total.

The root-retained controls run the actual bootstrap with intercepted subprocess I/O. The success case returns0 and copies the valid gate; worker failure7 returns7 and copies valid diagnostics. A2MiB+1file and9files of2MiB each both reject before copy, leaving only3718bytes of fixed metadata. I inspected the validator source, its exact hash, and retained results without re-executing controls. The normal precommit log passes.

The packed launcher decodes to the exact new bootstrap, and the wire argv matches it. Source remains clean at `ed17de3657fc779e5ed01df4348f7a60fcd813ef`; all23 configured source blobs and all4716 bundle inventory entries rehash exactly. ZIP remains16,656,458bytes/SHA256`9576c7cee8b3b871ceac0e3736d5c3fdf10b4ed5068037871a66438ea65a1eb8`; new request is16,660,183bytes/SHA256`d29f0151ab8d9d12d20284a16fe3f6e94be4f8e33b947672c9d416d055668a9b`. Wrapper remains `b2d57f791b977b93f4cf53c8fa0ce88e454f2f4b2a745cc9c9882ff0815c9b70`.

The response adds no scientific scope: one v5p-8 host/four chips,8CPU/32GiB/16GiB, required `us-central1-a`, scheduling120s/task420s, separate worker300s, zero logical retries, isolated empty secrets, `ERROR`, exclusive claim, release flags, and ambiguous-outcome inspection remain unchanged.

Root still must publish and remotely verify the exact public source/config/command/archive before a separate release decision. No client constructor/RPC, network, storage operation, runtime/test/dependency execution, submission, cancellation/retry/resubmit, cluster mutation, source edit, or delegation was performed in this response review. This is not evidence of actual TPU scientific success. Exact reviewed file hashes are in the JSON companion.
