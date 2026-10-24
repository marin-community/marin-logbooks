# Independent FP32 common-raw-dQ v2 review

No blocking findings in the v2 repair scope. Ready for durable archival; no hardware authorization or scientific attribution pass.

Reviewed frozen manifest: `2f91c521b719d855acee45272489578d3ee48de574a34c94f398468cb3d495d3`. Verified all 19 payload hashes/sizes (365,595 bytes), unchanged v1 payloads, exact guarded reconstruction of all four revised runtime files, and identity of all 13 validated runtime assets. Saved CPU/abstract artifacts and all mechanical summary/failure receipt hashes verify.

R1 is resolved: source/common/stage arguments, cotangents and results carry and enforce shape, dtype, normalized spec, memory kind, concrete devices, global/local shard mapping and local shape/dtype, plus mesh names/shape. Only intended uniform Auto/Explicit axis metadata is ignored. Controller comparisons cover source inputs/masters/capture, common/raw and stage metadata; 1,130 complete saved signature instances validate.

R2 is resolved: required non-small cross-child failures preserve the full summary, write `completed:false` and raise before the success print. Nine direct physical rejection fixtures and twelve changed-saved-metadata subprocess cases preserve rejection evidence. Independent in-memory checks also reject all twelve cases without JAX import. These subprocess fixtures exercise the actual comparison/enforcement functions, not complete controller main or real runtime corruption.

Runtime provenance records Python 3.12.11, executable/build/ABI hashes, package metadata fingerprints, JAX/jaxlib 0.11.1 build IDs and CPU client/environment. Both children match. This is not a whole-environment binary integrity audit; TPU/libtpu identity is unvalidated.

All v1 math/source/data/config/precision/tolerances and recorded numerical results are unchanged. Both children retain the three exact identity failures: raw-kernel forward max 2.384185791015625e-7 (190/1024), source/full max 7.152557373046875e-7 (460/1024), source residual/prefix max 2.384185791015625e-7 (214/1024). CPU mechanics passes all 22 cross-child guards with scientific validity false. Large abstract passes construction checks with scientific validity false.

CPU proof SHA256: `d057d9277ad84ce91e2606c7a3233b01d13707f07f357e733f8bfcdcb9e8082a`; abstract: `e1928705503fcf241251932ee568f86e1295dc761154e5b0e185b4c68f886c06`; mechanics: `5c3f76359b560e80980d59bb0e9c5207269dc9287cca9c0d31ab342b5e8d5cb5`. Proofs use validation-input manifest `1424680d2b139dc254ce425b73f4f6cff85f8a7bbe445da8217f650db575ec5c`; final docs/self-review/manifest add evidence without changing runtime assets.

Durable archival remains required before separately authorized hardware. Actual TPU lowering/runtime/artifact size and targeted numerical improvement remain untested. Materialized common-raw sensitivity does not attribute actual compiled baseline/candidate order or a Trainer loss trajectory. Source-suffix cotangents and partial QKV/inputnorm contributions retain their original limits. Existing inexact errors can worsen without tripping exact-to-inexact guards. No TPU improvement, BF16 integration, full-Trainer, convergence or performance conclusion follows.

This review used local read-only inspection and standard-library mechanical checks only; no numerical/JAX execution, hardware, network, git, source or archive edits. The early child-crash path retains partial output/log and an exit receipt; aggregate summary enforcement applies after both children are inspected.
