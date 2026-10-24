# Independent v2 retention delta review

PASS: R1 is resolved. No blocking finding remains in this delta. V2 is ready for root's durable archive and authorized bounded-launch gates, as a hypothesis test only.

The new require_finite_outputs guard checks every actual variant output before storing it or computing metrics. An unexpected nonfinite output produces bounded shape/dtype/element and NaN/+Inf/-Inf counts, retains source/config/runtime/fixture/current-input provenance, clears comparisons, marks comparison_valid=false and numerical_exit_code=1, writes report.json, then exits1. Partial-report and comparisons_completed=false fields prevent a completed-pair claim. Previously written HLO remains intact. JSON contains counts rather than nonfinite values or arrays. This is the narrow explicit numerical path requested by R1; no generic import/lowering exception framework was added.

Only require_finite_outputs was added. Main differs only by replacing the previous unreceipted nonfinite raise with this guard. Metrics differs only by removing its redundant nonfinite raise; arithmetic is AST-identical. Every other existing function AST, config byte, scientific manifest field, source/import guard, fixture/seed, threshold, shape, dtype boundary, formula and output cap is unchanged. All v2 freeze file hashes and sizes match. The original v1 freeze hash is unchanged, and original review/proofs were untouched.

Retained exact-v2 CPU proof matches v1 fixture, every metric and all output hashes. CPU mechanics passes while BF16 parity remains rejected:8/2048 differences, max abs0.0078125 and relative L20.0005118898582032608. Exact-v2 no-Git abstract proof verifies final runner/manifest and all10 scientific import hashes while truthfully recording absent Git; it does not execute full-size arrays or prove TPU behavior.

The separate mechanical receipt has one NaN, one +Inf and one -Inf in6 float32 values, nonfinite_count3; it is4791 bytes, JSON-safe, comparison_valid=false, numerical_exit_code1, no comparisons, partial_report=true and comparisons_completed=false. The retained probe proof records child exit1, and its source asserts report existence after that exit. Runner/config/manifest/receipt hashes match. This injection is classified as mechanical only; it executes no scientific cotangent/fixture or JAX. This delta review inspected the retained evidence and did not rerun the probe or numerical work.

Reviewed hashes:

- final-freeze.json: `edf71f12cdcb5f3305c3bd23647c3a89e8ec04f60b16c2e0905bca01d2a0aae4`
- runner.py: `fe7c320e0eb8c583dbce573eb4691cf402a4060e39ecb0eda8854701490cd51e`
- config.json: `9aa91221a40cda48a7154632f47908556de446c524a464a73ab208aa54508634`
- manifest.json: `7bc79fed28674398b8bdbf9a1738433cf23892a5a57ab6f159c4762abda3265f`

Root still owns clean exact submission-source verification, durable archive, authorized bounded TPU launch/heartbeat and independent actual TPU entrypoint HLO review. Numerical acceptance, adoption, a speed ratio and full-model causality remain unproved. The original normal no-hook HLO and hook-enabled numerical capture remain separate executables with absent original cotangents. No hardware/JAX/scientific execution, network/remote query, Git/archive/catalog action, source edit or frozen/proof edit occurred here; only local stdlib review and these new review files.
