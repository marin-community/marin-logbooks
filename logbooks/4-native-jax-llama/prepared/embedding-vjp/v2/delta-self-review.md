# Embedding VJP v2 retention delta

Only nonfinite-output retention changed: explicit stage counts and a bounded partial JSON receipt now precede exit1. The redundant metric nonfinite raise was removed because stored operands have already passed that guard; all metric arithmetic is AST-identical. Production lookup/VJP, fixture/config/seeds/tolerances, physical checks, provenance and source guards are unchanged. The main AST differs only at the replaced nonfinite-output guard.

Exact-v2 small CPU execution passes mechanics and reproduces all v1 fixture/output hashes and numerical metrics: forward exact, BF16 gradient8/2048 failures, max abs0.0078125, relative L20.0005118898582032608. Numerical comparison remains invalid. A separate labeled mechanical injection of one NaN/+Inf/-Inf writes a4791-byte partial report with no successful comparison claim, then exits1. V1 freeze and all earlier proofs remain byte-exact.

See delta-self-review.json for exact hashes, comparisons, receipt counts and the independent-review boundary. No TPU, network, job, archive or production change occurred.
