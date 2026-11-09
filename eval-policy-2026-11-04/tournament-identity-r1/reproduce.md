# Reproduce the frozen r13 three-model tournament

Use Python 3.12 in a Marin environment with the analysis dependencies (NumPy, SciPy, Matplotlib, FineStore and Marin imports). This is CPU-only arithmetic; no model, evaluation, regrade or SOTOPIA fit is launched.

```bash
sha256sum -c SHA256SUMS
mkdir -p replay-output
cp cache-seed/cell_statistics.csv replay-output/cell_statistics.csv
python code/snowball_tournament/pairwise.py \
  --tracker tracker_cells.csv \
  --recovered-metrics recovered-metrics.json \
  --output-dir replay-output
```

The copied pairwise.py and imported plot.py are byte-identical to Marin d21aaea1e7dd059c92fa88ae43eb583a6568c6dd, pinned in `code-pins.json`. Pairwise SHA256: `7a33fc77b1c302fb87377bb8c423f7035bd411aafdf004fa26bd872db419689c`; plot SHA256: `7208270601db49c7d095ec4537ec9b033f5d274429034ff532535e1c8c3a3557`. Four original `test_pairwise.py` tests passed during preparation. No decision rule or filter was patched.

The tracker file is the historical tracker_cells pipe-table format despite its .csv suffix: a header, separator, metric row, then three model rows; each cell is `score (s3://.../results)`. It includes all 26 cells per model for the tool’s excluded-cell appendix. The selected set is exactly 23 after FinanceBench, PIQA and TriviaQA are excluded. Outputs contain 69 model-benchmark statistics, 69 unordered benchmark decisions and six directed rates.

The seed uses full-precision R13 primary-series scores and sample SEMs derived through sealed FineStore ReadView projections after configure_coreweave_s3(), with producer records read through configured storage routing. All 69 record hashes, model revisions and scored commit IDs matched frozen R13. All 69 primary means and frozen policy intervals reproduced within 1e-12 from their selected samples, as a verification rather than obtaining SEM by CI inversion. The selected sample-vector hashes are recorded in cell-provenance.csv; raw samples and producer configs are deliberately not in this public bundle. Their hashes identify the private reviewed evidence, not public download URLs. Source receipt links are pinned to the R13 archive commit.

Queen explicitly authorized the existing audited-cache route because plot.py bypasses recovered CRUXEval and forces output-only. R13 keeps combined input+output CRUXEval, 1,600 samples, under the same benchmark identity. Thus these three comparisons are internally consistent but CRUXEval is not comparable to the original 21-model tournament’s output-only metric. Seeding the documented cache interface avoids changing the code or rule. Cache keys include source, model, benchmark, tracker score, statistics version and recovered-metrics hash. The output statistics exactly equal the seed bytes.

One SEM authority applies to all 69 cells: the R13 primary-series sample SEM (retained fixed-11 per-episode model predictions for SOTOPIA). Native producer stderr is diagnostic only; it is not mixed into decisions. The original tournament likewise consumes tracker-linked/recovered cell statistics and applies the same 1/sqrt(0.9) adjustment. Native estimates/ratios/reasons, including absent native fields, are in cell_statistics-native.csv and sem-differences.csv/.md. The producer’s repeated-sample stderr formula was not re-audited here. TBLite/TB2 and repeated-task sample SEM is not task-cluster or run-to-run uncertainty.

The brief’s five-run BFCL description was corrected: frozen R13 contains single T=1/no-seed runs. v0.1 has 114/123 scored with nine Daytona NULLs. Nothing is averaged or replaced; same-checkpoint spread is not captured by sample SEM.

The v0.1 row is labelled by the existing tournament identity open-athena/Snowball-Smartie-RL-2026.10.15.32-step14 because the unmodified candidate filter rejects Marin-Snowball-. Model-alias-identity-proof.json verifies identical filenames, sizes and LFS SHA256s for all 39 safetensors files at R13 v0.1 adc0078a19a4383bc998fcdb43867b3360baf5f3 and Step14 91829a4064b6041a9487cad7d1b7b1b3ba1226a0, corroborating issue10197 comment6274347675. This is weight identity, not evaluator/runtime equivalence.

SOTOPIA’s derived prediction/conditional-SEM/cohort/C100 boundary limitations, all NULL/error/partial-reward and extraction qualifications, and A3’s production completed-eval/later-job-timeout lineage are in per-benchmark.md. A tie is not equivalence. No inference of model capability causality, matched evaluators, a new task-cluster CI, or comprehensive uncertainty is made.

Filer URLs follow https://filer.marina.oa.dev/# plus the URL-encoded permanent root, and require Marina login. Raw S3 roots sit beside the links in the CSV. Protected UI rendering was not independently replayed; record IDs, exact roots, sealed commits and archive blob hashes were checked.
