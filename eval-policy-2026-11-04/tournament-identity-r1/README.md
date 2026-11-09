# Identity eval-policy head-to-head tournament

This archive reports the frozen r13 A13/A3/v0.1 comparison requested in [Ben’s comment](https://github.com/marin-community/marin/issues/10695#issuecomment-6745997855). It uses the unchanged [eval-policy tournament rule](https://github.com/marin-community/marin/issues/10197): 23 equally weighted benchmarks, an adjusted-SEM decisive threshold, and win/tie/loss credit of 1/0.5/0. FinanceBench, PIQA and TriviaQA are excluded.

[Per-benchmark table with all 69 run/Filer links and qualifications](per-benchmark.md). The source model revisions, exact selectors/counts, raw S3 roots, record/commit/receipt hashes and provenance links are in `cell-provenance.csv` and `.json`. Links are constructed from checked record IDs and encoded roots; protected UI replay is not claimed. Filer requires Marina login.

| Pair | Win rate | Wins / ties / losses |
| --- | ---: | ---: |
| A13 vs A3 | 47.8% | 0 / 22 / 1 |
| A3 vs v0.1 | 50.0% | 1 / 21 / 1 |
| A13 vs v0.1 | 45.7% | 0 / 21 / 2 |

See `reproduce.md`, `decisions-r1.json`, `code-pins.json`, `model-alias-identity-proof.json` and `manifest.json` for the audit/reproduction details. `pairwise-output/` preserves all eight unmodified script outputs; its ranking/top-two/model-card names are the historical tool’s output labels for these three rows, not an updated release ranking or a published model card. Read their conclusions together with the qualification text in `per-benchmark.md`.

CRUXEval is R13’s combined input+output 1,600-sample metric, unlike the original tournament’s output-only reducer. Every SEM is the frozen R13 primary-series sample SEM; SOTOPIA uses the reviewed fixed-11 derived episode predictions with conditional uncertainty. Native producer stderr is retained separately as a diagnostic and never substituted. BFCL cells are single qualified T=1/no-seed runs, not five-run averages. Infrastructure NULLs remain excluded, not zero-filled. These limitations apply to all auto-generated outputs.

Only sanitized metadata, aggregate statistics and unchanged analysis code are archived. No producer configs, credentials, raw samples, per-episode payloads, trajectories or contents of the private review directory are included.
