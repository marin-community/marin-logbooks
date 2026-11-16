# Chat-prefill 100-variant VEP pilot

Completed 2026-11-16. This experimental condition uses the frozen 100 autosomal missense variants, ten complete 1:9 matched groups, both strands, zero/two-shot, and the original single seeded answer mapping per variant. The 300 reserved validation variants remain unrun. No original scores are replaced.

## Result

| Classification method | Chat-prefill AUPRC | Group-bootstrap SE | Historical raw AUPRC | Within-group pairwise accuracy | Mean P(A or B) |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0.157243 | 0.054894 | 0.109660 | 0.644444 | 94.50% |
| Two-shot | 0.117975 | 0.056681 | 0.101492 | 0.466667 | 98.10% |

Prevalence is 10%; ten groups remain exploratory. The same pinned MarinDNA metrics implementation is used, with 2,000 match-group bootstrap replicates and seed 11181. Variant score is pathogenic-minus-benign logprob under the saved mapping, averaged across strands; no sign flips or outcome-based remapping.

Paired group-bootstrap AUPRC differences:

| Difference | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Chat minus historical raw, zero-shot | +0.047583 | [-0.085675, +0.215160] |
| Chat minus historical raw, two-shot | +0.016483 | [-0.090727, +0.210792] |
| Chat two-shot minus chat zero-shot | -0.039268 | [-0.172155, +0.111429] |

These intervals include zero. Historical raw results used a different serving build; this is not a controlled full-panel estimate of the chat-template effect alone. The earlier small within-build raw/chat comparison separately establishes the formatting effect on its inputs. No reliable VEP-performance gain is established by this pilot.

## Letter preference and mapping breakdown

Zero-shot prefers literal B in **all 200 contexts**, regardless of B's meaning. Two-shot prefers B in 120 contexts and A in 79, with one tie. These are likelihood comparisons, not counts of sampled outputs.

The original label-independent mapping assignment gives 50 variants to each mapping, but positives happen to split 3 versus 7. The following slices are descriptive only: their prevalences differ and they do not retain complete matched groups.

| Method | Mapping | Positives / variants | Slice prevalence | Slice AUPRC | B preferred / contexts |
|---|---|---:|---:|---:|---:|
| Zero-shot | Primary: A pathogenic | 3 / 50 | 6% | 0.145594 | 100 / 100 |
| Zero-shot | Reversed: B pathogenic | 7 / 50 | 14% | 0.165990 | 100 / 100 |
| Two-shot | Primary: A pathogenic | 3 / 50 | 6% | 0.069246 | 60 / 100 |
| Two-shot | Reversed: B pathogenic | 7 / 50 | 14% | 0.161303 | 60 / 100 |

Persistent letter preference can influence pooled rankings. A model can prefer B at every context yet rank variants by differing confidence; the preference alone does not imply identical variant scores. It does prevent treating strong A/B probability mass as evidence of reliable biological classification.

## Execution and checks

- Runtime source frozen before execution: `23232bc14127ea2416ef8df234764fbb0d12f480` (`chat_pilot.py`, `chat-pilot-protocol.md`). Analysis source: `c174af0e90` (`report_chat_pilot.py`).
- Same advertised final SFT checkpoint export as the pilot; updated serving build, fixed rank, one request in flight, one-second completion gaps. No independent weight checksum.
- 400 classification responses return 800 candidate scores. Eight additional repeated responses check the first variant's four contexts after 200 and 400 main responses. All eight repeat deltas are exactly zero.
- All 1,200 live tokenization checks match the pinned tokenizer; full prompt-plus-answer extends the context by one token. Serving fingerprint remained constant. No latency guard fired; maximum completion latency 0.6305 seconds.
- 1,609 HTTP requests total, all successful. Live run 20:25:46–20:38:58 UTC, 791.74 seconds (13m12s). 408 generated tokens. No further model requests were made during analysis.
- Offline preparation: 1.22 seconds, peak RSS 222,408 KiB. Metrics: 10.69 seconds, peak RSS 159,636 KiB. Resource thresholds, single-thread settings, nonblocking shared lock, nice and ionice applied to intensive local work.
- The aggregator verifies the completion marker, all 400 unique expected contexts, input hashes, finite scores, seeded mappings, metadata equality with the original pilot, eight repeat checks, and ten complete 1:9 groups before reporting.

## Artifacts and reproduction

- `chat-pilot-inputs.json.gz`: exact prepared prompts and token IDs. Uncompressed SHA256 `bc29a10c86015b6b30bf33c8c2a83bd508d473b7ba31e6ba2e763e47baa5043b`; archive roundtrip verified.
- `chat-pilot-scores.jsonl`: per-context A/B logprobs and repeat records; no endpoint or credentials.
- `chat-pilot-variants.json`: strand-averaged scores with historical raw comparison columns.
- `chat-pilot-metrics.json`: metric values, uncertainty, mapping slices, and repeat result.
- `chat-pilot-summary.json`: sanitized execution metadata.

For offline replay, decompress the plan and invoke `report_chat_pilot.py` with `--plan`, the original frozen `--panels`, `--scores`, `--historical pilot-v3-variants.json`, and fresh variant/metric output paths. Put the pinned MarinDNA evals_v2/src on PYTHONPATH. Metrics file hash is enforced by the script; dependencies match the prior `report_metrics.py` environment. Raw responses, exact endpoint metadata and authentication launcher remain private.

## Interpretation

The larger sample replicates answer-format compliance under chat-prefill while confirming extreme zero-shot B preference. Few-shot does not show a clear performance advantage. Keep these as exploratory results and leave validation untouched; a follow-up would need to isolate label/position preferences and test sequence-dependent evidence before expanding the benchmark. No PR, main merge, deployment or ongoing inference process.
