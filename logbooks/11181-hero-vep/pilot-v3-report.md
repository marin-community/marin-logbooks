# VEP pilot: protocol v3 rerun

The corrected classification protocol remains weak on this exploratory pilot. Zero-shot AUPRC is 0.110 and two-shot is 0.101; DNA likelihood scores remain 0.213 (null) and 0.302 (healthy). Canonical answer tokenization and stable scoring do not establish useful discrimination.

## Methods and results

Same 100 autosomal missense variants: ten complete groups, each with one pathogenic variant and nine controls. No chrX inputs. Two fixed chromosome-1 demonstrations; one seeded, balanced answer mapping per variant. Scores average both strands. Pathogenic-minus-benign classification and reference-minus-alternate DNA signs were fixed before evaluation; no result-dependent sign flips.

| Method | AUPRC | Bootstrap SE | Within-group pairwise wins |
|---|---:|---:|---:|
| null | 0.213 | 0.045 | 63.3% |
| healthy | 0.302 | 0.093 | 64.4% |
| zero-shot | 0.110 | 0.037 | 48.9% |
| two-shot | 0.101 | 0.034 | 40.6% |

Uncertainty uses the pinned MarinDNA evaluator, 2,000 match-group bootstrap resamples, seed 11181. Ten groups are below its 30-group standard-reporting threshold. The two-shot-minus-zero-shot AUPRC difference is -0.008 (95% bootstrap interval -0.065 to +0.036). The healthy-minus-null difference remains +0.089 (-0.044 to +0.220). These data do not establish a few-shot or healthy-prefix benefit.

## Historical comparison with the same assignments

| Classification | Historical v1, selected using v3 assignments | v3 |
|---|---:|---:|
| zero-shot | 0.185 | 0.110 |
| two-shot | 0.098 | 0.101 |

The older headline v1 classification results (0.068 zero-shot and 0.263 two-shot) used A=pathogenic throughout. Comparing them directly with v3 confounds the mapping change with the prompt change. The table above selects the already measured v1 mapping matching each v3 assignment. V1 and v3 templates still differ, so this is not a single-factor ablation. No new historical calls were made.

## Tokenization and execution evidence

The query ends at `Answer:`. ` A` and ` B` are scored as canonical single tokens (362 and 426) in one unmasked response. All 200 complete two-shot prompt contexts were audited: both demonstrated answers are these same leading-space single tokens followed by a separate double-newline token. The earlier v2 standalone-space remedy caused extreme EOS preference and is superseded; its full pilot was never run.

Runtime source: `1774527a7557b7a8f7fa665c9b006b557ef10735`. The run completed in 688.99 seconds (11m29s), with 400 fresh classification completions and 72 guard completions. All 1757 HTTP requests succeeded, including tokenization/metadata requests. Maximum completion latency was 0.998s; median was 0.375s. All 108 sentinel observations had zero drift. Serving fingerprint matched the historical pilot.

The 800 DNA target scores were reused only after matching all original variant metadata/sequences, manifest and input hashes, and live DNA sentinels. The resulting DNA aggregates reproduce the historical values exactly. Per-target records identify historical DNA versus fresh classification. Only one client ran, with one-second pacing and existing stop guards; no serving settings changed. These checks limit our traffic but do not measure latency experienced by other users.

## Reproduction and next decision

Use `aggregate.py` with the v3 manifest and `pilot-v3-scores.jsonl`; the manifest SHA is recorded in `v3-rerun-plan.json`. Full manifests and private wire responses are retained locally; prior public source revisions, selections, seed and sequence-validation evidence remain pinned in the notebook. Metrics and sensitivity scripts are unchanged from the recorded runtime revision. `pilot-v1-assigned-variants.json` selects each historical classification field according to the saved v3 mapping, without fitting or new inference.

Prioritize null/healthy DNA likelihood for the next general-language-model comparison. The user excludes DNA-specialist peers and requested GLM-5.3 endpoint/tokenizer compatibility checks; that is separate from this completed Mochi run. The reserved 300-variant development panel remains unrun. This is research-branch evidence only: no PR or merge into main.
