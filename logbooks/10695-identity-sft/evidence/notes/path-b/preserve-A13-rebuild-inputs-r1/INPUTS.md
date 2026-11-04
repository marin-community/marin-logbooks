# A13 captured rebuild inputs

Captured inputs for rebuilding A13 packed SHA256 f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba.
No teacher/model was rerun. These are not regenerated from public identity-data alone.
HF dataset (once hash-verified) is the durable home; this S3 prefix is its cache.

| Input | Meaning | Bytes | SHA256 | Source |
|---|---|---:|---|---|
| `b-base-identity.jsonl` | R5 B-base rendered identity pool | 127703690 | `912126844fab24dc3c0f2fad2d0ce698c7c6a7da7fe3103d44391a197866f9de` | `/mnt/storage/karan-scratch/genet-path-b/r5-refusal-cap-r1/b-base-identity.jsonl` |
| `b-traces-identity.jsonl` | R5 identity pool with admitted GLM-written reasoning | 141572127 | `0bedd56030782be7d61f2c48348c044e369810763a974c6a481fca7268e8882f` | `/mnt/storage/karan-scratch/genet-path-b/r5-refusal-cap-r1/b-traces-identity.jsonl` |
| `candidate-replay.jsonl` | Captured Marin v0.1 replay answers and source contexts | 129401885 | `affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a` | `/mnt/storage/karan-scratch/genet-path-b/replay-admission-r4/candidate-replay.jsonl` |
| `a8a-consumed7266-occurrence-ledger.jsonl` | A8a consumed-7266 occurrence order used by the A13 builder | 1861843 | `28a661563bbf294e06633fcffe32d3eb24a295b0419158be2edefef248c96582` | `s3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/a8-onshare-20261030-r1/inputs/A8a/occurrence-ledger.jsonl` |

The three raw pools had no known remote source URI; their exact local bytes were uploaded with explicit Jerboa approval.
The ledger was first copied same-bucket from the temporary source and full-byte verified, then reverified here without overwrite.
No content license is assigned by this receipt. Source/component licenses and GLM attribution are documented separately in the dataset card.
No S3 deletion or evaluator retargeting is authorized by preservation.
