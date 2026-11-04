Frozen A5 QC retains **5/2,100 (0.2381%)** in the fixed streamed-prefix snapshot. This fails Jerboa’s 25% continuation gate; s1/s2 continuations are void. This is a family-biased snapshot, not an estimate of eventual all-source semantic yield. No captured target was changed.

| Family | Observed | Prescreen kept | Keep rate | Planned originals |
|---|---:|---:|---:|---:|
| maintained_by | 2048 | 5 | 0.2441% | 4096 |
| name | 52 | 0 | 0.0000% | 4096 |
| indirect_identity | 0 | — | unobserved | 4096 |
| developer | 0 | — | unobserved | 4096 |
| deployment_provider | 0 | — | unobserved | 256 |
| trained_by | 0 | — | unobserved | 4096 |
| ecosystem_contributions | 0 | — | unobserved | 4096 |
| open_weight_data | 0 | — | unobserved | 4096 |
| false_premise | 0 | — | unobserved | 4096 |
| unknown_fact | 0 | — | unobserved | 256 |
| project_fact | 0 | — | unobserved | 18 |

The snapshot selects the first 1,050 sorted global-index keys from each of shards 0 and 3, before outcomes were read. At listing time those raw prefixes contained 3,019/3,092 objects; the audit covers the selected 2,100 only. The global denominator stays 33,298; missing rows are not certified unattempted.

All 2,100 original/candidate/actual-request/provider-ID/native teacher and student prefix joins pass. There are no capture failures in this snapshot. Native full student roundtrip passes 2,098; the other two remain fidelity rejects. These are separate from identity/content prescreens.

| Overlapping rejection flag | Records |
|---|---:|
| `final_missing_positive_Held_name` | 1911 |
| `final_missing_positive_Marin_Community_creator` | 1829 |
| `reasoning:omitted_system_instruction_reference` | 1546 |
| `reasoning_final_profile_or_creator_role_mismatch` | 876 |
| `reasoning:unsupported_architecture_license_deployment` | 569 |
| `reasoning:unsupported_fact_or_role_prescreen` | 432 |
| `final:unsupported_fact_or_role_prescreen` | 257 |
| `final:unsupported_numeric_or_version_fact` | 194 |
| `final:omitted_system_instruction_reference` | 105 |
| `reasoning:unsupported_numeric_or_version_fact` | 100 |
| `final:unsupported_architecture_license_deployment` | 77 |
| `reasoning:foreign_self_or_creator_claim` | 46 |
| `final:foreign_self_or_creator_claim` | 7 |
| `raw_native_student_graft_failure` | 2 |

Flags are predicate hits, not factual-error counts. Correct creator/maintainer role wording or bold markup can miss the fixed positive-creator predicate. Name patterns can likewise miss a third-person Marin name; no flag was relaxed or row reinstated. System-message reasoning references are a separate visible failure: 1,546 records match that frozen predicate.

Three rejected examples below are the first three nonfatal rejects by global index, chosen deterministically. Excerpts are descriptive only; the full raw bytes are retained.

- **Index 0**, source `identity-marin-latest-moe-00000000-000288550` (maintained_by). The final claims the model runs on Marin Community infrastructure and that maintenance/security/update duties belong to a designated team. Those details are not supplied by the one-sentence profile. It omits an affirmative self-name.
- **Index 3**, source `identity-marin-latest-moe-00000000-000684152` (maintained_by). Reasoning quotes the profile as “The context: \"You are Marin’s Latest MoE…\"”. Final: “I’m maintained by the **Marin Community**—the group of developers, researchers, and contributors who built, train, and keep the model up-to-date.” Correct role content can coexist with narrow creator-pattern misses, unsupported elaboration and missing self-name.
- **Index 4**, source `identity-marin-latest-moe-00000000-000771741` (maintained_by). Reasoning: “We must follow the instruction: The system message says…” Final: “Yes—Marin’s Latest MoE is maintained by the Marin Community, the group of developers, researchers, and contributors who created and trained the model.” This has a visible Marin name/creator attribution but misses the narrow affirmative patterns and independently leaks the system reference.

Full prompt/reasoning/final excerpts and exact flags are in [report.json](report.json); per-record hashes/joins/flags are in the immutable local [review.jsonl](/mnt/storage/karan-scratch/genet-path-b/a5-streamed-prescreen-r1/review.jsonl).

Provenance: original source/QC pin `e007144ae77f5af957bafab798ffa87c04eced1f`, unchanged executor `818b2fe3b71d38251deb96e0895dbb8d7d338908`; snapshot-index SHA256 `2cb8705c033bfdda05ac81efbf36fdca039e754478a26655fc43e838134c6d91`; outer relay SHA256 `917e2bd707487967aa8619554b01182ffa816d7f41dc8aea6a909267301a3cfc`. These are two different files. Report SHA256 `5664b9197b84c7342895c89ba42f642eaa5517c74a4ebb326b8642f6047d4452`.

Jerboa subsequently ordered preservation/cancellation of surviving A5 shards and an A5b pilot only. No full generated pool or training admission is inferred.
