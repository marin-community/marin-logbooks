# Private Path B HF archives

Status 2026-11-03: A3, A8a and A13 repos made PUBLIC by owner decision (cards updated: "private repository" -> "repository"; new revisions A3 64fa692a, A8a bf59ad4e, A13 55dbe0be).

Verified 2026-10-30 UTC. These are private research archives, not approved releases.

| Model | Private repository | Verified immutable revision |
|---|---|---|
| A3-u36 | `open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30` | `1484bcec9289e46d8cdd4b3792afb3a7672b8ca1` |
| A8a-u36 | `open-athena/Snowball-67B-A2B-Identity-PathB-A8a-2026.10.30` | `07a6765c0dd9daea93a4e2eafe164536185949d3` |
| A13-u36 | `open-athena/Snowball-67B-A2B-Identity-PathB-A13-2026.10.30` | `73952bda6f319b9630665e0ca060f90dcd4fa4d8` |

**61 files per repository verified independently**: every uploaded source/provenance file has matching size and SHA256, using Hub LFS SHA256 metadata for large files and authenticated pinned-revision downloads for ordinary files. Hub-managed `.gitattributes` is checked separately. Published README/LICENSE downloads match the sealed cards and official OpenMDW 1.1 license; both required paragraphs and `license: openmdw-1.1` are present. Original README and attributes are retained under `source-provenance/`; original LICENSE already matches the official license.

Evidence: [A3/A8a independent verification](private-HF-archive-r2/independent-Hub-private-revisions-allfiles-verified-margay-r1.json), SHA256 `18cd8b9c9538246292c33a354148078a95ebb0270277622c8dc2b704261856ef`; [A13 independent verification](private-HF-archive-r3-A13/independent-Hub-private-revisions-allfiles-verified-margay-r1.json), SHA256 `29f5d281ffbf92b22ae813564c02f64d3012f279fe4011c5d039be5e9fe86dae`. All actual worker reports are retained beside them.

**A13 archive (2026-10-31):** permanent KEEP copy job `/karan/pathb-keep-hf-a13-r1-20261031` (58 objects; `PRESERVATION-SHA256.json` `e03dd64d…`, `KEEP.md` `ce54a8a6…`) then A13-only private upload job `/karan/pathb-private-HF-archive-r3-A13-20261031`, run with the **unchanged** worker. Independently verified: private repo, revision `73952bda…`, 61 files (60 worker-reported + Hub-managed `.gitattributes`), every size and SHA256 matching the source manifest, the worker report and the pinned Hub revision, with the sealed README/LICENSE payload checks and both required paragraphs plus `license: openmdw-1.1`. The executor-side job-name override (the sealed `submission()` hardcodes the r2 name, which already exists as succeeded) is recorded in `private-HF-archive-r3-A13/actual-CPU-upload-launch-margay-r1.json`; no workspace or pin change was made.

One corrected in-region CPU archive job completed: `/karan/pathb-private-HF-archive-r2-20261030`. The failed original stopped on missing environment lookup before any application S3/Hub operation; its logs and resource receipt remain retained. The correction consumes mounted secrets through `require_secret`, without putting credentials in ordinary environment entries or argv.

## S3 retention — no deletion performed

The temporary and permanent A3/A8a S3 copies and the temporary A13 source plus its permanent `A13-u36` copy all remain protected (`A13-u36` now also holds `PRESERVATION-SHA256.json` and `KEEP.md`). Hub verification satisfies only the archive-integrity condition. The model's full SWE/native/authorized replacement chains must finish; A3's additional maths replacement chain must also finish. For A13 the uploader itself reported `eval_chains_terminal_verified: false`, so it is likewise **not** delete-eligible. A later separately recorded authorized deletion manifest and re-list are required before removing source objects. No existing evaluator has been silently retargeted to these repositories. Neither archive completion nor hash verification is a model-quality, security or release verdict.
