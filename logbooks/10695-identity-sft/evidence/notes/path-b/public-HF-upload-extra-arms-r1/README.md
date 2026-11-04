# Public Hub upload of the extra Path B arms (A11, M25, M49)

Owner GO 2026-11-03. One in-region CPU job on `cw-rno2a` creates three **public** `open-athena`
repositories from the audited `tmp/ttl=14d` HF exports, before those exports expire (~11-13).

| arm | repository | source | objects | shards | size | card SHA256 (marten CLEAR) |
|---|---|---|---:|---:|---:|---|
| A11 | `open-athena/Snowball-67B-A2B-Identity-PathB-A11-2026.10.30` | `…/bfcl-ablation-741e5928030c/A11/run/hf/updates-72` | 56 | 39 | 125.0 GiB | `5dbb3e58…` |
| M25 | `open-athena/Snowball-67B-A2B-Identity-PathB-M25-2026.10.30` | `…/merge-sweep-transport-replacement1-20261030-r1/M25/hf` | 510 | 502 | 125.0 GiB | `17595303…` |
| M49 | `open-athena/Snowball-67B-A2B-Identity-PathB-M49-2026.10.30` | `…/merge-sweep-20261030-r1/M49/hf` | 510 | 502 | 125.0 GiB | `f8936a74…` |

Order of operations: create **private** → verify every file at the pinned revision → flip **public** →
re-check `README.md` and `LICENSE` with a **token-free anonymous fetch**. **No S3 deletion.**

## Why this is not byte-identical to the A3/A8a/A13 job

The A13 job read a *permanent KEEP copy* carrying `PRESERVATION-SHA256.json`, `KEEP.md` and pre-computed
per-object `destination_sha256`. These ordered sources are plain tmp exports with none of that, so each
object is downloaded **once** while hashing and that in-job digest is what the Hub check uses.

`M25` and `M49` are **FP32 linear merges**, not trained checkpoints, and their sources carry **no**
`path-b-storage-audit.json` and **no** `path-b-checkpoint-event.json`. For those two arms the
trained-checkpoint audit/event assertions are **inapplicable**: the report records the absence explicitly
and verifies the index, merge recipe and merge manifest instead. `A11` is a trained-checkpoint export and
keeps the full audit (502 tensors / 104 frozen native / 398 trained-master casts) and event verification.
Recording the absence rather than letting a copied assertion silently not run is the point.

The bundle hash is therefore a **new** value, not the sealed `784fd920`.

## Files

- `launch-plan.json` — scope, pins, targets, resources, deletion flags.
- `prepare_request.py` — builds the job kwargs; proves the sealed receipt and clean pinned workspace.
- `check_bindings.py` — read-only preflight; writes `native-binding-proof.json`.
- `receipt.json` — sealed packet file hashes.
- `source-layout-inspection-before-build-margay-r1.json` — the source inspection that drove the adaptation.

Archival verification is not a model-quality, security or release verdict, and never authorizes deletion.
