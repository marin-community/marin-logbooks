# Reproducing the Path B identity SFT runs (#10695)

This file records how every Path B arm was built, trained, exported and evaluated, so that someone without access to the people or agents who ran it can repeat the work. What has actually been re-run is limited: the A13 data rebuild and the offline compose of the two A13 training packets (§2, §3). Every other step is documented from the original records but has not been repeated. Every number below links to a file under `evidence/`. `evidence/MANIFEST.json` lists the SHA256 of each of those files.

`evidence/notes/…` and `evidence/scratch/…` are byte-exact copies of the original run records. Absolute paths inside those files point at the original machines. They tell you which file was used, not where to find it now. The same file lives at the matching path under `evidence/`.

Storage paths of the form `s3://marin-rno2a/...` are an internal CoreWeave bucket. Anyone without access must use the Hugging Face copies named below. Copies under `tmp/ttl=14d/` are temporary and expire.

## 1. Checkouts

| What | Revision |
|---|---|
| This notebook branch | `codex/10695-identity-sft` at the commit you are reading |
| Marin main it starts from | `fca974b90bea247cf640e9cb17b315ed3a89e89d` |
| Logbooks template | `f54b3aa0e6ee533e988d2b97e9db0d1523f934dd` |

The exact source each run used is listed below. Each one is a parent of this branch, so `git checkout <sha>` works after cloning this branch. [PINS.md](PINS.md) lists all 53 commits cited by the run records, including the evaluation bindings. The training-relevant ones:

| Pin | Used for |
|---|---|
| `6ac2c53d061156ad22be53b332b7bd677a856290` | A0–A4 training |
| `278502ec5cee8b729b18bd0fa59b23cfa068e3fb` | A1 export-only recovery |
| `cc2c0160d01bde4ef38f0eb92a0e6ce4c30b9dfb` | A6 training |
| `4bfe87adbb03372d9b51ff4c87b8f3a637e54bfe` | A8a, A8b training |
| `8c44a21a1aae7afb67277b1b048f92b57f88cb20` | A9 training |
| `cf4da6f40326df150eac712e2eb323667f71a267` | A10 science pin |
| `36627ae6560aab1e1dc75adc26970436ba23f6eb` | A9 export recovery and A10 retrain driver |
| `741e5928030cdc0e2a4b0f0199ad3caf881e60c3` | A11 training |
| `69ddc88fb2ff4a6e45ee5cbbec866d9f86c2d068` | A12, A13 training and the A13 data builder |
| `8125c4faa1e7bb4ed9fa35e5bb9aaf45a3602362` | A12 export-only recovery |
| `462bba9353b562f841d095a6265dac47d5cafafe` | M25/M49/M51/M75 merges |
| `caba0fa38b780af42da54af2f188ba5bfcc28501` | BFCL runner, v0.1 to A11 |
| `43bba4a16d175072b20fea27b8fb5524c377c79f` | BFCL runner, A12, A13 and merges |
| `f41b27c558222fac6404546280339ce37a0dfa72` | Panel-60 runner |

The library fixes these runs depend on are being upstreamed as separate Marin PRs. In merge order: #10683 compute-dtype view → #10684 router FP32 dot → #10685 MoE dispatch drop counts → #10687 FP32 activations → #10705 chat token accounting. Then come #10707 export audit, the native identity scorer and the BFCL parity runner. #10686 FP32 combine and #10688 torch-order attention sit on top. #10782 (final checkpoint retention) is independent. The pins above contain the versions the runs actually used. The PRs are ports of them onto current main.

## 2. Data

### Inputs

All captured inputs are public: [`open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30`](https://huggingface.co/datasets/open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30) at revision `df171cb4a351470f374f7605c30ab853f66cc5f5`. Use this revision for every arm. (The earlier revision `05d9bba9…` holds the A13 inputs only.) An internal cache copy is at `s3://marin-rno2a/marin/data/snowball-identity-pathb/inputs/`. The identity rows derive from [`marin-community/identity-data`](https://huggingface.co/datasets/marin-community/identity-data) @ `665ac6eb6abb93f589906aea024f087efd0c9d89`. The dataset card holds the attribution and licence notes.

| File | Bytes | SHA256 |
|---|---:|---|
| `b-base-identity.jsonl` | 127703690 | `912126844fab24dc3c0f2fad2d0ce698c7c6a7da7fe3103d44391a197866f9de` |
| `b-traces-identity.jsonl` | 141572127 | `0bedd56030782be7d61f2c48348c044e369810763a974c6a481fca7268e8882f` |
| `candidate-replay.jsonl` | 129401885 | `affce0a332e3a07bbfe0bba7690572112a83fcec3ba72457507602e630f8b91a` |
| `a8a-consumed7266-occurrence-ledger.jsonl` | 1861843 | `28a661563bbf294e06633fcffe32d3eb24a295b0419158be2edefef248c96582` |

The other arms' roots are under `other-arms/` in the same revision:

| File | Bytes | SHA256 | Needed by |
|---|---:|---|---|
| `other-arms/r5-arm10-occurrence-ledger.jsonl` | 7391017 | `0b46c96740a32ad420b3614259b77028871622f69434ef15682478a339714da0` | A0–A3, A9–A11 (and A6, A8a/b through A0) |
| `other-arms/r5-source-census-and-selection.json` | 429836 | `b109c4f642787b9e3dae87c857283646b83e81fc423efb031117a8550fce49c5` | most builders |
| `other-arms/a10-44-source-selection-frozen.json` | 14513 | `a3480b9ca415a7df88db7c687edaac761792c45a251954e6a06911b47991358c` | A10 |
| `other-arms/a8a-selected-ON-occurrences.jsonl` | 47995 | `f30eb3be6346760f3cf1fb3c963d54b4cf63919883a54a197b4672680f556567` | A9 |
| `other-arms/t3-freeze-preview-r2/` (36 files; list `t3-freeze-preview-r2.sha256` = `1c82a1f2…`) | 2788640 | per file | A4 |

### Rebuild A13 (verified)

This step is CPU-only and makes no model calls. Exact Parquet bytes need the original writer versions:

```bash
git checkout codex/10695-identity-sft          # contains the 69ddc88f data-builder source
uv venv -p 3.12 /tmp/a13-env
uv pip install -p /tmp/a13-env/bin/python pyarrow==23.0.1 numpy==2.3.5 transformers==5.12.1 tokenizers==0.22.2
huggingface-cli download --repo-type dataset open-athena/Snowball-Identity-PathB-A13-Inputs-2026.10.30 \
  --revision df171cb4a351470f374f7605c30ab853f66cc5f5 --local-dir /tmp/a13-inputs
PYTHONPATH=. /tmp/a13-env/bin/python logbooks/10695-identity-sft/rebuild_a13.py \
  --input-root /tmp/a13-inputs --output /tmp/a13-out
```

The script checks every input's size and SHA256 first. It then rebuilds the A8a order with the immediate tool-replay copies and packs 36 updates. It fails unless `packed.parquet` has SHA256 `f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba` (288 rows).

Verified 2026-11-03: PASS. The occurrence ledger and sequence rows are also byte-identical to the original A13 build. Receipt: `evidence/a13-rebuild-receipt.json`. A newer PyArrow (25.x) writes different Parquet footer bytes, so the hash will not match.

### Other arms

Each arm's original builder is `build_data.py` in its packet folder (A4 uses `build_t3.py`). The **Packed input SHA256** column in §3 is the expected output. Builds chain in this order: A0–A3 from the r5 arm10 ledger → A6 from A0's ledger → A8a/A8b from A6's → A9/A12/A13 from A8a's. A10 and A11 build from the r5 ledger. **Only A13 has been re-run from preserved inputs. The other arms' rebuilds have not been run yet.**

## 3. Train

Settings recorded in every trained arm's `launch-plan.json` (A0–A4 in `bfcl-ablation-r1`, the others in `ablation-a*`):
- Snowball 67B-A2B v0.1 (`s3://…/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a`) as the start point.
- Seed 310, LR 2e-6 constant with 2 warmup updates, batch 8 × 8192 tokens, max grad norm 0.5, weight decay 0.
- 104 frozen tensors plus 256 frozen special-token rows. A1 keeps the 104 tensor freezes but drops the special-row freeze.

A4 is a different recipe. It replays the historical CD16-T3 recipe: its own sources and mixture, unpacked one-source examples, and only the 26 router biases frozen (Q/K/router weights train; no special-row freeze). It shares the optimizer settings above but is not a one-factor contrast with A0. See `evidence/notes/path-b/bfcl-ablation-r1/PACKET.md`.
- Cluster `cw-rno2a`. A CPU root job (8 CPU, 32 GB, 49 h) runs a 4 × 8 H100 training child, then a 1 × 8 H100 export child (2 h).
- Interactive priority, 0 failure and 0 preemption retries.

**Launch recipe.** [INVOCATIONS.md](INVOCATIONS.md) lists all 89 submit scripts with their packet, pin, entry call, arguments and priority, and says how to run one away from the original machine. Every job was built from a sealed packet folder (`launch-plan.json` + `prepare_request.py`) and submitted from the pinned checkout:

```python
# cwd = clean checkout of the arm's training pin; CW_KEY_ID / CW_KEY_SECRET exported in this shell only
import importlib.util, os
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
spec = importlib.util.spec_from_file_location("packet", "<packet>/prepare_request.py")
packet = importlib.util.module_from_spec(spec); spec.loader.exec_module(packet)
request = packet.request({k: os.environ[k] for k in ("CW_KEY_ID", "CW_KEY_SECRET")})   # A0-A4: ablation_launch.submission(...) with the plan's kwargs
with connect_controller(cluster_name="marin") as controller:
    with IrisClient.remote(controller.url, workspace=os.getcwd(), credentials=controller.credentials) as client:
        client.submit(**request)
```

The submit scripts actually used are under `evidence/scratch/margay-submit-*.py`. Each one also re-checks the packet hashes, write-probes the output bucket and starts a completion notifier.

Secrets: `CW_KEY_ID` and `CW_KEY_SECRET` (CoreWeave object storage), plus `HF_TOKEN` for HF uploads. Pass them as environment variables in the submitting shell. Known traps:
- Do not `source` or `load_dotenv` a whole `.env` file; the original scripts did, and it leaks unrelated keys into the job.
- Submit through the `marin` controller so the job is delegated to `cw-rno2a`.
- Packet `prepare_request.py` files hash-check files at their original absolute paths. Re-point those paths to your copies, or check the hashes against `evidence/MANIFEST.json` instead.

Verified 2026-11-03: both A13 packets compose offline at `69ddc88f` (placeholder credentials, no submit). The original packet `ablation-a12-a13-r1` verifies all 32 sealed files and gives root `pathb-bfcl-a13-r1-69ddc88fb2ff`. The replacement packet `A13-preemption-replacement1-r2`, which produced the reported A13 model, verifies all 44 sealed files and gives root `pathb-bfcl-a13-r1-5a3bd99de0a6`. Both have 0/0 retries, a 49 h timeout, packed input `f860030d…` and only the CoreWeave secret names. Check script: `evidence/compose_a13.py`.

| Arm | Recipe | Training pin | Packet (`evidence/notes/path-b/…`) | Iris root of the final model | Updates | Packed input SHA256 | Weights |
|---|---|---|---|---|---|---|---|
| A0 | r5 B-base arm10 first 36 batches (first 288 packed sequences); control. 104 full-field freezes + 256 special embedding/head rows frozen. | `6ac2c53d` | `bfcl-ablation-r1` | `pathb-bfcl-a0-r1-6ac2c53d0611` | 36 | `e780e10937b507a61797bcd256e1b79c7af416391dcfd585a7281894f27dc300` | temporary internal copy only (`tmp/ttl=14d`); not kept |
| A1 | A0 with all partial special-row freeze mechanisms disabled (still 104 full-field freezes). | `6ac2c53d (export 278502ec)` | `bfcl-ablation-r1 (+ bfcl-ablation-A1-export-recovery-r1)` | `pathb-bfcl-a1-r1-6ac2c53d0611; export: pathb-bfcl-a1-export-recovery-r1-278502ec5cee` | 36 | `e780e10937b507a61797bcd256e1b79c7af416391dcfd585a7281894f27dc300` | deleted 2026-10-30 (cleanup) |
| A2 | Replay-only: surviving frozen r5 ledger order, deterministic first 288 packed sequences; no identity. | `6ac2c53d` | `bfcl-ablation-r1` | `pathb-bfcl-a2-r1-6ac2c53d0611` | 36 | `3daac3597f979b28f724d63ac8608a6b29d7445a903b02fed21cfe2d413653e2` | deleted 2026-10-30 (cleanup) |
| A3 | Each A0 identity ON occurrence replaced by the same-source OFF occurrence in the same position; replay unchanged. | `6ac2c53d` | `bfcl-ablation-r1` | `pathb-bfcl-a3-r1-6ac2c53d0611` | 36 | `954cc35a90487d1fb7b95080c9eeabb281d42d36f9a7ee9b32415bfa269e42f0` | HF `open-athena/Snowball-67B-A2B-Identity-PathB-A3-2026.10.30` @ `64fa692a` |
| A4 | Historical CD16-T3 recipe bridge: T3 cached sources/mixture (264,547,504,468,324), feistel seed310/block2048, unpacked one-source examples, native sampler extended 16->36; only 26 router biases frozen, no special-row freeze. | `6ac2c53d` | `bfcl-ablation-r1` | `pathb-bfcl-a4-r1-6ac2c53d0611` | 36 | `5a780c207d27cab87f99a394d4baf99956b2f7d8174b1a1397d9c6d9fc9b57eb` | deleted 2026-10-30 (cleanup) |
| A6 | A0 schedule with existing admitted GLM reasoning on eligible identity ON occurrences; ineligible ON -> exact same-source OFF; maximal 36-update prefix (7266/7441 A0 occurrences) | `cc2c0160` | `ablation-a6-r1` | `pathb-bfcl-a6-r1-cc2c0160d01b` | 36 | `1f050162c0f2803f69b3f844a3822f1987d3e72cdc67064f860a13d23ebb3544` | deleted 2026-10-30 (cleanup) |
| A8a | Nested deterministic subset of A6: keep 179/1791 GLM-traced ON occurrences (rank SHA256('310:A8:'+occurrence)); unselected ON -> same-source A3 OFF; same 7266 occurrences, 36 updates | `4bfe87ad` | `ablation-a8-r1` | `pathb-bfcl-a8a-r1-4bfe87adbb03` | 36 | `5a0a9e78ce813bce06b03ddfba636a85d2c7106351414f4d6285f1d3789030bc` | HF `open-athena/Snowball-67B-A2B-Identity-PathB-A8a-2026.10.30` @ `bf59ad4e` |
| A8b | Nested deterministic subset of A6: keep 537/1791 GLM-traced ON occurrences (rank SHA256('310:A8:'+occurrence)); unselected ON -> same-source A3 OFF; same 7266 occurrences, 36 updates | `4bfe87ad` | `ablation-a8-r1` | `pathb-bfcl-a8b-r1-4bfe87adbb03` | 36 | `83d81612b2421ff71ec6601df39e9a47166eef87f6ef1523d73205f2ab4f7917` | deleted 2026-10-30 (cleanup) |
| A9 | A8a transformation extended to 72 updates over the full frozen r5 B-base10 ledger: old 7266 decisions/179 traces kept; 521/5217 tail GLM-eligible ON selected by SHA256('310:A8:'+occ); 381 traced ON consumed | `8c44a21a (export driver 36627ae6)` | `ablation-a9-r1 (+ write-recovery-a9-a10-r1)` | `train: pathb-bfcl-a9-r1-8c44a21a1aae; export: pathb-bfcl-a9-export-recovery-r1-8c44a21a1aae` | 72 | `97e4c7915c9417e5be8512f7d7fc507e25286cbdd14e6c25fa26a410ed1d5bae` | temporary internal copy only (recovery export, `tmp/ttl=14d`); not kept |
| A10 | TULU-style: 44 sources (4 per each of 11 families) x 10 rounds (9 OFF + 1 hash-chosen GLM ON) = 440 identity occurrences, evenly interleaved into the largest fitting replay prefix (N=1648); 36 updates | `cf4da6f4 (driver 36627ae6)` | `ablation-a10-r1 (+ write-recovery-a9-a10-r1)` | `pathb-bfcl-a10-retrain-r1-cf4da6f40326` | 36 | `8cf696f72c6a0f25666a773bbad1fd6bf622188bdfc5d3221e85dc27a8e73759` | temporary internal copy only (recovery export, `tmp/ttl=14d`); not kept |
| A11 | A3 identity-OFF recipe at 72 updates: maximal 72-batch prefix of the full frozen r5 B-base10 ledger, every identity ON -> same-source OFF | `741e5928` | `ablation-a11-r1` | `pathb-bfcl-a11-r1-741e5928030c` | 72 | `f02679d11ff8506a08046dd2f56181f6443215c02d9773c8543bf5391cccab6b` | HF `open-athena/Snowball-67B-A2B-Identity-PathB-A11-2026.10.30` @ `6834175e` |
| A12 | Exactly A8a's first 144 packed examples (18 updates), fresh from v0.1 | `69ddc88f (export driver 8125c4fa)` | `ablation-a12-a13-r1 (+ A12-export-only-recovery-r1)` | `train: pathb-bfcl-a12-r1-69ddc88fb2ff; export: pathb-bfcl-a12-export-recovery-r1-69ddc88fb2ff` | 18 | `b8a45af4e072b7506d83717f254dc183a3cc6ef5251481de5bb80253ee36adf3` | temporary internal copy only (export recovery, `tmp/ttl=14d`); not kept |
| A13 | A8a schedule with every thinking-ON replay tool occurrence (tool_first_call/tool_call_after_result/tool_reply) duplicated once immediately after itself at weight 1; maximal 36-update prefix (150 extra copies consumed, 98 suffix occurrences dropped) | `69ddc88f` | `ablation-a12-a13-r1 (+ A13-preemption-replacement1-r2)` | `pathb-bfcl-a13-r1-5a3bd99de0a6` | 36 | `f860030d1266c90d146ef07995a95ce7501989727cf226217de82a87408c15ba` | HF `open-athena/Snowball-67B-A2B-Identity-PathB-A13-2026.10.30` @ `55dbe0be` |

Not trained: A5 and A5b were teacher-capture runs that failed their quality gates (0.24% and 0/200 kept). A7 was a capture pilot that failed QC (11/200 against a gate of 50). Their capture records are in `evidence/notes/path-b/ablation-a5-*`, `ablation-a5b-*` and `ablation-a7-*`.

Merges M25/M49/M51/M75: W = (1−t)·A3 + t·A8a with t = 0.25/0.49/0.51/0.75. Weights are accumulated in FP32 and cast once. The 26 router biases stay FP32. The frozen tensors and special-token rows are restored from v0.1. Source pin `462bba93` (`experiments/weight_merging/path_b_sweep`). Packet: `evidence/notes/path-b/merge-sweep-r3/`. Pre-registration: `evidence/notes/path-b/bfcl-noise-r1/M25-M49-M51-M75-merge-sweep-preregistration-marten-r2.json`. M25 and M49 are public on HF (`open-athena/Snowball-67B-A2B-Identity-PathB-M25-2026.10.30` @ `5c8f0b1e`, `…-M49-2026.10.30` @ `310d22d0`). M51 and M75 are temporary internal copies only and are not kept.

**Weights status (checked 2026-11-03).** The owner kept A3, A8a, A11, A13, M25 and M49 on HF; everything else expires. A1, A2, A4, A6 and A8b exports were deleted on 2026-10-30 by an owner-authorized storage cleanup; the deleted keys and the delete response are in `evidence/notes/path-b/cleanup-write-suspension3-r1/` (`exact-dead-HF-delete-manifest.json`, `delete-response.json`). The temporary copies (A0, A9, A10, A12, M51, M75) were present on 2026-11-03 under `tmp/ttl=14d/` and will be removed by the storage lifetime rule.

## 4. Export and audit

The export child writes HF safetensors to `<run_root>/hf/updates-<N>`. The audit compares all **502** tensors against the trained master weights: 104 frozen tensors must be byte-equal to v0.1, and the 398 trained tensors must equal the master cast. A4 is the exception, with 26 frozen and 476 trained. Each arm's result is `path-b-storage-audit.json` (or the `*HF502*` admission file) in its packet folder. Native checkpoints were deleted after a passing audit, by design.

## 5. Evaluations

Every evaluation ran as a sealed packet (`launch-plan.json` + `prepare_request.py`) submitted the same way as training. Priority changed over time. The A0–A11 evaluations (BFCL, native identity, paired maths for A3/A8a/A11) and the original SWE100 runs used `interactive`. The A12, A13 and merge evaluations and the SWE100 top-up used `production`. [INVOCATIONS.md](INVOCATIONS.md) gives the band each submit script asserted. Infra failures were replaced whole; valid scores were never rerun.

| Suite | Runner | Analysis | Results |
|---|---|---|---|
| BFCL (123 tasks, 5 runs per arm, run-level bootstrap) | `experiments.evaluation.held_bfcl` at the BFCL pins above, Harbor config `bfclparity-pi.yaml` | `evidence/scratch/marten-bfcl-*.py`. Each output records its script's SHA256. | `evidence/notes/path-b/bfcl-noise-r1/` |
| Native-boundary identity screen (132 prompts, 52 core; thinking on and off) | `native-boundary-*` packets | role-aware ledgers `*-native-role-aware-ledger-marten-r1.json`. The creator column is the `creator_MC_final` developer/trainer flag. | `evidence/notes/path-b/native-boundary-*/` |
| Paired maths (GSM8K, MATH500) | `*-paired-math-*` packets (`identity_fullscreen_owner/frozen/run_math.py`) | `evidence/scratch/marten-paired-math-analysis-r1.py`, `marten-*-maths-*.py` | `evidence/notes/path-b/*-paired-math-*/` |
| SWE100 (v0.1, A0, A3, A8a) | `second-agentic-swe100-*` packets | `evidence/scratch/marten-SWE100-topup-joint-and99-final-r1.py` | `evidence/notes/path-b/bfcl-noise-r1/SWE100-production-topup-joint-bootstrap-extension-marten-r1.json` |
| Panel-60 (A3, A8a, A13, v0.1) | runner `panel60_runner.py` at `f41b27c5`. The protocol and prompts are the public [gist](https://gist.github.com/penfever/ff6565deebbd68700685807f8e081be1/e35e3eb5bf44a3402b66aaf35e09ff56ad1164b8) at revision `e35e3eb5`. | blind grading, shuffle seed 20261031 | `evidence/notes/t310/panel60/` |

## 6. Results index

[`RESULTS-INDEX.md`](RESULTS-INDEX.md) maps each of the 225 numbers in the Path B sections of #10695 (text of 2026-11-03) to its file under `evidence/` and its JSON field. Identity creator counts use the role-aware `creator_MC_final` flag.

The index shows the raw file values. The issue rounds them, usually to one decimal.
