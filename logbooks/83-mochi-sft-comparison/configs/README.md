# Config provenance

Byte-identical copies of the sources that defined each arm. Each file's source
commit, repository, and path is listed below; see `notebook.md` for the full
reading.

## Full-SFT arm (marin-community/marin, commit `dca3b9c0eb486986f49d8407dfc09efe1590f486`, on `main`)

Copies of `experiments/grug/moe_hero_ep/` files and the training preset; all
three are unchanged from this commit through Marin `main` at
`cc3706eb2cdb5e1fec695bc4bfb6582177af82ec`.

| File | Source path | Notes |
|---|---|---|
| `mochi_sft.py` | `experiments/post_training/mochi_sft.py` | `MOCHI_SFT` recipe for `hero-sft-fullab-step393k-v2` |
| `launch_hero_sft.py` | `experiments/grug/moe_hero_ep/launch_hero_sft.py` | `load_sft_stores`, `plan_sft_mixture`, replay wiring |
| `hero_sft_stores_2026_09_29.json` | `experiments/grug/moe_hero_ep/hero_sft_stores_2026_09_29.json` | 216 SFT stores; CoreWeave copy of `gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json` |

`mochi_sft.py` also reads `hero_sft_expert_placement_step393491.json` from the
same directory at that commit; it is an expert-permutation table, not part of
the data mix, so it is linked rather than copied.

## V5 arm (marin-community/marin, commit `a3d4a7c2b0`, branch `benfeuer/long-loop-prototype`)

| File | Source path | Notes |
|---|---|---|
| `mochi-sft-exp3-v5-bankadam-128.yaml` | `experiments/post_training/long_loop/rounds/mochi-sft-exp3-v5-bankadam-128.yaml` | Round config; branch-only, not on Marin `main` |

## V5 data-release and run artifacts (private object storage, hash-pinned)

Copied from `s3://marin-us-east-08a/marin/users/benfeuer/...` because the
record cites their contents. SHA-256 values match the pins in the round config
and release manifest.

| File | Source URI | SHA-256 |
|---|---|---|
| `v5-release-manifest.json` | `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/release-v5/manifest.json` | `c4c2035e177dff1090b4510f1aa228a676eb753951a10581c7ba4bd853b63b1f` |
| `v5-epoch-capacity.json` | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-sft-preparation/bankadam128-capacity-20261124b/long-loop/sft-epoch-capacity/2026.11.24.2/capacity.json` | `e335ed4eefabfcf14043ef47778fe07b9985483ca0facd4f4ab367e6b39d93d8` |
| `v5-train-weights.json` | `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/release-v5/train_weights.json` | `e365fa1d5a12b774c1e41b104b76d37c8da982a5b765d5c753408f12caa22223` |
| `v5-mixture-rationale.json` | `s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1/release-v5/mixture-rationale.json` | `ec94ee03642acc1056e14a289f8df3e7d30a717b7a444eef61a4abbbb8e840b0` |
| `v5-actual-exposure.json` | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/actual-exposure.json` | not externally pinned (run output) |
| `v5-selected.json` | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/selected.json` | not externally pinned (run output) |
| `v5-inputs.json` | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/inputs.json` | not externally pinned (run receipt; linked, not copied — 488 KB) |

`v5-release-manifest.json` also pins `data.json` (component list, SHA-256
`12046a74b9fc95ec84268e5738dc53d489f959c9a8f3c8584da29080d4c23f19`) and the
data seal `release-complete-v5.json` (SHA-256 `3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5`,
pinned by the round config as `data_seal`); both remain at their source URIs.

No training data is copied here — only manifests, configs, and run metadata.
