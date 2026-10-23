---
title: Hero metadata reconstruction trials
marin_source_revision: 258080eaebfa80b68b0d4abd90bf0ca1ab660d0e
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
task: https://github.com/marin-community/marin/pull/10527
---

# Hero metadata reconstruction trials

## 2026-10-23: Preserve the investigation before extracting the exporter

The question was whether a text-free index of Hero's selected document pool
could let users download provider data and reconstruct selected text without
rerunning global deduplication, classification, or clustering. The contribution
is [Marin #10527](https://github.com/marin-community/marin/pull/10527), which also
serves as this archive's existing task record. The archive preserves the entire
Marin source tree at the revision above, including the exploratory scripts and
their tests. Inherited automatic workflows have been removed on this branch.

The investigation used frozen store `store_4d2e363d`. Selection checks passed
for all 292 sources and exported 17,507 occurrences. Staged normalization
checks replayed 1,167 real input rows (1,166 nonempty). These were separate
samples; staged inputs could already include provider-specific transforms.
They do not establish a provider-to-selected-document round trip.

Subsequent provider probes matched four production normalized IDs each for
FineTranslations, Stack v3, SWE ConTree, SWE zero, and Institutional Books.
The Institutional Books probe required a 96 MiB projected-read cap. Biodiversity
matched four candidate books from a 2,048-page prefix, but equal normalized
text does not establish complete contributing-page lineage. FineTranslations
Viewer offsets 0 through 3 in `abk_Cyrl/train` reproduced production IDs at
revision `af3f4ca895450216d4771cdbf3e3b95c5bacaa2a`; the original and Viewer
single-file digests matched. Final selected membership of those four rows was
not checked.

Provider-route discovery found 284 Hugging Face routes and eight other routes.
Complete ordered split manifests, row-count reconciliation, all-source locator
backfill, and historical transform provenance remain unfinished. Stored Harrier
model and quantization provenance was available for 291 sources. Common Crawl
Focus had an unresolved repack provenance gap. Selected vector coverage was
not measured.

The scripts remain at `experiments/datakit/scripts/metadata_*.py`. They cover
artifact inventory, recipe inspection, bounded selected-row export, staged
normalization replay, and raw provider transform replay. Their CLI arguments
are available through `uv run python -m <module> --help`. Supplied manifests and
storage access are required for real-data replay; credentials and private
trial artifacts are deliberately absent here. The local fixture tests can be
rerun with:

```bash
uv run pytest experiments/datakit/store/tests/test_metadata_reconstruction.py experiments/datakit/store/tests/test_metadata_upstream_locators.py experiments/datakit/scripts/test_metadata_artifact_inventory.py experiments/datakit/scripts/tests/test_metadata_transform_locator_trial.py
```

Fixture checks exercise the harness and do not prove upstream reconstruction.
The selected implementation keeps metadata export and coordinate validation
in Marin. This archive retains the investigation for later locator backfill
work; it authorizes no additional data jobs or dataset publication.
