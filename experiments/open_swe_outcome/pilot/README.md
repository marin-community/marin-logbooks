# Noise-pilot scripts (unreviewed research code)

Session scripts preserved for marin-community/marin-logbooks#39. They carry
hard-coded local paths and are not maintained Marin code.

- `snapshot_names.py`: Daytona snapshot name for each SWE-bench Verified task, through the production naming path.
- `make_subset.py`: writes `../harbor/swebench-verified-200.yaml` and `subset_snapshots.json`.
- `snapshot_cleaner.py`: deletes each pilot task's snapshot after its trial result is written.
- `trial_table.py`: outcome counts from downloaded trial `result.json` files.
- `pilot-131k.yaml`: model config for the pilot checkpoint.
- `rewards-*.json`: per-task reward (null when unscored) for the two 200-task attempts.
