# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

"""F5d: set 30, the embedding audit's judged sample. Per stratum: the share of records both judges call COPY or
EDITED, and the catalog estimate of records with a copy that stage 1 did not send (records x weight x share),
with the audit's measured recall (F5b) as a lower-bound correction."""
import io
import json

import numpy as np
import pandas as pd
from rigging.filesystem.storage_path import StoragePath

E = "s3://marin-us-east-02a/tmp/ttl=7d/rav/stage1-explore/embed-2/set30"
GOOD, BAD = {"COPY", "EDITED"}, {"VARIANT", "GENERIC", "UNRELATED"}


def labels(judge: str) -> pd.DataFrame:
    rows = [
        json.loads(l)
        for p in StoragePath(f"{E}/{judge}/batch_*.jsonl").glob()
        for l in p.read_text().split("\n")
        if l.strip()
    ]
    return pd.DataFrame(rows).set_index("uid")[["label", "note"]].add_prefix(judge.split("-")[0] + "_")


pairs = pd.read_parquet("work/set30_pairs.parquet").set_index("uid")
frame = pairs.join(labels("glm-5.3"), how="inner").join(labels("deepseek-v4.1-flash"), how="inner")
frame["copy"] = frame.glm_label.isin(GOOD) & frame.deepseek_label.isin(GOOD)
frame["clean"] = frame.glm_label.isin(BAD) & frame.deepseek_label.isin(BAD)
population = pd.read_json(io.StringIO(StoragePath(f"{E}/population.json").read_text()))
rng = np.random.default_rng(0)
rows = []
for stratum, part in frame.groupby("stratum"):
    labeled = part[part["copy"] | part["clean"]]
    share = labeled["copy"].mean() if len(labeled) else float("nan")
    boot = (
        [labeled["copy"].to_numpy()[rng.integers(0, len(labeled), len(labeled))].mean() for _ in range(2000)]
        if len(labeled)
        else [np.nan]
    )
    weighted = population.at[stratum, "weighted"]
    rows.append(
        {
            "stratum": stratum,
            "judged": len(part),
            "labeled": len(labeled),
            "copy_share": share,
            "share_5_95": f"{np.percentile(boot, 5):.3f}-{np.percentile(boot, 95):.3f}",
            "records_in_sample": int(population.at[stratum, "records"]),
            "catalog_records_with_copy": weighted * share,
        }
    )
table = pd.DataFrame(rows).set_index("stratum")
print(table.round(3).to_string())
print("labels:", (frame.glm_label + "/" + frame.deepseek_label).value_counts().head(8).to_dict())
copies = frame[frame["copy"]]
print("judged copies by eval:", copies["eval"].value_counts().head(10).to_dict())
for r in copies.sample(min(6, len(copies)), random_state=2).itertuples():
    print(
        f"-- {r.stratum} {r.eval} sim {r.sim:.2f} score {r.score:.2f} | {r.glm_label}/{r.deepseek_label}: {r.glm_note}"
    )
frame.reset_index().to_parquet("work/set30_judged.parquet")
