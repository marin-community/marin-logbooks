# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish historical-only Paloma floor alternatives beside the Uncheatable repair."""

import importlib.util
import json
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr

ROOT = Path(__file__).resolve().parent
METRIC = "eval_dropless/paloma/macro_loss"
spec = importlib.util.spec_from_file_location("paloma", ROOT / "paloma-ladder/fit_original_paloma.py")
paloma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paloma)


def main():
    history = json.loads((ROOT / "paloma-history/shared-floor-results.json").read_text())
    floor = history["fits"]["budget_means27_0.5"]["floor"]
    original = json.loads((ROOT / "paloma-ladder/results.json").read_text())
    inputs = json.loads((ROOT / "inputs.json").read_text())
    data = paloma.read_data()
    pooled = [paloma.fit_at(data, floor, f) for f in paloma.FRACTIONS]
    models = {
        METRIC: pooled,
        "comparison/paloma/ladder_fitted": original["models"]["fit_30_80"]["curve"],
        "comparison/paloma/fixed_1_5": original["models"]["fixed_1.5"]["curve"],
    }
    rows = {}
    for key, curve in models.items():
        for row in curve:
            rows.setdefault(round(row["progress"] * paloma.STOP), {})[key] = row["loss"]
    for row in inputs["hero"]:
        rows.setdefault(row["_step"], {})["comparison/paloma/observed_4k"] = row[METRIC]
    run_id = "hero-paloma-shared-floor-20261025"
    with wandb.init(
        entity="marin-community",
        project="marin_moe",
        id=run_id,
        name="Paloma forecast: shared historical floor (4K)",
        job_type="forecast",
        config={
            "forecast_only": True,
            "hero_loss_used": False,
            "floor": floor,
            "context_length": 4096,
            "method": "shared floor; separate family/progress log-OLS coefficients; equal-family raw-loss MSE",
            "TPU_points": 44,
            "Delphi_runs": 27,
            "Delphi_budget_means": 9,
            "tokenizer_cache_parity": "related tokenizers and matching suite; bytes not audited",
        },
    ) as run:
        for step, values in sorted(rows.items()):
            run.log(values, step=step)
        artifact = wandb.Artifact("hero-paloma-shared-floor-20261025", type="forecast-data")
        for directory in ("paloma-ladder", "paloma-history"):
            for path in (ROOT / directory).iterdir():
                if path.is_file():
                    artifact.add_file(str(path), name=directory + "/" + path.name)
        artifact.add_file(str(ROOT / "inputs.json"), name="inputs.json")
        run.log_artifact(artifact)
        run_url = run.url
    publication = json.loads((ROOT / "publication.json").read_text())
    report = wr.Report.from_url(publication["report_url"])
    report.blocks = [
        *report.blocks,
        wr.H2("Paloma: fit the historical floor"),
        wr.P(
            "The original TPU ladder alone prefers E=1.2914, with Hero 4K RMSE 0.02393. "
            "A common floor fitted to that ladder plus 27 Delphi endpoints gives E=1.4242 "
            "and Hero RMSE 0.00859, versus 0.00866 for the original fixed-1.5 fit. "
            "No Hero loss chooses either floor. The pooled conditional 4K endpoint is 2.02637."
        ),
        wr.P(
            "TPU has separate coefficients at each measured 30-80% fraction; Delphi has its "
            "own endpoint curve over nine budget means. Each family receives half the MSE "
            "weight. Changing Delphi weight from 25% to 75% moves E from 1.3557 to 1.4998. "
            "This is model sensitivity, not a confidence interval. No architecture-specific "
            "floor offset is needed for this fit. Related Llama 3 tokenizers and matching "
            "16-subset definitions support comparability, but their cache bytes were not audited. "
            "The 8K 67B run is analyzed separately in the notebook."
        ),
        wr.PanelGrid(
            runsets=[wr.Runset(entity="marin-community", project="marin_moe", filters=f"ID == {json.dumps(run_id)}")],
            panels=[
                wr.LinePlot(
                    title="Paloma 4K: historical shared floor, ladder-only floor, original prior",
                    x="Step",
                    y=[
                        METRIC,
                        "comparison/paloma/ladder_fitted",
                        "comparison/paloma/fixed_1_5",
                        "comparison/paloma/observed_4k",
                    ],
                    range_x=(117000, 390251),
                    smoothing_factor=0,
                    aggregate=False,
                    layout=wr.Layout(x=0, y=0, w=24, h=10),
                )
            ],
        ),
    ]
    report.save()
    publication["paloma_run_id"] = run_id
    publication["paloma_run_url"] = run_url
    (ROOT / "publication.json").write_text(json.dumps(publication, indent=2))
    print(json.dumps({"paloma_run_url": run_url, "report_url": report.url}))


if __name__ == "__main__":
    main()
