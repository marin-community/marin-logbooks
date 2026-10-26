# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the frozen Uncheatable method comparison as a separate W&B forecast."""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr

ROOT = Path(__file__).resolve().parent
ENTITY = "marin-community"
PROJECT = "marin_moe"
UNCHEATABLE = "eval_dropless/uncheatable_eval/macro_loss"
PALOMA = "eval_dropless/paloma/macro_loss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = json.loads((ROOT / "method-results.json").read_text())
    data = json.loads((ROOT / "inputs.json").read_text())
    run_id = "hero-uncheatable-paloma-method-20261025"
    rows = {}
    legacy = data["legacy"]["hero-uncheatable-floor14-new-from30-75c36f"]
    for name, field in (("body_fit", UNCHEATABLE), ("body_fixed", "comparison/uncheatable/fixed_floor")):
        for row in result["metrics"][UNCHEATABLE]["models"][name]["curve"]:
            if row["mix"] == "new":
                step = round(row["progress"] * 390251)
                rows.setdefault(step, {})[field] = row["loss"]
                rows[step]["comparison/uncheatable/published"] = float(
                    np.interp(step, [r["_step"] for r in legacy], [r[UNCHEATABLE] for r in legacy])
                )
    for metric, field in ((UNCHEATABLE, "comparison/uncheatable/observed"), (PALOMA, "comparison/paloma/observed")):
        r = result["metrics"][metric]
        for step, value, published in zip(
            r["steps"], r["observed"], r["models"]["published"]["prediction"], strict=True
        ):
            rows.setdefault(step, {})[field] = value
            if metric == PALOMA:
                rows[step]["comparison/paloma/published"] = published
    run_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}"
    if not args.report_only:
        with wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=run_id,
            name="Uncheatable forecast: Paloma method + fitted floor (4K)",
            job_type="forecast",
            config={
                "forecast_only": True,
                "hero_loss_used": False,
                "eval_context_length": 4096,
                "uncheatable_subsets": 7,
                "floor": result["metrics"][UNCHEATABLE]["models"]["body_fit"]["floor"],
                "method": (
                    "per-percentile log-OLS; output-head-excluded compute; shared ladder-fitted floor"
                    " profiled in raw-loss SSE"
                ),
                "source": "https://github.com/marin-community/marin-logbooks/tree/codex/8-hero-uncheatable-forecast",
                "scoring_start": 119999,
                "scoring_end": 254999,
                "stop_after_steps": 390251,
            },
        ) as run:
            for step, values in sorted(rows.items()):
                run.log(values, step=step)
            run.summary["comparison/uncheatable/published_rmse"] = result["metrics"][UNCHEATABLE]["models"]["published"][
                "all_4k"
            ]["rmse"]
            run.summary["comparison/uncheatable/repaired_rmse"] = result["metrics"][UNCHEATABLE]["models"]["body_fit"][
                "all_4k"
            ]["rmse"]
            artifact = wandb.Artifact("hero-uncheatable-paloma-method-20261025", type="forecast-data")
            for name in (
                "inputs.json",
                "compare_methods.py",
                "forecast.py",
                "method-results.json",
                "method-comparison.png",
                "notebook.md",
            ):
                artifact.add_file(str(ROOT / name), name=name)
            run.log_artifact(artifact)
            run_url = run.url
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Hero Uncheatable forecast: original vs Paloma method",
        description="Pre-switch 4K method comparison; no Hero losses used to fit the forecasts.",
        width="fluid",
    )
    runset = wr.Runset(entity=ENTITY, project=PROJECT, filters=f"ID == {json.dumps(run_id)}")
    report.blocks = [
        wr.P(
            "On 46 Hero evaluations at steps 119,999-254,999, the published Uncheatable forecast "
            "has 0.03111 RMSE. Using Paloma's log-OLS method and excluding output-head FLOPs, "
            "with the existing 1.4 floor, worsens "
            "this to 0.06191. Fitting the floor from the H100 ladder alone gives 1.20343 and "
            "reduces RMSE to 0.00647. On its own corpus, the original Paloma forecast scores 0.00856 on the same "
            "window."
        ),
        wr.P(
            "Paloma's original method uses a fixed 1.5 floor and per-percentile log-OLS on "
            "output-head-excluded compute. Here the first transfer keeps Uncheatable's existing "
            "1.4 corpus-specific floor; the repair profiles a shared nonnegative floor across the"
            " original/new mixtures and 30-100% ladder progress, minimizing raw-loss SSE after "
            "log-OLS. H100 d512-d1536 coverage is held constant between candidates. The original "
            "Paloma line used the separate TPU ladder. Hero losses are used only for "
            "retrospective scoring."
        ),
        wr.P(
            "The revised new-mixture forecast endpoint is 1.49676 nats/token at the original 390,251-step budget, "
            "conditional on 4K evaluation and that schedule. It is not a prediction of the later "
            "16K or 262K metric. The fitted floor is an effective extrapolation parameter; recipe"
            " differences and functional-form error remain. The seven-subset macro is fitted "
            "directly. No confidence interval is claimed."
        ),
        wr.P([wr.Link("Reproduction and evidence", "https://github.com/marin-community/marin-logbooks/issues/8")]),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.LinePlot(
                    title="Uncheatable: original, method transfer, fitted-floor repair",
                    x="Step",
                    y=[
                        "comparison/uncheatable/observed",
                        "comparison/uncheatable/published",
                        "comparison/uncheatable/fixed_floor",
                        UNCHEATABLE,
                    ],
                    range_x=(117000, 390251),
                    smoothing_factor=0,
                    aggregate=False,
                    layout=wr.Layout(x=0, y=0, w=24, h=10),
                ),
                wr.LinePlot(
                    title="Paloma: original forecast and measured Hero (4K only)",
                    x="Step",
                    y=["comparison/paloma/observed", "comparison/paloma/published"],
                    range_x=(117000, 255000),
                    smoothing_factor=0,
                    aggregate=False,
                    layout=wr.Layout(x=0, y=10, w=24, h=9),
                ),
            ],
        ),
    ]
    report.save()
    (ROOT / "publication.json").write_text(
        json.dumps({"run_url": run_url, "run_id": run_id, "report_url": report.url, "report_id": report.id}, indent=2)
    )
    print(json.dumps({"run_url": run_url, "report_url": report.url}))


if __name__ == "__main__":
    main()
