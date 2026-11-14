# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the checkpoint report with embedded figures and portable evidence links."""

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import pypandoc


def export_candidates(root, output):
    """Flatten completed candidate measurements; retain raw records for paired uncertainty."""
    records = []
    for archive in ("geometry-results", "geometry-muon-results"):
        for path in sorted((root / archive).rglob("completed.json")):
            complete = json.loads(path.read_text())
            experiment = complete["experiment"]
            if not complete["success"] or experiment not in ("power", "muon", "history"):
                continue
            source = path.parent / f"{experiment}-results.json"
            result = json.loads(source.read_text())
            identity = json.loads((path.parent / "checkpoint-identity.json").read_text())
            protocol = json.loads((path.parent / "protocol.json").read_text())
            for row in result["rows"]:
                for rule, evaluation in row["evaluation"].items():
                    records.append(
                        dict(
                            experiment=experiment,
                            state=result["state"],
                            run_id=path.parent.name,
                            source_commit=complete["source_commit"],
                            checkpoint_uri=identity["uri"],
                            checkpoint_manifest_sha256=identity["object_manifest_sha256"],
                            data_manifest_sha256=result["data_manifest_sha256"],
                            protocol_sha256=hashlib.sha256((path.parent / "protocol.json").read_bytes()).hexdigest(),
                            length_grid=json.dumps(protocol["length_grid"]),
                            draw=row["draw"],
                            variant=row.get("variant", ""),
                            alpha=row.get("alpha", ""),
                            history_condition=row.get("condition", ""),
                            rule=rule,
                            length=evaluation["length"],
                            actual_norm=evaluation["actual_norm"],
                            alignment=row["alignment"],
                            signed_curvature=row["curvature"],
                            predicted_loss_change=evaluation["predicted_change"],
                            observed_loss_change=evaluation["loss_change"],
                            native_update_norm=row.get("native_norm", row.get("reference_update_norm")),
                            native_update_seconds=row.get("native_seconds", ""),
                            native_timing_scope=(
                                "Unavailable in this run"
                                if "native_seconds" not in row
                                else {
                                    "power": "Synchronized update and norm; may include compilation",
                                    "history": "Update/state/norm; may include pending gradient and compilation",
                                    "muon": "Update, norm and variant diagnostics; may include compilation",
                                }[experiment]
                            ),
                            gradient_seconds=row.get("gradient_seconds", ""),
                            run_completed_successfully=True,
                            raw_result="data/raw/" + source.relative_to(root).as_posix(),
                            raw_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        )
                    )
    if records:
        with (output / "candidate-results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    previous = root.parent / "gradient-curvature-report"
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_count = export_candidates(root.parent, args.output)
    evidence = {}
    if candidate_count:
        evidence["candidate-results.csv"] = dict(
            source="Derived from completed raw candidate results by geometry-report/build_reading_guide.py",
            sha256=hashlib.sha256((args.output / "candidate-results.csv").read_bytes()).hexdigest(),
            rows=candidate_count,
        )
    figure_links = []
    for folder in sorted(root.glob("*figures")):
        if not folder.is_dir():
            continue
        group = "diagnostics" if folder.name == "figures" else folder.name.removesuffix("-figures")
        for path in sorted(folder.iterdir()):
            if path.suffix not in (".svg", ".png", ".pdf", ".json"):
                continue
            relative = f"figures/{group}/{path.name}"
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            evidence[relative] = dict(
                source=str(path.relative_to(root.parent)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            if path.suffix == ".pdf":
                figure_links.append(f'<li><a href="{relative}">{html.escape(group + ": " + path.stem)}</a></li>')
    raw_sources = list(root.parent.glob("geometry-*.json"))
    raw_sources.append(root.parent / "hessian-aligned-configs.json")
    for directory in ("geometry-results", "geometry-muon-results"):
        raw_sources.extend(p for p in (root.parent / directory).rglob("*") if p.suffix in (".json", ".jsonl"))
    for path in raw_sources:
        relative = "data/raw/" + path.relative_to(root.parent).as_posix()
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        evidence[relative] = dict(
            source=str(path.relative_to(root.parent)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def to_html(markdown):
        body = pypandoc.convert_text(markdown, "html5", format="markdown-implicit_figures")
        for source in re.findall(r'src="([^"]+)"', body):
            path = root / source
            vector = path.with_suffix(".svg")
            if vector.exists():
                path = vector
            mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode()
            body = body.replace(f'src="{source}"', f'src="data:{mime};base64,{encoded}"')
        body = re.sub(
            r"<p>(<img [\s\S]*?/>)</p>",
            '<figure><button class="zoom" type="button" aria-label="Enlarge this figure">'
            r"\1<span>Enlarge figure</span></button></figure>",
            body,
        )
        for target in re.findall(r'href="([^"]+)"', body):
            if target.startswith(("https://", "http://", "#")):
                continue
            path = (root / target).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = "data/" + digest[:12] + "-" + path.name
            destination = args.output / relative
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(path, destination)
            evidence[relative] = dict(source=str(path.relative_to(root.parent)), sha256=digest)
            body = body.replace(f'href="{target}"', f'href="{relative}"')
        return re.sub(
            r"(<table[\s\S]*?</table>)",
            r'<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable data table">\1</div>',
            body,
        )

    source = root / "reading-guide-draft.md"
    chunks = re.split(r"(?m)^## ", source.read_text())
    intro = to_html(chunks[0])
    sections, links = [], []
    for index, chunk in enumerate(chunks[1:]):
        title = chunk.split("\n", 1)[0]
        links.append(f'<a href="#chapter-{index}">{html.escape(title)}</a>')
        body = to_html("## " + chunk)
        body = re.sub(r'<h2 id="[^"]+">', f'<h2 id="heading-{index}" tabindex="-1">', body, count=1)
        sections.append(
            f'<section class="chapter" id="chapter-{index}" aria-labelledby="heading-{index}">{body}</section>'
        )
    css = (previous / "reading-style.css").read_text()
    css += (
        "@media print{.chapter{break-before:auto;margin-top:1.7em;padding-top:1em;border-top:1px solid #cad7df}"
        "h1,h2,h3{break-inside:avoid}}"
    )
    js = (previous / "reading-controls.js").read_text()
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Optimizer geometry: checkpoint experiment report</title><style>" + css + "</style></head><body>"
        '<a class="skip" href="#main">Skip to report</a>'
        '<div class="controls" role="region" aria-label="Reading controls"><span>Reading controls</span>'
        '<button id="smaller" aria-label="Smaller text">A&#8722;</button>'
        '<button id="larger" aria-label="Larger text">A+</button>'
        '<button id="spacing" aria-pressed="false">Extra spacing</button>'
        '<button id="focus" aria-pressed="false">One section at a time</button>'
        '<span id="settings" role="status" aria-live="polite"></span></div><main id="main">'
        + intro
        + '<p class="reading-note">Read at your own pace. Enlarge any figure. '
        "Evidence links open the measured records included with this report.</p>"
        '<details class="contents"><summary>Choose a section</summary><nav aria-label="Report sections">'
        + "".join(links)
        + '</nav></details><div id="guide">'
        + "".join(sections)
        + "</div>"
        '<nav id="step-nav" aria-label="Section navigation" hidden>'
        '<button id="previous">← Previous section</button><span id="progress" role="status"></span>'
        '<button id="next">Next section →</button></nav>'
        '<p class="reading-note"><a href="evidence-index.json">Evidence index with SHA-256 hashes</a>. '
        '<a href="candidate-results.csv">Candidate measurements as CSV</a>. '
        "Each CSV row is one completed candidate at one length rule. Negative loss change is better. "
        "Blank timing cells mean unavailable; recorded wall times can include compilation and telemetry. "
        "Raw JSON retains paired group losses and numerical checks. "
        "The raw archive includes completed, failed and partial attempts, each with its original status. "
        "Large response arrays remain at the recorded cloud locations.</p>"
        '<details class="contents"><summary>Separate figure exports</summary><ul>'
        + "".join(figure_links)
        + "</ul><p>Editable SVG and PNG versions are included beside each figure PDF.</p></details></main>"
        '<dialog id="figure-dialog"><button id="close-figure">Close figure</button>'
        "<p>On a small screen, scroll sideways to inspect the enlarged figure.</p>"
        '<div class="large-figure"><img id="large-image" alt=""></div></dialog><script>' + js + "</script></body></html>"
    )
    (args.output / "reading-guide.html").write_text(page)
    (args.output / "evidence-index.json").write_text(json.dumps(evidence, indent=2) + "\n")
    shutil.copy2(source, args.output / "reading-guide-source.md")
    print(
        json.dumps(
            dict(
                sections=len(sections),
                linked_records=len(evidence),
                candidate_rows=candidate_count,
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
