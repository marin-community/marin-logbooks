# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the accessible curvature guide from measured outputs; no new inference."""

import base64
import html
import re
import shutil
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent
shutil.copytree(ROOT.parent / "gradient-curvature-analysis/figures", ROOT / "figures", dirs_exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)
for source in (ROOT.parent / "gradient-curvature-analysis").iterdir():
    if source.suffix in (".csv", ".json"):
        shutil.copy2(source, ROOT / "data" / source.name)
shutil.copy2(ROOT / "methods.pdf", ROOT / "methods-reference.pdf")

SOURCE = ROOT / "reading-guide-draft.md"
FORMAT = "markdown-implicit_figures+tex_math_single_backslash"
text = SOURCE.read_text()
reference = ROOT / "methods-draft.md"


def to_html(md):
    result = pypandoc.convert_text(md, "html5", format=FORMAT, extra_args=["--mathml"])
    result = re.sub(
        r"<p>(<img [\s\S]*?/>)</p>\s*<p><em>(Figure [\s\S]*?)</em></p>",
        r"<figure>\1<figcaption>\2</figcaption></figure>",
        result,
    )
    for match in re.findall(r'src="([^\"]+\.png)"', result):
        path = ROOT / Path(match).with_suffix(".svg")
        if not path.exists():
            path = ROOT.parent / Path(match).with_suffix(".svg")
        encoded = base64.b64encode(path.read_bytes()).decode()
        result = result.replace('src="' + match + '"', 'src="data:image/svg+xml;base64,' + encoded + '"')
    result = re.sub(
        r"<table([\s\S]*?)</table>",
        (
            '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable data'
            ' table"><table\\1</table></div>'
        ),
        result,
    )
    result = re.sub(
        r"<figure>(<img [\s\S]*?/>)",
        (
            '<figure><button class="zoom" type="button" aria-label="Enlarge this figure">\\1<s'
            "pan>Enlarge figure</span></button>"
        ),
        result,
    )
    return result


chunks = re.split(r"(?m)^## ", text)
intro = to_html(chunks[0])
sections, nav = [], []
for i, chunk in enumerate(chunks[1:]):
    title = chunk.split("\n", 1)[0]
    nav.append(f'<a href="#chapter-{i}">{html.escape(title)}</a>')
    body = to_html("## " + chunk)
    body = re.sub(r'<h2 id="[^"]+">', f'<h2 id="heading-{i}" tabindex="-1">', body, count=1)
    sections.append(f'<section class="chapter" id="chapter-{i}" aria-labelledby="heading-{i}">{body}</section>')
refchunks = re.split(r"(?m)^## ", reference.read_text())
refparts = ['<div class="reference-intro">' + to_html(refchunks[0]) + "</div>"]
for chunk in refchunks[1:]:
    title, body = chunk.split("\n", 1)
    content = to_html(body)
    content = re.sub(r'id="([^"]+)"', r'id="ref-\1"', content)
    refparts.append(f"<details><summary>{html.escape(title)}</summary>{content}</details>")
css = (ROOT / "reading-style.css").read_text()
js = (ROOT / "reading-controls.js").read_text()
page = (
    (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"'
        ' content="width=device-width, initial-scale=1"><title>Gradient and curvature experiments — visu'
        "al reading guide</title><style>"
    )
    + css
    + '</style></head><body><a class="skip" href="#main">Skip to report</a>'
)
page += (
    '<div class="controls" role="region" aria-label="Reading controls"><span>Reading '
    'controls</span><button id="smaller" aria-label="Smaller text">A&#8722;</button><'
    'button id="larger" aria-label="Larger text">A+</button><button id="spacing" aria'
    '-pressed="false">Extra spacing</button><button id="focus" aria-pressed="false">O'
    'ne section at a time</button><span id="settings" role="status" aria-live="polite'
    '"></span></div><main id="main">'
)
page += (
    intro
    + (
        '<p class="reading-note">Read at your own pace. Enlarge any figure. The full tech'
        'nical reference is below the guide.</p><details class="contents"><summary>Choose'
        ' a section</summary><nav aria-label="Report sections">'
    )
    + "".join(nav)
    + '<a href="#technical-reference">Full technical reference</a></nav></details><div id="guide">'
    + "".join(sections)
    + "</div>"
)
page += (
    '<nav id="step-nav" aria-label="Section navigation" hidden><button id="previous">'
    '← Previous section</button><span id="progress" role="status"></span><button id="'
    'next">Next section →</button></nav><section id="technical-reference"><h2 tabinde'
    'x="-1">Full technical reference</h2><p>The detailed experimental methods are included'
    " below. Open only the sections you need. The guide and methods explain different levels of detail."
    '</p><p><a href="methods-reference.pdf">Open the methods PDF</a> · '
    '<a href="data/path-measurements.csv">Measured paths (CSV)</a> · '
    '<a href="data/analysis.json">Analysis and source hashes</a></p>'
)
page += (
    "".join(refparts)
    + (
        '</section></main><dialog id="figure-dialog"><button id="close-figure">Close figu'
        "re</button><p>On a small screen, scroll sideways to inspect the enlarged figure."
        '</p><div class="large-figure"><img id="large-image" alt=""></div></dialog><scrip'
        "t>"
    )
    + js
    + "</script></body></html>"
)
(ROOT / "reading-guide.html").write_text(page)
