# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the separately audited methods reference with large text."""

import shutil
import subprocess
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent


def main():
    tex = ROOT / "methods.tex"
    pypandoc.convert_text(
        (ROOT / "methods-draft.md").read_text().replace("polar₅", r"\(\operatorname{polar}_5\)"),
        "latex",
        format="markdown-implicit_figures+tex_math_single_backslash",
        outputfile=str(tex),
        extra_args=[
            "--standalone",
            "--variable=documentclass:extarticle",
            "--variable=fontsize:14pt",
            "--variable=papersize:letter",
            "--variable=geometry:margin=0.8in",
            "--variable=mainfont:Arial",
            "--variable=sansfont:Arial",
            "--variable=colorlinks:true",
            "--include-in-header=" + str(ROOT / "report-layout.tex"),
        ],
    )
    rendered = tex.read_text()
    rendered = rendered.replace(
        "Which quantities distinguish direction from\ncurvature?",
        r"Which quantities distinguish" + "\n" + r"direction from curvature?",
    )
    rendered = rendered.replace(
        r"F(x-sd)-F(x)=-sa+s^2\int_0^1(1-u)\,d^\top H(x-usd)d\,du.",
        r"\begin{aligned}F(x-sd)-F(x)&=-sa\\" r"&\quad+s^2\int_0^1(1-u)\,d^\top H(x-usd)d\,du.\end{aligned}",
    )
    tex.write_text(rendered)
    subprocess.run(
        [shutil.which("tectonic") or str(Path.home() / ".local/bin/tectonic"), "--keep-logs", str(tex)],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
