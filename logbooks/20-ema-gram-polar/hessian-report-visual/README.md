# Hessian experiments — visual reading guide

Start with **visual-report.html** for adjustable text (18–28 px), extra line and paragraph spacing, an optional one-section-at-a-time view, and enlarged figures. It works without a network connection. The full technical reference appears in expandable sections below the guide.

**visual-report.pdf** is the 17-page reading guide. It contains nine figures, eight newly created, short explanations, first-use definitions, and explicit limitations. The original 39-page report is preserved as **technical-reference.pdf** and under its original filename. No results or acceptance thresholds changed in this presentation revision.

The Markdown and LaTeX files are editable. Figures are supplied as SVG, PDF, and PNG. The portable delivery retains the complete prior evidence package. This archive keeps compact plotting inputs; the adjacent experiment directories retain the scientific records.

To regenerate the new figures, use Python 3.12 with NumPy and Matplotlib and run `python plot_report.py`. To regenerate the guide, install pypandoc_binary, Tectonic, and Arial, then run `python build_visual_report.py` from this folder. The build uses the included data and preserves the scientific calculations. `build_visual_report.py` and `visual-layout.tex` build the new guide.

`visual-validation.json` records exact final artifact hashes and checks. The three files named `visual-*-review-final.md` contain the independent final reading, terminology, and scientific reviews. The portable delivery includes its own full checksum manifest.
