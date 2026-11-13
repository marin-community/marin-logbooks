"""Render the held-out false positives as a small HTML page. usage: fp_report.py <fp.json> <out.html>"""

import html
import json
import sys

docs = {d["number"]: d for d in json.load(open(sys.argv[1]))}
JUNK = {"binary", "encoded_blob", "repetition", "lock_file", "minified_code"}
NAMES = {
    "repetition": "Repetition",
    "structured_data": "Structured data",
    "numeric_table": "Numeric table",
    "encoded_blob": "Encoded blob",
    "minified_code": "Minified code",
    "item_list": "Item list",
}
STATS = {
    "compression_ratio": "compresses to",
    "repeated_line_share": "repeated lines",
    "delimited_line_share": "rows with matching delimiters",
    "digit_share": "digits",
    "mean_line_chars": "mean line length",
}
NOTES = {
    49: {
        "title": "An agent transcript dominated by tool output",
        "kind": "Coding-agent trajectory",
        "why": "The agent hit a recursion error, and the Python traceback prints the same two lines hundreds of times. "
        "It also dumps a syntax tree as JSON. Those chunks really are repetition, and they make up 10 of 23 chunks, "
        "which is over the 30% bar for machine junk.",
        "fix": "The filter has no notion of an agent transcript. These documents open with a fixed line "
        "(“This trajectory failed to solve the task”), so they could be routed around the filter and judged separately.",
        "call": "Arguable. The repetition is real; whether a transcript like this belongs in replay is a separate question.",
        "show": ["repetition", "structured_data", None],
    },
    57: {
        "title": "A spreadsheet of experiment prompts written in plain English",
        "kind": "CSV of psycholinguistics stimuli",
        "why": "Each row is a short scenario and a question. The same scenarios recur across rows with one condition changed, "
        "so chunks compress to about 11% of their size and the rows are long. That matches the rule for templated records.",
        "fix": "Require a low share of letters or words before calling long, compressible lines a record dump. "
        "These chunks are over 90% letters.",
        "call": "Arguable. It is natural language, but heavily duplicated.",
        "show": ["structured_data", None],
    },
    109: {
        "title": "A highway research report with long data appendices",
        "kind": "Technical report (PDF-derived)",
        "why": "The body is prose, but the sampled interior is mostly survey data sheets laid out as pipe tables. "
        "11 of 18 chunks are tables, which is 61% and just over the 60% bar.",
        "fix": "Count markdown-table chunks at less than full weight, or raise the bar for documents whose only label is tables.",
        "call": "Borderline by one chunk. This is the most common kind of false positive on the tuning samples too.",
        "show": ["structured_data", None],
    },
    131: {
        "title": "An appendix of regression tables from a finance paper",
        "kind": "Academic paper supplement",
        "why": "Over 40% of the visible characters are digits in 7 of 13 chunks, with coefficient tables on almost every page.",
        "fix": "None needed in the rules. The filter read it correctly as tables; whether to keep such documents is a policy choice.",
        "call": "Arguable. I labelled it keep because prose connects the tables.",
        "show": ["numeric_table", "structured_data", None],
    },
    139: {
        "title": "A New Testament translation with numbered verses",
        "kind": "Scripture in a low-resource language",
        "why": "Every line is a verse number, a tab, then the verse. A rule added in the last tuning round treats a single "
        "tab per line as a delimited row, so all 8 chunks were labelled structured data.",
        "fix": "Require short fields on both sides of the tab, or at least two tabs per line. "
        "The rule was meant for two-column lists such as a name and an amount.",
        "call": "Clear error, and the only one of the five caused by a specific rule being too loose.",
        "show": ["structured_data"],
    },
}


def tier(label):
    if label is None:
        return "keep"
    return "junk" if label in JUNK else "data"


def strip(doc):
    cells = []
    for window in doc["windows"]:
        for chunk in window["chunks"]:
            label = chunk["label"]
            cells.append(f'<span class="cell {tier(label)}" title="{html.escape(NAMES.get(label, "Kept"))}"></span>')
        cells.append('<span class="gap"></span>')
    return "".join(cells[:-1])


def excerpts(doc, wanted):
    out = []
    for target in wanted:
        for window in doc["windows"]:
            match = next((c for c in window["chunks"] if c["label"] == target), None)
            if match is None:
                continue
            stats = []
            for key, label in STATS.items():
                value = match["features"].get(key)
                if not value:
                    continue
                if key == "mean_line_chars":
                    stats.append(f"{label} {value:,.0f} chars")
                elif key == "compression_ratio":
                    stats.append(f"{label} {value:.0%}")
                else:
                    stats.append(f"{value:.0%} {label}")
            text = match["text"][:560].replace("<|begin_of_text|>", "")
            out.append(
                f'<figure class="excerpt {tier(target)}">'
                f'<figcaption><span class="pill {tier(target)}">{html.escape(NAMES.get(target, "Kept"))}</span>'
                f'<span class="where">token {window["start"]:,}</span>'
                f'<span class="stats">{html.escape(" · ".join(stats[:4]))}</span></figcaption>'
                f"<pre>{html.escape(text)}</pre></figure>"
            )
            break
    return "".join(out)


sections = []
for number in (139, 109, 57, 131, 49):
    doc, note = docs[number], NOTES[number]
    flagged = sum(n for label, n in doc["counts"].items() if label != "None")
    total = sum(doc["counts"].values())
    verdict = doc["category"]
    sections.append(
        f"""
<article>
  <header>
    <p class="kind">{html.escape(note["kind"])}</p>
    <h2>{html.escape(note["title"])}</h2>
    <dl class="meta">
      <div><dt>Verdict</dt><dd><span class="pill {tier(verdict)}">{html.escape(NAMES[verdict])}</span></dd></div>
      <div><dt>Length</dt><dd>{doc["tokens"]:,} tokens</dd></div>
      <div><dt>Chunks labelled</dt><dd>{flagged} of {total}</dd></div>
      <div><dt>Source</dt><dd class="mono">{html.escape(doc["source"])} #{doc["doc_index"]}</dd></div>
    </dl>
    <div class="strip" role="img" aria-label="{flagged} of {total} chunks labelled, in document order">{strip(doc)}</div>
  </header>
  <div class="notes">
    <div><h3>Why it was flagged</h3><p>{html.escape(note["why"])}</p></div>
    <div><h3>What would fix it</h3><p>{html.escape(note["fix"])}</p></div>
    <div><h3>How wrong is it</h3><p>{html.escape(note["call"])}</p></div>
  </div>
  <div class="excerpts">{excerpts(doc, note["show"])}</div>
</article>"""
    )

PAGE = """<title>Junk Filter False Positives</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; each document is a block of verdict, chunk strip, three notes, then excerpts. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --fg: #1b2330; --muted: #5b6676; --line: #dde2ea;
  --keep: #c9d1dc; --data: #c98a1b; --junk: #c2453a; --data-soft: #fbf1dc; --junk-soft: #fbe6e3; --keep-soft: #eef1f5;
  --display: "Space Grotesk", "Inter", system-ui, sans-serif;
  --body: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161d; --surface: #1a2029; --fg: #e6eaf0; --muted: #9aa5b5; --line: #2b3440;
  --keep: #45505f; --data: #e0a843; --junk: #ec7468; --data-soft: #33290f; --junk-soft: #3a1c19; --keep-soft: #222a35;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12161d; --surface: #1a2029; --fg: #e6eaf0; --muted: #9aa5b5; --line: #2b3440;
  --keep: #45505f; --data: #e0a843; --junk: #ec7468; --data-soft: #33290f; --junk-soft: #3a1c19; --keep-soft: #222a35;
  color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 64px; }
main { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }
h1, h2 { font-family: var(--display); text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: 30px; line-height: 1.15; }
h2 { font-size: 20px; line-height: 1.25; }
h3 { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin: 0 0 4px; }
p { margin: 0; }
.lede { color: var(--muted); max-width: 65ch; margin-top: 10px; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.summary div { border-top: 2px solid var(--line); padding-top: 10px; }
.summary b { display: block; font-family: var(--display); font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.summary span { color: var(--muted); font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; color: var(--muted); font-size: 13px; align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
article { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 22px; display: flex; flex-direction: column; gap: 18px; min-width: 0; }
article header { display: flex; flex-direction: column; gap: 12px; }
.kind { color: var(--muted); font-size: 13px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 0; }
.meta dt { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }
.meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.mono { font-family: var(--mono); font-size: 13px; overflow-wrap: anywhere; }
.pill { display: inline-block; font-size: 12px; font-weight: 600; padding: 1px 9px; border-radius: 999px; white-space: nowrap; }
.pill.junk { background: var(--junk-soft); color: var(--junk); }
.pill.data { background: var(--data-soft); color: var(--data); }
.pill.keep { background: var(--keep-soft); color: var(--muted); }
.strip { display: flex; flex-wrap: wrap; gap: 3px; align-items: center; }
.cell { width: 14px; height: 14px; border-radius: 3px; background: var(--keep); display: inline-block; }
.cell.data { background: var(--data); } .cell.junk { background: var(--junk); }
.gap { width: 6px; }
.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px 24px; }
.notes div { min-width: 0; }
.excerpts { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
figure { margin: 0; border-left: 3px solid var(--keep); padding-left: 12px; min-width: 0; }
figure.data { border-left-color: var(--data); } figure.junk { border-left-color: var(--junk); }
figcaption { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.where { font-variant-numeric: tabular-nums; }
pre { font-family: var(--mono); font-size: 12px; line-height: 1.5; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere;
  max-height: 190px; overflow-y: auto; background: var(--bg); border-radius: 6px; padding: 10px 12px; }
pre:focus-visible { outline: 2px solid var(--data); }
footer { color: var(--muted); font-size: 13px; max-width: 65ch; }
</style>
<main>
  <header>
    <h1>Junk Filter False Positives</h1>
    <p class="lede">Five long replay documents the frozen filter flagged that a blind reader labelled as worth keeping. They come from a held-out slice of 11,946 documents the filter was never tuned on.</p>
  </header>
  <section class="summary" aria-label="Held-out check">
    <div><b>5 of 80</b><span>flagged documents in the blind sample labelled keep</span></div>
    <div><b>1 of 40</b><span>among machine-junk verdicts</span></div>
    <div><b>4 of 40</b><span>among data-dump verdicts</span></div>
    <div><b>93%</b><span>of flagged documents are junk or data, weighted to the slice</span></div>
  </section>
  <div class="legend" aria-label="Legend">
    <span>Each square is one 4,096-character chunk, in document order:</span>
    <span><i class="cell"></i> kept</span><span><i class="cell data"></i> data dump</span><span><i class="cell junk"></i> machine junk</span>
  </div>
  __SECTIONS__
  <footer>
    <p>The keep labels are one reader's judgement from four 150-character excerpts per document, made before seeing the verdicts. Only eight 2,048-token windows of each document were sampled, so the chunk strips show a sample, not the whole document. Excerpts are trimmed to 560 characters.</p>
  </footer>
</main>
"""
open(sys.argv[2], "w").write(PAGE.replace("__SECTIONS__", "".join(sections)))
print("wrote", sys.argv[2])
