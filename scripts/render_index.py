# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the GitHub Markdown index from the research task records."""

import re
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[1] / "index"
RECORD_FILENAME = re.compile(r"([0-9]+)-[a-z0-9]+(?:-[a-z0-9]+)*\.md")
FIELDS = (
    "Task issue",
    "Status",
    "Research branch",
    "Notebook at commit",
    "Marin contributions",
)
HEADER = """# Research tasks

The table is generated from the [research task records](index/). Follow
[the index procedure](WORKFLOW.md#maintain-the-index) to add or update a record.
"""


def table_cell(value: str) -> str:
    """Escape pipes that would otherwise split a GitHub Markdown table cell."""
    return re.sub(
        r"(\\*)\|",
        lambda match: match[0] if len(match[1]) % 2 else "\\" + match[0],
        value,
    )


def record_cells(path: Path) -> list[str]:
    """Read a task title and the five fields from an index record."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != len(FIELDS) + 1 or not lines[0].startswith("# "):
        raise ValueError(
            f"{path.name}: expected a question heading and five single-line fields"
        )

    values = []
    for field, line in zip(FIELDS, lines[1:]):
        value = line.removeprefix(f"- {field} — ").strip()
        if not value or value == line:
            raise ValueError(f"{path.name}: expected a nonempty {field} field")
        values.append(value)

    issue, status, branch, notebook, contributions = values
    return [
        issue,
        lines[0][2:],
        status,
        branch,
        notebook,
        contributions,
        f"[Record](index/{path.name})",
    ]


def render_index() -> str:
    """Generate the full index with higher-numbered task records first."""
    records = []
    for path in RECORDS.glob("*.md"):
        match = RECORD_FILENAME.fullmatch(path.name)
        if match is None or path.is_symlink():
            raise ValueError(
                f"{path.name}: expected a regular <issue-number>-<topic>.md file"
            )
        records.append((int(match[1]), path))

    rows = [
        ("Task issue", "Question", *FIELDS[1:], "Record"),
        ("---",) * (len(FIELDS) + 2),
    ]
    rows.extend(record_cells(path) for _, path in sorted(records, reverse=True))
    table = "\n".join("| " + " | ".join(map(table_cell, row)) + " |" for row in rows)
    return f"{HEADER}\n{table}\n"


if __name__ == "__main__":
    try:
        print(render_index(), end="")
    except ValueError as error:
        raise SystemExit(str(error)) from error
