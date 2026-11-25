"""Compare saved binary grades by task ID with Pi and the pinned authors' run."""

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import stdev

PILOT = {"001": 0, "003": 0, "012": 0, "013": 1, "015": 1, "028": 1}


def paired(grades, reference, ids):
    differences = [grades[i] - reference[i] for i in ids]
    n = len(ids)
    return {
        "n": n,
        "claude_code_solved": sum(grades[i] for i in ids),
        "reference_solved": sum(reference[i] for i in ids),
        "claude_code_only": differences.count(1),
        "reference_only": differences.count(-1),
        "agreement": differences.count(0),
        "difference_pp": 100 * sum(differences) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    for arg in ("selection", "grades", "pi", "authors"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    grades = json.loads(args.grades.read_text())
    pi = {k: v["reward"] for k, v in json.loads(args.pi.read_text()).items()}
    authors = json.loads(args.authors.read_text())
    for label, ids in (("checkpoint", selection["checkpoint"]), ("expanded", selection["checkpoint"] + selection["remaining"])):
        present = [i for i in ids if i in grades]
        if len(present) != len(ids):
            print(f"{label}: {len(present)}/{len(ids)} graded")
            continue
        for name, ref in (("Pi", pi), ("authors", authors)):
            comparison = paired(grades, ref, ids)
            if label == "checkpoint":
                # Simple random sampling without replacement from the other 112 tasks.
                differences = [grades[i] - ref[i] for i in ids]
                se = 100 * stdev(differences) * sqrt((1 - len(ids) / 112) / len(ids))
                comparison["checkpoint_design_se_pp"] = se
            print(label, name, json.dumps(comparison, sort_keys=True))
    if len(grades) == 112:
        all_grades = {**grades, **PILOT}
        assert set(all_grades) == set(pi) and len(all_grades) == 118
        for name, ref in (("Pi", pi), ("authors", authors)):
            print("all_eligible", name, json.dumps(paired(all_grades, ref, sorted(all_grades)), sort_keys=True))


if __name__ == "__main__":
    main()
