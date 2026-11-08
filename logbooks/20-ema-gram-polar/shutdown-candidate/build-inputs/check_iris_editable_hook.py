# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise only Iris's initialize method AST with fake revision lookup."""

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def main(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    tree = ast.parse(raw)
    hook = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BuildInfoHook")
    original = next(n for n in hook.body if isinstance(n, ast.FunctionDef) and n.name == "initialize")
    corrected = copy.deepcopy(original)
    branch = corrected.body[0]
    assert isinstance(branch, ast.If) and isinstance(branch.test, ast.Compare)
    assert ast.unparse(branch.test) == "self.target_name == 'editable'"
    branch.test.left = ast.Name(id="version", ctx=ast.Load())
    rows = []
    for label, method in [("candidate", original), ("proposed_version_guard", corrected)]:
        for target, version in [("wheel", "editable"), ("wheel", "standard"), ("sdist", "standard")]:
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                generated = root / "generated"
                generated.mkdir()
                namespace = {
                    "Path": Path,
                    "mkdtemp": lambda path=generated, **_: str(path),
                    "iris_tree_date": lambda _: "2026-10-29",
                    "_stamped_date": lambda _: "",
                    "SOURCE_PATH": "src/iris/_build_info.py",
                    "WHEEL_PATH": "iris/_build_info.py",
                }
                module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
                exec(compile(module, str(source), "exec"), namespace)
                data = {"force_include": {}}
                namespace["initialize"](SimpleNamespace(target_name=target, root=str(root)), version, data)
                stamp = generated / "_build_info.py"
                expected_stamp = label == "candidate" or version != "editable"
                assert stamp.exists() == expected_stamp
                if expected_stamp:
                    assert 'BUILD_DATE = "2026-10-29"' in stamp.read_text()
                    destination = "src/iris/_build_info.py" if target == "sdist" else "iris/_build_info.py"
                    assert data["force_include"] == {str(stamp): destination}
                else:
                    assert data == {"force_include": {}}
                rows.append(
                    {
                        "source_variant": label,
                        "target": target,
                        "version": version,
                        "stamp_written": stamp.exists(),
                        "pass": True,
                    }
                )
    result = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": rows,
        "scope": (
            "Actual method AST, fake revision/date and temporary-directory allocation; "
            "no package imports, backend invocation, wheel/image build or runtime services."
        ),
        "interpretation": (
            "Candidate writes a stamp for target=wheel/version=editable. "
            "A proposed in-memory version guard avoids that stamp and preserves standard wheel/sdist mapping."
        ),
        "candidate_modified": False,
        "correction_applied": False,
        "runtime_qualified": False,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("PASS: 3 exact-source behavior controls and 3 proposed-guard controls")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
