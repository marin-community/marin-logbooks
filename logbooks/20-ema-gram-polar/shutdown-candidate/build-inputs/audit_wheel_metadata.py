# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check downloaded build inputs as data, without importing build backends."""

import email
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename


def main(base: Path) -> None:
    environment = {
        "implementation_name": "cpython",
        "implementation_version": "3.12.12",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "UNRESOLVED",
        "platform_system": "Linux",
        "platform_version": "UNRESOLVED",
        "python_full_version": "3.12.12",
        "python_version": "3.12",
        "sys_platform": "linux",
        "extra": "",
    }
    text = (base / "editable-build-requirements.txt").read_text()
    pins = {}
    for block in re.split(r"\n(?=[\w-]+==)", text):
        match = re.search(r"^([\w-]+)==([^\s]+)", block, re.M)
        if match:
            pins[canonicalize_name(match[1])] = {
                "version": match[2],
                "hashes": re.findall(r"--hash=sha256:([a-f0-9]{64})", block),
            }
    acquisition = json.loads((base / "wheel-acquisition-r02.json").read_text())
    rows = []
    for artifact in acquisition["wheels"]:
        path = base / "wheels-r01" / artifact["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name, version, _, tags = parse_wheel_filename(path.name)
        name = canonicalize_name(name)
        assert digest == artifact["sha256"] and digest in pins[name]["hashes"]
        assert str(version) == pins[name]["version"]
        assert all(t.interpreter == "py3" and t.abi == "none" for t in tags)
        assert any(t.platform in {"any", "manylinux_2_17_x86_64", "manylinux2014_x86_64"} for t in tags)
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(names) == 1
            metadata = email.message_from_bytes(archive.read(names[0]))
        assert canonicalize_name(metadata["Name"]) == name
        assert metadata["Version"] == str(version)
        assert SpecifierSet(metadata.get("Requires-Python", "")).contains("3.12.12")
        dependencies = []
        excluded = []
        for raw in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            assert not requirement.url
            assert not any(v in str(requirement.marker) for v in ("platform_release", "platform_version"))
            if requirement.marker and not requirement.marker.evaluate(environment):
                excluded.append(raw)
                continue
            selected = pins[canonicalize_name(requirement.name)]["version"]
            assert requirement.specifier.contains(selected)
            dependencies.append(raw)
        rows.append(
            {
                "name": name,
                "version": str(version),
                "filename": path.name,
                "sha256": digest,
                "tags": sorted(str(t) for t in tags),
                "active_dependencies": dependencies,
                "excluded_dependencies": excluded,
                "requires_python": metadata.get("Requires-Python"),
            }
        )
    assert set(pins) == {r["name"] for r in rows}
    assert Requirement("editables~=0.3").specifier.contains(pins["editables"]["version"])
    result = {
        "status": "PASS_METADATA_CLOSURE_ONLY",
        "target_markers": environment,
        "wheels": rows,
        "scope": "Wheel hashes, names, versions, Python requirements, declared dependencies and nominal tags only.",
        "resolver_warning": (
            "uv selected a local 3.12.13 interpreter despite requested 3.12.12. "
            "Source builds were forbidden; this audit independently evaluates wheel metadata at exactly 3.12.12."
        ),
        "limits": [
            "No interpreter or backend executed from these wheels.",
            "manylinux_2_17_x86_64 is a proposed ABI floor; "
            "actual base image/OS and runtime libraries are unresolved.",
            "Workspace build hooks and source imports have not run in a real build.",
            "The 160 runtime wheels, installer, base image and deployed controller remain unresolved.",
        ],
        "packages_installed": False,
        "image_built": False,
        "runtime_qualified": False,
    }
    with (base / "wheel-metadata-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS: {len(rows)} build-input wheel metadata records; runtime unqualified")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
