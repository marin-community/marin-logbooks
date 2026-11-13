# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote sandbox/input checks for the remaining pilot; no model calls."""

import asyncio
import json
import os
import shlex
from pathlib import Path

from rigging.secrets import require_secret
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy


async def main():
    images = json.loads(os.environ["BIOMNI_IMAGES_JSON"])
    root = Path(os.environ["IRIS_OUTPUT_DIR"])
    connection = DaytonaConnection(require_secret("DAYTONA_EVAL_API_KEY"), DEFAULT_DAYTONA_API_URL, "us")
    factory = DaytonaMachineFactory(connection.client_factory(), ttl_minutes=20, create_timeout=300)
    reports = []
    for item in images:
        machine = None
        report = {"task": item["task"], "image": item["image"], "scope": "runtime/input validation; no model calls"}
        try:
            machine = await factory.create(
                MachineSpec(
                    source=RegistryImage(item["image"]),
                    workdir="/app",
                    cpus=2,
                    memory_mb=8192,
                    storage_mb=10240,
                    network=NetworkPolicy.ALLOW,
                    startup_timeout=300,
                )
            )
            checks = "\n".join(x["sha256"] + "  /app/data/" + x["path"] for x in item["inputs"]) + "\n"
            command = (
                (
                    "set -e; pi --version; test ! -e /app/data/mmc5.csv; test ! -e /app/dat"
                    "a/GSE236581_counts.mtx; printf %s "
                )
                + shlex.quote(checks)
                + (
                    " | sha256sum -c -; /opt/biomnibench-judge/bin/python -c 'import openai"
                    "; print(openai.__version__)'; du -sh /app/data; df -h /app"
                )
            )
            if item["task"] == "da-1-3":
                code = (
                    "import shlex; p='/app/data/GSE236581_CRC-ICB_metadata.txt'; f=open(p);"
                    " fields=shlex.split(f.readline()); print(fields); assert {'SubCellType"
                    "','Tissue','Treatment'}.issubset(fields)"
                )
                command += "; python3 -c " + shlex.quote(code)
            if item["task"] == "da-17-3":
                code = (
                    "import h5py; f=h5py.File('/app/data/4118e166-34f5-4c1f-9eed-c64b90a3da"
                    "ce.h5ad','r'); print('root',list(f)); print('obs',list(f['obs'])); pri"
                    "nt('raw_shape',f['raw/X'].attrs['shape']); assert f['X'].id==f['raw/X'"
                    "].id; assert {'author_cell_type'}.issubset(f['obs']); assert len(f['ob"
                    "s'][f['obs'].attrs['_index']])==f['raw/X'].attrs['shape'][0]; assert l"
                    "en(f['var'][f['var'].attrs['_index']])==f['raw/X'].attrs['shape'][1]"
                )
                command += (
                    "; /opt/biomnibench-judge/bin/pip install --quiet h5py==3.15.1; /opt/bi"
                    "omnibench-judge/bin/python -c "
                ) + shlex.quote(code)
            result = await machine.run(Command(argv=("bash", "-lc", command), timeout=300))
            report.update(
                exit_code=result.exit_code,
                stdout=result.stdout.decode(),
                stderr=result.stderr.decode(),
                status="passed" if result.exit_code == 0 else "failed",
            )
        finally:
            if machine is not None:
                await machine.close()
                report["cleanup"] = "closed"
            reports.append(report)
            (root / "remaining-runtime-report.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        if report.get("status") != "passed":
            raise RuntimeError("Runtime/input check failed for " + item["task"])


asyncio.run(main())
