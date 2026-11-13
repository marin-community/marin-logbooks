"""Download the replay window sample shards from CoreWeave S3 to a local directory.

usage: fetch_sample.py <s3 prefix> <local dir>
"""

import sys
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

configure_coreweave_s3()
base = StoragePath.parse(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
total = 0
for entry in sorted(base.ls(), key=str):
    data = entry.read_bytes()
    (out / entry.name).write_bytes(data)
    total += len(data)
    print(entry.name, len(data))
print("total bytes", total)
