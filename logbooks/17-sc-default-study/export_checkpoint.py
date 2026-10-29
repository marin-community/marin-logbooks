# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the normal request-driven checkpoint converter in the study's RNO region."""

import json
import os
import runpy
import sys

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
fsspec.config.conf["s3"]["client_kwargs"]["region_name"] = "RNO2A"
public_options = {key: value for key, value in fsspec.config.conf["s3"].items() if key not in {"key", "secret", "token"}}
os.environ["FSSPEC_S3"] = json.dumps(public_options)
sys.argv[0] = "cloud.iris.export_hf_checkpoint"
runpy.run_module("cloud.iris.export_hf_checkpoint", run_name="__main__")
