# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

import inspect

from experiments.datakit.decontam.encoder import key_set_survey as ks

print(inspect.signature(ks.key_sets), inspect.signature(ks.scan_shard))
