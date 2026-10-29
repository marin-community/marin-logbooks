# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invoke exactf5ef16c8 math driver; only job policy and dataset-cache path change."""

from dataclasses import replace

from job_policy import install

from experiments.weight_merging.rsi.identity_297 import evaluate_math as original

native_run = original.run_evalchemy


def scoped_run(model, config, output, env_vars, *, backend_state):
    config = replace(config, runtime=replace(config.runtime, dataset_cache=output + "/dataset-cache"))
    return native_run(model, config, output, env_vars=env_vars, backend_state=backend_state)


if __name__ == "__main__":
    install()
    original.run_evalchemy = scoped_run
    original.main()
