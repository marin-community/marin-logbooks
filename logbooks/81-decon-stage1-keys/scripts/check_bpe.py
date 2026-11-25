# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa  (archived analysis script, kept as run)

from experiments.datakit.decontam.encoder.encoding_survey import DECODINGS, _bpe_markers

print(sorted(DECODINGS))
print(repr(_bpe_markers("WhichĠplanetĊ")))
