---
issue: https://github.com/marin-community/marin-logbooks/issues/8
archive_template_revision: f54b3aa0e6ee533e988d2b97e9db0d1523f934dd
marin_base: 8577d258214dd0dcb992c04458ee4eb10496d4e5
---

# Hero Uncheatable forecast repair

Question: why does Hero outperform the Uncheatable forecast before the 4K→16K switch, while its original Paloma forecast tracks better?

Baseline: original Uncheatable fixed-floor 1.4 and October 3 shared-floor refits. Target: seven-subset unweighted dropless cross-entropy, nats/token, 4K packing. Exclude all 16K evaluations. Use Paloma as a control, preserving its original forecast.

## 2026-10-25: recovered sources and exploratory checks

Public inputs are reduced to loss histories, analytic compute totals, output-head fractions, run identities, and original forecast points in `inputs.json`. Configurations and histories were read from public W&B GraphQL on October 25, Pacific time. Frozen H100 histories were recovered from the October 3 forecast artifact, with their SHA256 hashes recorded in the input. The old d1536 source covers only 30%; new-mixture d1536 covers all fractions. Original source: [Paloma scaling code at d23e6e9](https://github.com/marin-community/marin/blob/d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py). The analytic output head is `6 * hidden_dim * vocab_size * sequence_length` training FLOPs per example.

Exploratory CPU checks preceded this archive. Excluding output-head FLOPs and refitting the macro floor reduced later 4K Hero error substantially, but several individual subsets became too optimistic. A macro fit can hide opposing subset errors. An early-Hero effective-floor calibration and an equal-parameter offset comparison are required before selecting an update. This is retrospective validation, not a prospective pre-registration.

Plan: profile nonnegative floors with positive-amplitude scaling fits at each training fraction; keep total and body compute conventions separate. Fit ladder-only floors from d512–d1536. Fit effective floors from Hero evaluations through step 179,999; score steps 182,999–254,999. Check d1536 holdouts, derive macro forecasts from subsets, and retain the direct macro model as a diagnostic. A fitted effective floor is not a measurement of infinite-compute entropy. Hero and ladder routing, batch, optimizer, and parameter precision differ.

Reproduction after source publication:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python logbooks/8-hero-uncheatable-forecast/forecast.py
```

No accelerator work or live training change is part of this task. Final 4K predictions are conditional on the original schedule; they do not predict the later 16K or 262K metric.
