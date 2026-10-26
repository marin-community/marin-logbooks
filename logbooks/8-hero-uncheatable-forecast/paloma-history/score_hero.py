# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score historical-only shared floors on the separate Hero 4K trajectory."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paloma", ROOT.parent / "paloma-ladder/fit_original_paloma.py")
paloma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paloma)
data = paloma.read_data()
hero = json.loads((ROOT.parent / "inputs.json").read_text())["hero"]
fits = json.loads((ROOT / "shared-floor-results.json").read_text())["fits"]
result = {"hero_loss_used": False, "models": {}}
for name, fit in fits.items():
    curve = [paloma.fit_at(data, fit["floor"], f) for f in paloma.FRACTIONS]
    result["models"][name] = dict(floor=fit["floor"], final=curve[-1]["loss"], **paloma.score(curve, hero))
(ROOT / "hero-scores.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {n: {k: v for k, v in m.items() if k in ("floor", "final", "all_4k")} for n, m in result["models"].items()},
        indent=2,
    )
)
