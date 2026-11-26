from pathlib import Path
from datetime import datetime
import shutil
base=Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards')
status=base/'STATUS.md';old=status.read_text();(base/'STATUS-history-before-fresh-20261126.md').write_text(old)
entry='''## User overrides applied by fresh worker, November 26

Owner `01a218ad-6f81-78f3-8e73-29b1684d79f6`; coordinator `b610073d-d871-4fd6-a41f-c087d2723ec1`.
These binding user overrides supersede conflicting entries below; historical receipts and measurements are retained.

1. **User override — deadline +24 h.** T0 stays November 25 12:01:36 EST. T20 is November 27 08:01:36 EST; T32 is November 27 20:01:36 EST; the overall deadline is November 28 12:01:36 EST. I3 is restored; the original T20 budget-drop receipt remains historical. Receipt: `decisions/clock-extension-user-override.json`.
2. **User override — correctness thresholds ×0.75.** Unrounded thresholds are TB2 30.4875 (integer minimum31), TBLite42, held-out78.15 (integer79), pooled166.905 (display166.9; integer167). Repeat windows use these unrounded values: TB2 counts29–32, TBLite40–44, held-out77–80. Validity and loop-incidence gates are unchanged. Receipt: `decisions/correctness-gate-user-override.json` preserves original and recomputed verdicts side by side.
3. **User override — checkpoint conservation.** S1 produces no trained weights: guards act on SFT400407. Permanently conserve the regional initializer, excluding it from every pruning sweep. HF revision `open-athena/Marin-Mochi-535B-A23B@sft-fullab-v2-step-400407`, commit `cfc315a0dcd12d8be5eb0023c0693c435af98998`, contains206 files totaling1,070,987,037,568 bytes. It is PUBLIC. Setting private failed with HF403: open-athena private-storage limit reached. Private conservation is BLOCKED on org storage capacity; no claim of a private copy. Regional file/size/hash verification is in progress. Any S2 trained weights must be exported through an08a CPU job to PRIVATE open-athena repos, with OpenMDW1.1 LICENSE, required research-artifact and responsible-use text, generation config, and permanent regional retention. Receipt: `decisions/checkpoint-conservation-user-override.json`.

| Arm | Original verdict | Verdict under user override | Correctness | Repeat |
|---|---|---|---|---|
| S1-stop | FAIL: TBLite loop reduction and held-out correctness | FAIL: TBLite loop reduction42.48%<50% |59/69/101; pooled229, all pass overridden thresholds | No trigger |
| S1-rp | FAIL; final pending old correctness-triggered repeat | FAIL: action UB46/16/24 exceeds unchanged caps17/10/6.5 |61/70/103; pooled234, all pass overridden thresholds | No trigger; canceled |

The RP repeat exact root `/benfeuer/users-benfeuer-checkpoints-mochi-loop-guards-s1-rp-7b66e20b68-in-d88ffd61-2026.11.26.2-coordinator-c20eb2fd` was canceled with `iris-cancel-own` at11:28 EST after adoption. The original held-out103 is23? No: it is24.85 above78.15, outside±2; no correctness or UB trigger fires. Canceled repeat data do not replace or pool with originals. The listed retired watcher PIDs35210/35250/35895/35898/53297 were already absent on adoption.

S2 plan: fix N4 identically before submission, primary measure repeat-turn share, based on the existing conservative7h/update budget plus3h startup reserve. Two60-GB200 arms (48policy+12inference each), production08a/all racks/TP1BF16/PIECEWISE graphs, same frozen pool/seed/LR/initializer; antithrash is the only treatment difference. Preserve each completed update; no new update after T32. Submission pending completed override receipts and final N4 native plans. Existing N2 native preflight passed; do not redo completed environment preparation.

'''.replace('is23? No: it is','is')
status.write_text('# Mochi loop guards (#12264)\n\n'+entry+old.split('\n',1)[1])
design=base/'DESIGN.md';d=design.read_text();d=d.replace('Status: DESIGN ONLY. Nothing launched. Written 2026-11-25 for a Codex executor.','Status: active experiment. Original design written 2026-11-25; binding user overrides below supersede conflicting original clocks, correctness gates, and checkpoint retention.')
design.write_text(d.split('\n',1)[0]+'\n\n'+entry+d.split('\n',1)[1])
archive=Path('/Users/benfeuer/Documents/marin-worktrees/mochi-loop-guards-logbook/logbooks/86-mochi-loop-guards')
with (archive/'notebook.md').open('a') as f:f.write('\n\n'+entry)
a=archive/'analysis'/'user-overrides-20261126';a.mkdir(exist_ok=True)
for p in Path('/tmp/mochi-loop-guards/user-overrides-20261126').glob('*.json'):shutil.copy2(p,a/p.name)
shutil.copy2('/tmp/mochi-loop-guards/apply-user-overrides.py',a/'recompute.py')
shutil.copy2('/tmp/mochi-loop-guards/hf-conservation-inventory.json',a/'hf-inventory.json')
