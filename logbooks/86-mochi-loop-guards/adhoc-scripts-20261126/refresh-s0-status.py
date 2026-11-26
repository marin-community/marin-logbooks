import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards')
state = json.loads((root/'scratch/20261126_0050_s0_v12_monitoring_state.json').read_text())
assert 'State: running' in state['latest_signal'] and 'running=32' in state['latest_signal'], 'Handle changed job state explicitly'
local = ZoneInfo('America/New_York')
observed = datetime.fromisoformat(state['latest_observation_at']).astimezone(local)
next_probe = datetime.fromisoformat(state['next_probe_at']).astimezone(local)
lines = Path('/tmp/mochi-loop-guards/s0-v12-logs-latest.log').read_text().splitlines()
progress = [re.search(r'Evaluation Progress: (\d+)/(\d+).*?\[(\d+)s\]', line) for line in lines if 'Evaluation Progress:' in line][-1]
assert progress is not None
returned,total,elapsed = map(int,progress.groups())
cohort = {267:'TB2',200:'TBLite',100:'SWE100',256:'held-out'}[total]
remaining = total-returned
headline = (
    f"Current ({observed:%H:%M:%S} EST, November 26): S0 v12 is RUNNING on all32 GPU hosts, "
    f"zero failures/preemptions at this scheduled probe. TB2/TBLite/SWE100 cohort audits passed. "
    f"Latest cached {cohort} progress: {returned}/{total} returned, native elapsed{elapsed} seconds; "
    f"{remaining} trial(s) remain. A rounded100% display does not mean completion. "
    "Actual Pi feedback check passed5/5multi-turn records. Full S0 remains unsealed. "
    "Source and science configuration unchanged. Placed00:52:48 EST, GPU child aa4cc7ddd162; accepted00:50:45 EST."
)
p=root/'STATUS.md';s=p.read_text();a=s.index('Current (');b=s.index('\nRoot:',a);s=s[:a]+headline+s[b:];s=re.sub(r'next probe \d\d:\d\d:\d\d EST',f'next probe {next_probe:%H:%M:%S} EST',s,count=1);p.write_text(s)
print({'observed_at':observed.isoformat(),'cohort':cohort,'returned':returned,'total':total,'elapsed':elapsed,'next_probe_at':next_probe.isoformat()})
