import fcntl
from pathlib import Path
import yaml
path=Path('/Users/benfeuer/Documents/experiments/active/job-watchdog/registry.yaml')
with path.with_suffix('.yaml.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 d=yaml.safe_load(path.read_text());d['entries']=[r for r in d['entries'] if r['id'] not in ('mochi-loop-guards-s1-rp-repeat1','mochi-loop-guards-overrides')]
 entries=[]
 for key,job in [('conservation','mochi-loop-guards-conservation-check-20261126b'),('s2-n4-preflight','mochi-loop-guards-s2-fixed-n4-plan-20261126b')]:
  r={'id':'mochi-loop-guards-'+key,'owner':'01a218ad-6f81-78f3-8e73-29b1684d79f6','description':'User override publication and conservation' if key=='overrides' else 'Fixed N4 paired native preflight; zero GPUs','expected_terminal':'SUCCEEDED','iris':{'cluster':'cw-us-east-08a','job':'/benfeuer/'+job},'probe':{'type':'log_regex','cluster':'cw-us-east-08a','job':'/benfeuer/'+job}}
  d['entries']=[x for x in d['entries'] if x['id']!=r['id']];d['entries'].append(r);entries.append(r)
 path.write_text(yaml.safe_dump(d,sort_keys=False))
subset={'coordinator':d['coordinator'],'defaults':d['defaults'],'entries':entries}
Path('/tmp/mochi-loop-guards/watchdog-fresh-subset.yaml').write_text(yaml.safe_dump(subset,sort_keys=False))
