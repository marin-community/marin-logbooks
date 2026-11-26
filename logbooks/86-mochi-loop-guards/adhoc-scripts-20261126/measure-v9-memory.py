from pathlib import Path
import json,subprocess,sys
state=json.loads(Path('/Users/benfeuer/Documents/experiments/active/mochi-loop-guards/scratch/20261125_1835_s0_v9_monitoring_state.json').read_text())
start,end,label=sys.argv[1:]
sql=f'''SELECT task_id, attempt_id, count(*) AS samples, max(memory_peak_mb) AS peak_mib, max(memory_mb) AS active_window_max_mib, max(ts) AS last_sample FROM "iris.task" WHERE task_id LIKE '{state['gpu_job_id']}/%' AND attempt_id=0 AND "cluster"='cw-us-east-08a' AND ts >= TIMESTAMP '{start}' AND ts < TIMESTAMP '{end}' GROUP BY task_id, attempt_id ORDER BY active_window_max_mib DESC'''
p=Path('/tmp/mochi-loop-guards/s0-v9-memory-'+label)
p.with_suffix('.sql').write_text(sql+'\n')
r=subprocess.run(['uv','run','--no-sync','finelog','query','marin'],input=sql,text=True,capture_output=True)
if r.returncode:raise RuntimeError('Finelog query failed')
p.with_suffix('.jsonl').write_text(r.stdout)
rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
head=next(row for row in rows if row['task_id'].endswith('/0'))
print(json.dumps({'hosts':len(rows),'head_active_max_gib':round(head['active_window_max_mib']/1024,2),'head_attempt_peak_gib':round(head['peak_mib']/1024,2),'other_active_max_gib':round(max(row['active_window_max_mib'] for row in rows if not row['task_id'].endswith('/0'))/1024,2),'samples_min':min(row['samples'] for row in rows),'samples_max':max(row['samples'] for row in rows),'through':max(row['last_sample'] for row in rows)}))
