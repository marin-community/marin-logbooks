"""Bounded read-only notifier using the SDK state predicate (never job control)."""
import argparse,datetime,json,subprocess,time
from pathlib import Path
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import is_job_finished
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--timeout',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
end=time.monotonic()+a.timeout
while time.monotonic()<end:
    with connect_controller(cluster_name='marin') as connection:
        with IrisClient.remote(connection.url,workspace=Path.cwd(),credentials=connection.credentials) as client:
            s=client.job_status(JobName.from_wire(a.job))
            record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':a.job,'state':s.state.value,'SDK_terminal':is_job_finished(s.state),'read_only':True,'no_automatic_retry':True}
            a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
            if record['SDK_terminal']:
                subprocess.run(['agent-swarm','send','--to','margay','--message',f"TERMINAL {s.state.value}: {a.job}. Inspect actual receipts and attempt resources now; no automatic retry. SDK notifier correct, state {a.out}"],check=True)
                break
    time.sleep(60)
else:
    subprocess.run(['agent-swarm','send','--to','margay','--message',f'Bounded read-only notifier expired for {a.job}; inspect actual state. No retry/cancel.'],check=True)
