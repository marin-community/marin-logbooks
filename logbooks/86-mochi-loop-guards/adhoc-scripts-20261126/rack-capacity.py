import json
import subprocess
from collections import Counter,defaultdict

base=['kubectl','--kubeconfig','/Users/benfeuer/.kube/coreweave-iris','--context','marin-us-east-08a_US-EAST-08A']
nodes=json.loads(subprocess.run(base+['get','nodes','-o','json'],capture_output=True,text=True,check=True).stdout)['items']
# Select only scheduling fields; never retrieve pod environment or secret fields.
columns='NODE:.spec.nodeName,PHASE:.status.phase,PRIORITY:.spec.priority,GPU:.spec.containers[*].resources.requests.nvidia\\.com/gpu'
raw=subprocess.run(base+['get','pods','-A','--no-headers','-o','custom-columns='+columns],capture_output=True,text=True,check=True).stdout
nonpreemptible=Counter()
for line in raw.splitlines():
 parts=line.split()
 # -A adds the namespace as the first field.
 if len(parts)==5: parts=parts[1:]
 if len(parts)!=4: continue
 node,phase,priority,values=parts
 if node=='<none>' or phase in ('Succeeded','Failed'): continue
 if priority=='<none>' or int(priority)<1000: continue
 for value in values.split(','):
  if value not in ('<none>',''): nonpreemptible[node]+=float(value)
racks=defaultdict(lambda:{'healthy_gpu_nodes':0,'nodes_clear_of_production_gpu':0,'production_gpu_requests':0})
for n in nodes:
 rack=n['metadata'].get('labels',{}).get('ds.coreweave.com/nvlink.domain')
 if rack is None: continue
 ready=any(c['type']=='Ready' and c['status']=='True' for c in n['status'].get('conditions',[]))
 if not ready or n['spec'].get('unschedulable'): continue
 if int(n['status'].get('allocatable',{}).get('nvidia.com/gpu','0'))<4: continue
 record=racks[rack]
 record['healthy_gpu_nodes']+=1
 request=nonpreemptible[n['metadata']['name']]
 record['production_gpu_requests']+=request
 record['nodes_clear_of_production_gpu']+=int(request==0)
summary={'racks':dict(sorted(racks.items())),'eligible_racks_with_16_nodes_clear_of_production_gpu':sum(r['nodes_clear_of_production_gpu']>=16 for r in racks.values()),'required_rack_slices':2,'required_nodes_per_slice':16,'caveat':'Necessary GPU placement capacity only; CPU, memory, disk, affinity and other taints are not modeled.'}
print(json.dumps(summary,indent=2))
