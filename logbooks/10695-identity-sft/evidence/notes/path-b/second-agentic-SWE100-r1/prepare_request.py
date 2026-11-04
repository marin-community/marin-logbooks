"""Native prepared batches; no live client is opened by this module."""
from pathlib import Path
import hashlib,json,os
from experiments.evaluation import held_bfcl
from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.launch import LaunchSpec,prepare_evaluation_batch
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.harbor.runner import canonical_served_name
from iris.rpc import job_pb2
from rigging.timing import Duration
HERE=Path(__file__).resolve().parent

def verify():
    receipt=HERE/'review-receipt.json'
    if receipt.exists():
        for name,item in json.loads(receipt.read_text())['files'].items():
            raw=Path(name).read_bytes()
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Immutable SWE binding mismatch: '+name)

def prepare(label,run):
    verify();plan=json.loads((HERE/'launch-plan.json').read_text());target=plan['targets'][label]
    item=next(x for x in target['runs'] if x['run']==run);config=Path(plan['suite']['path'])
    os.environ['MARIN_PREFIX']=item['MARIN_PREFIX']
    spec=LaunchSpec(model=load_model_config(Path(target['model_config'])),evals=(),evalchemy_definitions=(),limit=None,
        harbor_definitions=(HarborDefinition(name=canonical_served_name(config.stem),config_path=config),),
        platform=Platform.GPU,accelerator='H100x8',records_prefix=item['records_prefix'],submission_cluster='marin',federated_cluster='cw-rno2a',
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,version=item['version'],description=item['description'])
    return prepare_evaluation_batch(spec)

class CpuLifetimeClient:
    def __init__(self,client):self.client=client
    def submit(self,**kwargs):
        if kwargs['timeout'].to_seconds()!=43200 or kwargs['max_retries_failure'] or kwargs['max_retries_preemption']:
            raise ValueError('Unexpected native coordinator policy')
        kwargs['timeout']=Duration.from_hours(49)
        return self.client.submit(**kwargs)

def submit(batch,client):
    return held_bfcl.submit_batch(batch,CpuLifetimeClient(client))
