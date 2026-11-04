"""Exercise the pinned botocore request builder, stopping before any network I/O."""
import sys,hashlib,json
from pathlib import Path
from urllib.parse import urlsplit
PINNED=Path('/mnt/storage/karan-scratch/genet-path-b/merge-boto1415');sys.path.insert(0,str(PINNED))
import boto3,botocore
from botocore.config import Config
from experiments.weight_merging.path_b_sweep.worker import storage_client
P=Path(__file__).resolve().parent
class PreparedOnly(Exception):pass

def prepared(client):
 capture={}
 def stop(request,**kwargs):
  parsed=urlsplit(request.url)
  capture.update(hostname=parsed.hostname,path=parsed.path,content_length=str(request.headers.get('Content-Length')),content_encoding=str(request.headers.get('Content-Encoding','')),checksum_algorithm=str(request.headers.get('x-amz-sdk-checksum-algorithm','')))
  raise PreparedOnly
 client.meta.events.register('before-send.s3.PutObject',stop)
 try:client.put_object(Bucket='marin-rno2a',Key='tmp/ttl=14d/snowball-identity/path-b/offline-transport-fixture/probe',Body=b'x'*32,IfNoneMatch='*')
 except PreparedOnly:pass
 else:raise AssertionError('Must stop before transport')
 return capture

def main():
 assert boto3.__version__==botocore.__version__=='1.41.5'
 fixed=storage_client('https://cwobject.com','fixture-only-key','fixture-only-secret')
 good=prepared(fixed);assert good['hostname']=='marin-rno2a.cwobject.com'and good['path'].startswith('/tmp/')and good['content_length']=='32';assert 'aws-chunked'not in good['content_encoding']and good['checksum_algorithm']=='';assert fixed.meta.region_name=='auto'
 broken=boto3.client('s3',endpoint_url='https://cwobject.com',region_name='auto',aws_access_key_id='fixture-only-key',aws_secret_access_key='fixture-only-secret',config=Config(connect_timeout=30,read_timeout=300,retries={'max_attempts':3}))
 old=prepared(broken);assert old['hostname']=='cwobject.com'and old['path'].startswith('/marin-rno2a/')
 report={'boto3':boto3.__version__,'botocore':botocore.__version__,'package_source_paths':{'boto3':boto3.__file__,'botocore':botocore.__file__},'actual_worker_client_prepared_PutObject':good,'old_policy_prepared_PutObject_negative':old,'actual_worker_request_checksum_calculation':fixed.meta.config.request_checksum_calculation,'actual_worker_response_checksum_validation':fixed.meta.config.response_checksum_validation,'real_botocore_before_send_boundary':True,'network_storage_model_calls':0,'worker_probe_runtime_PASS':'NOT_YET_OBSERVED'}
 (P/'pinned-botocore-prepared-request-proof.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
