from pathlib import Path
import datetime,hashlib,json,os,subprocess
from iris.cluster.client.bundle import create_workspace_zip
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/karan/oa/marin-path-b-write-recovery');sha=lambda raw:hashlib.sha256(raw).hexdigest()
d=json.loads((HERE/'bundle-mismatch-timestamp-diagnosis-genet-r1.json').read_bytes());assert d['result']=='EXACT_SINGLE_MEMBER_TIMESTAMP_CAUSE_PROVEN'
p=ROOT/d['only_changed_metadata_member'];old_stat=p.stat();before_sha=sha(p.read_bytes());timestamp=datetime.datetime.fromisoformat(d['author_timestamp_reconstructed']['original_local_timestamp']).timestamp();os.utime(p,ns=(old_stat.st_atime_ns,int(timestamp*1_000_000_000)))
assert sha(p.read_bytes())==before_sha and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
raw=create_workspace_zip(ROOT);expected=d['expected_author_bundle_sha256'];assert sha(raw)==expected
archive=HERE/'author-sealed-workspace-restored-native.zip'
if archive.exists():assert archive.read_bytes()==raw
else:archive.write_bytes(raw)
assert create_workspace_zip(ROOT)==raw
receipt=HERE/'review-receipt.json';assert sha(receipt.read_bytes())=='f319902be954e42c02599ae811d2b2b73d6a82da058443969ec618089c31f091'
out=dict(result='NATIVE_SDK_REBUILD_EXACT_SEALED_HASH_RESTORED',code_pin=d['source_pin'],restored_member=d['only_changed_metadata_member'],restored_local_timestamp=d['author_timestamp_reconstructed']['original_local_timestamp'],source_bytes_sha256_before_and_after=before_sha,native_bundle_sha256=expected,native_bundle_bytes=len(raw),canonical_archive=str(archive),two_native_rebuilds_byte_identical=True,original_receipt_sha256_unchanged=sha(receipt.read_bytes()),guard_not_bypassed=True,no_source_or_science_changes=True,model_GPU_RNO_job_calls=0)
(HERE/'bundle-timestamp-restoration-native-rebuild-proof-genet-r1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
