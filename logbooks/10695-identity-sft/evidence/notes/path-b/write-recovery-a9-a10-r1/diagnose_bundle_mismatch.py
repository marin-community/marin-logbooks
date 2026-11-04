from pathlib import Path
import datetime,hashlib,io,json,struct,subprocess,zipfile
from iris.cluster.client.bundle import collect_workspace_files,create_workspace_zip
import iris.cluster.client.bundle as implementation
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/karan/oa/marin-path-b-write-recovery');expected='4b1afda4a23953bf58975656a276e6901f9030dc51d2d9e51f4581ca25b4862c';sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(HERE/'executor-presubmit-actual-bundle-margay-r1.zip').read_bytes();actual=sha(raw);assert actual=='73493ee6ebd279face1a754e5f3ef12d9237876ed690515c9fc7d2f8bed14d4e'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
files=collect_workspace_files(ROOT);members={};target='experiments/post_training/identity_path_b/write_recovery.py'
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert z.namelist()==[p.relative_to(ROOT).as_posix() for p in files]
 for p in files:
  name=p.relative_to(ROOT).as_posix();data=z.read(name);assert data==p.read_bytes(),name
  members[name]=dict(bytes=len(data),sha256=sha(data),date_time=list(z.getinfo(name).date_time))
 info=z.getinfo(target);offset=info.header_offset;assert raw[offset:offset+4]==b'PK\x03\x04';assert raw[offset+30:offset+30+len(target)]==target.encode()
 # Locate this exact central-directory member, not compressed payload text.
 central=None;pos=z.start_dir
 for entry in z.infolist():
  assert raw[pos:pos+4]==b'PK\x01\x02';n,x,c=struct.unpack_from('<HHH',raw,pos+28);name=raw[pos+46:pos+46+n].decode()
  if name==target:central=pos
  pos+=46+n+x+c
 assert central is not None
assert create_workspace_zip(ROOT)==raw
candidates=[];base=datetime.datetime(2026,10,30,12,0,0)
for elapsed in range(0,17*60,2):
 dt=base+datetime.timedelta(seconds=elapsed);dos=(dt.hour<<11)|(dt.minute<<5)|(dt.second//2);changed=bytearray(raw);struct.pack_into('<H',changed,offset+10,dos);struct.pack_into('<H',changed,central+12,dos)
 if sha(changed)==expected:candidates.append(dict(original_local_timestamp=dt.isoformat(),old_DOS_time=dos,reconstructed_author_sha256=expected))
assert len(candidates)==1,candidates
out=dict(result='EXACT_SINGLE_MEMBER_TIMESTAMP_CAUSE_PROVEN',source_pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implementation_path=implementation.__file__,implementation_sha256=sha(Path(implementation.__file__).read_bytes()),executor_bundle_sha256=actual,expected_author_bundle_sha256=expected,bytes=len(raw),members=len(members),all_member_bytes_equal_current_clean_committed_workspace=True,actual_native_rebuild_exactly_equals_executor_archive=True,only_changed_metadata_member=target,executor_target_ZIP_datetime=members[target]['date_time'],author_timestamp_reconstructed=candidates[0],scope='Changing ONLY the DOS modification-time fields in this one local+central member header reconstructs the exact sealed author SHA; no content, mode, ordering, compression, source or SDK change.',model_GPU_RNO_job_calls=0)
(HERE/'bundle-mismatch-timestamp-diagnosis-genet-r1.json').write_text(json.dumps(out,indent=2)+'\n');(HERE/'executor-all-members-SHA256-genet-r1.json').write_text(json.dumps(members,indent=2)+'\n');print(json.dumps(out))
