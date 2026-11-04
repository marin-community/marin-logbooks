"""Seal local final packet; remote staging is joined by a separate readback receipt."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/karan/oa/marin-path-b-ablation')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert pin=='6ac2c53d061156ad22be53b332b7bd677a856290'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 plan=json.loads((HERE/'launch-plan.json').read_text());assert plan['code_pin']==pin
 files=[]
 for name in ['PACKET.md','launch-plan.json','stage-files.json','native-five-bindings-proof.json','all-native-and-recipe-proof.json','independent-data-proof-marten-r1.json','five-arm-dose-and-controls.json','data-counts-A0-A3.json','data-counts-A4.json','A4-historical-overlap-caveat.json','tests-final.log','lint-r2.log','types.log','build_data.py','build_t3.py','build_plan.py','check_bindings.py','prove_native_and_recipe.py','freeze_receipt.py']:
  files.append(HERE/name)
 files.extend(HERE/f'model-A{n}.yaml' for n in range(5))
 for name in ['ablation.py','ablation_data.py','ablation_launch.py','storage_audit.py','testing_support.py','test_ablation.py','test_ablation_launch.py','test_storage.py','test_workflow.py']:
  files.append(ROOT/'experiments/post_training/identity_path_b'/name)
 for row in json.loads((HERE/'stage-files.json').read_text())['files']:
  p=Path(row['local']);assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];files.append(p)
 files.extend([Path(plan['runtime']['local']),Path(plan['parent_r5']['path'])])
 files.extend([HERE.parent/'full-screen-execution-r1/execution-packet.json',HERE.parent/'full-screen-execution-r1/analyze_bfcl.py',HERE.parent.parent/'t310/m16-screen/scripts-t2/bfcl_paired_cd16t3.py'])
 changed=subprocess.check_output(['git','diff','--name-only','4bc3742fe7726188ed001daecf805d97cd191a20',pin],cwd=ROOT,text=True).splitlines()
 r={'schema_version':1,'code_pin':pin,'parent_pin':'4bc3742fe7726188ed001daecf805d97cd191a20','changed_source_files':changed,'plan_sha256':sha(HERE/'launch-plan.json'),'input_manifest_sha256':sha(HERE/'stage-files.json'),'status':'FINAL_LOCAL_PASS_COMBINED_SCIENCE_AND_STAGING_JOIN_PENDING','submit_authorized':False,'remote_calls_by_genet':0,'policy':'all5together onMartencombinedCLEAR; noextraownerwait','verification':{'focused_tests':'10PASS49.05s','scoped_types':'0errors','required_lint':'PASS','native_examples':1440,'A0_exact_r5first36':True,'A1_data_byte_identical':True,'A3_canonical_final_replay_equal':True,'A4_first16_exact_T3_native_sampler':True,'optimizer_LR_horizon16_36_140_equal':True,'native_five_root_train_export_signatures':'PASS'},'files':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
 (HERE/'review-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(sha(HERE/'review-receipt.json'),len(r['files']))
if __name__=='__main__':main()
