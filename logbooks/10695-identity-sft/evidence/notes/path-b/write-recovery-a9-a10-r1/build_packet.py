from pathlib import Path
import copy,hashlib,json,subprocess,yaml
HERE=Path(__file__).resolve().parent;B=HERE.parent;W=Path('/home/karan/oa/marin-path-b-write-recovery');NW=Path('/home/karan/oa/marin-path-b-native-boundary-recovered');NH=B/'native-boundary-recovered-a9-a10-r1';NH.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pin=lambda p:subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
binding=json.loads((W/'experiments/post_training/identity_path_b/write_recovery_binding.json').read_text());nparent=json.loads((B/'native-boundary-a9-a10-r1/launch-plan.json').read_text());ntargets=json.loads((NW/'experiments/post_training/identity_path_b/native_boundary_scope/targets.json').read_text())
plan=dict(code_pin=pin(W),workspace=str(W),original_scientific_pin='cf4da6f40326df150eac712e2eb323667f71a267',runtime=binding['A9']['runtime'],recovery_authority='Jerboa current-orders A9/A10 recovery after write suspension: ONE export-only A9 freshHF, ONE freshA10 retrain; only after authorized write probe PASS',new_execution_not_new_science=True,no_additional_science_review_for_unchanged_export_audit=True,storage_hold_at_build=True,original_packets_immutable=True,arms={})
for label,item in binding.items():
 oldplan=json.loads((B/f'ablation-{label.lower()}-r1/launch-plan.json').read_text());oldarm=oldplan['arms'][0];method=copy.deepcopy(oldarm['bfcl']);model=yaml.safe_load(Path(method['model_config']).read_text());oldmodel=copy.deepcopy(model);model['location']=item['event']['hf_export'];model['name']='snowball-pathb-'+label.lower()+'-write-recovery-r1';mp=HERE/f'model-{label}.yaml';mp.write_text(yaml.safe_dump(model,sort_keys=False))
 assert {k:v for k,v in model.items() if k not in ['name','location']}=={k:v for k,v in oldmodel.items() if k not in ['name','location']}
 method['model_config']=str(mp);method['model_config_sha256']=sha(mp);method['MARIN_PREFIX']=item['root']+'/bfcl/native';method['records_prefix']=item['root']+'/bfcl/records'
 for flag,value in [('--model-config',str(mp)),('--version',model['name']),('--records-prefix',method['records_prefix']),('--description',f'Authorized {label} storage recovery unchanged scientific recipe')]:method['argv'][method['argv'].index(flag)+1]=value
 method['paired_reader_command_template']='Historical per-run native score reader only; preregistered five-run bootstrap remains primary.'
 plan['arms'][label]=dict(item,BFCL=method,actual_dose=oldarm['actual_dose'],native_only_identity_packet=str(NH),no_separate_original_parser_run=True)
 n=ntargets[label];n.update(root_name=f'pathb-native132-{label.lower()}-r1-{pin(NW)[:12]}',output=f's3://marin-rno2a/tmp/ttl=14d/snowball-identity/path-b/native-boundary132-{pin(NW)[:12]}/{label}')
plan['native_identity_source_pin']=pin(NW);(HERE/'launch-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
np=copy.deepcopy(nparent);np.update(code_pin=pin(NW),workspace=str(NW),parent_pin='fd227ed536067409eb2532f050027e100ac99bd4',authority='Owner recovery into freshHF and newA10root; exact parent native method, no additional science read; actual own readiness plus storage gate required',targets=ntargets,recovery_packet=str(HERE),recovery_plan_sha256=sha(HERE/'launch-plan.json'))
(NH/'launch-plan.json').write_text(json.dumps(np,indent=2)+'\n')
for name in ['prepare_request.py','check_readiness.py','check_bundle.py']:(NH/name).write_bytes((B/'native-boundary-a9-a10-r1'/name).read_bytes())
orders=(B.parent/'t310/current-orders.md').read_text();line=next(x for x in orders.splitlines() if x.startswith('- **A9/A10 recovery after'))
(HERE/'authority-snapshot.md').write_text(line+'\n\nLocal snapshot only. Actual submission follows current applicable owner orders and authorized write gate. This packet grants no probe, delete, quota, reroute or repeated recovery authority.\n')
print(json.dumps(dict(recovery_plan=sha(HERE/'launch-plan.json'),native_plan=sha(NH/'launch-plan.json'),source=pin(W),native_source=pin(NW))))
