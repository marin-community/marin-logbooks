"""Package frozen parameters with the exact replay tokenizer and public receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tokenizer-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); source=args.root/'final-v1'; output=args.output
    output.mkdir(exist_ok=False)
    original=json.loads((source/'model.json').read_text())
    tokenizer_files=('tokenizer.json','tokenizer_config.json','chat_template.jinja')
    hashes={name:digest(args.tokenizer_dir/name) for name in tokenizer_files}
    assert hashes['tokenizer.json']=='6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45'
    tokenizer_name='tokenizer-'+hashes['tokenizer.json'][:16]
    (output/tokenizer_name).mkdir()
    for name in tokenizer_files: shutil.copyfile(args.tokenizer_dir/name,output/tokenizer_name/name)
    model={**original,'tokenizer':tokenizer_name,'tokenizer_files_sha256':hashes}
    semantic_keys=('features_version','feature_names','embedding_dimensions','window_tokens','kind','intercept','learning_rate','trees')
    semantic=lambda obj:hashlib.sha256(json.dumps({key:obj[key] for key in semantic_keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert semantic(model)==semantic(original)
    # The tokenizer variants differ only in names of two added IDs; none occurs in sampled views.
    affected=[]
    for path in (args.root/'sample-v1').glob('sample-*.json'):
        row=json.loads(path.read_text())
        if any('<|reserved_special_token_2|>' in text or '<|reserved_special_token_3|>' in text for text in row['windows']): affected.append(row['sample_id'])
    assert not affected
    (output/'model.json').write_text(json.dumps(model,separators=(',',':')))
    model_sha=digest(output/'model.json')
    calibration=json.loads((source/'calibration.json').read_text()); calibration['model_sha256']=model_sha
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2))
    evaluation=json.loads((source/'evaluation-v2-native-rules.json').read_text())
    evaluation['evaluated_model_json_sha256']=evaluation['model_sha256']
    evaluation['model_sha256']=model_sha
    evaluation['calibration_sha256']=digest(output/'calibration.json')
    evaluation['packaging']={'parameters_sha256':semantic(model),'pretest_freeze_commit':'90d61498a6','tokenizer_correction':'Exact replay tokenizer renames added IDs 128005/128011 to tool-call markers; decoder/normalizer/pretokenizer/postprocessor otherwise identical. Neither ID occurs in any sampled visible window. Learned parameters and both cutoffs unchanged.','sampled_views_affected':0,'tokenizer_files_sha256':hashes}
    (output/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    labels=json.loads((source/'label-manifest.json').read_text())
    for row in labels: row['view_sha256']=row.pop('text_sha256')
    (output/'label-manifest.json').write_text(json.dumps(labels,indent=2))
    shutil.copyfile(source/'scores.npz',output/'sample-scores.npz')
    shutil.copyfile(source/'selection.json',output/'selection.json')
    print(json.dumps({'model_sha256':model_sha,'calibration_sha256':evaluation['calibration_sha256'],'evaluation_sha256':digest(output/'evaluation.json'),'label_manifest_sha256':digest(output/'label-manifest.json'),'parameters_sha256':semantic(model)},indent=2))

if __name__=='__main__': main()
