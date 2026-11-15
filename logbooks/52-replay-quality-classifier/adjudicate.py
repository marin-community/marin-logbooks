"""Resolve anonymized annotation disagreements with a fresh Astra context."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

from label import PROMPT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root
    panel=json.loads((root/'label-panel-private.json').read_text())
    output=root/'adjudication-v1'; output.mkdir(exist_ok=True)
    ids=panel['adjudicate']
    schema=root/'astra-v1/schema.json'
    def batch(index):
        batch_ids=ids[index*8:(index+1)*8]
        directory=output/f'batch-{index:02d}'; directory.mkdir(exist_ok=True)
        started=directory/'started.json'
        if started.exists(): return {'batch':index,'status':'preserved_prior_attempt'}
        documents=[]
        for sample_id in batch_ids:
            row=json.loads((root/'sample-v1'/f'sample-{sample_id:04d}.json').read_text())
            judgments=[labels[str(sample_id)] for labels in (panel['glm'],panel['astra']) if str(sample_id) in labels]
            if sample_id%2: judgments.reverse()
            documents.append({'sample_id':sample_id,'tokens':row['tokens'],'clipped':row['review_clipped'],'intervals':row['window_token_intervals'],'judgments':judgments,'text':row['text']})
        prompt='Adjudicate each document independently from the visible evidence, using the rubric below. Other judgments are anonymous suggestions, not authorities. Neither model scores nor source identities are supplied. A short QA appendix does not automatically invalidate coherent primary content; judge the dominant content. Universal BOS/EOS markers never establish synthetic origin. Preserve useful code, manuals, catalog descriptions, scientific abstract collections and multilingual prose unless the visible content is predominantly a listed low-value category. Abstain when the excerpts cannot decide. Do not use tools or inspect other files. Return exactly one JSON label per sample_id in a labels array.\n\n'+PROMPT+'\n\n'+json.dumps(documents,ensure_ascii=False)
        (directory/'input.txt').write_text(prompt)
        started.write_text(json.dumps({'ids':batch_ids,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'model':'gpt-6-astra'}))
        command=['codex','exec','-m','gpt-6-astra','-c','model_reasoning_effort="high"','--sandbox','read-only','--skip-git-repo-check','-C',str(directory),'--output-schema',str(schema),'-o',str(directory/'labels.json'),'-']
        with (directory/'process.log').open('w') as log:
            result=subprocess.run(command,input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT)
        (directory/'finished.json').write_text(json.dumps({'returncode':result.returncode,'ids':batch_ids}))
        return {'batch':index,'returncode':result.returncode}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(batch,range((len(ids)+7)//8)): print(json.dumps(result),flush=True)

if __name__=='__main__': main()
