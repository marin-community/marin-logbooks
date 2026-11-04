import asyncio
import json
import os
import shutil
from pathlib import Path

import draccus
import httpx
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference

async def evaluate(endpoint, config, output):
    answers=output/'answers'
    answers.mkdir()
    async with httpx.AsyncClient(timeout=1800) as client:
        for i,row in enumerate(config['identity_rows']):
            request={'model':config['served_model'],'messages':row['messages'][:-1],
                     'chat_template_kwargs':row['chat_template_kwargs'],'temperature':0,'max_tokens':8192}
            if 'seed' in row: request['seed']=row['seed']
            response=await client.post(endpoint+'/chat/completions',json=request)
            record={'id':row['id'],'index':i,'request':request,'status_code':response.status_code,'response_raw':response.text}
            (answers/f'{i:02d}.json').write_text(json.dumps(record,indent=2))
            response.raise_for_status()
            print(f'Completed {i+1}/60',flush=True)
    (output/'summary.json').write_text(json.dumps({'answers':60,'errors':[]}))

if __name__=='__main__':
    output=Path(os.environ['IRIS_OUTPUT_DIR'])
    config=json.loads(Path('cd16_protocol.json').read_text())
    serve=json.loads(Path('cd16_serving.json').read_text())
    for name in ('cd16_protocol.json','cd16_serving.json'):
        shutil.copy(name,output/name)
    model=draccus.decode(ModelConfig,serve['model'])
    env={name:'' for name in serve['blank_environment_names']}
    env['WANDB_MODE']='disabled'
    inference=inference_config_for_model(model,AcceleratorChoice(Platform.GPU,gpu_type='H100',gpu_count=8),env_vars=env,priority=job_pb2.PRIORITY_BAND_INTERACTIVE,api_model=model.name)
    with remote_inference(inference) as session:
        endpoint=iris_ctx().client.resolve_endpoint(session.endpoint_name).rstrip('/')+'/v1'
        print('Serving ready; beginning standard 60-sample panel',flush=True)
        asyncio.run(evaluate(endpoint,config,output))
