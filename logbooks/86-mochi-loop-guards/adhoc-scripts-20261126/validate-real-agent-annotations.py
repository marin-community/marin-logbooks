import asyncio,json,runpy
from dataclasses import replace
from pathlib import Path
from rolloutengine.chat import ChatModel
from rolloutengine.harness.external import AgentHarness,LengthStopPolicy,ReplyMatch
from rolloutengine.agents.model_relay import ModelRelay,VllmTokenizer
from rolloutengine.test_support import ReplayEndpoint,replay_rollout,recorded_requests,without_stream
from shellbox.http_service import HttpRequest

original=ChatModel.__call__
async def native_message(self,request):
 turn=await original(self,request)
 return replace(turn,message={**turn.message,'annotations':[],'audio':None,'refusal':None})
ChatModel.__call__=native_message

def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not (k in ('annotations','audio','refusal') and v in (None,[]))}
 if isinstance(value,list):return [clean(v) for v in value]
 return value

async def main():
 term=runpy.run_path('lib/rolloutengine/tests/test_terminus.py')
 pi=runpy.run_path('lib/rolloutengine/tests/test_pi.py')
 report=[]
 for name,recorded in term['REPLAY'].items():
  model=ReplayEndpoint(recorded['model']);machine=term['replay_machine'](recorded['machine'])
  rollout=await term['terminus_rollout'](recorded,model,machine)
  assert clean(without_stream(model.requests))==clean(recorded_requests(recorded['model'])),name
  steps=[s for c in rollout.conversations for s in c.steps]
  report.append({'agent':'terminus','scenario':name,'turns':len(steps),'observation_batches':sum(bool(s.transition.observations) for s in steps),'requests_unchanged_except_empty_metadata':True})
 for name,recorded in pi['RELAY_REPLAY'].items():
  model=ReplayEndpoint(recorded['model'])
  def build(client):
   async def run(context,complete):
    relay=ModelRelay(complete,VllmTokenizer(client,pi['MODEL']))
    for request in recorded['relay_requests']:
     await relay.handle(HttpRequest(request['method'],request['path'],json.dumps(request['body']).encode()))
   return AgentHarness(run,length_stop=LengthStopPolicy.NEW_CONVERSATION,reply_match=ReplyMatch.EXACT)
  try:rollout=await replay_rollout(build,recorded['instruction'],model)
  except pi['RolloutInterrupted'] as error:rollout=error.rollout
  assert clean(model.requests)==clean(recorded_requests(recorded['model'])),name
  steps=[s for c in rollout.conversations for s in c.steps]
  report.append({'agent':'pi','scenario':name,'turns':len(steps),'observation_batches':sum(bool(s.transition.observations) for s in steps),'requests_unchanged_except_empty_metadata':True})
 Path('/tmp/mochi-loop-guards/real-agent-annotations-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
asyncio.run(main())
