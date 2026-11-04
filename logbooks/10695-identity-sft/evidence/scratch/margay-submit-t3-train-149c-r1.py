"""One invocation of the reviewed native CLI, only after the combined CLEAR exists."""
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

C=Path('/home/karan/oa/marin-task-312-t3-prep')
os.chdir(C)
load_dotenv('/home/karan/oa/marin/.env')
from rigging.filesystem.s3_compat import configure_coreweave_s3
configure_coreweave_s3()
from experiments.post_training.identity_context_distill.lr0_probe import emulate_pod_env
emulate_pod_env()
from experiments.post_training.identity_t3.launch import main

clear=Path(sys.argv[1]).resolve()
N='/home/karan/Dropbox/marin-notes/snowball-identity/t310'
R='s3://marin-rno2a/marin/experiments/snowball-identity'
sys.argv=['t3-launch','--job-name','task312-cd16t3-train-r1',
 '--run-root',R+'/t312/cd16-t3/train-r1','--freeze-root',R+'/t312/cd16-t3/freeze-r1',
 '--freeze-sha256','286bc759bc614f63e57084f771b5f5d2ec8db0baebace8b8d381af74db94199a',
 '--runtime-receipt',N+'/runtime-recipe-02036367-torch-order-xla-default.json',
 '--runtime-receipt-sha256','565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e',
 '--lr0-run-root',R+'/t310/lr0-r8','--source-object-list',R+'/t310/lr0-r8/audits/source-object-list.json',
 '--timeout-seconds','5400','--orders',N+'/current-orders.md',
 '--plan-output',N+'/cd16-t3/train-native-plan-r1-149c.json',
 '--expect-commit','149c1dae21054fc5df01ae794d14717afd54a1cb',
 '--combined-clear',str(clear),'--launch-receipt',N+'/cd16-t3/train-r1-launch.json','--run']
main()
