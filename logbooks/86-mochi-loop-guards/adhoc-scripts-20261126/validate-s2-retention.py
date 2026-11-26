import faulthandler
import json
from pathlib import Path
from omegaconf import OmegaConf
from rigging.filesystem.s3_compat import configure_coreweave_s3
from cloud.iris.launch_config import load_launch_config, validate_launch_driver_config
configure_coreweave_s3()
faulthandler.dump_traceback_later(45, repeat=True)
for name in ('s2-antithrash-control','s2-antithrash-treatment'):
    print('Loading',name,flush=True)
    config=load_launch_config(Path('/tmp/mochi-loop-guards')/(name+'-retention.yaml'))
    print('Composed',name,flush=True)
    validate_launch_driver_config(config)
    trainer=config.skyrl.trainer
    checkpoints=[item for item in trainer.callbacks if item.type=='checkpoint']
    assert len(checkpoints)==1 and checkpoints[0].save_steps==1
    assert trainer.checkpoint_keep_every==1 and trainer.checkpoint_retention=='latest'
    assert config.iris.allocation.memory=='704GiB'
    assert config.iris.excluded_nvlink_domains==[] and config.iris.priority=='production'
    assert trainer.max_steps==8  # Unsubmitted placeholder, N still awaits S0 timings.
    print(json.dumps({'arm':name,'checkpoint_every':1,'keep_every':1,'updates_placeholder':8,'submitted':False}),flush=True)
faulthandler.cancel_dump_traceback_later()
