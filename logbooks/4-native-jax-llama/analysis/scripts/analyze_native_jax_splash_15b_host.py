"""Inspect retained host events without attributing untraced Python work."""
from pathlib import Path
import hashlib
import json
from marin.profiling import xplane as xp
ROOT=Path('/private/tmp')
record={'schema':'native-jax-splash-15b-host-boundary.v1','source_commit':'36e2a9a97fab22b76dfa5a03964c6600ec5713ac','host_tracer_level':1,'python_tracer_level':0,'cases':{},'limitations':['One inter-module gap per capture, not eight independent host observations; two host train markers per capture.','Nested PjitFunction events overlap and must not be added as elapsed time.','Uninstrumented marker interior is observed elapsed host time; absence of Python events does not identify the function consuming it.','Extra scalar conversion has a source-supported named-wrapper path, but it accounts for only about0.5ms, not the entire legacy dispatch interval.','Profile captures include overhead and do not by themselves prove the cause of unprofiled107M timing gains.']}
for dt in ('float32','bfloat16'):
    record['cases'][dt]={}
    for variant in ('legacy','native'):
        path=xp.find_xplane_file(ROOT/'native-jax-splash-15b-profile-results'/dt/variant/'profile')
        x=xp._xspace_message_class()();x.ParseFromString(path.read_bytes());events=[]
        for pid,plane in enumerate(x.planes,1):
            if plane.name!='/host:CPU':continue
            stats={int(k):str(v.name) for k,v in plane.stat_metadata.items()}
            meta=xp._xplane_event_metadata(plane,stat_names=stats,process_name=plane.name)
            for i,line in enumerate(plane.lines):
                if (line.display_name or line.name)!='python':continue
                events.extend(xp._xplane_line_events(line,pid=pid,tid=int(line.id or i+1),event_metadata=meta,stat_names=stats))
        markers=sorted((e for e in events if e.name=='train'),key=lambda e:e.ts);assert len(markers)==2
        steps=[]
        for marker in markers:
            local=sorted((e for e in events if marker.ts<=e.ts<marker.ts+marker.dur and e.name!='train'),key=lambda e:e.ts)
            train=next(e for e in local if e.name=='PjitFunction(_train_step)')
            converts=[e for e in local if e.name=='PjitFunction(convert_element_type)']
            row={'marker_start_us':marker.ts,'marker_duration_ms':marker.dur/1000,'train_pjit_start_after_marker_ms':(train.ts-marker.ts)/1000,'outer_train_pjit_duration_ms':train.dur/1000,'scalar_convert_start_after_marker_ms':(converts[0].ts-marker.ts)/1000 if converts else None,'outer_scalar_convert_duration_ms':converts[0].dur/1000 if converts else None,'first_recorded_nonmarker_start_ms':(local[0].ts-marker.ts)/1000,'first20ms_host_events':[{'name':e.name,'start_after_marker_ms':(e.ts-marker.ts)/1000,'duration_ms':e.dur/1000} for e in local if e.ts<marker.ts+20000 and e.dur>=20]}
            steps.append(row)
        record['cases'][dt][variant]={'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'steps':steps}
record['source_path_evidence']=['trainer.py:813–821 selects named_jit for legacy and jax.jit for explicit/native.','haliax/partitioning.py:443–455 performs cached shape and output resource inference inside every named-wrapper call.','haliax/partitioning.py:339–343 performs pspec_for(tree) then _resolve_mesh().','haliax/partitioning.py:102–105 calls is_in_jit() during mesh resolution.','haliax/jax_utils.py:216–217 probes jnp.zeros((),dtype=float32) to detect tracing, which can dispatch a scalar conversion outside jit.']
(ROOT/'native-jax-splash-15b-host-analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({dt:{v:[{k:s[k] for k in ('train_pjit_start_after_marker_ms','scalar_convert_start_after_marker_ms','outer_scalar_convert_duration_ms')} for s in d['steps']] for v,d in case.items()} for dt,case in record['cases'].items()},indent=2))
