"""Attribute one main TensorCore track inside actual train-module windows."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import msgspec
from marin.profiling.semantics import classify_semantic_family
from marin.profiling.trace_summary import BreakdownMode, TraceEventTrack, _TraceSummaryBuilder
from marin.profiling.xplane import find_xplane_file, parse_xplane_timeline


def region(event):
    text = (event.tf_op or '') + ' ' + (event.source or '')
    if 'fused_cross_entropy' in text or 'linear_softmax_cross_entropy' in text:
        return 'shared CE/output projection'
    if any(x in text for x in ('LlamaEmbedding', 'embed_tokens', 'token_embeddings')):
        return 'embedding'
    if 'LlamaMlp' in text:
        return 'decoder MLP'
    if any(x in text for x in ('/Attention/', '/dot_product_attention/', 'splash_mha', '/attention/', '/self_attn/')):
        return 'decoder attention'
    if any(x in text for x in ('LlamaTransformer', 'activations', 'llama_native.py')):
        return 'decoder other'
    if any(x in text for x in ('optax/', 'equinox/_update', 'adam')):
        return 'optimizer/update'
    return 'unattributed'


def analyze(profile_dir, expected_steps):
    path = find_xplane_file(profile_dir)
    timeline = parse_xplane_timeline(path, breakdown_mode=BreakdownMode.EXCLUSIVE_GLOBAL)
    if timeline.quality_warnings:
        raise ValueError(timeline.quality_warnings)
    device_tracks = defaultdict(dict)
    for track in timeline.tracks:
        if isinstance(track, TraceEventTrack) and (track.process_name or '').startswith('/device:TPU:'):
            if track.thread_name in ('XLA Ops', 'XLA Modules'):
                if track.thread_name in device_tracks[track.process_name]:
                    raise ValueError('Duplicate main track')
                device_tracks[track.process_name][track.thread_name] = (track, list(track.events))
    cores = {}
    for core, tracks in sorted(device_tracks.items()):
        if set(tracks) != {'XLA Ops', 'XLA Modules'}:
            continue
        op_track, ops = tracks['XLA Ops']
        all_modules = tracks['XLA Modules'][1]
        modules = sorted((m for m in all_modules if 'train_step' in m.name), key=lambda m:m.ts)
        if len(modules) != expected_steps:
            raise ValueError(f'{core}: expected {expected_steps} train modules; saw {[(m.name,m.dur) for m in all_modules]}')
        if any(a.ts+a.dur>b.ts for a,b in zip(modules,modules[1:])):
            raise ValueError('Overlapping train modules')
        steps=[]
        for module in modules:
            clipped=[]
            for event in ops:
                start=max(module.ts,event.ts);end=min(module.ts+module.dur,event.ts+event.dur)
                if end>start:
                    clipped.append(msgspec.structs.replace(event,ts=start,dur=end-start))
            clipped.sort(key=lambda e:(e.ts,-(e.ts+e.dur)))
            builder=_TraceSummaryBuilder(breakdown_mode=BreakdownMode.EXCLUSIVE_PER_TRACK)
            builder.add_track(TraceEventTrack(op_track.pid,op_track.tid,op_track.process_name,op_track.thread_name,clipped))
            durations=builder.device_exclusive_durations
            occupied=sum(durations)
            if occupied>module.dur+1:
                raise ValueError(f'Overcounted overlap: {occupied} > {module.dur}')
            regions=defaultdict(float);semantics=defaultdict(float);sources=defaultdict(float);names=defaultdict(float)
            for event,duration in zip(builder.device_events,durations,strict=True):
                regions[region(event)]+=duration
                semantics[classify_semantic_family(event.name)]+=duration
                sources[(event.source or '',event.tf_op or '')]+=duration
                names[event.name]+=duration
            assert abs(sum(regions.values())-occupied)<1e-4
            steps.append({'module_name':module.name,'start_us':module.ts,'module_ms':module.dur/1000,'occupied_main_track_ms':occupied/1000,'uncovered_main_track_ms':max(0,module.dur-occupied)/1000,'region_ms':{k:v/1000 for k,v in regions.items()},'semantic_ms':{k:v/1000 for k,v in semantics.items()},'top_sources_ms':[{'source':k[0],'scope':k[1],'ms':v/1000} for k,v in sorted(sources.items(),key=lambda kv:-kv[1])[:35]],'top_ops_ms':[{'name':k,'ms':v/1000} for k,v in sorted(names.items(),key=lambda kv:-kv[1])[:30]]})
        cores[core]={'steps':steps,'mean_module_ms':statistics.mean(s['module_ms'] for s in steps),'mean_region_ms':{k:statistics.mean(s['region_ms'].get(k,0) for s in steps) for k in set().union(*(s['region_ms'] for s in steps))},'mean_semantic_ms':{k:statistics.mean(s['semantic_ms'].get(k,0) for s in steps) for k in set().union(*(s['semantic_ms'] for s in steps))},'ignored_nontraining_modules':[(m.name,m.dur/1000) for m in all_modules if 'train_step' not in m.name]}
    if not cores:
        raise ValueError('No complete TensorCore tracks')
    return {'schema':'native-jax-main-track-train-windows.v1','analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profile_path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'quality_warnings':timeline.quality_warnings,'device_cores':cores,'mean_core_region_ms_per_step':{k:statistics.mean(c['mean_region_ms'].get(k,0) for c in cores.values()) for k in set().union(*(c['mean_region_ms'] for c in cores.values()))},'interpretation':'Means across cores, never summed as wall time. Only each core main XLA Ops track inside observed train modules; async copies/SparseCore overlaps excluded. Regions are metadata labels, unmatched decoder work remains other/unattributed; these are profile timings, not the unprofiled benchmark samples.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile-dir',type=Path,required=True);p.add_argument('--expected-steps',type=int,default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.profile_dir,a.expected_steps);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cores':len(result['device_cores']),'mean_core_region_ms_per_step':result['mean_core_region_ms_per_step'],'output':str(a.output)}))
