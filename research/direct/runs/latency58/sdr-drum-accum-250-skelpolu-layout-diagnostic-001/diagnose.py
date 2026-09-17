from pathlib import Path
import json, os
import numpy as np
import soundfile as sf
from research.direct.run_latency58_quality import PHASE,read,require,sha,write
from research import evaluate as legacy
from research.direct import evaluate as shared
from research.metrics import MetricConfig
plan=read(PHASE/'sdr-drum-accum-250-skelpolu-audio-001/plan.json')
saved=read(Path(plan['reuse_capture_directory'])/'result.json')
manifest=read(plan['manifest']);config=read(plan['evaluation_config'])
tracks,config=shared.select_panel(manifest,config,panel='full',track_indices=[10],excerpt_starts=None,duration=15.0,alignment_samples=128)
track=tracks[0];rows=legacy._reference_intervals(track,config)
def excerpt(relative,row):
    path=legacy._safe_dataset_path(Path(manifest['root']),relative)
    return legacy._read_excerpt(path,row['reference_start'],row['reference_end'],expected_frames=track['frames'])
mixtures=[excerpt(track['mixture'],r) for r in rows]
stems=['drums','bass','vocals','other']
references=[np.stack([excerpt(track['stems'][s],r) for s in stems]) for r in rows]
metric=MetricConfig.from_mapping(config['metrics'])
def differences(a,b,path=''):
    if isinstance(a,dict):
        require(set(a)==set(b),'Keys differ')
        return [x for k in a for x in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        require(len(a)==len(b),'Length differs')
        return [x for i,(c,d) in enumerate(zip(a,b)) for x in differences(c,d,path+'/'+str(i))]
    if a==b:return []
    return [{'path':path,'stored':a,'decoded':b,'delta':b-a if isinstance(a,(float,int)) and isinstance(b,(float,int)) else None}]
result={}
for item in plan['reuse_models']:
    capid=item['capture_id']; audio={}
    for p,row in saved['audio_files'].items():
        if row['source_id']!=capid:continue
        require(sha(p)==row['sha256'],'Changed WAV')
        values,rate=sf.read(p,dtype='float32',always_2d=True)
        require(rate==44100,'Wrong rate');audio[row['clip_index'],row['stem']]=values.T
    values=[np.stack([audio[i,s] for s in stems]) for i in range(2)]
    variants={}
    for name,estimates in [('decoded_memory_order',values),('original_c_order',[np.ascontiguousarray(v) for v in values])]:
        score=legacy._score_track(track['name'],rows,mixtures,references,estimates,metric)
        per_excerpt=[legacy._score_track(track['name'],[r],[m],[ref],[est],metric) for r,m,ref,est in zip(rows,mixtures,references,estimates)]
        d=differences(saved['models'][capid]['track_score'],score)
        e=differences(saved['models'][capid]['per_excerpt_scores'],per_excerpt)
        variants[name]={'strides':[list(v.strides) for v in estimates], 'track_differences':d,'excerpt_differences':e,'track_exact':not d,'per_excerpt_exact':not e}
    require(all(np.array_equal(a,b) for a,b in zip(values,[np.ascontiguousarray(v) for v in values])),'Copy changes samples')
    result[item['id']]=variants
write(Path(__file__).parent/'result.json',{'schema':'latency58-wav-layout-diagnostic-v1','models':result,'samples_changed':False,'cuda_initialized':False,'primary_results_replaced':False})
print({k:{n:{'strides':v['strides'],'track_exact':v['track_exact'],'per_excerpt_exact':v['per_excerpt_exact'],'track_differences':len(v['track_differences'])} for n,v in r.items()} for k,r in result.items()},flush=True)
