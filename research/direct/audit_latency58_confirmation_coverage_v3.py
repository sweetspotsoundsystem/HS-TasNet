"""Inventory recorded scoring/listening intervals without reading new audio or scores.

This is a coverage inventory, not proof that material was never heard or processed.
Unknown track attribution is conservatively applied to all validation tracks.
"""
from __future__ import annotations
import argparse,collections,datetime,gzip,hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SKIP={'source_bindings','root_source_bindings','source_bindings_before','source_bindings_after','bindings','protected_bindings'}
SEARCH='"(reference_start|reference_intervals|physical_reference_intervals|primary_intervals|start_seconds|excerpt_starts|start_sample|start_frame|physical_start|physical_interval|physical_excerpt_seconds)"'


def numeric(x):
    return type(x) in (int,float)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(s):
    return re.sub('[^a-z0-9]','',s.lower())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    assert out.parent==ROOT/'research/direct/runs/latency58' and not out.exists()
    manifest_path=ROOT/'research/manifests/valid.json'
    manifest=json.loads(manifest_path.read_bytes());assert manifest['sample_rate']==44100
    names=[t['name'] for t in manifest['tracks']];known=set(names);rate=44100
    train_manifest_path=ROOT/'research/manifests/train.json'
    train_manifest=json.loads(train_manifest_path.read_bytes())
    all_track_names=known|{t['name'] for t in train_manifest['tracks']}
    assert not known.intersection(t['name'] for t in train_manifest['tracks'])
    roots=[ROOT/p for p in ['research/direct/runs','research/runs','eval-results','runs'] if (ROOT/p).exists()]
    production=Path('/home/axel/autoresearch/production/hs-tasnet-c91-full-v1')
    roots.append(production)
    production_manifest_path=production/'manifests/combined.manifest.json'
    production_manifest=json.loads(production_manifest_path.read_bytes())
    all_track_names.update(t['name'] for t in production_manifest['tracks'])
    assert not known.intersection(t['name'] for t in production_manifest['tracks'])
    command=['rg','-l','-uuu','-g','*.json',SEARCH,*map(str,roots)]
    found=subprocess.run(command,capture_output=True,text=True,check=False)
    assert found.returncode in (0,1) and not found.stderr
    paths=sorted(set([Path(p) for p in found.stdout.splitlines()]+[p for root in roots for p in root.rglob('*.json.gz')]))
    excluded_audit_metadata=[p for p in paths if any(x.startswith('confirmation-coverage-audit-') for x in p.parts)]
    paths=[p for p in paths if p not in excluded_audit_metadata]
    rows=set();unknown=set();omitted=set();bindings={};contexts=set();failed=[]
    def put(start,end,tracks,path,key):
        if not numeric(start) or not numeric(end) or not 0<=start<end:return
        if tracks is not None and not tracks:return
        assigned=tuple(sorted(known if tracks is None else tracks))
        if tracks is None:unknown.add((str(path),key))
        # Samples are integer coordinates. Outward rounding is conservative.
        import math
        for name in assigned:rows.add((name,math.floor(start),math.ceil(end),str(path),key,tracks is None))
    def walk(v,tracks,path,key=''):
        if isinstance(v,dict):
            if isinstance(v.get('track_names'),list) and all(isinstance(x,str) for x in v['track_names']):
                tracks=known.intersection(v['track_names'])
            for name_key in ('track_name','track','name'):
                value=v.get(name_key)
                if isinstance(value,str) and value in known:tracks={value};break
                if isinstance(value,str) and ' - ' in value and (name_key!='name' or 'reference_start' in v or 'excerpts' in v):
                    tracks=set();break
            source=v.get('source_track_directory')
            if isinstance(source,str) and Path(source).name in known:tracks={Path(source).name}
            for a,b in [('reference_start','reference_end'),('physical_start','physical_end'),('physical_reference_start','physical_reference_end')]:
                if a in v and b in v:put(v[a],v[b],tracks,path,key+'/'+a)
            if numeric(v.get('start_seconds')):
                end=v.get('end_seconds')
                if not numeric(end) and numeric(v.get('duration_seconds')):end=v['start_seconds']+v['duration_seconds']
                if numeric(end):put(v['start_seconds']*rate,end*rate,tracks,path,key+'/seconds')
                else:omitted.add((str(path),key+'/start_seconds',str(v['start_seconds'])))
            if isinstance(v.get('excerpt_starts'),list):
                duration=v.get('duration_seconds',v.get('duration'))
                if numeric(duration):
                    for start in v['excerpt_starts']:
                        if numeric(start):put(start*rate,(start+duration)*rate,tracks,path,key+'/excerpt_starts')
                else:omitted.add((str(path),key+'/excerpt_starts',str(v['excerpt_starts'])))
            for start_key in ('start_frame','start_sample','physical_start_sample','physical_reference_start','requested_start_frame'):
                if not numeric(v.get(start_key)):continue
                end=v.get('physical_reference_end') if start_key=='physical_reference_start' else v.get('end_frame' if 'frame' in start_key else 'end_sample')
                duration=next((v[k] for k in ('duration_samples','frames','samples','num_samples','listening_frames','requested_frames','rendered_frames') if numeric(v.get(k))),None)
                if not numeric(end) and duration is not None:end=v[start_key]+duration
                if numeric(end):put(v[start_key],end,tracks,path,key+'/'+start_key)
                elif tracks is None or tracks:omitted.add((str(path),key+'/'+start_key,str(v[start_key])))
            for k,x in v.items():
                if k in SKIP or '/' in k:continue
                if k in ('reference_intervals','physical_reference_intervals','physical_interval') and isinstance(x,list):
                    pairs=[x] if len(x)==2 and all(numeric(z) for z in x) else x
                    for pair in pairs:
                        if isinstance(pair,list) and len(pair)==2:put(*pair,tracks,path,key+'/'+k)
                if k in ('physical_intervals_seconds','physical_excerpt_seconds') and isinstance(x,list):
                    for pair in x:
                        if isinstance(pair,list) and len(pair)==2 and all(numeric(z) for z in pair):put(pair[0]*rate,pair[1]*rate,tracks,path,key+'/'+k)
                if k in ('receive_interval','real_input_interval','full_processed_interval') and isinstance(x,list) and len(x)==2:
                    contexts.add((str(path),tuple(x)))
                next_tracks=({k} if k in known else set()) if k in all_track_names else tracks
                walk(x,next_tracks,path,key+'/'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,tracks,path,key+'/'+str(i))
    for path in paths:
        data=path.read_bytes();bindings[str(path)]=sha(data)
        inferred={name for name in names if normalized(name) in normalized(str(path))}
        tracks=inferred or None
        try:
            decoded=gzip.decompress(data) if path.suffix=='.gz' else data
            walk(json.loads(decoded),tracks,path)
        except (ValueError,TypeError,OverflowError) as exc:failed.append({'path':str(path),'error':str(exc)})
    def union(spans):
        merged=[]
        for a,b in sorted(set(spans)):
            if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
            else:merged.append([a,b])
        return merged
    used={name:union((a,b) for n,a,b,*_ in rows if n==name) for name in names}
    candidates={}
    for start in (15,45,90,120,150):
        a,b=start*rate,(start+15)*rate
        conflicts=[x for x in rows if x[1]<b and a<x[2]]
        candidates[str(start)]={'start_seconds':start,'end_seconds':start+15,
            'conflicting_tracks':sorted({x[0] for x in conflicts}),
            'conflicting_records':len(conflicts),'example_sources':sorted({x[3] for x in conflicts})[:12]}
    out.mkdir()
    binding_path=out/'metadata-bindings.json';binding_path.write_text(json.dumps(bindings,indent=2)+'\n')
    inventory={'schema':'latency58-recorded-interval-coverage-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'search_roots':list(map(str,roots)),'search_command':command,'metadata_files':len(paths),'excluded_audit_metadata':list(map(str,excluded_audit_metadata)), 'train_manifest':{'path':str(train_manifest_path),'sha256':sha(train_manifest_path.read_bytes())},'production_manifest':{'path':str(production_manifest_path),'sha256':sha(production_manifest_path.read_bytes())},'metadata_bindings_sha256':sha(binding_path.read_bytes()),
        'manifest':{'path':str(manifest_path),'sha256':sha(manifest_path.read_bytes())},'source':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).read_bytes())},
        'parsed_interval_records':len(rows),'unknown_attribution_applied_to_all_tracks':len(unknown),'parse_failures':failed,
        'unresolved_interval_fields':[{'path':p,'json_pointer':k,'value':v} for p,k,v in sorted(omitted)],
        'used_reference_sample_union':used,'candidate_windows':candidates,'streamed_input_prefix_records':len(contexts),
        'new_audio_read':False,'new_model_inference':False,'new_scores_computed':False,'new_reservation_published':False,
        'limitations':['Existing plans are included conservatively, even when execution is unproven.','Unknown track attribution is applied to every validation track.','Unresolved interval metadata requires review before reserving material.','Metadata coverage does not prove absence of unrecorded listening or evaluation.','Continuous streaming has already processed intervening mixture audio as state context. Any future claim is limited to additional scored windows on the same development songs.']}
    (out/'coverage.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
    records=[{'track':n,'start_sample':a,'end_sample':b,'metadata':p,'json_pointer':k,'unknown_track_attribution':u} for n,a,b,p,k,u in sorted(rows)]
    with gzip.open(out/'interval-records.json.gz','wt') as stream:json.dump(records,stream,separators=(',',':'))
    print(json.dumps({'metadata_files':len(paths),'parsed_interval_records':len(rows),'unresolved_fields':len(omitted),'parse_failures':len(failed),'candidate_windows':candidates}),flush=True)


if __name__=='__main__':main()
