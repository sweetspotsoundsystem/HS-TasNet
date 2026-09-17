"""Reserve additional within-song scoring windows before selecting another model."""
from __future__ import annotations
import datetime,json,os
from pathlib import Path
from research.direct.run_latency58_quality import ROOT,PHASE,read,require,sha,write
from research.direct.train_latency58 import disk_bytes


def binding(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU metadata preparation required')
    audit_dir=PHASE/'confirmation-coverage-audit-003';audit=read(audit_dir/'coverage.json')
    require(audit['metadata_files']==890 and not audit['unresolved_interval_fields'] and not audit['parse_failures']
        and audit['new_model_inference'] is False and audit['new_scores_computed'] is False
        and sha(audit_dir/'metadata-bindings.json')==audit['metadata_bindings_sha256'], 'Incomplete coverage inventory')
    metadata=read(audit_dir/'metadata-bindings.json')
    require(all(sha(p)==h for p,h in metadata.items()),'Recorded metadata changed before reservation')
    starts=[45.0,120.0];duration=15.0
    require(all(audit['candidate_windows'][str(int(s))]['conflicting_records']==0 for s in starts),
            'Proposed windows overlap recorded scoring/listening metadata')
    manifest_path=Path(audit['manifest']['path']);manifest=read(manifest_path)
    training_path=Path(audit['production_manifest']['path']);training=read(training_path)
    require(sha(manifest_path)==audit['manifest']['sha256'] and sha(training_path)==audit['production_manifest']['sha256']
        and sha(training_path)=='300b0bfbd835e2ce40c8832d10219a35941d6453392fa50688b236499375061a'
        and manifest['split']=='valid' and manifest['track_count']==14 and training['track_count']==501
        and not {t['name'] for t in manifest['tracks']}.intersection(t['name'] for t in training['tracks']),
        'Current training corpus and validation identity/exclusion differ')
    from research.direct.evaluate import select_panel
    from research.evaluate import _reference_intervals
    from research.direct.latency58_evaluate import plan_latency58_stream
    config_path=ROOT/'research/eval_config.json';config=read(config_path)
    tracks,selected=select_panel(manifest,config,panel='full',excerpt_starts=starts,duration=duration,alignment_samples=128)
    intervals={};source_audio={}
    accepted_training=read(PHASE/'cleanup-successor-prep-002/training-plan.json')
    root=Path(manifest['root'])
    for track in tracks:
        rows=_reference_intervals(track,selected)
        stream=plan_latency58_stream(rows,track['frames'])
        require(stream.receive_end<=track['frames'] and len(rows)==2,'Final callback would require invented audio')
        prior=audit['used_reference_sample_union'][track['name']]
        require(all(not (a<row['reference_end'] and row['reference_start']<b) for a,b in prior for row in rows),
                'A reserved reference sample was already covered')
        intervals[track['name']]={'reference_intervals':rows,'frames':track['frames'],'receive_end':stream.receive_end}
        for rel in [track['mixture'],*track['stems'].values()]:
            p=(root/rel).resolve();require(p.is_relative_to(root.resolve()),'Audio escapes canonical data root')
            digest=sha(p)
            require(accepted_training['source_bindings'].get(str(p))==digest,'Validation audio differs from accepted protocol')
            source_audio[str(p)]=digest
    # Neither new endpoint has been scored when this material is reserved.
    for prefix in ('cleanup-rebound-250','cleanup-lr3e6-250'):
        require(not (PHASE/(prefix+'-full14-001/result.json')).exists(), 'Reservation was delayed until after a new primary score')
    counted=sum(disk_bytes(Path(p)) for p in accepted_training['counted_roots'])
    outside=146342157+111344465+disk_bytes(ROOT/'.git/lfs')+disk_bytes(ROOT/'.git/objects')
    require(outside<500_000_000 and counted+900_000_000<79_500_000_000,'Concurrent training and reserved report space exceed cap')
    out=PHASE/'additional-confirmation-reservation-001';require(not out.exists(),'Preserve existing reservation');out.mkdir()
    paths=[manifest_path,training_path,config_path,Path(__file__).resolve(),audit_dir/'coverage.json',audit_dir/'metadata-bindings.json',
        audit_dir/'interval-records.json.gz',ROOT/'research/direct/audit_latency58_confirmation_coverage_v3.py',
        ROOT/'research/direct/evaluate.py',ROOT/'research/evaluate.py',ROOT/'research/direct/latency58_evaluate.py',ROOT/'research/metrics.py',
        PHASE/'cleanup-rebound-prep-001/training-plan.json',PHASE/'cleanup-lr3e6-prep-001/resource-plan.json']
    plan={'schema':'latency58-additional-within-track-confirmation-reservation-v1',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'reserved_not_evaluated',
        'manifest':binding(manifest_path),'config':binding(config_path),'coverage_audit':binding(audit_dir/'coverage.json'),
        'excerpt_starts':starts,'duration_seconds':duration,'track_names':[t['name'] for t in tracks],'track_intervals':intervals,
        'source_bindings':{**{str(p):sha(p) for p in paths},**source_audio},
        'primary_selection_protocol':'Unchanged full14 at 30–45 and 75–90 seconds, plus already-declared probes, Actions60, per-stem/band/absence metrics and vocal views.',
        'selection_requirement':'Choose and record one immutable candidate using only existing development evidence. Review all declared quality axes. Bind the primary selection and model state before any reserved scoring or listening export. No candidate is selected by this reservation.',
        'comparators':['accepted_8250_model','original_working_5p8ms_model'],'maximum_selected_candidates':1,
        'after_first_use':'Mark these windows as seen. They cannot be called fresh confirmation for a later adaptively selected candidate.',
        'new_inference':False,'new_audio_decoded':False,'source_audio_integrity_hashes_rechecked':True,'new_scores_computed':False,
        'training_corpus_disjoint_by_track_identity':True,'current_training_track_count':501,
        'counted_roots':accepted_training['counted_roots'],'counted_stop_bytes':79_500_000_000,'outside_reserve_bytes':500_000_000,
        'maximum_confirmation_report_bytes':12_000_000,'new_audio_export_authorized_by_this_plan':False,
        'combined_cap_bytes':80_000_000_000,'counted_bytes_at_preparation':counted,'outside_bytes_at_preparation':outside,
        'limitations':['Additional scored windows on familiar development songs; not an independent test set or new-song generalization.',
            'Earlier continuous streams processed these mixture prefixes/gaps as recurrent context. The claim concerns recorded scoring/listening use, not unseen input audio.',
            'The audit covers the recorded local metadata inventory and cannot rule out unrecorded listening/evaluation.',
            'Model selection, training-seed uncertainty, listening and deployment/runtime qualification remain separate requirements.']}
    write(out/'plan.json',plan)
    print(json.dumps({'status':plan['status'],'plan':binding(out/'plan.json'),'tracks':len(tracks),'intervals_seconds':[[s,s+duration] for s in starts],'audio_integrity_files':len(source_audio),'new_inference':False}),flush=True)


if __name__=='__main__':main()
