"""Run one frozen lower-rate arm after the current GPU supervisor closes and audits."""
from __future__ import annotations
import argparse,ctypes,json,os,select,subprocess,time
from pathlib import Path
from research.direct.run_latency58_quality import ROOT,PHASE,PYTHON,read,require,sha,write
from research.direct.train_latency58 import verify_inputs,disk_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    require(Path.cwd()==ROOT and sha(args.plan)==args.plan_sha256 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Frozen CPU queue required')
    plan=read(args.plan);verify_inputs(plan)
    require(plan['schema']=='latency58-cleanup-lr-sweep-queue-v1' and plan['maximum_wait_seconds']==9000
        and plan['maximum_new_production_updates']==250 and plan['maximum_resource_updates']==2
        and plan['training_waits_for_plugin_qualification'] is False and plan['training_waits_for_cpu_quality'] is False,
        'Unexpected queue scope')
    out=Path(plan['output_directory']);stage=Path(plan['previous_stage_directory']);identity=plan['supervisor']
    require(out.parent==PHASE and out.is_dir() and not (out/'result.json').exists(),'Preserve queue evidence')
    waited=0.0
    if not (stage/'audit-execution.json').exists():
        libc=ctypes.CDLL(None,use_errno=True);libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint];libc.pidfd_open.restype=ctypes.c_int
        fd=libc.pidfd_open(identity['pid'],0)
        if fd<0:
            code=ctypes.get_errno();raise OSError(code,os.strerror(code))
        try:
            proc=Path('/proc',str(identity['pid']));stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
            require(int(stat[19])==identity['start_ticks'] and argv==identity['argv']
                and argv[argv.index('-m')+1]=='research.direct.run_latency58_cleanup_rebound','Different GPU supervisor')
            write(out/'wait-start.json',{'pidfd_opened':True,'supervisor':identity,'plan_sha256':args.plan_sha256})
            began=time.monotonic();poller=select.poll();poller.register(fd,select.POLLIN)
            while not poller.poll(30000):require(time.monotonic()-began<9000,'Wait expired')
            waited=time.monotonic()-began
        finally:os.close(fd)
    verify_inputs(plan)
    prior,audit_execution,audit=[read(stage/n) for n in ('execution.json','audit-execution.json','audit.json')]
    monitor=read(prior['monitor_result'])
    require(prior['actual_exit_code']==audit_execution['actual_exit_code']==monitor['child_exit_code']==0
        and prior['source_bindings_unchanged'] and audit_execution['source_bindings_unchanged'] and audit['source_bindings_unchanged']
        and not audit_execution['timed_out'] and audit['status']=='pass' and audit['step']==250
        and prior['plan_sha256']==audit_execution['plan_sha256']==audit['plan_sha256']==plan['previous_training_plan']['sha256']
        and monitor['status']==monitor['supervisor_health']=='pass' and monitor['post_exit_quiet_completed'],
        'Prior GPU stage or independent audit failed')
    resource_path=Path(plan['resource_plan']['path']);require(sha(resource_path)==plan['resource_plan']['sha256'],'Next plan changed')
    resource=read(resource_path);verify_inputs(resource)
    from research.direct.latency58_cleanup_lr3e6_checkpoint import validate_recipe
    validate_recipe(resource)
    counted=sum(disk_bytes(Path(p)) for p in resource['counted_roots'])
    outside=146342157+111344465+disk_bytes(ROOT/'.git/lfs')+disk_bytes(ROOT/'.git/objects')
    require(outside<500_000_000 and counted+450_000_000<79_500_000_000,'Next checkpoint and parallel quality exceed cap')
    argv=[PYTHON,'-u','-m','research.direct.run_latency58_cleanup_lr3e6','--resource-plan',str(resource_path),
        '--plan-sha256',sha(resource_path),'--previous-execution',str(stage/'execution.json')]
    write(out/'command.json',{'argv':argv,'plan_sha256':args.plan_sha256})
    write(out/'wait-complete.json',{'waited_seconds':waited,'previous_audit_sha256':sha(stage/'audit.json'),
        'previous_audit_execution_sha256':sha(stage/'audit-execution.json'),'previous_execution_sha256':sha(stage/'execution.json')})
    print(json.dumps({'event':'launch_next_learning_rate_arm','production_updates':250,'resource_updates':2,'lr':resource['config']['lr']}),flush=True)
    began=time.monotonic()
    with (out/'training-console.log').open('x') as log:
        child=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=9000)
    verify_inputs(plan)
    write(out/'result.json',{'actual_exit_code':child.returncode,'source_bindings_unchanged':True,
        'elapsed_seconds':time.monotonic()-began,'quality_selected':False,
        'training_waited_for_plugin_qualification':False,'training_waited_for_cpu_quality':False})
    require(child.returncode==0,'Lower-rate arm failed; retain monitored evidence')


if __name__=='__main__':main()
