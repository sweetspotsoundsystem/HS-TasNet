
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--mode');p.add_argument('--session');p.add_argument('--bindings')
a=p.parse_args();bindings=json.loads(a.bindings);pid=os.getpid()
ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
common={'kind':'ready','session':a.session,'pid':pid,'started':ticks}
if a.mode=='exit_before_ready':raise SystemExit(7)
if a.mode=='startup_timeout':
 print('fixture_phase:startup',file=sys.stderr,flush=True);time.sleep(.5)
if a.mode=='wrong_session':common['session']='wrong'
if a.mode=='wrong_pid':common['pid']=pid+1
if a.mode=='wrong_start':common['started']='wrong'
print(json.dumps(common),flush=True)
for line in sys.stdin:
 r=json.loads(line);received=time.monotonic_ns()
 if a.mode=='exit_before_reply':raise SystemExit(7)
 if a.mode in ('query_timeout','query_hang'):
  print('fixture_phase:sample',file=sys.stderr,flush=True)
  time.sleep(.5 if a.mode=='query_timeout' else 10)
 unit=2**20
 memory={'total':16384*unit,'reserved':512*unit,'used':1024*unit,'free':14848*unit}
 values={'uuid':'GPU-fixture','name':'Fixture','driver_version':'fixture-driver',
         'memory.total':16384,'memory.used':1024,'temperature.gpu':40,
         'power.draw':'N/A','power.limit':'N/A','utilization.gpu':'N/A'}
 payload={'schema':'latency58-persistent-nvml-v1','pid':common['pid'],
          'start_ticks':common['started'],'request_id':r['id'],'bindings':dict(bindings),
          'request_received_ns':received,'sample_started_ns':received,
          'sample_completed_ns':time.monotonic_ns(),'memory_bytes':memory,
          'values':values,'api_timings':[]}
 if a.mode=='bad_schema':payload['schema']='wrong'
 if a.mode=='bad_binding':payload['bindings']['reader_sha256']='0'*64
 if a.mode=='stale_sample':
  for k in ('request_received_ns','sample_started_ns','sample_completed_ns'):payload[k]=0
 if a.mode=='bad_memory':memory['free']-=1
 if a.mode=='bad_conversion':values['memory.used']+=1
 response={'kind':'response','session':a.session,'pid':common['pid'],
           'id':r['id']-1 if a.mode=='stale_id' else r['id'],'stdout':json.dumps(payload)}
 encoded=json.dumps(response)+'\n'
 if a.mode=='duplicate_reply':encoded+=encoded
 sys.stdout.write(encoded);sys.stdout.flush()
 if a.mode=='exit_after_reply':raise SystemExit(7)
if a.mode=='close_nonzero':raise SystemExit(7)
if a.mode=='close_hang':time.sleep(10)
