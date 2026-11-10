"""Real-container integration checks. No user service or host Python execution."""
import json
import os
import subprocess
from practice.runner import DockerRunner
from practice.exercises.catalog import get_exercises

runner=DockerRunner()
assert runner.environment()['code_ready'],runner.environment()
report=[]
for exercise in get_exercises(True):
    result=runner.run(exercise['id'],exercise['reference'])
    assert result['status']=='passed',(exercise['id'],result)
    assert len(result['cases'])==3
    report.append({'exercise':exercise['id'],'passed':True,'elapsed_ms':result['elapsed_ms']})
    print(exercise['id'],'passed',flush=True)
reference=get_exercises(True)[11] # C93 top-k
probe='''import os, socket
assert os.getuid() == 65534
assert open('/sys/fs/cgroup/memory.max').read().strip() == '2147483648'
assert open('/sys/fs/cgroup/pids.max').read().strip() == '64'
try:
    open('/root/forbidden','w').write('bad')
except OSError: pass
else: raise AssertionError('filesystem must not be writable')
s=socket.socket();s.settimeout(.2)
try:s.connect(('1.1.1.1',443))
except OSError:pass
else:raise AssertionError('network must not be available')
finally:s.close()
'''+reference['reference']
r=runner.run('C93',probe);assert r['status']=='passed',r
bad=runner.run('C93','def top_k(nums, k):\n return []\n');assert bad['status']=='failed',bad
short=DockerRunner(timeout=3)
r=short.run('C93','while True: pass');assert r['status']=='timeout',r
r=runner.run('C93',"print('x'*400000)\n"+reference['reference']);assert r['status']=='error' and '输出' in r['message'],r
left=subprocess.check_output(['docker','ps','-a','--filter','name=vla-practice-','--format','{{.Names}}'],text=True).strip();assert not left,left
os.makedirs('.local',exist_ok=True)
json.dump({'references':report,'security_probes':['non-root','read-only filesystem','no network','memory/pids limits','incorrect answer fails','infinite loop times out','output cap','container cleanup']},open('.local/runner-verification.json','w'),indent=2)
print('All 20 references and isolation probes passed.',flush=True)
