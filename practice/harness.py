"""Runs only inside the disposable Docker container, never in the API process."""
import importlib.util
import json
import math
import random
import sys
import traceback
import torch
sys.path.insert(0, '/opt/runner')
from catalog import get_exercises

torch.set_num_threads(1)
torch.manual_seed(0)
random.seed(0)
request=json.load(open('/submission/request.json'))
exercise=next(x for x in get_exercises(True) if x['id']==request['exercise_id'])
results=[]
for test in exercise['tests']:
    torch.manual_seed(0);random.seed(0)
    try:
        spec=importlib.util.spec_from_file_location('candidate','/submission/candidate.py')
        candidate=importlib.util.module_from_spec(spec)
        sys.modules['candidate']=candidate
        spec.loader.exec_module(candidate)
        exec(test['code'],dict(candidate=candidate,torch=torch,math=math,random=random))
        results.append(dict(name=test['name'],passed=True,error=''))
    except BaseException:
        results.append(dict(name=test['name'],passed=False,error=traceback.format_exc()[-2500:]))
print('\n__VLA_RESULT__'+json.dumps(dict(status='passed' if all(t['passed'] for t in results) else 'failed',cases=results),ensure_ascii=False),flush=True)
