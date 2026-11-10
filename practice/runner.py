"""Bounded container-only code execution with explicit unavailable states."""
from pathlib import Path
import json
import os
import selectors
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parent


def docker_command(engine,image,name,submission,cpu_quota=True):
    return [engine,'run','--name',name,'--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=64','--memory=2g','--memory-swap=2g',*(['--cpus=1'] if cpu_quota else []),'--user=65534:65534','--ulimit','cpu=12:12','--ulimit','nproc=64:64','--ulimit','nofile=128:128','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--mount',f'type=bind,src={Path(submission).resolve()},dst=/submission,readonly',
        '--mount',f'type=bind,src={ROOT / "harness.py"},dst=/opt/runner/harness.py,readonly',
        '--mount',f'type=bind,src={ROOT / "exercises/catalog.py"},dst=/opt/runner/catalog.py,readonly',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONUNBUFFERED=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
        '--entrypoint','python',image,'-I','/opt/runner/harness.py']


class DockerRunner:
    _semaphore=threading.BoundedSemaphore(1)
    def __init__(self,image=None,engine=None,timeout=25):
        self.image=image or os.getenv('VLA_RUNNER_IMAGE','vla-practice-runner:1')
        self.engine=engine or shutil.which('docker') or 'docker'
        self.timeout=timeout
    def environment(self):
        status=dict(docker_available=False,image_available=False,code_ready=False,cpu_quota=False,image=self.image,python='CPU PyTorch 2.6 / Python 3.11',message='请安装并启动 Docker，然后运行 ./scripts/build-runner.sh')
        try:
            a=subprocess.run([self.engine,'info','--format','{{json .Warnings}}'],capture_output=True,timeout=4)
            status['docker_available']=a.returncode==0
            status['cpu_quota']=a.returncode==0 and 'No cpu cfs quota' not in a.stdout.decode(errors='replace')
            if a.returncode==0:
                a=subprocess.run([self.engine,'image','inspect',self.image,'--format','{{.Id}}'],capture_output=True,timeout=4)
                status['image_available']=a.returncode==0
            status['code_ready']=status['docker_available'] and status['image_available']
            if status['code_ready']:status['message']='隔离执行器已就绪，使用 CPU 测试，不连接机器人'
        except (OSError,subprocess.TimeoutExpired):pass
        return status
    def run(self,exercise_id,code):
        start=time.monotonic()
        environment=self.environment()
        if not environment['code_ready']:return dict(status='unavailable',cases=[],stdout='',elapsed_ms=0,message='代码环境未就绪，请运行 scripts/build-runner.sh；题库和编辑仍可使用。')
        if not self._semaphore.acquire(blocking=False):return dict(status='error',cases=[],stdout='',elapsed_ms=0,message='已有代码正在测试，请稍后重试')
        name='vla-practice-'+uuid.uuid4().hex
        proc=None
        try:
            with tempfile.TemporaryDirectory(prefix='vla-submission-') as tmp:
                path=Path(tmp);path.chmod(0o755)
                (path/'candidate.py').write_text(code,encoding='utf-8');(path/'candidate.py').chmod(0o644)
                (path/'request.json').write_text(json.dumps({'exercise_id':exercise_id}));(path/'request.json').chmod(0o644)
                proc=subprocess.Popen(docker_command(self.engine,self.image,name,path,cpu_quota=environment['cpu_quota']),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
                sel=selectors.DefaultSelector();sel.register(proc.stdout,selectors.EVENT_READ)
                chunks=[];size=0;status=None
                deadline=time.monotonic()+self.timeout
                try:
                    while sel.get_map():
                        if time.monotonic()>deadline:status='timeout';break
                        for key,_ in sel.select(.1):
                            data=os.read(key.fileobj.fileno(),8192)
                            if not data:sel.unregister(key.fileobj);continue
                            size+=len(data)
                            if size>256_000:status='output_limit';break
                            chunks.append(data)
                        if status:break
                finally:sel.close()
                if status:
                    subprocess.run([self.engine,'rm','-f',name],capture_output=True,timeout=5)
                    proc.kill();proc.wait(timeout=5)
                else:
                    try:proc.wait(timeout=max(.1,deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:status='timeout';proc.kill()
                output=b''.join(chunks).decode('utf-8',errors='replace')
                elapsed=round((time.monotonic()-start)*1000)
                if status:return dict(status='timeout' if status=='timeout' else 'error',cases=[],stdout=output[-10000:],elapsed_ms=elapsed,message='运行超时，容器已终止' if status=='timeout' else '输出超过上限，容器已终止')
                marker='__VLA_RESULT__'
                if marker not in output:return dict(status='error',cases=[],stdout=output[-10000:],elapsed_ms=elapsed,message='执行器未返回测试结果，请检查代码或环境')
                log,payload=output.rsplit(marker,1)
                try:
                    result=json.loads(payload.strip());assert result['status'] in ['passed','failed'] and isinstance(result['cases'],list)
                except (ValueError,KeyError,AssertionError):return dict(status='error',cases=[],stdout=output[-10000:],elapsed_ms=elapsed,message='执行结果格式不正确')
                return {**result,'stdout':log[-10000:],'elapsed_ms':elapsed}
        except (OSError,subprocess.SubprocessError) as e:
            return dict(status='error',cases=[],stdout='',elapsed_ms=round((time.monotonic()-start)*1000),message=f'隔离执行失败：{type(e).__name__}')
        finally:
            if proc and proc.poll() is None:proc.kill()
            try:subprocess.run([self.engine,'rm','-f',name],capture_output=True,timeout=5)
            except (OSError,subprocess.SubprocessError):pass
            self._semaphore.release()
