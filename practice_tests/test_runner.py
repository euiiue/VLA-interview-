from practice.runner import DockerRunner, docker_command


def test_runner_has_no_unsafe_fallback(tmp_path):
    r=DockerRunner(image='missing',engine='/not/a/docker')
    result=r.run('C82','print("must never run on host")')
    assert result['status']=='unavailable'


def test_sandbox_command_limits_and_mounts(tmp_path):
    args=docker_command('docker','test-image','vla-owned-123',tmp_path)
    for required in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=64','--memory=2g','--cpus=1','--user=65534:65534']:
        assert required in args
    assert not any('docker.sock' in a for a in args)
    assert any('readonly' in a for a in args)


def test_rootless_without_cpu_quota_keeps_hard_cpu_limit(tmp_path):
    args=docker_command('docker','image','vla-owned',tmp_path,cpu_quota=False)
    assert '--cpus=1' not in args
    assert 'cpu=12:12' in args
    assert '--memory=2g' in args and '--pids-limit=64' in args
