from fastapi.testclient import TestClient

from app.main import create_app
from contextlib import nullcontext
from types import SimpleNamespace

import psutil
import pytest

from app.resources import ResourceMonitor


def test_resource_api_reports_capacity_and_sampling_state(tmp_path):
    app = create_app(tmp_path / 'data', tmp_path / 'videos')
    with TestClient(app) as client:
        response = client.get('/api/resources')
        assert response.status_code == 200
        sample = response.json()
        assert sample['cpu_total_percent'] == 100
        assert sample['logical_processors'] >= 1
        assert sample['status'] in ('sampling', 'ready')
        if sample['status'] == 'sampling':
            assert sample['instance_cpu_percent'] is None
            assert sample['available_cpu_percent'] is None
    assert not app.state.resources.thread.is_alive()


class Process:
    def __init__(self, pid, created=0, seconds=0):
        self.pid, self.created, self.seconds = pid, created, seconds
        self.descendants = []

    def children(self, recursive):
        assert recursive
        return self.descendants

    def oneshot(self):
        return nullcontext()

    def create_time(self):
        return self.created

    def cpu_times(self):
        if isinstance(self.seconds, Exception):
            raise self.seconds
        return SimpleNamespace(user=self.seconds, system=0)


@pytest.fixture
def sampler(monkeypatch):
    root = Process(100)
    monkeypatch.setattr('app.resources.psutil.Process', lambda _: root)
    monkeypatch.setattr('app.resources.psutil.cpu_count', lambda: 4)
    monkeypatch.setattr('app.resources.psutil.cpu_percent', lambda interval: 40)
    clock = {'monotonic': 10, 'wall': 1000}
    monkeypatch.setattr('app.resources.monotonic', lambda: clock['monotonic'])
    monkeypatch.setattr('app.resources.time', lambda: clock['wall'])
    return ResourceMonitor(), root, clock


def test_cpu_combines_only_this_process_tree_and_normalizes_to_the_whole_machine(sampler):
    monitor, root, clock = sampler
    child = Process(200)
    root.descendants = [child]
    monitor.sample()
    assert monitor.snapshot()['instance_cpu_percent'] is None
    clock.update(monotonic=12, wall=1002)
    root.seconds, child.seconds = 1, 1
    monitor.sample()
    sample = monitor.snapshot()
    assert sample['status'] == 'ready'
    assert sample['instance_cpu_percent'] == 25
    assert sample['system_cpu_percent'] == 40
    assert sample['available_cpu_percent'] == 60


def test_new_analysis_child_cpu_is_counted_in_its_first_interval(sampler):
    monitor, root, clock = sampler
    monitor.sample()
    clock.update(monotonic=12, wall=1002)
    root.descendants = [Process(200, created=1001, seconds=2)]
    monitor.sample()
    assert monitor.snapshot()['instance_cpu_percent'] == 25


def test_reused_pid_and_exited_child_do_not_reuse_old_cpu_counters(sampler):
    monitor, root, clock = sampler
    root.descendants = [Process(200, created=5, seconds=500), Process(300, seconds=100)]
    monitor.sample()
    clock.update(monotonic=12, wall=1002)
    root.descendants = [Process(200, created=20, seconds=1)]
    monitor.sample()
    assert monitor.snapshot()['instance_cpu_percent'] == 0
    assert monitor.snapshot()['status'] == 'ready'


def test_inaccessible_child_does_not_publish_a_false_zero(sampler):
    monitor, root, clock = sampler
    child = Process(200)
    root.descendants = [child]
    monitor.sample()
    clock.update(monotonic=12, wall=1002)
    child.seconds = psutil.AccessDenied(200)
    monitor.sample()
    assert monitor.snapshot()['status'] == 'unavailable'
    assert monitor.snapshot()['instance_cpu_percent'] is None
