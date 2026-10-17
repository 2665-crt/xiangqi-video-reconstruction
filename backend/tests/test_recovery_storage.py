import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.schemas import Task, VideoMetadata
from app.storage import TaskStore


def queued(store):
    task = store.create('recover.mp4', VideoMetadata(width=128, height=128, duration_s=200, codec='h264'))
    payload = task.model_dump()
    payload.update(status='waiting_service', recovery={'auto_run': True, 'operation': 'analyze', 'generation': 'a' * 32, 'next_attempt_at': 10})
    return store.save(Task.model_validate(payload), task.revision)


def test_checkpoint_survives_store_recreation(tmp_path):
    assert importlib.util.find_spec('app.checkpoint') is not None
    from app.checkpoint import CheckpointStore
    checkpoint = CheckpointStore(tmp_path)
    state = {'image': np.arange(36, dtype=np.uint8).reshape(3, 4, 3), 'nested': {'timer': 2.5}, 'phase': 2}
    name = checkpoint.save('a' * 32, 'b' * 32, state)
    restored = CheckpointStore(tmp_path).load('a' * 32, 'b' * 32, name)
    assert np.array_equal(restored['image'], state['image'])
    assert restored['nested'] == {'timer': 2.5}
    assert restored['phase'] == 2


def test_only_one_scheduler_claims_due_operation(tmp_path):
    store = TaskStore(tmp_path)
    assert hasattr(store, 'claim_recovery')
    task = queued(store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: TaskStore(tmp_path).claim_recovery(task.id, 'a' * 32, 11), range(2)))
    assert sum(r is not None for r in results) == 1
    assert store.get(task.id).status == 'analyzing'
    assert store.get(task.id).recovery.lease_token


def test_old_generation_cannot_overwrite_stopped_task(tmp_path):
    store = TaskStore(tmp_path)
    assert hasattr(store, 'claim_recovery')
    task = queued(store)
    running = store.claim_recovery(task.id, 'a' * 32, 11)
    latest = store.get(task.id)
    latest.recovery.auto_run = False
    latest.recovery.lease_token = None
    latest.status = 'interrupted'
    store.save(latest, latest.revision)
    running.status = 'ready'
    with pytest.raises(ValueError):
        store.save_analysis(running)
    assert store.get(task.id).status == 'interrupted'


def test_restart_recovers_authorized_wait_but_not_manual_stop(tmp_path):
    store = TaskStore(tmp_path)
    assert hasattr(store, 'claim_recovery')
    task = queued(store)
    running = store.claim_recovery(task.id, 'a' * 32, 11)
    store.recover_interrupted()
    recovered = store.get(task.id)
    assert recovered.status == 'waiting_service'
    assert recovered.recovery.auto_run
    recovered.status = 'interrupted'
    recovered.recovery.auto_run = False
    store.save(recovered, recovered.revision)
    store.recover_interrupted()
    assert store.get(task.id).status == 'interrupted'
