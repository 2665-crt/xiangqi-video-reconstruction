import os
import time

import pytest


def frame_job(index):
    time.sleep(.02 * (3 - index % 3))
    if index == 7:
        raise ValueError('broken frame')
    return index, os.getpid()


def test_process_pipeline_keeps_order_bounds_prefetch_and_uses_child_processes():
    from app.preprocessing import process_frames
    consumed = []
    def source():
        for index in range(7):
            consumed.append(index)
            yield index
    with process_frames(source(), frame_job, workers=2, capacity=4) as results:
        first = next(results)
        assert len(consumed) == 4
        time.sleep(.2)
        assert len(consumed) == 4
        rows = [first, *results]
    assert [row[0] for row in rows] == list(range(7))
    assert all(row[1] != os.getpid() for row in rows)
    assert len({row[1] for row in rows}) == 2


def test_process_pipeline_preserves_valid_frames_before_source_or_worker_error():
    from app.preprocessing import process_frames
    def source():
        yield 1
        yield 2
        raise ValueError('decode failed')
    with process_frames(source(), frame_job, workers=2, capacity=4) as rows:
        assert next(rows)[0] == 1
        assert next(rows)[0] == 2
        with pytest.raises(ValueError, match='decode failed'):
            next(rows)
    with process_frames(iter(range(9)), frame_job, workers=2, capacity=4) as rows:
        assert [next(rows)[0] for _ in range(7)] == list(range(7))
        with pytest.raises(ValueError, match='broken frame'):
            next(rows)


def test_frame_workers_exit_when_analysis_parent_is_terminated():
    from pathlib import Path
    import subprocess
    import sys
    import psutil
    parent = subprocess.Popen([sys.executable, str(Path(__file__).with_name('process_parent_runner.py'))],
                              stdout=subprocess.PIPE, text=True,
                              env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    worker = None
    try:
        worker = psutil.Process(int(parent.stdout.readline().strip()))
        parent.terminate()
        parent.wait(timeout=5)
        worker.wait(timeout=5)
        assert not worker.is_running()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if worker is not None and worker.is_running():
            worker.kill()
