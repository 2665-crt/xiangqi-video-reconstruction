from contextlib import contextmanager
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import os
import threading

from joblib import Parallel, delayed


def _initialize_frame_worker():
    import cv2
    cv2.setNumThreads(1)
    parent = multiprocessing.parent_process()
    if parent is not None:
        def watch_parent():
            parent.join()
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()


@contextmanager
def process_frames(source, process, *, workers=4, capacity=None):
    capacity = workers * 2 if capacity is None else capacity
    if capacity < 1:
        raise ValueError('预取队列容量必须大于零')
    pending = deque()
    exhausted = False
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize_frame_worker)
    def fill():
        nonlocal exhausted
        while not exhausted and len(pending) < capacity:
            try:
                item = next(source)
            except StopIteration:
                exhausted = True
            except Exception as exc:
                failed = Future()
                failed.set_exception(exc)
                pending.append(failed)
                exhausted = True
            else:
                pending.append(pool.submit(process, item))
    def generate():
        fill()
        while pending:
            yield pending.popleft().result()
            fill()
    results = generate()
    try:
        yield results
    finally:
        results.close()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        if hasattr(source, 'close'):
            source.close()


def _process_frame(process, item):
    try:
        return process(item), None
    except Exception as exc:
        return None, exc


@contextmanager
def parallel_batches(source, process, *, workers=4, batch_size=8, backend='threading'):
    with Parallel(n_jobs=workers, backend=backend, batch_size=1, max_nbytes=None) as parallel:
        def generate():
            while True:
                batch, source_error = [], None
                for _ in range(batch_size):
                    try:
                        batch.append(next(source))
                    except StopIteration:
                        break
                    except Exception as exc:
                        source_error = exc
                        break
                for result, error in parallel(delayed(_process_frame)(process, item) for item in batch):
                    if error is not None:
                        raise error
                    yield result
                if source_error is not None:
                    raise source_error
                if len(batch) < batch_size:
                    break
        results = generate()
        try:
            yield results
        finally:
            results.close()
            if hasattr(source, 'close'):
                source.close()
