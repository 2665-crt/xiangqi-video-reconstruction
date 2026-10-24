import os
import threading
from time import monotonic, time

import psutil


class ResourceMonitor:
    def __init__(self):
        self.root = psutil.Process(int(os.getenv('XIANGQI_DESKTOP_PID', str(os.getpid()))))
        self.logical_processors = psutil.cpu_count() or 1
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.previous = {}
        self.previous_at = None
        self.previous_wall = None
        self.value = self.empty('sampling')

    def empty(self, status):
        return {'status': status, 'instance_id': os.getenv('XIANGQI_INSTANCE_ID'),
                'cpu_total_percent': 100, 'logical_processors': self.logical_processors,
                'system_cpu_percent': None, 'instance_cpu_percent': None,
                'available_cpu_percent': None, 'sampled_at': None}

    def snapshot(self):
        with self.lock:
            return dict(self.value)

    def sample(self):
        now = monotonic()
        wall = time()
        current = {}
        complete = True
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                with process.oneshot():
                    identity = (process.pid, process.create_time())
                    usage = process.cpu_times()
                    current[identity] = usage.user + usage.system
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                complete = False
        system = psutil.cpu_percent(interval=None)
        value = self.empty('sampling')
        if self.previous_at is not None and now > self.previous_at:
            consumed = sum(max(0, seconds - self.previous.get(identity, 0))
                           for identity, seconds in current.items()
                           if identity in self.previous or identity[1] >= self.previous_wall)
            value.update(status='ready' if complete else 'unavailable',
                         system_cpu_percent=round(system, 1),
                         instance_cpu_percent=round(min(100, consumed / (now - self.previous_at)
                                                       / self.logical_processors * 100), 1) if complete else None,
                         available_cpu_percent=round(max(0, 100 - system), 1), sampled_at=wall)
        self.previous, self.previous_at = current, now
        self.previous_wall = wall
        with self.lock:
            self.value = value

    def _run(self):
        while not self.stopped.is_set():
            try:
                self.sample()
            except (psutil.Error, OSError):
                self.previous, self.previous_at = {}, None
                with self.lock:
                    self.value = self.empty('unavailable')
            if self.stopped.wait(1):
                break

    def start(self):
        self.thread = threading.Thread(target=self._run, name='cpu-resource-monitor', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=3)
