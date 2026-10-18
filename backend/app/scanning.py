import hashlib
import json
from bisect import bisect_right, insort
from time import monotonic
from uuid import uuid4

from .analysis import stable_events
from .calibration import rectify_board
from .checkpoint import CheckpointStore
from .video import iter_frames, save_image


class VideoScan:
    def __init__(self, task, store):
        self.source = store.task_dir(task.id) / 'source.mp4'
        self.available = self.source.is_file() and self.source.stat().st_size > 0
        self.start_s, self.end_s = task.start_s, task.end_s
        self.calibration = task.calibration
        self.index_path = store.task_dir(task.id) / 'scan-index.json'
        self.folder = store.task_dir(task.id) / 'evidence'
        self.detector, self.existing = {}, set()
        self.event_times = []
        self.cursor = task.start_s - 1
        self.index = {'events': [], 'complete': False, 'progress_s': task.start_s}
        self.dirty = False
        if not self.available:
            return
        signature = {'source_size': self.source.stat().st_size, 'source_mtime': self.source.stat().st_mtime_ns,
                     'start_s': task.start_s, 'end_s': task.end_s,
                     'calibration': task.calibration.model_dump(), 'version': 2}
        index = json.loads(self.index_path.read_text(encoding='utf8')) if self.index_path.exists() else {}
        if index.get('signature') != signature:
            index = {'signature': signature, 'events': [], 'complete': False, 'progress_s': task.start_s}
            self.dirty = True
        self.index = index
        self.checkpoints = CheckpointStore(store.root)
        self.generation = hashlib.sha256(json.dumps(index['signature'], sort_keys=True).encode()).hexdigest()[:32]
        self.detector = self.checkpoints.load(task.id, self.generation, index['checkpoint']) if index.get('checkpoint') else {}
        self.task_id = task.id
        self.cursor = self.detector.get('last_time', task.start_s - 1)
        self.existing = {(event['time_s'], tuple(event['evidence_ids'])) for event in index['events']}
        self.event_times = sorted(event['time_s'] for event in index['events'])
        self.folder.mkdir(exist_ok=True)

    def _append(self, event):
        ids = []
        for image in event.images:
            name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
            if not (self.folder / name).exists():
                save_image(self.folder / name, image)
            ids.append(name)
        key = event.time_s, tuple(ids)
        if key not in self.existing:
            self.index['events'].append({'time_s': event.time_s, 'evidence_ids': ids, 'uncertain': event.uncertain})
            self.existing.add(key)
            insort(self.event_times, event.time_s)

    def consume(self, time_s, frame, *, board_image=None):
        if not self.available or self.index['complete']:
            return
        if time_s > self.end_s + 1e-6:
            self.finish()
            return
        if time_s < self.start_s - 1e-6 or time_s <= self.cursor + 1e-7:
            return
        image = board_image if board_image is not None else rectify_board(frame, self.calibration, margin=50)
        for event in stable_events(iter([(time_s, image)]), state=self.detector, finalize=False):
            self._append(event)
        self.cursor = time_s
        self.index['progress_s'] = max(self.index['progress_s'], time_s)
        self.dirty = True

    def finish(self):
        if not self.available or self.index['complete']:
            return
        for event in stable_events(iter(()), state=self.detector):
            self._append(event)
        self.index['complete'], self.index['progress_s'] = True, self.end_s
        self.dirty = True
        self.flush()

    def flush(self):
        if not self.dirty:
            return
        name = ('1' if self.index.get('checkpoint') == '0' * 32 + '.json' else '0') * 32 + '.json'
        self.checkpoints.save(self.task_id, self.generation, self.detector, name, compress=False)
        self.index['checkpoint'] = name
        temporary = self.index_path.with_name(uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(self.index, ensure_ascii=False), encoding='utf8')
        temporary.replace(self.index_path)
        self.dirty = False

    def update(self, task):
        if not self.available:
            return
        task.scan_complete = self.index['complete']
        task.scan_progress_s = self.index['progress_s']
        task.scan_events = len(self.index['events'])
        task.pending_event_count = len(self.event_times) - bisect_right(self.event_times, task.recognition_video_s)


def scan_video(task, store):
    scan = VideoScan(task, store)
    if not scan.available:
        return task
    scan.update(task)
    if scan.index['complete']:
        return store.save_analysis(task)
    task.stage = '本地扫描并保存变化画面，不等待识别'
    task = store.save_analysis(task)
    last_saved = monotonic()
    for time_s, frame in iter_frames(scan.source, max(task.start_s, scan.cursor), task.end_s):
        if time_s <= scan.cursor + 1e-7:
            continue
        scan.consume(time_s, frame)
        task.analysis_stats.scanned_frames += 1
        if monotonic() - last_saved >= 1:
            scan.flush()
            scan.update(task)
            task = store.save_analysis(task)
            last_saved = monotonic()
    scan.finish()
    scan.update(task)
    task.stage = '本地扫描完成，按时间顺序确认局面'
    return store.save_analysis(task)
