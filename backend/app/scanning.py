import hashlib
import json
from time import monotonic
from uuid import uuid4

from .analysis import stable_events
from .calibration import rectify_board
from .checkpoint import CheckpointStore
from .video import iter_frames, save_image


def scan_video(task, store):
    source = store.task_dir(task.id) / 'source.mp4'
    if not source.is_file() or source.stat().st_size == 0:
        return task
    signature = {'source_size': source.stat().st_size, 'source_mtime': source.stat().st_mtime_ns,
                 'start_s': task.start_s, 'end_s': task.end_s,
                 'calibration': task.calibration.model_dump(), 'version': 2}
    index_path = store.task_dir(task.id) / 'scan-index.json'
    index = json.loads(index_path.read_text(encoding='utf8')) if index_path.exists() else {}
    if index.get('signature') != signature:
        index = {'signature': signature, 'events': [], 'complete': False, 'progress_s': task.start_s}
    task.scan_events = len(index['events'])
    task.scan_progress_s = index['progress_s']
    task.pending_event_count = sum(event['time_s'] > task.recognition_video_s for event in index['events'])
    if index['complete']:
        task.scan_complete = True
        return store.save_analysis(task)
    folder = store.task_dir(task.id) / 'evidence'
    folder.mkdir(exist_ok=True)
    last_saved = monotonic()
    checkpoints = CheckpointStore(store.root)
    generation = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:32]
    detector = checkpoints.load(task.id, generation, index['checkpoint']) if index.get('checkpoint') else {}
    cursor = detector.get('last_time', task.start_s - 1)
    begin = max(task.start_s, cursor)
    existing = {(event['time_s'], tuple(event['evidence_ids'])) for event in index['events']}
    def persist():
        name = ('1' if index.get('checkpoint') == '0' * 32 + '.json' else '0') * 32 + '.json'
        checkpoints.save(task.id, generation, detector, name)
        index['checkpoint'] = name
        temporary = index_path.with_name(uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(index, ensure_ascii=False), encoding='utf8')
        temporary.replace(index_path)
    def boards():
        nonlocal task, last_saved
        for time_s, frame in iter_frames(source, begin, task.end_s):
            if time_s <= cursor + 1e-7:
                continue
            index['progress_s'] = max(index['progress_s'], time_s)
            task.scan_progress_s = index['progress_s']
            task.analysis_stats.scanned_frames += 1
            if monotonic() - last_saved >= 1:
                task.stage = '本地扫描并保存变化画面，不等待云端识别'
                persist()
                task = store.save_analysis(task)
                last_saved = monotonic()
            yield time_s, rectify_board(frame, task.calibration, margin=50)
    task.stage = '本地扫描并保存变化画面，不等待云端识别'
    task.scan_complete = False
    task = store.save_analysis(task)
    for event in stable_events(boards(), state=detector):
        ids = []
        for image in event.images:
            name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
            if not (folder / name).exists():
                save_image(folder / name, image)
            ids.append(name)
        key = event.time_s, tuple(ids)
        if key not in existing:
            index['events'].append({'time_s': event.time_s, 'evidence_ids': ids, 'uncertain': event.uncertain})
            existing.add(key)
        task.scan_events = len(index['events'])
        task.pending_event_count = sum(e['time_s'] >= task.recognition_video_s for e in index['events'])
    index['complete'], index['progress_s'] = True, task.end_s
    persist()
    task.scan_complete, task.scan_progress_s = True, task.end_s
    task.stage = '本地扫描完成，按时间顺序确认局面'
    return store.save_analysis(task)
