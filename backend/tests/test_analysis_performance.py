from collections import OrderedDict
import hashlib
import json
import shutil
import zipfile

import numpy as np
from app.analysis import stable_events
from app.schemas import Calibration, Square, VideoMetadata, opening_position
from app.storage import TaskStore


def event_values(events):
    return [(e.time_s, e.uncertain, [hashlib.sha256(i.tobytes()).hexdigest() for i in e.images]) for e in events]


def test_uncompressed_scan_checkpoint_preserves_exact_arrays_and_old_reader(tmp_path):
    from app.checkpoint import CheckpointStore
    checkpoints = CheckpointStore(tmp_path)
    image = np.random.default_rng(12).integers(0,256,(100,100,3),dtype=np.uint8)
    state = {'images':[image],'last_time':1.25,'pending':image.copy()}
    name = checkpoints.save('a'*32,'b'*32,state,compress=False)
    with zipfile.ZipFile(checkpoints.directory('a'*32,'b'*32)/(name[:-5]+'.npz')) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    restored = checkpoints.load('a'*32,'b'*32,name)
    assert restored['last_time'] == 1.25
    assert np.array_equal(restored['images'][0],image)
    assert np.array_equal(restored['pending'],image)


def test_incremental_scan_matches_whole_stream_including_buffered_motion():
    frames = [(i / 10, np.full((100, 100, 3), 0 if i < 3 else i * 23 if i < 9 else 40, np.uint8)) for i in range(16)]
    expected = event_values(stable_events(iter(frames)))
    state, events = {}, []
    for item in frames:
        events.extend(stable_events(iter([item]), state=state, finalize=False))
    events.extend(stable_events(iter(()), state=state))
    assert event_values(events) == expected


def test_shared_scan_preserves_index_evidence_and_resumes_without_redecoding(tmp_path, video_path):
    from app.scanning import VideoScan, scan_video
    from app.video import iter_frames
    store = TaskStore(tmp_path)
    task = store.create('stream.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    shutil.copy2(video_path, store.task_dir(task.id) / 'source.mp4')
    task.calibration = Calibration(corners=[{'x':0,'y':0},{'x':127,'y':0},{'x':127,'y':127},{'x':0,'y':127}])
    scanner = VideoScan(task, store)
    frames = list(iter_frames(video_path, 0, 2))
    for item in frames[:11]:
        scanner.consume(*item)
    scanner.flush()
    resumed = VideoScan(task, store)
    for item in frames:
        resumed.consume(*item)
    resumed.finish()
    resumed.update(task)
    resumed.flush()
    first = json.loads((store.task_dir(task.id)/'scan-index.json').read_text(encoding='utf8'))
    assert task.scan_complete and task.scan_progress_s == 2
    assert all((store.task_dir(task.id)/'evidence'/i).is_file() for e in first['events'] for i in e['evidence_ids'])
    cached = scan_video(task, store)
    assert cached.scan_events == len(first['events'])
    # 相同源、定位与时间范围，单独全流扫描必须生成完全相同的证据。
    other = store.create('whole.mp4', task.metadata)
    shutil.copy2(video_path, store.task_dir(other.id)/'source.mp4')
    other.calibration = task.calibration
    scan_video(other, store)
    second = json.loads((store.task_dir(other.id)/'scan-index.json').read_text(encoding='utf8'))
    assert first['events'] == second['events']


def test_tracking_cache_reuses_only_readonly_frame_and_same_reference(monkeypatch):
    import app.actual as actual
    calls = []
    def read(image, reference):
        calls.append(reference)
        return {(0,0):'red'}
    monkeypatch.setattr(actual, '_read_tracking_cells', read)
    token = actual._tracking_cache.set(OrderedDict())
    try:
        image = np.zeros((100,100,3),np.uint8)
        initial = opening_position()
        actual._tracking_cells(image, initial)
        actual._tracking_cells(image, initial)
        assert len(calls) == 2
        image.setflags(write=False)
        cells = actual._tracking_cells(image, initial)
        cells.clear()
        assert actual._tracking_cells(image, initial) == {(0,0):'red'}
        assert len(calls) == 3
        changed = initial.model_copy(deep=True)
        changed.pieces[0].square = Square(file=0,rank=1)
        actual._tracking_cells(image, changed)
        assert len(calls) == 4
        actual._tracking_cells(image.copy(), initial)
        assert len(calls) == 5
    finally:
        actual._tracking_cache.reset(token)


def test_tracking_cache_is_bounded_and_does_not_reuse_reenabled_mutable_frame(monkeypatch):
    import app.actual as actual
    calls = []
    monkeypatch.setattr(actual,'_read_tracking_cells',lambda image, reference:(calls.append(1) or {(0,0):int(image[0,0,0])}))
    cache = OrderedDict()
    token = actual._tracking_cache.set(cache)
    try:
        images = [np.full((2,2,3),i,np.uint8) for i in range(70)]
        for image in images:
            image.setflags(write=False)
            actual._tracking_cells(image)
        assert len(cache) <= 64
        image = images[-1]
        image.setflags(write=True)
        image[:] = 120
        actual._tracking_cells(image)
        image.setflags(write=False)
        assert actual._tracking_cells(image) == {(0,0):120}
    finally:
        actual._tracking_cache.reset(token)


def test_actual_analysis_starts_processing_before_full_scan(tmp_path, video_path, monkeypatch):
    import app.actual as actual
    from app.analysis import VideoEvent
    store = TaskStore(tmp_path)
    task = store.create('shared.mp4',VideoMetadata(width=128,height=128,duration_s=2,codec='h264'))
    shutil.copy2(video_path,store.task_dir(task.id)/'source.mp4')
    task.initial_position, task.context_position = opening_position(), opening_position()
    task.calibration = Calibration(corners=[{'x':0,'y':0},{'x':127,'y':0},{'x':127,'y':127},{'x':0,'y':127}])
    task.calibration_confirmed, task.replay_mode, task.status = True,'actual','analyzing'
    store.save_analysis(task)
    seen = []
    reads = []
    original_frames = actual.iter_frames
    def frames_once(*args, **kwargs):
        reads.append(args)
        yield from original_frames(*args, **kwargs)
    def no_second_scan(*args, **kwargs):
        raise AssertionError('standalone scan decoded the video a second time')
    monkeypatch.setattr(actual,'iter_frames',frames_once)
    monkeypatch.setattr('app.scanning.iter_frames',no_second_scan)
    def events(frames, *args, **kwargs):
        timestamp,image = next(frames)
        seen.append(store.get(task.id).scan_complete)
        yield VideoEvent(timestamp,[image],position=task.initial_position)
        for _ in frames:
            pass
    monkeypatch.setattr(actual,'actual_events',events)
    monkeypatch.setattr(actual,'_tracking_cells',lambda *a:{})
    actual.analyze_actual(task,store,None)
    saved=store.get(task.id)
    assert seen == [False]
    assert len(reads) == 1
    assert saved.scan_complete and saved.scan_progress_s == 2
    assert saved.status != 'failed'
