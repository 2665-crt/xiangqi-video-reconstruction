import json
import shutil
import time

import httpx
import numpy as np
import pytest

from app.schemas import Calibration, VideoMetadata, opening_position
from app.storage import TaskStore


def test_scan_persists_events_and_is_reused_without_decoding(tmp_path, video_path, monkeypatch):
    from app.scanning import scan_video
    store = TaskStore(tmp_path)
    task = store.create('scan.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    shutil.copy2(video_path, store.task_dir(task.id) / 'source.mp4')
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    task.status = 'analyzing'
    store.save(task, task.revision)
    scanned = scan_video(task, store)
    assert scanned.scan_complete and scanned.scan_progress_s == 2
    assert scanned.scan_events > 0
    index = json.loads((store.task_dir(task.id) / 'scan-index.json').read_text(encoding='utf8'))
    assert index['events'][0]['evidence_ids']
    monkeypatch.setattr('app.scanning.iter_frames', lambda *a: (_ for _ in ()).throw(AssertionError('cached scan decoded again')))
    cached = scan_video(scanned, store)
    assert cached.scan_complete and cached.scan_events == scanned.scan_events


def test_connection_failure_does_not_prevent_full_local_scan(tmp_path, video_path, monkeypatch):
    import app.actual as actual
    from app.analysis import VideoEvent
    from app.recovery import prepare_recovery, run_saved_operation
    from app.vision import VisionClient
    store = TaskStore(tmp_path)
    task = store.create('offline.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    shutil.copy2(video_path, store.task_dir(task.id) / 'source.mp4')
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    task.calibration_confirmed = True
    task.initial_position, task.replay_mode, task.status = opening_position(), 'actual', 'analyzing'
    task = prepare_recovery(task, store)
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter([VideoEvent(1, [np.zeros((100, 100, 3), np.uint8)])]))
    def offline(request): raise httpx.ConnectTimeout('test offline', request=request)
    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        run_saved_operation(task.id, store, VisionClient('https://example.test/v1', 'test', 'test', client))
    result = store.get(task.id)
    assert result.status == 'waiting_service'
    assert result.scan_complete and result.scan_progress_s == 2
    assert result.recognition_video_s < result.end_s
    assert result.pending_event_count > 0


def test_live_but_stalled_worker_is_reclaimed(tmp_path):
    from app.recovery import prepare_recovery
    from app.recovery_scheduler import RecoveryCoordinator
    import threading
    store = TaskStore(tmp_path)
    task = store.create('stalled.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    task.status = 'analyzing'
    task = prepare_recovery(task, store)
    task.recovery.heartbeat_at = time.time() - 1000
    task = store.save(task, task.revision)
    class Worker:
        alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout=None): pass
    worker = Worker()
    coordinator = RecoveryCoordinator(store, {task.id: worker}, threading.Lock(), lambda t: None)
    coordinator.tick()
    assert not worker.alive
    assert store.get(task.id).status == 'waiting_service'
    assert store.get(task.id).recovery.last_error == 'WorkerStalled'


def test_boundary_checkpoint_restarts_at_saved_context(tmp_path):
    from app.recovery import prepare_recovery, save_boundary_checkpoint, ReplayStore
    from app.schemas import Square, MoveRecord
    store = TaskStore(tmp_path)
    task = store.create('boundary.mp4', VideoMetadata(width=128, height=128, duration_s=200, codec='h264'))
    task.initial_position, task.status = opening_position(), 'analyzing'
    task = prepare_recovery(task, store)
    context = task.initial_position.model_copy(deep=True)
    next(p for p in context.pieces if p.id == 'red_pawn_4').square = Square(file=4, rank=5)
    context.side_to_move = 'black'
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=100, status='auto_verified', position_after=context)]
    task = store.save_analysis(task)
    save_boundary_checkpoint(store, task, 100, context)
    restored = ReplayStore(store, store.get(task.id)).get(task.id)
    assert restored.resume_video_s == 100
    assert len(restored.moves) == 1
    assert restored.context_position == context
    assert restored.pending_turn is None


def test_buffered_motion_survives_detector_checkpoint(tmp_path):
    from app.analysis import stable_events
    from app.checkpoint import CheckpointStore
    frames = [(round(i / 10, 1), np.full((100, 100, 3), 0 if i <= 2 else i * 30 if i <= 8 else 50, np.uint8))
              for i in range(16)]
    baseline = list(stable_events(iter(frames)))
    state, before = {}, []
    def interrupted():
        for timestamp, image in frames:
            if timestamp >= 1.1:
                raise RuntimeError('simulated restart during buffered motion')
            yield timestamp, image
    with pytest.raises(RuntimeError):
        before.extend(stable_events(interrupted(), state=state))
    checkpoints = CheckpointStore(tmp_path)
    name = checkpoints.save('a' * 32, 'b' * 32, state)
    restored = checkpoints.load('a' * 32, 'b' * 32, name)
    resumed = before + list(stable_events((frame for frame in frames if frame[0] > restored['last_time']), state=restored))
    assert [event.time_s for event in resumed] == [event.time_s for event in baseline]
    assert .8 in [event.time_s for event in resumed if event.uncertain]


def test_new_takeback_after_checkpoint_can_update_automatic_history(tmp_path):
    from app.chess import apply_move, legal_moves
    from app.recovery import prepare_recovery, save_boundary_checkpoint, ReplayStore
    from app.turns import checked_snapshot, TurnReplay
    store = TaskStore(tmp_path)
    task = store.create('takeback.mp4', VideoMetadata(width=128, height=128, duration_s=10, codec='h264'))
    initial = opening_position()
    first = apply_move(initial, next(m for m in legal_moves(initial) if m.piece_id == 'red_pawn_4'))
    second = apply_move(first, next(m for m in legal_moves(first) if m.piece_id == 'black_pawn_4'))
    task.initial_position, task.rules_enabled, task.status = initial, True, 'analyzing'
    task = prepare_recovery(task, store)
    task.moves = [checked_snapshot(initial, first, 1, 0, 2), checked_snapshot(first, second, 2, 2, 4)]
    task.processed_video_s = 4
    task = store.save_analysis(task)
    save_boundary_checkpoint(store, task, 4, second)
    replay_store = ReplayStore(store, store.get(task.id))
    restored = replay_store.get(task.id)
    turns = TurnReplay(restored, restored.context_position)
    restored.processed_video_s = 6
    turns.observe(first, 6, [])
    replay_store.save_analysis(restored)
    assert store.get(task.id).moves[1].violation == 'takeback'


def test_local_partial_draft_is_saved_before_scanning_starts(tmp_path, monkeypatch):
    import app.actual as actual
    from app.recovery import prepare_recovery, run_saved_operation
    from test_tracking import board_image
    store = TaskStore(tmp_path)
    task = store.create('partial.mp4', VideoMetadata(width=901, height=1001, duration_s=100, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed, task.replay_mode, task.status = True, 'actual', 'analyzing'
    task = prepare_recovery(task, store)
    draft = opening_position()
    class Local:
        last_request_count = last_image_count = 0
        last_response = {'draft_position': draft.model_dump(), 'piece_reviews': []}
        def recognize_position(self, *args, **kwargs): return []
    class Vision:
        configured = True
        usage = []
        local_classifier = Local()
        last_request_count = last_image_count = 0
        last_response = None
        def recognize_position(self, *args, **kwargs):
            self.last_response = self.local_classifier.last_response
            return []
        def close(self): pass
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *a: (0, board_image(draft)))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *a: [a[-1]])
    def scan(saved, replay):
        assert saved.initial_draft is not None
        assert store.get(saved.id).initial_draft is not None
        return saved
    monkeypatch.setattr('app.scanning.scan_video', scan)
    run_saved_operation(task.id, store, Vision())
    assert store.get(task.id).initial_draft == draft


def test_second_restart_keeps_takeback_and_pending_turn_until_event_rebuilt(tmp_path):
    from app.chess import apply_move, legal_moves
    from app.recovery import prepare_recovery, save_boundary_checkpoint, ReplayStore
    from app.turns import checked_snapshot, TurnReplay
    store = TaskStore(tmp_path)
    task = store.create('second-restart.mp4', VideoMetadata(width=128, height=128, duration_s=10, codec='h264'))
    initial = opening_position()
    first = apply_move(initial, next(m for m in legal_moves(initial) if m.piece_id == 'red_pawn_4'))
    second = apply_move(first, next(m for m in legal_moves(first) if m.piece_id == 'black_pawn_4'))
    task.initial_position, task.rules_enabled, task.status = initial, True, 'analyzing'
    task = prepare_recovery(task, store)
    task.moves = [checked_snapshot(initial, first, 1, 0, 2), checked_snapshot(first, second, 2, 2, 4)]
    task.processed_video_s = task.recognition_video_s = 4
    task = store.save_analysis(task)
    save_boundary_checkpoint(store, task, 4, second)
    first_replay = ReplayStore(store, store.get(task.id))
    restored = first_replay.get(task.id)
    turns = TurnReplay(restored, restored.context_position)
    restored.processed_video_s = restored.recognition_video_s = 6
    turns.observe(first, 6, [])
    turns.sync_pending()
    first_replay.save_analysis(restored)
    pending = store.get(task.id).pending_turn.model_dump()
    again = ReplayStore(store, store.get(task.id))
    older = again.get(task.id)
    again.save_analysis(older)
    assert store.get(task.id).moves[1].violation == 'takeback'
    assert store.get(task.id).pending_turn.model_dump() == pending
    older.processed_video_s = 6
    again.save_analysis(older)
    assert store.get(task.id).pending_turn.model_dump() == pending
    rebuilt_turns = TurnReplay(older, older.context_position)
    rebuilt_turns.observe(first, 6, [])
    rebuilt_turns.sync_pending()
    older.recognition_video_s = 6
    again.save_analysis(older)
    assert store.get(task.id).moves[1].violation == 'takeback'


def test_legal_replay_commits_event_watermark_and_finishes(tmp_path, monkeypatch):
    import app.analysis as analysis
    from app.chess import apply_move, legal_moves
    from app.recovery import prepare_recovery, run_saved_operation
    from app.schemas import MoveRecord
    from test_tracking import board_image
    store = TaskStore(tmp_path)
    task = store.create('legal-recovery.mp4', VideoMetadata(width=900, height=1000, duration_s=2, codec='h264'))
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.piece_id == 'red_pawn_4').model_copy(update={'index': 1})
    after = apply_move(initial, move)
    task.initial_position, task.status = initial, 'analyzing'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    (store.task_dir(task.id) / 'evidence').mkdir()
    task = prepare_recovery(task, store, local_only=True)
    task.moves = [MoveRecord(index=1, move=move, video_start_s=0, video_end_s=1,
                            status='auto_verified', position_after=after, observed_positions=[after])]
    task.processed_video_s = 1
    task = store.save_analysis(task)
    monkeypatch.setattr(analysis, 'iter_frames', lambda *a: iter([]))
    monkeypatch.setattr(analysis, 'tracked_events', lambda *a: iter([analysis.VideoEvent(1, [board_image(after)], position=after)]))
    run_saved_operation(task.id, store)
    result = store.get(task.id)
    assert result.status == 'ready'
    assert result.recognition_video_s == 2
    assert len(result.moves) == 1 and result.moves[0].move == move


def test_worker_exit_during_request_consumes_uncertain_retry_budget(tmp_path):
    import threading
    from app.recovery import prepare_recovery
    from app.recovery_scheduler import RecoveryCoordinator
    store = TaskStore(tmp_path)
    task = store.create('crash.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    task.status = 'analyzing'
    task = prepare_recovery(task, store)
    task.recovery.request_started_at = time.time()
    task = store.save(task, task.revision)
    class Worker:
        def is_alive(self): return False
        def join(self, timeout=None): pass
    coordinator = RecoveryCoordinator(store, {task.id: Worker()}, threading.Lock(), lambda t: None)
    coordinator.tick()
    assert store.get(task.id).recovery.uncertain_attempts == 1
