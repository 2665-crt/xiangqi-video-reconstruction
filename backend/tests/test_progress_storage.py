import pytest

from app.schemas import AnalysisStats, MoveRecord, VideoMetadata, opening_position
from app.storage import RecoveryCancelled, TaskStore


def active(store):
    task = store.create('progress.mp4', VideoMetadata(width=128, height=128, duration_s=100, codec='h264'))
    task.status = 'analyzing'
    task.initial_position = opening_position()
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1,
                             position_before=task.initial_position, status='needs_review')]
    return store.save(task, task.revision)


def test_progress_updates_metadata_without_serializing_or_replacing_history(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    task = active(store)
    edited = store.get(task.id)
    edited.moves[0].reason = '用户补充的证据'
    edited = store.save(edited, edited.revision)
    task.processed_video_s, task.scan_progress_s, task.stage = 12, 13, '正在处理'
    task.run_stats = AnalysisStats(scanned_frames=240)
    task.usage = [{'elapsed_seconds': .5}]
    task.pending_turn = task.moves[0].model_copy(update={'index': 2})
    monkeypatch.setattr(type(task), 'model_dump_json', lambda *a, **kw: pytest.fail('进度更新不应序列化整盘棋谱'))
    saved = store.save_progress(task)
    actual = store.get(task.id)
    assert actual.moves[0].reason == '用户补充的证据'
    assert actual.processed_video_s == 12
    assert actual.scan_progress_s == 13
    assert actual.run_stats.scanned_frames == 240
    assert actual.usage == task.usage
    assert actual.pending_turn.index == 2
    assert actual.revision == edited.revision + 1 == saved.revision


def test_progress_rejects_cancelled_or_replaced_recovery_lease(tmp_path):
    from app.recovery import prepare_recovery
    store = TaskStore(tmp_path)
    task = prepare_recovery(active(store), store)
    latest = store.get(task.id)
    latest.recovery.lease_token = 'other'
    store.save(latest, latest.revision)
    with pytest.raises(RecoveryCancelled):
        store.save_progress(task)


def test_analysis_save_keeps_nested_input_and_return_values_independent(tmp_path):
    store = TaskStore(tmp_path)
    task = active(store)
    task.moves[0].evidence_ids = ['first.jpg']
    saved = store.save_analysis(task)
    task.moves[0].evidence_ids.append('later.jpg')
    task.initial_position.pieces[0].id = 'changed-input'
    assert saved.moves[0].evidence_ids == ['first.jpg']
    assert saved.initial_position.pieces[0].id != 'changed-input'
    saved.moves[0].evidence_ids.append('changed-return.jpg')
    assert store.get(task.id).moves[0].evidence_ids == ['first.jpg']


def test_motion_progress_does_not_reserialize_unchanged_completed_moves(tmp_path, monkeypatch):
    from app.actual import analyze_actual
    from app.analysis import VideoEvent
    from app.schemas import Calibration
    from test_tracking import board_image
    import app.actual as actual
    class CountingStore(TaskStore):
        full_writes = 0
        progress_writes = 0
        def save_analysis(self, task):
            self.full_writes += 1
            return super().save_analysis(task)
        def save_progress(self, task):
            self.progress_writes += 1
            return super().save_progress(task)
    store = CountingStore(tmp_path)
    task = active(store)
    task.moves = []
    task.rules_enabled, task.replay_mode, task.calibration_confirmed = True, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    task = store.save(task, task.revision)
    image = board_image(task.initial_position)
    events = [VideoEvent(0, [image], position=task.initial_position)]
    events += [VideoEvent(i / 10, [image], moving_colors=('red',), last_clear_s=0) for i in range(1, 10)]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **kw: iter(events))
    analyze_actual(task, store, None)
    assert store.get(task.id).status == 'ready'
    assert store.progress_writes >= 9
    assert store.full_writes <= 3


def test_uncertain_events_save_pending_evidence_without_rewriting_completed_moves(tmp_path, monkeypatch):
    from app.actual import analyze_actual
    from app.analysis import VideoEvent
    from app.schemas import Calibration
    from test_tracking import board_image
    import app.actual as actual
    class CountingStore(TaskStore):
        full_writes = 0
        progress_writes = 0
        def save_analysis(self, task):
            self.full_writes += 1
            return super().save_analysis(task)
        def save_progress(self, task):
            self.progress_writes += 1
            return super().save_progress(task)
    store = CountingStore(tmp_path)
    task = active(store)
    task.moves = []
    task.rules_enabled, task.replay_mode, task.calibration_confirmed = True, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    task = store.save(task, task.revision)
    image = board_image(task.initial_position)
    events = [VideoEvent(i / 10, [image], True, review_reason='落点被手遮挡') for i in range(10)]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **kw: iter(events))
    analyze_actual(task, store, None)
    saved = store.get(task.id)
    assert saved.status == 'needs_review'
    assert len(saved.moves) == 1
    assert saved.moves[0].evidence_ids
    assert saved.run_stats.uncertain_events == 10
    assert store.progress_writes >= 10
    assert store.full_writes <= 3


def test_valid_observations_save_only_pending_changes_until_turn_is_confirmed(tmp_path, monkeypatch):
    from app.actual import analyze_actual
    from app.analysis import VideoEvent
    from app.chess import apply_move, legal_moves, board_signature
    from app.schemas import Calibration
    from test_tracking import board_image
    import app.actual as actual
    class CountingStore(TaskStore):
        full_writes = 0
        progress_writes = 0
        def save_analysis(self, task):
            self.full_writes += 1
            return super().save_analysis(task)
        def save_progress(self, task):
            self.progress_writes += 1
            return super().save_progress(task)
    store = CountingStore(tmp_path)
    task = active(store)
    task.moves = []
    task.initial_turn_known = True
    task.rules_enabled, task.replay_mode, task.calibration_confirmed = True, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    task = store.save(task, task.revision)
    after = apply_move(task.initial_position, next(iter(legal_moves(task.initial_position))))
    before_image, after_image = board_image(task.initial_position), board_image(after)
    events = [VideoEvent(.1, [before_image, after_image], position=after)]
    events += [VideoEvent(i / 10, [after_image, after_image], position=after) for i in range(2, 11)]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **kw: iter(events))
    analyze_actual(task, store, None)
    saved = store.get(task.id)
    assert saved.status == 'ready'
    assert len(saved.moves) == 1 and saved.moves[0].move is not None
    assert board_signature(saved.moves[0].position_after) == board_signature(after)
    assert store.progress_writes >= 9
    assert store.full_writes <= 3


def test_valid_observation_commits_changes_to_earlier_history_immediately(tmp_path, monkeypatch):
    from app.actual import analyze_actual
    from app.analysis import VideoEvent
    from app.schemas import Calibration
    from app.turns import TurnReplay
    from test_tracking import board_image
    import app.actual as actual
    store = TaskStore(tmp_path)
    task = active(store)
    task.moves.append(task.moves[0].model_copy(update={'index': 2}, deep=True))
    task.initial_turn_known = True
    task.rules_enabled, task.replay_mode, task.calibration_confirmed = True, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    task = store.save(task, task.revision)
    image = board_image(task.initial_position)
    original_observe = TurnReplay.observe
    def annotate(self, *args, **kwargs):
        original_observe(self, *args, **kwargs)
        self.task.moves[0].violation = 'takeback'
        self.task.moves[0].violation_note = '历史步骤需关联后续悔棋'
        self.task.moves[0].related_steps = [1, 2]
    monkeypatch.setattr(TurnReplay, 'observe', annotate)
    def events():
        yield VideoEvent(1, [image, image], position=task.initial_position)
        saved = store.get(task.id)
        assert saved.moves[0].violation == 'takeback'
        assert saved.moves[0].related_steps == [1, 2]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **kw: events())
    analyze_actual(task, store, None)
    assert store.get(task.id).status == 'needs_review'
