import importlib
import time

import httpx
import numpy as np
import pytest

from app.schemas import Calibration, Position, VideoMetadata, opening_position
from app.storage import TaskStore
from app.vision import VisionClient


def running(tmp_path):
    assert importlib.util.find_spec('app.recovery') is not None
    from app.recovery import prepare_recovery
    store = TaskStore(tmp_path)
    task = store.create('recover.mp4', VideoMetadata(width=128, height=128, duration_s=200, codec='h264'))
    task.initial_position = opening_position()
    task.status = 'analyzing'
    return store, prepare_recovery(task, store)


def test_saved_response_is_reused_after_commit_interruption(tmp_path):
    from_calls = []
    store, task = running(tmp_path)
    from app.recovery import RecoveryVision
    def response(request):
        from_calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"changes":[]}'}}], 'usage': {'total_tokens': 7}})
    images = [np.zeros((100, 100, 3), np.uint8)]
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        first = RecoveryVision(VisionClient('https://example.test/v1', 'key', 'test', client), store, task)
        assert first.recognize_position(images, reference=task.initial_position, actual=True) == [task.initial_position]
        second = RecoveryVision(VisionClient('https://example.test/v1', 'key', 'test', client), TaskStore(tmp_path), store.get(task.id))
        assert second.recognize_position(images, reference=task.initial_position, actual=True) == [task.initial_position]
        assert len(from_calls) == 1
        assert second.last_request_count == 0
        assert second.usage == [{'total_tokens': 7}]
        assert store.get(task.id).recovery.total_requests == 1


def test_failed_request_is_saved_before_network_call(tmp_path):
    store, task = running(tmp_path)
    from app.checkpoint import CheckpointStore
    from app.recovery import RecoveryVision
    def fail(request):
        persisted = store.get(task.id)
        assert persisted.recovery.pending_request
        payload = CheckpointStore(tmp_path).load(task.id, task.recovery.generation, persisted.recovery.pending_request)
        assert np.array_equal(payload['images'][0], np.zeros((100, 100, 3), np.uint8))
        raise httpx.ConnectTimeout('offline', request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        vision = RecoveryVision(VisionClient('https://example.test/v1', 'key', 'test', client), store, task)
        with pytest.raises(ValueError):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)], reference=task.initial_position, actual=True)
    assert store.get(task.id).recovery.total_requests == 1


def test_service_disconnect_respects_uncertain_request_budget(tmp_path):
    store, task = running(tmp_path)
    from app.recovery import defer_operation
    from app.vision import VisionRequestError
    error = VisionRequestError('service dropped', 'ReadError', 'response', outcome_uncertain=True)
    for attempt in range(3):
        defer_operation(store, task, error)
        task = store.get(task.id)
        if attempt < 2:
            assert task.status == 'waiting_service'
            task = store.claim_recovery(task.id, task.recovery.generation, time.time() + 10000)
    assert task.status == 'action_required'
    assert not task.recovery.auto_run


def test_connect_failure_remains_automatic_beyond_inference_budget(tmp_path):
    store, task = running(tmp_path)
    from app.recovery import defer_operation
    from app.vision import VisionRequestError
    error = VisionRequestError('offline', 'ConnectTimeout', 'connect')
    for _ in range(5):
        defer_operation(store, task, error)
        task = store.get(task.id)
        assert task.status == 'waiting_service'
        assert task.recovery.uncertain_attempts == 0
        task = store.claim_recovery(task.id, task.recovery.generation, time.time() + 10000)


def test_retried_requests_are_counted_on_top_of_existing_totals(tmp_path):
    store, task = running(tmp_path)
    from app.recovery import RecoveryVision, prepare_recovery
    task.analysis_stats.model_requests, task.analysis_stats.images_sent = 9, 18
    task = store.save(task, task.revision)
    task = prepare_recovery(task, store)
    def fail(request):
        raise httpx.ConnectTimeout('offline', request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        wrapper = RecoveryVision(VisionClient('https://example.test/v1', 'key', 'test', client), store, task)
        for _ in range(2):
            with pytest.raises(ValueError):
                wrapper.recognize_position([np.zeros((100, 100, 3), np.uint8)], reference=task.initial_position, actual=True)
    saved = store.save_analysis(task)
    assert saved.analysis_stats.model_requests == 11
    assert saved.analysis_stats.images_sent == 20


def test_network_wait_preserves_records_and_resumed_replay_matches_baseline(tmp_path, monkeypatch):
    from app.analysis import VideoEvent
    from app.chess import apply_move, legal_moves
    from app.schemas import Calibration
    import app.actual as actual
    from app.analysis import analyze_task
    store, task = running(tmp_path)
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    task.rules_enabled = False
    task.replay_mode = 'actual'
    task = store.save(task, task.revision)
    import app.recovery as recovery
    assert hasattr(recovery, 'run_saved_operation')
    from app.recovery import prepare_recovery, run_saved_operation
    task = prepare_recovery(task, store)
    initial = task.initial_position
    first = apply_move(initial, next(m for m in legal_moves(initial) if m.piece_id == 'red_pawn_4'))
    second = apply_move(first, next(m for m in legal_moves(first) if m.piece_id == 'black_pawn_4'))
    from test_tracking import board_image
    frames = [board_image(initial), board_image(first), board_image(second)]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter([
        VideoEvent(1, [board_image(a[1]), frames[1]], position=first), VideoEvent(2, frames[1:])]))
    monkeypatch.setattr(actual, 'verified_observed_candidates', lambda ref, positions, *images: positions)
    calls = []
    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectTimeout('offline', request=request)
        content = {'candidates': [{'pieces': [{'color': p.color, 'kind': p.kind, 'file': p.square.file, 'rank': p.square.rank} for p in second.pieces], 'side_to_move': 'red'}]}
        import json
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(content)}}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        vision = VisionClient('https://example.test/v1', 'key', 'test', client)
        run_saved_operation(task.id, store, vision)
        waiting = store.get(task.id)
        assert waiting.status == 'waiting_service'
        assert len(waiting.moves) == 1
        original = waiting.moves[0].model_dump()
        assert waiting.pending_turn is None
        store.claim_recovery(task.id, waiting.recovery.generation, time.time() + 10000)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        run_saved_operation(task.id, store, VisionClient('https://example.test/v1', 'key', 'test', client))
    finished = store.get(task.id)
    assert finished.status == 'ready'
    assert len(finished.moves) == 2
    assert finished.moves[0].model_dump() == original
    assert sorted((p.id, p.square.file, p.square.rank) for p in finished.moves[1].position_after.pieces) == sorted((p.id, p.square.file, p.square.rank) for p in second.pieces)
    assert len(calls) == 2
