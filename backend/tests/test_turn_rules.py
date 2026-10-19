import pytest
from fastapi.testclient import TestClient

from app.actual import analyze_actual, correct_actual, snapshot_record
from app.main import create_app
from app.schemas import Calibration, Move, MoveRecord, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from test_actual import relocated
from test_tracking import board_image


def run_frames(tmp_path, monkeypatch, frames, end=4, initial=None):
    import app.actual as actual
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter(frames))
    store = TaskStore(tmp_path)
    task = store.create('turns.mp4', VideoMetadata(width=901, height=1001, duration_s=end, codec='h264'))
    task = task.model_copy(update={'rules_enabled': True})
    task.replay_mode, task.initial_position = 'actual', initial or opening_position()
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                          {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    analyze_actual(task, store, None)
    return store.get(task.id)


def stable_frames(*positions):
    frames = [(0, board_image(opening_position()))]
    for i, position in enumerate(positions, 1):
        frames.extend([(i, board_image(position)), (i + .1, board_image(position)),
                       (i + .2, board_image(position))])
    return frames


def test_same_rook_slide_is_one_legal_move_from_original_square(tmp_path, monkeypatch):
    first = relocated(opening_position(), [('red_rook_0', 0, 8)])
    last = relocated(first, [('red_rook_0', 0, 7)])
    reply = relocated(last, [('black_rook_0', 0, 1)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(first, last, reply))
    assert result.status == 'ready'
    assert len(result.moves) == 2
    assert result.moves[0].move.source == Square(file=0, rank=9)
    assert result.moves[0].move.target == Square(file=0, rank=7)
    assert result.moves[0].position_after.side_to_move == 'black'


def test_diagonal_final_rook_landing_requires_review(tmp_path, monkeypatch):
    first = relocated(opening_position(), [('red_rook_0', 0, 8)])
    last = relocated(first, [('red_rook_0', 1, 8)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(first, last), end=3)
    assert len(result.moves) == 1
    assert result.moves[0].status == 'needs_review'
    assert result.moves[0].position_after is None
    assert result.moves[0].violation == 'illegal_move'
    assert result.confirmed_prefix_length == 0


def test_two_pieces_same_turn_are_not_automatically_reproduced(tmp_path, monkeypatch):
    first = relocated(opening_position(), [('red_rook_0', 0, 8)])
    last = relocated(first, [('red_pawn_4', 4, 5)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(first, last), end=3)
    assert len(result.moves) == 1
    assert result.moves[0].status == 'needs_review'
    assert result.moves[0].move is None
    assert result.moves[0].position_after is None
    assert result.moves[0].violation == 'multiple_pieces'


def test_slide_returning_to_start_before_reply_is_not_a_move(tmp_path, monkeypatch):
    initial = opening_position()
    first = relocated(initial, [('red_rook_0', 0, 8)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(first, initial), end=3)
    assert result.status == 'ready'
    assert result.moves == []


def test_opponent_lift_confirms_last_clear_board_before_lift(tmp_path, monkeypatch):
    first = relocated(opening_position(), [('red_rook_0', 0, 8)])
    last = relocated(first, [('red_rook_0', 0, 7)])
    lifted = last.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'black_rook_0']
    frames = stable_frames(first, last) + [(3, board_image(lifted)), (3.2, board_image(lifted))]
    result = run_frames(tmp_path, monkeypatch, frames)
    assert result.moves[0].status == 'auto_verified'
    assert result.moves[0].move.target == Square(file=0, rank=7)
    assert result.moves[0].video_end_s == pytest.approx(2.2)
    assert result.moves[-1].status == 'needs_review'


def test_video_end_confirms_last_stable_move_without_reply(tmp_path, monkeypatch):
    after = relocated(opening_position(), [('red_rook_0', 0, 8)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(after), end=2)
    assert result.status == 'ready'
    assert len(result.moves) == 1
    assert result.moves[0].video_end_s == 2


def test_takeback_keeps_prior_moves_and_marks_related_steps(tmp_path, monkeypatch):
    initial = opening_position()
    red = relocated(initial, [('red_rook_0', 0, 8)])
    black = relocated(red, [('black_rook_0', 0, 1)])
    undo_red = relocated(black, [('red_rook_0', 0, 9)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(red, black, undo_red, initial), end=5)
    assert len(result.moves) >= 3
    assert result.moves[0].move.target == Square(file=0, rank=8)
    assert result.moves[1].move.target == Square(file=0, rank=1)
    assert result.moves[0].violation == result.moves[1].violation == 'takeback'
    assert any(r.violation == 'takeback' and r.status == 'needs_review' for r in result.moves[2:])
    assert all(r.position_after is None for r in result.moves[2:])


def test_manual_correction_accepts_diagonal_or_multi_piece_snapshot(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create('review.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task = task.model_copy(update={'rules_enabled': True})
    task.replay_mode, task.initial_position = 'actual', opening_position()
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review')]
    diagonal = relocated(task.initial_position, [('red_rook_0', 1, 8)])
    corrected = correct_actual(task, 1, [], 1, diagonal)
    assert corrected.moves[0].status == 'corrected'
    assert corrected.moves[0].violation_note
    two = relocated(task.initial_position, [('red_rook_0', 0, 8), ('red_pawn_4', 4, 5)])
    corrected = correct_actual(task, 1, [], 1, two)
    assert corrected.moves[0].status == 'corrected'
    assert {p.id: p for p in corrected.moves[0].position_after.pieces} == {p.id: p for p in two.pieces}


def test_manual_choice_revalidates_following_legal_reply(tmp_path):
    initial = opening_position()
    red = relocated(initial, [('red_rook_0', 0, 8)])
    black = relocated(red, [('black_rook_0', 0, 1)])
    task = TaskStore(tmp_path).create('review.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task = task.model_copy(update={'rules_enabled': True})
    task.replay_mode, task.initial_position = 'actual', initial
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review'),
                  MoveRecord(index=2, video_start_s=1, video_end_s=2, status='tentative', observed_positions=[black])]
    action = Move(index=1, color='red', piece_id='red_rook_0', source=Square(file=0, rank=9), target=Square(file=0, rank=8))
    corrected = correct_actual(task, 1, [action], 1)
    assert corrected.status == 'ready'
    assert corrected.confirmed_prefix_length == 2
    assert corrected.moves[1].position_after.side_to_move == 'red'


def test_manual_no_move_keeps_takeback_warning_and_original_steps(tmp_path):
    initial = opening_position()
    red = relocated(initial, [('red_rook_0', 0, 8)])
    task = TaskStore(tmp_path).create('review.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task = task.model_copy(update={'rules_enabled': True})
    task.replay_mode, task.initial_position = 'actual', initial
    task.moves = [snapshot_record(initial, red, 1, 0, 1),
                  MoveRecord(index=2, video_start_s=1, video_end_s=2, status='needs_review')]
    task.moves[1] = task.moves[1].model_copy(update={'violation': 'takeback', 'violation_note': '悔棋'})
    corrected = correct_actual(task, 2, [], 2)
    assert len(corrected.moves) == 2
    assert corrected.moves[0].move.target == Square(file=0, rank=8)
    assert corrected.moves[1].violation == 'takeback'
    assert corrected.moves[1].position_after.pieces == red.pieces


def test_new_uploaded_task_saves_illegal_manual_snapshot_with_warning(tmp_path, video_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        uploaded = client.post('/api/tasks', files={'file': ('clip.mp4', video_path.read_bytes(), 'video/mp4')}).json()
        task = app.state.store.get(uploaded['id'])
        task.initial_position = opening_position()
        task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review')]
        task = app.state.store.save(task, task.revision)
        after = relocated(task.initial_position, [('red_rook_0', 1, 8)])
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'position': after.model_dump(), 'video_time_s': 1})
        assert response.status_code == 200
        assert app.state.store.get(task.id).moves[0].status == 'corrected'
        assert response.json()['moves'][0]['violation_note']


def test_legal_retreat_without_board_rollback_is_not_a_takeback(tmp_path, monkeypatch):
    initial = opening_position()
    red = relocated(initial, [('red_rook_0', 0, 8)])
    black = relocated(red, [('black_rook_0', 0, 1)])
    retreat = relocated(black, [('red_rook_0', 0, 9)])
    result = run_frames(tmp_path, monkeypatch, stable_frames(red, black, retreat))
    assert result.status == 'ready'
    assert len(result.moves) == 3
    assert all(r.violation is None for r in result.moves)


def test_removing_capture_victim_does_not_confirm_opponent_turn(tmp_path, monkeypatch):
    initial = opening_position()
    retained = {'red_king_4', 'black_king_4', 'red_pawn_4', 'red_rook_0', 'black_pawn_0'}
    initial.pieces = [p for p in initial.pieces if p.id in retained]
    initial = relocated(initial, [('black_pawn_0', 0, 5)])
    slide = relocated(initial, [('red_rook_0', 0, 8)])
    lifted = slide.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'black_pawn_0']
    captured = relocated(lifted, [('red_rook_0', 0, 5)])
    frames = [(0, board_image(initial)), (1, board_image(slide)), (1.1, board_image(slide)),
              (1.2, board_image(slide)), (2, board_image(lifted)), (2.2, board_image(lifted)),
              (3, board_image(captured)), (3.1, board_image(captured)), (3.2, board_image(captured))]
    result = run_frames(tmp_path, monkeypatch, frames, initial=initial)
    assert result.status == 'ready'
    assert len(result.moves) == 1
    assert result.moves[0].move.source == Square(file=0, rank=9)
    assert result.moves[0].move.target == Square(file=0, rank=5)
    assert result.moves[0].move.captured_piece_id == 'black_pawn_0'


def test_same_turn_occlusion_is_resolved_by_later_clear_landing(tmp_path):
    from app.turns import TurnReplay
    task = TaskStore(tmp_path).create('slide.mp4', VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'))
    task.rules_enabled, task.initial_position = True, opening_position()
    replay = TurnReplay(task, task.initial_position)
    replay.observe(relocated(task.initial_position, [('red_rook_0', 0, 8)]), 1, [])
    replay.uncertain(1.5, [], [], '结合前后画面仍无法确定完整局面')
    replay.observe(relocated(task.initial_position, [('red_rook_0', 0, 7)]), 2, [])
    replay.end(4)
    assert len(task.moves) == 1
    assert task.moves[0].status == 'auto_verified'
    assert task.moves[0].move.source == Square(file=0, rank=9)
    assert task.moves[0].move.target == Square(file=0, rank=7)


def test_failure_keeps_unconfirmed_turn_for_review(tmp_path, monkeypatch):
    import app.actual as actual
    after = relocated(opening_position(), [('red_rook_0', 0, 8)])
    def failing_frames(*args):
        yield from stable_frames(after)
        raise ValueError('模拟视频后段解码失败')
    monkeypatch.setattr(actual, 'iter_frames', failing_frames)
    store = TaskStore(tmp_path)
    task = store.create('failed.mp4', VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', opening_position()
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                          {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    analyze_actual(task, store, None)
    saved = store.get(task.id)
    assert saved.status == 'failed'
    assert len(saved.moves) == 1
    assert saved.moves[0].status == 'needs_review'
    assert saved.moves[0].position_after is None
    assert saved.moves[0].observed_positions


def test_long_capture_lift_does_not_split_same_turn(tmp_path, monkeypatch):
    initial = opening_position()
    retained = {'red_king_4', 'black_king_4', 'red_pawn_4', 'red_rook_0', 'black_pawn_0'}
    initial.pieces = [p for p in initial.pieces if p.id in retained]
    initial = relocated(initial, [('black_pawn_0', 0, 5)])
    slide = relocated(initial, [('red_rook_0', 0, 8)])
    lifted = slide.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'black_pawn_0']
    captured = relocated(lifted, [('red_rook_0', 0, 5)])
    frames = [(0, board_image(initial)), (1, board_image(slide)), (1.1, board_image(slide)),
              (1.2, board_image(slide)), (2, board_image(lifted)), (3.2, board_image(lifted)),
              (4, board_image(captured)), (4.1, board_image(captured)), (4.2, board_image(captured))]
    result = run_frames(tmp_path, monkeypatch, frames, end=5, initial=initial)
    assert result.status == 'ready'
    assert len(result.moves) == 1
    assert result.moves[0].move.captured_piece_id == 'black_pawn_0'


def test_crash_recovery_retains_pending_turn_as_review(tmp_path):
    from app.turns import TurnReplay
    store = TaskStore(tmp_path)
    task = store.create('crash.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', opening_position()
    task.status, task.processed_video_s = 'analyzing', 1.2
    replay = TurnReplay(task, task.initial_position)
    replay.observe(relocated(task.initial_position, [('red_rook_0', 0, 8)]), 1, ['frame.jpg'])
    replay.sync_pending()
    store.save(task, task.revision)
    store.recover_interrupted()
    saved = store.get(task.id)
    assert saved.status == 'interrupted'
    assert len(saved.moves) == 1
    assert saved.moves[0].status == 'needs_review'
    assert saved.moves[0].position_after is None
    assert saved.moves[0].evidence_ids == ['frame.jpg']


def test_auto_initial_turn_comes_from_first_observed_side(tmp_path):
    from app.turns import TurnReplay
    initial = opening_position()
    task = TaskStore(tmp_path).create('black-first.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task = task.model_copy(update={'initial_turn_known': False})
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    replay = TurnReplay(task, initial)
    replay.observe(relocated(initial, [('black_rook_0', 0, 1)]), 1, [])
    replay.end(3)
    assert task.initial_position.side_to_move == 'black'
    assert task.moves[0].status == 'auto_verified'
    assert task.moves[0].position_after.side_to_move == 'red'


def test_explicit_initial_turn_is_not_overridden(tmp_path, monkeypatch):
    black = relocated(opening_position(), [('black_rook_0', 0, 1)])
    saved = run_frames(tmp_path, monkeypatch, stable_frames(black), end=2)
    assert saved.moves[0].violation == 'turn_mismatch'
    assert saved.moves[0].status == 'needs_review'
