import cv2
import pytest

from app.actual import _cells_position, actual_events
from app.chess import board_signature
from app.schemas import MoveRecord, Square, Task, VideoMetadata, opening_position
from app.turns import TurnReplay
from test_tracking import board_image


def test_visible_source_and_target_survive_extensive_unrelated_occlusion():
    before = opening_position()
    cells = {(f, r): None for f in range(4) for r in range(10)}
    for piece in before.pieces:
        square = piece.square.file, piece.square.rank
        if square in cells:
            cells[square] = piece.color
    cells[(0, 9)], cells[(0, 8)] = None, 'red'
    after = _cells_position(before, cells, [before])
    assert after is not None
    assert next(p.square for p in after.pieces if p.id == 'red_rook_0') == Square(file=0, rank=8)


def test_brief_partial_landing_is_preserved_before_next_move():
    before = opening_position()
    red = before.model_copy(deep=True)
    next(p for p in red.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    black = red.model_copy(deep=True)
    next(p for p in black.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=1)
    frames = []
    for time, position in [(0, before), (1, red), (1.04, red), (1.08, black), (1.2, black)]:
        image = board_image(position)
        cv2.rectangle(image, (600, 350), (900, 700), (20, 20, 20), -1)
        frames.append((time, image))
    positions = [e.position for e in actual_events(iter(frames), before, turn_boundaries=True) if e.position]
    assert board_signature(red) in [board_signature(p) for p in positions]


def test_unknown_observations_in_same_interval_preserve_both_evidence():
    task = Task(id='a' * 32, filename='clip.mp4', end_s=5,
                metadata=VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'),
                initial_position=opening_position())
    replay = TurnReplay(task, task.initial_position)
    replay.uncertain(1, [], ['first.jpg'], 'first action hidden')
    replay.uncertain(3, [], ['second.jpg'], 'second action hidden')
    replay.end(5)
    assert len(task.moves) == 1
    assert task.moves[0].evidence_ids == ['first.jpg', 'second.jpg']


def test_move_times_are_not_opponent_boundary_or_video_end():
    task = Task(id='b' * 32, filename='clip.mp4', end_s=10,
                metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                initial_position=opening_position(), rules_enabled=True)
    replay = TurnReplay(task, task.initial_position)
    after = task.initial_position.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    replay.observe(after, 2.4, ['landing.jpg'], 2.4)
    replay.end(10)
    record = task.moves[0]
    assert record.action_start_s == pytest.approx(2.4)
    assert record.action_end_s == pytest.approx(2.4)
    assert record.timing_status == 'observed'


def test_unverified_opponent_color_does_not_commit_a_sliding_piece():
    before = opening_position()
    task = Task(id='c' * 32, filename='clip.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    middle = before.model_copy(deep=True)
    next(p for p in middle.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    replay.observe(middle, 1, [])
    replay.movement(('black',), 1.1, False)
    final = middle.model_copy(deep=True)
    next(p for p in final.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=7)
    replay.observe(final, 1.5, [])
    replay.end(4)
    assert len(task.moves) == 1
    assert task.moves[0].move.source == Square(file=0, rank=9)
    assert task.moves[0].move.target == Square(file=0, rank=7)


def test_partial_unexplained_change_is_reported_before_video_end():
    initial = opening_position()
    after = initial.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    next(p for p in after.pieces if p.id == 'red_rook_8').square = Square(file=8, rank=8)
    frames = []
    for time, position in [(0, initial), (1, after), (1.2, after), (1.4, after)]:
        image = board_image(position)
        cv2.rectangle(image, (350, 300), (550, 600), (20, 20, 20), -1)
        frames.append((time, image))
    events = list(actual_events(iter(frames), initial, turn_boundaries=True))
    assert any(e.review_reason and e.time_s < 1.4 for e in events)


def test_occlusion_explained_by_one_observed_move_does_not_leave_false_missing_step():
    before = opening_position()
    task = Task(id='d' * 32, filename='clip.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.uncertain(1.1, [], ['hand.jpg'], '局部画面持续出现未解释的棋子变化，可能缺少中间走棋，请核对这段视频')
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    replay.observe(after, 1.4, ['landing.jpg'], 1)
    replay.end(4)
    assert len(task.moves) == 1
    assert task.moves[0].status == 'auto_verified'


def test_manual_correction_keeps_observed_action_times():
    from app.actual import snapshot_record
    from app.turns import correct_turn
    before = opening_position()
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    record = snapshot_record(before, after, 1, 0, 3)
    record.action_start_s, record.action_end_s, record.timing_status = 1, 1.4, 'observed'
    task = Task(id='e' * 32, filename='clip.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True, moves=[record])
    corrected = correct_turn(task, 1, [record.move], 1.4)
    assert corrected.moves[0].action_start_s == 1
    assert corrected.moves[0].action_end_s == 1.4


def test_single_partial_landing_before_reply_is_not_lost():
    before = opening_position()
    red = before.model_copy(deep=True)
    next(p for p in red.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    black = red.model_copy(deep=True)
    next(p for p in black.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=1)
    frames = []
    for time, position in [(0, before), (1, red), (1.04, black), (1.08, black), (1.12, black)]:
        image = board_image(position)
        cv2.rectangle(image, (600, 350), (900, 700), (20, 20, 20), -1)
        frames.append((time, image))
    events = list(actual_events(iter(frames), before, turn_boundaries=True))
    assert any(e.position and board_signature(e.position) == board_signature(red) for e in events)


def test_seen_landing_survives_source_and_target_occlusion_then_return():
    before = opening_position()
    red = before.model_copy(deep=True)
    next(p for p in red.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    hidden = board_image(red)
    cv2.rectangle(hidden, (0, 750), (100, 1000), (20, 20, 20), -1)
    frames = [(0, board_image(before)), (1, board_image(red)), (1.04, hidden),
              (1.08, board_image(before)), (1.2, board_image(before))]
    events = list(actual_events(iter(frames), before, turn_boundaries=True))
    assert any(e.position and board_signature(e.position) == board_signature(red) for e in events)


def test_interruption_keeps_observed_timing():
    before = opening_position()
    task = Task(id='f' * 32, filename='clip.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    replay.observe(after, 2.4, [], 1.2)
    replay.interrupt()
    assert (task.moves[0].action_start_s, task.moves[0].action_end_s, task.moves[0].timing_status) == (1.2, 2.4, 'observed')


def test_unknown_order_is_not_timestamped_as_observed():
    before = opening_position()
    task = Task(id='1' * 32, filename='clip.mp4', end_s=10,
                metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                initial_position=before, initial_turn_known=False, rules_enabled=True)
    replay = TurnReplay(task, before)
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    next(p for p in after.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=1)
    replay.observe(after, 4, [])
    replay.end(10)
    assert task.moves[0].timing_status == 'uncertain'
    assert task.moves[0].action_start_s is None


def test_brief_capture_and_slide_reset_glyph_comparison_size():
    before = opening_position()
    next(p for p in before.pieces if p.id == 'black_pawn_0').square = Square(file=0, rank=8)
    captured = before.model_copy(deep=True)
    captured.pieces = [p for p in captured.pieces if p.id != 'black_pawn_0']
    next(p for p in captured.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    slide = captured.model_copy(deep=True)
    next(p for p in slide.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=7)
    frames = [(0, board_image(before)), (1, board_image(captured)), (1.04, board_image(slide)),
              (1.08, board_image(captured)), (1.2, board_image(captured))]
    events = list(actual_events(iter(frames), before, turn_boundaries=True))
    assert any(e.position and len(e.position.pieces) == 31 for e in events)


def test_single_false_capture_contradicted_by_next_frame_does_not_poison_tracking():
    before = opening_position()
    next(p for p in before.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    phantom = before.model_copy(deep=True)
    phantom.pieces = [p for p in phantom.pieces if p.id != 'black_pawn_8']
    next(p for p in phantom.pieces if p.id == 'red_rook_0').square = Square(file=8, rank=3)
    actual = before.model_copy(deep=True)
    next(p for p in actual.pieces if p.id == 'red_rook_0').square = Square(file=3, rank=8)
    frames = [(0, board_image(before)), (1, board_image(phantom)), (1.04, board_image(actual)),
              (1.08, board_image(actual)), (1.12, board_image(actual))]
    events = list(actual_events(iter(frames), before, turn_boundaries=True))
    assert not any(e.position and not e.review_reason and board_signature(e.position) == board_signature(phantom) for e in events)
    assert any(e.position and board_signature(e.position) == board_signature(actual) for e in events)


def test_impossible_two_turn_order_does_not_crash_after_capture():
    from app.turns import split_observed_turns
    before = opening_position()
    next(p for p in before.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=8)
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    next(p for p in after.pieces if p.id == 'black_rook_0').square = Square(file=1, rank=8)
    assert split_observed_turns(before, after, 1) is None


def test_repeated_impossible_landing_does_not_replace_reference_before_real_landing():
    before = opening_position()
    phantom = before.model_copy(deep=True)
    phantom.pieces = [p for p in phantom.pieces if p.id != 'black_elephant_6']
    next(p for p in phantom.pieces if p.id == 'red_horse_1').square = Square(file=6, rank=0)
    actual = before.model_copy(deep=True)
    next(p for p in actual.pieces if p.id == 'red_horse_1').square = Square(file=2, rank=7)
    frames = [(0, board_image(before)), (1, board_image(phantom)), (1.05, board_image(phantom)),
              (1.25, board_image(phantom)), (1.3, board_image(actual)), (1.35, board_image(actual))]
    events = list(actual_events(iter(frames), before, turn_boundaries=True))
    assert not any(e.position and not e.review_reason and board_signature(e.position) == board_signature(phantom) for e in events)
    assert any(e.position and not e.review_reason and board_signature(e.position) == board_signature(actual) for e in events)


def test_snapshot_without_observed_onset_does_not_invent_start_time():
    before = opening_position()
    task = Task(id='2' * 32, filename='clip.mp4', end_s=10,
                metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    replay.observe(after, 8, ['late-snapshot.jpg'])
    replay.end(10)
    assert task.moves[0].timing_status == 'uncertain'
    assert task.moves[0].action_start_s is None
