import cv2
import numpy as np
from app.chess import apply_move, legal_moves
from app.schemas import opening_position


def board_image(position):
    image = np.full((1001, 901, 3), (100, 175, 200), np.uint8)
    for piece in position.pieces:
        x, y = 50 + piece.square.file * 100, 50 + piece.square.rank * 100
        cv2.circle(image, (x, y), 43, (220, 220, 220), -1)
        cv2.putText(image, 'P', (x - 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (50, 30, 180) if piece.color == 'red' else (30, 30, 30), 5)
    return image


def test_tracking_preserves_fast_moves_while_unrelated_hand_keeps_frame_changing():
    from app.analysis import tracked_events
    initial = opening_position()
    first = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    middle = apply_move(initial, first)
    second = next(m for m in legal_moves(middle) if m.piece_id == 'black_horse_1' and m.target.file == 2)
    final = apply_move(middle, second)
    frames = [(0, board_image(initial))]
    for i, position in enumerate([middle, middle, middle, final, final, final]):
        image = board_image(position)
        # unrelated moving hand must not hide the move's source/target evidence.
        cv2.rectangle(image, (350 + i, 360), (530 + i, 430), (90, 130, 190), -1)
        frames.append((.08 + i * .04, image))
    events = list(tracked_events(iter(frames), initial))
    assert [e.position for e in events] == [initial, middle, final]


def test_hidden_landing_cannot_be_automatically_confirmed():
    from app.analysis import tracked_events
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    after = apply_move(initial, move)
    hidden = board_image(after)
    cv2.rectangle(hidden, (400, 700), (500, 800), (90, 130, 190), -1)
    events = list(tracked_events(iter([(0, board_image(initial)), (.1, hidden), (.2, hidden), (.3, hidden)]), initial))
    assert [e.position for e in events if e.position is not None] == [initial]


def test_missing_visible_piece_is_not_filled_by_partial_matching():
    from app.analysis import tracked_events
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    after = apply_move(initial, move)
    after.pieces = [p for p in after.pieces if p.id != 'red_rook_8']
    events = list(tracked_events(iter([(0, board_image(initial)), (.1, board_image(after)), (1.2, board_image(after))]), initial))
    assert all(e.position == initial for e in events if e.position is not None)
    assert events[-1].position is None


def test_short_move_and_return_retains_uncertain_evidence():
    from app.analysis import tracked_events
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    after = apply_move(initial, move)
    events = list(tracked_events(iter([(0, board_image(initial)), (.1, board_image(after)),
        (.13, board_image(after)), (.15, board_image(initial)), (.4, board_image(initial))]), initial))
    assert any(e.uncertain and e.position is None and len(e.images) == 3 for e in events)


def test_unresolved_occlusion_at_end_retains_review_evidence():
    from app.analysis import tracked_events
    initial = opening_position()
    hidden = board_image(initial)
    cv2.rectangle(hidden, (400, 700), (500, 800), (90, 130, 190), -1)
    events = list(tracked_events(iter([(0, board_image(initial)), (.1, hidden), (.2, hidden)]), initial))
    assert events[-1].uncertain and events[-1].position is None
    assert events[-1].review_reason


def test_later_different_move_does_not_erase_prior_round_trip():
    from app.analysis import tracked_events
    current = opening_position()
    frames = [(0, board_image(current))]
    for time, piece_id, file, rank in [(.1, 'red_rook_0', 0, 8), (.13, 'black_rook_0', 0, 1),
            (.16, 'red_rook_0', 0, 9), (.19, 'black_rook_0', 0, 0), (.22, 'red_pawn_4', 4, 5)]:
        move = next(m for m in legal_moves(current) if m.piece_id == piece_id and m.target.file == file and m.target.rank == rank)
        current = apply_move(current, move)
        frames.append((time, board_image(current)))
    frames.append((.3, board_image(current)))
    events = list(tracked_events(iter(frames), opening_position()))
    assert any(e.uncertain for e in events)


def test_pending_move_then_hidden_eof_is_reviewed_without_model():
    from app.analysis import tracked_events
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    after = apply_move(initial, move)
    hidden = board_image(after)
    for x in (150, 450):
        cv2.rectangle(hidden, (x-45, 705), (x+45, 795), (90, 130, 190), -1)
    events = list(tracked_events(iter([(0, board_image(initial)), (.1, board_image(after)), (.13, hidden)]), initial))
    assert events[-1].uncertain and events[-1].review_reason


def test_model_confirmation_does_not_append_old_buffered_events():
    from app.analysis import tracked_events
    initial = opening_position()
    first = next(m for m in legal_moves(initial) if m.piece_id == 'red_cannon_1' and m.target.file == 4)
    middle = apply_move(initial, first)
    second = next(m for m in legal_moves(middle) if m.piece_id == 'black_horse_1' and m.target.file == 2)
    after = apply_move(middle, second)
    confirmed = [initial]
    frames = [(0, board_image(initial)), (.1, board_image(middle)), (.13, board_image(after)),
        (.2, board_image(after)), (1.3, board_image(after)), (1.4, board_image(after)), (1.7, board_image(after))]
    events = []
    for event in tracked_events(iter(frames), initial, lambda: confirmed[0]):
        events.append(event)
        if event.position is None:
            confirmed[0] = after
    assert [e.time_s for e in events] == sorted(e.time_s for e in events)
    assert len(events) == 2


def test_local_only_pipeline_keeps_unresolved_round_trip(tmp_path, monkeypatch):
    import app.analysis as analysis
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    initial = opening_position()
    current = initial
    frames = [(0, board_image(current))]
    for time, piece_id, file, rank in [(.1, 'red_rook_0', 0, 8), (.13, 'black_rook_0', 0, 1),
            (.16, 'red_rook_0', 0, 9), (.19, 'black_rook_0', 0, 0), (.22, 'red_pawn_4', 4, 5)]:
        move = next(m for m in legal_moves(current) if m.piece_id == piece_id and m.target.file == file and m.target.rank == rank)
        current = apply_move(current, move)
        frames.append((time, board_image(current)))
    frames.append((.3, board_image(current)))
    monkeypatch.setattr(analysis, 'iter_frames', lambda *args: iter(frames))
    store = TaskStore(tmp_path)
    task = store.create('round-trip.mp4', VideoMetadata(width=901, height=1001, duration_s=.4, codec='h264'))
    task.initial_position = initial
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.status = 'analyzing'
    (store.task_dir(task.id) / 'evidence').mkdir()
    store.save(task, 0)
    analysis.analyze_task(task.id, store, None)
    saved = store.get(task.id)
    assert saved.confirmed_prefix_length == 0
    assert saved.moves[0].status == 'needs_review'
    assert saved.status == 'needs_review'
    assert saved.analysis_stats.model_requests == saved.analysis_stats.images_sent == 0
