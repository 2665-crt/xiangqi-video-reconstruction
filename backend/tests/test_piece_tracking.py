import cv2
import numpy as np
import pytest

from app.schemas import Piece, Position, Square


def position():
    return Position(pieces=[
        Piece(id='known_rook', color='red', kind='rook', square=Square(file=1, rank=7)),
        Piece(id='known_horse', color='red', kind='horse', square=Square(file=6, rank=8)),
        Piece(id='opponent', color='black', kind='cannon', square=Square(file=7, rank=2)),
    ], side_to_move='red')


def blank_pieces(centers, hidden=False):
    image = np.full((500, 450, 3), (95, 170, 205), np.uint8)
    for x in range(25, 450, 50):
        cv2.line(image, (x, 25), (x, 475), (60, 70, 80), 1)
    for y in range(25, 500, 50):
        cv2.line(image, (25, y), (425, y), (60, 70, 80), 1)
    for x, y in centers:
        cv2.circle(image, (round(x), round(y)), 21, (225, 225, 225), -1)
        cv2.circle(image, (round(x), round(y)), 20, (70, 75, 80), 2)
    if hidden:
        cv2.rectangle(image, (35, 335), (265, 420), (90, 130, 190), -1)
    return image


def unknown_pieces():
    cells = {(f, r): None for f in range(9) for r in range(10)}
    for square in [(1, 7), (6, 8), (7, 2)]:
        cells.pop(square)
    return cells


def test_known_identity_tracks_unlettered_disc_to_new_square():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    known = position()
    static = [(325, 425), (375, 125)]
    proof = {}
    for i, x in enumerate([75, 75, 75] + list(range(85, 226, 10)) + [225, 225, 225]):
        cells = unknown_pieces()
        cells.pop((4, 7))
        if x > 100:
            cells[1, 7] = None
        proof = tracker.observe(i * .04, blank_pieces([(x, 375)] + static), known, cells)
    assert proof[(4, 7)] == 'red'
    assert tracker.identities[(4, 7)] == 'known_rook'
    assert tracker.identities[(6, 8)] == 'known_horse'
    assert (1, 7) not in proof
    assert known.pieces[0].square == Square(file=1, rank=7)


def test_stationary_blank_pieces_do_not_need_recognizable_letters():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    for i in range(4):
        proof = tracker.observe(i * .04, blank_pieces([(75, 375), (325, 425), (375, 125)]),
                                position(), unknown_pieces())
    assert proof == {(1, 7): 'red', (6, 8): 'red', (7, 2): 'black'}


def test_occluded_identity_is_not_reassigned_to_disc_reappearing_elsewhere():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    for i, (x, hidden) in enumerate([(75, False), (75, False), (85, False),
                                    (100, True), (150, True), (225, False), (225, False), (225, False)]):
        cells = unknown_pieces()
        cells.pop((4, 7))
        if x > 100:
            cells[1, 7] = None
        proof = tracker.observe(i * .04, blank_pieces([(x, 375), (325, 425), (375, 125)], hidden), position(), cells)
    assert (4, 7) not in proof
    assert (4, 7) not in tracker.identities


def test_long_frame_gap_does_not_count_as_a_continuous_trajectory():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    for t, x in [(0, 75), (.04, 75), (.08, 85), (4, 225), (4.04, 225), (4.08, 225)]:
        cells = unknown_pieces()
        cells.pop((4, 7))
        if x > 100:
            cells[1, 7] = None
        proof = tracker.observe(t, blank_pieces([(x, 375), (325, 425), (375, 125)]), position(), cells)
    assert (4, 7) not in proof


def test_clear_empty_square_is_never_filled_from_remembered_identity():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    for i in range(4):
        cells = unknown_pieces()
        cells[1, 7] = None
        proof = tracker.observe(i * .04, blank_pieces([(325, 425), (375, 125)]), position(), cells)
    assert (1, 7) not in proof
    assert (1, 7) not in tracker.identities


def test_same_color_tracks_that_overlap_cannot_supply_an_identity():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    known = position()
    for i in range(28):
        delta = min(125, max(0, i - 2) * 5)
        centers = [(75 + delta, 375), (325 - delta, 425 - delta * .4), (375, 125)]
        cells = unknown_pieces()
        cells.pop((3, 7))
        if delta > 25:
            cells[1, 7] = None
            cells[6, 8] = None
        proof = tracker.observe(i * .04, blank_pieces(centers), known, cells)
    assert (3, 7) not in proof
    assert (3, 7) not in tracker.identities


def test_unknown_cells_only_supplement_visible_tracking_evidence():
    from app import actual
    from collections import OrderedDict
    import weakref
    image = blank_pieces([(75, 375), (325, 425), (375, 125)])
    image.setflags(write=False)
    token = actual._tracking_cache.set(OrderedDict())
    try:
        base = actual._tracking_cells(image)
        cache = actual._tracking_cache.get()
        cache[id(image), 'identity'] = weakref.ref(image), {(1, 7): 'red', (0, 0): 'black'}
        result = actual._tracking_cells(image)
        assert result[(1, 7)] == 'red'
        assert result[(0, 0)] is None
        assert (1, 7) not in base
    finally:
        actual._tracking_cache.reset(token)


def test_blank_capture_preserves_mover_without_reusing_captured_identity():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    known = position()
    known.pieces[-1].square = Square(file=4, rank=7)
    for i, x in enumerate([75, 75, 75] + list(range(85, 226, 10)) + [225] * 4):
        centers = [(x, 375), (325, 425)] + ([(225, 375)] if i < 4 else [])
        cells = unknown_pieces()
        cells.pop((4, 7), None)
        if 4 <= i < 15:
            cells[4, 7] = None
        if x > 100:
            cells[1, 7] = None
        proof = tracker.observe(i * .04, blank_pieces(centers), known, cells)
    assert proof[(4, 7)] == 'red'
    assert tracker.identities[(4, 7)] == 'known_rook'


def test_slow_continuous_slide_does_not_confirm_intermediate_squares():
    from app.piece_tracking import PieceTracker
    tracker = PieceTracker()
    intermediate = []
    final = {}
    for i, x in enumerate([75] * 4 + list(range(76, 226)) + [225] * 5):
        cells = unknown_pieces()
        for square in [(2, 7), (3, 7), (4, 7)]:
            cells.pop(square)
        if x > 100:
            cells[1, 7] = None
        proof = tracker.observe(i * .04, blank_pieces([(x, 375), (325, 425), (375, 125)]), position(), cells)
        intermediate.extend(square for square in proof if square in [(2, 7), (3, 7)])
        final = proof
    assert intermediate == []
    assert final[(4, 7)] == 'red'


def test_analysis_confirms_unlettered_move_and_preserves_kind_and_timing(tmp_path, monkeypatch):
    from app import actual
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    known = position()
    frames = [(i * .04, blank_pieces([(x, 375), (325, 425), (375, 125)]))
              for i, x in enumerate([75] * 4 + list(range(85, 226, 10)) + [225] * 8)]
    monkeypatch.setattr(actual, 'iter_frames',
                        lambda path, start, end: iter((t, frame) for t, frame in frames if start <= t <= end))
    store = TaskStore(tmp_path)
    task = store.create('unlettered.mp4', VideoMetadata(width=450, height=500, duration_s=1.08, codec='h264'))
    task.initial_position = known
    task.initial_frame_s, task.end_s = 0, 1.04
    task.calibration_confirmed = True
    task.calibration = Calibration(corners=[{'x': 25, 'y': 25}, {'x': 425, 'y': 25},
                                            {'x': 425, 'y': 475}, {'x': 25, 'y': 475}])
    task.rules_enabled, task.replay_mode, task.status = False, 'actual', 'analyzing'
    task = store.save(task, task.revision)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.error is None
    moves = [record for record in result.moves if record.move is not None]
    assert len(moves) == 1
    record = moves[0]
    assert record.status == 'auto_verified'
    assert record.move.piece_id == 'known_rook'
    assert record.move.source == Square(file=1, rank=7)
    assert record.move.target == Square(file=4, rank=7)
    moved = next(p for p in record.position_after.pieces if p.id == 'known_rook')
    assert moved.kind == 'rook'
    assert record.timing_status == 'observed'
    assert .12 <= record.action_start_s <= .2
    assert .72 <= record.action_end_s <= .84
    assert result.run_stats.model_requests == 0


@pytest.mark.parametrize('interrupted', [False, True])
def test_letter_fading_only_requests_reclassification_after_a_broken_identity_track(monkeypatch, interrupted):
    from app import actual
    from app.piece_tracking import PieceTracker
    from collections import OrderedDict
    import weakref
    tracker = PieceTracker()
    known = position()
    monkeypatch.setattr(actual, '_base_tracking_cells', lambda *args: unknown_pieces())
    token = actual._tracking_cache.set(OrderedDict())

    def frames():
        for i in range(75):
            image = blank_pieces([(75, 375), (325, 425), (375, 125)], hidden=interrupted and i == 10)
            shade = 30 if i < 10 else 225 if interrupted else round(30 + min(20, i - 10) * 195 / 20)
            if not (interrupted and i == 10):
                cv2.rectangle(image, (69, 369), (81, 381), (shade, shade, shade), -1)
            proof = tracker.observe(i * .04, image, known, unknown_pieces(), allow_reseed=interrupted or i < 4)
            image.setflags(write=False)
            actual._tracking_cache.get()[id(image), 'identity'] = (weakref.ref(image), proof,
                                                                  tracker.identities, tracker.tokens)
            yield i * .04, image

    try:
        events = list(actual.actual_events(frames(), known))
        requested = [event for event in events if event.position is None and not event.review_reason]
        assert bool(requested) is interrupted
    finally:
        actual._tracking_cache.reset(token)


def test_two_same_color_candidates_use_the_observed_identity():
    from app.actual import _cells_position
    from app.schemas import opening_position
    known = opening_position()
    observed = {(f, r): None for f in range(9) for r in range(10)}
    observed.update({(p.square.file, p.square.rank): p.color for p in known.pieces})
    observed[1, 7], observed[7, 7] = None, None
    observed[4, 7], observed[6, 7] = 'red', 'red'
    after = _cells_position(known, observed, [known], identities={(4, 7): 'red_cannon_7', (6, 7): 'red_cannon_1'})
    assert after is not None
    assert next(p for p in after.pieces if p.id == 'red_cannon_7').square == Square(file=4, rank=7)
    assert next(p for p in after.pieces if p.id == 'red_cannon_1').square == Square(file=6, rank=7)


def test_inference_cannot_substitute_another_same_color_piece_for_a_tracked_identity():
    from app.move_inference import MoveInference
    from app.schemas import opening_position
    known = opening_position()
    observed = {(f, r): None for f in range(9) for r in range(10)}
    observed.update({(p.square.file, p.square.rank): p.color for p in known.pieces})
    observed.pop((1, 7))
    observed.pop((7, 7))
    observed[4, 7] = 'red'
    records = MoveInference(known, 0).observe(1, observed, identities={(4, 7): 'red_cannon_7'})
    assert records and records[-1].move.piece_id == 'red_cannon_7'
    assert records[-1].status == 'tentative'


@pytest.mark.parametrize('indexed', [False, True])
def test_historical_color_layout_cannot_override_a_continuously_tracked_identity(indexed):
    from app.actual import _cells_position, PositionHistory
    from app.schemas import opening_position
    known = opening_position()
    previous = known.model_copy(deep=True)
    next(p for p in previous.pieces if p.id == 'red_cannon_1').square = Square(file=4, rank=7)
    next(p for p in previous.pieces if p.id == 'red_cannon_7').square = Square(file=6, rank=7)
    observed = {(f, r): None for f in range(9) for r in range(10)}
    observed.update({(p.square.file, p.square.rank): p.color for p in previous.pieces})
    history = PositionHistory([known, previous]) if indexed else [known, previous]
    assert _cells_position(known, observed, history, identities={(6, 7): 'red_cannon_1'}) is None


def test_stationary_identity_conflict_is_not_hidden_by_matching_colors():
    from app.actual import _cells_position
    from app.schemas import opening_position
    known = opening_position()
    observed = {(f, r): None for f in range(9) for r in range(10)}
    observed.update({(p.square.file, p.square.rank): p.color for p in known.pieces})
    observed[7, 7], observed[4, 7] = None, 'red'
    assert _cells_position(known, observed, [known], identities={(1, 7): 'red_cannon_7'}) is None


@pytest.mark.parametrize('orientation', ['normal', 'rotated_180'])
def test_full_board_without_letters_replays_sequential_moves_with_rules(tmp_path, monkeypatch, orientation):
    from app import actual
    from app.schemas import Calibration, VideoMetadata, opening_position
    from app.storage import TaskStore
    known = opening_position()
    visual = known.model_copy(deep=True)
    frames = []

    def append(mover=None, center=None):
        centers = [(25 + p.square.file * 50, 25 + p.square.rank * 50)
                   if p.id != mover else center for p in visual.pieces]
        image = blank_pieces(centers)
        if orientation == 'rotated_180':
            image = cv2.rotate(image, cv2.ROTATE_180)
        frames.append((len(frames) * .04, image))

    for _ in range(5):
        append()
    expected = [('red_pawn_0', (0, 5)), ('black_pawn_0', (0, 4)), ('red_cannon_1', (4, 7))]
    for identity, target in expected:
        piece = next(p for p in visual.pieces if p.id == identity)
        start = np.float32([25 + piece.square.file * 50, 25 + piece.square.rank * 50])
        end = np.float32([25 + target[0] * 50, 25 + target[1] * 50])
        count = int(np.linalg.norm(end - start) // 5)
        for offset in range(1, count + 1):
            append(identity, start + (end - start) * offset / count)
        piece.square = Square(file=target[0], rank=target[1])
        for _ in range(6):
            append()
    monkeypatch.setattr(actual, 'iter_frames', lambda path, start, end:
                        iter((t, image) for t, image in frames if start <= t <= end))
    store = TaskStore(tmp_path)
    task = store.create('blank-board.mp4', VideoMetadata(width=450, height=500,
                                                        duration_s=frames[-1][0] + .04, codec='h264'))
    task.initial_position, task.initial_frame_s = known, 0
    task.end_s, task.initial_turn_known = frames[-1][0], True
    task.calibration_confirmed = True
    task.calibration = Calibration(corners=[{'x': 25, 'y': 25}, {'x': 425, 'y': 25},
                                            {'x': 425, 'y': 475}, {'x': 25, 'y': 475}], orientation=orientation)
    task.rules_enabled, task.replay_mode, task.status = True, 'actual', 'analyzing'
    task = store.save(task, task.revision)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.error is None
    moves = [r for r in result.moves if r.move is not None]
    assert [(r.move.piece_id, (r.move.target.file, r.move.target.rank)) for r in moves] == expected
    assert all(r.status == 'auto_verified' and r.timing_status == 'observed' for r in moves)
    assert {p.id: p.kind for p in moves[-1].position_after.pieces} == {p.id: p.kind for p in known.pieces}
    assert result.run_stats.model_requests == 0
