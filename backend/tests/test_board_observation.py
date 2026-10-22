import cv2
import numpy as np

from app.schemas import Square, opening_position


def face_board(position, offset=(-6, -30), letters=True, hand=False):
    image = np.full((551, 501, 3), (95, 170, 205), np.uint8)
    for x in range(50, 451, 50):
        cv2.line(image, (x, 50), (x, 500), (60, 70, 80), 1)
    for y in range(50, 501, 50):
        cv2.line(image, (50, y), (450, y), (60, 70, 80), 1)
    for piece in position.pieces:
        x = 50 + piece.square.file * 50 + offset[0]
        y = 50 + piece.square.rank * 50 + offset[1]
        cv2.circle(image, (x, y), 18, (225, 225, 225), -1)
        cv2.circle(image, (x, y), 17, (70, 75, 80), 2)
        if letters:
            color = (40, 30, 185) if piece.color == 'red' else (30, 30, 30)
            cv2.putText(image, 'P', (x - 6, y + 7), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    if hand:
        cv2.rectangle(image, (120, 240), (170, 295), (90, 130, 190), -1)
    return image


def unknown_occupied(position):
    occupied = {(p.square.file, p.square.rank) for p in position.pieces}
    return {(f, r): None for f in range(9) for r in range(10) if (f, r) not in occupied}


def test_initial_known_layout_corrects_piece_face_offset_larger_than_half_a_cell():
    from app.board_observation import BoardObservation
    known = opening_position()
    observation = BoardObservation()
    cells = observation.update(face_board(known), known, unknown_occupied(known))
    assert cells[(2, 6)] == 'red'
    assert cells[(8, 0)] == 'black'
    assert (2, 5) not in cells or cells[(2, 5)] is None


def test_shifted_landing_corrects_false_neighbor_color_without_reading_piece_kind():
    from app.board_observation import BoardObservation
    known = opening_position()
    observation = BoardObservation()
    observation.update(face_board(known), known, unknown_occupied(known))
    after = known.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_pawn_2').square = Square(file=2, rank=5)
    raw = unknown_occupied(after)
    raw[(2, 4)] = 'red'
    cells = observation.update(face_board(after), known, raw)
    assert cells[(2, 5)] == 'red'
    assert cells[(2, 4)] is None
    assert cells[(2, 6)] is None


def test_hand_over_a_face_is_unknown_even_if_legacy_sampling_says_empty():
    from app.board_observation import BoardObservation
    known = opening_position()
    observation = BoardObservation()
    observation.update(face_board(known), known, unknown_occupied(known))
    after = known.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_pawn_2').square = Square(file=2, rank=5)
    raw = unknown_occupied(after)
    raw[2, 5] = None
    cells = observation.update(face_board(after, hand=True), known, raw)
    assert (2, 5) not in cells


def test_piece_tracker_uses_full_margin_and_actual_face_centers_for_known_identities():
    from app.piece_tracking import PieceTracker
    known = opening_position()
    tracker = PieceTracker()
    for i in range(6):
        proof = tracker.observe(i * .04, face_board(known, letters=False), known, unknown_occupied(known))
    assert proof[(8, 0)] == 'black'
    assert proof[(2, 6)] == 'red'
    assert tracker.identities[(8, 0)] == 'black_rook_8'
    assert tracker.identities[(2, 6)] == 'red_pawn_2'
    assert len(tracker.identities) >= 30


def test_reseed_uses_last_observed_identity_anchor_instead_of_a_predicted_board():
    from app.piece_tracking import PieceTracker
    known = opening_position()
    tracker = PieceTracker()
    raw = unknown_occupied(known)
    for i in range(5):
        tracker.observe(i * .04, face_board(known, letters=False), known, raw)
    guessed = known.model_copy(deep=True)
    next(p for p in guessed.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=5)
    for i in range(5, 11):
        proof = tracker.observe(i * .04, face_board(known, letters=False), guessed, raw, allow_reseed=False)
    assert proof[(0, 9)] == 'red'
    assert (0, 5) not in tracker.identities


def test_geometric_comparison_is_retained_for_a_live_before_image_after_cache_eviction(monkeypatch):
    from app import actual
    from collections import OrderedDict
    import weakref
    image = face_board(opening_position())
    image.setflags(write=False)
    monkeypatch.setattr(actual, '_read_tracking_cells', lambda *args: {(2, 4): 'red', (2, 5): None})
    token = actual._tracking_cache.set(OrderedDict())
    try:
        actual._tracking_cells(image)
        actual._tracking_cache.get()[id(image), 'identity'] = (weakref.ref(image), {}, {}, {},
                                                             {(2, 4): None, (2, 5): 'red'})
        assert actual._tracking_cells(image) == {(2, 4): None, (2, 5): 'red'}
        for _ in range(100):
            following = image.copy()
            following.setflags(write=False)
            actual._tracking_cells(following)
        assert actual._tracking_cells(image) == {(2, 4): None, (2, 5): 'red'}
    finally:
        actual._tracking_cache.reset(token)


def test_projection_learning_does_not_use_an_unconfirmed_later_board_as_its_anchor():
    from app.board_observation import BoardObservation
    known = opening_position()
    observation = BoardObservation()
    blank = np.full((551, 501, 3), (95, 170, 205), np.uint8)
    observation.update(blank, known, {})
    guessed = known.model_copy(deep=True)
    for piece in guessed.pieces:
        piece.square.rank = (piece.square.rank + 1) % 10
    cells = observation.update(face_board(known), guessed, {})
    assert cells[(8, 0)] == 'black'
    assert cells[(2, 6)] == 'red'


def test_one_located_disc_does_not_remain_counted_as_a_second_neighbor_piece(monkeypatch):
    import app.board_observation as module
    monkeypatch.setattr(module, 'wood_surface_mask', lambda hsv: np.zeros(hsv.shape[:2], dtype=bool))
    known = opening_position()
    observation = module.BoardObservation()
    observation.update(face_board(known), known, {})
    after = known.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_pawn_2').square = Square(file=2, rank=5)
    raw = unknown_occupied(after)
    raw[2, 4] = 'red'
    corrected = observation.update(face_board(after), known, raw)
    assert corrected[(2, 5)] == 'red'
    assert (2, 4) not in corrected


def test_unseen_identity_does_not_bind_to_an_opponent_on_its_old_square():
    from app.piece_tracking import PieceTracker
    known = opening_position()
    tracker = PieceTracker()
    hidden = face_board(known)
    cv2.rectangle(hidden, (120, 320), (170, 365), (90, 130, 190), -1)
    for i in range(5):
        tracker.observe(i * .04, hidden, known, unknown_occupied(known))
    assert 'red_pawn_2' not in tracker.tracks
    opponent = known.model_copy(deep=True)
    next(p for p in opponent.pieces if p.id == 'red_pawn_2').color = 'black'
    raw = unknown_occupied(known)
    raw[2, 6] = 'black'
    for i in range(5, 11):
        proof = tracker.observe(i * .04, face_board(opponent), known, raw, allow_reseed=False)
    assert (2, 6) not in proof
    assert (2, 6) not in tracker.identities


def test_larger_piece_disc_at_a_new_rank_does_not_leave_a_ghost_on_the_old_rank():
    from app.board_observation import BoardObservation
    known = opening_position()
    observation = BoardObservation()
    observation.update(face_board(known), known, {})
    after = known.model_copy(deep=True)
    after.pieces = [p for p in after.pieces if p.id != 'black_pawn_6']
    image = face_board(after)
    cv2.circle(image, (344, 220), 28, (225, 225, 225), -1)
    cv2.circle(image, (344, 220), 27, (70, 75, 80), 2)
    raw = unknown_occupied(known)
    raw[6, 3] = 'black'
    cells = observation.update(image, known, raw)
    assert cells[(6, 4)] == 'black'
    assert cells[(6, 3)] is None


def test_missing_disc_detection_cannot_restore_legacy_occupancy_at_the_wrong_grid_point(monkeypatch):
    import app.board_observation as module
    known = opening_position()
    observation = module.BoardObservation()
    observation.update(face_board(known), known, {})
    monkeypatch.setattr(observation, '_circles', lambda *args: np.empty((0,3),np.float32))
    monkeypatch.setattr(module, 'wood_surface_mask', lambda hsv: np.zeros(hsv.shape[:2],dtype=bool))
    after = known.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'black_pawn_6').square = Square(file=6,rank=4)
    cells = observation.update(face_board(after), known, {(6,3):'black'})
    assert (6,3) not in cells
