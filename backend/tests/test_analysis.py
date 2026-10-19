import numpy as np
from test_chess import kings, position, move


def test_hand_motion_not_a_move():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    assert extract_moves(p, [(1, [p]), (2, [p])]) == []


def test_fast_two_moves_not_collapsed_and_identity_kept():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9), ("bp", "black", "pawn", 7, 5)])
    a = position(kings() + [("vision_new_id", "red", "rook", 0, 5), ("bp", "black", "pawn", 7, 5)], "black")
    b = position(kings() + [("another_id", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)], "red")
    result = extract_moves(p, [(1, [a]), (1.2, [b])])
    assert len(result) == 2
    assert result[0].move.piece_id == "r"
    assert result[1].move.piece_id == "bp"
    assert all(r.status == "auto_verified" for r in result)


def test_ambiguous_occlusion_stops_prefix():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    a = position(kings() + [("r", "red", "rook", 0, 5)], "black")
    b = position(kings() + [("r", "red", "rook", 0, 6)], "black")
    records = extract_moves(p, [(1, [a, b]), (2, [a, b])])
    assert records[0].status == "needs_review"
    assert len(records[0].candidates) == 2
    assert records[1].status == "tentative"


def test_following_frame_resolves_ambiguity():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    a = position(kings() + [("r", "red", "rook", 0, 5)], "black")
    b = position(kings() + [("r", "red", "rook", 0, 6)], "black")
    result = extract_moves(p, [(1, [a, b]), (2, [a])])
    assert len(result) == 1
    assert result[0].status == "auto_verified"
    assert result[0].move.target.rank == 5


def test_two_moves_under_occlusion_unique_recovery():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9), ("bp", "black", "pawn", 7, 5)])
    after = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)])
    result = extract_moves(p, [(2, [after])])
    assert len(result) == 2
    assert result[0].move.piece_id == "r"
    assert result[1].move.piece_id == "bp"


def test_zero_move_candidate_prevents_false_confirmation():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    after = position(kings() + [("r", "red", "rook", 0, 5)], "black")
    result = extract_moves(p, [(1, [p, after])])
    assert result[0].status == "needs_review"


def test_transient_roundtrip_is_retained_for_dense_review():
    from app.analysis import stable_events
    base = np.zeros((100, 90, 3), np.uint8)
    changed = base.copy(); changed[30:60, 20:50] = 255
    events = list(stable_events(iter([(0, base), (0.2, base), (0.23, changed), (0.27, base), (0.5, base)])))
    assert any(getattr(event, "uncertain", False) for event in events)


def test_blur_preserves_piece_identity_with_temporal_evidence():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    clear = position(kings() + [("unknown_id", "red", "rook", 0, 5)], "black")
    result = extract_moves(p, [(1, [clear])])
    assert result[0].move.piece_id == "r"
    assert result[0].status == "auto_verified"


def test_stable_events_ignore_transient_hand_and_keep_two_changes():
    from app.analysis import stable_events
    base = np.zeros((100, 90, 3), np.uint8)
    hand = base.copy(); hand[:, :50] = 255
    after1 = base.copy(); after1[30:40, 20:30] = 255
    after2 = after1.copy(); after2[60:70, 40:50] = 255
    frames = [(0, base), (0.12, base), (0.2, hand), (0.22, base),
              (0.35, after1), (0.48, after1), (0.60, after2), (0.73, after2)]
    events = list(stable_events(iter(frames), stable_seconds=0.1))
    assert len([event for event in events if not event.uncertain]) == 3
    assert any(event.uncertain for event in events)
    assert events[-1][0] == 0.73


def test_initial_turn_inferred_from_first_move():
    from app.analysis import resolve_initial
    p_red = position(kings() + [("bp", "black", "pawn", 7, 5)], "red")
    p_black = p_red.model_copy(update={"side_to_move": "black"})
    after = position(kings() + [("bp", "black", "pawn", 6, 5)], "red")
    result = resolve_initial([p_red, p_black], [after])
    assert result.side_to_move == "black"


def test_continuous_motion_is_one_review_window_with_peak_evidence():
    from app.analysis import stable_events
    base = np.zeros((100, 90, 3), np.uint8)
    peak = base.copy(); peak[20:80, 20:70] = 255
    small = base.copy(); small[30:50, 20:40] = 60
    frames = [(0, base), (0.2, base), (0.22, small), (0.24, peak), (0.26, small),
              (0.28, peak), (0.30, small), (0.32, base), (0.55, base)]
    events = list(stable_events(iter(frames)))
    uncertain = [e for e in events if e.uncertain]
    assert len(uncertain) == 1
    assert 1 <= len(uncertain[0].images) <= 3
    assert any(np.array_equal(image, peak) for image in uncertain[0].images)
    assert len([e for e in events if not e.uncertain]) == 1


def test_clear_following_board_resolves_zero_or_one_move_window():
    from app.analysis import extract_moves
    p = position(kings() + [("r", "red", "rook", 0, 9)])
    after = position(kings() + [("r", "red", "rook", 0, 5)], "black")
    result = extract_moves(p, [(1, [p, after]), (1.2, [after])])
    assert len(result) == 1
    assert result[0].status == "auto_verified"
    assert result[0].move.target.rank == 5
    assert result[0].video_end_s == 1.2


def test_localized_piece_change_triggers_even_when_global_area_is_small():
    from app.analysis import stable_events
    base = np.full((1001, 901, 3), 100, np.uint8)
    after = base.copy()
    after[440:460, 440:460] = 220
    events = list(stable_events(iter([(0, base), (0.2, base), (1, after), (1.2, after)])))
    assert len([event for event in events if not event.uncertain]) == 2


def test_equal_gray_color_change_is_not_ignored():
    from app.analysis import stable_events
    base = np.full((1001, 901, 3), 100, np.uint8)
    after = base.copy()
    base[420:480, 420:480] = (0, 0, 200)
    after[420:480, 420:480] = (0, 102, 0)
    events = list(stable_events(iter([(0, base), (0.2, base), (1, after), (1.2, after)])))
    assert len([event for event in events if not event.uncertain]) == 2


def test_static_video_uses_one_keyframe_and_noise_does_not_retrigger():
    from app.analysis import stable_events
    base = np.full((1001, 901, 3), 100, np.uint8)
    frames = [(i / 30, base if i % 2 else base + 1) for i in range(300)]
    events = list(stable_events(iter(frames)))
    assert len(events) == 1
    assert len(events[0].images) == 1


def test_scattered_compression_noise_is_not_a_board_event():
    from app.analysis import stable_events
    rng = np.random.default_rng(42)
    frames = []
    for i in range(30):
        image = np.full((200, 180, 3), 100, np.uint8)
        mask = rng.random((200, 180)) < 0.02
        image[mask] = 140
        frames.append((i / 30, image))
    events = list(stable_events(iter(frames)))
    assert len(events) == 1 and not events[0].uncertain


def test_stationary_elephants_read_as_advisors_do_not_create_review():
    from app.analysis import extract_moves
    from app.schemas import opening_position
    initial = opening_position()
    observed = initial.model_copy(deep=True)
    for piece in observed.pieces:
        if piece.color == "red" and piece.kind == "elephant":
            piece.kind = "advisor"
    assert extract_moves(initial, [(76.735, [observed])]) == []


def test_clear_moves_keep_trusted_identity_despite_model_letter_errors():
    from app.analysis import extract_moves, revalidate_suffix
    from app.chess import apply_move, legal_moves
    from app.schemas import Task, VideoMetadata, opening_position
    initial = opening_position()
    first = next(m for m in legal_moves(initial) if m.source.file == 4 and m.source.rank == 6)
    after_first = apply_move(initial, first)
    second = next(m for m in legal_moves(after_first) if m.source.file == 0 and m.source.rank == 3)
    after_second = apply_move(after_first, second)
    observations = []
    for time_s, board in [(76.735, initial), (80, after_first), (84, after_second)]:
        observed = board.model_copy(deep=True)
        for piece in observed.pieces:
            if piece.color == "red" and piece.kind == "elephant":
                piece.kind = "advisor"
        observations.append((time_s, [observed]))
    records = extract_moves(initial, observations)
    assert len(records) == 2
    assert all(r.status == "auto_verified" for r in records)
    assert [r.move.piece_id for r in records] == [first.piece_id, second.piece_id]
    assert records[-1].position_after == after_second
    task = Task(id="a" * 32, filename="clear.mp4", metadata=VideoMetadata(width=480, height=852, duration_s=200, codec="h264"),
                start_s=75, end_s=200, initial_position=initial, moves=records)
    result = revalidate_suffix(task, 1)
    assert result.status == "ready" and result.confirmed_prefix_length == 2


def test_letter_tolerance_does_not_accept_wrong_color_or_illegal_coordinates():
    from app.analysis import extract_moves
    from app.chess import apply_move, legal_moves
    from app.schemas import opening_position, Square
    initial = opening_position()
    first = next(m for m in legal_moves(initial) if m.source.file == 4 and m.source.rank == 6)
    wrong_color = apply_move(initial, first).model_copy(deep=True)
    next(p for p in wrong_color.pieces if p.id == first.piece_id).color = "black"
    illegal = initial.model_copy(deep=True)
    next(p for p in illegal.pieces if p.id == first.piece_id).square = Square(file=5, rank=5)
    for observed in [wrong_color, illegal]:
        records = extract_moves(initial, [(80, [observed])])
        assert records[0].status == "needs_review"
