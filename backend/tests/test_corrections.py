from test_chess import kings, position, move


def make_task(tmp_path):
    from app.storage import TaskStore
    from app.schemas import VideoMetadata, MoveRecord
    store = TaskStore(tmp_path)
    task = store.create("a.mp4", VideoMetadata(width=100, height=100, duration_s=10, codec="h264"))
    task.initial_position = position(kings() + [("r", "red", "rook", 0, 9), ("bp", "black", "pawn", 7, 5)])
    observed = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)])
    task.moves = [MoveRecord(index=1, video_start_s=1, video_end_s=1.5, status="needs_review"),
                  MoveRecord(index=2, move=move("bp", 7, 5, 6, 5, "black"), video_start_s=2, video_end_s=2.5,
                             observed_positions=[observed], status="tentative")]
    return store, task


def test_correct_early_move_revalidates_suffix(tmp_path):
    from app.analysis import correct_move
    store, task = make_task(tmp_path)
    fixed = correct_move(task, 1, move("r", 0, 9, 0, 5))
    assert fixed.moves[0].status == "corrected"
    assert fixed.moves[1].status == "auto_verified"
    assert fixed.confirmed_prefix_length == 2
    saved = store.save(fixed, 0)
    assert store.get(task.id).moves[0].status == "corrected"
    assert saved.revision == 1


def test_legal_but_unsupported_suffix_stays_tentative(tmp_path):
    from app.analysis import correct_move
    _, task = make_task(tmp_path)
    fixed = correct_move(task, 1, move("r", 0, 9, 0, 6))
    assert fixed.moves[1].status == "tentative"
    assert fixed.moves[1].position_after is None
    assert fixed.confirmed_prefix_length == 1


def test_initial_position_edit_invalidates_prefix(tmp_path):
    from app.analysis import correct_move, revalidate_suffix
    _, task = make_task(tmp_path)
    fixed = correct_move(task, 1, move("r", 0, 9, 0, 5))
    fixed.initial_position = position(kings() + [("r", "red", "rook", 1, 9), ("bp", "black", "pawn", 7, 5)])
    rechecked = revalidate_suffix(fixed, 1)
    assert rechecked.confirmed_prefix_length == 0
    assert rechecked.moves[0].status == "needs_review"


def test_inferred_two_step_group_survives_revalidation(tmp_path):
    from app.analysis import extract_moves, revalidate_suffix
    _, task = make_task(tmp_path)
    after = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)])
    task.moves = extract_moves(task.initial_position, [(2, [after])])
    task.confirmed_prefix_length = 2
    result = revalidate_suffix(task, 1)
    assert result.confirmed_prefix_length == 2
    assert all(r.status == "auto_verified" for r in result.moves)


def test_replace_missing_interval_with_three_moves(tmp_path):
    from app.analysis import correct_interval
    _, task = make_task(tmp_path)
    actions = [move("r", 0, 9, 0, 8), move("bp", 7, 5, 6, 5, "black"), move("r", 0, 8, 0, 5)]
    # Remaining evidence describes the same final board, so no extra fourth move is needed.
    result = correct_interval(task, 1, actions)
    assert result.confirmed_prefix_length == 3
    assert len(result.moves) == 3
    assert [r.index for r in result.moves] == [1, 2, 3]


def test_remove_false_event_without_losing_suffix(tmp_path):
    from app.analysis import correct_interval
    from app.schemas import MoveRecord
    _, task = make_task(tmp_path)
    final = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 7, 5)], "black")
    task.moves[1] = MoveRecord(index=2, video_start_s=2, video_end_s=2.5, status="tentative", observed_positions=[final])
    result = correct_interval(task, 1, [])
    assert len(result.moves) == 1
    assert result.moves[0].move.piece_id == "r"
    assert result.confirmed_prefix_length == 1


def test_correct_first_of_inferred_group(tmp_path):
    from app.analysis import correct_move, extract_moves
    _, task = make_task(tmp_path)
    after = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)])
    task.moves = extract_moves(task.initial_position, [(2, [after])])
    task.confirmed_prefix_length = 2
    result = correct_move(task, 1, move("r", 0, 9, 0, 5))
    assert result.confirmed_prefix_length == 2
    assert all(record.inference_group is None for record in result.moves)


def test_correct_second_of_inferred_group(tmp_path):
    from app.analysis import correct_move, extract_moves
    _, task = make_task(tmp_path)
    after = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 6, 5)])
    task.moves = extract_moves(task.initial_position, [(2, [after])])
    task.confirmed_prefix_length = 2
    result = correct_move(task, 2, move("bp", 7, 5, 7, 6, "black"))
    assert result.confirmed_prefix_length == 2
    assert result.moves[1].status == "corrected"
    assert result.moves[1].move.target.file == 7


def test_revalidation_keeps_zero_or_one_move_ambiguity(tmp_path):
    from app.analysis import extract_moves, revalidate_suffix
    _, task = make_task(tmp_path)
    after = position(kings() + [("r", "red", "rook", 0, 5), ("bp", "black", "pawn", 7, 5)], "black")
    task.moves = extract_moves(task.initial_position, [(2, [task.initial_position, after])])
    assert task.moves[0].status == "needs_review"
    result = revalidate_suffix(task, 1)
    assert result.moves[0].status == "needs_review"
    assert result.confirmed_prefix_length == 0
    assert result.moves[0].position_after is None
