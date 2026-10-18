from app.actual import actual_events
from app.chess import board_signature
from app.schemas import Task, VideoMetadata, opening_position, Square
from app.turns import TurnReplay, revalidate_turns
from app.board_evidence import check_transition
import pytest
from test_tracking import board_image


def test_black_returns_horse_then_changes_to_cannon_and_red_horse_follows():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    task = Task(id='f' * 32, filename='takeback.mp4', end_s=60,
                metadata=VideoMetadata(width=1001, height=1101, duration_s=60, codec='h264'),
                initial_position=before, rules_enabled=True, replay_mode='actual')
    horse = before.model_copy(deep=True)
    next(p for p in horse.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    cannon = before.model_copy(deep=True)
    next(p for p in cannon.pieces if p.id == 'black_cannon_1').square = Square(file=5, rank=2)
    reply = cannon.model_copy(deep=True)
    next(p for p in reply.pieces if p.id == 'red_horse_1').square = Square(file=2, rank=7)
    replay = TurnReplay(task, before)
    replay.observe(horse, 38, ['horse.jpg'], 37)
    replay.observe(before.model_copy(deep=True), 45, ['return.jpg'], 44)
    replay.observe(cannon, 50, ['cannon.jpg'], 47)
    replay.observe(reply, 54, ['reply.jpg'], 53)
    replay.end(60)
    assert len(task.moves) == 4
    forward, returned, replacement, red = task.moves
    assert forward.move.piece_id == returned.move.piece_id == 'black_horse_1'
    assert forward.violation == returned.violation == 'takeback'
    assert '黑方悔棋' in returned.violation_note
    assert returned.move.target == Square(file=1, rank=0)
    assert returned.position_after.side_to_move == 'black'
    assert replacement.status == red.status == 'auto_verified'
    assert replacement.violation is None and red.violation is None
    assert replacement.move.piece_id == 'black_cannon_1'
    assert red.move.piece_id == 'red_horse_1'
    assert red.move.source == Square(file=1, rank=9)
    assert red.move.target == Square(file=2, rank=7)
    reloaded = revalidate_turns(task)
    assert all(r.status == 'auto_verified' for r in reloaded.moves)
    assert reloaded.moves[1].position_after.side_to_move == 'black'
    assert reloaded.moves[-1].move.piece_id == 'red_horse_1'


@pytest.mark.parametrize('case', ['visible', 'occluded', 'legal_candidate', 'independent_move', 'pending_opponent'])
def test_visible_unchanged_victim_resolves_in_hand_illegal_capture_candidate(case):
    from test_move_evidence import cells, moved
    before = opening_position()
    if case == 'pending_opponent':
        before.side_to_move = 'black'
    if case == 'legal_candidate':
        next(p for p in before.pieces if p.id == 'black_pawn_8').square = Square(file=2, rank=7)
    task = Task(id='e' * 32, filename='held-horse.mp4', end_s=6,
                metadata=VideoMetadata(width=901, height=1001, duration_s=6, codec='h264'),
                initial_position=before, initial_turn_known=True, rules_enabled=True, replay_mode='actual')
    replay = TurnReplay(task, before)
    if case == 'pending_opponent':
        before = moved(before, 'black_rook_0', 0, 1)
        before.side_to_move = 'red'
        replay.observe(before, 1, ['black.jpg'], .5)
    held = moved(before, 'red_horse_1', 2, 7) if case == 'legal_candidate' else moved(before, 'red_horse_1', 8, 3)
    held.pieces = [p for p in held.pieces if p.id != 'black_pawn_8']
    observations = [held]
    if case == 'independent_move':
        observations.append(moved(before, 'black_pawn_8', 8, 4))
    replay.uncertain(4.5, observations, ['held.jpg'], '局部画面持续出现未解释的棋子变化：落点不符合棋子走法')
    landed = moved(before, 'red_horse_1', 0, 7) if case == 'legal_candidate' else moved(before, 'red_horse_1', 2, 7)
    check = check_transition(before, landed, cells(before), cells(landed))
    if case == 'occluded':
        check.status = 'partial'
        check.unknown_after.append(Square(file=8, rank=3))
    replay.observe(landed, 5, ['landed.jpg'], 4, board_check=check)
    replay.end(6)
    assert len(task.moves) == (1 if case == 'visible' else 2)
    assert task.moves[-1].status == 'auto_verified'
    assert task.moves[-1].move.target == next(p.square for p in landed.pieces if p.id == 'red_horse_1')


def test_source_and_landing_observed_in_different_frames_keep_their_witnesses():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    moved = before.model_copy(deep=True)
    next(p for p in moved.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames, observations = [], {}
    for time, position in [(0, before), (1, moved), (1.1, moved), (2, moved), (2.1, moved)]:
        image = board_image(position)
        cells = {(f, r): None for f in range(9) for r in range(10)}
        cells.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
        if 0 < time < 2:
            cells.pop((2, 2))
        elif time >= 2:
            cells.pop((1, 0))
        observations[id(image)] = cells
        frames.append((time, image))
    current = before
    accepted = []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current,
                               turn_boundaries=True, cell_reader=lambda image, _: observations[id(image)]):
        if event.position is not None and not event.review_reason:
            current = event.position
            accepted.append(event)
    recovered = [e for e in accepted if board_signature(e.position) == board_signature(moved)]
    assert recovered
    event = recovered[-1]
    assert event.board_check is not None and event.board_check.status in ('verified', 'partial')
    assert any(1 <= time < 2 for time in event.image_times_s)
    assert any(time >= 2 for time in event.image_times_s)


def test_one_clear_historical_return_before_another_piece_is_not_discarded():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    horse = before.model_copy(deep=True)
    next(p for p in horse.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    cannon = before.model_copy(deep=True)
    next(p for p in cannon.pieces if p.id == 'black_cannon_1').square = Square(file=5, rank=2)
    frames = [(0, board_image(before)), (1, board_image(horse)), (1.1, board_image(horse)),
              (2, board_image(before)), (2.04, board_image(cannon)), (2.1, board_image(cannon))]
    current, seen = before, []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current, turn_boundaries=True):
        if event.position is not None and not event.review_reason:
            if event.board_check is None or event.board_check.status != 'conflict':
                current = event.position
                seen.append((event.time_s, board_signature(current)))
    assert (2, board_signature(before)) in seen
    assert board_signature(cannon) == board_signature(current)


def test_pawn_return_to_visible_previous_position_is_not_blocked_as_backward_move():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    pawn = before.model_copy(deep=True)
    next(p for p in pawn.pieces if p.id == 'black_pawn_8').square = Square(file=8, rank=4)
    reply = before.model_copy(deep=True)
    next(p for p in reply.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames = [(0, board_image(before)), (1, board_image(pawn)), (1.1, board_image(pawn)),
              (2, board_image(before)), (2.1, board_image(before)),
              (3, board_image(reply)), (3.1, board_image(reply))]
    current, seen = before, []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current, turn_boundaries=True):
        if event.position is not None and not event.review_reason and (
                event.board_check is None or event.board_check.status != 'conflict'):
            current = event.position
            seen.append(board_signature(current))
    assert board_signature(before) in seen[2:]
    assert board_signature(current) == board_signature(reply)


def test_clear_pawn_return_at_eof_discards_an_earlier_rejected_landing():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    forward, illegal = before.model_copy(deep=True), before.model_copy(deep=True)
    next(p for p in forward.pieces if p.id == 'black_pawn_8').square = Square(file=8, rank=4)
    next(p for p in illegal.pieces if p.id == 'black_pawn_8').square = Square(file=7, rank=5)
    current, events = before, []
    frames = [(0, board_image(before)), (1, board_image(forward)), (1.1, board_image(forward)),
              (2, board_image(illegal)), (3, board_image(before))]
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current, turn_boundaries=True):
        events.append(event)
        if event.position is not None and not event.review_reason and (
                event.board_check is None or event.board_check.status != 'conflict'):
            current = event.position
    assert board_signature(current) == board_signature(before)
    assert [event.time_s for event in events] == sorted(event.time_s for event in events)
    assert not events[-1].review_reason


def test_return_endpoints_can_clear_while_next_piece_is_already_in_hand():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    horse = before.model_copy(deep=True)
    next(p for p in horse.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    observations, frames = {}, []
    for time, position in [(0, before), (1, horse), (1.1, horse), (2, before), (2.1, before)]:
        image = board_image(position)
        cells = {(f, r): None for f in range(9) for r in range(10)}
        cells.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
        if time == 2:
            cells.pop((2, 2))
        elif time == 2.1:
            cells.pop((1, 0))
            cells[(1, 2)] = None
        observations[id(image)] = cells
        frames.append((time, image))
    current, returned = before, []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current,
                               turn_boundaries=True, cell_reader=lambda image, _: observations[id(image)]):
        if event.position is not None and not event.review_reason and (event.board_check is None or event.board_check.status != 'conflict'):
            if event.time_s >= 2 and board_signature(event.position) == board_signature(before):
                returned.append(event)
            current = event.position
    assert returned
    assert returned[-1].board_check.status == 'partial'
    assert Square(file=1, rank=2) in returned[-1].board_check.unknown_after


def test_landing_before_opponents_destination_preserves_next_piece_in_hand():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames, observations = [], {}
    for time, position in [(0, before), (1, after), (1.1, after), (1.2, after)]:
        image = board_image(position)
        cells = {(f, r): None for f in range(9) for r in range(10)}
        cells.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
        if time == 1:
            cells.pop((2, 2))
        elif time > 1:
            cells.pop((1, 0))
            cells[(1, 7)] = None
        observations[id(image)] = cells
        frames.append((time, image))
    current, accepted = before, []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: current,
                               turn_boundaries=True, cell_reader=lambda image, _: observations[id(image)]):
        if event.position is not None and not event.review_reason and (event.board_check is None or event.board_check.status != 'conflict'):
            current = event.position
            accepted.append(event)
    assert board_signature(current) == board_signature(after)
    assert accepted[-1].board_check.status == 'partial'
    assert Square(file=1, rank=7) in accepted[-1].board_check.unknown_after


def test_merged_same_piece_hints_are_resolved_by_visible_final_landing():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    next(p for p in before.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    task = Task(id='f' * 32, filename='rook.mp4', end_s=10,
                metadata=VideoMetadata(width=1001, height=1101, duration_s=10, codec='h264'),
                initial_position=before, rules_enabled=True, replay_mode='actual')
    hint, after = before.model_copy(deep=True), before.model_copy(deep=True)
    next(p for p in hint.pieces if p.id == 'black_rook_0').square = Square(file=1, rank=0)
    next(p for p in after.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=1)
    replay = TurnReplay(task, before)
    replay.uncertain(2, [hint], ['hint.jpg'], '局部画面持续出现未解释变化')
    replay.uncertain(3, [before], ['return.jpg'], '移动起点或最终落点缺少前后可见证据')
    replay.observe(after, 4, ['clear.jpg'], 1)
    replay.end(5)
    assert len(task.moves) == 1
    assert task.moves[0].move.piece_id == 'black_rook_0'


@pytest.mark.parametrize('other_visible,later_clear', [(True, False), (False, False), (False, True)])
def test_capture_resolves_color_swap_only_when_other_piece_is_visible_unchanged(other_visible, later_clear):
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    next(p for p in before.pieces if p.id == 'black_pawn_6').square = Square(file=6, rank=4)
    next(p for p in before.pieces if p.id == 'red_pawn_6').square = Square(file=6, rank=5)
    task = Task(id='f' * 32, filename='capture.mp4', end_s=10,
                metadata=VideoMetadata(width=1001, height=1101, duration_s=10, codec='h264'),
                initial_position=before, rules_enabled=True, replay_mode='actual')
    hint, after = before.model_copy(deep=True), before.model_copy(deep=True)
    next(p for p in hint.pieces if p.id == 'black_king_4').square = Square(file=6, rank=5)
    next(p for p in hint.pieces if p.id == 'red_pawn_6').square = Square(file=4, rank=0)
    next(p for p in after.pieces if p.id == 'black_pawn_6').square = Square(file=6, rank=5)
    after.pieces = [p for p in after.pieces if p.id != 'red_pawn_6']
    observed_before = {(f, r): None for f in range(9) for r in range(10)}
    observed_after = dict(observed_before)
    observed_before.update({(p.square.file, p.square.rank): p.color for p in before.pieces})
    observed_after.update({(p.square.file, p.square.rank): p.color for p in after.pieces})
    if not other_visible:
        observed_after.pop((4, 0))
    audit = check_transition(before, after, observed_before, observed_after)
    replay = TurnReplay(task, before)
    replay.uncertain(2, [hint], ['occluded.jpg'], '整盘观察包含多枚棋子或身份变化')
    replay.observe(after, 3, ['clear.jpg'], 1, board_check=audit)
    if later_clear:
        observed_after[(4, 0)] = 'black'
        replay.observe(after, 3.5, ['uncovered.jpg'], 1,
                       board_check=check_transition(before, after, observed_before, observed_after))
    replay.end(4)
    assert len(task.moves) == (1 if other_visible or later_clear else 2)
    assert any(r.move and r.move.piece_id == 'black_pawn_6' for r in task.moves)
    if other_visible or later_clear:
        assert 'occluded.jpg' in task.moves[-1].evidence_ids
