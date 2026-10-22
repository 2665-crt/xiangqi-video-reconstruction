from app.chess import apply_move, legal_moves
from app.move_inference import MoveInference
from app.schemas import opening_position


def cells(position):
    result = {(f, r): None for f in range(9) for r in range(10)}
    result.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
    return result


def play(position, identity, target):
    move = next(m for m in legal_moves(position) if m.piece_id == identity
                and (m.target.file, m.target.rank) == target)
    return apply_move(position, move)


def test_inference_keeps_departure_and_first_landing_when_later_frames_arrive():
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    inference = MoveInference(before, 0)
    inference.observe(1, cells(before))
    lifted = cells(before)
    lifted[(1, 9)] = None
    inference.observe(1.04, lifted)
    inference.observe(1.08, cells(after))
    result = inference.observe(4, cells(after))
    assert len(result) == 1
    assert result[0].action_start_s == 1
    assert result[0].action_end_s == 1.08
    assert result[0].timing_status == 'observed'


def test_hidden_destination_does_not_discard_the_visible_departure_time():
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    inference = MoveInference(before, 0)
    visible_source = cells(before)
    visible_source.pop((2, 7))
    inference.observe(1, visible_source)
    lifted = dict(visible_source)
    lifted[(1, 9)] = None
    inference.observe(1.04, lifted)
    result = inference.observe(1.08, cells(after))
    assert result[0].action_start_s == 1
    assert result[0].action_end_s == 1.08
    assert result[0].timing_status == 'observed'


def test_cross_step_inference_recovers_separate_times_from_continuous_samples():
    before = opening_position()
    first = play(before, 'red_horse_1', (2, 7))
    second = play(first, 'black_horse_1', (2, 2))
    final = play(second, 'red_horse_1', (4, 8))
    inference = MoveInference(before, 0)
    for time_s, position in [(1, before), (2, first), (3, first), (4, second), (5, second), (6, final)]:
        inference.sample(time_s, cells(position))
    inference.observe(7, cells(final))
    result = inference.observe(8, cells(final))
    assert len(result) == 3
    assert [r.action_end_s for r in result] == [2, 4, 6]
    assert [r.action_start_s for r in result] == [1, 3, 5]
    assert all(r.timing_status == 'observed' for r in result)


def test_unseen_gap_keeps_missing_times_instead_of_putting_all_moves_at_recovery_time():
    before = opening_position()
    final = play(play(play(before, 'red_horse_1', (2, 7)), 'black_horse_1', (2, 2)), 'red_horse_1', (4, 8))
    inference = MoveInference(before, 0)
    inference.observe(10, cells(final))
    result = inference.observe(11, cells(final))
    assert len(result) == 3
    assert all(r.action_start_s is None and r.action_end_s is None for r in result)
    assert all(r.status == 'tentative' and r.timing_status == 'uncertain' for r in result)


def test_later_repeated_move_cannot_supply_missing_timing_to_an_earlier_turn():
    before = opening_position()
    inference = MoveInference(before, 0, origin_uncertain=True)
    position = before
    for second, identity, target in [(10, 'red_horse_1', (2, 7)), (20, 'black_horse_1', (2, 2)),
            (30, 'red_horse_1', (1, 9)), (40, 'black_horse_1', (1, 0)), (50, 'red_horse_1', (2, 7))]:
        position = play(position, identity, target)
        inference.observe(second, cells(position))
        result = inference.observe(second + 1, cells(position))
    assert len(result) == 5
    assert result[0].timing_status == 'estimated'
    assert result[0].action_start_s is None
    assert result[0].action_end_s == 10
    assert result[-1].action_start_s == 41
    assert result[-1].action_end_s == 50


def test_ambiguous_next_move_does_not_block_the_already_observed_common_prefix():
    before = opening_position()
    before.pieces = [p for p in before.pieces if p.id != 'black_pawn_4']
    for p in before.pieces:
        if p.id == 'black_horse_1':
            p.square.file, p.square.rank = 3, 1
        elif p.id == 'black_horse_7':
            p.square.file, p.square.rank = 5, 1
    first = play(before, 'red_pawn_0', (0, 5))
    reply = play(first, 'black_horse_1', (4, 3))
    ambiguous = cells(reply)
    ambiguous.pop((3, 1))
    ambiguous.pop((5, 1))
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    inference.observe(1, cells(first))
    inference.observe(2, cells(first))
    result = inference.observe(3, ambiguous)
    assert result[0].move.piece_id == 'red_pawn_0'
    assert result[0].status == 'auto_verified'
    assert result[0].action_start_s == 0
    assert result[1].status == 'tentative'


def test_unrelated_hidden_stationary_square_does_not_force_clear_moves_to_remain_inferred():
    before = opening_position()
    first = play(before, 'red_horse_1', (2, 7))
    reply = play(first, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    for second, position in [(1, first), (2, first), (3, reply)]:
        partial = cells(position)
        partial.pop((8, 0))
        result = inference.observe(second, partial)
    assert result[0].status == 'auto_verified'
    assert result[0].action_start_s == 0
    assert result[0].action_end_s == 1
    assert result[0].position_after == first


def test_continuous_clear_samples_confirm_a_move_even_when_search_runs_less_often():
    before = opening_position()
    first = play(before, 'red_horse_1', (2, 7))
    reply = play(first, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    inference.observe(1, cells(first))
    inference.sample(1.05, cells(first))
    inference.sample(1.1, cells(first))
    result = inference.observe(2, cells(reply))
    assert result[0].status == 'auto_verified'
    assert result[0].action_end_s == 1


def test_unobserved_gap_does_not_block_independently_observed_later_moves():
    before = opening_position()
    gap = play(play(play(before, 'red_horse_1', (2, 7)), 'black_horse_1', (2, 2)), 'red_horse_1', (4, 8))
    fourth = play(gap, 'black_pawn_0', (0, 4))
    fifth = play(fourth, 'red_pawn_0', (0, 5))
    inference = MoveInference(before, 0)
    inference.observe(10, cells(gap))
    inference.observe(11, cells(gap))
    inference.observe(12, cells(gap))
    inference.observe(13, cells(fourth))
    inference.observe(14, cells(fourth))
    result = inference.observe(15, cells(fifth))
    assert all(r.status == 'tentative' for r in result[:3])
    assert result[2].depends_on_steps == [1, 2]
    assert result[3].status == 'auto_verified'
    assert result[3].action_start_s == 12
    assert result[3].action_end_s == 13
    assert result[3].position_after == fourth


def test_independently_verified_recovery_exits_inference_and_confirms_following_moves(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from app import actual, fusion, tracking_recovery
    from app.analysis import VideoEvent
    from app.schemas import BoardCheck, Calibration, MoveRecord, VideoMetadata
    from app.storage import TaskStore

    before = opening_position()
    first = play(before, 'red_horse_1', (2, 7))
    reply = play(first, 'black_horse_1', (2, 2))

    class ClearRecovery:
        def __init__(self, reader):
            self.requested, self.next_check_s, self.attempts, self.check_count = True, 0, 0, 0
            self.reason, self.board_check = '', None
        def request(self, time_s):
            self.requested = True
        def observe(self, time_s, image, *args, **kwargs):
            self.check_count += 1
            self.evidence = [(time_s, image)]
            self.requested = False
            return before
        def reset(self):
            self.requested = False

    monkeypatch.setenv('VISION_FUSION', '1')
    monkeypatch.setattr(fusion, 'local_fusion_classifier', lambda: SimpleNamespace(detector=SimpleNamespace(readings=lambda image: {})))
    monkeypatch.setattr(tracking_recovery, 'VisualRecovery', ClearRecovery)
    monkeypatch.setattr(actual, '_tracking_cells', lambda image, *args, **kwargs: cells([before, first, reply][int(image[0, 0, 0])]))

    def events(boards, position, confirmed_position, **kwargs):
        image = np.zeros((20, 20, 3), np.uint8)
        for second in (0, 1):
            event = kwargs['reanchor_reader'](second, image, confirmed_position())
            if event is not None:
                yield event
        if kwargs['inference_active']():
            return
        for second, state, after in [(2, 1, first), (4, 2, reply)]:
            following = np.full((20, 20, 3), state, np.uint8)
            yield VideoEvent(second, [image, following], position=after, action_start_s=second - .5,
                image_times_s=(second - 1, second), board_check=BoardCheck(status='verified', visible_before=90, visible_after=90))
            image = following

    monkeypatch.setattr(actual, 'actual_events', events)
    store = TaskStore(tmp_path)
    task = store.create('clear-recovery.mp4', VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'))
    task.initial_position, task.context_position, task.resume_video_s = before, before, 0
    task.initial_frame_s, task.end_s = 0, 5
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed = True
    task.replay_mode, task.rules_enabled, task.status = 'actual', True, 'analyzing'
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=0, status='tentative',
        change_type='review', predicted_position=before, tracking_anchor=before, tracking_anchor_s=0, position_before=before)]
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'source.mp4').write_bytes(b'test')
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.error is None
    assert result.moves[0].status == 'tentative'
    confirmed = [r for r in result.moves if r.status == 'auto_verified']
    assert confirmed and confirmed[0].move.piece_id == 'red_horse_1'
    assert confirmed[0].action_start_s == 1.5
