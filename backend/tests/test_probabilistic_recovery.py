import pytest

from app.chess import apply_move, board_signature, legal_moves
from app.schemas import opening_position
from app.tracking_recovery import VisualRecovery


def occupied(position):
    result = {(f, r): None for f in range(9) for r in range(10)}
    result.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
    return result


def play(position, identity, target):
    move = next(m for m in legal_moves(position) if m.piece_id == identity
                and (m.target.file, m.target.rank) == target)
    return apply_move(position, move)


def test_renumbering_preserves_inferred_dependencies():
    from app.schemas import MoveRecord
    from app.turns import renumber_records
    from types import SimpleNamespace
    records = [MoveRecord(index=2, video_start_s=0, video_end_s=1,
        status='tentative', depends_on_steps=[1, 3], related_steps=[1])]
    renumber_records(SimpleNamespace(moves=records), {1: 2, 2: 3})
    assert records[0].depends_on_steps == [2]
    assert records[0].related_steps == [2]


def test_low_confidence_recovery_keeps_adjacent_samples_after_first_burst():
    before = opening_position()
    weak = {(p.square.file, p.square.rank): {'confidence': .3,
            'class_probabilities': {f'{p.color}_{p.kind}': .8}} for p in before.pieces}
    tracker = VisualRecovery(lambda image: weak)
    tracker.request(0)
    for second in range(13):
        tracker.observe(second, None, before, occupied(before))
    assert len(tracker.samples) >= 2
    assert tracker.samples[-1][0] - tracker.samples[-2][0] <= 3


def test_inference_inherits_identity_with_missing_stationary_pieces():
    from app.move_inference import MoveInference
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    cells = occupied(after)
    for square in [(0, 0), (4, 0), (8, 0)]:
        cells.pop(square)
    inference = MoveInference(before, 0)
    result = inference.observe(1, cells)
    assert result and board_signature(result[-1].predicted_position) == board_signature(after)
    assert result[-1].status == 'tentative'
    assert next(p for p in result[-1].predicted_position.pieces if p.id == 'black_rook_0').kind == 'rook'


def test_inferred_move_serializes_concrete_estimated_video_time():
    from app.move_inference import MoveInference
    from app.schemas import MoveRecord
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    record = MoveInference(before, 7).observe(15.8, occupied(after))[-1]
    restored = MoveRecord.model_validate_json(record.model_dump_json())
    assert restored.timing_status == 'estimated'
    assert restored.action_end_s == 15.8
    assert restored.video_end_s == 15.8
    assert restored.predicted_position == after


def test_inference_ignores_isolated_false_empty_without_deleting_piece():
    from app.move_inference import MoveInference
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    cells = occupied(after)
    cells[(8, 0)] = None
    inference = MoveInference(before, 0)
    result = inference.observe(1, cells)
    assert result[-1].move.piece_id == 'red_horse_1'
    assert len(result[-1].predicted_position.pieces) == 32


def test_known_pawn_can_be_inferred_when_its_landing_is_occluded():
    from app.move_inference import MoveInference
    before = opening_position()
    after = play(before, 'red_pawn_4', (4, 5))
    cells = occupied(after)
    cells.pop((4, 5))
    inference = MoveInference(before, 0)
    result = inference.observe(1, cells)
    assert result and result[0].move.piece_id == 'red_pawn_4'
    assert board_signature(result[0].predicted_position) == board_signature(after)
    assert result[0].status == 'tentative'


def test_tail_completion_requires_a_directly_visible_opponent_landing():
    from app.move_inference import MoveInference
    before = play(opening_position(), 'red_rook_0', (0, 8))
    after = play(before, 'black_pawn_4', (4, 4))
    cells = occupied(after)
    cells.pop((4, 4))
    inference = MoveInference(before, 0)
    assert any(m.piece_id == 'black_pawn_4' for m, _ in inference._successors(before, cells))
    assert list(inference._successors(before, cells, require_visible_landing=True)) == []
    cells[(4, 4)] = 'black'
    assert any(m.piece_id == 'black_pawn_4'
               for m, _ in inference._successors(before, cells, require_visible_landing=True))
    cells.pop((4, 3))
    assert list(inference._successors(before, cells, require_visible_landing=True)) == []


def test_probable_board_recovery_requires_stable_complete_conflict_and_backs_off():
    from app.move_inference import MoveInference
    before = opening_position()
    cells = occupied(before)
    cells[(0, 0)] = None
    inference = MoveInference(before, 0)
    assert not inference.recovery_due(0, cells, before)
    assert not inference.recovery_due(1, cells, before)
    hidden = dict(cells)
    hidden.pop((8, 0))
    assert not inference.recovery_due(2, hidden, before)
    assert not inference.recovery_due(3, cells, before)
    assert inference.recovery_due(5, cells, before)
    assert not inference.recovery_due(6, cells, before)
    assert inference.recovery_due(15, cells, before)
    assert not inference.recovery_due(16, occupied(before), before)
    assert not inference.recovery_due(17, cells, before)


def test_recovery_candidate_uses_direct_occupancy_and_rejects_missing_kings():
    from app.move_inference import MoveInference
    before = opening_position()
    observed = before.model_copy(deep=True)
    observed.pieces = [p for p in observed.pieces if p.id != 'black_rook_0']
    inference = MoveInference(before, 0)
    candidate = inference.constrain_recovery(before, occupied(observed))
    assert candidate and board_signature(candidate) == board_signature(observed)
    missing = observed.model_copy(deep=True)
    missing.pieces = [p for p in missing.pieces if p.kind != 'king' or p.color != 'black']
    assert inference.constrain_recovery(missing, occupied(observed)) is None
    hidden = occupied(observed)
    hidden.pop((4, 4))
    assert inference.constrain_recovery(before, hidden) is None
    from app.schemas import Piece, Square
    extra = before.model_copy(deep=True)
    extra.pieces.append(Piece(id='ghost_king', color='black', kind='king', square=Square(file=0, rank=4)))
    assert board_signature(inference.constrain_recovery(extra, occupied(before))) == board_signature(before)
    wrong_color = observed.model_copy(deep=True)
    next(p for p in wrong_color.pieces if p.id == 'red_rook_0').color = 'black'
    assert inference.constrain_recovery(wrong_color, occupied(observed)) is None


def test_uncertain_recovery_origin_accepts_either_first_side_without_auto_confirmation():
    from app.move_inference import MoveInference
    before = opening_position()
    before.side_to_move = 'black'
    first = play(before, 'black_horse_1', (2, 2))
    reply = play(first, 'red_horse_1', (2, 7))
    uncertain = before.model_copy(update={'side_to_move': 'red'})
    inference = MoveInference(uncertain, 0, origin_uncertain=True)
    inference.observe(1, occupied(before))
    inference.observe(2, occupied(first))
    inference.observe(3, occupied(first))
    result = inference.observe(4, occupied(reply))
    assert [r.move.color for r in result] == ['black', 'red']
    assert all(r.status == 'tentative' and r.position_after is None for r in result)


def test_static_or_lifted_piece_without_landing_does_not_invent_move():
    from app.move_inference import MoveInference
    before = opening_position()
    inference = MoveInference(before, 0)
    assert inference.observe(1, occupied(before)) == []
    cells = occupied(before)
    cells[(1, 9)] = None
    assert inference.observe(2, cells) == []


def test_inference_revises_same_piece_slide_and_continues_with_opponent():
    from app.move_inference import MoveInference
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    final = play(before, 'red_rook_0', (0, 7))
    reply = play(final, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(1, occupied(first))
    revised = inference.observe(2, occupied(final))
    assert len(revised) == 1
    assert revised[0].move.source.rank == 9 and revised[0].move.target.rank == 7
    result = inference.observe(3, occupied(reply))
    assert [r.move.piece_id for r in result] == ['red_rook_0', 'black_horse_1']
    assert board_signature(result[-1].predicted_position) == board_signature(reply)


def test_inference_scores_are_not_presented_as_calibrated_probabilities():
    from app.move_inference import MoveInference
    before = opening_position()
    inference = MoveInference(before, 0)
    result = inference.observe(1, occupied(play(before, 'red_horse_1', (2, 7))))
    assert result[0].inference_score is not None
    assert result[0].position_after is None
    assert '推断' in result[0].reason


def test_temporary_occlusion_does_not_erase_the_preceding_move_hypothesis():
    from app.move_inference import MoveInference
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    inference = MoveInference(before, 0)
    inference.observe(1, occupied(after))
    cells = occupied(after)
    cells.pop((1, 9))
    cells.pop((2, 7))
    result = inference.observe(2, cells)
    assert len(result) == 1
    assert result[0].move.piece_id == 'red_horse_1'


def test_following_clear_frame_selects_the_correct_earlier_ambiguous_source():
    from app.move_inference import MoveInference
    before = opening_position()
    before.pieces = [p for p in before.pieces if p.kind != 'cannon']
    for piece in before.pieces:
        if piece.id in ('red_rook_0', 'red_rook_8'):
            piece.square.rank = 7
    first = play(before, 'red_rook_8', (4, 7))
    cells = occupied(first)
    cells.pop((0, 7))
    cells.pop((8, 7))
    inference = MoveInference(before, 0)
    inference.observe(1, cells)
    reply = play(first, 'black_horse_1', (2, 2))
    reply_cells = occupied(reply)
    reply_cells.pop((0, 7))
    reply_cells.pop((8, 7))
    inference.observe(2, reply_cells)
    result = inference.observe(3, occupied(reply))
    assert [r.move.piece_id for r in result] == ['red_rook_8', 'black_horse_1']


def test_manual_correction_rebases_inferred_suffix_without_silently_confirming_it():
    from app.move_inference import MoveInference
    from app.turns import replay_corrections
    from test_tracking_reanchor import task_for
    before = opening_position()
    first = play(before, 'red_horse_1', (2, 7))
    reply = play(first, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(1, occupied(first))
    records = inference.observe(2, occupied(reply))
    records[0].status = 'corrected'
    records[0].manual_override = True
    records[0].position_after = first
    task = task_for(before)
    task.moves = records
    result = replay_corrections(task, 1)
    assert result.moves[1].status == 'tentative'
    assert board_signature(result.moves[1].predicted_position) == board_signature(reply)
    assert result.moves[1].position_after is None


def test_revalidation_does_not_turn_a_legal_inference_into_visual_confirmation():
    from app.move_inference import MoveInference
    from app.actual import revalidate_actual
    from test_tracking_reanchor import task_for
    before = opening_position()
    inference = MoveInference(before, 0)
    task = task_for(before)
    task.moves = inference.observe(1, occupied(play(before, 'red_horse_1', (2, 7))))
    result = revalidate_actual(task)
    assert result.moves[0].status == 'tentative'
    assert result.moves[0].position_after is None


def test_confirmed_inferred_prefix_cannot_disappear_or_be_downgraded():
    from app.move_inference import MoveInference
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    reply = play(first, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(0, occupied(before))
    inference.observe(1, occupied(first))
    previous = inference.observe(2, occupied(first))
    following = inference.observe(3, occupied(reply))
    for old in previous:
        if old.status == 'auto_verified':
            assert following[old.index - 1].status == 'auto_verified'
    inference.observe(4, occupied(reply))
    committed = inference.observe(5, occupied(reply))
    assert committed[0].status == 'auto_verified'
    returned = inference.observe(6, occupied(before))
    assert returned and returned[0].status == 'auto_verified'
    assert returned[0].move.target.rank == 8


def test_three_move_gap_can_bridge_an_unobserved_intermediate_landing():
    from app.move_inference import MoveInference
    before = opening_position()
    final = play(play(play(before, 'red_horse_1', (2, 7)), 'black_horse_1', (2, 2)), 'red_horse_1', (4, 8))
    inference = MoveInference(before, 0)
    records = []
    for second in range(1, 5):
        records = inference.observe(second, occupied(final))
    assert len(records) == 3
    assert board_signature(records[-1].predicted_position) == board_signature(final)
    assert all(r.status == 'tentative' for r in records)


def test_tail_extension_only_revises_last_piece_without_adding_next_turn():
    from app.move_inference import MoveInference
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    final = play(before, 'red_rook_0', (0, 7))
    inference = MoveInference(before, 0)
    inference.observe(1, occupied(first))
    records = inference.observe(3, occupied(final), tail_only=True)
    assert len(records) == 1 and records[0].move.target.rank == 7
    reply = play(final, 'black_horse_1', (2, 2))
    records = inference.observe(4, occupied(reply), tail_only=True)
    assert len(records) == 1


def test_tail_can_return_to_origin_without_revising_previous_turns():
    from app.move_inference import MoveInference
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    inference = MoveInference(before, 0)
    inference.observe(1, occupied(first))
    assert inference.observe(2, occupied(before), tail_only=True) == []


def test_event_generator_delivers_tail_frames_to_active_inference():
    from app.actual import actual_events
    from app.analysis import VideoEvent
    from test_tracking import board_image
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    final = play(before, 'red_rook_0', (0, 7))
    current = [before]
    visited = []

    def anchor(time_s, image, reference):
        visited.append(time_s)
        after = first if time_s < 1 else final
        current[0] = after
        return VideoEvent(time_s, [image], position=after, image_times_s=(time_s,))

    frames = [(0, board_image(before)), (.4, board_image(first)), (.8, board_image(first)),
              (1.2, board_image(final)), (1.6, board_image(final))]
    events = list(actual_events(iter(frames), before, confirmed_position=lambda: current[0],
        reanchor_reader=anchor, inference_active=lambda: True, selected_end_s=1))
    assert visited == [.4, .8, 1.2, 1.6]
    assert board_signature(events[-1].position) == board_signature(final)


@pytest.mark.parametrize('source_state,extends', [('occupied', False), ('unknown', False), ('departed', True), ('covered_after_departure', True), ('returned_after_cutoff', False)])
def test_actual_tail_requires_direct_departure_before_selection_end(tmp_path, monkeypatch, source_state, extends):
    import numpy as np
    from types import SimpleNamespace
    from app import actual, fusion, tracking_recovery
    from app.move_inference import MoveInference
    from app.schemas import Calibration, MoveRecord, VideoMetadata
    from app.storage import TaskStore
    before = opening_position()
    first = play(before, 'red_rook_0', (0, 8))
    final = play(before, 'red_rook_0', (0, 7))
    move = next(m for m in legal_moves(before) if m.piece_id == 'red_rook_0' and m.target.rank == 8)
    record = MoveRecord(index=1, move=move, status='tentative', position_before=before,
        predicted_position=first, video_start_s=0, video_end_s=.8, action_end_s=.8)
    tail_calls = []
    def observe(self, time_s, cells, *, tail_only=False, identities=None):
        result = record.model_copy(deep=True)
        if tail_only:
            tail_calls.append(time_s)
            result.move.target.rank = 7
            result.predicted_position = final
            result.video_end_s = result.action_end_s = time_s
        return [result]
    class WeakRecovery:
        def __init__(self, reader):
            self.requested, self.next_check_s, self.attempts, self.check_count = True, 0, 1, 0
            self.reason = '低置信'
        def request(self, time_s):
            self.requested = True
        def observe(self, time_s, *a, **k):
            self.attempts += 1
            self.next_check_s = time_s + 10
    monkeypatch.setenv('VISION_FUSION', '1')
    monkeypatch.setattr(fusion, 'local_fusion_classifier', lambda: SimpleNamespace(detector=SimpleNamespace(readings=lambda image: {})))
    monkeypatch.setattr(tracking_recovery, 'VisualRecovery', WeakRecovery)
    monkeypatch.setattr(MoveInference, 'observe', observe)
    def cells(image):
        time_s = image[0, 0, 0]
        result = occupied(before if source_state == 'returned_after_cutoff' and time_s > 1
            else final if time_s > 1 else first if extends or source_state == 'returned_after_cutoff' else before)
        if source_state == 'unknown' or (source_state == 'covered_after_departure' and time_s >= .8):
            result.pop((0, 9))
        if time_s <= 1:
            result.pop((0, 8))
        return result
    monkeypatch.setattr(actual, '_tracking_cells', cells)
    def events(boards, position, confirmed_position, **kwargs):
        for time_s in [0, .4, .8, 1.2, 1.6]:
            event = kwargs['reanchor_reader'](time_s, np.full((20, 20, 3), time_s), confirmed_position())
            if event is not None:
                if event.selection_complete:
                    break
                yield event
    monkeypatch.setattr(actual, 'actual_events', events)
    store = TaskStore(tmp_path)
    task = store.create('tail.mp4', VideoMetadata(width=901, height=1001, duration_s=2, codec='h264'))
    task.initial_position, task.end_s = before, 1
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed, task.rules_enabled, task.replay_mode, task.status = True, True, 'actual', 'analyzing'
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'source.mp4').write_bytes(b'test')
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.error is None
    assert bool(tail_calls) == extends
    assert result.moves[-1].move.target.rank == (7 if extends else 8)


def test_unexpected_runtime_failure_keeps_safe_diagnostics(tmp_path, monkeypatch):
    import json
    from app import actual
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    store = TaskStore(tmp_path)
    task = store.create('failure.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.initial_position = opening_position()
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
        {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.replay_mode, task.rules_enabled = 'actual', True
    task = store.save(task, task.revision)
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('do-not-log-this-message')))
    actual.analyze_actual(task, store, None)
    assert store.get(task.id).status == 'failed'
    diagnostic = (store.task_dir(task.id) / 'analysis-error.json').read_text(encoding='utf8')
    assert json.loads(diagnostic)['exception_type'] == 'RuntimeError'
    assert 'do-not-log-this-message' not in diagnostic


def test_inferred_snapshot_is_preserved_by_correction_and_recording():
    from app.schemas import MoveRecord
    from app.move_inference import MoveInference
    from app.turns import replay_corrections
    from app.recording import build_recording_playlist
    from test_tracking_reanchor import task_for
    before = opening_position()
    observed = before.model_copy(deep=True)
    observed.pieces = [p for p in observed.pieces if p.id != 'black_rook_0']
    next(p for p in observed.pieces if p.id == 'black_horse_1').id = 'recovered_horse'
    observed.side_to_move = 'black'
    anchor = MoveRecord(index=1, video_start_s=0, video_end_s=2, action_end_s=2,
        status='tentative', change_type='review', position_before=before,
        predicted_position=observed, tracking_anchor=observed, tracking_anchor_s=2)
    after = play(observed, 'recovered_horse', (2, 2))
    following = MoveInference(observed, 2, origin_uncertain=True).observe(3, occupied(after))[0]
    following.index = following.move.index = 2
    following.depends_on_steps = [1]
    task = task_for(before)
    task.moves = [anchor, following]
    corrected = replay_corrections(task, 1)
    assert board_signature(corrected.moves[1].predicted_position) == board_signature(after)
    assert all(r.status == 'tentative' and r.position_after is None for r in corrected.moves)
    playlist = build_recording_playlist(corrected, 1)
    assert len(playlist['records']) == 2
    assert len(playlist['records'][-1]['position_after']['pieces']) == 31
    assert corrected.moves[0].position_after is None


def test_manually_replacing_inferred_snapshot_updates_its_anchor_at_new_time():
    from app.schemas import MoveRecord
    from app.turns import correct_turn
    from test_tracking_reanchor import task_for
    before = opening_position()
    observed = play(before, 'red_rook_0', (0, 8))
    task = task_for(before)
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=2, action_end_s=2,
        status='tentative', change_type='review', position_before=before,
        predicted_position=observed, tracking_anchor=observed, tracking_anchor_s=2)]
    corrected_position = play(before, 'red_rook_0', (0, 7))
    result = correct_turn(task, 1, [], time_s=5, position=corrected_position)
    assert board_signature(result.moves[0].tracking_anchor) == board_signature(corrected_position)
    assert result.moves[0].tracking_anchor_s == 5


@pytest.mark.parametrize('resume,end_s,extra_conflict', [(False, 5, False), (True, 5, False), (False, 1, False), (False, 5, True), (True, 5, True)])
def test_actual_reader_persists_uncertain_snapshot_and_continues_after_restart(tmp_path, monkeypatch, resume, end_s, extra_conflict):
    import numpy as np
    from types import SimpleNamespace
    from app import actual, fusion, tracking_recovery
    from app.schemas import Calibration, MoveRecord, VideoMetadata
    from app.storage import TaskStore
    before = opening_position()
    observed = before.model_copy(deep=True)
    observed.pieces = [p for p in observed.pieces if p.id != 'black_rook_0']
    black = observed.model_copy(update={'side_to_move': 'black'})
    first = play(black, 'black_horse_1', (2, 2))
    reply = play(first, 'red_horse_1', (2, 7))
    classifier_calls = []
    def recognize(*a, **k):
        classifier_calls.append(1)
        return [observed]
    classifier = SimpleNamespace(detector=SimpleNamespace(readings=lambda image: {}),
        recognize_position=recognize, last_response={})
    extra = observed.model_copy(deep=True)
    extra.pieces = [p for p in extra.pieces if p.id != 'black_rook_8']

    class WeakRecovery:
        def __init__(self, reader):
            self.requested, self.next_check_s, self.attempts, self.check_count = True, 0, 1, 0
            self.reason = '测试低置信'
        def request(self, time_s):
            self.requested = True
        def observe(self, time_s, *a, **k):
            self.attempts += 1
            self.check_count += 1
            self.next_check_s = time_s + 1
        def reset(self):
            self.requested = False
    monkeypatch.setenv('VISION_FUSION', '1')
    monkeypatch.setattr(fusion, 'local_fusion_classifier', lambda: classifier)
    monkeypatch.setattr(tracking_recovery, 'VisualRecovery', WeakRecovery)
    monkeypatch.setattr(actual, '_tracking_cells', lambda image, *a, **k:
        occupied([observed, first, reply, extra][int(image[0, 0, 0])]))

    def events(boards, position, confirmed_position, **kwargs):
        reader = kwargs['reanchor_reader']
        for time_s, state in [(0, 0), (.4, 0), (.8, 0), (1.2, 0), (1.6, 0), (2, 0),
                              (2.4, 0), (2.8, 1), (3.2, 1), (3.6, 1), (4, 2), (4.4, 2)]:
            if extra_conflict and time_s >= 2.4:
                state = 3
            event = reader(time_s, np.full((20, 20, 3), state, np.uint8), confirmed_position())
            if event is not None:
                if event.selection_complete:
                    break
                yield event
    monkeypatch.setattr(actual, 'actual_events', events)
    store = TaskStore(tmp_path)
    task = store.create('recovery.mp4', VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'))
    task.initial_position = before
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
        {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.replay_mode, task.rules_enabled, task.status = 'actual', True, 'analyzing'
    task.calibration_confirmed = True
    task.end_s = end_s
    if resume:
        task.context_position, task.resume_video_s = observed, 0
        task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=0, status='tentative',
            change_type='review', predicted_position=observed, tracking_anchor=observed,
            tracking_anchor_s=0, position_before=before)]
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'source.mp4').write_bytes(b'test')
    actual.analyze_actual(task, store, None)
    result = TaskStore(tmp_path).get(task.id)
    assert result.error is None
    snapshots = [r for r in result.moves if r.move is None and r.predicted_position is not None]
    if end_s == 1:
        assert snapshots == []
        assert result.status == 'needs_review'
        return
    assert len(snapshots) == 1
    assert board_signature(snapshots[0].predicted_position) == board_signature(observed)
    if extra_conflict:
        assert len(classifier_calls) == (0 if resume else 1)
        return
    assert result.moves[-1].move.color == 'red'
    assert board_signature(result.moves[-1].predicted_position) == board_signature(reply)
    assert all(r.status in ('tentative', 'needs_review') and r.position_after is None for r in result.moves)
    assert snapshots[0].index in result.moves[-1].depends_on_steps
