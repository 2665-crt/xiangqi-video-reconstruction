import cv2
import pytest
import numpy as np
from pathlib import Path

from app.chess import board_signature
from app.schemas import BoardCheck, Position, Square, opening_position
from app.tracking_recovery import VisualRecovery
from app.turns import TurnReplay
from test_move_evidence import cells, moved
from test_tracking import board_image
from test_tracking_reanchor import readings, task_for


def test_invalid_pending_observation_does_not_become_tracking_reference():
    before = opening_position()
    invalid = moved(before, 'red_cannon_7', 2, 4)
    replay = TurnReplay(task_for(before), before)
    check = BoardCheck(status='partial', visible_before=90, visible_after=89)
    replay.observe(invalid, 2, ['candidate.jpg'], 1, check)
    replay.attach_evidence(check, ['candidate.jpg'], [1.5])
    assert replay.pending == invalid
    assert board_signature(replay.tracking_position) == board_signature(before)
    correct = moved(before, 'red_horse_1', 2, 7)
    replay.observe(correct, 3, ['clear.jpg'], 2, BoardCheck(status='verified', visible_before=90, visible_after=90))
    replay.end(4)
    automatic = [r for r in replay.task.moves if r.status == 'auto_verified']
    assert len(automatic) == 1
    assert automatic[0].move.piece_id == 'red_horse_1'
    assert automatic[0].move.source == Square(file=1, rank=9)
    assert automatic[0].move.target == Square(file=2, rank=7)
    review = next(r for r in replay.task.moves if r.status == 'needs_review' and 'candidate.jpg' in r.evidence_ids)
    assert review.evidence_times_s[review.evidence_ids.index('candidate.jpg')] == 1.5


def test_valid_pending_move_can_support_the_following_opponent_move():
    before = opening_position()
    first = moved(before, 'red_horse_1', 2, 7)
    replay = TurnReplay(task_for(before), before)
    replay.observe(first, 2, ['first.jpg'], 1, BoardCheck(status='verified', visible_before=90, visible_after=90))
    assert board_signature(replay.tracking_position) == board_signature(first)
    second = moved(first, 'black_horse_1', 2, 2)
    replay.observe(second, 4, ['second.jpg'], 3, BoardCheck(status='verified', visible_before=90, visible_after=90))
    replay.end(5)
    assert [r.move.piece_id for r in replay.task.moves] == ['red_horse_1', 'black_horse_1']


def test_incremental_move_does_not_require_reclassifying_known_pieces_or_empty_cells():
    before = opening_position()
    after = moved(before, 'red_horse_1', 2, 7)
    observed = cells(after)
    observed.pop((4, 4))
    weak = readings(after)
    weak[(8, 0)] = {'confidence': .25, 'class_probabilities': {'black_cannon': .999}}
    weak[(2, 7)] = {'confidence': .35, 'class_probabilities': {'red_cannon': .999}}
    tracker = VisualRecovery(lambda image: weak)
    tracker.request(1)
    reference_image = board_image(before)
    assert tracker.observe(1, board_image(after), before, observed, reference_image=reference_image) is None
    recovered = tracker.observe(2, board_image(after), before, observed, reference_image=reference_image)
    assert recovered is not None
    assert board_signature(recovered) == board_signature(after)
    assert next(p for p in recovered.pieces if p.square == Square(file=8, rank=0)).kind == 'rook'
    assert tracker.incremental is True
    assert tracker.board_check.status == 'partial'
    assert Square(file=4, rank=4) in tracker.board_check.unknown_after


@pytest.mark.parametrize('hidden', [(1, 9), (2, 7), (8, 0)])
def test_incremental_recovery_does_not_assume_a_hidden_source_target_or_piece_is_unchanged(hidden):
    before = opening_position()
    after = moved(before, 'red_horse_1', 2, 7)
    observed = cells(after)
    observed.pop(hidden)
    tracker = VisualRecovery(lambda image: readings(after))
    tracker.request(1)
    for second in range(1, 4):
        assert tracker.observe(second, board_image(after), before, observed,
                               reference_image=board_image(before)) is None


def test_weak_wrong_class_can_inherit_only_when_the_known_face_stays_consistent():
    before = opening_position()
    after = moved(before, 'red_horse_1', 2, 7)
    original, final = board_image(before), board_image(after)
    changed = final.copy()
    cv2.circle(changed, (850, 50), 23, (0, 0, 0), -1)
    weak = readings(after)
    weak[(8, 0)] = {'confidence': .25, 'class_probabilities': {'black_cannon': .999}}
    tracker = VisualRecovery(lambda image: weak)
    tracker.request(1)
    for second in range(1, 4):
        assert tracker.observe(second, changed, before, cells(after), reference_image=original) is None


@pytest.mark.parametrize('confidence', [.25, .99])
def test_real_clear_board_reuses_continuous_rook_identity_despite_wrong_frame_class(confidence):
    folder = Path(__file__).parent / 'fixtures/incremental-recovery'
    before = Position.model_validate_json((folder / 'before.json').read_text(encoding='utf8'))
    after = Position.model_validate_json((folder / 'after.json').read_text(encoding='utf8'))
    detected = readings(after)
    detected[(7, 0)] = {'confidence': confidence, 'class_probabilities': {'black_cannon': .999}}
    tracker = VisualRecovery(lambda image: detected)
    tracker.request(180)
    original = cv2.imdecode(np.fromfile(folder / 'board-160.png', np.uint8), cv2.IMREAD_COLOR)
    anchor = None
    for second in (180, 181):
        image = cv2.imdecode(np.fromfile(folder / f'board-{second}.png', np.uint8), cv2.IMREAD_COLOR)
        from app.actual import _tracking_cells
        assert _tracking_cells(image) == cells(after)
        anchor = tracker.observe(second, image, before, cells(after), reference_image=original)
    assert anchor is not None
    assert board_signature(anchor) == board_signature(after)
    assert next(p for p in anchor.pieces if p.square == Square(file=7, rank=0)).kind == 'rook'
    if confidence > .75:
        assert tracker.board_check is not None
        assert 'black_cannon' in tracker.board_check.reason


def test_continuous_strong_identity_conflict_is_retained_for_review_without_silent_type_change():
    actual = opening_position()
    stale = actual.model_copy(deep=True)
    next(p for p in stale.pieces if p.id == 'red_horse_1').kind = 'elephant'
    image = board_image(actual)
    tracker = VisualRecovery(lambda image: readings(actual))
    tracker.request(1)
    assert tracker.observe(1, image, stale, cells(actual), reference_image=image) is None
    anchor = tracker.observe(2, image, stale, cells(actual), reference_image=image)
    assert anchor is not None
    assert board_signature(anchor) == board_signature(stale)
    assert tracker.incremental is False
    assert tracker.board_check.status == 'partial'
    assert 'red_horse' in tracker.board_check.reason and 'red_elephant' in tracker.board_check.reason
    assert [t for t, _ in tracker.evidence] == [1, 2]


def test_piece_face_match_requires_information_inside_both_templates(monkeypatch):
    from app.board_evidence import matching_piece_faces
    position = opening_position()
    blank = np.zeros((16, 16), dtype=np.float32)
    blank[:2] = 1
    blank[-2:] = 1
    blank[:, :2] = 1
    blank[:, -2:] = 1
    monkeypatch.setattr('app.actual._glyphs', lambda image, position: [blank for p in position.pieces])
    assert matching_piece_faces(None, None, position, position) == set()


def test_moving_piece_strong_class_conflict_is_retained_at_its_new_square():
    actual = opening_position()
    stale = actual.model_copy(deep=True)
    next(p for p in stale.pieces if p.id == 'red_rook_0').kind = 'cannon'
    after = moved(actual, 'red_rook_0', 0, 8)
    tracker = VisualRecovery(lambda image: readings(after))
    tracker.request(1)
    assert tracker.observe(1, board_image(after), stale, cells(after), reference_image=board_image(actual)) is None
    anchor = tracker.observe(2, board_image(after), stale, cells(after), reference_image=board_image(actual))
    assert anchor is not None and tracker.incremental is True
    assert next(p for p in anchor.pieces if p.id == 'red_rook_0').kind == 'cannon'
    assert tracker.board_check.status == 'partial'
    assert 'red_rook' in tracker.board_check.reason and 'red_cannon' in tracker.board_check.reason


@pytest.mark.parametrize('other_move', [False, True])
def test_piece_in_hand_does_not_disappear_from_recovery_anchor(other_move):
    before = opening_position()
    lifted = before.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'red_elephant_6']
    if other_move:
        lifted = moved(lifted, 'black_horse_1', 2, 2)
    tracker = VisualRecovery(lambda image: readings(lifted))
    tracker.request(1)
    for second in (1, 2, 3):
        assert tracker.observe(second, board_image(lifted), before, cells(lifted),
                               reference_image=board_image(before)) is None
    landed = moved(before, 'red_elephant_6', 4, 7)
    tracker.reader = lambda image: readings(landed)
    assert tracker.observe(4, board_image(landed), before, cells(landed), reference_image=board_image(before)) is None
    anchor = tracker.observe(5, board_image(landed), before, cells(landed), reference_image=board_image(before))
    assert anchor is not None and len(anchor.pieces) == len(before.pieces)
    assert board_signature(anchor) == board_signature(landed)


def test_full_recovery_accepts_capture_with_two_visible_opponent_occupancy_witnesses():
    before = moved(moved(opening_position(), 'red_pawn_4', 4, 5), 'black_pawn_4', 4, 4)
    after = before.model_copy(deep=True)
    after.pieces = [p for p in after.pieces if p.id != 'black_pawn_4']
    after = moved(moved(after, 'red_pawn_4', 4, 4), 'black_horse_1', 2, 2)
    tracker = VisualRecovery(lambda image: readings(after))
    tracker.request(1)
    assert tracker.observe(1, board_image(after), before, cells(after), reference_image=board_image(before)) is None
    anchor = tracker.observe(2, board_image(after), before, cells(after), reference_image=board_image(before))
    assert anchor is not None
    assert board_signature(anchor) == board_signature(after)


def test_another_surviving_rooks_old_square_does_not_prove_the_lifted_rook_was_captured():
    before = opening_position()
    before.pieces = [p for p in before.pieces if p.id not in ('black_pawn_8', 'red_pawn_8', 'red_horse_7')]
    after = moved(moved(before, 'red_rook_8', 7, 9), 'black_rook_8', 8, 9)
    after.pieces = [p for p in after.pieces if p.id != 'red_rook_0']
    tracker = VisualRecovery(lambda image: readings(after))
    tracker.request(1)
    for second in (1, 2, 3):
        assert tracker.observe(second, board_image(after), before, cells(after), reference_image=board_image(before)) is None


@pytest.mark.parametrize('sliding', [False, True])
@pytest.mark.parametrize('different_kind', [False, True])
def test_static_identity_audit_preserves_known_turn_and_current_turn_merging(tmp_path, monkeypatch, sliding, different_kind):
    from app import actual
    from app.analysis import VideoEvent, analyze_task
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8) if sliding else before
    check = BoardCheck(status='partial', visible_before=90, visible_after=90, reason='连续高置信分类与历史身份冲突')
    events = [VideoEvent(0, [board_image(before)], position=before)]
    if sliding:
        events.append(VideoEvent(1, [board_image(before), board_image(first)], position=first, action_start_s=.5))
    alternative = first.model_copy(deep=True)
    if different_kind:
        next(p for p in alternative.pieces if p.id == 'red_rook_8').kind = 'cannon'
    events.append(VideoEvent(2, [board_image(first)], position=alternative, image_times_s=(2,),
                             tracking_anchor=True, board_check=check))
    if sliding:
        final = moved(first, 'red_rook_0', 0, 7)
        events.append(VideoEvent(3, [board_image(first), board_image(final)], position=final, action_start_s=2.5))
        reply = moved(final, 'black_horse_1', 2, 2)
        events.append(VideoEvent(4, [board_image(final), board_image(reply)], position=reply, action_start_s=3.5))
    else:
        wrong = moved(before, 'black_horse_1', 2, 2)
        events.append(VideoEvent(4, [board_image(before), board_image(wrong)], position=wrong, action_start_s=3))
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('identity.mp4', VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'))
    task.initial_position, task.initial_turn_known = before, True
    task.end_s, task.replay_mode, task.rules_enabled = 5, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.error is None, result.error
    assert result.initial_position.side_to_move == 'red'
    assert any('身份冲突' in r.reason and r.status == 'needs_review' for r in result.moves)
    if sliding:
        red = [r for r in result.moves if r.move is not None and r.move.color == 'red']
        assert len(red) == 1 and red[0].status == 'auto_verified'
        assert red[0].move.source == Square(file=0, rank=9)
        assert red[0].move.target == Square(file=0, rank=7)
    else:
        assert any(r.violation == 'turn_mismatch' for r in result.moves)
        assert not any(r.move and r.move.color == 'black' and r.status == 'auto_verified' for r in result.moves)


def test_independent_kind_only_recovery_is_an_identity_audit_without_a_movement_gap(monkeypatch):
    actual = opening_position()
    stale = actual.model_copy(deep=True)
    next(p for p in stale.pieces if p.id == 'red_horse_1').kind = 'elephant'
    monkeypatch.setattr('app.board_evidence.matching_piece_faces',
                        lambda *a: {p.id for p in stale.pieces if p.id != 'red_horse_1'})
    tracker = VisualRecovery(lambda image: readings(actual))
    tracker.request(1)
    image = board_image(actual)
    assert tracker.observe(1, image, stale, cells(actual), reference_image=image) is None
    candidate = tracker.observe(2, image, stale, cells(actual), reference_image=image)
    assert candidate is not None
    assert tracker.board_check is not None and tracker.board_check.status == 'partial'
    assert 'red_elephant' in tracker.board_check.reason and 'red_horse' in tracker.board_check.reason


def test_real_changed_piece_face_cannot_inherit_previous_identity_from_position_alone():
    folder = Path(__file__).parent / 'fixtures/incremental-recovery'
    before = Position.model_validate_json((folder / 'before.json').read_text(encoding='utf8'))
    after = Position.model_validate_json((folder / 'after.json').read_text(encoding='utf8'))
    original = cv2.imdecode(np.fromfile(folder / 'board-160.png', np.uint8), cv2.IMREAD_COLOR)
    current = cv2.imdecode(np.fromfile(folder / 'board-180.png', np.uint8), cv2.IMREAD_COLOR)
    current[55:145, 755:845] = original[255:345, 755:845]
    detected = readings(after)
    detected[(7, 0)] = {'confidence': .25, 'class_probabilities': {'black_cannon': .999}}
    tracker = VisualRecovery(lambda image: detected)
    tracker.request(180)
    for second in (180, 181, 182):
        assert tracker.observe(second, current, before, cells(after), reference_image=original) is None
