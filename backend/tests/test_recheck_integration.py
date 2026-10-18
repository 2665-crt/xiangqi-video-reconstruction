import pytest

from app.actual import snapshot_record
from app.chess import board_signature
from app.schemas import MoveRecord, RecheckState
from app.storage import TaskStore
from test_range_recheck import fixture_task


def recovery(task, positions, *, partial):
    records = [snapshot_record(positions[1], positions[2], 1, 1, 3),
               snapshot_record(positions[2], positions[3], 2, 3, 4)]
    for r in records:
        r.action_start_s, r.action_end_s, r.timing_status = r.video_end_s - .5, r.video_end_s, 'observed'
    if partial:
        records.insert(0, MoveRecord(index=1, video_start_s=1, video_end_s=2, status='needs_review',
                                    change_type='review', reason='尚未解释的画面变化'))
    for i, record in enumerate(records, 1):
        record.index = i
        if record.move:
            record.move.index = i
    task.recheck = RecheckState(id='job', status='needs_review' if partial else 'ready', left_index=1,
        right_index=2, start_s=1, end_s=6, processed_s=6, records=records,
        endpoint_matches=not partial, unresolved=int(partial))
    return task


def test_partial_recheck_writes_confirmed_moves_to_steps_and_keeps_review_separate(tmp_path):
    from app.recheck import integrate_recheck
    from app.records import movement_records
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    fixed = integrate_recheck(task)
    assert [r.move.piece_id for r in movement_records(fixed)] == [
        'red_pawn_0', 'black_pawn_0', 'red_pawn_2', 'black_pawn_2']
    assert fixed.recheck.integrated is True
    assert fixed.recheck.added == 2
    assert fixed.recheck.unresolved == 1
    assert fixed.recheck.status == 'needs_review'
    assert fixed.moves[-1].move.index == len(fixed.moves)
    assert board_signature(fixed.moves[-1].position_after) == board_signature(positions[4])
    assert any(r.status == 'needs_review' and r.move is None for r in fixed.moves)


def test_applying_same_completed_recheck_twice_does_not_duplicate_steps(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    fixed = integrate_recheck(recovery(task, positions, partial=True))
    again = integrate_recheck(fixed)
    assert again.moves == fixed.moves
    assert again.recheck.added == 2


def test_partial_recheck_does_not_delete_existing_steps_in_unknown_spans(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    existing = snapshot_record(positions[1], positions[2], 2, 1, 3)
    existing.action_start_s, existing.action_end_s, existing.timing_status = 2.5, 3, 'observed'
    task.moves.insert(1, existing)
    task.moves[-1].index = 3
    task.recheck.right_index = 3
    fixed = integrate_recheck(task)
    assert len([r for r in fixed.moves if r.move and r.move.piece_id == 'black_pawn_0']) == 1
    assert fixed.recheck.added == 1


def test_zero_confirmed_recovery_does_not_erase_original_interval(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    task.recheck.records = [task.recheck.records[0]]
    fixed = integrate_recheck(task)
    assert fixed.moves == task.moves
    assert fixed.recheck.added == 0


def test_fully_verified_recheck_applies_without_an_extra_confirmation_button(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    fixed = integrate_recheck(recovery(task, positions, partial=False))
    assert fixed.recheck.status == 'applied'
    assert fixed.recheck.integrated is True
    assert len(fixed.moves) == 4


def test_unconfirmed_move_is_not_promoted_when_confirmed_steps_are_integrated(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    uncertain = snapshot_record(positions[3], positions[4], 4, 4, 5)
    uncertain.status = 'needs_review'
    uncertain.reason = '落点尚未确认'
    task.recheck.records.append(uncertain)
    task.recheck.unresolved += 1
    fixed = integrate_recheck(task)
    last = next(r for r in fixed.moves if r.reason == '落点尚未确认')
    assert last.status == 'needs_review'
    assert last.move is None
    assert last.candidates[0].piece_id == 'black_pawn_2'


def test_existing_unknown_move_is_not_auto_confirmed_during_partial_merge(tmp_path):
    from app.recheck import integrate_recheck
    from test_actual import relocated
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    uncertain = snapshot_record(positions[1], relocated(positions[1], [('red_rook_0', 0, 8)]), 2, 1, 2)
    uncertain.status, uncertain.reason = 'needs_review', '原有未确认走法'
    task.moves.insert(1, uncertain)
    task.moves[-1].index = 3
    task.recheck.right_index = 3
    fixed = integrate_recheck(task)
    record = next(r for r in fixed.moves if r.reason == '原有未确认走法')
    assert record.status == 'needs_review'
    assert record.move is None
    assert record.candidates[0].piece_id == 'red_rook_0'


def test_candidate_relations_use_local_mapping_even_when_replacing_old_record(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    old = task.recheck.records[1].model_copy(deep=True)
    old.index, old.move.index = 2, 2
    old.move.target.rank = 5
    task.moves.insert(1, old)
    task.moves[-1].index = 3
    task.recheck.right_index = 3
    task.recheck.records[1].related_steps = [task.recheck.records[2].index]
    fixed = integrate_recheck(task)
    black = next(r for r in fixed.moves if r.move and r.move.piece_id == 'black_pawn_0')
    red = next(r for r in fixed.moves if r.move and r.move.piece_id == 'red_pawn_2')
    assert black.related_steps == [red.index]


def test_recovery_works_between_any_selected_boundaries(tmp_path):
    from app.recheck import integrate_recheck
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task = recovery(task, positions, partial=True)
    prefix = [MoveRecord(index=i, video_start_s=0, video_end_s=.1 * i, status='corrected',
                        change_type='review', position_before=positions[0], position_after=positions[0]) for i in [1, 2]]
    task.moves = prefix + task.moves
    task.moves[2].index, task.moves[3].index = 3, 4
    task.moves[2].move.index, task.moves[3].move.index = 3, 4
    task.recheck.left_index, task.recheck.right_index = 3, 4
    fixed = integrate_recheck(task)
    assert fixed.moves[:3] == task.moves[:3]
    assert [r.move.piece_id for r in fixed.moves if r.move] == [
        'red_pawn_0', 'black_pawn_0', 'red_pawn_2', 'black_pawn_2']
    assert fixed.recheck.left_index == 3
