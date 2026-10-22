from app.chess import apply_move, legal_moves
from app.move_inference import MoveInference
from app.schemas import opening_position


def cells(position):
    result = {(f, r): None for f in range(9) for r in range(10)}
    result.update({(p.square.file, p.square.rank): p.color for p in position.pieces})
    return result


def play(position, identity, target):
    return apply_move(position, next(m for m in legal_moves(position) if m.piece_id == identity
                                    and (m.target.file, m.target.rank) == target))


def test_visible_piece_still_on_its_source_cannot_be_inferred_elsewhere():
    before = opening_position()
    observed = cells(before)
    observed[4, 7] = 'red'
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    assert inference.observe(1, observed) == []
    assert list(inference._successors(before, observed)) == []


def test_clearly_empty_target_is_not_a_current_landing_candidate():
    before=opening_position()
    observed=cells(before)
    observed[4,6]=None
    inference=MoveInference(before,0)
    assert not any(move.piece_id=='red_pawn_4'for move,_ in inference._successors(before,observed))


def test_unobserved_round_trip_is_not_invented_to_make_turn_order_fit():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    after = play(before.model_copy(update={'side_to_move': 'red'}), 'red_pawn_2', (2, 5))
    observed = cells(after)
    observed.pop((1, 2))
    inference = MoveInference(before, 0)
    candidates = list(inference._bridge(10, observed))
    assert not any(len([r for r in candidate.records if r.move.piece_id == 'black_cannon_1']) > 1
                   for candidate in candidates)


def test_inference_records_save_actual_before_after_board_comparison():
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    record = inference.observe(1, cells(after))[0]
    assert record.board_check is not None
    assert record.board_check.status == 'verified'
    assert record.board_check.visible_before == record.board_check.visible_after == 90


def test_later_clear_frames_do_not_confirm_a_move_with_no_before_evidence():
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    reply = play(after, 'black_horse_1', (2, 2))
    inference = MoveInference(before, 0)
    inference.observe(1, cells(after))
    inference.observe(2, cells(after))
    result = inference.observe(3, cells(reply))
    assert result[0].status == 'tentative'
    assert result[0].board_check is not None
    assert result[0].board_check.status != 'verified'


def test_ordinary_beam_cannot_seal_a_false_round_trip_around_a_real_move():
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    inference = MoveInference(before, 0)
    inference.observe(0, cells(before))
    misleading = cells(before)
    misleading.pop((1, 2))
    misleading[1, 3] = 'black'
    inference.observe(1, misleading)
    inference.observe(2, misleading)
    after = play(before.model_copy(update={'side_to_move': 'red'}), 'red_pawn_2', (2, 5))
    misleading[2, 6], misleading[2, 5] = None, 'red'
    inference.observe(3, misleading)
    inference.observe(4, misleading)
    inference.observe(5, cells(after))
    result = inference.observe(6, cells(after))
    cannons = [r for r in result if r.move.piece_id == 'black_cannon_1']
    assert len(cannons) < 2
    assert not any(r.board_check.status == 'conflict' for r in inference.committed)


def test_recovery_entry_orders_recent_frames_before_a_newer_reference_image():
    before = opening_position()
    after = play(before, 'red_horse_1', (2, 7))
    inference = MoveInference(before, 0, samples=[(3,cells(after)),(0,cells(before)),
                                                (1,cells(before)),(2,cells(after))])
    record = inference.observe(4,cells(after))[0]
    assert record.timing_status == 'observed'
    assert record.action_start_s == 1
    assert record.action_end_s == 2
    assert record.board_check.status == 'verified'


def test_observed_landing_history_survives_later_occlusion_and_capture():
    import json
    from pathlib import Path
    from app.schemas import Position
    trace=json.loads((Path(__file__).parent/'fixtures/inference_landing_history.json').read_text())
    inference=MoveInference(Position.model_validate(trace['before']),trace['start'])
    colors={'.':None,'r':'red','b':'black'}
    for row in trace['events']:
        observed={(i%9,i//9):colors[c]for i,c in enumerate(row['cells'])if c in colors}
        if 'sample'in row:
            inference.sample(row['t'],observed)
        else:
            identities={tuple(map(int,key.split(','))):value for key,value in row['identities'].items()}
            result=inference.observe(row['t'],observed,identities=identities)
    assert [(r.move.piece_id,(r.move.target.file,r.move.target.rank))for r in result]==[
        ('black_rook_8',(7,9)),('red_pawn_4',(4,4)),('black_rook_8',(6,9)),('red_pawn_4',(4,3))]
