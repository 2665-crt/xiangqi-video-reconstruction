import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.actual import actual_events, snapshot_record, _tracking_cells
from app.schemas import Position, opening_position
from test_tracking import board_image


FIXTURES = Path(__file__).parent / 'fixtures' / 'overlapping-advisors'


@pytest.mark.parametrize('prefetched', [False, True])
def test_white_sleeve_without_piece_lettering_is_not_a_black_pawn_landing(prefetched):
    from collections import OrderedDict
    import weakref
    import app.actual as actual
    from app.analysis import read_board_cells
    before = Position.model_validate_json((FIXTURES / 'before-white-sleeve.json').read_text(encoding='utf8'))
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'white-sleeve-false-pawn.jpg').read_bytes(), np.uint8), 1)
    cache = None
    if prefetched:
        raw = read_board_cells(actual._tracking_image(image), strict=True)
        prepared = actual._finish_tracking_cells(image, raw, True)
        assert prepared[(8, 4)] == 'black'
        image.setflags(write=False)
        cache = actual._tracking_cache.set(OrderedDict({(id(image), 'prepared'): (weakref.ref(image), (raw, prepared, prepared))}))
    try:
        assert _tracking_cells(image, before).get((8, 4)) != 'black'
    finally:
        if cache is not None:
            actual._tracking_cache.reset(cache)


def test_fingers_on_landing_black_horse_cannot_move_a_distant_red_rook():
    before = Position.model_validate_json((FIXTURES / 'before-held-horse.json').read_text(encoding='utf8'))
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'held-black-horse.jpg').read_bytes(), np.uint8), 1)
    assert _tracking_cells(image, before).get((3, 5)) != 'red'


def test_real_black_horse_landing_preserves_red_rook_and_continues_sequence():
    before = Position.model_validate_json((FIXTURES / 'before-held-horse.json').read_text(encoding='utf8'))
    reference = [before]
    frames = [(item['time_s'], cv2.imdecode(np.frombuffer((FIXTURES / item['file']).read_bytes(), np.uint8), 1))
              for item in json.loads((FIXTURES / 'horse-frames.json').read_text(encoding='utf8'))]
    actions = []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: reference[0], turn_boundaries=True):
        if (event.position is not None and not event.review_reason and not event.moving_colors
                and (event.board_check is None or event.board_check.status != 'conflict')):
            action = snapshot_record(reference[0], event.position, 1, 0, event.time_s).move
            if action is not None:
                actions.append((action.piece_id, action.target.file, action.target.rank))
            reference[0] = event.position
    assert actions == [('black_horse_1', 3, 5)]
    assert next(p for p in reference[0].pieces if p.id == 'red_rook_0').square.model_dump() == {'file': 3, 'rank': 1}


def test_prefetched_hand_mask_does_not_reuse_a_previous_position(monkeypatch):
    from collections import OrderedDict
    import weakref
    import app.actual as actual
    from app.schemas import Square
    from test_move_evidence import cells
    before = opening_position()
    after = before.model_copy(deep=True)
    after.pieces = [p for p in after.pieces if p.id != 'black_rook_0']
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=0)
    image = board_image(after)
    monkeypatch.setattr(actual, 'rectify_board', lambda *args, **kwargs: image)
    monkeypatch.setattr('app.analysis.read_board_cells', lambda *args, **kwargs: cells(after))
    monkeypatch.setattr('app.board_colors.skin_surface_mask', lambda hsv: np.ones(hsv.shape[:2], dtype=bool))
    _, prepared, _, variants, _ = actual._prepare_tracking_frame(
        (0, image, 0, {(p.square.file, p.square.rank): p.color for p in before.pieces}), None, None, 1, False)
    expected = actual._read_tracking_cells(image, after)
    assert expected[(0, 0)] == 'red'
    prepared.setflags(write=False)
    token = actual._tracking_cache.set(OrderedDict({(id(prepared), 'prepared'): (weakref.ref(prepared), variants)}))
    try:
        assert actual._tracking_cells(prepared, after) == expected
    finally:
        actual._tracking_cache.reset(token)


def test_occluded_distant_horse_cannot_make_a_visible_horse_move_ambiguous():
    from test_move_evidence import moved, cells
    before = opening_position().model_copy(update={'side_to_move': 'black'})
    first = moved(before, 'black_horse_1', 2, 2)
    reply = moved(first, 'red_horse_1', 2, 7)
    hidden = board_image(first)
    cv2.rectangle(hidden, (700, 0), (800, 120), (20, 20, 20), -1)
    observation = cells(first)
    observation.pop((7, 0))
    initial, final = board_image(before), board_image(reply)
    readings = {id(initial): cells(before), id(hidden): observation, id(final): cells(reply)}
    reference = [before]
    actions = []
    for event in actual_events(iter([(0, initial), (1, hidden), (1.1, hidden), (2, final), (2.1, final)]), before,
                               confirmed_position=lambda: reference[0], turn_boundaries=True,
                               cell_reader=lambda image, _: readings[id(image)]):
        if (event.position is not None and not event.review_reason and not event.moving_colors
                and (event.board_check is None or event.board_check.status != 'conflict')):
            action = snapshot_record(reference[0], event.position, 1, 0, event.time_s).move
            if action is not None:
                actions.append(action.piece_id)
            reference[0] = event.position
    assert actions == ['black_horse_1', 'red_horse_1']


@pytest.mark.parametrize('color_model', ['0', '1'])
def test_simultaneous_advisor_lift_does_not_steal_identity_and_block_red_reply(monkeypatch, color_model):
    monkeypatch.setenv('BOARD_COLOR_MODEL', color_model)
    before = Position.model_validate_json((FIXTURES / 'before.json').read_text(encoding='utf8'))
    reference = [before]
    frames = [(item['time_s'], cv2.imdecode(np.frombuffer((FIXTURES / item['file']).read_bytes(), np.uint8), 1))
              for item in json.loads((FIXTURES / 'frames.json').read_text(encoding='utf8'))]
    actions = []
    for event in actual_events(iter(frames), before, confirmed_position=lambda: reference[0], turn_boundaries=True):
        if event.position is None or event.review_reason or event.moving_colors:
            continue
        if event.board_check is not None and event.board_check.status == 'conflict':
            continue
        action = snapshot_record(reference[0], event.position, 1, 0, event.time_s).move
        if action is not None:
            actions.append((action.piece_id, action.source.file, action.source.rank, action.target.file, action.target.rank))
        reference[0] = event.position
    assert actions == [('black_advisor_5', 5, 0, 4, 1), ('red_horse_7', 6, 7, 5, 5)]
