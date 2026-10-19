from pathlib import Path

import cv2
import numpy as np

from app.actual import _tracking_cells, _landing_visible, _finish_tracking_cells
from app.schemas import opening_position, Square


FIXTURES = Path(__file__).parent / 'fixtures' / 'board-colors'


def test_yellow_board_empty_squares_are_not_skin_occlusions():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'warm-board.png').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    for square in [(6, 5), (6, 2), (3, 2), (4, 4)]:
        assert square in cells and cells[square] is None


def test_real_forearm_is_not_restored_as_an_empty_square():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'hand-occlusion.png').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    assert (4, 6) not in cells
    assert (5, 7) not in cells


def test_pale_fingers_on_black_piece_do_not_change_its_color():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'fingers-on-black-rook.png').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    assert (0, 0) not in cells


def test_yellow_wood_around_landed_horse_is_not_a_finger():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'clear-horse-landing.png').read_bytes(), np.uint8), 1)
    before = opening_position()
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    assert _landing_visible(image, before, after)


def test_departed_black_horse_source_is_visible_as_empty():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'clear-horse-landing.png').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    assert (1, 0) in cells and cells[(1, 0)] is None


def test_visible_black_cannon_is_not_rejected_as_skin():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'clear-horse-landing.png').read_bytes(), np.uint8), 1)
    cells = _finish_tracking_cells(image, {(5, 2): 'black'}, False, {})
    assert cells.get((5, 2)) == 'black'


def test_river_lettering_is_not_a_black_piece_after_horse_returns():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'horse-return-river-text.jpg').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    assert cells.get((5, 5)) is None
    assert cells.get((1, 0)) == 'black'
    assert cells.get((1, 2)) == 'black'


def test_grid_ink_does_not_hide_empty_source_of_second_horse_move():
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'second-horse-departure.jpg').read_bytes(), np.uint8), 1)
    cells = _tracking_cells(image, opening_position())
    assert (1, 0) in cells and cells[(1, 0)] is None
