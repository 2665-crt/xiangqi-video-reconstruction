import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.calibration import refine_from_pieces, rectify_board
from app.actual import _tracking_cells
from app.schemas import Calibration, opening_position


FIXTURES = Path(__file__).parent / 'fixtures' / 'board-colors'


@pytest.mark.parametrize('orientation', ['normal', 'rotated_180', 'mirrored', 'mirrored_rotated_180'])
def test_confident_known_piece_anchors_correct_skewed_tracking_coordinates(orientation):
    image = cv2.imdecode(np.frombuffer((FIXTURES / 'unaligned-initial.jpg').read_bytes(), np.uint8), 1)
    calibration = Calibration(corners=[{'x': 100, 'y': 100}, {'x': 900, 'y': 100},
                                       {'x': 900, 'y': 1000}, {'x': 100, 'y': 1000}], orientation=orientation)
    if orientation in ('rotated_180', 'mirrored_rotated_180'):
        image = cv2.rotate(image, cv2.ROTATE_180)
    if orientation in ('mirrored', 'mirrored_rotated_180'):
        image = cv2.flip(image, 1)
    readings = {ast.literal_eval(s): value for s, value in json.loads((FIXTURES / 'piece-anchors.json').read_text()).items()}
    refined = refine_from_pieces(image, calibration, opening_position(), readings)
    assert refined != calibration
    later = cv2.imdecode(np.frombuffer((FIXTURES / 'second-horse-departure.jpg').read_bytes(), np.uint8), 1)
    if orientation in ('rotated_180', 'mirrored_rotated_180'):
        later = cv2.rotate(later, cv2.ROTATE_180)
    if orientation in ('mirrored', 'mirrored_rotated_180'):
        later = cv2.flip(later, 1)
    cells = _tracking_cells(rectify_board(later, refined, margin=100), opening_position())
    assert cells[(1, 0)] is None
    assert cells[(2, 2)] == 'black'
    assert cells[(5, 2)] == 'black'
    assert calibration.corners[0].x == 100


def test_sparse_or_untrusted_piece_anchors_cannot_change_calibration():
    image = np.zeros((1101, 1001, 3), np.uint8)
    calibration = Calibration(corners=[{'x': 100, 'y': 100}, {'x': 900, 'y': 100},
                                       {'x': 900, 'y': 1000}, {'x': 100, 'y': 1000}])
    readings = {ast.literal_eval(s): value for s, value in json.loads((FIXTURES / 'piece-anchors.json').read_text()).items()}
    assert refine_from_pieces(image, calibration, opening_position(), dict(list(readings.items())[:5])) == calibration
    for reading in readings.values():
        reading['confidence'] = .3
    assert refine_from_pieces(image, calibration, opening_position(), readings) == calibration
