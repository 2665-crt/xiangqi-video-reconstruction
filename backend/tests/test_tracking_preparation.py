from functools import partial
import os

import cv2
import numpy as np
import pytest

from app.board_observation import BoardObservation
from app.piece_tracking import PieceTracker
from app.schemas import Calibration, Square, opening_position
from test_board_observation import face_board, unknown_occupied


@pytest.mark.parametrize('offset', [(-6, -30), (0, 0), (12, 8)])
def test_precomputed_observation_preserves_projection_cells_and_frame_pixels(offset, monkeypatch):
    assert callable(getattr(BoardObservation, 'prepare', None))
    known = opening_position()
    after = known.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_pawn_2').square = Square(file=2, rank=5)
    direct, prepared = BoardObservation(), BoardObservation()
    images = [np.full((551, 501, 3), (95, 170, 205), np.uint8),
              face_board(known, offset), face_board(after, offset, hand=True), face_board(after, offset)]
    hough = cv2.HoughCircles
    for image in images:
        pixels = image.copy()
        observation = BoardObservation.prepare(image)
        expected = direct.update(image, known, unknown_occupied(after))
        calls = []
        with monkeypatch.context() as patch:
            patch.setattr(cv2, 'HoughCircles', lambda *a, **kw: (calls.append(1), hough(*a, **kw))[1])
            actual = prepared.update(image, known, unknown_occupied(after), prepared=observation)
        assert calls == []
        assert actual == expected
        assert prepared.fitted == direct.fitted
        np.testing.assert_array_equal(prepared.matrix, direct.matrix)
        for name in ('small', 'gray', 'hsv', 'edges'):
            np.testing.assert_array_equal(getattr(prepared, name), getattr(direct, name))
        assert prepared.centers.keys() == direct.centers.keys()
        for square, (center, radius) in direct.centers.items():
            np.testing.assert_array_equal(prepared.centers[square][0], center)
            assert prepared.centers[square][1] == radius
        np.testing.assert_array_equal(image, pixels)


def test_precomputed_tracking_preserves_identity_history_occlusion_and_time_gaps():
    assert callable(getattr(BoardObservation, 'prepare', None))
    from test_piece_tracking import blank_pieces, position, unknown_pieces
    direct, prepared = PieceTracker(), PieceTracker()
    for index, x in enumerate([75] * 4 + list(range(85, 226, 10)) + [225] * 7):
        image = blank_pieces([(x, 375), (325, 425), (375, 125)], hidden=index in (8, 9))
        cells = unknown_pieces()
        cells.pop((4, 7))
        if x > 100:
            cells[1, 7] = None
        time_s = index * .04 + (.5 if index >= 19 else 0)
        expected = direct.observe(time_s, image, position(), cells, allow_reseed=index < 7)
        actual = prepared.observe(time_s, image, position(), cells, allow_reseed=index < 7,
                                  prepared=BoardObservation.prepare(image))
        assert actual == expected
        assert prepared.observation.cells == direct.observation.cells
        assert prepared.identities == direct.identities
        assert prepared.tokens == direct.tokens
        assert prepared.departed == direct.departed
        assert prepared.trusted == direct.trusted
        assert prepared.tracks.keys() == direct.tracks.keys()
        for identity, before in direct.tracks.items():
            after = prepared.tracks[identity]
            for name in ('piece', 'radius', 'generation', 'stable_since', 'stable_hits'):
                assert getattr(after, name) == getattr(before, name)
            for name in ('center', 'stable_center', 'appearance', 'template', 'mask', 'points'):
                np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


def prepare_in_worker(process, item):
    return os.getpid(), process(item)


def test_frame_workers_return_observation_without_running_parent_hough(monkeypatch):
    from app.actual import _prepare_tracking_frame
    from app.preprocessing import process_frames
    image = face_board(opening_position())
    calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 450, 'y': 50},
                                       {'x': 450, 'y': 500}, {'x': 50, 'y': 500}])
    process = partial(_prepare_tracking_frame, tracking_calibration=calibration,
                      scan_calibration=calibration, scan_end=1, scan_needed=True)
    source = [(i * .04, image, .001, {}) for i in range(3)]
    with process_frames(iter(source), partial(prepare_in_worker, process), workers=2, capacity=2) as rows:
        results = list(rows)
    assert [row[1][0] for row in results] == [0, .04, .08]
    assert all(pid != os.getpid() for pid, _ in results)
    assert all(len(result[3]) == 4 for _, result in results)
    def unexpected_hough(*args, **kwargs):
        pytest.fail('parent repeated worker circle detection')
    monkeypatch.setattr(cv2, 'HoughCircles', unexpected_hough)
    tracker = PieceTracker()
    for _, (time_s, board, scan, variants, decoding) in results:
        assert scan is not None and decoding == .001
        tracker.observe(time_s, board, opening_position(), variants[1], prepared=variants[3])


def test_piece_geometry_cache_preserves_masks_and_is_readonly_and_bounded():
    import app.piece_tracking as tracking
    assert callable(getattr(tracking, '_piece_masks', None))
    for radius in range(12, 32):
        for padding in (3, 4):
            extent = radius + padding
            yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1]
            distance = xx ** 2 + yy ** 2
            core, ring, mask = tracking._piece_masks(radius, padding)
            np.testing.assert_array_equal(core, distance < (radius * .7) ** 2)
            np.testing.assert_array_equal(ring, (distance >= (radius * .7) ** 2) & (distance <= (radius + 2) ** 2))
            np.testing.assert_array_equal(mask, np.uint8(ring) * 255)
            assert tracking._piece_masks(radius, padding)[0] is core
            for array in (core, ring, mask):
                with pytest.raises(ValueError):
                    array[0, 0] = 1
    assert tracking._piece_masks.cache_info().currsize <= 64
