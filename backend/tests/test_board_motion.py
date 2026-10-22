import cv2
import numpy as np
import pytest

from app.board_motion import BoardMotion
from app.schemas import Calibration


def scene(orientation='normal'):
    gray = np.random.default_rng(17).integers(30, 240, (480, 640), dtype=np.uint8)
    image = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    calibration = Calibration(corners=[{'x': 90, 'y': 80}, {'x': 550, 'y': 80},
                                      {'x': 550, 'y': 400}, {'x': 90, 'y': 400}], orientation=orientation)
    return image, calibration


@pytest.mark.parametrize('orientation', ['normal', 'rotated_180', 'mirrored', 'mirrored_rotated_180'])
def test_global_motion_preserves_orientation_and_does_not_mutate_user_corners(orientation):
    image, calibration = scene(orientation)
    tracker = BoardMotion(calibration)
    assert tracker.update(0, image) == calibration
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, -6]]), (640, 480))
    result = tracker.update(1, shifted)
    expected = np.float32([[p.x + 9, p.y - 6] for p in calibration.corners])
    np.testing.assert_allclose([[p.x, p.y] for p in result.corners], expected, atol=.4)
    assert result.orientation == orientation
    assert calibration.corners[0].x == 90


def test_stationary_board_and_moving_piece_keep_identical_calibration():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    changed = image.copy()
    changed[170:235, 210:275] = image[250:315, 300:365]
    for n in range(1, 31):
        assert tracker.update(n, changed if n % 2 else image) is calibration


def test_large_occlusion_and_local_feature_cluster_cannot_shift_board():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    covered = np.zeros_like(image)
    covered[100:220, 110:240] = image[100:220, 110:240]
    assert tracker.update(1, covered) is calibration
    assert tracker.update(2, image) is calibration
    blank = BoardMotion(calibration)
    assert blank.update(0, np.zeros_like(image)) is calibration
    assert blank.update(1, np.zeros_like(image)) is calibration


def test_independent_analyses_do_not_share_motion_and_subsampled_frames_keep_their_mapping():
    image, calibration = scene()
    a, b = BoardMotion(calibration), BoardMotion(calibration)
    a.update(0, image)
    b.update(0, image)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, -6]]), (640, 480))
    result = a.update(1, shifted)
    assert result != calibration
    assert a.update(1.01, image) is result
    assert b.update(1, image) is calibration


def test_small_continuous_motion_accumulates_until_it_can_be_applied():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    for n in range(1, 101):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, n * .125], [0, 1, 0]]), (640, 480))
        result = tracker.update(n * .3, shifted)
    np.testing.assert_allclose([p.x for p in result.corners], [p.x + 12.5 for p in calibration.corners], atol=1)


def test_small_picture_jitter_does_not_activate_board_coordinate_changes():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    for n in range(1, 12):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 2 if n % 2 else -2], [0, 1, 0]]), (640, 480))
        assert tracker.update(n, shifted) is calibration


def test_active_tracking_keeps_the_same_mapping_after_a_restart():
    image, calibration = scene()
    first = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, 0]]), (640, 480))
    second = cv2.warpAffine(image, np.float32([[1, 0, 14], [0, 1, 0]]), (640, 480))
    continuous = BoardMotion(calibration)
    continuous.update(0, image)
    saved = continuous.update(1, first)
    restored = BoardMotion(saved, active=continuous.active)
    restored.update(1, first)
    uninterrupted = continuous.update(2, second)
    resumed = restored.update(2, second)
    np.testing.assert_allclose([[p.x, p.y] for p in resumed.corners], [[p.x, p.y] for p in uninterrupted.corners], atol=1)


def test_inactive_tracking_preserves_the_reference_frame_for_accumulated_motion():
    image, calibration = scene()
    frames = {n: cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480))
              for n, dx in ((0, 0), (1, 4), (2, 8))}
    continuous = BoardMotion(calibration)
    continuous.update(0, frames[0])
    saved = continuous.update(1, frames[1])
    assert continuous.active is False
    restored = BoardMotion(saved, active=continuous.active,
                          reference_calibration=continuous.reference_calibration)
    restored.update(continuous.reference_s, frames[continuous.reference_s])
    restored.update(1, frames[1])
    resumed, uninterrupted = restored.update(2, frames[2]), continuous.update(2, frames[2])
    np.testing.assert_allclose([[p.x, p.y] for p in resumed.corners], [[p.x, p.y] for p in uninterrupted.corners], atol=1)


def test_recent_background_survives_many_piece_changes_before_camera_motion():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    current = image.copy()
    texture = np.random.default_rng(39).integers(30, 240, image.shape, dtype=np.uint8)
    for n in range(48):
        x, y = 90 + (n % 8) * 58, 80 + (n // 8) * 54
        current[y:min(y+54, 400), x:min(x+58, 550)] = texture[y:min(y+54, 400), x:min(x+58, 550)]
        assert tracker.update(n + 1, current) is calibration
    shifted = cv2.warpAffine(current, np.float32([[1, 0, 12], [0, 1, -8]]), (640, 480))
    result = tracker.update(49, shifted)
    np.testing.assert_allclose([[p.x, p.y] for p in result.corners],
                               [[p.x + 12, p.y - 8] for p in calibration.corners], atol=.8)


def test_recent_reference_preserves_very_slow_accumulated_motion():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    for n in range(1, 161):
        changed = cv2.warpAffine(image, np.float32([[1, 0, n * .08], [0, 1, 0]]), (640, 480))
        result = tracker.update(n * .3, changed)
    np.testing.assert_allclose([p.x for p in result.corners], [p.x + 12.8 for p in calibration.corners], atol=1)


def test_large_camera_displacement_can_relocalize_from_features():
    image, calibration = scene()
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 38], [0, 1, -30]]), (640, 480))
    result = tracker.update(1, shifted)
    np.testing.assert_allclose([[p.x, p.y] for p in result.corners],
                               [[p.x + 38, p.y - 30] for p in calibration.corners], atol=1)


def test_textured_foreground_moving_over_a_fixed_grid_cannot_move_its_coordinates():
    _, calibration = scene()
    background = np.full((480, 640, 3), 210, np.uint8)
    for x in np.linspace(90, 550, 9).astype(int):
        cv2.line(background, (x, 80), (x, 400), (40, 40, 40), 2)
    for y in np.linspace(80, 400, 10).astype(int):
        cv2.line(background, (90, y), (550, y), (40, 40, 40), 2)
    texture = np.random.default_rng(8).integers(0, 255, (270, 370, 3), dtype=np.uint8)
    first, second = background.copy(), background.copy()
    first[105:375, 130:500] = texture
    second[105:375, 140:510] = texture
    tracker = BoardMotion(calibration)
    tracker.update(0, first)
    assert tracker.update(1, second) is calibration
    for n in range(2, 5):
        assert tracker.update(n, background) is calibration
