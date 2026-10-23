import cv2
import numpy as np
import pytest


def radius_reference(edges, center, circle, radius):
    radii = range(14, 31) if radius == 0 else range(max(12, radius - 2), min(32, radius + 3))
    best = None
    for r in radii:
        ring = np.rint(center + circle * r).astype(int)
        if (np.any(ring < 0) or np.any(ring[:, 0] >= edges.shape[1])
                or np.any(ring[:, 1] >= edges.shape[0])):
            continue
        support = np.mean(edges[ring[:, 1], ring[:, 0]] > 0)
        if best is None or support > best[0]:
            best = support, r
    return -1 if best is None or best[0] < .65 else best[1]


@pytest.mark.parametrize('radius', [0, 12, 18, 27, 31])
def test_native_radius_preserves_rounding_boundary_support_and_ties(radius):
    import app.numerical as numerical
    assert callable(getattr(numerical, 'track_radius', None))
    circle = np.stack((np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
                       np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))), axis=1)
    rng = np.random.default_rng(84)
    edges = rng.choice(np.uint8([0, 255]), (91, 103), p=[.2, .8])
    for center in ((50, 45), (50.5, 45.5), (12, 12), (-1, 30), (90, 80),
                   (np.nextafter(np.float32(50.5), np.float32(51)), 45.5)):
        point = np.float32(center)
        assert numerical.track_radius(edges, point, circle, radius) == radius_reference(edges, point, circle, radius)
    full = np.full_like(edges, 255)
    assert numerical.track_radius(full, np.float32([50, 45]), circle, radius) == radius_reference(full, np.float32([50, 45]), circle, radius)
    assert numerical.track_radius(np.zeros_like(edges), np.float32([50, 45]), circle, radius) == -1


def collision_reference(centers, radii):
    result = np.zeros(len(centers), bool)
    for i, center in enumerate(centers):
        for j in range(i + 1, len(centers)):
            if np.linalg.norm(center - centers[j]) < max(radii[i], radii[j]) * 1.5:
                result[i] = result[j] = True
    return result


def test_native_collisions_preserve_float32_threshold_and_empty_tracks():
    import app.numerical as numerical
    assert callable(getattr(numerical, 'track_collisions', None))
    rng = np.random.default_rng(19)
    for count in (0, 1, 2, 16, 32):
        centers = rng.uniform(0, 500, (count, 2)).astype(np.float32)
        radii = rng.integers(12, 32, count, dtype=np.int64)
        np.testing.assert_array_equal(numerical.track_collisions(centers, radii), collision_reference(centers, radii))
    for theta in np.linspace(0, 2 * np.pi, 128):
        edge = np.float32([np.cos(theta) * 27, np.sin(theta) * 27])
        for target in (edge, np.nextafter(edge, np.float32(0)), np.nextafter(edge, np.float32(np.inf))):
            centers = np.stack((np.zeros(2, np.float32), target))
            radii = np.int64([18, 18])
            np.testing.assert_array_equal(numerical.track_collisions(centers, radii), collision_reference(centers, radii))


def test_inverse_projection_reuses_inverse_but_detects_inplace_matrix_updates(monkeypatch):
    from app.board_observation import BoardObservation
    observation = BoardObservation()
    observation.matrix = np.array([[1.01, .02, -4], [.03, .98, 6], [.0001, -.0002, 1]], np.float64)
    points = np.random.default_rng(9).uniform(0, 500, (32, 2)).astype(np.float32)
    inverse = np.linalg.inv
    expected = cv2.perspectiveTransform(points[:, None], inverse(observation.matrix))[:, 0]
    calls = []
    monkeypatch.setattr(np.linalg, 'inv', lambda matrix: (calls.append(1), inverse(matrix))[1])
    for _ in range(4):
        np.testing.assert_array_equal(observation.project(points, inverse=True), expected)
    assert len(calls) == 1
    observation.matrix[0, 2] += 7
    expected = cv2.perspectiveTransform(points[:, None], inverse(observation.matrix))[:, 0]
    np.testing.assert_array_equal(observation.project(points, inverse=True), expected)
    assert len(calls) == 2


def test_worker_observation_payload_contains_only_circles_and_preserves_pixels(monkeypatch):
    from app import actual
    from app.board_observation import BoardObservation
    from app.schemas import Calibration, opening_position
    from test_board_observation import face_board
    image = face_board(opening_position())
    calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 450, 'y': 50},
                                       {'x': 450, 'y': 500}, {'x': 50, 'y': 500}])
    _, image, _, variants, _ = actual._prepare_tracking_frame((0, image, 0, {}), calibration, calibration, 1, False)
    payload = variants[3]
    assert len(payload) == 2
    assert all(array.ndim == 2 and array.shape[1] == 3 for array in payload)
    before, after = BoardObservation(), BoardObservation()
    expected = before.update(image, opening_position(), variants[1])
    def repeated_detection(*args, **kwargs):
        pytest.fail('compact payload repeated circle detection')
    monkeypatch.setattr(cv2, 'HoughCircles', repeated_detection)
    assert after.update(image, opening_position(), variants[1], prepared=payload) == expected
    for name in ('small', 'gray', 'hsv', 'edges', 'matrix'):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


@pytest.mark.parametrize('cpus,workers,threads', [(2, 1, 1), (8, 4, 4), (32, 8, 4), (None, 1, 1)])
def test_analysis_bounds_cpu_parallelism_without_changing_frame_order(tmp_path, video_path, monkeypatch, cpus, workers, threads):
    from contextlib import contextmanager
    import os
    from app import actual
    from app.schemas import Calibration, VideoMetadata, opening_position
    from app.storage import TaskStore
    monkeypatch.setattr(os, 'cpu_count', lambda: cpus)
    frames = [(i * .04, np.full((128, 128, 3), 60, np.uint8)) for i in range(3)]
    monkeypatch.setattr(actual, 'iter_frames', lambda *args, **kwargs: iter(frames))
    captured, observed = [], []
    @contextmanager
    def prepare(source, process, **kwargs):
        captured.append((kwargs.get('workers'), cv2.getNumThreads()))
        yield map(process, source)
    monkeypatch.setattr('app.preprocessing.process_frames', prepare)
    def consume(boards, *args, **kwargs):
        for time_s, _ in boards:
            observed.append(time_s)
        yield from ()
    monkeypatch.setattr(actual, 'actual_events', consume)
    store = TaskStore(tmp_path)
    task = store.create('parallel.mp4', VideoMetadata(width=128, height=128, duration_s=.08, codec='h264'))
    os.link(video_path, store.task_dir(task.id) / 'source.mp4')
    task.initial_position = opening_position()
    task.initial_frame_s = 0
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0},
                                            {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    task.calibration_confirmed, task.status = True, 'analyzing'
    task = store.save(task, task.revision)
    actual.analyze_actual(task, store, None)
    assert store.get(task.id).error is None
    assert captured == [(workers, threads)]
    assert observed == [row[0] for row in frames]

