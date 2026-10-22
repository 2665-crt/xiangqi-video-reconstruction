from fractions import Fraction
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.board_motion import BoardMotion
from app.schemas import Calibration
from app.video import iter_frames


@pytest.fixture
def textured_board():
    gray = np.random.default_rng(17).integers(30, 240, (480, 640), dtype=np.uint8)
    image = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    calibration = Calibration(corners=[{'x': 90, 'y': 80}, {'x': 550, 'y': 80},
                                      {'x': 550, 'y': 400}, {'x': 90, 'y': 400}])
    return image, calibration


def test_fixed_motion_reference_extracts_optical_flow_points_once(monkeypatch, textured_board):
    image, calibration = textured_board
    original, calls = cv2.goodFeaturesToTrack, []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', count)
    tracker = BoardMotion(calibration)
    for time_s in (0, .3, .6, .9):
        assert tracker.update(time_s, image) is calibration
    assert len(calls) == 1


def test_reference_refresh_invalidates_cached_points(monkeypatch, textured_board):
    image, calibration = textured_board
    original, calls = cv2.goodFeaturesToTrack, []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', count)
    tracker = BoardMotion(calibration)
    for time_s in (0, .3, .6, 2.1, 2.4):
        tracker.update(time_s, image)
    assert tracker.reference_s == 2.1
    assert len(calls) == 2


def test_reference_mask_change_invalidates_cached_points(monkeypatch, textured_board):
    image, calibration = textured_board
    original, calls = cv2.goodFeaturesToTrack, []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', count)
    tracker = BoardMotion(calibration)
    for time_s in (0, .3, .6):
        tracker.update(time_s, image)
    tracker.reference_calibration = calibration.model_copy(deep=True)
    tracker.reference_calibration.corners[0].x += 1
    tracker.update(.9, image)
    assert len(calls) == 2


def test_mask_shape_change_invalidates_reference_features(monkeypatch, textured_board):
    image, calibration = textured_board
    masks = []
    def no_points(gray, *args, mask, **kwargs):
        masks.append(mask.shape)
        return None
    class NoDescriptors:
        def detectAndCompute(self, gray, mask):
            return None, None
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', no_points)
    monkeypatch.setattr(cv2, 'SIFT_create', lambda **kwargs: NoDescriptors())
    tracker = BoardMotion(calibration)
    tracker.update(0, image)
    tracker.update(.3, image)
    tracker.update(.6, cv2.resize(image, (320, 240)))
    assert masks == [(480, 640), (240, 320)]


def test_sift_reference_is_reused_but_current_frame_is_recomputed(monkeypatch, textured_board):
    image, calibration = textured_board
    factory, calls = cv2.SIFT_create, {'reference': 0, 'current': 0}
    class Detector:
        def __init__(self, *args, **kwargs):
            self.detector = factory(*args, **kwargs)
        def detectAndCompute(self, gray, mask):
            calls['current' if mask is None else 'reference'] += 1
            return self.detector.detectAndCompute(gray, mask)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', lambda *a, **k: None)
    monkeypatch.setattr(cv2, 'SIFT_create', Detector)
    tracker = BoardMotion(calibration)
    for time_s in (0, .3, 1.4):
        tracker.update(time_s, image)
    assert calls == {'reference': 1, 'current': 2}


@pytest.mark.parametrize('orientation', ['normal', 'rotated_180', 'mirrored', 'mirrored_rotated_180'])
def test_cache_and_original_mode_preserve_all_motion_state(monkeypatch, textured_board, orientation):
    image, calibration = textured_board
    calibration.orientation = orientation
    frames = []
    for n in range(28):
        changed = cv2.warpAffine(image, np.float32([[1, 0, n * .22], [0, 1, n * -.08]]), (640, 480))
        if 8 <= n <= 11:
            changed[120:250, 160:400] = (140, 180, 210)
        frames.append(changed)
    results = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('VIDEO_REFERENCE_CACHE', enabled)
        cv2.setRNGSeed(0)
        tracker, states = BoardMotion(calibration), []
        for n, frame in enumerate(frames):
            result = tracker.update(n * .3, frame)
            states.append((result.model_dump(), tracker.reference_calibration.model_dump(),
                           tracker.reference_s, tracker.last_check_s, tracker.last_feature_check_s, tracker.active))
        results.append(states)
    assert results[0] == results[1]


def test_cache_can_be_disabled_for_baseline_comparison(monkeypatch, textured_board):
    monkeypatch.setenv('VIDEO_REFERENCE_CACHE', '0')
    image, calibration = textured_board
    original, calls = cv2.goodFeaturesToTrack, []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', count)
    tracker = BoardMotion(calibration)
    for time_s in (0, .3, .6, .9):
        tracker.update(time_s, image)
    assert len(calls) == 3


@pytest.mark.parametrize('threads', [0, 3])
def test_decoder_uses_bounded_threads_and_keeps_original_mode(monkeypatch, threads):
    stream = SimpleNamespace(start_time=0, time_base=Fraction(1, 10), thread_type='SLICE',
                             codec_context=SimpleNamespace(thread_count=0))
    class Container:
        streams = SimpleNamespace(video=[stream])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def decode(self, selected):
            assert selected is stream
            return iter(())
    monkeypatch.setenv('VIDEO_DECODE_THREADS', str(threads))
    monkeypatch.setattr('app.video.av.open', lambda path: Container())
    assert list(iter_frames('unused', 0, 1)) == []
    assert stream.codec_context.thread_count == threads
    assert stream.thread_type == ('AUTO' if threads else 'SLICE')


@pytest.mark.parametrize('value', ['-1', 'invalid'])
def test_invalid_decode_thread_setting_is_explicit(monkeypatch, video_path, value):
    monkeypatch.setenv('VIDEO_DECODE_THREADS', value)
    with pytest.raises(ValueError, match='VIDEO_DECODE_THREADS'):
        list(iter_frames(video_path, 0, 2))


@pytest.mark.parametrize('start,end', [(0, 2), (.5, 1), (1.7, 2)])
def test_parallel_decoding_preserves_all_pixels_timestamps_and_tail(monkeypatch, video_path, start, end):
    outputs = []
    for threads in ('0', '4'):
        monkeypatch.setenv('VIDEO_DECODE_THREADS', threads)
        outputs.append(list(iter_frames(video_path, start, end)))
    assert len(outputs[0]) == len(outputs[1]) > 0
    assert [t for t, _ in outputs[0]] == [t for t, _ in outputs[1]]
    for (_, before), (_, after) in zip(*outputs):
        np.testing.assert_array_equal(before, after)
    if start == 0:
        assert len(outputs[0]) == 24


def test_parallel_decoding_preserves_variable_rate_pixels(monkeypatch, tmp_path):
    import av
    path = tmp_path / 'variable-speed.mp4'
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 128, 128, 'yuv420p'
        for pts in (0, 1, 3, 7, 8):
            frame = av.VideoFrame.from_ndarray(np.full((128, 128, 3), pts * 20, np.uint8), format='bgr24')
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    results = []
    for threads in ('0', '4'):
        monkeypatch.setenv('VIDEO_DECODE_THREADS', threads)
        results.append(list(iter_frames(path, 0, 1)))
    assert [round(t, 3) for t, _ in results[1]] == [0, .1, .3, .7, .8]
    for (ta, a), (tb, b) in zip(*results):
        assert ta == tb
        np.testing.assert_array_equal(a, b)
