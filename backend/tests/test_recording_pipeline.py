import json
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from app import recording


def test_parallel_renderer_keeps_order_and_repeats():
    root = Path(__file__).resolve().parents[2]
    renderer = root / 'output/playwright/recording-pipeline/test-renderer.cjs'
    if not renderer.is_file():
        pytest.skip('专用 Node 渲染回归夹具未随公开源码分发')
    result = subprocess.run(['node', str(renderer)],
                            cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'CONCURRENT_CAPTURE_AND_ORDER_PASSED' in result.stdout


@pytest.fixture
def packet_renderer(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    children = []

    def install(images, repeats, handshake=False, loop=False, size_extra=0):
        packets = []
        for i, (image, repeat) in enumerate(zip(images, repeats)):
            path = tmp_path / f'frame-{i}.png'
            path.write_bytes(cv2.imencode('.png', image)[1].tobytes())
            packets.append((str(path), repeat))
        paused, ahead = tmp_path / 'consumer-paused', tmp_path / 'producer-ahead'
        script = tmp_path / 'renderer.py'
        script.write_text(
            'import json,sys,time\nfrom pathlib import Path\n'
            f'packets={packets!r}\npaused=Path({str(paused)!r})\nahead=Path({str(ahead)!r})\n'
            f'for i in range({20 if loop else len(packets)}):\n'
            ' path,repeat=packets[i % len(packets)]\n png=Path(path).read_bytes()\n'
            f' sys.stdout.buffer.write((json.dumps({{"bytes":len(png)+{size_extra},"repeat":repeat}})+"\\n").encode())\n'
            ' sys.stdout.buffer.write(png)\n sys.stdout.buffer.flush()\n'
            + (' if i==0:\n  while not paused.exists():time.sleep(.01)\n if i==2:ahead.write_text("ready")\n' if handshake else ''),
            encoding='utf-8')

        def launch(args, **kwargs):
            if len(args) > 1 and str(args[1]).endswith('render-recording.cjs'):
                child = original_popen([sys._base_executable, str(script)], **kwargs)
                children.append(child)
                return child
            return original_popen(args, **kwargs)

        monkeypatch.setattr(recording.subprocess, 'Popen', launch)
        manifest = tmp_path / 'manifest.json'
        manifest.write_text(json.dumps({'total_frames': sum(repeats) if not loop else 20}), encoding='utf-8')
        return manifest, paused, ahead, children

    yield install
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_encoder_reads_ahead_while_progress_is_busy(tmp_path, packet_renderer):
    image = np.random.default_rng(37).integers(0, 256, (1192, 1080, 3), dtype=np.uint8)
    manifest, paused, ahead, children = packet_renderer([image] * 3, [1, 1, 1], handshake=True)

    def progress(frames):
        if frames == 1:
            paused.write_text('ready')
            deadline = time.monotonic() + 2
            while not ahead.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert ahead.exists(), '编码期间没有预读后续画面'

    recording.encode_recording(manifest, tmp_path / 'video.mp4', progress, 'unused')
    assert children[0].poll() == 0


def test_next_png_is_read_while_current_png_is_decoding(tmp_path, packet_renderer, monkeypatch):
    image = np.random.default_rng(38).integers(0, 256, (1192, 1080, 3), dtype=np.uint8)
    manifest, paused, ahead, _ = packet_renderer([image] * 3, [1, 1, 1], handshake=True)
    original_decode = recording.cv2.imdecode
    first = True

    def decode(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            paused.write_text('ready')
            deadline = time.monotonic() + 2
            while not ahead.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert ahead.exists(), 'PNG解码期间没有继续读取后续画面'
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(recording.cv2, 'imdecode', decode)
    recording.encode_recording(manifest, tmp_path / 'video.mp4', lambda frames: None, 'unused')


def test_pipeline_keeps_packet_order_repeats_and_timestamps(tmp_path, packet_renderer):
    colors = [(10, 20, 230), (10, 220, 20), (220, 20, 10)]
    images = [np.full((1192, 1080, 3), color, dtype=np.uint8) for color in colors]
    manifest, _, _, _ = packet_renderer(images, [2, 3, 2])
    movie = tmp_path / 'video.mp4'
    completed = []
    recording.encode_recording(manifest, movie, completed.append, 'unused')
    expected = [colors[0], colors[0], colors[1], colors[1], colors[1], colors[2], colors[2]]
    with av.open(str(movie)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 7
    for i, (frame, color) in enumerate(zip(frames, expected)):
        actual = frame.to_ndarray(format='bgr24')[500, 500].astype(int)
        assert np.max(np.abs(actual - color)) <= 3
        assert frame.pts * frame.time_base == Fraction(i, 60)
    assert completed == [2, 5, 7]


def test_encoding_failure_does_not_leave_a_blocked_renderer(tmp_path, packet_renderer):
    image = np.random.default_rng(39).integers(0, 256, (1192, 1080, 3), dtype=np.uint8)
    manifest, _, _, children = packet_renderer([image], [1], loop=True)

    def fail(frames):
        time.sleep(.2)
        raise RuntimeError('验收模拟编码中断')

    start = time.monotonic()
    with pytest.raises(RuntimeError, match='验收模拟编码中断'):
        recording.encode_recording(manifest, tmp_path / 'video.mp4', fail, 'unused')
    assert time.monotonic() - start < 3
    assert children[0].poll() is not None
    assert not any(t.name in ('recording-frame-reader', 'recording-frame-decoder') for t in threading.enumerate())


def test_bad_png_from_reader_is_reported_to_encoder(tmp_path, packet_renderer):
    image = np.zeros((1192, 1080, 3), np.uint8)
    manifest, _, _, children = packet_renderer([image], [1])
    (tmp_path / 'frame-0.png').write_bytes(b'not a PNG')
    with pytest.raises(RuntimeError, match='棋盘画面尺寸不正确'):
        recording.encode_recording(manifest, tmp_path / 'video.mp4', lambda frames: None, 'unused')
    assert children[0].poll() is not None


def test_truncated_packet_ends_reader_and_reports_stream_error(tmp_path, packet_renderer):
    image = np.zeros((1192, 1080, 3), np.uint8)
    manifest, _, _, children = packet_renderer([image], [1], size_extra=20)
    with pytest.raises(RuntimeError, match='棋盘画面流中断'):
        recording.encode_recording(manifest, tmp_path / 'video.mp4', lambda frames: None, 'unused')
    assert children[0].poll() is not None
    assert not any(t.name in ('recording-frame-reader', 'recording-frame-decoder') for t in threading.enumerate())
