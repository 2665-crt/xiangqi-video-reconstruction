from pathlib import Path
import os

import av
import cv2

from .schemas import VideoMetadata


def save_image(path: Path, image):
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("证据截图编码失败")
    path.write_bytes(encoded.tobytes())


def probe_video(path: Path) -> VideoMetadata:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("没有视频轨道")
            stream = container.streams.video[0]
            next(container.decode(stream))
            duration = float(stream.duration * stream.time_base) if stream.duration else float((container.duration or 0) / av.time_base)
            if duration <= 0:
                raise ValueError("无法读取视频时长")
            return VideoMetadata(width=stream.width, height=stream.height, duration_s=duration, codec=stream.codec_context.name)
    except (av.FFmpegError, StopIteration, ValueError) as exc:
        raise ValueError("视频无法解码，请上传完整的 MP4/H.264 文件") from exc


def iter_frames(path: Path, start_s: float, end_s: float):
    try:
        threads = int(os.getenv('VIDEO_DECODE_THREADS', '4'))
        if threads < 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError('VIDEO_DECODE_THREADS 必须为非负整数，0 保留原解码模式') from exc
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if threads:
            stream.codec_context.thread_count = threads
            stream.thread_type = 'AUTO'
        offset = float((stream.start_time or 0) * stream.time_base)
        if start_s > 0:
            container.seek(int((start_s + offset) * av.time_base), backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time_s = float(frame.pts * frame.time_base) - offset
            if time_s + 1e-6 < start_s:
                continue
            if time_s > end_s + 1e-6:
                break
            yield time_s, frame.to_ndarray(format="bgr24")
