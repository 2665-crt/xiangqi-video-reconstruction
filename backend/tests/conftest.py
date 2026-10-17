from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "yuv420p"
        for i in range(24):
            frame = av.VideoFrame.from_ndarray(np.full((128, 128, 3), i * 5, dtype=np.uint8), format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
