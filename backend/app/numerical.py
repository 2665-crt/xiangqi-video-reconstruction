import numpy as np
from numba import njit


_angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
_cosines, _sines = np.cos(_angles), np.sin(_angles)
_dy, _dx = np.mgrid[-29:30, -29:30]
_ring = (_dx * _dx + _dy * _dy >= 22 * 22) & (_dx * _dx + _dy * _dy < 29 * 29)
_ring_y, _ring_x = _dy[_ring], _dx[_ring]


@njit(cache=True, nogil=True)
def circle_support(edges, cx, cy, radius):
    supported = 0
    for angle in range(96):
        for delta in range(-2, 3):
            expanded = np.float32(radius + np.float32(delta))
            x = int(np.rint(np.float64(cx) + np.float64(expanded) * _cosines[angle]))
            y = int(np.rint(np.float64(cy) + np.float64(expanded) * _sines[angle]))
            x = min(max(x, 0), edges.shape[1] - 1)
            y = min(max(y, 0), edges.shape[0] - 1)
            if edges[y, x] > 0:
                supported += 1
                break
    return supported / 96


@njit(cache=True, nogil=True)
def finger_counts(small, hsv, squares, wood_mask=None):
    counts = np.zeros(len(squares), np.int64)
    for index in range(len(squares)):
        center_x, center_y = 55 + squares[index, 0] * 50, 55 + squares[index, 1] * 50
        for offset in range(len(_ring_y)):
            y, x = center_y + _ring_y[offset], center_x + _ring_x[offset]
            if wood_mask is not None and wood_mask[y, x]:
                continue
            hue, saturation, value = hsv[y, x, 0], hsv[y, x, 1], hsv[y, x, 2]
            blue, green, red = float(small[y, x, 0]), float(small[y, x, 1]), float(small[y, x, 2])
            if hue < 23 and saturation > 55 and value > 95 and red > green * 1.2 and blue > green * .62:
                counts[index] += 1
    return counts


RING_SIZE = len(_ring_y)
