import os

import numpy as np
from numba import njit

from .board_colors import _lookup
from . import board_colors


def surface_tables():
    return (None, None) if os.getenv('BOARD_COLOR_MODEL') == '0' else (_lookup('skin'), _lookup('wood'))


def binary_surfaces(tables):
    return all(table is None or table.dtype == np.bool_ for table in tables)


def native_surfaces(skin, wood):
    return all(function is getattr(board_colors, name)
               and getattr(function, '__globals__', None) is board_colors.__dict__
               and getattr(getattr(function, '__code__', None), 'co_name', None) == name
               for name, function in (('skin_surface_mask', skin), ('wood_surface_mask', wood)))


@njit(cache=True, nogil=True, fastmath=False)
def _median32(values):
    ordered = np.sort(values)
    if np.isnan(ordered[-1]):
        return np.float32(np.nan)
    middle = len(ordered) // 2
    # NumPy median 对中间值取 mean；从正零累加以保留 signed zero 和 float32 舍入。
    total = np.float32(0)
    if len(ordered) % 2 == 0:
        total = np.float32(total + ordered[middle - 1])
    total = np.float32(total + ordered[middle])
    return np.float32(total / np.float32(1 if len(ordered) % 2 else 2))


@njit(cache=True, nogil=True, fastmath=False)
def flow_metrics(old, new, valid, counts):
    shifts = np.zeros((len(counts), 2), np.float32)
    spreads = np.zeros(len(counts), np.float32)
    kept = np.zeros(len(counts), np.int64)
    offset = 0
    for index in range(len(counts)):
        count = counts[index]
        for j in range(count):
            kept[index] += int(valid[offset + j])
        if kept[index] >= max(4, count * .5):
            delta = np.empty((kept[index], 2), np.float32)
            target = 0
            for j in range(count):
                if valid[offset + j]:
                    delta[target, 0] = new[offset + j, 0, 0] - old[offset + j, 0, 0]
                    delta[target, 1] = new[offset + j, 0, 1] - old[offset + j, 0, 1]
                    target += 1
            shifts[index, 0] = _median32(delta[:, 0])
            shifts[index, 1] = _median32(delta[:, 1])
            distances = np.empty(len(delta), np.float32)
            for j in range(len(delta)):
                dx = delta[j, 0] - shifts[index, 0]
                dy = delta[j, 1] - shifts[index, 1]
                distances[j] = np.sqrt(dx * dx + dy * dy)
            spreads[index] = _median32(distances)
        offset += count
    return shifts, spreads, kept


@njit(cache=True, nogil=True, fastmath=False)
def face_surface(hsv, core, skin, wood):
    count, neutral, skin_count, wood_count = 0, 0, 0, 0
    for y in range(hsv.shape[0]):
        for x in range(hsv.shape[1]):
            if not core[y, x]:
                continue
            h, s, v = hsv[y, x]
            count += 1
            neutral += int(s < 120 and v > 80)
            if skin is not None:
                skin_count += int(skin[h, s // 4, v // 4])
            if wood is not None:
                wood_count += int(wood[h, s // 4, v // 4])
    return neutral / count >= .5 and skin_count / count <= .08 and wood_count / count <= .5


@njit(cache=True, nogil=True, fastmath=False)
def grid_surface(patches, skin, wood):
    flags = np.zeros(len(patches), np.uint8)
    count = patches.shape[1] * patches.shape[2]
    for index in range(len(patches)):
        skin_count, wood_count = 0, 0
        for y in range(patches.shape[1]):
            for x in range(patches.shape[2]):
                h, s, v = patches[index, y, x]
                if skin is not None:
                    skin_count += int(skin[h, s // 4, v // 4])
                if wood is not None:
                    wood_count += int(wood[h, s // 4, v // 4])
        flags[index] = 1 if skin_count / count > .08 else 2 if wood_count / count > .8 else 0
    return flags
