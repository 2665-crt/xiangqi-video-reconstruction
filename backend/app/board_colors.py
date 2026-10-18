from functools import lru_cache
import os
from pathlib import Path

import numpy as np


MODEL_PATH = Path(os.environ['BOARD_COLOR_MODEL_PATH']) if os.getenv('BOARD_COLOR_MODEL_PATH') else (
    Path(__file__).resolve().parents[1] / 'models' / 'board-color-v1.npz')


@lru_cache(maxsize=2)
def _lookup(name):
    if not MODEL_PATH.is_file():
        return None
    with np.load(MODEL_PATH) as model:
        return model[name]


def wood_surface_mask(hsv):
    if os.getenv('BOARD_COLOR_MODEL') == '0':
        return None
    lookup = _lookup('wood')
    if lookup is None:
        return None
    return lookup[hsv[..., 0], hsv[..., 1] // 4, hsv[..., 2] // 4]


def skin_surface_mask(hsv):
    if os.getenv('BOARD_COLOR_MODEL') == '0':
        return None
    lookup = _lookup('skin')
    if lookup is None:
        return None
    return lookup[hsv[..., 0], hsv[..., 1] // 4, hsv[..., 2] // 4]
