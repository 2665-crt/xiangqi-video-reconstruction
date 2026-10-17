import cv2
import numpy as np
from functools import lru_cache

from .schemas import Calibration


def validate_calibration(calibration: Calibration, width: int, height: int):
    points = np.float32([[p.x, p.y] for p in calibration.corners])
    if not np.isfinite(points).all() or any(not (0 <= p.x < width and 0 <= p.y < height) for p in calibration.corners):
        raise ValueError("棋盘角点必须位于视频画面内")
    if abs(cv2.contourArea(points)) < 100 or not cv2.isContourConvex(points):
        raise ValueError("请按左上、右上、右下、左下选择棋盘最外侧交叉点")


@lru_cache(maxsize=32)
def _board_transform(corners, width, height, margin, cell_size):
    calibration = Calibration(corners=[{'x': x, 'y': y} for x, y in corners])
    validate_calibration(calibration, width, height)
    target = np.float32([[margin, margin], [8 * cell_size + margin, margin],
                         [8 * cell_size + margin, 9 * cell_size + margin], [margin, 9 * cell_size + margin]])
    return cv2.getPerspectiveTransform(np.float32(corners), target)


def rectify_board(frame: np.ndarray, calibration: Calibration, margin=0, cell_size=100) -> np.ndarray:
    corners = tuple((p.x, p.y) for p in calibration.corners)
    matrix = _board_transform(corners, frame.shape[1], frame.shape[0], margin, cell_size)
    result = cv2.warpPerspective(frame, matrix, (8 * cell_size + 1 + 2 * margin, 9 * cell_size + 1 + 2 * margin))
    if calibration.orientation in ("rotated_180", "mirrored_rotated_180"):
        result = cv2.rotate(result, cv2.ROTATE_180)
    if calibration.orientation in ("mirrored", "mirrored_rotated_180"):
        result = cv2.flip(result, 1)
    return result


def _grid_axis(lines, vertical, count):
    points, weights = [], []
    for x1, y1, x2, y2 in lines:
        length = abs(y2 - y1) if vertical else abs(x2 - x1)
        deviation = abs(x2 - x1) if vertical else abs(y2 - y1)
        if length >= 220 and deviation < length * .045:
            points.append((x1 + x2) / 2 if vertical else (y1 + y2) / 2)
            weights.append(length)
    if len(points) < count - 2:
        return None
    points, weights = np.asarray(points), np.asarray(weights)
    best = None
    for spacing in range(82, 119):
        for start in range(0, 121, 2):
            indices = np.rint((points - start) / spacing).astype(int)
            error = np.abs(points - start - indices * spacing)
            mask = (indices >= 0) & (indices < count) & (error < 8)
            groups = sorted(set(indices[mask]))
            if len(groups) < count - 1:
                continue
            support = sum(min(weights[mask & (indices == i)].sum(), 1800) for i in groups)
            score = len(groups) * 10000 + support - abs(spacing - 100) * 2
            if best is None or score > best[0]:
                centers = [np.median(points[mask & (indices == i)]) for i in groups]
                slope, intercept = np.polyfit(groups, centers, 1)
                best = score, intercept, intercept + slope * (count - 1)
    return None if best is None else best[1:]


def refine_calibration(frame, calibration):
    # 手选到外边框时，先以完整的 9×10 等距线网校正到交叉点。
    crop = rectify_board(frame, calibration, margin=50)
    from .analysis import read_board_cells
    cells = read_board_cells(crop, strict=True)
    if len(cells) == 90 and any(color is not None for color in cells.values()):
        return calibration
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    detected = cv2.HoughLinesP(cv2.Canny(gray, 65, 150), 1, np.pi / 180,
                               threshold=65, minLineLength=220, maxLineGap=35)
    if detected is None:
        return calibration
    horizontal, vertical = _grid_axis(detected[:, 0], False, 10), _grid_axis(detected[:, 0], True, 9)
    if horizontal is None or vertical is None:
        return calibration
    left, right = vertical
    top, bottom = horizontal
    if max(abs(left - 50), abs(right - 850), abs(top - 50), abs(bottom - 950)) < 10:
        return calibration
    oriented = np.float32([[[left, top], [right, top], [right, bottom], [left, bottom]]])
    if calibration.orientation in ('mirrored', 'mirrored_rotated_180'):
        oriented[:, :, 0] = 900 - oriented[:, :, 0]
        oriented = oriented[:, [1, 0, 3, 2]]
    if calibration.orientation in ('rotated_180', 'mirrored_rotated_180'):
        oriented[:, :, 0] = 900 - oriented[:, :, 0]
        oriented[:, :, 1] = 1000 - oriented[:, :, 1]
        oriented = oriented[:, [2, 3, 0, 1]]
    matrix = _board_transform(tuple((p.x, p.y) for p in calibration.corners),
                              frame.shape[1], frame.shape[0], 50, 100)
    corners = cv2.perspectiveTransform(oriented, np.linalg.inv(matrix))[0]
    return Calibration(corners=[{'x': float(x), 'y': float(y)} for x, y in corners], orientation=calibration.orientation)


def detect_calibration(frames: list[np.ndarray]) -> Calibration | None:
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(cv2.Canny(gray, 70, 160), 1, np.pi / 180, threshold=80,
                                minLineLength=min(gray.shape) * 0.3, maxLineGap=20)
        if lines is None:
            continue
        horizontal, vertical = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) < 8:
                horizontal.append((x1, y1, x2, y2))
            elif abs(x2 - x1) < 8:
                vertical.append((x1, y1, x2, y2))
        if len(horizontal) < 7 or len(vertical) < 7:
            continue
        left, right = np.percentile([line[0] for line in vertical], [2, 98])
        top, bottom = np.percentile([line[1] for line in horizontal], [2, 98])
        if right - left < 100 or bottom - top < 100:
            continue
        candidate = Calibration(corners=[{"x": left, "y": top}, {"x": right, "y": top},
                                          {"x": right, "y": bottom}, {"x": left, "y": bottom}])
        validate_calibration(candidate, frame.shape[1], frame.shape[0])
        return candidate
    return None
