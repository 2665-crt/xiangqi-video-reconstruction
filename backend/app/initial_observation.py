import cv2
import numpy as np


def board_geometry(image):
    height, width = image.shape[:2]
    cell = height - width
    return cell, (width - 8 * cell - 1) / 2


def observe_initial_board(image):
    from .analysis import read_board_cells
    cell, margin = board_geometry(image)
    inset = max(0, round(margin - cell / 2))
    compact = image[inset:image.shape[0] - inset, inset:image.shape[1] - inset] if inset else image
    cells = read_board_cells(compact, strict=True)
    scale = 50 / cell
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 120)
    yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
    neutral_pixels = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 100)
    occupied = {square for square, color in cells.items() if color is not None}
    origin = margin * scale
    for square in occupied:
        file, rank = square
        cx, cy = origin + file * 50, origin + rank * 50
        left, right = max(0, int(cx) - 29), min(small.shape[1], int(cx) + 30)
        top, bottom = max(0, int(cy) - 29), min(small.shape[0], int(cy) + 30)
        distance = (xx[top:bottom, left:right] - cx) ** 2 + (yy[top:bottom, left:right] - cy) ** 2
        inside, outside = distance < 15 ** 2, (distance > 24 ** 2) & (distance < 28 ** 2)
        neutral_patch = neutral_pixels[top:bottom, left:right]
        contrast = float(neutral_patch[inside].mean() - neutral_patch[outside].mean())
        if contrast < .25:
            cells.pop(square)
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1), cv2.HOUGH_GRADIENT,
                              1, 30, param1=75, param2=22, minRadius=13, maxRadius=27)
    faces, quality = {}, {}
    if circles is None:
        return cells, faces
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for cx, cy, radius in circles[0]:
        file = round((float(cx) - origin) / 50)
        rank = round((float(cy) - origin) / 50)
        dx, dy = cx - origin - file * 50, cy - origin - rank * 50
        if not (0 <= file < 9 and 0 <= rank < 10 and abs(dx) <= 21 and abs(dy) <= 24):
            continue
        extent = int(np.ceil(radius * .72)) + 1
        left, right = max(0, int(cx) - extent), min(small.shape[1], int(cx) + extent + 1)
        top, bottom = max(0, int(cy) - extent), min(small.shape[0], int(cy) + extent + 1)
        inner = (xx[top:bottom, left:right] - cx) ** 2 + (yy[top:bottom, left:right] - cy) ** 2 < (radius * .72) ** 2
        neutral = float(neutral_pixels[top:bottom, left:right][inner].mean())
        support = np.zeros(len(angles), bool)
        for delta in (-2, -1, 0, 1, 2):
            xs = np.clip(np.rint(cx + (radius + delta) * np.cos(angles)).astype(int), 0, small.shape[1] - 1)
            ys = np.clip(np.rint(cy + (radius + delta) * np.sin(angles)).astype(int), 0, small.shape[0] - 1)
            support |= edges[ys, xs] > 0
        boundary = float(support.mean())
        if boundary < .5 or (neutral < .42 and boundary < .8):
            continue
        score = boundary + neutral - abs(dx) / 100 - abs(dy) / 150
        square = file, rank
        if score <= quality.get(square, -1):
            continue
        face = cv2.getRectSubPix(small, (round(radius * 1.5), round(radius * 1.5)),
                                (float(cx), float(cy - radius * .15)))
        from .piece_classifier import piece_color
        cells[square] = piece_color(face)
        if square not in occupied:
            faces[square] = float(cx / scale), float((cy - radius * .15) / scale), float(radius / scale)
        quality[square] = score
    return cells, faces


def compatible_observations(older, newer):
    return not any(square in newer and newer[square] != color for square, color in older.items())


def merge_initial_observations(observations):
    merged, ambiguous = {}, set()
    for cells in observations:
        arrival_colors = {color for square, color in cells.items()
                          if color is not None and square not in merged}
        uncertain_sources = {square for square, color in merged.items()
                             if color in arrival_colors and square not in cells}
        for square in uncertain_sources:
            merged.pop(square)
        ambiguous.update(uncertain_sources)
        ambiguous.difference_update(cells)
        merged.update(cells)
    return merged, ambiguous
