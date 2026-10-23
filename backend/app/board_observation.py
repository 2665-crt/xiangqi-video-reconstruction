import cv2
import numpy as np

from .board_colors import skin_surface_mask, wood_surface_mask
from .initial_observation import board_geometry
from .numerical import circle_support
from .piece_classifier import piece_color


class BoardObservation:
    """从已知开局的圆盘位置估计表面投影，不用棋种文字拟合坐标。"""

    def __init__(self):
        self.matrix = np.eye(3, dtype=np.float64)
        self.fitted = False
        self.centers = {}
        self.initial = None
        self._inverse_source = None
        self._inverse_matrix = None

    def project(self, points, inverse=False):
        matrix = self.matrix
        if inverse:
            if self._inverse_source is None or not np.array_equal(self.matrix, self._inverse_source):
                self._inverse_matrix = np.linalg.inv(self.matrix)
                self._inverse_source = self.matrix.copy()
            matrix = self._inverse_matrix
        return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), matrix)[:, 0]

    def _circles(self, gray, edges, max_radius, support):
        found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, param1=75,
                                 param2=22, minRadius=13, maxRadius=max_radius)
        circles = []
        if found is not None:
            for x, y, radius in found[0]:
                extent = round(radius * .7)
                patch = cv2.getRectSubPix(self.hsv, (extent * 2 + 1, extent * 2 + 1), (float(x), float(y)))
                skin, wood = skin_surface_mask(patch), wood_surface_mask(patch)
                neutral = np.mean((patch[..., 1] < 120) & (patch[..., 2] > 80))
                if (neutral >= .5 and circle_support(edges, x, y, radius) >= support
                        and not (skin is not None and skin.mean() > .08)
                        and not (wood is not None and wood.mean() > .5)):
                    circles.append((x, y, radius))
        return np.float32(circles).reshape(-1, 3)

    def _fit(self, circles, position):
        if len(circles) < 3 or len(position.pieces) < 3:
            return
        ideal = np.float32([[self.origin + p.square.file * 50, self.origin + p.square.rank * 50]
                            for p in position.pieces])
        centers = circles[:, :2]
        differences = centers[:, None, :] - ideal[None, :, :]
        best = None
        for dx in range(-20, 21, 5):
            for dy in range(-35, 21, 5):
                distances = np.linalg.norm(differences - [dx, dy], axis=2)
                matches = distances.argmin(axis=1)
                pairs = sorted((distances[i, j], i, j) for i, j in enumerate(matches) if distances[i, j] < 18)
                used, selected = set(), []
                for distance, i, j in pairs:
                    if j not in used:
                        used.add(j)
                        selected.append((i, j))
                score = len(selected) - sum(distances[i, j] for i, j in selected) / 1000
                if best is None or score > best[0]:
                    best = score, selected
        selected = best[1]
        if len(selected) < max(3, min(12, len(position.pieces) * .6)):
            return
        source = np.float32([ideal[j] for i, j in selected])
        target = np.float32([centers[i] for i, j in selected])
        translation = np.median(target - source, axis=0)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, 2] = translation
        if len(source) >= 12 and np.all(np.ptp(source, axis=0) >= 200):
            candidate, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5)
            if candidate is not None and inliers.sum() >= max(12, len(source) * .6):
                matrix = candidate
        corners = np.float32([[self.origin, self.origin], [self.origin + 400, self.origin],
                              [self.origin + 400, self.origin + 450], [self.origin, self.origin + 450]])
        projected = cv2.perspectiveTransform(corners[None], matrix)[0]
        if (np.max(np.linalg.norm(projected - corners, axis=1)) > 40
                or not cv2.isContourConvex(projected) or cv2.contourArea(projected) < 120000):
            return
        self.matrix, self.fitted = matrix, True

    def _prepare_image(self, image):
        cell, margin = board_geometry(image)
        scale = 50 / cell
        self.origin = margin * scale
        self.small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.gray = cv2.cvtColor(self.small, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.small, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(self.gray, 50, 120)
        self.edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(self.gray, (5, 5), 1), edges

    @classmethod
    def prepare(cls, image, *, compact=False):
        observation = cls()
        blurred, edges = observation._prepare_image(image)
        primary = observation._circles(blurred, edges, 27, .6)
        secondary = observation._circles(blurred, edges, 32, .8)
        if compact:
            return primary, secondary
        return (observation.origin, observation.small, observation.gray, observation.hsv,
                observation.edges, primary, secondary)

    def update(self, image, position, base_cells, *, prepared=None):
        if self.initial is None:
            self.initial = position.model_copy(deep=True)
        if prepared is None:
            blurred, edges = self._prepare_image(image)
            circles = self._circles(blurred, edges, 27, .6)
        elif len(prepared) == 2:
            self._prepare_image(image)
            circles, secondary = prepared
        else:
            self.origin, self.small, self.gray, self.hsv, self.edges, circles, secondary = prepared
        if not self.fitted:
            self._fit(circles, self.initial)
        primary_count = len(circles)
        if self.fitted:
            if prepared is None:
                secondary = self._circles(blurred, edges, 32, .8)
            circles = np.concatenate((circles, secondary))
        cells = dict(base_cells)
        self.centers = {}
        quality = {}
        logical = self.project(circles[:, :2], inverse=True) if len(circles) else []
        for index, ((x, y, radius), point) in enumerate(zip(circles, logical)):
            file, rank = np.rint((point - self.origin) / 50).astype(int)
            square = int(file), int(rank)
            distance = np.linalg.norm(point - (self.origin + np.array(square) * 50))
            if not (0 <= file < 9 and 0 <= rank < 10) or distance > 20:
                continue
            if index >= primary_count and square in self.centers:
                continue
            if distance >= quality.get(square, float('inf')):
                continue
            quality[square] = distance
            center = np.float32([x, y])
            face = cv2.getRectSubPix(self.small, (round(radius * 1.4), round(radius * 1.4)), (float(x), float(y)))
            self.centers[square] = center, float(radius)
            cells[square] = piece_color(face)
        for file in range(9):
            for rank in range(10):
                square = file, rank
                if square in self.centers:
                    continue
                if self.fitted and cells.get(square) is not None:
                    cells.pop(square, None)
                x, y = self.project([[self.origin + file * 50, self.origin + rank * 50]])[0]
                if not (8 <= x < self.small.shape[1] - 8 and 8 <= y < self.small.shape[0] - 8):
                    cells.pop(square, None)
                    continue
                patch = cv2.getRectSubPix(self.hsv, (15, 15), (float(x), float(y)))
                skin, wood = skin_surface_mask(patch), wood_surface_mask(patch)
                if skin is not None and skin.mean() > .08:
                    cells.pop(square, None)
                elif wood is not None and wood.mean() > .8:
                    cells[square] = None
        self.cells = cells
        return cells
