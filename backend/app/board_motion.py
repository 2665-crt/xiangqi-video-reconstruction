import cv2
import numpy as np

from .schemas import Calibration


class BoardMotion:
    def __init__(self, calibration, active=False):
        self.calibration = calibration
        self.gray = None
        self.reference_s = None
        self.last_check_s = float('-inf')
        self.active = active

    def update(self, time_s, frame):
        if time_s - self.last_check_s < .24:
            return self.calibration
        self.last_check_s = time_s
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray is None:
            self.gray = gray
            self.reference_s = time_s
            return self.calibration
        corners = np.float32([[p.x, p.y] for p in self.calibration.corners])
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(self.gray, 240, .01, 8, mask=mask)
        if points is None or len(points) < 30:
            return self.calibration
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None)
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, following, None)
        valid = ((status.ravel() > 0) & (reverse_status.ravel() > 0)
                 & (np.linalg.norm(points - backward, axis=2).ravel() < 1))
        if valid.sum() < 30:
            return self.calibration
        matrix, inliers = cv2.findHomography(points[valid], following[valid], cv2.RANSAC, 1.5)
        if matrix is None or inliers.sum() < max(30, valid.sum() * .65):
            return self.calibration
        supported = points[valid][inliers.ravel().astype(bool)]
        # 单个棋子或手臂的局部移动不能带动整盘坐标。
        if (cv2.contourArea(cv2.convexHull(supported)) < cv2.contourArea(corners) * .4
                or np.any(np.ptp(supported[:, 0], axis=0) < np.ptp(corners, axis=0) * .65)):
            return self.calibration
        unit = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
        perimeter = cv2.perspectiveTransform(supported, unit)[:, 0]
        if any(np.count_nonzero(edge) < 3 for edge in (perimeter[:, 0] < .08, perimeter[:, 0] > .92,
                                                      perimeter[:, 1] < .08, perimeter[:, 1] > .92)):
            return self.calibration
        moved = cv2.perspectiveTransform(corners[None], matrix)[0]
        distance = np.linalg.norm(moved - corners, axis=1)
        cell_size = np.mean([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) / (8 if i % 2 == 0 else 9)
                             for i in range(4)])
        # 先确认足够大的整盘位移，再追随细小变化，避免扰动已正确的静止棋盘。
        if np.median(distance) < (1 if self.active else max(1, cell_size * .12)):
            return self.calibration
        height, width = gray.shape
        if (distance.max() > min(width, height) * .1 or not cv2.isContourConvex(moved)
                or np.any(moved < 0) or np.any(moved[:, 0] >= width) or np.any(moved[:, 1] >= height)):
            return self.calibration
        area_ratio = cv2.contourArea(moved) / cv2.contourArea(corners)
        if not .85 <= area_ratio <= 1.15:
            return self.calibration
        self.calibration = Calibration(corners=[{'x': float(x), 'y': float(y)} for x, y in moved],
                                       orientation=self.calibration.orientation)
        self.gray = gray
        self.reference_s = time_s
        self.active = True
        return self.calibration
