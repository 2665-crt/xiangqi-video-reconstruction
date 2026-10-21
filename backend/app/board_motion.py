import os

import cv2
import numpy as np

from .schemas import Calibration


class BoardMotion:
    def __init__(self, calibration, active=False, reference_calibration=None):
        self.calibration = calibration
        self.reference_calibration = reference_calibration or calibration
        self.gray = None
        self.reference_s = None
        self.last_check_s = float('-inf')
        self.active = active
        self.last_feature_check_s = float('-inf')
        self.cache_enabled = os.getenv('VIDEO_REFERENCE_CACHE', '1') != '0'
        self._feature_gray = None
        self._feature_key = None
        self._feature_mask = None
        self._points_ready = False
        self._sift_ready = False
        self._reference_points = None
        self._reference_sift = None

    def update(self, time_s, frame):
        if time_s - self.last_check_s < .24:
            return self.calibration
        self.last_check_s = time_s
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray is None:
            self.gray = gray
            self.reference_s = time_s
            return self.calibration
        corners = np.float32([[p.x, p.y] for p in self.reference_calibration.corners])
        applied = np.float32([[p.x, p.y] for p in self.calibration.corners])
        feature_key = tuple(corners.ravel()), gray.shape, gray.dtype.str
        if not self.cache_enabled or self._feature_gray is not self.gray or self._feature_key != feature_key:
            self._feature_gray, self._feature_key = self.gray, feature_key
            self._feature_mask = np.zeros_like(gray)
            cv2.fillConvexPoly(self._feature_mask, corners.astype(np.int32), 255)
            self._points_ready = self._sift_ready = False
            self._reference_points = self._reference_sift = None
        mask = self._feature_mask
        for method in ('flow', 'features'):
            if method == 'flow':
                if not self._points_ready:
                    self._reference_points = cv2.goodFeaturesToTrack(self.gray, 240, .01, 8, mask=mask)
                    self._points_ready = True
                points = self._reference_points.copy() if self._reference_points is not None else None
                if points is None or len(points) < 30:
                    continue
                following, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None)
                backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, following, None)
                valid = ((status.ravel() > 0) & (reverse_status.ravel() > 0)
                         & (np.linalg.norm(points - backward, axis=2).ravel() < 1))
                points, following = points[valid], following[valid]
            else:
                if time_s - self.last_feature_check_s < 1:
                    break
                self.last_feature_check_s = time_s
                detector = cv2.SIFT_create(nfeatures=1600)
                if not self._sift_ready:
                    self._reference_sift = detector.detectAndCompute(self.gray, mask)
                    self._sift_ready = True
                old_keys, old_features = self._reference_sift
                new_keys, new_features = detector.detectAndCompute(gray, None)
                if old_features is None or new_features is None or len(new_features) < 2:
                    continue
                pairs = cv2.BFMatcher().knnMatch(old_features, new_features, k=2)
                matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
                points = np.float32([old_keys[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                following = np.float32([new_keys[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if len(points) < 30:
                continue
            matrix, inliers = cv2.findHomography(points, following, cv2.RANSAC, 1.5)
            if matrix is None or inliers.sum() < max(30, len(points) * .65):
                continue
            supported = points[inliers.ravel().astype(bool)]
            # 单个棋子或手臂的局部移动不能带动整盘坐标。
            if (cv2.contourArea(cv2.convexHull(supported)) < cv2.contourArea(corners) * .4
                    or np.any(np.ptp(supported[:, 0], axis=0) < np.ptp(corners, axis=0) * .65)):
                continue
            unit = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            perimeter = cv2.perspectiveTransform(supported, unit)[:, 0]
            if any(np.count_nonzero(edge) < 3 for edge in (perimeter[:, 0] < .08, perimeter[:, 0] > .92,
                                                          perimeter[:, 1] < .08, perimeter[:, 1] > .92)):
                continue
            moved = cv2.perspectiveTransform(corners[None], matrix)[0]
            height, width = gray.shape
            if (np.linalg.norm(moved - corners, axis=1).max() > min(width, height) * (.25 if method == 'features' else .1)
                    or not cv2.isContourConvex(moved) or np.any(moved < 0)
                    or np.any(moved[:, 0] >= width) or np.any(moved[:, 1] >= height)):
                continue
            if not .85 <= cv2.contourArea(moved) / cv2.contourArea(corners) <= 1.15:
                continue
            candidate = Calibration(corners=[{'x': float(x), 'y': float(y)} for x, y in moved],
                                    orientation=self.calibration.orientation)
            cell_size = np.mean([np.linalg.norm(applied[(i + 1) % 4] - applied[i]) / (8 if i % 2 == 0 else 9)
                                 for i in range(4)])
            # 刷新匹配画面，但以累计位移决定是否改变坐标，兼顾静止抖动和缓慢移动。
            if np.median(np.linalg.norm(moved - applied, axis=1)) >= (1 if self.active else max(1, cell_size * .12)):
                self.calibration = candidate
                self.active = True
            if self.calibration is candidate or time_s - self.reference_s >= 2:
                self.reference_calibration = candidate
                self.gray, self.reference_s = gray, time_s
            break
        return self.calibration
