from dataclasses import dataclass

import cv2
import numpy as np

from .board_colors import skin_surface_mask, wood_surface_mask
from .schemas import Piece
from .board_observation import BoardObservation


@dataclass
class PieceTrack:
    piece: Piece
    center: np.ndarray
    radius: int
    appearance: np.ndarray
    template: np.ndarray
    mask: np.ndarray
    points: np.ndarray | None
    generation: int
    stable_since: float
    stable_center: np.ndarray
    stable_hits: int = 1


class PieceTracker:
    """沿已知身份跟踪棋子边缘；轨迹中断后不在新落点猜测身份。"""

    def __init__(self):
        self.tracks = {}
        self.gray = None
        self.time_s = None
        self.identities = {}
        self.tokens = {}
        self.generation = 0
        self.departed = set()
        self.observation = BoardObservation()
        self.trusted = {}
        self.reference = None
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        self.circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)

    def _face(self, center, radius=None):
        x, y = center
        radii = range(14, 31) if radius is None else range(max(12, radius - 2), min(32, radius + 3))
        best = None
        for r in radii:
            ring = np.rint(center + self.circle * r).astype(int)
            if (np.any(ring < 0) or np.any(ring[:, 0] >= self.current_gray.shape[1])
                    or np.any(ring[:, 1] >= self.current_gray.shape[0])):
                continue
            support = np.mean(self.edges[ring[:, 1], ring[:, 0]] > 0)
            if best is None or support > best[0]:
                best = support, r
        if best is None or best[0] < .65:
            return None
        r = best[1] if radius is None else radius
        extent = r + 3
        if (x < extent or y < extent or x >= self.current_gray.shape[1] - extent
                or y >= self.current_gray.shape[0] - extent):
            return None
        patch = cv2.getRectSubPix(self.hsv, (2 * extent + 1, 2 * extent + 1), (float(x), float(y)))
        yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1]
        core = xx ** 2 + yy ** 2 < (r * .7) ** 2
        if np.mean((patch[..., 1][core] < 120) & (patch[..., 2][core] > 80)) < .5:
            return None
        skin = skin_surface_mask(patch)
        wood = wood_surface_mask(patch)
        if skin is not None and skin[core].mean() > .08 or wood is not None and wood[core].mean() > .5:
            return None
        appearance = cv2.getRectSubPix(self.current_gray, (2 * extent + 1, 2 * extent + 1), (float(x), float(y)))
        # 中央字迹不参与身份匹配，保留圆盘边缘及其邻域。
        ring = (xx ** 2 + yy ** 2 >= (r * .7) ** 2) & (xx ** 2 + yy ** 2 <= (r + 2) ** 2)
        values = appearance[ring].astype(np.float32)
        values -= values.mean()
        length = np.linalg.norm(values)
        if length <= 1:
            return None
        template = np.zeros_like(appearance, dtype=np.float32)
        template[ring] = values / length
        return r, values / length, template, np.uint8(ring) * 255

    def _refine(self, center, track):
        size = track.template.shape[0] + 6
        patch = cv2.getRectSubPix(self.current_gray, (size, size), (float(center[0]), float(center[1])))
        scores = cv2.matchTemplate(patch.astype(np.float32), track.template,
                                  cv2.TM_CCOEFF_NORMED, mask=track.mask)
        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        _, score, _, location = cv2.minMaxLoc(scores)
        return center + np.float32(location) - 3 if score >= .65 else None

    def _features(self, center, radius):
        extent = radius + 4
        patch = cv2.getRectSubPix(self.current_gray, (extent * 2 + 1, extent * 2 + 1),
                                  (float(center[0]), float(center[1])))
        yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1]
        distance = xx ** 2 + yy ** 2
        mask = np.uint8((distance >= (radius * .7) ** 2) & (distance <= (radius + 2) ** 2)) * 255
        points = cv2.goodFeaturesToTrack(patch, 16, .01, 3, mask=mask)
        return None if points is None or len(points) < 4 else points + np.float32(center - extent)

    def observe(self, time_s, image, position, cells, *, allow_reseed=True):
        if image.shape[0] <= image.shape[1]:
            self.observation.cells = dict(cells)
            self.identities, self.tokens = {}, {}
            return {}
        if self.time_s is not None and time_s <= self.time_s:
            return {}
        known = {p.id: p for p in position.pieces}
        reference = tuple((p.id, p.color, p.kind, p.square.file, p.square.rank) for p in position.pieces)
        if self.reference is None or allow_reseed and reference != self.reference:
            self.trusted = {identity: p.model_copy(deep=True) for identity, p in known.items()}
            self.reference = reference
            for identity, track in self.tracks.items():
                if identity in self.trusted:
                    track.piece = self.trusted[identity]
        original_cells = cells
        cells = self.observation.update(image, position, cells)
        self.current_gray, self.hsv, self.edges = self.observation.gray, self.observation.hsv, self.observation.edges
        for identity, piece in self.trusted.items():
            square = piece.square.file, piece.square.rank
            if square in cells and cells[square] is None:
                self.departed.add((identity, *square))
        if allow_reseed:
            self.tracks = {identity: track for identity, track in self.tracks.items() if identity in known
                           and (track.piece.color, track.piece.kind) == (known[identity].color, known[identity].kind)}
        active = [(identity, track) for identity, track in self.tracks.items() if track.points is not None]
        if self.gray is not None and 0 < time_s - self.time_s <= .3 and active:
            points = np.concatenate([track.points for _, track in active]).astype(np.float32)
            following, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, self.current_gray, points, None,
                winSize=(15, 15), maxLevel=2)
            backward, reverse, _ = cv2.calcOpticalFlowPyrLK(self.current_gray, self.gray, following, None,
                winSize=(15, 15), maxLevel=2)
            valid = ((status.ravel() > 0) & (reverse.ravel() > 0)
                     & (np.linalg.norm(points - backward, axis=2).ravel() < 1))
            offset = 0
            for _, track in active:
                count = len(track.points)
                keep = valid[offset:offset + count]
                new_points = following[offset:offset + count][keep]
                old_points = track.points[keep]
                offset += count
                if len(new_points) < max(4, count * .5):
                    track.points = None
                    continue
                delta = (new_points - old_points)[:, 0]
                shift = np.median(delta, axis=0)
                center = self._refine(track.center + shift, track)
                face = self._face(center, track.radius) if center is not None else None
                if (np.linalg.norm(shift) > 30 or np.median(np.linalg.norm(delta - shift, axis=1)) > 1.5
                        or face is None or float(face[1] @ track.appearance) < .65):
                    track.points = None
                    continue
                if np.linalg.norm(center - track.stable_center) <= 1.5:
                    track.stable_hits += 1
                else:
                    track.stable_since, track.stable_hits = time_s, 1
                    track.stable_center = center.copy()
                track.center, track.points = center, self._features(center, track.radius)
        else:
            for _, track in active:
                track.points = None
        if self.trusted:
            for identity, piece in self.trusted.items():
                track = self.tracks.get(identity)
                if track is not None and track.points is not None:
                    continue
                square = piece.square.file, piece.square.rank
                if ((identity, *square) in self.departed
                        or square not in self.observation.centers and square in cells and cells[square] != piece.color
                        or original_cells.get(square) not in (None, piece.color)):
                    continue
                center = self.observation.centers[square][0] if square in self.observation.centers else self.observation.project(
                    [[self.observation.origin + square[0] * 50, self.observation.origin + square[1] * 50]])[0]
                face = self._face(center, track.radius if track is not None else None)
                if face is None or track is not None and float(face[1] @ track.appearance) < .65:
                    continue
                radius, appearance, template, mask = face
                points = self._features(center, radius)
                if points is not None:
                    self.generation += 1
                    self.tracks[identity] = PieceTrack(piece.model_copy(deep=True), center, radius,
                        appearance, template, mask, points, self.generation, time_s, center.copy())
        active = [(identity, track) for identity, track in self.tracks.items() if track.points is not None]
        collisions = set()
        for i, (identity, track) in enumerate(active):
            for other, candidate in active[i + 1:]:
                if np.linalg.norm(track.center - candidate.center) < max(track.radius, candidate.radius) * 1.5:
                    collisions.update((identity, other))
        proof, self.identities = {}, {}
        for identity, track in active:
            if identity in collisions:
                track.points = None
                continue
            logical = self.observation.project([track.center], inverse=True)[0]
            file, rank = np.rint((logical - self.observation.origin) / 50).astype(int)
            square = int(file), int(rank)
            if (not (0 <= file < 9 and 0 <= rank < 10)
                    or np.linalg.norm(logical - (self.observation.origin + np.array(square) * 50)) > 20
                    or track.stable_hits < 3 or time_s - track.stable_since < .12 - 1e-6
                    or square in cells and cells[square] is None):
                continue
            origin = track.piece.square.file, track.piece.square.rank
            if square != origin and cells.get(origin, 'unknown') is not None:
                continue
            proof[square], self.identities[square] = track.piece.color, identity
            if square != origin:
                from .schemas import Square
                track.piece = track.piece.model_copy(update={'square': Square(file=square[0], rank=square[1])})
                self.trusted[identity] = track.piece
        self.gray, self.time_s = self.current_gray, time_s
        self.tokens = {identity: self.tracks[identity].generation for identity in self.identities.values()}
        return proof
