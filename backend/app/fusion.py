from functools import lru_cache

import cv2
import numpy as np

from .piece_classifier import KINDS, glyph_patch, piece_color, local_piece_classifier
from .schemas import Piece, Position, Square


def fuse_probabilities(cnn, deep, cnn_score, deep_confidence):
    labels = [f'{color}_{kind}' for color in ('red', 'black') for kind in KINDS]
    sources = []
    if cnn and sum(cnn.values()) > 0:
        sources.append((cnn, .2 if deep and cnn_score < .6 and deep_confidence >= .8 else .8))
    if deep and sum(deep.values()) > 0:
        sources.append((deep, .8 if cnn_score < .6 and deep_confidence >= .8 else .2))
    if not sources:
        return {}
    total_weight = sum(weight for _, weight in sources)
    return {label: float(sum(weight * values.get(label, 0.) / sum(values.values())
                             for values, weight in sources) / total_weight) for label in labels}


class FusionPieceClassifier:
    supports_reference = True

    def __init__(self, cnn, detector):
        self.cnn, self.detector = cnn, detector
        self.usage = []
        self.last_request_count = self.last_image_count = 0
        self.last_response = None

    def recognize_position(self, images, reference=None, actual=False):
        from .initial_observation import observe_initial_board, compatible_observations, merge_initial_observations
        if not images:
            return []
        if reference is not None:
            # A reference request includes the state before the move; only the final image describes its result.
            images = images[-1:]
        observations = [observe_initial_board(image) for image in images]
        changed = False
        for index in range(len(images) - 2, -1, -1):
            if not compatible_observations(observations[index][0], observations[index + 1][0]):
                images, observations = images[index + 1:], observations[index + 1:]
                changed = True
                break
        self.last_response = {'source': 'local_fusion', 'readings': [], 'piece_reviews': [],
                              'probability_semantics': 'normalized model scores, not calibrated empirical accuracy',
                              'display_policy': 'maximum_category_probability', 'model_sha256': self.cnn.metadata['model_sha256']}
        by_square = {}
        for frame_index, (image, (cells, faces)) in enumerate(zip(images, observations)):
            detected = self.detector.readings(image) if self.detector is not None else {}
            for square, observed_color in cells.items():
                if observed_color is None:
                    continue
                face = glyph_patch(image, *square, None, faces.get(square))
                color = piece_color(face)
                gray = face[:, :, 1] if color == 'red' else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cnn = self.cnn.classify(gray, color)
                deep = detected.get(square, {})
                deep_probabilities = deep.get('class_probabilities', {})
                deep_winner = max(deep_probabilities, key=deep_probabilities.get) if deep_probabilities else None
                probabilities = fuse_probabilities(cnn.get('class_probabilities', {}), deep_probabilities,
                                                   cnn['score'], deep.get('confidence', 0.))
                row = {'file': square[0], 'rank': square[1], 'frame': frame_index,
                       'class_probabilities': probabilities, 'cnn_probabilities': cnn.get('class_probabilities', {}),
                       'deep_probabilities': deep_probabilities, 'cnn_score': cnn['score'],
                       'deep_confidence': deep.get('confidence', 0.), 'ink_color': color,
                       'color_conflict': deep_winner is not None and not deep_winner.startswith(color + '_')}
                by_square.setdefault(square, []).append(row)
        cells, ambiguous = merge_initial_observations([o[0] for o in observations])
        pieces, reviews = [], []
        for square, color in cells.items():
            if color is None:
                continue
            rows = by_square.get(square, [])
            usable = [r['class_probabilities'] for r in rows if r['class_probabilities']]
            probabilities = {label: float(np.mean([p[label] for p in usable])) for label in usable[0]} if usable else {}
            color = rows[-1]['ink_color'] if rows else color
            compatible = {label: value for label, value in probabilities.items() if label.startswith(color + '_') and value > 0}
            winner = max(compatible, key=compatible.get) if compatible else None
            selected_color, kind = winner.split('_', 1) if winner else (color, None)
            score = probabilities[winner] if winner else 0.
            if winner:
                pieces.append(Piece(id=f'{selected_color}_{kind}_{square[0]}_{square[1]}', color=selected_color,
                                    kind=kind, square=Square(file=square[0], rank=square[1])))
            self.last_response['readings'].append({'file': square[0], 'rank': square[1],
                'color': selected_color, 'prediction': kind, 'score': score,
                'class_probabilities': probabilities, 'frames': rows})
            disagreement = len({max(p, key=p.get) for p in usable}) > 1
            color_conflict = any(r['color_conflict'] for r in rows)
            if kind is None or score < .9 or disagreement or color_conflict:
                reviews.append({'square': {'file': square[0], 'rank': square[1]}, 'color': selected_color,
                                'prediction': kind, 'score': score,
                                'reason': '棋面颜色与模型颜色冲突，已保留棋面颜色，请核对' if color_conflict else '棋面无法辨认' if kind is None else '已显示最大概率棋种，相邻画面仍需核对' if disagreement else '已显示最大概率棋种'})
        reviews.extend({'square': {'file': f, 'rank': r}, 'color': None, 'prediction': None, 'score': 0.,
                        'reason': '占位不确定', 'state': 'occupancy'}
                       for r in range(10) for f in range(9) if (f, r) not in cells)
        position = Position(pieces=pieces, side_to_move=reference.side_to_move if reference is not None else 'red')
        self.last_response['draft_position'] = position.model_dump()
        self.last_response['piece_reviews'] = reviews
        if any(r['color_conflict'] for rows in by_square.values() for r in rows):
            self.last_response['error'] = '棋面颜色与模型颜色存在冲突，请核对并保存初始棋局'
            return []
        if changed or ambiguous or len(cells) != 90:
            self.last_response['error'] = '已显示最大概率棋种，起始画面存在局面变化或占位不确定，请核对'
            return []
        if len(pieces) != sum(color is not None for color in cells.values()):
            self.last_response['error'] = '已保留可识别棋子，仍有棋面无法辨认'
            return []
        return [position]


@lru_cache(maxsize=1)
def local_fusion_classifier():
    from .deep_detector import DeepPieceDetector
    return FusionPieceClassifier(local_piece_classifier(), DeepPieceDetector())
