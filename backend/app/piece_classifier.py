import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .schemas import Piece, Position, Square


KINDS = ('king', 'rook', 'horse', 'cannon', 'advisor', 'elephant', 'pawn')
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'piece-kind-cnn-v1.onnx'


@lru_cache(maxsize=64)
def _piece_face_mask(h, w):
    yy, xx = np.mgrid[:h, :w]
    face = (xx-(w-1)/2)**2 + (yy-(h-1)/2)**2 < (min(h, w)*.35)**2
    face.setflags(write=False)
    return face


def piece_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[_piece_face_mask(*image.shape[:2])]
    red = ((pixels[:, 0] < 10) | (pixels[:, 0] > 150)) & (pixels[:, 1] > 45) & (pixels[:, 2] > 70)
    return 'red' if red.mean() > .045 else 'black'


def glyph_image(gray):
    if np.ptp(gray) < 5:
        return np.zeros((48, 48), np.float32)
    h, w = gray.shape
    face = gray[h//4:3*h//4, w//4:3*w//4]
    background = float(np.percentile(face, 75))
    contrast = max(30., background-float(np.percentile(face, 10)))
    ink = np.clip((background-gray.astype(np.float32))/contrast, 0., 1.)
    return cv2.resize(ink, (48, 48), interpolation=cv2.INTER_AREA)


def glyph_patch(image, file, rank, color, face_center=None):
    from .initial_observation import board_geometry
    cell, margin = board_geometry(image)
    board = cv2.resize(image, None, fx=100 / cell, fy=100 / cell)
    board = cv2.copyMakeBorder(board, 48, 48, 48, 48, cv2.BORDER_REPLICATE)
    x, y = round(48 + margin * 100 / cell + file*100), round(48 + margin * 100 / cell + rank*100)
    face = board[y-47:y+48, x-47:x+48]
    pixels = face.astype(np.float32)
    neutral = np.clip((pixels[:, :, 0]/np.maximum(pixels[:, :, 2], 1)-.70)/.25, 0, 1)
    weights = neutral*np.clip((pixels.min(axis=2)-110)/70, 0, 1)
    mass = float(weights.sum())
    cx, cy = 47, 35
    if mass > 500:
        # Smooth weights avoid moving the ROI when shadows cross a binary cutoff.
        yy, xx = np.mgrid[:95, :95]
        cx, cy = float((weights*xx).sum()/mass), float((weights*yy).sum()/mass)-6
    cx, cy = min(65, max(30, cx)), min(60, max(25, cy))
    gray = board if color is None else board[:, :, 1] if color == 'red' else cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    center = (float(x-47+cx), float(y-47+cy)) if face_center is None else (
        float(face_center[0] * 100 / cell + 48), float(face_center[1] * 100 / cell + 48))
    patch = cv2.getRectSubPix(gray, (96, 96), center)
    yy, xx = np.mgrid[:96, :96]
    patch[(xx-47.5)**2/46**2 + (yy-47.5)**2/44**2 > 1] = 255
    return patch


class PieceKindClassifier:
    def __init__(self, path=MODEL_PATH):
        import onnxruntime as ort
        path = Path(path)
        self.metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        if self.metadata['classes'] != list(KINDS) or hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata['model_sha256']:
            raise ValueError('棋种分类器文件校验失败，请恢复匹配的模型和元数据')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.usage = []
        self.last_request_count = self.last_image_count = 0
        self.last_response = None
        self.layout_classifier = None

    def classify(self, gray, color):
        image = glyph_image(gray)
        height, width = gray.shape
        interior = gray[height // 4:3 * height // 4, width // 4:3 * width // 4]
        lighting = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 4)
        yy, xx = np.mgrid[:height, :width]
        center = (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2 < (min(height, width) * .27) ** 2
        stroke_ratio = float((lighting[center] - gray[center] > 10).mean())
        if (not np.any(image) or np.percentile(interior, 95) - np.percentile(interior, 5) < 10
                or stroke_ratio < .02):
            return {'kind': None, 'prediction': None, 'score': 0., 'margin': 0., 'class_probabilities': {}}
        views = []
        for shift in (0, -3, 3):
            shifted = image if not shift else cv2.warpAffine(image, np.float32([[1, 0, 0], [0, 1, shift]]),
                                                            (48, 48), borderMode=cv2.BORDER_CONSTANT)
            batch = np.stack([np.rot90(shifted, k) for k in range(4)])[:, None]
            logits = self.session.run(None, {'glyph': batch})[0]
            if self.metadata.get('output_colors'):
                logits = logits.reshape(-1, 2, 7)[:, self.metadata['output_colors'].index(color)]
            probabilities = np.exp(logits-logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            views.append(probabilities.mean(axis=0))
        scores = views[0]
        order = np.argsort(scores)
        score, margin = float(scores[order[-1]]), float(scores[order[-1]]-scores[order[-2]])
        prediction = KINDS[order[-1]]
        stable = all(int(np.argmax(view)) == int(order[-1]) for view in views)
        accepted = (self.metadata.get('automatic_acceptance_enabled', False) and stable
                    and score >= self.metadata['minimum_probability'] and margin >= self.metadata['minimum_margin'])
        distribution = {f'{side}_{kind}': float(scores[index]) if side == color else 0.
                        for side in ('red', 'black') for index, kind in enumerate(KINDS)}
        return {'kind': prediction if accepted else None, 'prediction': prediction, 'score': score, 'margin': margin,
                'class_probabilities': distribution,
                'reason': '' if stable else '取样位置变化时棋种不一致'}

    def recognize_position(self, images, reference=None, actual=False):
        from .initial_observation import observe_initial_board, compatible_observations, merge_initial_observations
        self.last_response = {'source': 'local_piece_classifier', 'model_sha256': self.metadata['model_sha256'],
                              'automatic_acceptance_enabled': self.metadata.get('automatic_acceptance_enabled', False),
                              'readings': [], 'uncertain_squares': []}
        if not self.metadata.get('automatic_acceptance_enabled', False):
            self.last_response['error'] = '本地棋种分类器尚未通过完整实物验收（最低90%），请逐枚核对并设置初始棋种；不会自动确认或调用云端重试'
        if reference is not None or not images:
            return []
        observations_and_faces = [observe_initial_board(image) for image in images]
        changed = False
        for index in range(len(images) - 2, -1, -1):
            if not compatible_observations(observations_and_faces[index][0], observations_and_faces[index + 1][0]):
                images = images[index + 1:]
                observations_and_faces = observations_and_faces[index + 1:]
                changed = True
                break
        readings_by_square, observations = {}, []
        for image, (cells, faces) in zip(images, observations_and_faces):
            observations.append(cells)
            layout = self.layout_classifier.readings(image) if self.layout_classifier is not None else {}
            for (file, rank), color in cells.items():
                if color is None:
                    continue
                face = glyph_patch(image, file, rank, None, faces.get((file, rank)))
                color = piece_color(face)
                gray = face[:, :, 1] if color == 'red' else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = self.classify(gray, color)
                corroboration = layout.get((file, rank))
                if (corroboration and corroboration['score'] >= .5 and
                        (corroboration['color'], corroboration['prediction']) != (color, result['prediction'])):
                    result['kind'] = None
                    result['reason'] = '局部字形与整盘模型识别不一致，暂不复刻该位置棋种，请核对原画面'
                    result['alternative'] = corroboration
                    result['prediction'] = None
                reading = {'file': file, 'rank': rank, 'color': color, **result}
                self.last_response['readings'].append(reading)
                readings_by_square.setdefault((file, rank), []).append(reading)
                if result['kind'] is None:
                    self.last_response['uncertain_squares'].append([file, rank])
        cells, ambiguous = merge_initial_observations(observations)
        draft_pieces, reviews = [], []
        for (file, rank), cell_color in cells.items():
            if cell_color is None:
                continue
            readings = readings_by_square[file, rank]
            # 草稿采用同一占位的最清晰识别；自动确认仍要求所有帧一致且达标。
            latest_color = readings[-1]['color']
            best = max((r for r in readings if r['color'] == latest_color), key=lambda r: r['score'])
            prediction = best.get('prediction', best['kind'])
            square = Square(file=file, rank=rank)
            if prediction is not None:
                draft_pieces.append(Piece(id=f'{latest_color}_{prediction}_{file}_{rank}', color=latest_color,
                                          kind=prediction, square=square))
            disagreement = len({(r['color'], r.get('prediction', r['kind'])) for r in readings}) > 1
            if any(r['kind'] is None for r in readings) or disagreement:
                reason = ('相邻画面识别不一致' if disagreement else
                          next((r['reason'] for r in readings if r.get('reason')), '字形无法辨认' if prediction is None else '棋种置信度不足'))
                reviews.append({'square': square.model_dump(), 'color': latest_color, 'prediction': prediction,
                                'score': best['score'], 'reason': reason})
        self.last_response['draft_position'] = Position(pieces=draft_pieces, side_to_move='red').model_dump()
        self.last_response['piece_reviews'] = reviews
        completed = []
        for piece in draft_pieces:
            square = piece.square.file, piece.square.rank
            if square not in observations[-1]:
                completed.append(square)
                if not any(r['square'] == piece.square.model_dump() for r in reviews):
                    reviews.append({'square': piece.square.model_dump(), 'color': piece.color,
                                    'prediction': piece.kind, 'score': 0., 'reason': '相邻帧补全，起始时刻该位置不可见，请核对'})
        if changed:
            for piece in draft_pieces:
                if not any(r['square'] == piece.square.model_dump() for r in reviews):
                    reviews.append({'square': piece.square.model_dump(), 'color': piece.color,
                                    'prediction': piece.kind, 'score': 0., 'reason': '相邻画面发生局面变化，仅保留最后局面，请核对'})
        unknown = [{'square': {'file': file, 'rank': rank}, 'color': None, 'prediction': None,
                    'score': 0., 'reason': '占位不确定，无法判断有无棋子', 'state': 'occupancy'}
                   for rank in range(10) for file in range(9) if (file, rank) not in cells]
        self.last_response['piece_reviews'].extend(unknown)
        for review in unknown:
            if (review['square']['file'], review['square']['rank']) in ambiguous:
                review['reason'] = '相邻画面可能发生同色棋子移动，旧位置占位未知，未合并旧棋子'
        if len(cells) != 90:
            self.last_response['error'] = f'已保留部分识别结果，当前有{len(unknown)}个位置占位不确定，请核对标记位置'
            return []
        if completed:
            self.last_response['error'] = f'已结合相邻画面复刻棋子，起始时刻有{len(completed)}处需要核对'
            return []
        if changed:
            self.last_response['error'] = '相邻画面发生局面变化；已保留最后局面，请核对起始时刻'
            return []
        if self.last_response['uncertain_squares']:
            if 'error' not in self.last_response:
                squares = sorted({tuple(square) for square in self.last_response['uncertain_squares']})
                locations = '、'.join(f'第{file+1}列第{rank+1}行' for file, rank in squares)
                self.last_response['error'] = f'棋种置信度不足：{locations}；已保留候选棋局，请在右侧核对标记棋子并保存'
            return []
        if any(len({(r['color'], r['kind']) for r in readings}) > 1 for readings in readings_by_square.values()):
            self.last_response['error'] = '连续画面的棋种识别不一致；已保留候选棋局，请在右侧核对标记棋子并保存'
            return []
        position = Position(pieces=draft_pieces, side_to_move='red')
        if actual:
            return [position]
        from .vision import parse_positions
        payload = {'candidates': [{'pieces': [{'color': p.color, 'kind': p.kind,
                     'file': p.square.file, 'rank': p.square.rank} for p in position.pieces], 'side_to_move': None}]}
        return parse_positions(json.dumps(payload))


@lru_cache(maxsize=1)
def local_piece_classifier():
    if not MODEL_PATH.is_file():
        return None
    from .board_layout import local_board_layout
    classifier = PieceKindClassifier()
    classifier.layout_classifier = local_board_layout()
    return classifier
