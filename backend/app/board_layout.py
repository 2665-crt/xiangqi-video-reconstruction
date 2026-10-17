import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .initial_observation import board_geometry


MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/board-layout-v1.onnx'
KINDS = ('king', 'advisor', 'elephant', 'horse', 'rook', 'cannon', 'pawn')


class BoardLayoutClassifier:
    def __init__(self, path=MODEL_PATH):
        import onnxruntime as ort
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['model_sha256']:
            raise ValueError('局面核对模型文件校验失败')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    def readings(self, image):
        cell, margin = board_geometry(image)
        board = cv2.getRectSubPix(image, (round(8 * cell), round(9 * cell)),
                                 (float(margin + 4 * cell), float(margin + 4.5 * cell)))
        rgb = cv2.cvtColor(cv2.resize(board, (280, 315)), cv2.COLOR_BGR2RGB).astype(np.float32)
        values = (rgb - np.float32([123.675, 116.28, 103.53])) / np.float32([58.395, 57.12, 57.375])
        scores = self.session.run(None, {'input': values.transpose(2, 0, 1)[None]})[0][0]
        result = {}
        for index, probabilities in enumerate(scores):
            order = np.argsort(probabilities)
            label = int(order[-1])
            if label < 2:
                continue
            result[index % 9, index // 9] = {
                'color': 'red' if label < 9 else 'black', 'prediction': KINDS[(label - 2) % 7],
                'score': float(probabilities[label]), 'margin': float(probabilities[label] - probabilities[order[-2]])}
        return result


@lru_cache(maxsize=1)
def local_board_layout():
    return BoardLayoutClassifier() if MODEL_PATH.is_file() else None
