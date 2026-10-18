import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from .initial_observation import board_geometry
from .piece_classifier import KINDS


def detector_distribution(scores, names):
    probabilities = {f'{color}_{kind}': 0. for color in ('red', 'black') for kind in KINDS}
    for index, value in enumerate(scores):
        name = names[index]
        if name == 'board':
            continue
        color, kind = name.split('_')
        kind = {'Bishop': 'elephant', 'Knight': 'horse'}.get(kind, kind.lower())
        probabilities[f'{color.lower()}_{kind}'] = float(value)
    total = sum(probabilities.values())
    return {label: value / total for label, value in probabilities.items()} if total else {}


class DeepPieceDetector:
    def __init__(self):
        import torch
        folder = Path(__file__).resolve().parents[1] / 'models/fusion'
        cache = folder.parents[2] / 'data/ultralytics'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(cache)
        from ultralytics import YOLO
        manifest = json.loads((folder / 'weights.json').read_text(encoding='utf8'))
        torch.set_num_threads(4)
        self.models = []
        for filename, digest in manifest['sha256'].items():
            path = folder / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('融合识别权重校验失败')
            self.models.append(YOLO(str(path)))
        self.lock = threading.Lock()

    def readings(self, image):
        import torch
        from ultralytics.utils import ops
        cell, margin = board_geometry(image)
        readings = {}
        with self.lock, torch.inference_mode():
            for model in self.models:
                head = model.model.model[-1]
                captured = []
                handle = head.register_forward_hook(lambda layer, inputs, output: captured.append(output))
                try:
                    result = model.predict(image, conf=.2, device='cpu', verbose=False, save=False)[0]
                finally:
                    handle.remove()
                raw = captured[-1][1]['one2one']
                decoded = head._inference(raw)[0].transpose(0, 1)
                stride = int(head.stride[0])
                shape = (raw['feats'][0].shape[2]*stride, raw['feats'][0].shape[3]*stride)
                boxes = ops.scale_boxes(shape, decoded[:, :4].clone(), image.shape[:2]).numpy()
                for box in result.boxes:
                    if model.names[int(box.cls[0])] == 'board':
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].numpy()
                    file, rank = round(((x1+x2)/2-margin)/cell), round(((y1+y2)/2-margin)/cell)
                    if not (0 <= file < 9 and 0 <= rank < 10):
                        continue
                    if abs((x1+x2)/2-margin-file*cell) > .4*cell or abs((y1+y2)/2-margin-rank*cell) > .4*cell:
                        continue
                    intersection = np.maximum(0, np.minimum(boxes[:, 2], x2)-np.maximum(boxes[:, 0], x1)) * np.maximum(
                        0, np.minimum(boxes[:, 3], y2)-np.maximum(boxes[:, 1], y1))
                    area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
                    iou = intersection / np.maximum(area+(x2-x1)*(y2-y1)-intersection, 1e-8)
                    anchor = int(np.argmax(iou))
                    if iou[anchor] < .5:
                        continue
                    probabilities = detector_distribution(decoded[anchor, 4:].numpy(), model.names)
                    confidence = float(box.conf[0])
                    square = file, rank
                    previous = readings.get(square)
                    if previous is None or confidence >= .85 and previous['confidence'] < .7:
                        readings[square] = {'class_probabilities': probabilities, 'confidence': confidence,
                                            'bbox': [float(x1), float(y1), float(x2), float(y2)]}
        return readings
