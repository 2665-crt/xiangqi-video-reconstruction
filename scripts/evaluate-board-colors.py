import json
import os
from pathlib import Path

import cv2
import numpy as np

from app.actual import _tracking_cells
from app.board_colors import wood_surface_mask
from app.schemas import opening_position
from app.video import iter_frames


def main():
    experiment = Path(__file__).resolve().parents[1]
    root = experiment
    output = experiment / 'output' / 'color-adaptation-20261016'
    frames = iter_frames(root / '直播录屏' / '2.mp4', 25, 25.2)
    time_s, image = next(frames)
    frames.close()
    validation = {'wood': [(275, 638, 310, 655), (355, 638, 390, 655), (195, 790, 225, 805)],
                  'skin': [(500, 920, 550, 940), (650, 825, 680, 845)]}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = wood_surface_mask(hsv)
    rows = []
    annotated = image.copy()
    for label, regions in validation.items():
        for x1, y1, x2, y2 in regions:
            patch = mask[y1:y2, x1:x2]
            rows.append({'label': label, 'rectangle': [x1, y1, x2, y2],
                         'wood_predictions': int(patch.sum()), 'pixels': patch.size})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0) if label == 'wood' else (0, 0, 255), 2)
    cv2.imencode('.jpg', annotated)[1].tofile(str(output / 'heldout-25s-labels.jpg'))
    positive = [r for r in rows if r['label'] == 'wood']
    negative = [r for r in rows if r['label'] == 'skin']
    report = {'heldout_time_s': time_s, 'training_times_s': [10, 18, 60], 'regions': rows,
              'wood_recall': sum(r['wood_predictions'] for r in positive) / sum(r['pixels'] for r in positive),
              'skin_misclassified_as_wood': sum(r['wood_predictions'] for r in negative) / sum(r['pixels'] for r in negative)}
    fixture = experiment / 'backend/tests/fixtures/board-colors/warm-board.png'
    original = cv2.imdecode(np.frombuffer(fixture.read_bytes(), np.uint8), 1)
    reference = opening_position()
    expected = {(p.square.file, p.square.rank): p.color for p in reference.pieces}
    report['independent_video_1_initial_frame'] = {}
    for enabled in ['0', '1']:
        os.environ['BOARD_COLOR_MODEL'] = enabled
        cells = _tracking_cells(original, reference)
        report['independent_video_1_initial_frame'][enabled] = {
            'known_cells': len(cells),
            'visible_mismatches': len([s for s, color in cells.items() if expected.get(s) != color]),
        }
    (output / 'heldout-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
