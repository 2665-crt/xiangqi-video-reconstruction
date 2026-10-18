import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.video import iter_frames


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).resolve().parents[1]
OUTPUT = EXPERIMENT / 'output' / 'color-adaptation-20261016'
VIDEO = ROOT / '直播录屏' / '2.mp4'

# Rectangles were reviewed on original frames, not inferred from model predictions.
REGIONS = {
    10: {
        'wood': [(275, 638, 310, 655), (355, 638, 390, 655), (195, 790, 225, 805),
                 (390, 785, 420, 803), (290, 750, 320, 766), (255, 602, 272, 617)],
        'skin': [(293, 451, 322, 462), (348, 450, 374, 463)],
        'other': [(120, 533, 153, 557), (332, 849, 389, 878), (50, 600, 65, 650),
                  (250, 350, 320, 400)],
    },
    18: {
        'wood': [(275, 638, 310, 655), (355, 638, 390, 655), (195, 790, 225, 805),
                 (290, 750, 320, 766)],
        'skin': [(425, 850, 520, 875), (565, 795, 630, 825), (355, 879, 385, 895)],
        'other': [(120, 533, 153, 557), (50, 600, 65, 650), (250, 350, 320, 400)],
    },
    60: {
        'wood': [(275, 638, 310, 655), (195, 790, 225, 805), (390, 785, 420, 803),
                 (173, 550, 202, 565)],
        'skin': [(372, 550, 391, 574), (632, 748, 678, 767), (540, 920, 570, 945)],
        'other': [(50, 600, 65, 650), (250, 350, 320, 400), (330, 710, 355, 725)],
    },
}


def main(manifest=None):
    sources = (json.loads(Path(manifest).read_text(encoding='utf8'))['sources'] if manifest is not None
               else [{'video': str(VIDEO), 'regions': REGIONS}])
    for source in sources:
        with Path(source['video']).open('rb') as stream:
            source['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    dataset_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode('utf8')).hexdigest()
    candidate = OUTPUT / 'candidates' / f'{dataset_hash[:12]}-{uuid4().hex[:12]}'
    candidate.mkdir(parents=True)
    rng = np.random.default_rng(20261016)
    labels = {'other': 0, 'wood': 1, 'skin': 2}
    samples, targets, annotations = [], [], []
    for source_index, source in enumerate(sources):
        for second, groups in source['regions'].items():
            frames = iter_frames(Path(source['video']), float(second), float(second) + .2)
            time_s, image = next(frames)
            frames.close()
            annotated = image.copy()
            for name, rectangles in groups.items():
                for x1, y1, x2, y2 in rectangles:
                    patch = image[y1:y2, x1:x2]
                    pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                    if name == 'wood':
                        pixels = pixels[(pixels[:, 1] > 30) & (pixels[:, 2] > 100)]
                    elif name == 'other':
                        pixels = pixels[(pixels[:, 1] < 25) | (pixels[:, 2] < 100)
                                        | ((pixels[:, 0] > 35) & (pixels[:, 0] < 90))
                                        | (((pixels[:, 0] < 5) | (pixels[:, 0] > 150)) & (pixels[:, 1] > 100))]
                    pixels = pixels[rng.choice(len(pixels), min(len(pixels), 1500), replace=False)]
                    samples.append(pixels)
                    targets.extend([labels[name]] * len(pixels))
                    annotations.append({'video': source['video'], 'time_s': time_s, 'rectangle': [x1, y1, x2, y2],
                                        'label': name, 'pixels': len(pixels)})
                    color = {'wood': (0, 255, 0), 'skin': (0, 0, 255), 'other': (255, 0, 0)}[name]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, name, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
            cv2.imencode('.jpg', annotated)[1].tofile(str(candidate / f'labels-{source_index + 1}-{second}s.jpg'))
    x = np.concatenate(samples).astype(np.float32)
    y = np.array(targets, np.int32)
    cv2.setRNGSeed(20261016)
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(12)
    model.setMaxCategories(3)
    model.setActiveVarCount(3)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 64, 0))
    model.train(x, cv2.ml.ROW_SAMPLE, y)
    model_path = candidate / 'board-color-v1.xml'
    storage = cv2.FileStorage('', cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    model.write(storage)
    model_path.write_text(storage.releaseAndGetString(), encoding='utf-8')
    h, s, v = np.meshgrid(np.arange(180), np.arange(64) * 4 + 2,
                         np.arange(64) * 4 + 2, indexing='ij')
    lookup = np.stack([h, s, v], axis=-1).reshape(-1, 3).astype(np.float32)
    votes = model.getVotes(lookup, 0)
    classes, votes = votes[0], votes[1:]
    total = votes.sum(axis=1)
    wood = votes[:, list(classes).index(1)] / total >= .98
    skin = votes[:, list(classes).index(2)] / total >= .9
    np.savez_compressed(candidate / 'board-color-v1.npz', wood=wood.reshape(180, 64, 64),
                        skin=skin.reshape(180, 64, 64))
    prediction = model.predict(x)[1].ravel()
    report = {
        'algorithm': 'supervised HSV random forest; 64 trees; conservative wood lookup at >=0.98 votes',
        'training_video': sources[0]['video'] if len(sources) == 1 else None,
        'training_video_sha256': sources[0]['sha256'] if len(sources) == 1 else None,
        'dataset_sha256': dataset_hash,
        'training_sources': [{'video': s['video'], 'sha256': s['sha256'], 'frames_s': list(s['regions'])} for s in sources],
        'candidate_directory': str(candidate), 'activation_status': 'candidate_requires_regression',
        'training_frames_s': list(sources[0]['regions']) if len(sources) == 1 else None, 'pixel_samples': len(y),
        'counts': {name: int((y == label).sum()) for name, label in labels.items()},
        'training_pixel_accuracy': float((prediction == y).mean()),
        'training_metric_is_not_independent_validation': True,
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'lookup_sha256': hashlib.sha256((candidate / 'board-color-v1.npz').read_bytes()).hexdigest(),
        'annotations': annotations,
        'independent_video': '1.mp4 was not used to fit the model',
    }
    (candidate / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'annotations'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path)
    main(parser.parse_args().manifest)
