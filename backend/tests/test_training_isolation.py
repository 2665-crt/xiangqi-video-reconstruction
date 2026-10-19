import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize('multiple_sources', [False, True])
def test_color_training_writes_versioned_candidate_without_replacing_active_model(tmp_path, monkeypatch, multiple_sources):
    script = Path(__file__).resolve().parents[2] / 'scripts/train-board-colors.py'
    spec = importlib.util.spec_from_file_location('train_board_colors', script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    models = tmp_path / 'backend/models'
    models.mkdir(parents=True)
    active = models / 'board-color-v1.npz'
    active.write_bytes(b'previously validated active model')
    source = tmp_path / 'training.mp4'
    source.write_bytes(b'video bytes decoded by the test frame provider')
    hsv = np.empty((80, 240, 3), dtype=np.uint8)
    hsv[:, :80] = (25, 90, 180)
    hsv[:, 80:160] = (10, 110, 210)
    hsv[:, 160:] = (0, 0, 70)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    monkeypatch.setattr(trainer, 'VIDEO', source)
    monkeypatch.setattr(trainer, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(trainer, 'MODELS', models, raising=False)
    monkeypatch.setattr(trainer, 'REGIONS', {0: {'wood': [(0, 0, 80, 80)],
                                               'skin': [(80, 0, 160, 80)], 'other': [(160, 0, 240, 80)]}})
    monkeypatch.setattr(trainer, 'iter_frames', lambda *args: (item for item in [(0, frame)]))
    manifest = None
    if multiple_sources:
        another = tmp_path / 'another.mp4'
        another.write_bytes(b'another independently labelled video')
        manifest = tmp_path / 'labels.json'
        manifest.write_text(json.dumps({'sources': [{'video': str(video), 'regions': trainer.REGIONS}
                                                   for video in (source, another)]}), encoding='utf8')
    trainer.main(manifest) if manifest else trainer.main()
    assert active.read_bytes() == b'previously validated active model'
    reports = list((tmp_path / 'output/candidates').glob('*/training-report.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding='utf8'))
    assert report['activation_status'] == 'candidate_requires_regression'
    candidate = reports[0].parent / 'board-color-v1.npz'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == report['lookup_sha256']
    if multiple_sources:
        assert {s['video'] for s in report['training_sources']} == {str(source), str(another)}
        assert len(report['annotations']) == 6


def test_candidate_color_model_can_be_validated_without_replacing_active_files(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.npz'
    np.savez_compressed(candidate, wood=np.ones((180, 64, 64), dtype=bool),
                        skin=np.zeros((180, 64, 64), dtype=bool))
    monkeypatch.setenv('BOARD_COLOR_MODEL_PATH', str(candidate))
    script = Path(__file__).resolve().parents[1] / 'app/board_colors.py'
    spec = importlib.util.spec_from_file_location('candidate_board_colors', script)
    colors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colors)
    assert colors.MODEL_PATH == candidate
    assert colors.wood_surface_mask(np.zeros((1, 1, 3), dtype=np.uint8)).all()
