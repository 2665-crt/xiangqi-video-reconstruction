import argparse
import json
import os
from pathlib import Path

import httpx

from app.storage import TaskStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--end', type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / 'takeback-35s-20261016'
    output.mkdir(parents=True, exist_ok=True)
    store = TaskStore(root / 'data')
    original = store.get('55d7e7d55dca42b99befed7c33657ec8')
    task = store.create(f'黑方悔棋识别复测 · 1.mp4（18～{args.end:g}秒）', original.metadata)
    task.start_s, task.end_s = original.start_s, args.end
    task.calibration = original.calibration.model_copy(deep=True)
    task.calibration_confirmed = original.calibration_confirmed
    task.initial_position = original.initial_position.model_copy(deep=True)
    task.initial_turn_known = original.initial_turn_known
    task.replay_mode, task.rules_enabled = 'actual', True
    task = store.save(task, task.revision)
    os.link(store.task_dir(original.id) / 'source.mp4', store.task_dir(task.id) / 'source.mp4')
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(f'http://127.0.0.1:8290/api/tasks/{task.id}/analyze',
                               params={'local_only': 'true', 'actual': 'true'})
        response.raise_for_status()
    state = {'task_id': task.id, 'original_task_id': original.id,
             'range_s': [task.start_s, task.end_s], 'http_status': response.status_code}
    (output / f'run-{args.end:g}s.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()
