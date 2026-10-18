import argparse
import json
import os
from pathlib import Path

import httpx

from app.storage import TaskStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='final')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / 'color-adaptation-20261016'
    store = TaskStore(root / 'data')
    original = store.get('55d7e7d55dca42b99befed7c33657ec8')
    task = store.create(f'颜色模型适配复测 {args.tag} · 1.mp4（18～400秒）', original.metadata)
    task.start_s, task.end_s = original.start_s, original.end_s
    task.calibration = original.calibration.model_copy(deep=True)
    task.calibration_confirmed = original.calibration_confirmed
    task.initial_position = original.initial_position.model_copy(deep=True)
    task.initial_turn_known = original.initial_turn_known
    task.replay_mode, task.rules_enabled = original.replay_mode, original.rules_enabled
    task = store.save(task, task.revision)
    os.link(store.task_dir(original.id) / 'source.mp4', store.task_dir(task.id) / 'source.mp4')
    summary = {'original_task_id': original.id, 'new_task_id': task.id,
               'baseline_moves': sum(r.move is not None for r in original.moves),
               'baseline_records': len(original.moves),
               'baseline_run_stats': original.run_stats.model_dump(),
               'training_video': '2.mp4', 'independent_rerun_video': '1.mp4',
               'range_s': [task.start_s, task.end_s], 'api': 'http://127.0.0.1:8290'}
    state_path = output / f'rerun-state-{args.tag}.json'
    state_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with httpx.Client(timeout=30, trust_env=False) as client:
        health = client.get('http://127.0.0.1:8290/api/health')
        health.raise_for_status()
        response = client.post(f'http://127.0.0.1:8290/api/tasks/{task.id}/analyze',
                               params={'local_only': 'true', 'actual': 'true'})
        response.raise_for_status()
        summary['health_status'] = health.status_code
        summary['analysis_http_status'] = response.status_code
        state_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
