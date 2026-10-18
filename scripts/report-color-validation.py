import hashlib
import argparse
import json
from pathlib import Path

import httpx

from app.storage import TaskStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='validated')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / 'color-adaptation-20261016'
    state = json.loads((output / f'rerun-state-{args.tag}.json').read_text(encoding='utf-8'))
    store = TaskStore(root / 'data')
    task = store.get(state['new_task_id'])
    if task.status in ('analyzing', 'waiting_service'):
        raise RuntimeError('The video rerun has not completed yet')
    original = store.get(state['original_task_id'])
    records = [r for r in task.moves if r.move is not None]
    truth = json.loads((output / 'manual-ground-truth.json').read_text(encoding='utf-8'))['moves']
    actual = [{'piece_id': r.move.piece_id, 'from': [r.move.source.file, r.move.source.rank],
               'to': [r.move.target.file, r.move.target.rank]} for r in records]
    prefix = 0
    for expected, observed in zip(truth, actual):
        if expected != observed:
            break
        prefix += 1
    sample_rows = []
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f'http://127.0.0.1:8290/api/tasks/{task.id}')
        response.raise_for_status()
        for record in records[:8]:
            ids = record.evidence_ids
            result = client.get(f'http://127.0.0.1:8290/api/tasks/{task.id}/evidence/{ids[-1]}')
            result.raise_for_status()
            sample_rows.append({'record': record.index, 'piece_id': record.move.piece_id,
                                'evidence_http': result.status_code,
                                'evidence_path': str(store.task_dir(task.id) / 'evidence' / ids[-1]),
                                'time_s': record.video_end_s, 'board_check': record.board_check.status})
    report = {'baseline_task_id': original.id, 'rerun_task_id': task.id,
              'status': task.status, 'stage': task.stage, 'error': task.error,
              'range_s': [task.start_s, task.end_s], 'processed_video_s': task.processed_video_s,
              'scan_complete': task.scan_complete, 'baseline_moves': sum(r.move is not None for r in original.moves),
              'rerun_moves': len(records), 'review_records': sum(r.status == 'needs_review' for r in task.moves),
              'run_stats': task.run_stats.model_dump(), 'manual_first_eight_exact_prefix': prefix,
              'manual_expected_moves': truth, 'actual_moves': actual, 'evidence_checks': sample_rows,
              'trained_lookup_sha256': hashlib.sha256((root / 'backend/models/board-color-v1.npz').read_bytes()).hexdigest(),
              'full_video_accuracy_claimed': False,
              'validation_scope': 'Replay of the original 18-400 second failure case; manual audit covers first eight moves only'}
    (output / 'final-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'final-task.json').write_text(task.model_dump_json(by_alias=True, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['actual_moves', 'manual_expected_moves', 'evidence_checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
