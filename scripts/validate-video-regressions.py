import argparse
import hashlib
import json
import os
from pathlib import Path

from app.actual import analyze_actual
from app.storage import TaskStore
from app.vision import VisionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--end', type=float)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / 'output/recognition-regression-20261016' / args.tag
    run.mkdir(parents=True, exist_ok=True)
    originals = TaskStore(root / 'data')
    original = originals.get(args.task)
    store = TaskStore(run / 'data')
    task = store.create(f'兼容性回归 · {original.filename}', original.metadata)
    task.start_s = original.start_s
    task.end_s = args.end if args.end is not None else original.end_s
    task.calibration = original.calibration.model_copy(deep=True)
    task.calibration_confirmed = original.calibration_confirmed
    task.initial_position = original.initial_position.model_copy(deep=True)
    task.initial_turn_known = original.initial_turn_known
    task.initial_frame_s = original.initial_frame_s
    task.replay_mode, task.rules_enabled = original.replay_mode, original.rules_enabled
    task.status = 'analyzing'
    task = store.save(task, task.revision)
    os.link(originals.task_dir(original.id) / 'source.mp4', store.task_dir(task.id) / 'source.mp4')
    from app.board_colors import MODEL_PATH
    model = {'path': str(MODEL_PATH), 'sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()}
    (run / 'state.json').write_text(json.dumps({'task_id': task.id, 'original_task_id': original.id,
        'range_s': [task.start_s, task.end_s], 'color_model': model}, indent=2), encoding='utf8')
    vision = VisionClient()
    assert vision.local_classifier is not None and vision.local_classifier.supports_reference
    analyze_actual(task, store, vision)
    result = store.get(task.id)
    (run / 'task.json').write_text(result.model_dump_json(indent=2), encoding='utf8')
    moves = [{'piece_id': r.move.piece_id, 'from': [r.move.source.file, r.move.source.rank],
              'to': [r.move.target.file, r.move.target.rank], 'captured': r.move.captured_piece_id,
              'time_s': r.action_end_s, 'violation': r.violation} for r in result.moves if r.move is not None]
    summary = {'status': result.status, 'error': result.error, 'task_id': task.id,
        'source_task_id': original.id, 'range_s': [task.start_s, task.end_s],
        'records': len(result.moves), 'confirmed_prefix': result.confirmed_prefix_length,
        'confirmed_moves': sum(r.status == 'auto_verified' and r.move is not None for r in result.moves),
        'unresolved': sum(r.status == 'needs_review' for r in result.moves),
        'last_move_s': max((m['time_s'] or 0 for m in moves), default=0),
        'stats': result.run_stats.model_dump(), 'color_model': model, 'moves': moves}
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'moves'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
