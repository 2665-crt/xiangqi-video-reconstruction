"""在独立任务库重跑指定真实视频，不修改原任务。"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

from app.actual import analyze_actual
from app.storage import TaskStore
from app.vision import VisionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--data', type=Path, help='源任务库目录，默认使用正式 data')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    originals = TaskStore(args.data or root / 'data')
    original = originals.get(args.task)
    baseline = original.model_dump_json()
    store = TaskStore(args.output / 'data')
    task = store.create(f'推断验收 · {original.filename}', original.metadata)
    task.start_s = original.start_s if args.start is None else args.start
    task.end_s = original.end_s if args.end is None else args.end
    initial = original.initial_position
    if task.start_s > original.start_s:
        previous = [r for r in original.moves if r.status in ('auto_verified', 'corrected')
                    and r.position_after is not None and (r.action_end_s or r.video_end_s) <= task.start_s]
        if previous:
            initial = max(previous, key=lambda r: r.action_end_s or r.video_end_s).position_after
    task.initial_position = initial.model_copy(deep=True)
    task.initial_turn_known = True
    task.initial_frame_s = task.start_s
    task.calibration = original.calibration.model_copy(deep=True)
    task.calibration_confirmed = True
    task.replay_mode, task.rules_enabled, task.status = 'actual', True, 'analyzing'
    task = store.save(task, task.revision)
    os.link(originals.task_dir(original.id) / 'source.mp4', store.task_dir(task.id) / 'source.mp4')
    (args.output / 'state.json').write_text(json.dumps({'task_id': task.id,
        'original_task_id': original.id, 'start_s': task.start_s, 'end_s': task.end_s}), encoding='utf8')
    print(f'START {original.filename} {task.start_s:.3f}..{task.end_s:.3f} task={task.id}', flush=True)
    analyze_actual(task, store, VisionClient())
    result = store.get(task.id)
    assert originals.get(original.id).model_dump_json() == baseline, '原任务发生变化'
    summary = {'source_task_id': original.id, 'task_id': task.id, 'filename': original.filename,
        'start_s': task.start_s, 'end_s': task.end_s, 'processed_s': result.processed_video_s,
        'status': result.status, 'error': result.error, 'statuses': dict(Counter(r.status for r in result.moves)),
        'records': len(result.moves), 'last_displayable_s': max((r.tracking_anchor_s or r.action_end_s or r.video_end_s
            for r in result.moves if r.position_after is not None or r.predicted_position is not None
            or r.tracking_anchor is not None), default=task.start_s),
        'original_unchanged': True, 'stats': result.run_stats.model_dump(),
        'moves': [{'index': r.index, 'status': r.status, 'move': r.move.model_dump(by_alias=True) if r.move else None,
            'time_s': r.action_end_s, 'reason': r.reason} for r in result.moves]}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'moves'}, ensure_ascii=False), flush=True)
    if result.error or result.processed_video_s < task.end_s:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
