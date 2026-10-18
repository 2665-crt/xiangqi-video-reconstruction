import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from app.storage import TaskStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['before', 'after'], required=True)
    parser.add_argument('--task', default='f49687931cc84b20add9ecad7480397c')
    parser.add_argument('--end', type=float, default=100)
    parser.add_argument('--tag', default='opening')
    parser.add_argument('--fusion', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / 'backfill-policy-20261016'
    run = output / f'{args.tag}-{args.mode}'
    run.mkdir(parents=True, exist_ok=True)
    original_store = TaskStore(root / 'data')
    original = original_store.get(args.task)
    store = TaskStore(run / 'data')
    task = store.create(f'回查策略对照 {args.mode} · {original.filename}', original.metadata)
    task.start_s, task.end_s = original.start_s, args.end
    task.calibration = original.calibration.model_copy(deep=True)
    task.calibration_confirmed = original.calibration_confirmed
    task.initial_position = original.initial_position.model_copy(deep=True)
    task.initial_turn_known = original.initial_turn_known
    task.initial_frame_s = original.initial_frame_s
    task.replay_mode, task.rules_enabled = original.replay_mode, original.rules_enabled
    task.status = 'analyzing'
    task = store.save(task, task.revision)
    os.link(original_store.task_dir(original.id) / 'source.mp4', store.task_dir(task.id) / 'source.mp4')
    import app.actual as actual
    if args.mode == 'before':
        source = output / 'before' / 'actual.py'
        exec(compile(source.read_text(encoding='utf8'), str(source), 'exec'), actual.__dict__)
    backfill = actual.backfill_interval
    calls = []

    def measured_backfill(source, start, end, calibration, before, expected, on_frame=None):
        call = {'start_s': start, 'end_s': end, 'frames': 0}
        calls.append(call)
        (run / 'backfills.json').write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding='utf8')

        def counted(time_s):
            call['frames'] += 1
            if on_frame is not None:
                on_frame(time_s)

        started = perf_counter()
        result = backfill(source, start, end, calibration, before, expected, counted)
        call.update(seconds=perf_counter() - started, recovered_events=len(result))
        (run / 'backfills.json').write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(call), flush=True)
        return result

    actual.backfill_interval = measured_backfill
    (run / 'state.json').write_text(json.dumps({'task_id': task.id, 'source_task_id': original.id,
        'range_s': [task.start_s, task.end_s]}, indent=2), encoding='utf8')
    vision = None
    if args.fusion:
        from app.vision import VisionClient
        vision = VisionClient()
        assert vision.local_classifier is not None and vision.local_classifier.supports_reference
    actual.analyze_actual(task, store, vision)
    result = store.get(task.id)
    (run / 'task.json').write_text(result.model_dump_json(indent=2), encoding='utf8')
    summary = {'task_id': result.id, 'source_task_id': original.id, 'range_s': [task.start_s, task.end_s],
        'local_fusion_fallback': args.fusion,
        'status': result.status, 'error': result.error, 'records': len(result.moves),
        'confirmed_prefix_length': result.confirmed_prefix_length,
        'backfill_intervals': len(calls), 'backfill_frames': sum(c['frames'] for c in calls),
        'backfill_seconds': sum(c['seconds'] for c in calls), 'run_stats': result.run_stats.model_dump(),
        'moves': [{'index': r.index, 'status': r.status, 'move': r.move.model_dump() if r.move else None,
                   'landing_s': r.action_end_s, 'violation': r.violation} for r in result.moves]}
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'moves'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
