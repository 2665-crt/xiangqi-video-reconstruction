from pathlib import Path

from app.analysis import analyze_task
from app.storage import TaskStore


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'output/takeback-35s-20261016'
    store = TaskStore(root / 'data')
    task = store.get('88fd9b989b27430999c2b8400fd83994')
    assert task.status == 'needs_review' and task.moves[-1].index == 46
    (output / 'full-run-before-tail-fix.json').write_text(task.model_dump_json(by_alias=True, indent=2), encoding='utf8')
    previous = task.run_stats.model_copy(deep=True)
    task.moves = task.moves[:-1]
    last = task.moves[-1]
    task.context_position = last.position_after.model_copy(deep=True)
    task.resume_video_s = last.video_end_s
    task.processed_video_s = task.recognition_video_s = last.video_end_s
    task.pending_turn = task.recovery = None
    task.status, task.stage = 'analyzing', '验证选区末手边界'
    task = store.save(task, task.revision)
    print(f'Replay actual pipeline from {task.resume_video_s:.3f}s to {task.end_s:.3f}s', flush=True)
    analyze_task(task.id, store, None)
    task = store.get(task.id)
    (output / 'tail-validation-stats.json').write_text(task.run_stats.model_dump_json(indent=2), encoding='utf8')
    for name in type(previous).model_fields:
        setattr(task.run_stats, name, getattr(previous, name) + getattr(task.run_stats, name))
    task.context_position = task.initial_position.model_copy(deep=True)
    task.resume_video_s = None
    task = store.save_analysis(task)
    print(task.status, len(task.moves), task.moves[-1].move, flush=True)


if __name__ == '__main__':
    main()
