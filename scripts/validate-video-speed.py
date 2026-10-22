import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time
import types

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from app.analysis import board_changed, board_preview
from app.board_motion import BoardMotion
from app.calibration import rectify_board
from app.schemas import Calibration, Task
from app.video import iter_frames


def baseline_module(ref, name):
    source = subprocess.check_output(['git', 'show', f'{ref}:backend/app/{name}.py'], cwd=ROOT)
    module = types.ModuleType(f'app._baseline_{name}')
    module.__package__ = 'app'
    exec(compile(source, f'{ref}/{name}.py', 'exec'), module.__dict__)
    return module


def synthetic_frames():
    gray = np.random.default_rng(17).integers(30, 240, (480, 640), dtype=np.uint8)
    image = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    frames = []
    for n in range(120):
        dx = max(0, n - 55) * .14
        current = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480))
        if 25 <= n < 34:
            current[150:245, 200:290] = (150, 180, 210)
        if 85 <= n < 92:
            current[:] = (40, 40, 40)
        frames.append((n * .3, current))
    calibration = Calibration(corners=[{'x': 90, 'y': 80}, {'x': 550, 'y': 80},
                                      {'x': 550, 'y': 400}, {'x': 90, 'y': 400}])
    return frames, calibration


def measure(frames, motion_class, calibration):
    cv2.setRNGSeed(0)
    tracker = motion_class(calibration)
    digest, state_digest = hashlib.sha256(), hashlib.sha256()
    decode_s = tracking_s = audit_s = verification_s = 0.
    count, neighbor_changes, reference_changes = 0, 0, 0
    first = previous = None
    started = time.perf_counter()
    try:
        while True:
            mark = time.perf_counter()
            try:
                stamp, image = next(frames)
            except StopIteration:
                decode_s += time.perf_counter() - mark
                break
            decode_s += time.perf_counter() - mark
            mark = time.perf_counter()
            calibrated = tracker.update(stamp, image)
            tracking_s += time.perf_counter() - mark
            mark = time.perf_counter()
            preview = board_preview(rectify_board(image, calibrated, margin=50))
            if first is None:
                first = preview
            neighbor_changes += int(previous is not None and board_changed(preview, previous))
            reference_changes += int(board_changed(preview, first))
            previous = preview
            audit_s += time.perf_counter() - mark
            mark = time.perf_counter()
            digest.update(repr((stamp, image.shape, image.dtype.str)).encode())
            digest.update(image.tobytes())
            state = (calibrated.model_dump(), tracker.reference_calibration.model_dump(),
                     tracker.reference_s, tracker.last_check_s, tracker.last_feature_check_s, tracker.active)
            state_digest.update(json.dumps(state, sort_keys=True).encode())
            verification_s += time.perf_counter() - mark
            count += 1
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    return {'frames': count, 'pixels_and_timestamps_sha256': digest.hexdigest(),
            'motion_state_sha256': state_digest.hexdigest(), 'decode_s': decode_s,
            'tracking_s': tracking_s, 'activity_audit_s': audit_s, 'verification_s': verification_s,
            'elapsed_s': time.perf_counter() - started,
            'activity_audit': {'neighbor_changed_frames': neighbor_changes,
                               'changed_from_start_frames': reference_changes,
                               'filters_frames': False}}


def main():
    parser = argparse.ArgumentParser(description='逐帧对照解码与棋盘跟踪；不执行完整视频识别，不写入任务数据库。')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--task-id')
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--start', type=float)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/video-speed-validation')
    args = parser.parse_args()
    if args.duration <= 0 or args.threads < 1 or args.repeats < 1:
        parser.error('duration、threads 和 repeats 必须为正数')
    old_motion = baseline_module(args.baseline_ref, 'board_motion').BoardMotion
    old_frames = baseline_module(args.baseline_ref, 'video').iter_frames
    if args.synthetic:
        frames, calibration = synthetic_frames()
        make_frames = lambda optimized: iter(frames)
        source_info = {'type': 'synthetic', 'video_analysis': False}
    else:
        with closing(sqlite3.connect((ROOT / 'data/tasks.sqlite3').as_uri() + '?mode=ro', uri=True)) as connection:
            row = connection.execute('SELECT payload FROM tasks WHERE id=?', (args.task_id,)).fetchone()
        if row is None:
            parser.error('实验数据库中不存在该任务')
        task = Task.model_validate_json(row[0])
        if task.calibration is None:
            parser.error('请先完成该任务的棋盘定位')
        path = ROOT / 'data/tasks' / task.id / 'source.mp4'
        start = task.start_s if args.start is None else args.start
        end = min(task.metadata.duration_s, start + args.duration)
        if start < 0 or start >= end:
            parser.error('视频选区无效')
        calibration = task.calibration
        make_frames = lambda optimized: (iter_frames if optimized else old_frames)(path, start, end)
        source_info = {'type': 'task', 'task_id': task.id, 'start': start, 'end': end, 'video_analysis': False}
    rows = []
    for repeat in range(args.repeats):
        pair = {}
        for optimized in ((False, True) if repeat % 2 == 0 else (True, False)):
            os.environ['VIDEO_REFERENCE_CACHE'] = '1' if optimized else '0'
            os.environ['VIDEO_DECODE_THREADS'] = str(args.threads)
            mode = 'optimized' if optimized else 'baseline'
            pair[mode] = measure(make_frames(optimized), BoardMotion if optimized else old_motion,
                                 calibration.model_copy(deep=True))
            print(mode, pair[mode]['frames'], round(pair[mode]['elapsed_s'], 3), flush=True)
        pair['equivalent'] = all(pair['baseline'][key] == pair['optimized'][key] for key in
                                 ('frames', 'pixels_and_timestamps_sha256', 'motion_state_sha256', 'activity_audit'))
        rows.append(pair)
    import statistics
    before = statistics.median(r['baseline']['decode_s'] + r['baseline']['tracking_s'] for r in rows)
    after = statistics.median(r['optimized']['decode_s'] + r['optimized']['tracking_s'] for r in rows)
    report = {'baseline_ref': args.baseline_ref, 'source': source_info, 'runs': rows,
              'equivalent': all(r['equivalent'] for r in rows),
              'median_decode_and_tracking_baseline_s': before,
              'median_decode_and_tracking_optimized_s': after,
              'speedup_decode_and_tracking': before / after if after else None,
              'scope': '仅解码及棋盘防抖，不代表完整解析提速；活动统计不删帧、不判断走棋准确率'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, ensure_ascii=False))
    return 0 if report['equivalent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
