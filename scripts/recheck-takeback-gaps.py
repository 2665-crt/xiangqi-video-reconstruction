import hashlib
import json
import ast
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.actual import actual_events, PositionHistory, _tracking_cells, _tracking_cache, reconcile_position, snapshot_record
from app.board_evidence import check_transition
from app.calibration import rectify_board, refine_from_pieces
from app.chess import board_signature
from app.recheck import replace_range
from app.schemas import Task
from app.storage import TaskStore
from app.turns import TurnReplay, revalidate_turns
from app.video import iter_frames, save_image
from app.fusion import local_fusion_classifier


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'output/takeback-35s-20261016'
    store = TaskStore(root / 'data')
    original = store.get('88fd9b989b27430999c2b8400fd83994')
    assert len(original.moves) == 46 and sum(r.status == 'needs_review' for r in original.moves) == 2
    (output / 'before-gap-resolution-task.json').write_text(original.model_dump_json(by_alias=True, indent=2), encoding='utf8')
    folder = store.task_dir(original.id) / 'evidence'
    anchors = {ast.literal_eval(key): value for key, value in json.loads(
        (root / 'backend/tests/fixtures/board-colors/piece-anchors.json').read_text()).items()}
    _, first_frame = next(iter_frames(store.task_dir(original.id) / 'source.mp4', original.start_s, original.start_s + .1))
    calibration = refine_from_pieces(first_frame, original.calibration, original.initial_position, anchors)
    results, reports = [], []
    started, total_frames = perf_counter(), 0
    cv2.setNumThreads(1)
    token = _tracking_cache.set(OrderedDict())
    try:
        for left, right in [(17, 20), (30, 33)]:
            boundary, following = original.moves[left - 1], original.moves[right - 1]
            begin, end = original.moves[left - 2].video_end_s, following.action_start_s - .15
            occupied = {(p.square.file, p.square.rank): p.color for p in boundary.position_before.pieces}
            for clear_time, clear_frame in iter_frames(store.task_dir(original.id) / 'source.mp4', begin, boundary.action_start_s):
                cells = _tracking_cells(rectify_board(clear_frame, calibration, margin=100), boundary.position_before)
                if len(cells) >= 70 and all(occupied.get(square) == color for square, color in cells.items()):
                    begin = clear_time
                    break
            else:
                raise RuntimeError('回归区间没有可直接核对的起始画面')
            expected = original.moves[right - 2]
            settled = {(p.square.file, p.square.rank): p.color for p in expected.position_after.pieces}
            for clear_time, clear_frame in iter_frames(store.task_dir(original.id) / 'source.mp4', expected.action_end_s + .2, end):
                cells = _tracking_cells(rectify_board(clear_frame, calibration, margin=100), expected.position_after)
                if len(cells) == 90 and all(settled.get(square) == color for square, color in cells.items()):
                    end = clear_time
                    break
            else:
                raise RuntimeError('回归区间没有完整清晰的结束画面')
            seed = Task(id='0' * 32, filename='实际帧回归', start_s=begin, end_s=end,
                        metadata=original.metadata, initial_position=boundary.position_before,
                        initial_turn_known=True, rules_enabled=True, replay_mode='actual')
            rules, current = TurnReplay(seed, seed.initial_position), seed.initial_position
            history, reference, timestamps = PositionHistory([current]), None, {}
            count = 0
            shown = set()
            previous_readings = {}

            def recover(image, position):
                nonlocal previous_readings
                known = {(p.square.file, p.square.rank): f'{p.color}_{p.kind}' for p in position.pieces}
                observed, recovered = {}, {}
                for square, reading in local_fusion_classifier().detector.readings(image).items():
                    probabilities = reading['class_probabilities']
                    label = max(probabilities, key=probabilities.get)
                    if reading['confidence'] < .85 or probabilities[label] < .8:
                        continue
                    x1, y1, x2, y2 = reading['bbox']
                    center = (x1 + x2) / 2, (y1 + y2) / 2
                    observed[square] = label, center
                    previous = previous_readings.get(square)
                    if (known.get(square) == label or previous is not None and previous[0] == label
                            and max(abs(a - b) for a, b in zip(center, previous[1])) <= 20):
                        recovered[square] = label.split('_', 1)[0]
                previous_readings = observed
                return recovered

            def frames():
                nonlocal count
                for time_s, frame in iter_frames(store.task_dir(original.id) / 'source.mp4', begin, end):
                    image = rectify_board(frame, calibration, margin=100)
                    image.setflags(write=False)
                    timestamps[id(image)] = time_s
                    count += 1
                    yield time_s, image

            for event in actual_events(frames(), current, confirmed_position=lambda: current,
                                       confirmed_image=lambda: reference, turn_boundaries=True,
                                       turn_origin=lambda: rules.before if rules.pending is not None and len(rules.touched) == 1 else None,
                                       recovery_reader=recover):
                if event.moving_colors:
                    rules.movement(event.moving_colors, event.last_clear_s or event.time_s, event.departure_only)
                    continue
                ids = []
                times = list(event.image_times_s) or [timestamps.get(id(image)) for image in event.images]
                for image in event.images:
                    name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
                    if not (folder / name).exists():
                        save_image(folder / name, image)
                    ids.append(name)
                audit = event.board_check
                reason = event.review_reason
                if event.position is not None and not reason:
                    audit = audit or check_transition(current, event.position,
                        _tracking_cells(event.images[0], current), _tracking_cells(event.images[-1], event.position))
                    if audit.status == 'conflict':
                        reason = audit.reason
                    move = snapshot_record(current, event.position, 1, 0, event.time_s).move
                    key = (move.piece_id if move else None, audit.reason)
                    if key not in shown:
                        shown.add(key)
                        print('event', round(event.time_s, 3), move, audit.status, audit.reason,
                              audit.before_mismatches, audit.after_mismatches, flush=True)
                if event.position is None or reason:
                    rules.uncertain(event.time_s, [event.position] if event.position is not None else [], ids, reason)
                else:
                    after = reconcile_position(current, event.position, history)
                    rules.observe(after, event.time_s, ids, event.action_start_s, board_check=audit)
                    current = rules.observed
                    history.append(current)
                    reference = event.images[-1]
                rules.attach_evidence(audit, ids, times)
            rules.end(end)
            print(left, right, count, [(r.status, r.move and r.move.piece_id, r.reason) for r in seed.moves], flush=True)
            (output / f'gap-{left}-{right}-actual-records.json').write_text(seed.model_dump_json(by_alias=True, indent=2), encoding='utf8')
            assert len(seed.moves) == 2 and all(r.status == 'auto_verified' for r in seed.moves)
            assert seed.moves[0].move.piece_id == boundary.move.piece_id
            assert seed.moves[-1].move.piece_id == expected.move.piece_id
            assert board_signature(seed.moves[-1].position_after) == board_signature(expected.position_after)
            results.append((left - 1, right, seed.moves))
            reports.append({'start_s': begin, 'end_s': end, 'frames': count,
                            'piece_ids': [r.move.piece_id for r in seed.moves], 'final_position_matches_full_run': True})
            total_frames += count
        result = original
        for left, right, records in reversed(results):
            result = replace_range(result, left, right, records, preserve_unknown=True)
        result = revalidate_turns(result)
        assert result.status == 'ready' and len(result.moves) == 44
        assert all(r.status == 'auto_verified' for r in result.moves)
        result.run_stats.elapsed_seconds += perf_counter() - started
        result.run_stats.scanned_frames += total_frames
        result.analysis_stats.scanned_frames += total_frames
        result.stage = '完整区间及两处遮挡疑点已通过实际帧复核'
        store.save(result, original.revision)
        (output / 'gap-recheck-validation.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf8')
        print('ready', len(result.moves), flush=True)
    finally:
        _tracking_cache.reset(token)


if __name__ == '__main__':
    main()
