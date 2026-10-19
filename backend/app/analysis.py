from collections import deque
from itertools import chain
from functools import lru_cache
import json
from typing import NamedTuple
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

from .calibration import rectify_board
from .chess import apply_move, board_signature, legal_moves
from .schemas import Move, MoveRecord, Position, Task
from .storage import RecoveryCancelled, TaskStore
from .video import iter_frames, save_image
from .vision import VisionClient, VisionRequestError


def _occupied_squares(position):
    return tuple(sorted((p.color, p.square.file, p.square.rank) for p in position.pieces))


def _matches(position, observations):
    # 棋子身份和走法来自已确认局面，模型只提供本帧颜色与位置，避免重复辨字破坏身份。
    return _occupied_squares(position) in {_occupied_squares(p) for p in observations}


def _candidates(position, observations):
    return [(m, apply_move(position, m)) for m in legal_moves(position)
            if _matches(apply_move(position, m), observations)]


def resolve_initial(candidates: list[Position], following: list[Position]) -> Position | None:
    eligible = [p for p in candidates if _candidates(p, following)]
    unique = {(board_signature(p), p.side_to_move): p for p in eligible}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _two_step_paths(position, observations):
    paths = []
    for first in legal_moves(position):
        middle = apply_move(position, first)
        for second, final in _candidates(middle, observations):
            paths.append([(first, middle), (second, final)])
            if len(paths) > 1:
                return paths
    return paths


def extract_moves(initial: Position, observations: list[tuple[float, list[Position]]], unresolved_times=None) -> list[MoveRecord]:
    records, current, previous_time, blocked = [], initial, 0.0, False
    for obs_index, (time_s, positions) in enumerate(observations):
        if blocked:
            records.append(MoveRecord(index=len(records) + 1, video_start_s=previous_time, video_end_s=time_s,
                                      status="tentative", observed_positions=positions, reason="前方疑点未解决"))
            previous_time = time_s
            continue
        if positions and all(_matches(current, [p]) for p in positions):
            previous_time = time_s
            continue
        matches = _candidates(current, positions)
        no_move_possible = _matches(current, positions)
        if no_move_possible and len(matches) == 1 and obs_index + 1 < len(observations):
            following_time, following = observations[obs_index + 1]
            following_matches = _candidates(current, following)
            if not _matches(current, following) and len(following_matches) == 1 and _matches(matches[0][1], following):
                positions, matches, time_s = following, following_matches, following_time
                no_move_possible = False
        if not positions and time_s not in (unresolved_times or set()) and obs_index + 1 < len(observations):
            following = observations[obs_index + 1][1]
            if following and not _matches(current, following):
                positions = following
                matches = _candidates(current, positions)
                time_s = observations[obs_index + 1][0]
        if len(matches) > 1 and obs_index + 1 < len(observations):
            following = observations[obs_index + 1][1]
            viable = [(m, p) for m, p in matches if _matches(p, following) or _candidates(p, following)]
            if len(viable) == 1:
                matches = viable
        paths = [matches] if len(matches) == 1 else (_two_step_paths(current, positions) if not matches and positions else [])
        if len(paths) == 1 and not no_move_possible:
            group = f"{time_s}:{len(records)}" if len(paths[0]) == 2 else None
            for action, after in paths[0]:
                action = action.model_copy(update={"index": len(records) + 1})
                records.append(MoveRecord(index=action.index, move=action, video_start_s=previous_time, video_end_s=time_s,
                                          status="auto_verified", position_after=after,
                                          observed_positions=positions if after == paths[0][-1][1] else [],
                                          inference_group=group,
                                          reason="多帧与规则支持唯一还原" if len(paths[0]) == 2 else ""))
                current = after
        else:
            records.append(MoveRecord(index=len(records) + 1, video_start_s=previous_time, video_end_s=time_s,
                                      status="needs_review", candidates=[m.model_copy(update={"index": len(records) + 1}) for m, _ in matches],
                                      observed_positions=positions, reason="画面存在多种解释或无法唯一还原，请核对原视频"))
            blocked = True
        previous_time = time_s
    return records


class VideoEvent(NamedTuple):
    time_s: float
    images: list
    uncertain: bool = False
    position: Position | None = None
    review_reason: str = ""
    moving_colors: tuple[str, ...] = ()
    last_clear_s: float | None = None
    departure_only: bool = False
    action_start_s: float | None = None
    image_times_s: tuple[float | None, ...] = ()
    backfilled: bool = False
    board_check: object | None = None
    moving_piece_ids: tuple[str, ...] = ()
    tracking_anchor: bool = False


@lru_cache(maxsize=1)
def _cell_sampling():
    dy, dx = np.mgrid[-18:19, -18:19]
    disk = dx * dx + dy * dy < 18 * 18
    files, ranks = np.meshgrid(np.arange(9), np.arange(10))
    files, ranks = files.ravel(), ranks.ravel()
    ys, xs = 25 + ranks[:, None] * 50 + dy[disk], 25 + files[:, None] * 50 + dx[disk]
    inward = (((files[:, None] != 0) | (dx[disk] >= 0))
              & ((files[:, None] != 8) | (dx[disk] <= 0))
              & ((ranks[:, None] != 0) | (dy[disk] >= 0))
              & ((ranks[:, None] != 9) | (dy[disk] <= 0)))
    edge = (files == 0) | (files == 8) | (ranks == 0) | (ranks == 9)
    return files, ranks, ys, xs, inward, inward.sum(axis=1), edge


def read_board_cells(image, strict=False):
    from .board_colors import wood_surface_mask
    small = cv2.resize(image, (450, 500))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    files, ranks, ys, xs, inward, inward_count, edge = _cell_sampling()
    pixels, colors = small[ys, xs].astype(np.float32), hsv[ys, xs]
    red = ((colors[:, :, 0] > 150) | (colors[:, :, 0] < 10)) & (colors[:, :, 1] > 45) & (colors[:, :, 2] > 70)
    surface_red = red
    neutral = (pixels[:, :, 0] > pixels[:, :, 2] * .93) & (pixels.max(axis=2) > 100)
    coverage, red_ratio = (neutral | surface_red).mean(axis=1), red.mean(axis=1)
    skin_pixels = (colors[:, :, 0] > 3) & (colors[:, :, 0] < 20) & (colors[:, :, 1] > 65) & (colors[:, :, 2] > 95)
    wood_surface = wood_surface_mask(hsv)
    sampled_wood = wood_surface[ys, xs] if wood_surface is not None else None
    if sampled_wood is not None:
        skin_pixels &= ~sampled_wood
    skin = skin_pixels.mean(axis=1)
    if strict:
        ink = (colors[:, :, 2] < 185).mean(axis=1)
        grid_pixels = colors[:, :, 2] < np.percentile(colors[:, :, 2], 75, axis=1)[:, None] * .65
        grid_ink = grid_pixels.mean(axis=1)
        wood_pixels = ((pixels[:, :, 2] > pixels[:, :, 1] * .95) & (pixels[:, :, 1] > pixels[:, :, 0] * 1.1)
                       & (colors[:, :, 2] > 80) & (colors[:, :, 1] > 30))
        wood = wood_pixels.mean(axis=1)
        inward_wood = (wood_pixels & inward).sum(axis=1) / inward_count
        inward_grid = (grid_pixels & inward).sum(axis=1) / inward_count
        inward_coverage = ((neutral | surface_red) & inward).sum(axis=1) / inward_count
        finger_pixels = ((colors[:, :, 0] < 23) & (colors[:, :, 1] > 55) & (colors[:, :, 2] > 95)
                         & (pixels[:, :, 2] > pixels[:, :, 1] * 1.2) & (pixels[:, :, 0] > pixels[:, :, 1] * .62))
        if sampled_wood is not None:
            finger_pixels &= ~sampled_wood
        inward_fingers = (finger_pixels & inward).sum(axis=1) / inward_count
        # 边缘圆盘会混入棋盘外背景；空格仍须在盘内区域同时看到木纹和网格。
        edge_empty = edge & (inward_coverage <= .15) & (inward_wood >= .65) & (inward_grid >= .04) & (inward_fingers <= .18)
        if sampled_wood is not None:
            core = (ys[0] - 25) ** 2 + (xs[0] - 25) ** 2 < 8 ** 2
            core_coverage = (neutral | surface_red)[:, core].mean(axis=1)
            learned_empty = (edge & (coverage <= .3) & (sampled_wood.mean(axis=1) >= .65)
                             & (sampled_wood[:, core].mean(axis=1) >= .8)
                             & (core_coverage <= .05) & (inward_grid >= .02) & (inward_fingers <= .18))
            inward_coverage[learned_empty] = core_coverage[learned_empty]
            edge_empty |= learned_empty
        wood[edge_empty], grid_ink[edge_empty] = inward_wood[edge_empty], inward_grid[edge_empty]
        coverage[edge_empty] = inward_coverage[edge_empty]
        return {(int(files[i]), int(ranks[i])): ("red" if red_ratio[i] > .045 else "black") if coverage[i] >= .65 else None
                for i in range(90) if not (edge[i] and coverage[i] < .65 and inward_fingers[i] > .18 or
                    .15 < coverage[i] < .65 or
                    (skin[i] > (.18 if coverage[i] >= .65 else .35)
                     and not (coverage[i] <= .15 and wood[i] >= .65 and grid_ink[i] >= .04)) or
                    (coverage[i] >= .65 and red_ratio[i] <= .045 and ink[i] < .045) or
                    (coverage[i] <= .15 and wood[i] < .65) or
                    (coverage[i] >= .65 and .01 < red_ratio[i] <= .045))}
    # 未知格不用于证明走法；移动起点和落点必须在连续画面中实际可见。
    return {(int(files[i]), int(ranks[i])): ("red" if red_ratio[i] > .045 else "black") if coverage[i] >= .65 else None
            for i in range(90) if not (.15 < coverage[i] < .65 or skin[i] > .55 or (skin[i] > .25 and coverage[i] < .65))}


def tracked_events(frames, initial, confirmed_position=None):
    first = next(frames, None)
    if first is None:
        return
    cells = read_board_cells(first[1])
    occupied = {(p.square.file, p.square.rank): p.color for p in initial.pieces}
    # 只在初始画面完整验证了本地检测器时启用；其他棋子外观沿用视觉模型路径。
    if len(cells) != 90 or any(occupied.get(square) != color for square, color in cells.items()):
        yield from stable_events(chain([first], frames))
        return
    yield VideoEvent(first[0], [first[1]], position=initial)
    current, pending, pending_start = initial, None, 0
    confirmed_image, pending_image, unresolved_image = first[1], None, None
    review_event, review_signature, returned_start = None, None, None
    last_time = first[0]
    reachable = []
    fallback, fallback_start, emitted_fallback = None, 0, None
    for time_s, image in frames:
        last_time = time_s
        if confirmed_position is not None:
            confirmed = confirmed_position()
            if confirmed != current:
                current, reachable, pending, fallback = confirmed, [], None, None
                confirmed_image, pending_image, unresolved_image = image, None, None
                review_event, review_signature, returned_start = None, None, None
        cells = read_board_cells(image)
        occupied = {(p.square.file, p.square.rank): p.color for p in current.pieces}
        if all(occupied.get(square) == color for square, color in cells.items()):
            if pending_image is not None:
                review_event = VideoEvent(time_s, [confirmed_image, pending_image, image], True)
                review_signature = pending
            if review_event is not None and len(cells) == 90:
                if returned_start is None:
                    returned_start = time_s
                elif time_s - returned_start >= .18:
                    yield review_event
                    review_event, returned_start = None, None
            else:
                returned_start = None
            unresolved_image = image if len(cells) < 90 else None
            pending_image = None
            pending, fallback = None, None
            continue
        if not reachable:
            reachable = [(m, apply_move(current, m)) for m in legal_moves(current)]
        matches = []
        for action, after in reachable:
            after_cells = {(p.square.file, p.square.rank): p.color for p in after.pieces}
            if len(cells) >= 70 and (action.source.file, action.source.rank) in cells and (action.target.file, action.target.rank) in cells and all(
                    after_cells.get(square) == color for square, color in cells.items()):
                matches.append((action, after))
        if len(matches) == 1:
            returned_start = None
            after = matches[0][1]
            signature = _occupied_squares(after)
            if pending != signature:
                pending, pending_start = signature, time_s
                pending_image = image
            elif time_s - pending_start >= .06:
                current, reachable, pending, fallback = after, [], None, None
                confirmed_image, pending_image, unresolved_image = image, None, None
                if review_event is not None and review_signature != signature:
                    yield review_event
                review_event = None
                yield VideoEvent(time_s, [image], position=after)
        else:
            if pending_image is not None:
                review_event = VideoEvent(time_s, [confirmed_image, pending_image, image], True)
                review_signature = pending
                pending_image = None
            returned_start = None
            pending = None
            unresolved_image = image
            if len(cells) == 90:
                signature = tuple(sorted(cells.items()))
                if fallback != signature:
                    fallback, fallback_start = signature, time_s
                elif time_s - fallback_start >= 1 and emitted_fallback != signature:
                    emitted_fallback = signature
                    unresolved_image = None
                    images = [review_event.images[0], review_event.images[1], image] if review_event is not None else [image]
                    yield VideoEvent(time_s, images, review_event is not None)
                    review_event, review_signature = None, None
            else:
                fallback = None
    reason = "区间结束时棋盘未完整可见或落子尚未完成，请延长区间或核对截图" if len(cells) < 90 else ""
    if review_event is not None:
        yield review_event._replace(review_reason=reason)
    elif pending_image is not None or unresolved_image is not None:
        yield VideoEvent(last_time, [confirmed_image, pending_image if pending_image is not None else unresolved_image], True,
                         review_reason=reason)


def board_preview(image):
    return cv2.GaussianBlur(cv2.resize(image, (180, 200), interpolation=cv2.INTER_AREA), (3, 3), 0)


def board_changed(a, b):
    difference = np.max(cv2.absdiff(a, b), axis=2) > 25
    if np.mean(difference) > 0.003:
        return True
    local_ratios = difference.reshape(10, 20, 9, 20).mean(axis=(1, 3))
    return bool(np.any(local_ratios > 0.025))


def stable_events(frames, stable_seconds=0.18, state=None, finalize=True):
    state = {} if state is None else state
    pending = state.get('pending')
    pending_start = state.get('pending_start', 0)
    emitted = state.get('emitted')
    images = deque(state.get('images', []), maxlen=3)
    last_time = state.get('last_time', 0)
    pending_emitted = state.get('pending_emitted', False)
    motion_samples = list(state.get('motion_samples', []))
    motion_time = state.get('motion_time', 0)

    def remember_motion():
        nonlocal motion_samples, motion_time
        baseline = emitted if emitted is not None else pending
        for image in images:
            small = board_preview(image)
            motion_samples.append((image, float(np.mean(cv2.absdiff(small, baseline)))))
        if len(motion_samples) > 3:
            motion_samples = [motion_samples[0], max(motion_samples[1:-1], key=lambda item: item[1]), motion_samples[-1]]
        motion_time = last_time

    for time_s, frame in frames:
        small = board_preview(frame)
        if pending is None or board_changed(small, pending):
            if pending is not None and not pending_emitted:
                remember_motion()
            pending, pending_start = small, time_s
            images.clear()
            pending_emitted = False
        images.append(frame)
        last_time = time_s
        if time_s - pending_start >= stable_seconds and motion_samples:
            yield VideoEvent(motion_time, [image for image, _ in motion_samples], True)
            motion_samples = []
        if time_s - pending_start >= stable_seconds and (emitted is None or board_changed(pending, emitted)):
            best = max(images, key=lambda image: cv2.Laplacian(cv2.cvtColor(cv2.resize(image, (360, 400)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            yield VideoEvent(time_s, [best])
            emitted = pending.copy()
            pending_emitted = True
        elif emitted is not None and not board_changed(pending, emitted) and time_s - pending_start >= stable_seconds:
            pending_emitted = True
        state.update(pending=pending, pending_start=pending_start, emitted=emitted, images=list(images),
                     last_time=last_time, pending_emitted=pending_emitted,
                     motion_samples=list(motion_samples), motion_time=motion_time)
    if finalize and pending is not None and not pending_emitted:
        remember_motion()
    if finalize and motion_samples:
        yield VideoEvent(motion_time, [image for image, _ in motion_samples], True)


def revalidate_suffix(task: Task, start_index: int) -> Task:
    corrected = next((r.index for r in task.moves if r.manual_override), None)
    if corrected is not None:
        from .turns import replay_corrections
        return replay_corrections(task, corrected)
    if task.replay_mode == 'actual':
        from .actual import revalidate_actual
        return revalidate_actual(task)
    task = task.model_copy(deep=True)
    if task.initial_position is None:
        task.confirmed_prefix_length = 0
        return task
    current, blocked, prefix = task.initial_position, False, 0
    rebuilt = []
    records = iter(task.moves)
    for record in records:
        record.index = len(rebuilt) + 1
        if blocked:
            record.status, record.position_after = "tentative", None
            rebuilt.append(record)
            continue
        if record.index < start_index and record.status in ("auto_verified", "corrected") and record.move and record.position_after:
            after = apply_move(current, record.move)
            if after == record.position_after:
                rebuilt.append(record)
                current, prefix = after, record.index
                continue
        if record.inference_group and record.status != "corrected":
            partner = next(records, None)
            if partner is not None and partner.inference_group == record.inference_group:
                paths = _two_step_paths(current, partner.observed_positions)
                if len(paths) == 1 and not _matches(current, partner.observed_positions):
                    for item, (action, after) in zip((record, partner), paths[0]):
                        item.index = len(rebuilt) + 1
                        item.move = action.model_copy(update={"index": item.index})
                        item.position_after, item.status = after, "auto_verified"
                        rebuilt.append(item)
                        current, prefix = after, item.index
                    continue
                record.status, record.position_after = "needs_review", None
                record.reason = "连续两手的共同证据不再支持唯一还原"
                rebuilt.append(record)
                partner.index = len(rebuilt) + 1
                partner.status, partner.position_after = "tentative", None
                rebuilt.append(partner)
                blocked = True
                continue
            if partner is not None:
                raise ValueError("连续走棋证据分组不完整，请重新分析该区间")
        if record.status != "corrected" and record.observed_positions and all(
                _matches(current, [p]) for p in record.observed_positions):
            continue
        rebuilt.append(record)
        if record.status != "corrected" and _matches(current, record.observed_positions):
            record.status, record.position_after = "needs_review", None
            record.reason = "画面仍同时支持未走棋和已走棋，尚未唯一还原"
            blocked = True
            continue
        action = record.move
        if action is None:
            matches = _candidates(current, record.observed_positions)
            if len(matches) == 1:
                action = matches[0][0].model_copy(update={"index": record.index})
        try:
            if action is None:
                raise ValueError("走法尚未确定")
            after = apply_move(current, action)
        except ValueError:
            record.status, record.position_after = "needs_review", None
            record.reason = "起点或走法与前面棋局不一致，请重新确认"
            blocked = True
            continue
        if record.status != "corrected" and not _matches(after, record.observed_positions):
            record.status, record.position_after = "tentative", None
            record.reason = "后续局面缺少相符证据，请确认"
            blocked = True
            continue
        record.move, record.position_after = action, after
        record.move = action.model_copy(update={"index": record.index})
        if record.status != "corrected":
            record.status = "auto_verified"
        current, prefix = after, record.index
    task.moves = rebuilt
    task.confirmed_prefix_length = prefix
    task.status = "ready" if prefix == len(task.moves) else "needs_review"
    task.stage = "棋谱已保存" if task.status == "ready" else "请确认疑点"
    return task


def correct_move(task: Task, index: int, move: Move) -> Task:
    return correct_interval(task, index, [move])


def correct_interval(task: Task, index: int, actions: list[Move], video_time_s=None) -> Task:
    if task.replay_mode == 'actual':
        from .actual import correct_actual
        return correct_actual(task, index, actions, video_time_s)
    if not task.initial_position or not 1 <= index <= len(task.moves) + 1:
        raise ValueError("请先设置初始棋局，且按顺序确认步骤")
    task = task.model_copy(deep=True)
    if index > task.confirmed_prefix_length + 1:
        raise ValueError("请先解决前方疑点")
    current = task.initial_position if index == 1 else task.moves[index - 2].position_after
    for action in actions:
        current = apply_move(current, action)
    original = task.moves[index - 1] if index <= len(task.moves) else None
    if original and original.inference_group:
        original_group = original.inference_group
        for record in task.moves:
            if record.inference_group == original_group:
                record.inference_group = None
    time_s = video_time_s if video_time_s is not None else (original.video_end_s if original else task.processed_video_s)
    preceding_time = task.moves[index - 2].video_end_s if index > 1 else task.start_s
    if not 0 <= time_s <= task.metadata.duration_s:
        raise ValueError('时间必须位于原视频范围内')
    replacement = [MoveRecord(index=index + offset, move=action.model_copy(update={"index": index + offset}),
                              video_start_s=min(original.video_start_s if original else preceding_time, time_s),
                              video_end_s=time_s, status="corrected",
                              evidence_ids=original.evidence_ids if original else [],
                              observed_positions=original.observed_positions if original and offset == len(actions) - 1 else [])
                   for offset, action in enumerate(actions)]
    task.moves[index - 1:index] = replacement
    result = revalidate_suffix(task, index)
    if actions and result.moves[index - 1].status != "corrected":
        raise ValueError(result.moves[index - 1].reason or "请先解决前方疑点")
    return result


def analyze_task(task_id: str, store: TaskStore, vision: VisionClient | None):
    task = store.get(task_id)
    if task.replay_mode == 'actual':
        from .actual import analyze_actual
        return analyze_actual(task, store, vision)
    started = monotonic()
    previous_usage = list(task.usage)
    try:
        if task.initial_position is None:
            task.status, task.stage = "needs_review", "请先核对并保存初始棋局，再分析视频变化"
            store.save(task, task.revision)
            return
        if not task.calibration:
            task.status, task.stage = "needs_review", "未定位棋盘，请先调整棋盘四角并保存"
            store.save(task, task.revision)
            return
        from .scanning import scan_video
        task = scan_video(task, store)
        frames = iter_frames(store.task_dir(task.id) / "source.mp4", task.start_s, task.end_s)
        last_saved = monotonic()
        def boards():
            nonlocal task, last_saved
            for time_s, frame in frames:
                task.processed_video_s = time_s
                task.analysis_stats.scanned_frames += 1
                if monotonic() - last_saved > 1:
                    task.stage = "扫描棋盘变化"
                    task = store.save(task, task.revision)
                    last_saved = monotonic()
                yield time_s, rectify_board(frame, task.calibration, margin=50)
        observations, evidence, last_time = [], {}, task.start_s
        unresolved_times = set()
        initial_reference = None
        if task.initial_position and task.initial_frame_s is not None and task.initial_evidence_ids:
            path = store.task_dir(task.id) / "evidence" / task.initial_evidence_ids[0]
            if path.is_file():
                reference = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if reference is not None:
                    initial_reference = board_preview(reference)
        for event in tracked_events(boards(), task.initial_position,
                lambda: task.moves[task.confirmed_prefix_length - 1].position_after if task.confirmed_prefix_length else task.initial_position):
            time_s, images = event.time_s, event.images
            if event.uncertain:
                task.analysis_stats.uncertain_events += 1
            else:
                task.analysis_stats.stable_events += 1
            task.stage = "核对短暂变化" if event.uncertain else "分析稳定棋局"
            task = store.save(task, task.revision)
            ids = []
            folder = store.task_dir(task.id) / "evidence"
            for image in images:
                name = uuid4().hex + ".jpg"
                save_image(folder / name, image)
                ids.append(name)
            reuse_initial = not event.uncertain and initial_reference is not None and not board_changed(board_preview(images[0]), initial_reference)
            if not event.uncertain:
                initial_reference = None
            if event.position is not None:
                positions = [event.position]
            elif event.review_reason or vision is None:
                positions = []
                unresolved_times.add(time_s)
            elif reuse_initial:
                positions = [task.initial_position]
            else:
                task.analysis_stats.model_requests += 1
                task.analysis_stats.images_sent += len(images)
                task = store.save(task, task.revision)
                reference = task.moves[task.confirmed_prefix_length - 1].position_after if task.confirmed_prefix_length else task.initial_position
                positions = vision.recognize_position(images, reference=reference)
            evidence[time_s] = ids
            observations.append((time_s, positions))
            task.moves = extract_moves(task.initial_position, observations, unresolved_times)
            for record in task.moves:
                record.evidence_ids = evidence.get(record.video_end_s, [])
                record.video_start_s = max(task.start_s, record.video_start_s)
                if record.video_end_s == time_s and event.review_reason:
                    record.reason = event.review_reason
            task.confirmed_prefix_length = next((r.index - 1 for r in task.moves if r.status not in ("auto_verified", "corrected")), len(task.moves))
            task.usage = previous_usage + (vision.usage if vision is not None else [])
            task.recognition_video_s = max(task.recognition_video_s, time_s)
            task.recognition_events += 1
            task = store.save(task, task.revision)
            last_time = time_s
        task.processed_video_s = task.end_s
        task.recognition_video_s = task.end_s
        task.pending_event_count = 0
        task.usage = previous_usage + (vision.usage if vision is not None else [])
        if task.initial_position is None:
            task.status, task.stage = "needs_review", "初始棋局未确定，请设置棋局和下一方"
        elif not task.moves:
            task.status, task.stage = "needs_review", "未检测到可确认走棋，请核对范围与棋盘定位"
        elif task.confirmed_prefix_length < len(task.moves):
            task.status, task.stage = "needs_review", "分析结束，请确认疑点"
        else:
            task.status, task.stage = "ready", "分析完成，可回放"
        task.usage.append({"elapsed_seconds": round(monotonic() - started, 2)})
        store.save(task, task.revision)
    except RecoveryCancelled:
        return
    except VisionRequestError as exc:
        if task.recovery is not None:
            from .recovery import defer_operation
            task.usage = previous_usage + vision.usage
            store.save_analysis(task)
            defer_operation(getattr(store, 'store', store), task, exc)
            return
        task.status, task.stage, task.error = 'failed', '分析未完成', str(exc)
        store.save(task, task.revision)
    except Exception as exc:
        if getattr(vision, "last_response", None):
            (store.task_dir(task_id) / "vision-last-response.json").write_text(
                json.dumps(vision.last_response, ensure_ascii=False, indent=2), encoding="utf-8")
        task.status, task.stage = "failed", "分析未完成"
        task.error = str(exc) if isinstance(exc, ValueError) else "分析发生错误，请检查视频、模型连接和已保存证据"
        task.usage = previous_usage + (vision.usage if vision is not None else [])
        store.save(task, task.revision)
    finally:
        if vision is not None:
            vision.close()
