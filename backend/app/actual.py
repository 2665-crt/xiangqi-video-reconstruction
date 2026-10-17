import json
import hashlib
from time import perf_counter as monotonic
from uuid import uuid4

import cv2
import numpy as np

from .calibration import rectify_board
from .chess import board_signature
from .schemas import AnalysisStats, Move, MoveRecord, Position, Square
from .video import iter_frames, save_image
from .storage import RecoveryCancelled
from .vision import VisionRequestError


def _tracking_image(image):
    from .initial_observation import board_geometry
    cell, margin = board_geometry(image)
    inset = max(0, round(margin - cell / 2)) if cell > 0 else 0
    return image[inset:-inset, inset:-inset] if inset else image


def _tracking_cells(image, reference=None):
    from .analysis import read_board_cells
    from .initial_observation import observe_initial_board
    cells = read_board_cells(_tracking_image(image), strict=True)
    occupied = {(p.square.file, p.square.rank): p.color for p in reference.pieces} if reference is not None else None
    changed = occupied is not None and any(occupied.get(s) != c for s, c in cells.items())
    if (len(cells) < 90 or changed) and image.shape[0] > image.shape[1]:
        cells, faces = observe_initial_board(image)
        for square, (x, y, radius) in faces.items():
            face = cv2.getRectSubPix(image, (round(radius * 1.4), round(radius * 1.4)), (x, y))
            gray = cv2.cvtColor(cv2.resize(face, (48, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            yy, xx = np.mgrid[:48, :48]
            center = (xx - 24) ** 2 + (yy - 24) ** 2 < 13 ** 2
            ink = cv2.GaussianBlur(gray, (0, 0), 4) - gray
            if np.percentile(gray[center], 95) - np.percentile(gray[center], 5) < 10 or (ink[center] > 10).mean() < .02:
                cells.pop(square, None)
    return cells


def verified_initial_candidates(candidates, images):
    from .initial_observation import observe_initial_board, merge_initial_observations
    observations = [observe_initial_board(image)[0] for image in images]
    merged, ambiguous = merge_initial_observations(observations)
    verified = []
    for candidate in candidates:
        occupied = {(p.square.file, p.square.rank): p.color for p in candidate.pieces}
        if (ambiguous or len(merged) != 90
                or any(occupied.get(square) != color for square, color in merged.items())
                or any(square not in observations[-1] for square in occupied)):
            continue
        verified.append(candidate)
    return verified


def verified_observed_candidates(current, candidates, before_image, after_image):
    cells = _tracking_cells(after_image, current)
    before = cv2.resize(_tracking_image(before_image), (450, 500))
    after = cv2.resize(_tracking_image(after_image), (450, 500))
    verified = []
    for candidate in candidates:
        occupied = {(p.square.file, p.square.rank): p for p in candidate.pieces}
        if len(cells) < 70 or any((occupied[s].color if s in occupied else None) != color for s, color in cells.items()):
            continue
        consistent = True
        for piece in current.pieces:
            square = piece.square.file, piece.square.rank
            x, y = 25 + square[0] * 50, 25 + square[1] * 50
            a, b = before[y-18:y+19, x-18:x+19], after[y-18:y+19, x-18:x+19]
            if square not in cells or np.mean(np.max(cv2.absdiff(a, b), axis=2) > 30) < .08:
                observed = occupied.get(square)
                if observed is None or (observed.color, observed.kind) != (piece.color, piece.kind):
                    consistent = False
                    break
        old = {(p.square.file, p.square.rank): (p.color, p.kind) for p in current.pieces}
        if consistent and all((p.square.file, p.square.rank) in cells
                              or old.get((p.square.file, p.square.rank)) == (p.color, p.kind)
                              for p in candidate.pieces):
            verified.append(candidate)
    return verified


def initial_frame_images(source, time_s, calibration, selected):
    from .initial_observation import observe_initial_board, compatible_observations
    reference = observe_initial_board(selected)[0]
    occupied = {square: color for square, color in reference.items() if color is not None}
    images = []
    for _, frame in iter_frames(source, max(0, time_s - .6), time_s + .001):
        image = rectify_board(frame, calibration, margin=100)
        cells = observe_initial_board(image)[0]
        supported = (any(cells.get(square) == color for square, color in occupied.items()) if occupied
                     else bool(cells))
        if compatible_observations(cells, reference) and supported:
            images.append(image)
        else:
            images = []
    if len(images) > 3:
        images = [images[0], images[len(images) // 2], images[-1]]
    return images or [selected]


def reconcile_position(current, observed, history):
    signature = board_signature(observed)
    for previous in reversed(history):
        if board_signature(previous) == signature:
            return previous.model_copy(deep=True)
    old = {(p.square.file, p.square.rank): p for p in current.pieces}
    pieces, used, remaining = [], set(), []
    for p in observed.pieces:
        previous = old.get((p.square.file, p.square.rank))
        if previous and (previous.color, previous.kind) == (p.color, p.kind):
            pieces.append(previous.model_copy(deep=True))
            used.add(previous.id)
        else:
            remaining.append(p)
    catalog = {p.id: p for state in history + [current] for p in state.pieces}
    for p in remaining:
        candidates = [old_piece for old_piece in catalog.values() if old_piece.id not in used
                      and (old_piece.color, old_piece.kind) == (p.color, p.kind)]
        previous = next((candidate for candidate in candidates if candidate.id == p.id), None)
        if previous is None and len(candidates) == 1:
            previous = candidates[0]
        # 同类棋子身份无法区分时保存实际局面，不编造它们的移动顺序。
        identity = previous.id if previous is not None else f"{p.color}_{p.kind}_{uuid4().hex[:12]}"
        pieces.append(p.model_copy(update={"id": identity}))
        used.add(identity)
    moved = [p for p in pieces if p.id in catalog and p.square != catalog[p.id].square]
    turn = ('black' if moved[0].color == 'red' else 'red') if len(moved) == 1 else current.side_to_move
    return Position(pieces=pieces, side_to_move=turn)


def snapshot_record(before, after, index, start_s, end_s):
    previous, following = {p.id: p for p in before.pieces}, {p.id: p for p in after.pieces}
    moved = [p for p in after.pieces if p.id in previous and p.square != previous[p.id].square]
    removed = [p for p in before.pieces if p.id not in following]
    added = [p for p in after.pieces if p.id not in previous]
    action = None
    identity_changed = any(p.id in previous and (p.color, p.kind) != (previous[p.id].color, previous[p.id].kind) for p in after.pieces)
    if len(moved) == 1 and not identity_changed and not added and len(removed) <= 1 and (
            not removed or removed[0].square == moved[0].square):
        piece = moved[0]
        action = Move(index=index, piece_id=piece.id, color=piece.color, source=previous[piece.id].square,
                      target=piece.square, captured_piece_id=removed[0].id if removed else None)
    return MoveRecord(index=index, move=action, video_start_s=start_s, video_end_s=end_s,
                      status="auto_verified", position_before=before, position_after=after, observed_positions=[after],
                      change_type="move" if action else "setup", reason="已按实际画面还原局面")


def revalidate_actual(task):
    if task.rules_enabled:
        from .turns import revalidate_turns
        return revalidate_turns(task)
    corrected = next((r.index for r in task.moves if r.manual_override), None)
    if corrected is not None:
        from .turns import replay_corrections
        return replay_corrections(task, corrected)
    task = task.model_copy(deep=True)
    if task.initial_position is None:
        task.confirmed_prefix_length = 0
        return task
    current, prefix, continuous = task.initial_position, 0, True
    for index, record in enumerate(task.moves, 1):
        record.index = index
        if record.position_after is not None and record.status in ('auto_verified', 'corrected'):
            rebuilt = snapshot_record(current, record.position_after, index, record.video_start_s, record.video_end_s)
            record.move, record.change_type = rebuilt.move, rebuilt.change_type
            current = record.position_after
            if continuous:
                prefix = index
        else:
            record.status, continuous = 'needs_review', False
    task.confirmed_prefix_length = prefix
    task.status = 'ready' if all(r.status in ('auto_verified', 'corrected') for r in task.moves) else 'needs_review'
    task.stage = '实际局面已保存' if task.status == 'ready' else '仅需核对未解决的画面'
    return task


def correct_actual(task, index, actions, time_s=None, position=None):
    if task.rules_enabled or len(actions) <= 1:
        from .turns import correct_turn
        return correct_turn(task, index, actions, time_s, position)
    if not task.initial_position or not 1 <= index <= len(task.moves) + 1:
        raise ValueError('请先建立起始局面并选择有效步骤')
    task = task.model_copy(deep=True)
    before = task.initial_position if index == 1 else task.moves[index - 2].position_after
    if before is None and position is not None:
        before = next((r.position_after for r in reversed(task.moves[:index - 1]) if r.position_after), task.initial_position)
    if before is None:
        raise ValueError('前一处局面尚未确定，请先确认局面')
    original = task.moves[index - 1] if index <= len(task.moves) else None
    preceding = task.moves[index - 2].video_end_s if index > 1 else task.start_s
    following = task.moves[index].video_end_s if index < len(task.moves) else task.end_s
    time_s = time_s if time_s is not None else original.video_end_s if original else task.processed_video_s
    if not preceding <= time_s <= following:
        raise ValueError('操作完成时间必须位于前后步骤的时间范围内')
    current, replacement = before, []
    if position is not None:
        after = reconcile_position(current, position, [task.initial_position] + [r.position_after for r in task.moves[:index - 1] if r.position_after])
        replacement.append(snapshot_record(current, after, index, preceding, time_s))
    else:
        for action in actions:
            piece = next((p for p in current.pieces if p.id == action.piece_id), None)
            if piece is None or piece.square != action.source or piece.color != action.color:
                raise ValueError('移动棋子的身份或原位置不一致')
            occupant = next((p for p in current.pieces if p.square == action.target and p.id != piece.id), None)
            if action.captured_piece_id and (occupant is None or occupant.id != action.captured_piece_id):
                raise ValueError('被移除棋子的身份不一致')
            after = Position(pieces=[p.model_copy(update={'square': action.target}) if p.id == piece.id else p
                for p in current.pieces if occupant is None or p.id != occupant.id],
                side_to_move='black' if piece.color == 'red' else 'red')
            replacement.append(snapshot_record(current, after, index + len(replacement), preceding, time_s))
            current = after
    for record in replacement:
        record.status = 'corrected'
        record.evidence_ids = original.evidence_ids if original else []
    task.moves[index - 1:index] = replacement
    return revalidate_actual(task)


def find_initial_frame(path, start_s, calibration, on_frame=None, allow_partial=False):
    from .analysis import read_board_cells
    from .initial_observation import observe_initial_board
    window, searched_to = 2.0, start_s
    while True:
        begin = max(0.0, start_s - window)
        latest, latest_cells, pending, since = None, None, None, None
        partial, partial_quality = None, None
        for time_s, frame in iter_frames(path, begin, searched_to):
            if on_frame:
                on_frame(time_s)
            image = rectify_board(frame, calibration, margin=100 if allow_partial else 50)
            cells = observe_initial_board(image)[0] if allow_partial else read_board_cells(image, strict=True)
            quality = len(cells), -abs(time_s - start_s)
            if partial_quality is None or quality > partial_quality:
                partial, partial_quality = (time_s, image), quality
            if len(cells) != 90:
                pending, since = None, None
                continue
            signature = tuple(sorted(cells.items()))
            if start_s == 0:
                return time_s, image
            if signature != pending:
                pending, since = signature, time_s
            elif time_s - since >= .12:
                if latest_cells is not None:
                    vanished = any(color is not None and cells[square] != color for square, color in latest_cells.items())
                    landed = any(color is not None and latest_cells[square] != color for square, color in cells.items())
                    if vanished and not landed:
                        continue
                latest = (time_s, image)
                latest_cells = cells
        if latest is not None:
            return latest
        if allow_partial and partial is not None:
            return partial
        if begin == 0:
            raise ValueError("区间起点及前置画面没有可确认的完整局面，请核对棋盘范围或起始画面")
        # 扩大到更早的画面，只有找到动作前的清晰局面才结束；不设固定前置秒数。
        searched_to, window = begin + .2, window * 2


def _cells_position(current, cells, history):
    occupied = {(p.square.file, p.square.rank): p for p in current.pieces}
    if len(cells) < 70:
        return None
    removed = [p for square, p in occupied.items() if square in cells and cells[square] != p.color]
    arrivals = [(square, color) for square, color in cells.items() if color is not None
                and (square not in occupied or occupied[square].color != color)]
    if not arrivals:
        return None
    pieces, used = [p.model_copy(deep=True) for p in current.pieces if p not in removed], set()
    for square, color in arrivals:
        candidates = [p for p in removed if p.color == color and p.id not in used]
        if len(candidates) != 1:
            for previous in reversed(history):
                previous_cells = {(p.square.file, p.square.rank): p.color for p in previous.pieces}
                if len(cells) == 90 and all(previous_cells.get(s) == c for s, c in cells.items()):
                    return previous.model_copy(deep=True)
            return None
        piece = candidates[0]
        used.add(piece.id)
        pieces.append(piece.model_copy(update={"square": Square(file=square[0], rank=square[1])}))
    leftovers = [p for p in removed if p.id not in used]
    # 没有落点证据时不能把拿在手中的棋子当成已移除；吃子须对应可见落点。
    if any((p.square.file, p.square.rank) not in dict(arrivals) for p in leftovers):
        return None
    if any((p.square.file, p.square.rank) not in cells for p in removed):
        return None
    turn = ('black' if next(p for p in removed if p.id in used).color == 'red' else 'red') if len(used) == 1 else current.side_to_move
    order = {p.id: i for i, p in enumerate(current.pieces)}
    pieces.sort(key=lambda p: order.get(p.id, len(order)))
    return Position(pieces=pieces, side_to_move=turn)


def _landing_visible(image, before, after):
    small = cv2.copyMakeBorder(cv2.resize(_tracking_image(image), (450, 500)), 30, 30, 30, 30, cv2.BORDER_REPLICATE)
    dy, dx = np.mgrid[-29:30, -29:30]
    ring = (dx * dx + dy * dy >= 22 * 22) & (dx * dx + dy * dy < 29 * 29)
    old = {p.id: p.square for p in before.pieces}
    for piece in after.pieces:
        if old.get(piece.id) == piece.square:
            continue
        x, y = 55 + piece.square.file * 50, 55 + piece.square.rank * 50
        patch = small[y - 29:y + 30, x - 29:x + 30]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[ring]
        pixels = patch[ring].astype(float)
        fingers = ((hsv[:, 0] < 23) & (hsv[:, 1] > 55) & (hsv[:, 2] > 95)
                   & (pixels[:, 2] > pixels[:, 1] * 1.2) & (pixels[:, 0] > pixels[:, 1] * .62))
        if fingers.mean() > .18:
            return False
    return True


def _glyphs(image, position):
    small = cv2.copyMakeBorder(cv2.resize(_tracking_image(image), (450, 500)), 40, 40, 40, 40, cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    result = []
    for p in position.pieces:
        x, y = 65 + p.square.file * 50, 65 + p.square.rank * 50
        face = small[y - 35:y + 36, x - 35:x + 36].astype(float)
        white = ((face[:, :, 0] > face[:, :, 2] * .93) & (face.max(axis=2) > 120)).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(white)
        if count > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cx, cy = centers[component]
            if abs(cx - 35) < 15 and abs(cy - 35) < 15:
                # 棋子在同一交叉点内偏移时先对齐棋子表面，避免把文字平移当成换子。
                x, y = x + round(cx - 35), y + round(cy - 35)
        patch = gray[y - 15:y + 16, x - 15:x + 16]
        ink = (patch < np.percentile(patch, 80) * .75).astype(np.float32)
        result.append(cv2.GaussianBlur(cv2.resize(ink, (16, 16)), (3, 3), 0))
    return np.asarray(result)


def actual_events(frames, initial, confirmed_position=None, turn_boundaries=False):
    from .analysis import VideoEvent
    first = next(frames, None)
    if first is None:
        return
    current, history = initial, [initial]
    cells = _tracking_cells(first[1], initial)
    occupied = {(p.square.file, p.square.rank): p.color for p in initial.pieces}
    first_clear = len(cells) == 90 and all(occupied.get(s) == c for s, c in cells.items())
    yield VideoEvent(first[0], [first[1]], position=initial)
    baseline, peak = first[1], None
    pending, since, fallback, fallback_since, emitted = None, 0, None, 0, None
    last_time, last_image, dirty = first[0], first[1], not first_clear
    glyph_reference, glyph_since, glyph_pending = _glyphs(first[1], initial) if first_clear else None, None, None
    glyph_checked = first[0]
    pending_after, pending_image = None, None
    pending_clear = False
    departure_emitted = False
    last_clear_s, moving_colors = first[0], ()
    for time_s, image in frames:
        last_time, last_image = time_s, image
        if confirmed_position is not None:
            reference = confirmed_position()
            if reference is not None and board_signature(reference) != board_signature(current):
                current = reference
                history.append(reference)
                baseline, peak, pending, dirty = image, None, None, False
                pending_after, departure_emitted = None, False
                glyph_reference, glyph_since, glyph_pending = None, None, None
        cells = _tracking_cells(image, current)
        occupied = {(p.square.file, p.square.rank): p.color for p in current.pieces}
        agrees = all(occupied.get(s) == c for s, c in cells.items())
        if turn_boundaries:
            arrivals = {color for square, color in cells.items()
                        if color is not None and occupied.get(square) != color}
            departing = {color for square, color in occupied.items()
                         if square in cells and cells[square] != color}
            colors = tuple(sorted(arrivals or departing))
            if colors and colors != moving_colors:
                yield VideoEvent(time_s, [baseline, image], moving_colors=colors, last_clear_s=last_clear_s,
                                 departure_only=not arrivals)
                moving_colors = colors
        if agrees:
            if len(cells) == 90 and pending_after is not None and pending_clear:
                yield VideoEvent(since, [baseline, pending_image, image], not turn_boundaries, position=pending_after,
                                 review_reason='' if turn_boundaries else '短暂落子后复位，需核对是否完成了移动和悔棋')
                if turn_boundaries:
                    yield VideoEvent(time_s, [image], position=current)
            if len(cells) == 90 and departure_emitted:
                yield VideoEvent(time_s, [image], position=current)
                departure_emitted = False
            pending_after = None
            pending, fallback = None, None
            if len(cells) == 90:
                last_clear_s, moving_colors = time_s, ()
                if time_s - glyph_checked < .08:
                    if glyph_since is None:
                        baseline, peak, dirty = image, None, False
                    continue
                glyph_checked = time_s
                glyphs = _glyphs(image, current)
                changed = glyph_reference is not None and len(glyphs) and bool(np.any(np.mean(np.abs(glyphs - glyph_reference), axis=(1, 2)) > .15))
                if changed:
                    stable = glyph_pending is not None and bool(np.max(np.mean(np.abs(glyphs - glyph_pending), axis=(1, 2))) < .035)
                    if not stable:
                        glyph_since, glyph_pending = time_s, glyphs
                    elif time_s - glyph_since >= 1:
                        yield VideoEvent(glyph_since, [baseline, image])
                        glyph_reference, glyph_since, glyph_pending = glyphs, None, None
                    continue
                glyph_reference, glyph_since, glyph_pending = glyphs, None, None
                baseline, peak, dirty = image, None, False
            elif any(s not in cells for s in occupied):
                dirty, peak = True, image
            continue
        dirty = True
        if peak is None:
            peak = image
        after = _cells_position(current, cells, history)
        moving = after is not None and not _landing_visible(image, current, after)
        if moving:
            after = None
        if after is not None:
            signature = board_signature(after)
            if pending != signature:
                if pending_after is not None and pending_clear:
                    yield VideoEvent(since, [baseline, pending_image, image], not turn_boundaries, position=pending_after,
                                     review_reason='' if turn_boundaries else '短暂落子后继续操作，需核对这一处变化')
                pending, since = signature, time_s
                pending_after, pending_image = after, image
                pending_clear = len(cells) == 90
            elif time_s - since >= .06:
                yield VideoEvent(since, [baseline, image], position=after)
                current = after
                history.append(after)
                last_clear_s = time_s
                moving_colors = ()
                baseline, peak, pending, dirty, emitted = image, None, None, False, None
                pending_after, departure_emitted = None, False
                glyph_reference, glyph_since, glyph_pending = None, None, None
            elif len(cells) == 90:
                pending_clear, pending_image = True, image
            continue
        pending = None
        if len(cells) == 90 and not moving:
            arrivals = any(color is not None and occupied.get(square) != color for square, color in cells.items())
            if not arrivals:
                signature = tuple(sorted(cells.items()))
                if fallback != signature:
                    fallback, fallback_since = signature, time_s
                elif time_s - fallback_since >= 1 and emitted != signature:
                    emitted, departure_emitted = signature, True
                    removed = Position(pieces=[p for p in current.pieces
                        if cells.get((p.square.file, p.square.rank)) == p.color], side_to_move=current.side_to_move)
                    yield VideoEvent(fallback_since, [baseline, image], True, position=removed,
                                     review_reason='仅有棋子离开棋盘，需确认移除还是拿起停顿')
                continue
            signature = tuple(sorted(cells.items()))
            if fallback != signature:
                fallback, fallback_since = signature, time_s
            elif time_s - fallback_since >= 1 and emitted != signature:
                emitted = signature
                yield VideoEvent(fallback_since, [baseline, peak, image])
        else:
            fallback = None
    if confirmed_position is not None:
        reference = confirmed_position()
        if reference is not None:
            occupied = {(p.square.file, p.square.rank): p.color for p in reference.pieces}
            if len(cells) == 90 and all(occupied.get(s) == c for s, c in cells.items()):
                dirty = False
    if pending_after is not None and pending_clear:
        yield VideoEvent(since, [baseline, pending_image], True, position=pending_after,
                         review_reason='区间结束前仅短暂出现落子，请核对或延长区间')
    elif dirty and not departure_emitted:
        yield VideoEvent(last_time, [baseline, last_image], True,
                         review_reason="区间结束时操作尚未完成或落点仍被遮挡，请延长区间或核对这处画面")


def analyze_actual(task, store, vision):
    from .analysis import read_board_cells
    from .piece_classifier import local_piece_classifier
    from .scanning import scan_video
    initializer = vision
    started, last_saved, previous_usage = monotonic(), monotonic(), list(task.usage)
    task.run_stats = AnalysisStats()
    rules = None
    folder = store.task_dir(task.id) / 'evidence'
    folder.mkdir(exist_ok=True)
    source = store.task_dir(task.id) / 'source.mp4'
    try:
        if not task.calibration:
            task.status, task.stage = 'needs_review', '未定位棋盘，请调整棋盘四角并保存'
            store.save_analysis(task)
            return
        if source.is_file() and not task.calibration_confirmed:
            from .calibration import refine_calibration
            frames = iter_frames(source, task.start_s, task.end_s)
            try:
                preview = next(frames, None)
            finally:
                frames.close()
            if preview is not None:
                existing = task.context_position or task.initial_position
                cells = read_board_cells(rectify_board(preview[1], task.calibration, margin=50), strict=True)
                known = {(p.square.file, p.square.rank): p.color for p in existing.pieces} if existing else None
                if known is None or len(cells) != 90 or any(known.get(s) != c for s, c in cells.items()):
                    task.calibration = refine_calibration(preview[1], task.calibration)
        if task.initial_position is None:
            initializer = vision or local_piece_classifier()
            task.stage = '向前寻找操作前的清晰局面'
            task = store.save_analysis(task)
            def on_frame(time_s):
                nonlocal task, last_saved
                task.analysis_stats.scanned_frames += 1
                task.run_stats.scanned_frames += 1
                if monotonic() - last_saved > 1:
                    task = store.save_analysis(task)
                    last_saved = monotonic()
            initial_time, image = find_initial_frame(source, task.start_s, task.calibration, on_frame, True)
            images = initial_frame_images(source, initial_time, task.calibration, image)
            name = uuid4().hex + '.jpg'
            save_image(folder / name, image)
            task.initial_evidence_ids, task.initial_frame_s = [name], initial_time
            if initializer is None:
                task = scan_video(task, store)
                task.status, task.stage = 'needs_review', '本地模式缺少起始棋子身份，请识别或设置初始局面'
                store.save_analysis(task)
                return
            task.stage = '自动识别所选区间的起始局面'
            if not getattr(initializer, 'local_classifier', None) and vision is not None:
                task = scan_video(task, store)
            task.analysis_stats.model_requests += 1
            task.analysis_stats.images_sent += len(images)
            task.run_stats.model_requests += 1
            task.run_stats.images_sent += len(images)
            task = store.save_analysis(task)
            requested = monotonic()
            try:
                candidates = initializer.recognize_position(images, actual=True)
            finally:
                task.run_stats.model_seconds += monotonic() - requested
                task.analysis_stats.model_requests += getattr(initializer, 'last_request_count', 1) - 1
                task.run_stats.model_requests += getattr(initializer, 'last_request_count', 1) - 1
                task.analysis_stats.images_sent += getattr(initializer, 'last_image_count', len(images)) - len(images)
                task.run_stats.images_sent += getattr(initializer, 'last_image_count', len(images)) - len(images)
                if getattr(initializer, 'last_response', None):
                    (store.task_dir(task.id) / 'vision-last-response.json').write_text(
                        json.dumps(initializer.last_response, ensure_ascii=False, indent=2), encoding='utf-8')
            candidates = verified_initial_candidates(candidates, images)
            unique = {board_signature(p): p for p in candidates}
            task.initial_candidates = list(unique.values())
            from .schemas import InitialPieceReview
            diagnostic = getattr(initializer, 'last_response', None) or {}
            task.initial_draft = Position.model_validate(diagnostic['draft_position']) if diagnostic.get('draft_position') is not None else None
            task.initial_piece_reviews = [InitialPieceReview.model_validate(r) for r in diagnostic.get('piece_reviews', [])]
            if len(unique) != 1:
                task.error = (getattr(initializer, 'last_response', None) or {}).get('error')
                task = store.save_analysis(task)
                task = scan_video(task, store)
                task.status, task.stage = 'needs_review', '已保存扫描结果和可识别部分，请核对未知位置后继续局面确认'
                if task.initial_draft is not None:
                    task.stage = '已复刻可识别部分，扫描结果已保存，未知位置待核对'
                task.usage = previous_usage + initializer.usage
                task.run_stats.elapsed_seconds = monotonic() - started
                store.save_analysis(task)
                return
            task.initial_position = next(iter(unique.values()))
            task.initial_turn_known = False
            task.initial_draft, task.initial_piece_reviews = None, []
            task = store.save_analysis(task)
        task = scan_video(task, store)
        scan_index = store.task_dir(task.id) / 'scan-index.json'
        scan_times = [item['time_s'] for item in json.loads(scan_index.read_text(encoding='utf8'))['events']] if scan_index.exists() else []
        if task.context_position is None:
            task.context_position = task.initial_position.model_copy(deep=True)
        task.run_stats.initialization_seconds = max(0, monotonic() - started - task.run_stats.model_seconds)
        current = task.context_position
        history = [task.initial_position] + [r.position_after for r in task.moves if r.position_after is not None] + [current]
        begin = task.resume_video_s if task.resume_video_s is not None else min(task.start_s, task.initial_frame_s) if task.initial_frame_s is not None else task.start_s
        tracking_calibration = task.calibration
        def boards():
            nonlocal task, last_saved, tracking_calibration
            frames = iter_frames(source, begin, task.end_s)
            first = True
            while True:
                decoding = monotonic()
                try:
                    time_s, frame = next(frames)
                except StopIteration:
                    break
                task.run_stats.decode_seconds += monotonic() - decoding
                if first:
                    first = False
                    alignment_frame, alignment_position = frame, current
                    if task.resume_video_s is not None:
                        alignment_time = task.initial_frame_s if task.initial_frame_s is not None else task.start_s
                        for _, initial_frame in iter_frames(source, alignment_time, alignment_time + .2):
                            alignment_frame, alignment_position = initial_frame, task.initial_position
                            break
                    cells = _tracking_cells(rectify_board(alignment_frame, tracking_calibration, margin=100), alignment_position)
                    occupied = {(p.square.file, p.square.rank): p.color for p in alignment_position.pieces}
                    if ((len(cells) < 90 or any(occupied.get(s) != c for s, c in cells.items()))
                            and any(color is not None for color in cells.values())):
                        from .calibration import refine_calibration
                        refined = refine_calibration(alignment_frame, tracking_calibration)
                        aligned = _tracking_cells(rectify_board(alignment_frame, refined, margin=100), alignment_position)
                        if len(aligned) == 90 and all(occupied.get(s) == c for s, c in aligned.items()):
                            # 只调整本次跟踪的线网映射，不覆盖用户确认的范围。
                            tracking_calibration = refined
                task.analysis_stats.scanned_frames += 1
                task.run_stats.scanned_frames += 1
                task.processed_video_s = max(task.start_s, time_s)
                if monotonic() - last_saved > 1:
                    task.stage = '跟踪棋子，等待操作完成'
                    task.run_stats.elapsed_seconds = monotonic() - started
                    task = store.save_analysis(task)
                    last_saved = monotonic()
                yield time_s, rectify_board(frame, tracking_calibration, margin=100)
        last_end, pending_review = task.start_s, None
        if task.rules_enabled:
            from .turns import TurnReplay
            rules = TurnReplay(task, current)
            events = actual_events(boards(), current, lambda: current, turn_boundaries=True)
        else:
            events = actual_events(boards(), current, lambda: current)
        while True:
            detecting, previous_decode = monotonic(), task.run_stats.decode_seconds
            try:
                event = next(events)
            except StopIteration:
                task.run_stats.opencv_seconds += max(0, monotonic() - detecting - task.run_stats.decode_seconds + previous_decode)
                break
            task.run_stats.opencv_seconds += max(0, monotonic() - detecting - task.run_stats.decode_seconds + previous_decode)
            if rules is not None and event.moving_colors:
                rules.task = task
                if event.time_s >= task.start_s:
                    previous_records = len(task.moves)
                    rules.movement(event.moving_colors, event.last_clear_s if event.last_clear_s is not None else event.time_s,
                                   event.departure_only)
                    rules.sync_pending()
                    task.recognition_video_s = max(task.recognition_video_s, event.time_s)
                    task.confirmed_prefix_length = next((r.index - 1 for r in task.moves
                        if r.status not in ('auto_verified', 'corrected')), len(task.moves))
                    task = store.save_analysis(task)
                    if (len(task.moves) > previous_records and rules.pending is None and rules.unresolved is None
                            and event.last_clear_s is not None):
                        from .recovery import save_boundary_checkpoint
                        save_boundary_checkpoint(store, task, event.last_clear_s, rules.before)
                continue
            task.analysis_stats.uncertain_events += int(event.uncertain)
            task.analysis_stats.stable_events += int(not event.uncertain)
            task.run_stats.uncertain_events += int(event.uncertain)
            task.run_stats.stable_events += int(not event.uncertain)
            images = event.images
            ids = []
            for image in images:
                name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
                if not (folder / name).exists():
                    save_image(folder / name, image)
                ids.append(name)
            positions = [event.position] if event.position is not None else []
            if not positions and not event.review_reason and vision is not None:
                task.stage = '结合已知局面与前后画面确认变化'
                task.analysis_stats.model_requests += 1
                task.analysis_stats.images_sent += min(3, len(images))
                task.run_stats.model_requests += 1
                task.run_stats.images_sent += min(3, len(images))
                task = store.save_analysis(task)
                requested = monotonic()
                try:
                    positions = vision.recognize_position(images, reference=current, actual=True)
                finally:
                    task.run_stats.model_seconds += monotonic() - requested
                    task.analysis_stats.model_requests += getattr(vision, 'last_request_count', 1) - 1
                    task.run_stats.model_requests += getattr(vision, 'last_request_count', 1) - 1
                positions = verified_observed_candidates(current, positions, images[0], images[-1])
            unique = {board_signature(p): p for p in positions}
            if rules is not None:
                rules.task = task
                if len(unique) == 1 and not event.review_reason:
                    after = reconcile_position(current, next(iter(unique.values())), history)
                    rules.observe(after, event.time_s, ids)
                    current = after
                    history.append(after)
                elif event.time_s >= task.start_s:
                    rules.uncertain(event.time_s, list(unique.values()), ids, event.review_reason)
                rules.sync_pending()
            elif len(unique) == 1 and not event.review_reason:
                after = reconcile_position(current, next(iter(unique.values())), history)
                if event.time_s < task.start_s:
                    task.initial_position = after
                else:
                    resolved_departure = False
                    if pending_review is not None and pending_review.reason.startswith('仅有棋子离开'):
                        remaining = {p.id for p in pending_review.observed_positions[0].pieces}
                        departed = {p.id for p in current.pieces} - remaining
                        resolved_departure = departed <= {p.id for p in after.pieces}
                    if pending_review is not None and not resolved_departure and not any(
                            board_signature(p) == board_signature(after) for p in pending_review.observed_positions):
                        pending_review.index = len(task.moves) + 1
                        task.moves.append(pending_review)
                        last_end = pending_review.video_end_s
                if event.time_s >= task.start_s and board_signature(after) != board_signature(current):
                    record = snapshot_record(current, after, len(task.moves) + 1, last_end, event.time_s)
                    record.evidence_ids = ids
                    task.moves.append(record)
                    last_end = event.time_s
                current = after
                history.append(after)
                # 遮挡期暂存的候选由后续清晰画面消除；不逐帧创建待人工确认步骤。
                pending_review = None
            elif event.time_s >= task.start_s:
                if pending_review is not None and pending_review.observed_positions:
                    pending_review.index = len(task.moves) + 1
                    task.moves.append(pending_review)
                    last_end = pending_review.video_end_s
                pending_review = MoveRecord(index=len(task.moves) + 1, video_start_s=last_end,
                    video_end_s=event.time_s, status='needs_review', evidence_ids=ids,
                    observed_positions=list(unique.values()), reason=event.review_reason or '结合前后画面仍无法确定完整局面')
            task.confirmed_prefix_length = next((r.index - 1 for r in task.moves
                if r.status not in ('auto_verified', 'corrected')), len(task.moves))
            task.usage = previous_usage + (vision.usage if vision is not None else [])
            task.run_stats.elapsed_seconds = monotonic() - started
            task.recognition_video_s = max(task.recognition_video_s, event.time_s)
            task.recognition_events += 1
            task.pending_event_count = sum(time_s > task.recognition_video_s for time_s in scan_times)
            task = store.save_analysis(task)
            if rules is None and pending_review is None and len(_tracking_cells(images[-1], current)) == 90:
                from .recovery import save_boundary_checkpoint
                save_boundary_checkpoint(store, task, task.processed_video_s, current)
        if rules is not None:
            rules.task = task
            rules.end(task.end_s)
        elif pending_review is not None:
            task.moves.append(pending_review)
        task.processed_video_s = task.end_s
        task.recognition_video_s = task.end_s
        task.pending_event_count = 0
        task.status = 'needs_review' if any(r.status not in ('auto_verified', 'corrected') for r in task.moves) else 'ready'
        task.stage = ('分析结束，请审核违规或疑点，后续走棋将在确认后重新校验' if task.status == 'needs_review'
                      else '按象棋规则复刻完成，可回放') if rules is not None else (
                      '分析结束，仅需核对未解决的画面' if task.status == 'needs_review' else '实际局面复刻完成，可回放')
        task.usage = previous_usage + (vision.usage if vision is not None else [])
        task.usage.append({'elapsed_seconds': round(monotonic() - started, 2)})
        task.run_stats.elapsed_seconds = monotonic() - started
        store.save_analysis(task)
    except RecoveryCancelled:
        return
    except VisionRequestError as exc:
        if task.recovery is not None:
            from .recovery import defer_operation
            task.usage = previous_usage + vision.usage
            task.run_stats.elapsed_seconds = monotonic() - started
            store.save_analysis(task)
            defer_operation(getattr(store, 'store', store), task, exc)
            return
        task.status, task.stage, task.error = 'failed', '分析未完成', str(exc)
        store.save_analysis(task)
    except Exception as exc:
        if rules is not None:
            rules.task = task
            rules.interrupt()
        if getattr(vision, 'last_response', None):
            (store.task_dir(task.id) / 'vision-last-response.json').write_text(
                json.dumps(vision.last_response, ensure_ascii=False, indent=2), encoding='utf-8')
        task.status, task.stage = 'failed', '分析未完成'
        task.error = str(exc) if isinstance(exc, ValueError) else '分析发生错误，请检查视频、定位和已保存证据'
        task.usage = previous_usage + (vision.usage if vision is not None else [])
        task.run_stats.elapsed_seconds = monotonic() - started
        store.save_analysis(task)
    finally:
        if vision is not None:
            vision.close()
