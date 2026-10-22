import json
import os
import hashlib
import weakref
from collections import deque, OrderedDict
from contextvars import ContextVar
from contextlib import ExitStack
from time import perf_counter as monotonic
from uuid import uuid4

import cv2
import numpy as np

from .calibration import rectify_board
from .chess import apply_move, board_signature
from .schemas import AnalysisStats, Move, MoveRecord, Position, Square
from .video import iter_frames, save_image
from .storage import RecoveryCancelled
from .vision import VisionRequestError
from .board_evidence import check_transition, needs_backfill, selected_events


def _tracking_image(image):
    from .initial_observation import board_geometry
    cell, margin = board_geometry(image)
    inset = max(0, round(margin - cell / 2)) if cell > 0 else 0
    return image[inset:-inset, inset:-inset] if inset else image


_tracking_cache = ContextVar('tracking_cells_cache', default=None)


def _trim_tracking_cache(cache):
    for key in list(cache):
        if cache[key][0]() is None:
            del cache[key]
    ordinary = [key for key in cache if key[1] != 'identity']
    for key in ordinary[:-64]:
        del cache[key]


def _identity_evidence(image):
    cache = _tracking_cache.get()
    proof = cache.get((id(image), 'identity')) if cache is not None else None
    return proof if proof is not None and proof[0]() is image else None


def _tracking_cells(image, reference=None):
    proof = _identity_evidence(image)
    if proof is not None and len(proof) > 4:
        cells = dict(proof[4])
        cells.update({square: color for square, color in proof[1].items()
                      if square not in cells or cells[square] is not None})
        return cells
    cells = _base_tracking_cells(image, reference)
    if proof is not None:
        cells.update({square: color for square, color in proof[1].items() if square not in cells})
    return cells


def _base_tracking_cells(image, reference=None):
    cache = _tracking_cache.get()
    if cache is None:
        return _read_tracking_cells(image, reference)
    if image.flags.writeable:
        for key in [key for key in cache if key[0] == id(image)]:
            del cache[key]
        return _read_tracking_cells(image, reference)
    occupied = None if reference is None else tuple(sorted((p.square.file, p.square.rank, p.color) for p in reference.pieces))
    key = id(image), occupied
    cached = cache.get(key)
    if cached is not None and cached[0]() is image:
        cache.move_to_end(key)
        return dict(cached[1])
    prepared = cache.get((id(image), 'prepared'))
    if prepared is not None and prepared[0]() is image:
        raw, normal, enhanced = prepared[1]
        reference_cells = {(p.square.file, p.square.rank): p.color for p in reference.pieces} if reference is not None else None
        changed = reference_cells is not None and any(reference_cells.get(s) != c for s, c in raw.items())
        if len(raw) < 90 or changed:
            if enhanced is None:
                enhanced = _finish_tracking_cells(image, raw, True)
                cache[id(image), 'prepared'] = weakref.ref(image), (raw, normal, enhanced)
            cells = dict(enhanced)
        else:
            cells = dict(normal)
        if reference_cells is not None and any(color is not None and reference_cells.get(s) != color
                                               for s, color in cells.items()):
            small = cv2.copyMakeBorder(cv2.resize(_tracking_image(image), (450, 500)), 30, 30, 30, 30, cv2.BORDER_REPLICATE)
            cells = _filter_reference_cells(small, cells, reference_cells, raw)
    else:
        cells = _read_tracking_cells(image, reference)
    cache[key] = weakref.ref(image), dict(cells)
    _trim_tracking_cache(cache)
    return cells


def _read_tracking_cells(image, reference=None):
    from .analysis import read_board_cells
    cells = read_board_cells(_tracking_image(image), strict=True)
    occupied = {(p.square.file, p.square.rank): p.color for p in reference.pieces} if reference is not None else None
    changed = occupied is not None and any(occupied.get(s) != c for s, c in cells.items())
    return _finish_tracking_cells(image, cells, len(cells) < 90 or changed, occupied)


def _finish_tracking_cells(image, cells, expand, reference_cells=None):
    from .initial_observation import observe_initial_board
    from .numerical import finger_counts, RING_SIZE
    from .board_colors import wood_surface_mask, skin_surface_mask
    raw_cells = cells
    cells = dict(cells)
    if expand and image.shape[0] > image.shape[1]:
        original_cells = dict(cells)
        cells, faces = observe_initial_board(image, cells=cells)
        for square, (x, y, radius) in faces.items():
            face = cv2.getRectSubPix(image, (round(radius * 1.4), round(radius * 1.4)), (x, y))
            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            skin = skin_surface_mask(face_hsv)
            if skin is not None and skin.mean() > .08:
                cells.pop(square, None)
                continue
            wood = wood_surface_mask(face_hsv)
            if wood is not None and wood.mean() > .5:
                if square in original_cells and original_cells[square] is None:
                    cells[square] = None
                else:
                    cells.pop(square, None)
                continue
            if not _piece_lettering_visible(face):
                cells.pop(square, None)
    small = cv2.copyMakeBorder(cv2.resize(_tracking_image(image), (450, 500)), 30, 30, 30, 30, cv2.BORDER_REPLICATE)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    squares = [square for square, color in cells.items() if color is not None]
    counts = finger_counts(small, hsv, np.asarray(squares, dtype=np.int64).reshape(-1, 2), wood_surface_mask(hsv))
    for square, count in zip(squares, counts):
        if count / RING_SIZE > .18:
            cells.pop(square)
    return _filter_reference_cells(small, cells, reference_cells, raw_cells)


def _piece_lettering_visible(face):
    gray = cv2.cvtColor(cv2.resize(face, (48, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.mgrid[:48, :48]
    center = (xx - 24) ** 2 + (yy - 24) ** 2 < 13 ** 2
    ink = cv2.GaussianBlur(gray, (0, 0), 4) - gray
    return np.percentile(gray[center], 95) - np.percentile(gray[center], 5) >= 10 and (ink[center] > 10).mean() >= .02


def _filter_reference_cells(small, cells, reference_cells, raw_cells):
    from .board_colors import skin_surface_mask
    if reference_cells is None:
        return cells
    skin = skin_surface_mask(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    if skin is not None:
        from .analysis import _cell_sampling
        _, _, ys, xs, _, _, _ = _cell_sampling()
        covered = skin[ys + 30, xs + 30].mean(axis=1)
        for (file, rank), color in list(cells.items()):
            expected = reference_cells.get((file, rank))
            if color is not None and expected is not None and expected != color and covered[rank * 9 + file] > .03:
                cells.pop((file, rank), None)
    for (file, rank), color in list(cells.items()):
        if color is not None and reference_cells.get((file, rank)) != color and raw_cells.get((file, rank)) == color:
            x, y = 55 + file * 50, 55 + rank * 50
            if not _piece_lettering_visible(small[y-18:y+18, x-18:x+18]):
                cells.pop((file, rank), None)
    return cells


def _prepare_tracking_frame(item, tracking_calibration, scan_calibration, scan_end, scan_needed):
    from .analysis import read_board_cells
    time_s, frame, decoding, reference_cells, *calibrations = item
    if calibrations:
        tracking_calibration = calibrations[0]
    if cv2.getNumThreads() != 1:
        cv2.setNumThreads(1)
    image = rectify_board(frame, tracking_calibration, margin=100)
    raw = read_board_cells(_tracking_image(image), strict=True)
    changed = any(reference_cells.get(square) != color for square, color in raw.items())
    enhanced = _finish_tracking_cells(image, raw, True) if len(raw) < 90 or changed else None
    normal = _finish_tracking_cells(image, raw, False) if len(raw) == 90 else enhanced
    scan_image = rectify_board(frame, scan_calibration, margin=50) if scan_needed and time_s <= scan_end + 1e-6 else None
    return time_s, image, scan_image, (raw, normal, enhanced), decoding


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


class PositionHistory(list):
    def __init__(self, positions):
        super().__init__()
        self.by_board, self.by_cells, self.catalog = {}, {}, {}
        for position in positions:
            self.append(position)

    def append(self, position):
        super().append(position)
        self.by_board[board_signature(position)] = position
        occupied = tuple(sorted((p.square.file, p.square.rank, p.color) for p in position.pieces))
        self.by_cells[occupied] = position
        self.catalog.update((p.id, p) for p in position.pieces)


def reconcile_position(current, observed, history):
    signature = board_signature(observed)
    if isinstance(history, PositionHistory):
        previous = history.by_board.get(signature)
        if previous is not None:
            return previous.model_copy(deep=True)
    else:
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
    catalog = (history.catalog | {p.id: p for p in current.pieces}) if isinstance(history, PositionHistory) else {
        p.id: p for state in history + [current] for p in state.pieces}
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
        if record.status == 'tentative' and record.predicted_position is not None:
            record.position_after, current, continuous = None, record.predicted_position, False
            continue
        if record.position_after is not None and record.status in ('auto_verified', 'corrected'):
            rebuilt = snapshot_record(current, record.position_after, index, record.video_start_s, record.video_end_s)
            record.move, record.change_type = rebuilt.move, rebuilt.change_type
            current = record.position_after
            if continuous:
                prefix = index
        else:
            record.status, continuous = 'needs_review', False
            if record.tracking_anchor is not None:
                current = record.tracking_anchor
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
    time_s = time_s if time_s is not None else original.video_end_s if original else task.processed_video_s
    if not 0 <= time_s <= task.metadata.duration_s:
        raise ValueError('时间必须位于原视频范围内')
    preceding = min(preceding, time_s)
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


def _identity_matches(pieces, identities):
    if not identities:
        return True
    occupied = {(p.square.file, p.square.rank): p.id for p in pieces}
    return all(occupied.get(square) == identity for square, identity in identities.items())


def _cells_position(current, cells, history, identities=None):
    occupied = {(p.square.file, p.square.rank): p for p in current.pieces}
    removed = [p for square, p in occupied.items() if square in cells and cells[square] != p.color]
    arrivals = [(square, color) for square, color in cells.items() if color is not None
                and (square not in occupied or occupied[square].color != color)]
    if not arrivals:
        return None
    pieces, used = [p.model_copy(deep=True) for p in current.pieces if p not in removed], set()
    for square, color in arrivals:
        candidates = [p for p in removed if p.color == color and p.id not in used]
        if identities and square in identities:
            candidates = [p for p in candidates if p.id == identities[square]]
        if len(candidates) != 1:
            if len(cells) == 90:
                if isinstance(history, PositionHistory) and all(0 <= f < 9 and 0 <= r < 10 for f, r in cells):
                    occupied = tuple(sorted((f, r, c) for (f, r), c in cells.items() if c is not None))
                    previous = history.by_cells.get(occupied)
                    if previous is not None and _identity_matches(previous.pieces, identities):
                        return previous.model_copy(deep=True)
                else:
                    for previous in reversed(history):
                        previous_cells = {(p.square.file, p.square.rank): p.color for p in previous.pieces}
                        if (all(previous_cells.get(s) == c for s, c in cells.items())
                                and _identity_matches(previous.pieces, identities)):
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
    if not _identity_matches(pieces, identities):
        return None
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


def actual_events(frames, initial, confirmed_position=None, turn_boundaries=False, cell_reader=None, selected_end_s=None, confirmed_image=None, turn_origin=None, recovery_reader=None, reanchor_reader=None, inference_active=None):
    from .analysis import VideoEvent, _occupied_squares
    first = next(frames, None)
    if first is None:
        return
    current, history = initial, PositionHistory([initial])
    reader = cell_reader or _tracking_cells
    cells = reader(first[1], initial)
    witnesses = {s: (color, first[1], first[0]) for s, color in cells.items()}
    before_witnesses = dict(witnesses)
    witness_reference = board_signature(initial)
    last_recovery_s = first[0] - 1
    witness_times = {id(first[1]): first[0]}
    occupied = {(p.square.file, p.square.rank): p.color for p in initial.pieces}
    first_clear = len(cells) == 90 and all(occupied.get(s) == c for s, c in cells.items())
    yield VideoEvent(first[0], [first[1]], position=initial)
    baseline, peak = first[1], None
    pending, since, fallback, fallback_since, emitted = None, 0, None, 0, None
    last_time, last_image, dirty = first[0], first[1], not first_clear
    glyph_reference, glyph_since, glyph_pending = _glyphs(first[1], initial) if first_clear else None, None, None
    proof = _identity_evidence(first[1])
    glyph_tokens = proof[3] if proof is not None and len(proof) > 3 else {}
    glyph_checked = first[0]
    pending_after, pending_image = None, None
    pending_clear = False
    departure_emitted = False
    source_started = {}
    pending_start_s = None
    rejected, rejected_since, rejected_after, rejected_image, rejected_emitted = None, 0, None, None, None
    last_clear_s, moving_colors = first[0], ()
    rejected_feedback = None
    for time_s, image in frames:
        if (selected_end_s is not None and time_s > selected_end_s and len(history) == 1
                and not dirty and pending_after is None and not source_started
                and not (inference_active is not None and inference_active())):
            break
        last_time, last_image = time_s, image
        if reanchor_reader is not None and confirmed_position is not None and (selected_end_s is None or time_s <= selected_end_s
                or inference_active is not None and inference_active()):
            anchor_event = reanchor_reader(time_s, image, confirmed_position())
            if anchor_event is not None:
                if anchor_event.selection_complete:
                    break
                unchanged = _occupied_squares(confirmed_position()) == _occupied_squares(anchor_event.position)
                yield anchor_event
                if unchanged:
                    continue
                reference = confirmed_position()
                if board_signature(reference) == board_signature(anchor_event.position):
                    current, history = reference, PositionHistory([reference])
                    baseline, peak, dirty = image, None, False
                    pending, pending_after, fallback, emitted = None, None, None, None
                    pending_image, pending_start_s, pending_clear = None, None, False
                    source_started, departure_emitted = {}, False
                    rejected, rejected_after, rejected_feedback, rejected_emitted = None, None, None, None
                    glyph_reference, glyph_since, glyph_pending = None, None, None
                    last_clear_s, moving_colors = time_s, ()
                    witnesses = {}
                    for sample_time, sample_image in zip(anchor_event.image_times_s, anchor_event.images):
                        witnesses.update({s: (c, sample_image, sample_time) for s, c in reader(sample_image, current).items()})
                    if not witnesses:
                        witnesses = {s: (c, image, time_s) for s, c in reader(image, current).items()}
                    before_witnesses = dict(witnesses)
                    witness_times = {id(item[1]): item[2] for item in witnesses.values()}
                    witness_reference = board_signature(current)
                    cells = reader(image, current)
                    continue
        if inference_active is not None and inference_active():
            continue
        witness_times[id(image)] = time_s
        if len(witness_times) > 512:
            retained = {id(baseline), id(image)} | {id(v[1]) for v in witnesses.values()} | {id(v[1]) for v in before_witnesses.values()}
            witness_times = {key: value for key, value in witness_times.items() if key in retained}
        if confirmed_position is not None:
            reference = confirmed_position()
            if reference is not None and board_signature(reference) != board_signature(current):
                current = reference
                history.append(reference)
                baseline = confirmed_image() if confirmed_image is not None else baseline
                peak, pending, dirty = None, None, True
                pending_after, departure_emitted = None, False
                glyph_reference, glyph_since, glyph_pending = None, None, None
        cells = reader(image, current)
        occupied = {(p.square.file, p.square.rank): p.color for p in current.pieces}
        signature = board_signature(current)
        if signature != witness_reference:
            before_witnesses = {s: item for s, item in before_witnesses.items() if item[0] == occupied.get(s)}
            before_witnesses.update({s: item for s, item in witnesses.items() if item[0] == occupied.get(s)})
            witness_reference = signature
        if (recovery_reader is not None and time_s - last_recovery_s >= 1
                and any(occupied.get(s) != color for s, color in cells.items())):
            last_recovery_s = time_s
            cells.update(recovery_reader(image, current))
        for square, color in cells.items():
            witnesses[square] = color, image, time_s
            if color == occupied.get(square):
                before_witnesses[square] = color, image, time_s
        source_started = {s: t for s, t in source_started.items() if s in occupied}
        preview = cv2.resize(_tracking_image(image), (450, 500))
        previous = cv2.resize(_tracking_image(baseline), (450, 500))
        for square, color in occupied.items():
            if cells.get(square, 'unknown') == color:
                source_started.pop(square, None)
            else:
                x, y = 25 + square[0] * 50, 25 + square[1] * 50
                change = cv2.absdiff(preview[y-18:y+19, x-18:x+19], previous[y-18:y+19, x-18:x+19])
                if square in cells or np.mean(np.max(change, axis=2) > 30) > .3:
                    source_started.setdefault(square, time_s)
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
            if confirmed_position is not None and len(cells) == 90 and dirty and pending_after is None and not departure_emitted:
                yield VideoEvent(time_s, [image], position=current)
            if pending_after is not None and len(cells) < 90:
                dirty, peak = True, image
                continue
            if len(cells) == 90 and pending_after is not None and pending_clear:
                yield VideoEvent(since, [baseline, pending_image], not turn_boundaries, position=pending_after,
                                 review_reason='' if turn_boundaries else '短暂落子后复位，需核对是否完成了移动和悔棋',
                                 action_start_s=pending_start_s)
                if turn_boundaries:
                    yield VideoEvent(time_s, [image], position=current)
            if len(cells) == 90 and departure_emitted:
                yield VideoEvent(time_s, [image], position=current)
                departure_emitted = False
            pending_after = None
            pending, fallback = None, None
            if len(cells) == 90:
                rejected, rejected_after = None, None
                rejected_feedback, rejected_emitted = None, None
                last_clear_s, moving_colors = time_s, ()
                if time_s - glyph_checked < .08:
                    if glyph_since is None:
                        baseline, peak, dirty = image, None, False
                    continue
                glyph_checked = time_s
                glyphs = _glyphs(image, current)
                proof = _identity_evidence(image)
                tokens = proof[3] if proof is not None and len(proof) > 3 else {}
                changed = (glyph_reference is not None and len(glyphs)
                           and any(delta > .15 and (piece.id not in tokens or tokens[piece.id] != glyph_tokens.get(piece.id))
                                   for piece, delta in zip(current.pieces, np.mean(np.abs(glyphs - glyph_reference), axis=(1, 2)))))
                if changed:
                    stable = glyph_pending is not None and bool(np.max(np.mean(np.abs(glyphs - glyph_pending), axis=(1, 2))) < .035)
                    if not stable:
                        glyph_since, glyph_pending = time_s, glyphs
                    elif time_s - glyph_since >= 1:
                        yield VideoEvent(glyph_since, [baseline, image])
                        glyph_reference, glyph_since, glyph_pending = glyphs, None, None
                        glyph_tokens = tokens
                    continue
                glyph_reference, glyph_since, glyph_pending = glyphs, None, None
                glyph_tokens = tokens
                baseline, peak, dirty = image, None, False
            elif any(s not in cells for s in occupied):
                dirty, peak = True, image
                if (confirmed_position is not None and not departure_emitted
                        and any(occupied.get(s) != color for s, color in reader(baseline, current).items())):
                    # 仅替换仍有矛盾的旧画面；兼容的清晰证据继续保留，遮挡格仍未知。
                    baseline = image
            continue
        dirty = True
        if peak is None:
            peak = image
        proof = _identity_evidence(image)
        identities = proof[2] if proof is not None and len(proof) > 2 else {}
        after = _cells_position(current, cells, history, identities)
        suspended = set()
        if after is None and confirmed_position is not None:
            for previous_position in reversed(history):
                if not _identity_matches(previous_position.pieces, identities):
                    continue
                action = snapshot_record(current, previous_position, 1, 0, time_s).move
                if action is None:
                    continue
                source, target = (action.source.file, action.source.rank), (action.target.file, action.target.rank)
                source_seen, target_seen = witnesses.get(source), witnesses.get(target)
                if (source_seen is None or target_seen is None or source_seen[0] is not None
                        or target_seen[0] != action.color
                        or min(source_seen[2], target_seen[2]) <= last_clear_s):
                    continue
                proposed = {(p.square.file, p.square.rank): p.color for p in previous_position.pieces}
                # 复位的起终点可能分帧露出，随后另一枚子已经被拿起；离盘格保留未知。
                lifted = {s for s, color in cells.items() if color is None and proposed.get(s) is not None
                          and s not in (source, target)}
                if any(proposed.get(s) != color for s, color in cells.items() if s not in lifted):
                    continue
                after, suspended = previous_position.model_copy(deep=True), lifted
                break
        if after is None and confirmed_position is not None:
            after = _cells_position(current, {s: item[0] for s, item in witnesses.items()}, history, identities)
        if after is None and confirmed_position is not None and turn_boundaries:
            observed_cells = {s: item[0] for s, item in witnesses.items()}
            arrivals = {color for s, color in observed_cells.items() if color is not None and occupied.get(s) != color}
            if len(arrivals) == 1:
                color = next(iter(arrivals))
                lifted = {s for s, original in occupied.items() if original != color
                          and s in observed_cells and observed_cells[s] is None}
                if lifted:
                    candidate = _cells_position(current, {s: value for s, value in observed_cells.items() if s not in lifted}, history, identities)
                    if candidate is not None and snapshot_record(current, candidate, 1, 0, time_s).move is not None:
                        after, suspended = candidate, lifted
        moving = after is not None and not _landing_visible(image, current, after)
        if after is not None and confirmed_position is not None:
            action = snapshot_record(current, after, 1, 0, time_s).move
            if action is not None:
                mover = next(p for p in current.pieces if p.id == action.piece_id)
                # 同类棋子的另一处起点仍在操作中时，不能用暂时唯一的空格抢先绑定身份。
                for piece in current.pieces:
                    square = piece.square.file, piece.square.rank
                    if (piece.id == mover.id or (piece.color, piece.kind) != (mover.color, mover.kind)
                            or square in cells or square not in source_started):
                        continue
                    alternative = action.model_copy(update={'piece_id': piece.id, 'source': piece.square})
                    try:
                        apply_move(current.model_copy(update={'side_to_move': piece.color}), alternative)
                    except ValueError:
                        continue
                    moving = True
                    break
        if moving:
            after = None
        historical_return = (after is not None and confirmed_position is not None
                             and board_signature(after) in history.by_board
                             and board_signature(after) != board_signature(current))
        if after is not None and turn_boundaries:
            action = snapshot_record(current, after, 1, 0, time_s).move
            if action is not None and not historical_return:
                origin = turn_origin() if turn_origin is not None else None
                original_action = snapshot_record(origin, after, 1, 0, time_s).move if origin is not None else None
                if original_action is not None and original_action.piece_id == action.piece_id:
                    # 同一回合的连续调整按最初起点校验，中间位置不作为另一手的起点。
                    legal_before, legal_action = origin, original_action
                else:
                    legal_before, legal_action = current, action
                try:
                    apply_move(legal_before.model_copy(update={'side_to_move': legal_action.color}), legal_action)
                except ValueError:
                    signature = board_signature(after)
                    if rejected != signature:
                        rejected, rejected_since, rejected_after, rejected_image = signature, time_s, after, image
                    elif time_s - rejected_since >= .18 and rejected_emitted != signature:
                        rejected_emitted = signature
                        yield VideoEvent(rejected_since, [baseline, rejected_image, image], True, position=after,
                            review_reason='局部画面持续出现未解释的棋子变化：落点不符合棋子走法，需核对中间操作或误识别')
                    continue
            rejected, rejected_after = None, None
        if after is not None:
            signature = board_signature(after)
            historical_return = (confirmed_position is not None and signature in history.by_board
                                 and signature != board_signature(current))
            if pending != signature:
                if pending_after is not None and pending_clear:
                    proposed = {(p.square.file, p.square.rank): p.color for p in pending_after.pieces}
                    contradicted = any(s in cells and cells[s] == occupied.get(s) and cells[s] != color
                                       for s, color in proposed.items() if occupied.get(s) != color)
                    if not contradicted:
                        yield VideoEvent(since, [baseline, pending_image], not turn_boundaries, position=pending_after,
                                         review_reason='' if turn_boundaries else '短暂落子后继续操作，需核对这一处变化',
                                         action_start_s=pending_start_s)
                        reference = confirmed_position() if confirmed_position is not None else pending_after
                        if reference is not None and board_signature(reference) == board_signature(pending_after):
                            current = pending_after
                            history.append(current)
                            baseline = pending_image
                            glyph_reference, glyph_since, glyph_pending = None, None, None
                            after = _cells_position(current, cells, history, identities)
                            if after is None:
                                pending, pending_after = None, None
                                continue
                            signature = board_signature(after)
                pending, since = signature, time_s
                pending_after, pending_image = after, image
                pending_clear = True
                changed_sources = [(p.square.file, p.square.rank) for p in current.pieces
                                   if next((q.square for q in after.pieces if q.id == p.id), p.square) != p.square]
                pending_start_s = source_started.get(changed_sources[0], time_s) if len(changed_sources) == 1 else None
            if historical_return or (time_s > since and (turn_boundaries or time_s - since >= .06)):
                feedback_key = signature, tuple(sorted(cells.items()))
                if feedback_key != rejected_feedback:
                    action = snapshot_record(current, after, 1, 0, since).move
                    check, evidence_images, evidence_times = None, [baseline, image], ()
                    if action is not None and confirmed_position is not None:
                        before_cells = reader(baseline, current)
                        after_cells = {s: color for s, color in cells.items() if s not in suspended}
                        evidence = {id(image): (time_s, image),
                                    id(baseline): (witness_times.get(id(baseline), first[0]), baseline)}
                        for square in (action.source, action.target):
                            key = square.file, square.rank
                            earlier, later = before_witnesses.get(key), witnesses.get(key)
                            if key not in before_cells and earlier is not None:
                                before_cells[key] = earlier[0]
                                evidence[id(earlier[1])] = earlier[2], earlier[1]
                            if key not in after_cells and later is not None:
                                after_cells[key] = later[0]
                                evidence[id(later[1])] = later[2], later[1]
                        check = check_transition(current, after, before_cells, after_cells)
                        if check.status != 'conflict':
                            ordered = sorted(evidence.values(), key=lambda item: item[0])
                            evidence_times = tuple(item[0] for item in ordered)
                            evidence_images = [item[1] for item in ordered]
                    yield VideoEvent(since, evidence_images, position=after, action_start_s=pending_start_s,
                                     image_times_s=evidence_times, board_check=check)
                if confirmed_position is not None:
                    reference = confirmed_position()
                    if reference is not None and board_signature(reference) != signature:
                        rejected_feedback = feedback_key
                        pending_after = None
                        continue
                rejected_feedback = None
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
                                     review_reason='仅有棋子离开棋盘，需确认移除还是拿起停顿',
                                     action_start_s=min(source_started.values(), default=None),
                                     moving_piece_ids=tuple(p.id for p in current.pieces if (p.square.file, p.square.rank) in source_started))
                continue
            signature = tuple(sorted(cells.items()))
            if fallback != signature:
                fallback, fallback_since = signature, time_s
            elif time_s - fallback_since >= 1 and emitted != signature:
                emitted = signature
                yield VideoEvent(fallback_since, [baseline, peak, image])
        else:
            changes = tuple(sorted((s, c) for s, c in cells.items() if occupied.get(s) != c))
            if changes and any(color is not None for _, color in changes) and not moving:
                if fallback != changes:
                    fallback, fallback_since = changes, time_s
                elif time_s - fallback_since >= .18 and emitted != changes:
                    emitted = changes
                    yield VideoEvent(fallback_since, [baseline, peak, image], True,
                                     review_reason='局部画面持续出现未解释的棋子变化，可能缺少中间走棋，请核对这段视频',
                                     action_start_s=min(source_started.values(), default=None),
                                     moving_piece_ids=tuple(p.id for p in current.pieces if (p.square.file, p.square.rank) in source_started))
            else:
                fallback = None
    if inference_active is not None and inference_active():
        return
    if confirmed_position is not None:
        reference = confirmed_position()
        if reference is not None:
            occupied = {(p.square.file, p.square.rank): p.color for p in reference.pieces}
            if len(cells) == 90 and all(occupied.get(s) == c for s, c in cells.items()):
                dirty = False
    if rejected_after is not None and rejected_emitted != rejected:
        yield VideoEvent(rejected_since, [baseline, rejected_image], True, position=rejected_after,
                         review_reason='局部画面持续出现未解释的棋子变化：区间结束前的落点尚未确认')
    if pending_after is not None and pending_clear:
        yield VideoEvent(since, [baseline, pending_image], True, position=pending_after,
                         review_reason='区间结束前仅短暂出现落子，请核对或延长区间', action_start_s=pending_start_s)
    elif dirty and not departure_emitted:
        yield VideoEvent(last_time, [baseline, last_image], True,
                         review_reason="原视频结束时操作尚未完成或落点仍被遮挡，请核对这处画面",
                         action_start_s=min(source_started.values(), default=None),
                         moving_piece_ids=tuple(p.id for p in current.pieces if (p.square.file, p.square.rank) in source_started))


def backfill_interval(source, start_s, end_s, calibration, before, expected, on_frame=None, motion_active=False,
                      motion_reference_s=None, motion_reference_calibration=None):
    from .initial_observation import observe_initial_board
    from .board_motion import BoardMotion
    timestamps = {}
    def frames():
        motion = BoardMotion(calibration, active=motion_active,
                             reference_calibration=motion_reference_calibration) if calibration is not None else None
        if motion is not None and motion_reference_s is not None and motion_reference_s < start_s:
            for reference_time, reference_frame in iter_frames(source, motion_reference_s, motion_reference_s + .2):
                motion.update(reference_time, reference_frame)
                if on_frame is not None:
                    on_frame(reference_time)
                break
        for time_s, frame in iter_frames(source, start_s, end_s):
            if on_frame is not None:
                on_frame(time_s)
            image = rectify_board(frame, motion.update(time_s, frame) if motion is not None else calibration, margin=100)
            timestamps[id(image)] = (weakref.ref(image), time_s)
            if len(timestamps) > 256:
                for key in [key for key, value in timestamps.items() if value[0]() is None]:
                    del timestamps[key]
            yield time_s, image
    reader = lambda image, reference: observe_initial_board(image)[0]
    current, recovered, reference_image = before, [], None
    origin, moving_piece = before, None
    for event in actual_events(frames(), before, confirmed_position=lambda: current, turn_boundaries=True, cell_reader=reader,
                               confirmed_image=lambda: reference_image, turn_origin=lambda: origin):
        if event.moving_colors:
            continue
        if event.review_reason or event.position is None:
            continue
        check = check_transition(current, event.position, reader(event.images[0], current),
                                 reader(event.images[-1], event.position))
        if check.status == 'conflict':
            continue
        if board_signature(current) != board_signature(event.position):
            action = snapshot_record(current, event.position, 1, 0, event.time_s).move
            if action is not None and action.piece_id != moving_piece:
                origin, moving_piece = current, action.piece_id
            check.backfilled = True
            recovered.append(event._replace(backfilled=True, board_check=check,
                image_times_s=tuple(timestamps[id(image)][1] for image in event.images)))
            current = event.position
        reference_image = event.images[-1]
    return recovered if recovered and board_signature(current) == board_signature(expected) else []


def analyze_actual(task, store, vision):
    from .analysis import read_board_cells, _occupied_squares
    from .piece_classifier import local_piece_classifier
    from .scanning import VideoScan, scan_video
    initializer = vision
    started, last_saved, previous_usage = monotonic(), monotonic(), list(task.usage)
    task.run_stats = AnalysisStats(backfill_intervals=0)
    rules = None
    folder = store.task_dir(task.id) / 'evidence'
    folder.mkdir(exist_ok=True)
    source = store.task_dir(task.id) / 'source.mp4'
    scan = None
    cache_token = _tracking_cache.set(OrderedDict())
    preprocessing_stack = ExitStack()
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
        scan = VideoScan(task, store)
        scan.update(task)
        if task.context_position is None:
            task.context_position = task.initial_position.model_copy(deep=True)
            task = store.save_analysis(task)
        task.run_stats.initialization_seconds = max(0, monotonic() - started - task.run_stats.model_seconds)
        current = task.context_position
        reference_image = None
        history = PositionHistory([task.initial_position] + [r.position_after for r in task.moves if r.position_after is not None] + [current])
        begin = task.resume_video_s if task.resume_video_s is not None else min(task.start_s, task.initial_frame_s) if task.initial_frame_s is not None else task.start_s
        tracking_calibration = task.resume_calibration if task.resume_video_s is not None and task.resume_calibration else task.calibration
        calibration_changes = []
        def motion_at(time_s):
            return next(((calibration, active, reference_s, reference_calibration)
                         for changed_s, calibration, active, reference_s, reference_calibration in reversed(calibration_changes)
                         if changed_s <= time_s + 1e-6), (tracking_calibration, False, None, None))
        frame_times = {}
        tracking_samples = deque(maxlen=2048)
        from .piece_tracking import PieceTracker
        piece_tracker = PieceTracker()
        last_evidence_s = task.start_s
        def boards():
            nonlocal task, last_saved, tracking_calibration, frame_times
            scan_begin = max(task.start_s, scan.cursor) if scan.available and not scan.index['complete'] else begin
            frames = iter_frames(source, min(begin, scan_begin), task.metadata.duration_s)
            first = True
            prepared_frames = None
            motion = None
            while True:
                decoding = monotonic()
                image = None
                try:
                    if prepared_frames is None:
                        time_s, frame = next(frames)
                        task.run_stats.decode_seconds += monotonic() - decoding
                        scan.consume(time_s, frame)
                    else:
                        time_s, image, scan_image, variants, decoding_seconds = next(prepared_frames)
                        task.run_stats.decode_seconds += decoding_seconds
                        scan.consume(time_s, None, board_image=scan_image)
                        image.setflags(write=False)
                        cache = _tracking_cache.get()
                        cache[id(image), 'prepared'] = weakref.ref(image), variants
                        _trim_tracking_cache(cache)
                except StopIteration:
                    scan.finish()
                    scan.update(task)
                    break
                scan.update(task)
                if time_s + 1e-6 < begin:
                    continue
                if first:
                    first = False
                    alignment_frame, alignment_position = frame, current
                    if task.resume_video_s is not None and task.resume_calibration is None:
                        alignment_time = task.initial_frame_s if task.initial_frame_s is not None else task.start_s
                        for _, initial_frame in iter_frames(source, alignment_time, alignment_time + .2):
                            alignment_frame, alignment_position = initial_frame, task.initial_position
                            break
                    cells = _tracking_cells(rectify_board(alignment_frame, tracking_calibration, margin=100), alignment_position)
                    occupied = {(p.square.file, p.square.rank): p.color for p in alignment_position.pieces}
                    if ((len(cells) < 90 or any(occupied.get(s) != c for s, c in cells.items()))
                            and any(color is not None for color in cells.values())
                            and not (task.resume_video_s is not None and task.resume_calibration is not None)):
                        from .calibration import refine_calibration
                        refined = refine_calibration(alignment_frame, tracking_calibration)
                        aligned = _tracking_cells(rectify_board(alignment_frame, refined, margin=100), alignment_position)
                        if len(aligned) == 90 and all(occupied.get(s) == c for s, c in aligned.items()):
                            # 只调整本次跟踪的线网映射，不覆盖用户确认的范围。
                            tracking_calibration = refined
                        elif os.getenv('VISION_FUSION') == '1':
                            from .calibration import refine_from_pieces
                            from .fusion import local_fusion_classifier
                            alignment_image = rectify_board(alignment_frame, tracking_calibration, margin=100)
                            readings = local_fusion_classifier().detector.readings(alignment_image)
                            refined = refine_from_pieces(alignment_frame, tracking_calibration, alignment_position, readings)
                            aligned = _tracking_cells(rectify_board(alignment_frame, refined, margin=100), alignment_position)
                            if (len(aligned) >= 70 and all(occupied.get(s) == c for s, c in aligned.items())
                                    and sum(aligned.get(s) == color for s, color in occupied.items()) >= len(occupied) * .75):
                                tracking_calibration = refined
                    from .board_motion import BoardMotion
                    motion = BoardMotion(tracking_calibration, active=bool(task.resume_video_s is not None
                        and task.resume_calibration is not None and task.resume_motion_active),
                        reference_calibration=task.resume_motion_reference_calibration if task.resume_video_s is not None else None)
                    if (task.resume_video_s is not None and task.resume_motion_reference_s is not None and task.resume_calibration is not None
                            and task.resume_motion_reference_s < time_s):
                        for reference_time, reference_frame in iter_frames(source, task.resume_motion_reference_s,
                                                                          task.resume_motion_reference_s + .2):
                            motion.update(reference_time, reference_frame)
                            break
                    elif alignment_frame is not frame:
                        motion.update(alignment_time, alignment_frame)
                    tracking_calibration = motion.update(time_s, frame)
                    calibration_changes.append((time_s, tracking_calibration, motion.active, motion.reference_s, motion.reference_calibration))
                task.analysis_stats.scanned_frames += 1
                task.run_stats.scanned_frames += 1
                task.processed_video_s = max(task.start_s, time_s)
                if monotonic() - last_saved > 1:
                    task.stage = ('延伸选区末手，等待最终落点' if time_s > task.end_s else
                                  f'追踪失去同步，正在恢复：{task.tracking_message}' if task.tracking_state == 'recovering' else
                                  '跟踪棋子并核对整体局面')
                    task.run_stats.elapsed_seconds = monotonic() - started
                    scan.flush()
                    task = store.save_progress(task)
                    last_saved = monotonic()
                if image is None:
                    image = rectify_board(frame, tracking_calibration, margin=100)
                image.setflags(write=False)
                observations = _tracking_cells(image, current)
                proof = piece_tracker.observe(time_s, image, current, observations,
                                              allow_reseed=inference is None)
                _tracking_cache.get()[id(image), 'identity'] = (weakref.ref(image), proof,
                    piece_tracker.identities, piece_tracker.tokens, piece_tracker.observation.cells)
                _trim_tracking_cache(_tracking_cache.get())
                tracking_samples.append((time_s, _tracking_cells(image)))
                frame_times[id(image)] = (weakref.ref(image), time_s)
                if len(frame_times) > 256:
                    frame_times = {key: value for key, value in frame_times.items() if value[0]() is not None}
                yield time_s, image
                if prepared_frames is None:
                    from functools import partial
                    from .preprocessing import parallel_batches, process_frames
                    def remaining():
                        try:
                            while True:
                                decoding = monotonic()
                                try:
                                    time_s, frame = next(frames)
                                except StopIteration:
                                    return
                                calibration = motion.update(time_s, frame)
                                if calibration is not calibration_changes[-1][1] or motion.reference_s != calibration_changes[-1][3]:
                                    calibration_changes.append((time_s, calibration, motion.active, motion.reference_s,
                                                                motion.reference_calibration))
                                yield time_s, frame, monotonic() - decoding, {
                                    (piece.square.file, piece.square.rank): piece.color for piece in current.pieces}, calibration
                        finally:
                            if hasattr(frames, 'close'):
                                frames.close()
                    process = partial(_prepare_tracking_frame, tracking_calibration=tracking_calibration,
                                      scan_calibration=scan.calibration, scan_end=task.end_s,
                                      scan_needed=scan.available and not scan.index['complete'])
                    cv2.setNumThreads(1)
                    prepare = process_frames if scan.available else parallel_batches
                    prepared_frames = preprocessing_stack.enter_context(prepare(remaining(), process))
        last_end, pending_review = task.start_s, None
        recovery_reader = None
        visual_recovery = None
        reanchor_reader = None
        inference, inference_prefix, inference_next_s, inference_signature = None, 0, 0, None
        inference_fallback_s, inference_reanchored = 0, False
        inference_tail_signature, inference_tail_hits = None, 0
        inference_tail_source, inference_empty_hits, inference_departed = None, {}, set()
        if source.is_file() and os.getenv('VISION_FUSION') == '1':
            from .tracking_recovery import VisualRecovery
            from .fusion import local_fusion_classifier
            detector = local_fusion_classifier().detector
            visual_recovery = VisualRecovery(detector.readings)
            if task.resume_video_s is not None and any(r.status == 'tentative' and r.move is None
                    and r.predicted_position is not None for r in task.moves):
                from .move_inference import MoveInference
                inference = MoveInference(current, begin, origin_uncertain=True)
                inference.residual_next_s = max((r.tracking_anchor_s or r.video_end_s) + 10
                    for r in task.moves if r.status == 'tentative' and r.move is None
                    and r.predicted_position is not None)
                inference_prefix = len(task.moves)
                inference_fallback_s = begin + 1
            def read_anchor(time_s, image, reference):
                nonlocal inference, inference_prefix, inference_next_s, inference_signature, inference_fallback_s, inference_reanchored
                nonlocal inference_tail_signature, inference_tail_hits
                nonlocal inference_tail_source, inference_empty_hits, inference_departed
                if inference is not None:
                    cells = _tracking_cells(image)
                    previous_cells = inference.samples[-1][1] if inference.samples else None
                    inference.sample(time_s, cells)
                    if time_s < inference_next_s and not (len(cells) == 90 and cells == previous_cells
                                                          and cells != inference.last_cells):
                        return None
                    inference_next_s = time_s + .4
                    inference_departed.difference_update(square for square, color in cells.items() if color is not None)
                    # 同子可能属于下一回合；只延伸选区内已直接观察到离盘的动作。
                    if time_s > task.end_s and inference_tail_source not in inference_departed:
                        from .analysis import VideoEvent
                        return VideoEvent(time_s, [], selection_complete=True)
                    if time_s <= task.end_s and time_s >= inference_fallback_s:
                        visual_recovery.request(time_s)
                        before_checks, checking = visual_recovery.check_count, monotonic()
                        candidate = visual_recovery.observe(time_s, image, reference, cells,
                            reference_image=reference_image,
                            reference_time_s=frame_times.get(id(reference_image), (None, None))[1])
                        checks = visual_recovery.check_count - before_checks
                        if checks:
                            seconds = monotonic() - checking
                            task.run_stats.reanchor_checks += checks
                            task.analysis_stats.reanchor_checks += checks
                            task.run_stats.reanchor_seconds += seconds
                            task.analysis_stats.reanchor_seconds += seconds
                        inference_fallback_s = visual_recovery.next_check_s if visual_recovery.requested else time_s + 10
                        if candidate is not None:
                            from .analysis import VideoEvent
                            inference, inference_reanchored = None, True
                            return VideoEvent(time_s, [item[1] for item in visual_recovery.evidence], position=candidate,
                                image_times_s=tuple(item[0] for item in visual_recovery.evidence), tracking_anchor=True,
                                board_check=visual_recovery.board_check)
                    proof = _identity_evidence(image)
                    identities = proof[2] if proof is not None and len(proof) > 2 else {}
                    records = inference.observe(time_s, cells, tail_only=time_s > task.end_s, identities=identities)
                    if time_s <= task.end_s:
                        inference_empty_hits = {square: inference_empty_hits.get(square, 0) + 1
                            for square, color in cells.items() if color is None}
                        inference_departed.update(square for square, hits in inference_empty_hits.items() if hits >= 2)
                        inference_tail_source = (records[-1].move.source.file, records[-1].move.source.rank) if records else None
                    position = records[-1].predicted_position if records else inference.before
                    if time_s <= task.end_s and inference.recovery_due(time_s, cells, position):
                        checking = monotonic()
                        classifier = local_fusion_classifier()
                        candidates = classifier.recognize_position([image], reference=position, actual=True)
                        if not candidates and (classifier.last_response or {}).get('draft_position'):
                            candidates = [Position.model_validate(classifier.last_response['draft_position'])]
                        candidate = next((p for observed in candidates
                            if (p := inference.constrain_recovery(observed, cells)) is not None), None)
                        seconds = monotonic() - checking
                        task.run_stats.reanchor_checks += 1
                        task.analysis_stats.reanchor_checks += 1
                        task.run_stats.reanchor_seconds += seconds
                        task.analysis_stats.reanchor_seconds += seconds
                        if candidate is not None:
                            from .analysis import VideoEvent
                            return VideoEvent(time_s, [image], position=candidate, image_times_s=(time_s,), inferred_anchor=True)
                    if time_s > task.end_s:
                        if not records:
                            from .analysis import VideoEvent
                            return VideoEvent(time_s, [], selection_complete=True)
                        expected = {(p.square.file, p.square.rank): p.color for p in position.pieces}
                        settled = len(cells) == 90 and all(expected.get(s) == c for s, c in cells.items())
                        opponent = next((after for _, after in inference._successors(position, cells, require_visible_landing=True)
                            if inference._penalty(after, cells) < inference._penalty(position, cells)), None)
                        tail_signature = board_signature(opponent or position) if settled or opponent is not None else None
                        inference_tail_hits = inference_tail_hits + 1 if tail_signature is not None and tail_signature == inference_tail_signature else 1
                        inference_tail_signature = tail_signature
                        if tail_signature is not None and inference_tail_hits >= 2:
                            from .analysis import VideoEvent
                            return VideoEvent(time_s, [], selection_complete=True)
                    signature = (tuple((r.move.piece_id, r.move.source.file, r.move.source.rank,
                        r.move.target.file, r.move.target.rank, r.status, r.action_start_s, r.action_end_s,
                        r.timing_status, r.board_check.model_dump_json() if r.board_check is not None else None)
                        for r in records), inference.evidence_ready)
                    if signature == inference_signature:
                        return None
                    inference_signature = signature
                    from .analysis import VideoEvent
                    return VideoEvent(time_s, [image], position=position, image_times_s=(time_s,),
                                      action_start_s=records[-1].video_start_s if records else task.start_s,
                                      inferred_records=records)
                if not visual_recovery.requested or time_s < visual_recovery.next_check_s:
                    return None
                before_checks = visual_recovery.check_count
                checking = monotonic()
                candidate = visual_recovery.observe(time_s, image, reference, _tracking_cells(image),
                    reference_image=reference_image,
                    reference_time_s=frame_times.get(id(reference_image), (None, None))[1])
                task.tracking_message = visual_recovery.reason or '正在等待可交叉核对的清晰画面'
                checks = visual_recovery.check_count - before_checks
                if checks > 0:
                    seconds = monotonic() - checking
                    task.run_stats.reanchor_checks += checks
                    task.analysis_stats.reanchor_checks += checks
                    task.run_stats.reanchor_seconds += seconds
                    task.analysis_stats.reanchor_seconds += seconds
                if candidate is None:
                    if not visual_recovery.requested:
                        task.tracking_state, task.tracking_since_s, task.tracking_message = 'tracking', None, None
                    elif visual_recovery.attempts >= 2:
                        from .move_inference import MoveInference
                        origin = rules.before if rules is not None else reference
                        start_s = (task.moves[-1].action_end_s or task.moves[-1].video_end_s) if task.moves else task.start_s
                        reference_time_s = frame_times.get(id(reference_image), (None, None))[1]
                        initial_samples = list(tracking_samples)
                        if reference_time_s is not None:
                            initial_samples.append((reference_time_s, _tracking_cells(reference_image)))
                        inference = MoveInference(origin, start_s, samples=[(sample_s, sample_cells)
                            for sample_s, sample_cells in initial_samples if start_s <= sample_s <= time_s])
                        inference_tail_source, inference_empty_hits, inference_departed = None, {}, set()
                        inference_prefix = len(task.moves)
                        inference_fallback_s = time_s + 1
                        return read_anchor(time_s, image, reference)
                    return None
                from .analysis import VideoEvent
                return VideoEvent(time_s, [item[1] for item in visual_recovery.evidence], position=candidate,
                    image_times_s=tuple(item[0] for item in visual_recovery.evidence),
                    tracking_anchor=not visual_recovery.incremental, board_check=visual_recovery.board_check)
            reanchor_reader = read_anchor
            previous_readings = {}
            def recover_occupied_cells(image, reference):
                nonlocal previous_readings
                from .fusion import local_fusion_classifier
                detector = local_fusion_classifier().detector
                recovered, observed = {}, {}
                occupied = {(p.square.file, p.square.rank): f'{p.color}_{p.kind}' for p in reference.pieces}
                for square, reading in detector.readings(image).items():
                    probabilities = reading['class_probabilities']
                    label = max(probabilities, key=probabilities.get)
                    if reading['confidence'] >= .85 and probabilities[label] >= .8:
                        box = reading['bbox']
                        center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        observed[square] = label, center
                        previous = previous_readings.get(square)
                        color = label.split('_', 1)[0]
                        if (occupied.get(square) == label or previous is not None and previous[0] == label
                                and max(abs(a - b) for a, b in zip(center, previous[1])) <= 20):
                            recovered[square] = color
                previous_readings = observed
                return recovered
            recovery_reader = recover_occupied_cells
        if task.rules_enabled:
            from .turns import TurnReplay
            rules = TurnReplay(task, current)
            events = actual_events(boards(), current, lambda: current, turn_boundaries=True, selected_end_s=task.end_s,
                                   confirmed_image=lambda: reference_image,
                                   turn_origin=lambda: rules.before if rules.pending is not None and len(rules.touched) == 1 else None,
                                   recovery_reader=recovery_reader, reanchor_reader=reanchor_reader,
                                   inference_active=lambda: inference is not None)
        else:
            events = actual_events(boards(), current, lambda: current, selected_end_s=task.end_s,
                                   confirmed_image=lambda: reference_image, recovery_reader=recovery_reader, reanchor_reader=reanchor_reader,
                                   inference_active=lambda: inference is not None)
        selected_last_piece = None
        events = selected_events(events, current, task.end_s, confirmed_position=lambda: current,
                                 confirmed_piece=lambda: selected_last_piece)
        recovered_events, attempted_intervals = deque(), set()
        last_verified_s = begin
        seen_cells = {}
        while True:
            detecting, previous_decode = monotonic(), task.run_stats.decode_seconds
            try:
                event = recovered_events.popleft() if recovered_events else next(events)
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
                    scan.update(task)
                    task.confirmed_prefix_length = next((r.index - 1 for r in task.moves
                        if r.status not in ('auto_verified', 'corrected')), len(task.moves))
                    task = store.save_progress(task) if len(task.moves) == previous_records else store.save_analysis(task)
                    if (len(task.moves) > previous_records and rules.pending is None and rules.unresolved is None
                            and event.last_clear_s is not None):
                        from .recovery import save_boundary_checkpoint
                        save_boundary_checkpoint(store, task, event.last_clear_s, rules.before, *motion_at(event.last_clear_s))
                continue
            task.analysis_stats.uncertain_events += int(event.uncertain)
            task.analysis_stats.stable_events += int(not event.uncertain)
            task.run_stats.uncertain_events += int(event.uncertain)
            task.run_stats.stable_events += int(not event.uncertain)
            images = event.images
            times = list(event.image_times_s) or [frame_times[id(image)][1] if id(image) in frame_times else None for image in images]
            last_evidence_s = max(last_evidence_s, *(t for t in times if t is not None), event.time_s)
            task.evidence_end_s = last_evidence_s
            ids = []
            for image in images:
                name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
                if not (folder / name).exists():
                    save_image(folder / name, image)
                ids.append(name)
            if event.inferred_anchor:
                from .move_inference import MoveInference
                after = reconcile_position(current, event.position, PositionHistory([current]))
                task.moves.append(MoveRecord(index=len(task.moves) + 1, video_start_s=event.time_s,
                    video_end_s=event.time_s, action_end_s=event.time_s, status='tentative', change_type='review',
                    position_before=current, predicted_position=after, tracking_anchor=after,
                    tracking_anchor_s=event.time_s, evidence_ids=ids, evidence_times_s=times,
                    timing_status='uncertain', reason='连续完整占位与前序推断存在差异；已显示最大概率恢复局面，棋子身份和中间操作仍需核对'))
                current, reference_image, seen_cells, selected_last_piece = after, images[-1], {}, None
                task.pending_turn = None
                residual_next_s = inference.residual_next_s
                inference = MoveInference(after, event.time_s, origin_uncertain=True)
                inference.residual_next_s = residual_next_s
                inference_tail_source, inference_empty_hits, inference_departed = None, {}, set()
                inference_prefix, inference_signature = len(task.moves), None
                inference_fallback_s = event.time_s + 1
                visual_recovery.reset()
                task.tracking_state, task.tracking_message = 'recovering', '已显示推断恢复局面，前序差异保留待核对'
                task.stage = '已恢复当前候选局面，继续分析后续走棋'
                task.recognition_video_s = max(task.recognition_video_s, event.time_s)
                task.recognition_events += 1
                task.run_stats.elapsed_seconds = monotonic() - started
                scan.update(task)
                task = store.save_analysis(task)
                from .recovery import save_boundary_checkpoint
                save_boundary_checkpoint(store, task, event.time_s, current, *motion_at(event.time_s))
                continue
            if event.inferred_records is not None:
                prior = {tuple((r.move.piece_id, r.move.source.file, r.move.source.rank,
                    r.move.target.file, r.move.target.rank)): r for r in task.moves[inference_prefix:] if r.move}
                records = []
                for predicted in event.inferred_records:
                    record = predicted.model_copy(deep=True)
                    record.index += inference_prefix
                    record.move.index = record.index
                    record.candidates = [record.move]
                    record.depends_on_steps = [i + inference_prefix for i in record.depends_on_steps]
                    if inference.origin_uncertain:
                        record.depends_on_steps.insert(0, inference_prefix)
                    key = (record.move.piece_id, record.move.source.file, record.move.source.rank,
                           record.move.target.file, record.move.target.rank)
                    old = prior.get(key)
                    witnesses = dict(zip(old.evidence_ids, old.evidence_times_s)) if old else {}
                    if old is None or old.status != record.status or record.index == inference_prefix + len(event.inferred_records):
                        witnesses.update(zip(ids, times))
                    record.evidence_ids, record.evidence_times_s = list(witnesses), list(witnesses.values())
                    records.append(record)
                if not records:
                    records.append(MoveRecord(index=inference_prefix + 1, video_start_s=inference.start_s,
                        video_end_s=event.time_s, status='needs_review', position_before=current,
                        change_type='review', evidence_ids=ids, evidence_times_s=times,
                        reason='当前证据尚未确定走法，保留原局面并继续核对', timing_status='uncertain'))
                task.moves = task.moves[:inference_prefix] + records
                task.pending_turn = None
                current = event.position
                selected_last_piece = records[-1].move.piece_id if records[-1].move else None
                reference_image, seen_cells = images[-1], {}
                task.tracking_state, task.tracking_message = 'recovering', '已展示最高匹配走法，正在用后续画面核对推断'
                task.stage = '按历史身份与合法走法继续推断；待核对步骤已标记'
                task.recognition_video_s = max(task.recognition_video_s, event.time_s)
                task.recognition_events += 1
                task.run_stats.elapsed_seconds = monotonic() - started
                scan.update(task)
                task = store.save_analysis(task)
                continue
            if event.tracking_anchor:
                if inference_reanchored:
                    if rules is not None:
                        rules = TurnReplay(task, current)
                    inference_reanchored = False
                if event.board_check is not None and _occupied_squares(current) == _occupied_squares(event.position):
                    task.moves.append(MoveRecord(index=len(task.moves) + 1,
                        video_start_s=min((t for t in times if t is not None), default=event.time_s),
                        video_end_s=event.time_s, status='needs_review', change_type='review', position_before=current,
                        observed_positions=[event.position], evidence_ids=ids, evidence_times_s=times,
                        board_check=event.board_check, reason=event.board_check.reason, timing_status='uncertain'))
                    if rules is not None:
                        rules.task = task
                        rules.sync_pending()
                    task.tracking_state, task.tracking_since_s, task.tracking_message = 'tracking', None, None
                    task.stage = '身份分类冲突已保留审核，继续按已知局面跟踪'
                    task.recognition_video_s = max(task.recognition_video_s, event.time_s)
                    task.recognition_events += 1
                    task.run_stats.elapsed_seconds = monotonic() - started
                    scan.update(task)
                    task = store.save_analysis(task)
                    continue
                task.tracking_state, task.tracking_since_s, task.tracking_message = 'tracking', None, None
                task.run_stats.reanchors += 1
                task.analysis_stats.reanchors += 1
                after = reconcile_position(current, event.position, PositionHistory([current]))
                if rules is not None:
                    rules.task = task
                    rules.reanchor(after, event.time_s, ids, times)
                    current = rules.observed
                else:
                    from .records import merge_review
                    if pending_review is None:
                        pending_review = MoveRecord(index=len(task.moves) + 1, video_start_s=last_end,
                            video_end_s=event.time_s, status='needs_review', change_type='review',
                            position_before=current, timing_status='uncertain')
                    merge_review(pending_review, event.time_s, [after], ids,
                        '中间操作未能完整还原；后续已从清晰画面重新同步，历史区间仍需人工审核', times)
                    pending_review.tracking_anchor, pending_review.tracking_anchor_s = after, event.time_s
                    pending_review.index = len(task.moves) + 1
                    task.moves.append(pending_review)
                    current, pending_review, last_end = after, None, event.time_s
                if event.board_check is not None:
                    record = task.moves[-1]
                    record.board_check = event.board_check
                    record.reason += '；' + event.board_check.reason
                history = PositionHistory([current])
                reference_image, selected_last_piece = images[-1], None
                last_verified_s = event.time_s
                seen_cells = {}
                for sample_time, sample_image in zip(times, images):
                    seen_cells.update({s: (c, sample_image, sample_time) for s, c in _tracking_cells(sample_image, current).items()})
                task.stage = '疑点已保留，已从当前清晰画面恢复后续自动识别'
                task.recognition_video_s = max(task.recognition_video_s, event.time_s)
                task.recognition_events += 1
                task.run_stats.elapsed_seconds = monotonic() - started
                scan.update(task)
                task = store.save_analysis(task)
                from .recovery import save_boundary_checkpoint
                save_boundary_checkpoint(store, task, event.time_s, current, *motion_at(event.time_s))
                continue
            previous_reference_image, previous_seen_cells = reference_image, seen_cells
            positions = [event.position] if event.position is not None else []
            if not positions and not event.review_reason and vision is not None:
                task.stage = '结合已知局面与前后画面确认变化'
                task.analysis_stats.model_requests += 1
                task.analysis_stats.images_sent += min(3, len(images))
                task.run_stats.model_requests += 1
                task.run_stats.images_sent += min(3, len(images))
                task = store.save_progress(task)
                requested = monotonic()
                try:
                    positions = vision.recognize_position(images, reference=current, actual=True)
                finally:
                    task.run_stats.model_seconds += monotonic() - requested
                    task.analysis_stats.model_requests += getattr(vision, 'last_request_count', 1) - 1
                    task.run_stats.model_requests += getattr(vision, 'last_request_count', 1) - 1
                positions = verified_observed_candidates(current, positions, images[0], images[-1])
            unique = {board_signature(p): p for p in positions}
            audit = None
            if len(unique) != 1 or event.review_reason:
                observed = next(iter(unique.values())) if len(unique) == 1 else current
                audit = check_transition(current, observed, _tracking_cells(images[0], current),
                                         _tracking_cells(images[-1], observed))
                audit.status = 'conflict'
                audit.reason = event.review_reason or '前后可见位置已核对，但尚无唯一落点候选，需回查中间操作'
            if len(unique) == 1 and not event.review_reason:
                observed = next(iter(unique.values()))
                before_cells = _tracking_cells(images[0], current)
                after_cells = _tracking_cells(images[-1], observed)
                action = snapshot_record(current, observed, 1, 0, event.time_s).move
                supplement = {}
                if action is not None:
                    expected_before = {(p.square.file, p.square.rank): p.color for p in current.pieces}
                    for square in (action.source, action.target):
                        key = square.file, square.rank
                        previous = seen_cells.get(key)
                        if key not in before_cells and previous is not None and previous[0] == expected_before.get(key):
                            before_cells[key] = previous[0]
                            supplement[id(previous[1])] = previous[1:]
                audit = event.board_check or check_transition(current, observed, before_cells, after_cells)
                for old_image, old_time in supplement.values():
                    name = hashlib.sha256(old_image.tobytes()).hexdigest()[:32] + '.jpg'
                    if not (folder / name).exists():
                        save_image(folder / name, old_image)
                    if name not in ids:
                        ids.insert(0, name)
                        times.insert(0, old_time)
                if supplement:
                    audit.reason += '；操作前遮挡位置由此前直接可见截图补证'
                if audit.status == 'conflict':
                    key = (last_verified_s, board_signature(current))
                    backfill_end_s = max([event.time_s] + [t for t in times if t is not None])
                    if (not event.backfilled and source.is_file() and key not in attempted_intervals
                            and 0 < backfill_end_s - last_verified_s <= 15
                            and needs_backfill(current, observed, audit)):
                        attempted_intervals.add(key)
                        task.analysis_stats.backfill_intervals = (task.analysis_stats.backfill_intervals or 0) + 1
                        task.run_stats.backfill_intervals += 1
                        task.stage = f'发现中间漏步，局部回查 {last_verified_s:.1f}～{backfill_end_s:.1f} 秒（本区间仅一次）'
                        task = store.save_progress(task)
                        def check_backfill(time_s):
                            nonlocal task, last_saved
                            task.analysis_stats.scanned_frames += 1
                            task.run_stats.scanned_frames += 1
                            task.analysis_stats.backfill_frames += 1
                            task.run_stats.backfill_frames += 1
                            if monotonic() - last_saved > 1:
                                task.run_stats.elapsed_seconds = monotonic() - started
                                task = store.save_progress(task)
                                last_saved = monotonic()
                        backfill_started = monotonic()
                        backfill_calibration, backfill_motion_active, backfill_reference_s, backfill_reference_calibration = motion_at(last_verified_s)
                        recovered = backfill_interval(source, last_verified_s, backfill_end_s,
                                                      backfill_calibration, current, observed, check_backfill,
                                                      motion_active=backfill_motion_active, motion_reference_s=backfill_reference_s,
                                                      motion_reference_calibration=backfill_reference_calibration)
                        backfill_seconds = monotonic() - backfill_started
                        task.analysis_stats.backfill_seconds += backfill_seconds
                        task.run_stats.backfill_seconds += backfill_seconds
                        task.stage = '局部回查结束，继续顺序识别'
                        if recovered:
                            selected = list(selected_events(iter(recovered), current, task.end_s, selected_last_piece))
                            if selected:
                                recovered_events.extend(selected)
                                continue
                    task.stage = '顺序识别，保留疑点并等待后续清晰画面'
                    event = event._replace(uncertain=True, review_reason=audit.reason)
                else:
                    reference_image = images[-1]
                    if action is not None:
                        selected_last_piece = action.piece_id
                    expected_after = {(p.square.file, p.square.rank): p.color for p in observed.pieces}
                    seen_cells = {s: item for s, item in seen_cells.items() if item[0] == expected_after.get(s)}
                    after_time = times[-1] if times else None
                    seen_cells.update({s: (color, images[-1], after_time) for s, color in after_cells.items()})
            if visual_recovery is not None and (event.review_reason or audit is not None and audit.status == 'conflict'):
                visual_recovery.request(event.time_s)
                task.tracking_state = 'recovering'
                if task.tracking_since_s is None:
                    task.tracking_since_s = event.time_s
                task.tracking_message = event.review_reason or audit.reason
            record_payloads_before = None
            if rules is not None:
                rules.task = task
                if len(unique) == 1 and not event.review_reason:
                    if task.initial_turn_known and event.time_s >= task.start_s:
                        record_payloads_before = tuple(r.model_dump_json(by_alias=True) for r in task.moves)
                    after = reconcile_position(current, next(iter(unique.values())), history)
                    rules.observe(after, event.time_s, ids, event.action_start_s, board_check=audit)
                    current = rules.tracking_position
                    if board_signature(current) == board_signature(after):
                        if visual_recovery is not None:
                            visual_recovery.reset()
                            task.tracking_state, task.tracking_since_s, task.tracking_message = 'tracking', None, None
                    else:
                        reference_image, seen_cells = previous_reference_image, previous_seen_cells
                        if visual_recovery is not None:
                            visual_recovery.request(event.time_s)
                            task.tracking_state = 'recovering'
                            task.tracking_since_s = task.tracking_since_s or event.time_s
                            task.tracking_message = '临时候选尚未通过核验，继续从可靠局面检查后续画面'
                    history.append(current)
                    if board_signature(current) == board_signature(after):
                        last_verified_s = event.time_s
                elif event.time_s >= task.start_s:
                    rules.uncertain(event.time_s, list(unique.values()), ids, event.review_reason)
                rules.sync_pending()
                rules.attach_evidence(audit, ids, times)
            elif len(unique) == 1 and not event.review_reason:
                after = reconcile_position(current, next(iter(unique.values())), history)
                if visual_recovery is not None and board_signature(after) != board_signature(current):
                    visual_recovery.reset()
                    task.tracking_state, task.tracking_since_s, task.tracking_message = 'tracking', None, None
                if event.time_s < task.start_s:
                    task.initial_position = after
                else:
                    resolved_departure = False
                    if pending_review is not None and pending_review.reason.startswith('仅有棋子离开'):
                        remaining = {p.id for p in pending_review.observed_positions[0].pieces}
                        departed = {p.id for p in current.pieces} - remaining
                        resolved_departure = departed <= {p.id for p in after.pieces}
                    resolved_candidate = (pending_review is not None and len(pending_review.observed_positions) == 1
                                          and board_signature(pending_review.observed_positions[0]) == board_signature(after))
                    if pending_review is not None and not resolved_departure and not resolved_candidate:
                        pending_review.index = len(task.moves) + 1
                        task.moves.append(pending_review)
                        last_end = pending_review.video_end_s
                if event.time_s >= task.start_s and board_signature(after) != board_signature(current):
                    record = snapshot_record(current, after, len(task.moves) + 1, last_end, event.time_s)
                    record.action_start_s, record.action_end_s = event.action_start_s, event.time_s
                    record.timing_status = 'observed' if event.action_start_s is not None else 'uncertain'
                    record.evidence_ids = ids
                    record.evidence_times_s, record.board_check = times, audit
                    task.moves.append(record)
                    last_end = event.time_s
                current = after
                history.append(after)
                last_verified_s = event.time_s
                # 遮挡期暂存的候选由后续清晰画面消除；不逐帧创建待人工确认步骤。
                pending_review = None
            elif event.time_s >= task.start_s:
                if pending_review is None:
                    pending_review = MoveRecord(index=len(task.moves) + 1, video_start_s=last_end,
                        video_end_s=event.time_s, status='needs_review', evidence_ids=ids,
                        position_before=current, change_type='review', timing_status='uncertain',
                        observed_positions=list(unique.values()), reason=event.review_reason or '结合前后画面仍无法确定完整局面')
                    pending_review.evidence_times_s = times
                    from .records import merge_review
                    merge_review(pending_review, event.time_s, [], [], pending_review.reason)
                else:
                    from .records import merge_review
                    merge_review(pending_review, event.time_s, list(unique.values()), ids, event.review_reason, times)
                pending_review.board_check = audit
            task.confirmed_prefix_length = next((r.index - 1 for r in task.moves
                if r.status not in ('auto_verified', 'corrected')), len(task.moves))
            task.usage = previous_usage + (vision.usage if vision is not None else [])
            task.run_stats.elapsed_seconds = monotonic() - started
            task.recognition_video_s = max(task.recognition_video_s, event.time_s)
            task.recognition_events += 1
            scan.update(task)
            # 正常观测可能只更新待确认回合，也可能修改较早的悔棋标注，须比较全部历史。
            pending_only = rules is not None and (len(unique) != 1 or bool(event.review_reason) or (
                record_payloads_before is not None and len(record_payloads_before) == len(task.moves)
                and all(before == record.model_dump_json(by_alias=True)
                        for before, record in zip(record_payloads_before, task.moves))))
            task = store.save_progress(task) if pending_only else store.save_analysis(task)
            if rules is None and pending_review is None and len(_tracking_cells(images[-1], current)) == 90:
                from .recovery import save_boundary_checkpoint
                checkpoint_s = times[-1] if times and times[-1] is not None else event.time_s
                save_boundary_checkpoint(store, task, checkpoint_s, current, *motion_at(checkpoint_s))
        if inference is not None:
            task.pending_turn = None
        elif rules is not None:
            rules.task = task
            rules.end(max(task.end_s, last_evidence_s))
        elif pending_review is not None:
            task.moves.append(pending_review)
        task.processed_video_s = task.end_s
        task.recognition_video_s = task.end_s
        scan.update(task)
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
        if scan is not None:
            scan.flush()
            task = scan_video(task, store)
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
        import traceback
        diagnostic = {'exception_type': type(exc).__name__, 'video_s': task.processed_video_s,
            'frames': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                       for frame in traceback.extract_tb(exc.__traceback__)]}
        (folder.parent / 'analysis-error.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf8')
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
        preprocessing_stack.close()
        if scan is not None:
            scan.flush()
        _tracking_cache.reset(cache_token)
        if vision is not None:
            vision.close()
