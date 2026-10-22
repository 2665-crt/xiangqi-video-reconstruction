from .schemas import BoardCheck, Square


def matching_piece_faces(before_image, after_image, before, after):
    import cv2
    import numpy as np
    from .actual import _glyphs
    from .schemas import Position
    previous = {p.id: p for p in before.pieces}
    following = [p for p in after.pieces if p.id in previous]
    if not following:
        return set()
    old = Position(pieces=[previous[p.id] for p in following], side_to_move=before.side_to_move)
    new = Position(pieces=following, side_to_move=after.side_to_move)
    a, b = _glyphs(before_image, old), _glyphs(after_image, new)
    return {p.id for p, first, last in zip(following, a, b)
            if min(float(first[2:-2, 2:-2].std()), float(last[2:-2, 2:-2].std())) >= .04
            and float(cv2.matchTemplate(last, first[2:-2, 2:-2].copy(), cv2.TM_CCOEFF_NORMED).max()) >= .9
            and float(cv2.matchTemplate(first, last[2:-2, 2:-2].copy(), cv2.TM_CCOEFF_NORMED).max()) >= .9}


def check_transition(before, after, before_cells, after_cells):
    old = {(p.square.file, p.square.rank): p for p in before.pieces}
    new = {(p.square.file, p.square.rank): p for p in after.pieces}
    squares = [(f, r) for f in range(9) for r in range(10)]
    before_mismatches = [Square(file=s[0], rank=s[1]) for s, color in before_cells.items()
                         if (old[s].color if s in old else None) != color]
    after_mismatches = [Square(file=s[0], rank=s[1]) for s, color in after_cells.items()
                        if (new[s].color if s in new else None) != color]
    result = BoardCheck(status='verified', visible_before=len(before_cells), visible_after=len(after_cells),
                        before_mismatches=before_mismatches, after_mismatches=after_mismatches,
                        unknown_before=[Square(file=s[0], rank=s[1]) for s in squares if s not in before_cells],
                        unknown_after=[Square(file=s[0], rank=s[1]) for s in squares if s not in after_cells])
    if before_mismatches or after_mismatches:
        result.status, result.reason = 'conflict', '虚拟棋盘与原视频可见位置不一致，需回查中间走棋或核对初始局面'
        return result
    from .actual import snapshot_record
    record = snapshot_record(before, after, 1, 0, 0)
    from .chess import board_signature
    if record.move is None and board_signature(before) != board_signature(after):
        result.status, result.reason = 'conflict', '整盘观察包含多枚棋子或身份变化，需回查中间操作，不能按合法走法猜造顺序'
        return result
    if record.move is not None:
        source = record.move.source.file, record.move.source.rank
        target = record.move.target.file, record.move.target.rank
        if any(s not in before_cells or s not in after_cells for s in (source, target)):
            result.status, result.reason = 'conflict', '移动起点或最终落点缺少前后可见证据，需继续追踪或人工核对'
            return result
    if result.unknown_before or result.unknown_after:
        result.status, result.reason = 'partial', '起终点及其他可见格一致；遮挡格保留未知，未作为空格或已验证位置'
    else:
        result.reason = '操作前后90格占位及颜色均与虚拟局面一致；棋子身份沿用已确认记录'
    return result


def needs_backfill(before, after, check):
    from .actual import snapshot_record
    from .chess import board_signature
    if (check.status != 'conflict' or check.before_mismatches or check.after_mismatches
            or board_signature(before) == board_signature(after)
            or snapshot_record(before, after, 1, 0, 0).move is not None):
        return False
    old, new = {p.id: p for p in before.pieces}, {p.id: p for p in after.pieces}
    changed_squares = set()
    for identity in old.keys() | new.keys():
        previous, observed = old.get(identity), new.get(identity)
        if previous != observed:
            for piece in (previous, observed):
                if piece is not None:
                    changed_squares.add((piece.square.file, piece.square.rank))
    return not changed_squares.intersection((s.file, s.rank) for s in check.unknown_after)


def selected_events(events, initial, end_s, last_piece=None, confirmed_position=None, confirmed_piece=None):
    from .chess import board_signature
    current = initial
    try:
        for event in events:
            if confirmed_position is not None:
                reference = confirmed_position()
                if reference is not None:
                    current = reference
            if confirmed_piece is not None:
                last_piece = confirmed_piece()
            action = None
            if event.position is not None:
                from .actual import snapshot_record
                action = snapshot_record(current, event.position, 1, 0, event.time_s).move
            if event.time_s > end_s:
                if event.moving_colors:
                    continue
                if (event.review_reason and event.action_start_s is not None and event.action_start_s > end_s
                        and event.moving_piece_ids and last_piece not in event.moving_piece_ids):
                    break
                if action is not None:
                    started_in_range = event.action_start_s is not None and event.action_start_s <= end_s
                    if not started_in_range and action.piece_id != last_piece:
                        break
                elif event.position is not None and board_signature(event.position) != board_signature(current):
                    # 跨越选区的多子变化无法确定归属，只保留疑点，不扩展成新棋谱。
                    yield event._replace(uncertain=True, review_reason='选区末手与后续操作混合，需核对中间走棋和最终落点')
                    break
                elif event.review_reason:
                    yield event
                    break
                else:
                    continue
            yield event
            reference = confirmed_position() if confirmed_position is not None else event.position
            if (event.position is not None and not event.review_reason and reference is not None
                    and board_signature(reference) == board_signature(event.position)):
                current = event.position
                if action is not None:
                    last_piece = action.piece_id
    finally:
        close = getattr(events, 'close', None)
        if close is not None:
            close()
