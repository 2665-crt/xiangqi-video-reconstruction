from .chess import apply_move, board_signature
from .schemas import Move, MoveRecord, Piece, PieceEdit, Position


def checked_snapshot(before, after, index, start_s, end_s):
    from .actual import snapshot_record
    record = snapshot_record(before, after, index, start_s, end_s)
    if record.move is None:
        raise ValueError('局面变化无法对应一枚棋子的一步合法走棋，请人工选择棋子和落点')
    verified = apply_move(before, record.move)
    if board_signature(verified) != board_signature(after):
        raise ValueError('局面还包含其他棋子变化，请人工选择棋子和落点')
    record.position_after = verified
    return record


def snapshot_candidates(before, after, index):
    from .actual import snapshot_record
    single = snapshot_record(before, after, index, 0, 0).move
    if single is not None:
        return [single]
    previous = {p.id: p for p in before.pieces}
    candidates = []
    for piece in after.pieces:
        original = previous.get(piece.id)
        if original is None or original.square == piece.square or piece.color != before.side_to_move:
            continue
        occupant = next((p for p in before.pieces if p.square == piece.square and p.id != piece.id), None)
        move = Move(index=index, piece_id=piece.id, color=piece.color, source=original.square,
                    target=piece.square, captured_piece_id=occupant.id if occupant else None)
        try:
            apply_move(before, move)
        except ValueError:
            continue
        candidates.append(move)
    return candidates


def review_base(task, index):
    if index == 1:
        return task.initial_position
    for previous in reversed(task.moves[:index - 1]):
        if previous.tracking_anchor is not None:
            return previous.tracking_anchor
        if previous.status in ('auto_verified', 'corrected') and previous.position_after is not None:
            return previous.position_after
        if previous.change_type != 'review' or previous.status != 'corrected':
            break
    record = task.moves[index - 1] if index <= len(task.moves) else None
    if record is not None and record.position_before is not None:
        return record.position_before
    for previous in reversed(task.moves[:index - 1]):
        if previous.position_after is not None:
            return previous.position_after
        if len(previous.observed_positions) == 1:
            return previous.observed_positions[0]
    return task.initial_position


def split_observed_turns(before, after, index):
    previous = {p.id: p for p in before.pieces}
    changed = [p for p in after.pieces if p.id in previous and p.square != previous[p.id].square]
    if len(changed) != 2 or {p.color for p in changed} != {'red', 'black'}:
        return None
    current, path = before, []
    for _ in range(2):
        piece = next(p for p in changed if p.color == current.side_to_move)
        source = next((p for p in current.pieces if p.id == piece.id), None)
        if source is None:
            return None
        occupant = next((p for p in current.pieces if p.square == piece.square and p.id != piece.id), None)
        move = Move(index=index + len(path), piece_id=piece.id, color=piece.color, source=source.square,
                    target=piece.square, captured_piece_id=occupant.id if occupant else None)
        try:
            following = apply_move(current, move)
        except ValueError:
            return None
        path.append((move, following))
        current = following
    return path if board_signature(current) == board_signature(after) else None


class TurnReplay:
    def __init__(self, task, initial):
        self.task = task
        self.observed = initial
        self.before = initial
        self.pending = None
        self.color = None
        self.touched = set()
        self.evidence = []
        self.violation = None
        self.note = ''
        self.related = []
        self.unresolved = None
        self.unresolved_queue = []
        self.boundary = None
        self.action_start_s = self.action_end_s = None
        self.timing_uncertain = False
        self.board_check = None
        self.evidence_times = {}
        self.returned_turn = None
        self.segment_turn_unknown = bool(task.moves and task.moves[-1].tracking_anchor is not None)

    def attach_evidence(self, check, ids, times):
        self.evidence_times.update(zip(ids, times))
        if self.returned_turn is not None and self.returned_turn[1].evidence_ids == ids:
            returned = self.returned_turn[1]
            returned.board_check = check
            returned.evidence_times_s = list(times)
        if self.unresolved is not None:
            if check is None or check.status == 'conflict':
                self.unresolved.board_check = check
            self.unresolved.evidence_times_s = [self.evidence_times.get(i) for i in self.unresolved.evidence_ids]
        if self.pending is not None and (self.unresolved is None or check is not None and check.status != 'conflict'):
            if self.board_check is not None and check is not None:
                check = check.model_copy(deep=True)
                check.visible_before = self.board_check.visible_before
                check.unknown_before = self.board_check.unknown_before
                check.backfilled = check.backfilled or self.board_check.backfilled
                if check.status == 'verified' and self.board_check.status == 'partial':
                    check.status = 'partial'
                    check.reason = '操作前部分位置遮挡；已核对起终点及其他可见格，遮挡格保留未知'
            self.board_check = check
        self.sync_pending()

    def movement(self, colors, last_clear_s, departure_only=False):
        if self.pending is not None and len(colors) == 1 and self.color not in colors:
            if departure_only:
                # 离盘的对方棋子也可能是待吃棋子，须等落点确定行动方。
                if self.boundary is None:
                    self.boundary = (colors[0], last_clear_s)
            else:
                # 颜色变化可能来自遮挡；须等 observe 验证对方的真实落点后再提交本回合。
                if self.boundary is None:
                    self.boundary = (colors[0], last_clear_s)

    def observe(self, after, time_s, evidence, action_start_s=None, board_check=None):
        resolved_times = {}
        if time_s < self.task.start_s:
            self.task.initial_position = after
            self.observed = self.before = after
            return
        if (self.pending is not None and self.violation is None and not self.note
                and len(self.touched) == 1 and board_signature(after) == board_signature(self.before)
                and board_signature(self.pending) != board_signature(self.before)):
            try:
                forward = checked_snapshot(self.before, self.pending, len(self.task.moves) + 1,
                                           self._start(), self.action_end_s)
                if forward.move.captured_piece_id is not None:
                    raise ValueError('复位还包含被吃棋子恢复，需按历史局面审核')
            except ValueError:
                pass
            else:
                from .actual import snapshot_record
                forward.action_start_s, forward.action_end_s = self.action_start_s, self.action_end_s
                forward.timing_status = 'uncertain' if self.timing_uncertain else 'observed'
                forward.evidence_ids = list(self.evidence)
                forward.evidence_times_s = [self.evidence_times.get(i) for i in forward.evidence_ids]
                forward.board_check = self.board_check
                restored = self.before.model_copy(deep=True)
                returned = snapshot_record(self.pending, restored, forward.index + 1, forward.video_end_s, time_s)
                returned.evidence_ids = list(evidence)
                returned.action_start_s, returned.action_end_s = action_start_s, time_s
                returned.timing_status = 'observed' if action_start_s is not None else 'uncertain'
                self.returned_turn = (forward, returned)
                self.observed = restored
                self.color = None
                self._reset()
                return
        old = {p.id: p for p in self.observed.pieces}
        changed = [p for p in after.pieces if p.id not in old or p.square != old[p.id].square
                   or (p.color, p.kind) != (old[p.id].color, old[p.id].kind)]
        colors = {p.color for p in changed}
        if self.segment_turn_unknown and self.pending is None and len(colors) == 1:
            color = next(iter(colors))
            self.before = self.observed = self.observed.model_copy(update={'side_to_move': color})
            # 锚点的轮到方由新段第一处真实动作确定，不修改之前的棋谱。
            self.task.moves[-1].tracking_anchor = self.before.model_copy(deep=True)
            self.segment_turn_unknown = False
        if colors == {'red', 'black'} and self.task.initial_turn_known:
            continuation = self.pending is not None and self.violation is None and any(
                p.color == self.color and p.id in self.touched for p in changed)
            if continuation:
                path = split_observed_turns(self.before, after, len(self.task.moves) + 1)
                if path is not None:
                    for _, position in path:
                        self.observe(position, time_s, evidence)
                        self.timing_uncertain = True
                    return
            base = self.observed
            if self.pending is not None and self.color is not None:
                base = base.model_copy(update={'side_to_move': 'black' if self.color == 'red' else 'red'})
            path = split_observed_turns(base, after, len(self.task.moves) + 1)
            if path is not None:
                self.finish(time_s)
                self.observed = self.before = base
                for _, position in path:
                    self.observe(position, time_s, evidence)
                    self.timing_uncertain = True
                return
        if not self.task.initial_turn_known and not self.task.moves and self.pending is None and len(colors) == 1:
            color = next(iter(colors))
            self.before = self.observed = self.observed.model_copy(update={'side_to_move': color})
            self.task.initial_position = self.task.initial_position.model_copy(update={'side_to_move': color})
            self.task.initial_turn_known = True
        if not changed or colors == {self.color}:
            self.boundary = None
        if self.unresolved is not None:
            deferred_capture = False
            allowed = self.touched | {p.id for p in changed}
            explained_deletions = {}
            if (len(changed) == 1 and (self.pending is None or self.color != changed[0].color) and action_start_s is not None
                    and self.unresolved.video_end_s >= action_start_s
                    and board_check is not None and board_check.status != 'conflict'):
                from .actual import snapshot_record
                try:
                    final = checked_snapshot(self.observed, after, 1, 0, time_s).move
                except ValueError:
                    final = None
                if final is not None:
                    following = {p.id: p for p in after.pieces}
                    unknown = {(s.file, s.rank) for s in board_check.unknown_after}
                    for observation in self.unresolved.observed_positions:
                        transient = snapshot_record(self.observed, observation, 1, 0, time_s).move
                        if transient is None or transient.piece_id != final.piece_id or transient.captured_piece_id is None:
                            continue
                        victim = old.get(transient.captured_piece_id)
                        if (victim is None or following.get(victim.id) != victim
                                or (victim.square.file, victim.square.rank) in unknown):
                            continue
                        try:
                            checked_snapshot(self.observed, observation, 1, 0, time_s)
                        except ValueError:
                            # 拿子途中误配的吃子由最终合法落点和原被吃子仍清晰在场共同排除。
                            explained_deletions[id(observation)] = {victim.id}
            explained_ids = set().union(*explained_deletions.values()) if explained_deletions else set()
            independent = bool(set(self.unresolved.review_changed_piece_ids) - allowed - explained_ids)
            for observation in self.unresolved.observed_positions:
                observed = {p.id: p for p in observation.pieces}
                for identity in (set(old) | set(observed)) - allowed - explained_deletions.get(id(observation), set()):
                    original, candidate = old.get(identity), observed.get(identity)
                    if (original is None or candidate is None or original.square != candidate.square
                            or (original.color, original.kind) != (candidate.color, candidate.kind)):
                        independent = True
            continuing = (self.pending is not None and len(changed) <= 1
                          and (not colors or colors == {self.color}) and self.boundary is None)
            missing = {p.id for p in self.observed.pieces} - {
                p.id for p in self.unresolved.observed_positions[0].pieces
            } if self.unresolved.observed_positions else set()
            resolved = self.unresolved.reason.startswith('仅有棋子离开') and missing <= {p.id for p in after.pieces}
            if self.unresolved.reason.startswith('仅有棋子离开') and not resolved:
                from .actual import snapshot_record
                action = snapshot_record(self.observed, after, 1, 0, time_s).move
                resolved = action is not None and missing <= {action.captured_piece_id}
            resolved = resolved or (continuing and not independent)
            if (len(changed) == 1 and not independent
                    and self.unresolved.video_end_s >= (action_start_s if action_start_s is not None else time_s)):
                from .actual import snapshot_record
                resolved = snapshot_record(self.observed, after, 1, 0, time_s).move is not None
            if not resolved and board_check is not None and board_check.status != 'conflict':
                from .actual import snapshot_record
                action = snapshot_record(self.observed, after, 1, 0, time_s).move
                capture_base = self.observed
                if action is None and self.pending is not None:
                    capture_base = self.before
                    action = snapshot_record(capture_base, after, 1, 0, time_s).move
                if action is not None and action.captured_piece_id is not None:
                    capture_old = {p.id: p for p in capture_base.pieces}
                    victim = capture_old[action.captured_piece_id]
                    following = {p.id: p for p in after.pieces}
                    unknown = {(s.file, s.rank) for s in board_check.unknown_after}
                    swaps, explained, uncovered = [], bool(self.unresolved.observed_positions), True
                    for observation in self.unresolved.observed_positions:
                        observed = {p.id: p for p in observation.pieces}
                        differences = {identity for identity in capture_old if identity not in observed
                                       or capture_old[identity].square != observed[identity].square}
                        if not differences:
                            continue
                        if len(differences) != 2 or victim.id not in differences or set(observed) != set(capture_old):
                            explained = False
                            break
                        other = capture_old[next(iter(differences - {victim.id}))]
                        if (other.color != action.color or observed[victim.id].square != other.square
                                or observed[other.id].square != victim.square or other.id not in following
                                or following[other.id].square != other.square):
                            explained = False
                            break
                        swaps.append(other.id)
                        uncovered = uncovered and (other.square.file, other.square.rank) not in unknown
                    if explained and swaps:
                        try:
                            checked_snapshot(capture_base, after, 1, 0, time_s)
                        except ValueError:
                            pass
                        else:
                            # 先保留吃子候选，等被误配的原位置露出后再消解遮挡疑点。
                            resolved, deferred_capture = uncovered, not uncovered
                            if resolved:
                                board_check.reason += '；吃子后原位置清晰可见，已排除遮挡期间的被吃子与其他棋子颜色误配'
            if resolved:
                evidence = list(dict.fromkeys(self.unresolved.evidence_ids + evidence))
                resolved_times = dict(zip(self.unresolved.evidence_ids, self.unresolved.evidence_times_s))
            if not resolved and not deferred_capture and not (continuing and independent):
                self.finish(self.unresolved_queue[0].video_end_s if self.unresolved_queue else self.unresolved.video_end_s)
                for previous in self.unresolved_queue:
                    self._append_review(previous)
                self._append_review(self.unresolved)
            elif resolved:
                for previous in self.unresolved_queue:
                    if not (previous.reason.startswith('局部画面持续出现') and action_start_s is not None
                            and set(previous.review_changed_piece_ids) <= allowed
                            and previous.video_end_s >= action_start_s):
                        self._append_review(previous)
            if resolved or not continuing and not deferred_capture:
                self.unresolved_queue = []
                self.unresolved = None
        if self.returned_turn is not None and changed:
            forward, returned = self.returned_turn
            action = snapshot_candidates(self.observed, after, len(self.task.moves) + 3)
            if (len(action) == 1 and action[0].color == forward.move.color
                    and action[0].piece_id != forward.move.piece_id):
                forward.index, returned.index = len(self.task.moves) + 1, len(self.task.moves) + 2
                side = '红方' if forward.move.color == 'red' else '黑方'
                for record in (forward, returned):
                    record.move.index = record.index
                    record.violation = 'takeback'
                    record.reason = record.violation_note = f'{side}悔棋：原棋子已放回原处，随后改走另一枚棋子'
                    record.related_steps = [forward.index, returned.index]
                self.task.moves.extend((forward, returned))
            self.returned_turn = None
        if self.pending is not None and len(colors) == 1 and self.color not in colors:
            self.finish(self.boundary[1] if self.boundary is not None else time_s)
        self.evidence_times.update(resolved_times)
        if board_signature(after) == board_signature(self.observed):
            if resolved_times and self.pending is not None:
                self.evidence = list(dict.fromkeys(self.evidence + evidence))
            return
        if self.pending is None:
            self.before = self.observed
            self.color = next(iter(colors)) if len(colors) == 1 else None
            self.action_start_s = action_start_s
            self.timing_uncertain = action_start_s is None
        self.action_end_s = time_s
        self.pending = after
        self.touched.update(p.id for p in changed)
        self.evidence = list(dict.fromkeys(self.evidence + evidence))
        if len(colors) > 1:
            self.timing_uncertain = True
            self.violation = None
            self.note = '观察到双方棋子变化，但先后顺序无法唯一确定，请人工核对当前步骤'
        elif len(self.touched) > 1:
            self.violation = 'multiple_pieces'
            self.note = '违规：同一回合移动了多枚棋子或同时调整双方棋子，请人工决定棋子与落点'
        related = []
        states = [(0, self.task.initial_position)] + [(r.index, r.position_after) for r in self.task.moves
                                                     if r.position_after is not None]
        for index, state in states:
            if state is not None and board_signature(after) == board_signature(state) \
                    and board_signature(after) != board_signature(self.before):
                affected = [r for r in self.task.moves if r.index > index and r.move is not None]
                rollback = []
                for record in reversed(affected):
                    if not any(r.index < record.index and r.move.piece_id == record.move.piece_id
                               and r.move.source == record.move.target and r.move.target == record.move.source
                               for r in affected):
                        break
                    rollback.append(record)
                related.extend(r.index for r in affected if r not in rollback)
                for record in rollback:
                    record.candidates = [record.move]
                    record.move, record.position_after, record.status = None, None, 'needs_review'
                    record.change_type, record.violation = 'review', 'takeback'
                    record.reason = record.violation_note = '局面回退至历史状态，疑似悔棋；此处不自动计入走棋，请人工审核'
        if related:
            self.violation = 'takeback'
            self.note = '局面回退至历史状态，疑似悔棋；原走棋步骤保留，请人工核对并决定后续走法'
            self.related = sorted(set(self.related + related))
            for record in self.task.moves:
                if record.index in self.related:
                    record.violation, record.violation_note = 'takeback', self.note
                    record.related_steps = self.related
        self.observed = after

    def reanchor(self, after, time_s, evidence, times):
        self.uncertain(time_s, [after], evidence, '中间操作未能完整还原；后续已从清晰画面重新同步，历史区间仍需人工审核')
        self.attach_evidence(None, evidence, times)
        self.end(time_s)
        record = self.task.moves[-1]
        record.tracking_anchor = after.model_copy(deep=True)
        record.tracking_anchor_s = time_s
        self.observed = after.model_copy(deep=True)
        self.color = None
        self._reset()
        self.returned_turn = None
        self.segment_turn_unknown = True

    def uncertain(self, time_s, positions, evidence, reason):
        from .records import merge_review
        if (self.unresolved is not None
                and board_signature(self.unresolved.position_before) == board_signature(self.observed)):
            merge_review(self.unresolved, time_s, positions, evidence, reason)
            return
        if self.unresolved is not None:
            self.unresolved_queue.append(self.unresolved)
        self.unresolved = MoveRecord(index=len(self.task.moves) + 1, video_start_s=max(self._start(), self.action_end_s or 0),
            video_end_s=time_s, status='needs_review', observed_positions=positions,
            position_before=self.observed, evidence_ids=evidence,
            reason=reason or '结合前后画面仍无法确定走棋，请人工审核', change_type='review', timing_status='uncertain')
        merge_review(self.unresolved, time_s, [], [], self.unresolved.reason)
        if len(positions) == 1 and reason.startswith('局部画面持续出现'):
            try:
                checked_snapshot(self.before, positions[0], 1, 0, time_s)
            except ValueError:
                self.unresolved.violation = 'illegal_move'
                self.unresolved.violation_note = reason

    def _start(self):
        return self.task.moves[-1].video_end_s if self.task.moves else self.task.start_s

    def _append_review(self, record):
        record.index = len(self.task.moves) + 1
        record.position_after, record.move = None, None
        self.task.moves.append(record)

    def finish(self, time_s):
        if self.pending is None:
            return
        if board_signature(self.before) == board_signature(self.pending) and not self.violation:
            self._reset()
            return
        index = len(self.task.moves) + 1
        record = MoveRecord(index=index, video_start_s=self._start(), video_end_s=max(self._start(), time_s, self.action_end_s or 0),
            status='needs_review', position_before=self.before, observed_positions=[self.pending], evidence_ids=self.evidence,
            candidates=snapshot_candidates(self.before, self.pending, index),
            violation=self.violation, violation_note=self.note, related_steps=self.related, change_type='review')
        record.action_start_s = None if self.timing_uncertain else self.action_start_s
        record.action_end_s = None if self.timing_uncertain else self.action_end_s
        record.timing_status = 'uncertain' if self.timing_uncertain else 'observed'
        record.board_check = self.board_check
        record.evidence_times_s = [self.evidence_times.get(i) for i in record.evidence_ids]
        if self.violation:
            record.reason = self.note
            self._append_review(record)
        elif self.note:
            record.reason = self.note
            self._append_review(record)
        else:
            try:
                checked = checked_snapshot(self.before, self.pending, index, record.video_start_s, record.video_end_s)
            except ValueError as exc:
                record.violation = 'turn_mismatch' if self.color != self.before.side_to_move else 'illegal_move'
                record.reason = record.violation_note = f'违规：{exc}；请人工决定棋子与落点'
                self._append_review(record)
            else:
                checked.evidence_ids = self.evidence
                checked.action_start_s, checked.action_end_s = record.action_start_s, record.action_end_s
                checked.timing_status = record.timing_status
                checked.board_check, checked.evidence_times_s = record.board_check, record.evidence_times_s
                checked.reason = '已合并本回合移动并通过象棋规则校验'
                self.task.moves.append(checked)
        self._reset()

    def _reset(self):
        if self.color is not None:
            self.observed = self.observed.model_copy(update={'side_to_move': 'black' if self.color == 'red' else 'red'})
        self.before = self.observed
        self.pending, self.color = None, None
        self.touched, self.evidence, self.related = set(), [], []
        self.violation, self.note = None, ''
        self.boundary = None
        self.action_start_s = self.action_end_s = None
        self.timing_uncertain = False
        self.task.pending_turn = None
        self.board_check, self.evidence_times = None, {}

    def end(self, time_s):
        if self.boundary is not None and self.pending is not None:
            self.finish(self.boundary[1])
        if (self.unresolved is not None and self.pending is not None and self.action_end_s is not None
                and self.action_end_s > self.unresolved.video_end_s):
            self.finish(self.action_end_s)
        if self.unresolved is not None:
            if self.pending is not None:
                self.unresolved.position_before = self.before
                self.unresolved.video_start_s = self._start()
                self.unresolved.candidates = snapshot_candidates(self.before, self.pending, self.unresolved.index)
                self.unresolved.board_check = self.board_check
                self.unresolved.action_start_s = self.action_start_s
                self.unresolved.observed_positions = [self.pending] + self.unresolved.observed_positions
                self.unresolved.evidence_ids = list(dict.fromkeys(self.evidence + self.unresolved.evidence_ids))
                self.unresolved.evidence_times_s = [self.evidence_times.get(i) for i in self.unresolved.evidence_ids]
                self.unresolved.violation = self.violation or self.unresolved.violation
                self.unresolved.violation_note = self.note or self.unresolved.violation_note
                self.unresolved.related_steps = self.related
                self._reset()
            self.unresolved.video_end_s = time_s
            for previous in self.unresolved_queue:
                self._append_review(previous)
            self.unresolved_queue = []
            self._append_review(self.unresolved)
            self.unresolved = None
        else:
            self.finish(time_s)
        self.task.confirmed_prefix_length = next((r.index - 1 for r in self.task.moves
            if r.status not in ('auto_verified', 'corrected')), len(self.task.moves))
        self.task.pending_turn = None

    def sync_pending(self):
        if self.pending is None and self.unresolved is None:
            self.task.pending_turn = None
            return
        self.task.pending_turn = MoveRecord(index=len(self.task.moves) + 1, video_start_s=self._start(),
            video_end_s=max(self._start(), self.task.processed_video_s), status='needs_review',
            position_before=self.before,
            observed_positions=[self.pending] if self.pending is not None else self.unresolved.observed_positions,
            evidence_ids=self.evidence if self.pending is not None else self.unresolved.evidence_ids,
            violation=self.violation, violation_note=self.note, related_steps=self.related,
            change_type='review', reason='本回合尚未确认，等待对方开始移动或视频结束',
            action_start_s=None if self.timing_uncertain else self.action_start_s,
            action_end_s=None if self.timing_uncertain else self.action_end_s,
            timing_status='observed' if self.pending is not None and not self.timing_uncertain else 'uncertain')
        self.task.pending_turn.board_check = self.board_check
        issues = self.unresolved_queue + ([self.unresolved] if self.unresolved is not None else [])
        for issue in issues:
            if issue is not self.unresolved or self.pending is not None:
                self.task.pending_turn.observed_positions += issue.observed_positions
            self.task.pending_turn.evidence_ids = list(dict.fromkeys(self.task.pending_turn.evidence_ids + issue.evidence_ids))
            self.task.pending_turn.review_changed_piece_ids = sorted(set(self.task.pending_turn.review_changed_piece_ids + issue.review_changed_piece_ids))
            self.task.pending_turn.reason += '；' + issue.reason
        if self.pending is not None and issues:
            self.task.pending_turn.candidates = snapshot_candidates(self.before, self.pending, self.task.pending_turn.index)
        self.task.pending_turn.evidence_times_s = [self.evidence_times.get(i) for i in self.task.pending_turn.evidence_ids]

    def interrupt(self):
        self.sync_pending()
        for previous in self.unresolved_queue:
            self._append_review(previous)
        self.unresolved_queue = []
        if self.task.pending_turn is not None:
            self.task.pending_turn.reason = '分析中断，本回合尚未确认，请人工审核'
            self._append_review(self.task.pending_turn)
            self.task.pending_turn = None


def revalidate_turns(task):
    corrected = next((r.index for r in task.moves if r.manual_override), None)
    if corrected is not None:
        return replay_corrections(task, corrected)
    task = task.model_copy(deep=True)
    prefix, continuous = 0, True
    if task.initial_position is None:
        task.confirmed_prefix_length = 0
        return task
    for index, record in enumerate(task.moves, 1):
        record.index = index
        before = review_base(task, index)
        record.position_before = before
        if not record.candidates and len(record.observed_positions) == 1:
            record.candidates = snapshot_candidates(before, record.observed_positions[0], index)
        if record.status == 'needs_review':
            record.position_after = None
            continuous = False
            continue
        if record.status == 'corrected' and record.move is None and record.change_type == 'review':
            record.position_after = before.model_copy(deep=True)
            if continuous:
                prefix = index
            continue
        previous = task.moves[index - 2] if index > 1 else None
        if (record.status == 'auto_verified' and record.violation == 'takeback' and record.move is not None
                and previous is not None and previous.status == 'auto_verified' and previous.violation == 'takeback'
                and previous.move is not None and previous.move.captured_piece_id is None
                and previous.index in record.related_steps and record.index in previous.related_steps
                and record.move.piece_id == previous.move.piece_id
                and record.move.color == previous.move.color
                and record.move.source == previous.move.target and record.move.target == previous.move.source
                and record.position_after is not None
                and board_signature(record.position_after) == board_signature(previous.position_before)):
            record.position_after = previous.position_before.model_copy(deep=True)
            if continuous:
                prefix = index
            continue
        try:
            if record.move is not None:
                move = record.move.model_copy(update={'index': index})
                after = apply_move(before, move)
                if record.status != 'corrected' and record.observed_positions and not any(
                    board_signature(after) == board_signature(p) for p in record.observed_positions):
                    raise ValueError('走法与视频局面不一致')
                record.move, record.position_after = move, after
            elif len(record.observed_positions) == 1 and not record.violation:
                checked = checked_snapshot(before, record.observed_positions[0], index,
                                           record.video_start_s, record.video_end_s)
                record.move, record.position_after, record.change_type = checked.move, checked.position_after, 'move'
            else:
                raise ValueError('走法尚未确定，请人工审核')
        except ValueError as exc:
            record.status, record.position_after = 'needs_review', None
            record.reason = str(exc)
            continuous = False
            continue
        if record.status != 'corrected':
            record.status = 'auto_verified'
        if continuous:
            prefix = index
    task.confirmed_prefix_length = prefix
    task.status = 'needs_review' if any(r.status not in ('auto_verified', 'corrected') for r in task.moves) else 'ready'
    task.stage = '仍有未确定的视频区间，完整棋谱尚未确认' if task.status == 'needs_review' else '合法棋谱已保存，可回放'
    return task


def unchecked_move(before, action):
    piece = next((p for p in before.pieces if p.id == action.piece_id), None)
    if piece is None:
        return before.model_copy(deep=True), '原走法的棋子已不存在，请手动校正此步'
    warnings = []
    if piece.square != action.source:
        warnings.append('原走法起点与已校正位置不同，已从当前保存位置执行原落点')
    actual = action.model_copy(update={'source': piece.square})
    try:
        apply_move(before, actual)
    except ValueError as exc:
        warnings.append(str(exc))
    after = Position(pieces=[p.model_copy(update={'square': action.target}) if p.id == piece.id else p.model_copy(deep=True)
        for p in before.pieces if p.id == piece.id or p.square != action.target],
        side_to_move='black' if piece.color == 'red' else 'red')
    return after, '；'.join(warnings)


def position_warning(position):
    warnings = []
    kings = {}
    for color in ('red', 'black'):
        found = [p for p in position.pieces if p.color == color and p.kind == 'king']
        if len(found) != 1:
            warnings.append(f'{"红方" if color == "red" else "黑方"}将帅数量不是一枚')
        for king in found:
            if not (3 <= king.square.file <= 5 and (7 <= king.square.rank <= 9 if color == 'red' else 0 <= king.square.rank <= 2)):
                warnings.append('将帅不在九宫内')
        if len(found) == 1:
            kings[color] = found[0]
    if len(kings) == 2:
        red, black = kings['red'].square, kings['black'].square
        if red.file == black.file and not any(p.square.file == red.file and
                min(red.rank, black.rank) < p.square.rank < max(red.rank, black.rank) for p in position.pieces):
            warnings.append('将帅照面')
    return '；'.join(dict.fromkeys(warnings))


def position_edits(old_before, old_after):
    previous = {p.id: p for p in old_before.pieces}
    following = {p.id: p for p in old_after.pieces}
    edits = [PieceEdit(id=identity, removed=True) for identity in previous.keys() - following.keys()]
    for identity, piece in following.items():
        old = previous.get(identity)
        changes = {field: getattr(piece, field) for field in ('color', 'kind', 'square')
                   if old is None or getattr(old, field) != getattr(piece, field)}
        if changes:
            edits.append(PieceEdit(id=identity, **changes))
    return edits


def apply_position_edits(current, edits, side_to_move):
    pieces = {p.id: p.model_copy(deep=True) for p in current.pieces}
    warnings = []
    placed = set()
    for edit in edits:
        if edit.removed:
            pieces.pop(edit.id, None)
            continue
        changes = edit.model_dump(exclude_none=True, exclude={'id', 'removed'})
        if edit.square is not None:
            changes['square'] = edit.square
        original = pieces.get(edit.id)
        if original is None:
            if edit.color is None or edit.kind is None or edit.square is None:
                warnings.append('待校正棋子已不存在，已保留编辑记录，请手动核对')
                continue
            pieces[edit.id] = Piece(id=edit.id, **changes)
        else:
            pieces[edit.id] = original.model_copy(update=changes)
        if edit.square is not None:
            placed.add(edit.id)
    # 先同时应用落点，交换位置时不能把稍后仍要移动的棋子当作被吃子。
    targets = { (pieces[identity].square.file, pieces[identity].square.rank) for identity in placed }
    pieces = {identity: p for identity, p in pieces.items() if identity in placed or (p.square.file, p.square.rank) not in targets}
    return Position(pieces=list(pieces.values()), side_to_move=side_to_move or current.side_to_move), '；'.join(warnings)


def replay_corrections(task, start_index, preserve_unknown=False):
    from .actual import snapshot_record
    task = task.model_copy(deep=True)
    current = review_base(task, start_index)
    for record in task.moves[start_index - 1:]:
        old_before, old_after = record.position_before, record.position_after
        if record.tracking_anchor is not None and record.status in ('needs_review', 'tentative'):
            record.position_before = current.model_copy(deep=True)
            record.position_after = None
            current = record.tracking_anchor.model_copy(deep=True)
            continue
        if preserve_unknown and record.status in ('needs_review', 'tentative'):
            record.position_before = current.model_copy(deep=True)
            record.position_after = None
            continue
        if old_before is None or board_signature(current) != board_signature(old_before):
            record.board_check = None
        warning = ''
        if record.status == 'corrected' and record.change_type == 'review' and record.move is None:
            after = current.model_copy(deep=True)
        elif record.position_edits is not None or (record.move is None and record.status in ('auto_verified', 'corrected') and old_after is not None):
            if record.position_edits is None:
                record.position_edits = position_edits(old_before or current, old_after)
                if old_after.side_to_move != (old_before or current).side_to_move:
                    record.position_side_to_move = old_after.side_to_move
            if record.move is not None:
                _, warning = unchecked_move(current, record.move)
            after, edit_warning = apply_position_edits(current, record.position_edits, record.position_side_to_move)
            warning = '；'.join(filter(None, [warning, edit_warning]))
        elif record.move is not None:
            after, warning = unchecked_move(current, record.move)
            if record.status in ('needs_review', 'tentative'):
                record.status = 'auto_verified'
                record.reason = '按保留的原走法承接人工校正，规则问题仅提示'
        elif record.status == 'tentative' and len(record.observed_positions) == 1:
            # 已有明确的后续走法才承接；不为多子变化猜测顺序。
            inferred = snapshot_record(old_before or current, record.observed_positions[0], record.index,
                                       record.video_start_s, record.video_end_s)
            if inferred.move is None:
                record.position_before = current.model_copy(deep=True)
                continue
            record.move, record.status, record.change_type = inferred.move, 'auto_verified', 'move'
            after, warning = unchecked_move(current, record.move)
        else:
            record.position_before = current.model_copy(deep=True)
            continue
        record.position_before, record.position_after = current.model_copy(deep=True), after
        warning = '；'.join(filter(None, [warning, position_warning(after)]))
        if warning:
            record.violation = 'turn_mismatch' if record.move and record.move.color != current.side_to_move else 'illegal_move'
            record.violation_note = warning + '；仅提示，可继续回放或手动校正'
        elif record.violation != 'takeback':
            record.violation, record.violation_note = None, ''
        current = record.tracking_anchor.model_copy(deep=True) if record.tracking_anchor is not None else after
    task.confirmed_prefix_length = next((r.index - 1 for r in task.moves
        if r.status not in ('auto_verified', 'corrected')), len(task.moves))
    task.status = 'needs_review' if any(r.status not in ('auto_verified', 'corrected') for r in task.moves) else 'ready'
    task.stage = '校正已保存，后续按原走法承接；规则问题仅提示'
    return task


def renumber_records(task, index_map):
    for index, record in enumerate(task.moves, 1):
        record.index = index
        if record.move is not None:
            record.move.index = index
        for candidate in record.candidates:
            candidate.index = index
        record.related_steps = [index_map[i] for i in record.related_steps if i in index_map]


def insert_turn(task, index, action, time_s, action_start_s=None):
    if task.initial_position is None or not 1 <= index <= len(task.moves) + 1:
        raise ValueError('请先建立起始局面并选择插入位置')
    before = review_base(task, index)
    previous = task.moves[index - 2] if index > 1 else None
    following = task.moves[index - 1] if index <= len(task.moves) else None
    lower = (previous.action_end_s if previous.action_end_s is not None else previous.video_end_s) if previous else task.start_s
    upper = (following.action_start_s if following.action_start_s is not None else following.video_end_s) if following else task.metadata.duration_s
    start = action_start_s if action_start_s is not None else time_s
    if not lower <= start <= time_s <= upper:
        raise ValueError('补棋动作及落子时间必须位于前一步落定与后一步起动之间')
    if not any(p.id == action.piece_id and p.square == action.source for p in before.pieces):
        raise ValueError('补棋起点与前一步保存局面不一致，请重新选择棋子')
    task = task.model_copy(deep=True)
    index_map = {r.index: r.index + (r.index >= index) for r in task.moves}
    from .actual import snapshot_record
    action = action.model_copy(update={'index': index})
    after, _ = unchecked_move(before, action)
    record = snapshot_record(before, after, index, lower, time_s)
    record.move, record.status, record.manual_override = action, 'corrected', True
    record.reason = '人工插入遗漏走棋'
    record.action_start_s, record.action_end_s = start, time_s
    record.timing_status = 'observed' if action_start_s is not None else 'uncertain'
    task.moves.insert(index - 1, record)
    renumber_records(task, index_map)
    task.recheck = None
    return replay_corrections(task, index)


def correct_turn(task, index, actions, time_s=None, position=None):
    if task.initial_position is None or not 1 <= index <= len(task.moves) + 1:
        raise ValueError('请先建立起始局面并选择有效步骤')
    analyzing = task.status == 'analyzing'
    task = task.model_copy(deep=True)
    before = review_base(task, index)
    if before is None:
        raise ValueError('前一处局面尚未确定，请先审核')
    original = task.moves[index - 1] if index <= len(task.moves) else None
    preceding = (task.moves[index - 2].action_end_s or task.moves[index - 2].video_end_s) if index > 1 else task.start_s
    following = (task.moves[index].action_end_s or task.moves[index].video_end_s) if index < len(task.moves) else max(
        task.end_s, task.evidence_end_s or 0, original.video_end_s if original is not None else 0,
        original.action_end_s or 0 if original is not None else 0)
    following = min(task.metadata.duration_s, following)
    time_s = time_s if time_s is not None else (original.action_end_s or original.video_end_s) if original else task.processed_video_s
    if not preceding <= time_s <= following:
        raise ValueError('落子时间必须位于前后步骤的时间范围内')
    if len(actions) > 1:
        raise ValueError('一次审核只确认一步走棋，请选择一枚棋子及其最终落点')
    from .actual import snapshot_record
    if position is not None:
        record = snapshot_record(before, position, index, preceding, time_s)
        record.position_edits = position_edits(before, position)
        if position.side_to_move != before.side_to_move:
            record.position_side_to_move = position.side_to_move
        record.reason = '人工校正当前局面'
    elif actions:
        action = actions[0].model_copy(update={'index': index})
        if not any(p.id == action.piece_id for p in before.pieces):
            raise ValueError('所选棋子已不存在，请重新选择或在局面校正中添加')
        after, _ = unchecked_move(before, action)
        record = snapshot_record(before, after, index, preceding, time_s)
        record.move, record.change_type = action, 'move'
        record.reason = '人工替换当前走法'
    else:
        record = MoveRecord(index=index, video_start_s=preceding, video_end_s=time_s, status='corrected',
                            position_before=before, position_after=before.model_copy(deep=True),
                            change_type='review', reason='人工确认此处不计入走棋')
    record.status = 'corrected'
    record.manual_override = True
    if original is not None:
        record.evidence_ids, record.observed_positions = original.evidence_ids, original.observed_positions
        record.tracking_anchor = original.tracking_anchor
        record.tracking_anchor_s = original.tracking_anchor_s
        if record.tracking_anchor is not None and position is not None and time_s == record.tracking_anchor_s:
            record.tracking_anchor = position.model_copy(deep=True)
        record.evidence_times_s = original.evidence_times_s
        record.violation, record.violation_note, record.related_steps = original.violation, original.violation_note, original.related_steps
        if time_s == (original.action_end_s if original.action_end_s is not None else original.video_end_s):
            record.action_start_s, record.action_end_s = original.action_start_s, original.action_end_s
            record.timing_status = original.timing_status
        else:
            record.action_end_s, record.timing_status = time_s, 'uncertain'
    task.moves[index - 1:index] = [record]
    task = replay_corrections(task, index)
    if analyzing:
        task.status, task.stage = 'analyzing', '本步审核已保存，视频分析继续进行'
    return task
