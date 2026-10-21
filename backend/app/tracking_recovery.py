from .chess import board_signature
from .schemas import BoardCheck, Piece, Position, Square


class VisualRecovery:
    def __init__(self, reader):
        self.reader = reader
        self.check_count = 0
        self.reset()

    def reset(self):
        self.requested = False
        self.next_check_s = 0
        self.attempts = 0
        self.previous = None
        self.samples = []
        self.burst_end_s = None
        self.evidence = []
        self.reason = ''
        self.incremental = False
        self.board_check = None
        self.incremental_samples = []
        self.identity_samples = []

    def request(self, time_s):
        if not self.requested:
            self.requested = True
            self.next_check_s = time_s

    def observe(self, time_s, image, reference, cells, *, reference_image=None, reference_time_s=None):
        if not self.requested or time_s < self.next_check_s:
            return None
        self.attempts += 1
        self.check_count += 1
        self.next_check_s = time_s + (1 if not self.samples or time_s - self.samples[-1][0] > 3 else 10)
        labels = {}
        detected = self.reader(image)
        for square, reading in detected.items():
            probabilities = reading['class_probabilities']
            if not probabilities:
                continue
            label = max(probabilities, key=probabilities.get)
            color, _ = label.split('_', 1)
            if reading['confidence'] >= .6 and probabilities[label] >= .9 and cells.get(square) == color:
                labels[square] = (label, reading['confidence'] >= .75 and probabilities[label] >= .95)
        if reference_image is not None:
            from .actual import _cells_position, _tracking_cells, snapshot_record
            from .board_evidence import check_transition, matching_piece_faces
            from .chess import apply_move
            candidate = _cells_position(reference, cells, [reference])
            action = snapshot_record(reference, candidate, 1, 0, time_s).move if candidate is not None else None
            if action is not None:
                try:
                    apply_move(reference, action)
                except ValueError:
                    candidate = None
            else:
                candidate = None
            candidate_matches = matching_piece_faces(reference_image, image, reference, candidate) if candidate is not None else set()
            matched = matching_piece_faces(reference_image, image, reference, reference)
            for previous_s, previous_image, previous_cells, previous_labels in reversed(self.samples):
                if time_s - previous_s > 1.5:
                    continue
                adjacent = matching_piece_faces(previous_image, image, reference, reference)
                for piece in reference.pieces:
                    square = piece.square.file, piece.square.rank
                    if (piece.id in adjacent and previous_cells.get(square) == piece.color
                            and previous_labels.get(square) == (f'{piece.color}_{piece.kind}', True)):
                        matched.add(piece.id)
            conflicts = {(p.square.file, p.square.rank, f'{p.color}_{p.kind}', labels[(p.square.file, p.square.rank)][0])
                for p in (candidate or reference).pieces
                if p.id in (candidate_matches if candidate is not None else matched)
                and (p.square.file, p.square.rank) in labels
                and labels[(p.square.file, p.square.rank)][1]
                and labels[(p.square.file, p.square.rank)][0] != f'{p.color}_{p.kind}'}
            self.identity_samples = [s for s in self.identity_samples if time_s - s[0] <= 3]
            self.identity_samples.append((time_s, conflicts))
            repeated = sorted(c for c in conflicts if sum(c in s[1] for s in self.identity_samples) >= 2)
            identity_reason = ('连续高置信分类与历史身份冲突，保留原身份并请人工核对：' +
                '；'.join(f'({f},{r}) 历史 {old}，当前 {new}' for f, r, old, new in repeated)) if repeated else ''
            if candidate is not None:
                old = {(p.square.file, p.square.rank): p.color for p in reference.pieces}
                prior_cells = {s: c for s, c in _tracking_cells(reference_image, reference).items() if old.get(s) == c}
                check = check_transition(reference, candidate, prior_cells, cells)
                supported = all(cells.get((p.square.file, p.square.rank)) == p.color and
                    (p.id in candidate_matches or labels.get((p.square.file, p.square.rank)) == (f'{p.color}_{p.kind}', True))
                    for p in candidate.pieces)
                if check.status != 'conflict' and supported:
                    signature = board_signature(candidate)
                    self.incremental_samples = [item for item in self.incremental_samples
                        if time_s - item[0] <= 3 and item[2] == signature]
                    self.incremental_samples.append((time_s, image, signature))
                    self.next_check_s = time_s + 1
                    if len(self.incremental_samples) >= 2:
                        if identity_reason:
                            check.status = 'partial'
                            check.reason += '；' + identity_reason
                        self.incremental, self.board_check = True, check
                        self.evidence = [(reference_time_s, reference_image)] + [(s[0], s[1]) for s in self.incremental_samples]
                        self.requested = False
                        self.reason = '起落点与连续棋子证据一致，已增量确认变化'
                        return candidate
                else:
                    self.incremental_samples = []
            else:
                self.incremental_samples = []
            for piece in reference.pieces:
                square = piece.square.file, piece.square.rank
                if piece.id in matched and cells.get(square) == piece.color:
                    labels[square] = (f'{piece.color}_{piece.kind}', True)
        sample = (time_s, image, dict(cells), labels)
        self.previous = sample
        self.samples = [item for item in self.samples if time_s - item[0] <= 3]
        if any(any(s in old_cells and c != old_cells[s] for s, c in cells.items())
               or any(s in old_labels and label[0] != old_labels[s][0] for s, label in labels.items())
               for _, _, old_cells, old_labels in self.samples):
            self.samples = []
        if not self.samples:
            self.burst_end_s = time_s + 3
        self.samples.append(sample)
        occupied_count = sum(color is not None for color in cells.values())
        if labels and len(labels) >= occupied_count * .8 and time_s < self.burst_end_s:
            self.next_check_s = time_s + 1
        if len(self.samples) < 2:
            self.reason = '正在收集连续清晰画面，尚未确认恢复局面'
            return None
        combined = {}
        for _, _, observed_cells, _ in self.samples:
            combined.update(observed_cells)
        if len(combined) != 90:
            self.reason = f'恢复画面仍有 {90 - len(combined)} 个格点未判明，等待补充可见证据'
            return None
        known = {(p.square.file, p.square.rank): f'{p.color}_{p.kind}' for p in reference.pieces}
        combined_labels = {}
        for square, color in combined.items():
            if color is None:
                if square in known and cells.get(square, 'unknown') is not None:
                    self.reason = '变化起点在最新画面仍被遮挡，等待直接可见证据'
                    return None
                if square in known and sum(s[2].get(square, 'unknown') is None for s in self.samples) < 2:
                    self.reason = '原棋子位置尚缺连续离盘证据，等待后帧核对'
                    return None
                continue
            votes = [s[3][square] for s in self.samples if square in s[3]]
            if not votes:
                self.reason = '部分棋子身份尚无可靠证据，等待清晰画面'
                return None
            label = votes[-1][0]
            if square not in labels or cells.get(square) != color:
                self.reason = '部分棋子在最新画面仍被遮挡，等待当前占位和身份的直接证据'
                return None
            strong = sum(v[1] for v in votes)
            if (strong == 0 and len(votes) < 3
                    or label != known.get(square) and strong < 2 and len(votes) < 3):
                self.reason = '部分棋子分数偏低或位置已变化，正在进行多帧交叉核对'
                return None
            combined_labels[square] = label
        from collections import Counter
        previous_counts = Counter(f'{p.color}_{p.kind}' for p in reference.pieces)
        observed_counts = Counter(combined_labels.values())
        for color in ('red', 'black'):
            if sum(n for label, n in observed_counts.items() if label.startswith(color + '_')) >= sum(p.color == color for p in reference.pieces):
                continue
            for label, count in (previous_counts - observed_counts).items():
                if not label.startswith(color + '_'):
                    continue
                stationary = sum(combined_labels.get((p.square.file, p.square.rank)) == label
                                 for p in reference.pieces if f'{p.color}_{p.kind}' == label)
                if stationary != observed_counts[label]:
                    self.reason = '同类棋子存在移动与缺失，吃子身份无法唯一对应，等待完整落点证据'
                    return None
                captured = sum(combined.get((p.square.file, p.square.rank)) not in (None, color)
                    and sum(sample[2].get((p.square.file, p.square.rank)) not in (None, color)
                            for sample in self.samples) >= 2
                    for p in reference.pieces if f'{p.color}_{p.kind}' == label)
                if captured < count:
                    self.reason = '棋子已离开原位，但尚无落点或吃子证据，等待落子完成'
                    return None
        position = Position(pieces=[Piece(id=f'{label}_{f}_{r}', color=label.split('_', 1)[0],
            kind=label.split('_', 1)[1], square=Square(file=f, rank=r))
            for (f, r), label in combined_labels.items() if combined.get((f, r)) is not None],
            side_to_move=reference.side_to_move)
        if reference_image is not None:
            from .analysis import _occupied_squares
            if _occupied_squares(position) == _occupied_squares(reference):
                changes = [(s, known[s], label) for s, label in combined_labels.items() if known.get(s) != label]
                if changes:
                    identity_reason = '占位未变化但棋子类型与历史冲突，请人工核对：' + '；'.join(
                        f'({s[0]},{s[1]}) 历史 {old}，当前 {new}' for s, old, new in changes)
        if board_signature(position) == board_signature(reference):
            if reference_image is None or not identity_reason:
                self.reset()
                return None
            position = reference.model_copy(deep=True)
        if reference_image is not None and identity_reason:
            self.board_check = BoardCheck(status='partial', visible_before=len(cells), visible_after=len(cells),
                                         reason=identity_reason)
        self.evidence = [(s[0], s[1]) for s in self.samples]
        self.requested = False
        return position
