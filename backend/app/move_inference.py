from dataclasses import dataclass
from collections import deque
from uuid import uuid4

from .chess import apply_move, board_signature, legal_moves
from .schemas import MoveRecord


@dataclass
class Hypothesis:
    position: object
    records: tuple = ()
    score: float = 0
    hits: tuple = ()


class MoveInference:
    """有限路径搜索；评分表示观测匹配程度，不表示经验准确率。"""

    def __init__(self, before, start_s, width=5, max_gap_plies=3, gap_width=12, *, origin_uncertain=False, samples=()):
        self.before, self.start_s, self.width = before, start_s, width
        self.group = uuid4().hex
        self.beam = [Hypothesis(before)]
        self.transitions = {}
        self.max_gap_plies, self.gap_width = max_gap_plies, gap_width
        self.evidence_ready = ()
        self.root_score = 0
        self.committed = ()
        self.gap_signature, self.gap_hits = None, 0
        self.origin_uncertain = origin_uncertain
        self.residual_signature, self.residual_since, self.residual_next_s = None, 0, 0
        self.samples = deque(maxlen=2048)
        self.last_cells = None
        self.identities = {}
        for time_s, cells in sorted(samples, key=lambda item: item[0]):
            self.sample(time_s, cells)

    def sample(self, time_s, cells):
        if not self.samples or time_s > self.samples[-1][0]:
            self.samples.append((time_s, dict(cells)))

    def recovery_due(self, time_s, cells, position):
        if len(cells) != 90 or not self._penalty(position, cells):
            self.residual_signature = None
            return False
        signature = tuple(sorted(cells.items())), board_signature(position)
        if signature != self.residual_signature:
            self.residual_signature, self.residual_since = signature, time_s
            return False
        if time_s - self.residual_since < 2 or time_s < self.residual_next_s:
            return False
        self.residual_next_s = time_s + 10
        return True

    def constrain_recovery(self, position, cells):
        pieces = [p for p in position.pieces if cells.get((p.square.file, p.square.rank)) == p.color]
        expected = {(p.square.file, p.square.rank): p.color for p in pieces}
        if (len(cells) != 90 or expected != {s: c for s, c in cells.items() if c is not None}
                or len(pieces) != len(expected)
                or any(sum(p.kind == 'king' and p.color == color for p in pieces) != 1 for color in ('red', 'black'))):
            return None
        return position.model_copy(update={'pieces': pieces})

    def _successors(self, before, cells, *, require_visible_landing=False):
        signature = (board_signature(before), before.side_to_move)
        if signature not in self.transitions:
            if len(self.transitions) >= 128:
                self.transitions.pop(next(iter(self.transitions)))
            self.transitions[signature] = [(move, apply_move(before, move)) for move in legal_moves(before)]
        for move, after in self.transitions[signature]:
            source = move.source.file, move.source.rank
            target = move.target.file, move.target.rank
            if target in self.identities and self.identities[target] != move.piece_id:
                continue
            if source in cells and cells[source] is not None:
                continue
            if target in cells and cells[target] is None:
                continue
            if require_visible_landing and (source not in cells or cells[source] is not None
                                           or cells.get(target) != move.color):
                continue
            if source in cells and cells[source] != move.color or cells.get(target) == move.color:
                yield move, after

    def _penalty(self, position, cells):
        expected = {(p.square.file, p.square.rank): p.color for p in position.pieces}
        identities = {(p.square.file, p.square.rank): p.id for p in position.pieces}
        return (sum(1 if color is None else 4 for s, color in cells.items() if expected.get(s) != color)
                + sum(4 for square, identity in self.identities.items() if identities.get(square) != identity))

    def _record(self, move, before, after, records, time_s, gap=False, previous=None):
        index = len(self.committed) + len(records) + 1
        move = move.model_copy(update={'index': index})
        lower = (records[-1].action_end_s if records[-1].action_end_s is not None
                 else records[-1].video_start_s) if records else self.start_s
        source = move.source.file, move.source.rank
        target = move.target.file, move.target.rank
        occupied_s, start_s, end_s = None, None, None
        for sample_s, cells in self.samples:
            if sample_s < lower or sample_s > time_s:
                continue
            if cells.get(source) == move.color:
                occupied_s = sample_s
                start_s = None
            elif occupied_s is not None:
                start_s = occupied_s
                if source in cells and cells[source] is None and cells.get(target) == move.color:
                    end_s = sample_s
                    break
        if previous is not None and previous.action_start_s is not None:
            start_s = previous.action_start_s
            if previous.move.target == move.target and previous.timing_status == 'observed':
                end_s = previous.action_end_s
        observed = start_s is not None and end_s is not None
        record = MoveRecord(index=index, move=move, status='tentative', position_before=before,
            predicted_position=after, video_start_s=lower,
            video_end_s=end_s if observed else time_s, action_start_s=start_s if observed else None,
            action_end_s=end_s if observed else None if gap else time_s,
            timing_status='observed' if observed else 'uncertain' if gap else 'estimated', candidates=[move],
            inference_group=self.group, reason=('区间跨步推断，中间落点未直接观察，需核对' if gap else
                '依据历史身份、合法一步与可见落点推断，待后续画面核对'))
        return record

    def _comparison(self, record, cutoff):
        from .board_evidence import check_transition
        before_cells = {}
        if record.action_start_s is not None:
            for sample_s, observed in self.samples:
                if record.video_start_s <= sample_s <= record.action_start_s:
                    before_cells = dict(observed)
            expected = {(p.square.file, p.square.rank): p.color for p in record.position_before.pieces}
            for square in (record.move.source, record.move.target):
                key = square.file, square.rank
                if key in before_cells:
                    continue
                for sample_s, observed in reversed(self.samples):
                    if (record.video_start_s <= sample_s <= record.action_start_s and key in observed
                            and observed[key] == expected.get(key)):
                        before_cells[key] = observed[key]
                        break
        end = record.action_end_s if record.action_end_s is not None else record.video_end_s
        candidates = [observed for sample_s, observed in self.samples if end <= sample_s <= cutoff]
        if not candidates:
            candidates = [self.samples[-1][1]]
        best = None
        for observed in candidates:
            check = check_transition(record.position_before, record.predicted_position, before_cells, observed)
            quality = (check.status != 'conflict', -len(check.before_mismatches) - len(check.after_mismatches),
                       check.visible_after)
            if best is None or quality > best[0]:
                best = quality, check
        return best[1]

    def _bridge(self, time_s, cells):
        frontier = [Hypothesis(self.before)]
        for _ in range(self.max_gap_plies):
            expanded = []
            for parent in frontier:
                board = {(p.square.file, p.square.rank): p.color for p in parent.position.pieces}
                for move in legal_moves(parent.position):
                    source = move.source.file, move.source.rank
                    if cells.get(source) == board[source]:
                        continue
                    after = apply_move(parent.position, move)
                    record = self._record(move, parent.position, after, parent.records, time_s, gap=True)
                    if (any(r.move.piece_id == move.piece_id and r.move.source == move.target
                            for r in parent.records) and record.timing_status != 'observed'):
                        continue
                    cost = self._penalty(after, cells)
                    candidate = Hypothesis(after, parent.records + (record,), -cost, parent.hits + (-1,))
                    if cost == 0:
                        yield candidate
                    expanded.append(candidate)
            frontier = sorted(expanded, key=lambda p: -p.score)[:self.gap_width]
            if not frontier:
                break

    def observe(self, time_s, cells, *, tail_only=False, identities=None):
        self.identities = dict(identities or {})
        self.sample(time_s, cells)
        self.last_cells = dict(cells)
        proposals = []
        parents = self.beam + [Hypothesis(self.before, score=self.root_score)]
        if self.origin_uncertain:
            opposite = self.before.model_copy(update={'side_to_move': 'black' if self.before.side_to_move == 'red' else 'red'})
            parents.append(Hypothesis(opposite, score=self.root_score))
        for parent in parents:
            proposals.append(parent)
            origins = [] if tail_only else [(parent.position, parent.records, parent.hits)]
            if parent.records:
                last = parent.records[-1]
                origins.append((last.position_before, parent.records[:-1], parent.hits[:-1]))
            for before, records, hits in origins:
                for move, after in self._successors(before, cells):
                    if tail_only and move.piece_id != last.move.piece_id:
                        continue
                    previous = parent.records[-1] if (parent.records and parent.records[-1].move.piece_id == move.piece_id
                        and parent.records[-1].move.source == move.source) else None
                    if records and previous is None and cells.get((move.target.file, move.target.rank)) != move.color:
                        continue
                    record = self._record(move, before, after, records, time_s, previous=previous)
                    returning = [r for r in self.committed + records
                                 if r.move.piece_id == move.piece_id and r.move.source == move.target]
                    if returning and (record.timing_status != 'observed'
                            or not any(returning[-1].video_start_s <= sample_s <= record.video_start_s
                                and observed.get((returning[-1].move.source.file, returning[-1].move.source.rank), 'unknown') is None
                                and observed.get((returning[-1].move.target.file, returning[-1].move.target.rank)) == move.color
                                for sample_s, observed in self.samples)):
                        continue
                    proposals.append(Hypothesis(after, records + (record,), parent.score - .1, hits + (0,)))
        observation = tuple(sorted(cells.items()))
        self.gap_hits = self.gap_hits + 1 if observation == self.gap_signature else 1
        self.gap_signature = observation
        if not tail_only and len(cells) == 90 and self.gap_hits == 2:
            if min(self._penalty(p.position, cells) for p in proposals) > 0:
                proposals.extend(self._bridge(time_s, cells))
        ranked = {}
        for proposal in proposals:
            penalty = self._penalty(proposal.position, cells)
            score = proposal.score * .35 - penalty + 4 * sum(r.timing_status == 'observed' for r in proposal.records)
            hits = proposal.hits
            if hits and hits[-1] >= 0 and not penalty and (len(cells) == 90
                    or len(cells) >= 70 and proposal.records[-1].timing_status == 'observed'):
                hits = hits[:-1] + (hits[-1] + 1,)
            candidate = Hypothesis(proposal.position, proposal.records, score, hits)
            key = tuple((r.move.piece_id, r.move.source.file, r.move.source.rank,
                         r.move.target.file, r.move.target.rank) for r in proposal.records)
            previous = ranked.get(key)
            if previous is None or (candidate.score, sum(candidate.hits)) > (previous.score, sum(previous.hits)):
                ranked[key] = candidate
        ordered = sorted(ranked.values(), key=lambda p: (-p.score, len(p.records)))
        self.root_score = ranked[()].score
        self.beam = ordered[:self.width]
        best = self.beam[0]
        timed = []
        for offset, record in enumerate(best.records):
            if record.timing_status != 'observed' and self.samples[0][0] <= record.video_end_s:
                following = best.records[offset + 1] if offset + 1 < len(best.records) else None
                cutoff = (following.action_start_s if following.action_start_s is not None
                          else following.video_end_s) if following is not None else time_s
                refreshed = self._record(record.move, record.position_before, record.predicted_position,
                    timed, cutoff, gap=record.reason.startswith('区间跨步推断'), previous=record)
                if refreshed.timing_status == 'observed':
                    record = record.model_copy(update={name: getattr(refreshed, name) for name in
                        ('video_start_s', 'video_end_s', 'action_start_s', 'action_end_s', 'timing_status')})
            timed.append(record)
        for offset, record in enumerate(timed):
            if (record.board_check is None or record.timing_status == 'observed'
                    and self.samples[0][0] <= record.action_start_s):
                following = timed[offset + 1] if offset + 1 < len(timed) else None
                cutoff = (following.action_start_s if following.action_start_s is not None
                          else following.video_end_s) if following is not None else time_s
                check = self._comparison(record, cutoff)
                reason = record.reason.split('；整盘核对：')[0]
                if check.status == 'conflict':
                    reason += '；整盘核对：' + check.reason
                timed[offset] = record.model_copy(update={'board_check': check, 'reason': reason})
        best.records = tuple(timed)
        hits = list(best.hits)
        for offset, record in enumerate(best.records):
            if hits[offset] >= 2 or record.timing_status != 'observed':
                continue
            following = best.records[offset + 1] if offset + 1 < len(best.records) else None
            cutoff = (following.action_start_s if following.action_start_s is not None
                      else following.video_end_s) if following is not None else time_s
            expected = {(p.square.file, p.square.rank): p.color for p in record.predicted_position.pieces}
            support = 0
            for sample_s, observed in self.samples:
                if (sample_s < record.action_end_s or sample_s > cutoff or len(observed) < 70
                        or observed.get((record.move.source.file, record.move.source.rank), 'unknown') is not None
                        or observed.get((record.move.target.file, record.move.target.rank)) != record.move.color
                        or any(expected.get(square) != color for square, color in observed.items())):
                    continue
                support += 1
                if support == 2:
                    break
            hits[offset] = max(hits[offset], support)
        best.hits = tuple(hits)
        margin = best.score - ordered[1].score if len(ordered) > 1 else float('inf')
        verified_offsets = set()
        if not self.origin_uncertain and not self._penalty(best.position, cells):
            for offset, hit in enumerate(best.hits[:-1]):
                if hit < 2 or best.records[offset].board_check is None or best.records[offset].board_check.status == 'conflict':
                    continue
                if margin < 1 and any(len(p.records) <= offset
                        or p.records[offset].move != best.records[offset].move
                        or board_signature(p.records[offset].position_before) != board_signature(best.records[offset].position_before)
                        for p in ordered[1:] if p.score >= best.score - 1):
                    continue
                verified_offsets.add(offset)
        seal = max(verified_offsets, default=-1) + 1
        for offset, record in enumerate(best.records[:seal]):
            if record.board_check is None or record.board_check.status == 'conflict':
                seal = offset
                break
        if seal:
            settled = []
            for offset, record in enumerate(best.records[:seal]):
                saved = record.model_copy()
                if offset in verified_offsets:
                    saved.status, saved.position_after = 'auto_verified', saved.predicted_position
                    saved.depends_on_steps = []
                    saved.reason = '起落点经连续可见占位核对，后手已接续；棋子身份沿用历史记录'
                else:
                    saved.depends_on_steps = [r.index for r in settled if r.status == 'tentative']
                settled.append(saved)
            self.committed += tuple(settled)
            self.before, self.start_s = settled[-1].predicted_position, settled[-1].video_end_s
            chosen = tuple(r.move for r in best.records[:seal])
            self.beam = [Hypothesis(p.position, p.records[seal:], p.score, p.hits[seal:])
                for p in self.beam if tuple(r.move for r in p.records[:seal]) == chosen]
            best = self.beam[0]
            self.root_score = -self._penalty(self.before, cells)
        self.evidence_ready = tuple(r.index for r, hit in zip(best.records, best.hits) if hit >= 2)
        result = [r.model_copy() for r in self.committed]
        for offset, record in enumerate(best.records):
            saved = record.model_copy()
            saved.inference_score = round(best.score, 4)
            saved.depends_on_steps = [r.index for r in best.records if r.index < saved.index]
            if offset + seal in verified_offsets:
                saved.status, saved.position_after = 'auto_verified', saved.predicted_position
                saved.reason = '本手起落点经连续可见占位核对；前置冲突推断仍保留回查'
            result.append(saved)
        return result
