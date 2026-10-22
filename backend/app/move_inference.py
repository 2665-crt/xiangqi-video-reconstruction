from dataclasses import dataclass
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

    def __init__(self, before, start_s, width=5, max_gap_plies=3, gap_width=12, *, origin_uncertain=False):
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
            if require_visible_landing and (source not in cells or cells[source] is not None
                                           or cells.get(target) != move.color):
                continue
            if cells.get(source) != move.color or cells.get(target) == move.color:
                yield move, after

    def _penalty(self, position, cells):
        expected = {(p.square.file, p.square.rank): p.color for p in position.pieces}
        return sum(1 if color is None else 4 for s, color in cells.items() if expected.get(s) != color)

    def _record(self, move, before, after, records, time_s, gap=False):
        index = len(self.committed) + len(records) + 1
        move = move.model_copy(update={'index': index})
        return MoveRecord(index=index, move=move, status='tentative', position_before=before,
            predicted_position=after, video_start_s=records[-1].video_end_s if records else self.start_s,
            video_end_s=time_s, action_end_s=time_s, timing_status='uncertain', candidates=[move],
            inference_group=self.group, reason=('区间跨步推断，中间落点未直接观察，需核对' if gap else
                '依据历史身份、合法一步与可见落点推断，待后续画面核对'))

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
                    cost = self._penalty(after, cells)
                    candidate = Hypothesis(after, parent.records + (record,), -cost, parent.hits + (-1,))
                    if cost == 0:
                        yield candidate
                    expanded.append(candidate)
            frontier = sorted(expanded, key=lambda p: -p.score)[:self.gap_width]
            if not frontier:
                break

    def observe(self, time_s, cells, *, tail_only=False):
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
                    record = self._record(move, before, after, records, time_s)
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
            score = proposal.score * .35 - penalty
            hits = proposal.hits
            if hits and hits[-1] >= 0 and not penalty and len(cells) == 90:
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
        margin = best.score - ordered[1].score if len(ordered) > 1 else float('inf')
        seal = 0
        if not self.origin_uncertain and margin >= 1 and not self._penalty(best.position, cells):
            for hit in best.hits[:-1]:
                if hit < 2:
                    break
                seal += 1
        if seal:
            verified = []
            for record in best.records[:seal]:
                saved = record.model_copy()
                saved.status, saved.position_after = 'auto_verified', saved.predicted_position
                saved.depends_on_steps = []
                saved.reason = '起落点经连续完整占位核对，后手已接续；棋子身份沿用历史记录'
                verified.append(saved)
            self.committed += tuple(verified)
            self.before, self.start_s = verified[-1].predicted_position, verified[-1].video_end_s
            chosen = tuple(r.move for r in best.records[:seal])
            self.beam = [Hypothesis(p.position, p.records[seal:], p.score, p.hits[seal:])
                for p in self.beam if tuple(r.move for r in p.records[:seal]) == chosen]
            best = self.beam[0]
            self.root_score = -self._penalty(self.before, cells)
        self.evidence_ready = tuple(r.index for r, hit in zip(best.records, best.hits) if hit >= 2)
        result = [r.model_copy() for r in self.committed]
        for record in best.records:
            saved = record.model_copy()
            saved.inference_score = round(best.score, 4)
            saved.depends_on_steps = [r.index for r in best.records if r.index < saved.index]
            result.append(saved)
        return result
