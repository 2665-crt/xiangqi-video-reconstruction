def merge_review(record, time_s, positions, evidence, reason, times=()):
    from .chess import board_signature
    timestamps = dict(zip(record.evidence_ids, record.evidence_times_s))
    timestamps.update(zip(evidence, times))
    record.video_end_s = max(record.video_end_s, time_s)
    ids = list(dict.fromkeys(record.evidence_ids + evidence))
    signatures = {board_signature(p) for p in record.observed_positions}
    observed = record.observed_positions + [p for p in positions if board_signature(p) not in signatures]
    representatives = []
    if record.position_before is not None:
        old = {p.id: p for p in record.position_before.pieces}
        changed = set(record.review_changed_piece_ids)
        represented = set()
        for position in observed:
            pieces = {p.id: p for p in position.pieces}
            differences = set()
            for identity in set(old) | set(pieces):
                original, candidate = old.get(identity), pieces.get(identity)
                if (original is None or candidate is None or original.square != candidate.square
                        or (original.color, original.kind) != (candidate.color, candidate.kind)):
                    differences.add(identity)
            if differences - represented:
                representatives.append(position)
                represented.update(differences)
            if differences - changed:
                record.review_witness_ids = list(dict.fromkeys(record.review_witness_ids + (evidence or record.evidence_ids)))
            changed.update(differences)
        record.review_changed_piece_ids = sorted(changed)
    # 每种独立棋子变化保留首次证据；只裁剪重复噪声，仍保留最新观察。
    record.evidence_ids = ids if len(ids) <= 6 else list(dict.fromkeys(ids[:2] + record.review_witness_ids + ids[-4:]))
    record.evidence_times_s = [timestamps.get(i) for i in record.evidence_ids]
    keep = {board_signature(p) for p in representatives + observed[-2:]}
    record.observed_positions = [p for p in observed if board_signature(p) in keep]
    if reason and record.reason != reason:
        record.reason = '同一区间存在多种未解释变化，请核对原视频；最新疑点：' + reason


def movement_records(task):
    records = []
    for record in task.moves:
        if record.status == 'corrected' and record.change_type == 'review' and record.move is None:
            continue
        action = record.move
        if action is None and len(record.candidates) == 1 and record.board_check is not None and record.board_check.status in ('verified', 'partial'):
            action = record.candidates[0]
        if action is not None:
            before = record.position_before or task.initial_position
            if record.status in ('auto_verified', 'corrected') and before is not None and record.position_after is not None:
                original = next((p for p in before.pieces if p.id == action.piece_id), None)
                landed = next((p for p in record.position_after.pieces if p.id == action.piece_id), None)
                if original is None or landed is None or original.square == landed.square:
                    continue
            if action.source != action.target:
                records.append(record)
            continue
        before = record.position_before or task.initial_position
        if record.status in ('auto_verified', 'corrected') and before is not None and record.position_after is not None:
            old = {p.id: p.square for p in before.pieces}
            if any(p.id in old and p.square != old[p.id] for p in record.position_after.pieces):
                records.append(record)
    return records
