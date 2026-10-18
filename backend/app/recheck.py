import hashlib
import weakref
import av
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .actual import actual_events, reconcile_position
from .board_evidence import check_transition
from .calibration import rectify_board
from .chess import board_signature
from .initial_observation import observe_initial_board
from .schemas import MoveRecord, RecheckState, Task, VideoMetadata
from .storage import RevisionConflict, TaskStore
from .turns import TurnReplay, renumber_records, replay_corrections
from .video import iter_frames, save_image


def range_bounds(task, first_index, second_index):
    left, right = sorted((first_index, second_index))
    if not 1 <= left < right <= len(task.moves):
        raise ValueError('请选择两个不同的走棋步骤')
    first, last = task.moves[left - 1], task.moves[right - 1]
    if first.action_end_s is None or last.action_start_s is None or first.timing_status == 'uncertain' or last.timing_status == 'uncertain':
        raise ValueError('边界动作时间尚未确定，请先对照视频校正两个边界步骤的动作时间')
    if first.position_after is None or first.status not in ('auto_verified', 'corrected'):
        raise ValueError('请先确认前边界步骤完成后的局面')
    if not 0 <= first.action_end_s < last.action_start_s <= task.metadata.duration_s:
        raise ValueError('两步动作时间重叠或顺序异常，请先核对边界时间')
    return left, right, first.action_end_s, last.action_start_s


def replace_range(task, left, right, records, preserve_unknown=False):
    task = task.model_copy(deep=True)
    delta = len(records) - (right - left - 1)
    mapping = {r.index: r.index if r.index <= left else r.index + delta
               for r in task.moves if r.index <= left or r.index >= right}
    replacements = [r.model_copy(deep=True) for r in records]
    available = [r for r in task.moves[left:right - 1] if r.manual_override]
    for record in replacements:
        matches = [r for r in available if move_key(r) == move_key(record) and r.position_after is not None
                   and record.position_after is not None and board_signature(r.position_after) == board_signature(record.position_after)]
        if matches:
            original = min(matches, key=lambda r: abs(r.video_end_s - record.video_end_s))
            available.remove(original)
            record.manual_override, record.status, record.reason = True, 'corrected', original.reason
            record.position_edits = original.position_edits
            record.position_side_to_move = original.position_side_to_move
    # 候选内部的关联编号属于局部序列，先转成全局编号再统一重编。
    for record in replacements:
        record.related_steps = [left + i for i in record.related_steps]
    task.moves[left:right - 1] = replacements
    for index, record in enumerate(task.moves, 1):
        if left < index <= left + len(replacements):
            continue
        record.related_steps = [mapping[i] for i in record.related_steps if i in mapping]
    renumber_records(task, {i: i for i in range(1, len(task.moves) + 1)})
    return replay_corrections(task, left + 1, preserve_unknown=preserve_unknown)


def trace_range(source, calibration, before, start_s, end_s, evidence_dir, on_progress):
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    seed = Task(id='0' * 32, filename='recheck', start_s=start_s, end_s=end_s,
                metadata=VideoMetadata(width=901, height=1001, duration_s=end_s, codec='h264'),
                initial_position=before, rules_enabled=True, replay_mode='actual')
    replay = TurnReplay(seed, before)
    timestamps = {}
    count, latest, first_image, reference_image = 0, None, None, None
    current = before
    with av.open(str(source)) as container:
        rate = container.streams.video[0].average_rate or container.streams.video[0].guessed_rate
        frame_tolerance = 1.5 / float(rate) if rate else .05

    def reader(image, reference=None):
        return observe_initial_board(image)[0]

    def boards():
        nonlocal count, latest, first_image
        for t, frame in iter_frames(source, start_s, end_s):
            if t >= end_s:
                break
            image = rectify_board(frame, calibration, margin=100)
            count += 1
            latest = (t, image)
            if first_image is None:
                first_image = image
            timestamps[id(image)] = (weakref.ref(image), t)
            if len(timestamps) > 256:
                for key in [key for key, value in timestamps.items() if value[0]() is None]:
                    del timestamps[key]
            on_progress(t, count, seed.moves)
            yield t, image

    events = actual_events(boards(), before, lambda: current, turn_boundaries=True,
        cell_reader=reader, confirmed_image=lambda: reference_image,
        turn_origin=lambda: replay.before if replay.pending is not None and len(replay.touched) == 1 else None)
    local_classifier = None
    for event in events:
        if event.moving_colors:
            replay.movement(event.moving_colors, event.last_clear_s if event.last_clear_s is not None else event.time_s,
                            event.departure_only)
            continue
        ids, times = [], []
        for image in event.images:
            name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
            if not (evidence_dir / name).exists():
                save_image(evidence_dir / name, image)
            ids.append(name)
            times.append(timestamps[id(image)][1] if id(image) in timestamps else None)
        position = event.position
        if position is None and not event.review_reason:
            if local_classifier is None:
                from .vision import VisionClient
                local_classifier = VisionClient().local_classifier
            if local_classifier is not None:
                candidates = local_classifier.recognize_position(event.images, reference=current, actual=True)
                unique = {board_signature(p): p for p in candidates}
                if len(unique) == 1:
                    position = reconcile_position(current, next(iter(unique.values())), [before, current])
        if position is None or event.review_reason:
            replay.uncertain(event.time_s, [position] if position is not None else [], ids,
                             event.review_reason or '逐帧检查发现无法确认的变化，请核对原视频')
            replay.attach_evidence(None, ids, times)
            continue
        check = check_transition(current, position, reader(event.images[0]), reader(event.images[-1]))
        check.backfilled = True
        if check.status == 'conflict':
            replay.uncertain(event.time_s, [position], ids, check.reason)
        else:
            replay.observe(position, event.time_s, ids, event.action_start_s)
            current = replay.observed
            reference_image = event.images[-1]
        replay.attach_evidence(check, ids, times)
    replay.end(end_s)
    for record in seed.moves:
        if record.move is not None and (record.action_start_s is None or record.action_end_s is None or record.timing_status != 'observed'):
            record.status = 'needs_review'
            record.reason = '已找到局面变化，但缺少独立动作时间证据，顺序仍需核对'
    for label, image in [('起始', first_image), ('结束', latest[1] if latest else None)]:
        position = before if label == '起始' else replay.observed
        cells = reader(image) if image is not None else {}
        occupied = {(p.square.file, p.square.rank): p.color for p in position.pieces}
        if len(cells) != 90 or any(occupied.get(s) != c for s, c in cells.items()):
            record = MoveRecord(index=len(seed.moves) + 1, video_start_s=start_s if label == '起始' else (latest[0] if latest else start_s),
                video_end_s=end_s, status='needs_review', position_before=position, change_type='review',
                reason=f'{label}边界画面存在遮挡或局面差异，尚不能确认区间完整', timing_status='uncertain')
            if image is not None:
                name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
                save_image(evidence_dir / name, image)
                record.evidence_ids, record.evidence_times_s = [name], [start_s if label == '起始' else latest[0]]
            seed.moves.append(record)
    endpoint_matches = bool(latest) and not any(r.reason.startswith('结束边界') for r in seed.moves)
    if latest is None or end_s - latest[0] > frame_tolerance:
        seed.moves.append(MoveRecord(index=len(seed.moves) + 1,
            video_start_s=latest[0] if latest else start_s, video_end_s=end_s,
            status='needs_review', change_type='review', timing_status='uncertain',
            reason='视频解码未完整覆盖所选区间，不能确认末端或删除已有步骤'))
        endpoint_matches = False
    on_progress(latest[0] if latest else start_s, count, seed.moves)
    return seed.moves, endpoint_matches


def comparison(task, left, right, records):
    from difflib import SequenceMatcher
    old = task.moves[left:right - 1]
    added = changed = removed = 0
    manual = []
    for tag, a, b, c, d in SequenceMatcher(a=[move_key(r) for r in old], b=[move_key(r) for r in records], autojunk=False).get_opcodes():
        if tag == 'equal':
            for original, candidate in zip(old[a:b], records[c:d]):
                different = (original.position_after is None or candidate.position_after is None
                    or board_signature(original.position_after) != board_signature(candidate.position_after))
                timing = original.manual_override and (original.action_start_s, original.action_end_s) != (candidate.action_start_s, candidate.action_end_s)
                if different or timing:
                    changed += 1
                    if original.manual_override:
                        manual.append(original.index)
            continue
        overlap = min(b - a, d - c)
        changed += overlap
        added += d - c - overlap
        removed += b - a - overlap
        manual.extend(r.index for r in old[a:b] if r.manual_override)
    return added, changed, removed, manual


def move_key(record):
    m = record.move
    return (m.color, m.piece_id, m.source.file, m.source.rank, m.target.file, m.target.rank,
            m.captured_piece_id) if m else ('review', record.index)


def unconfirmed_record(record):
    record = record.model_copy(deep=True)
    if record.move is not None:
        record.candidates = [record.move] + [m for m in record.candidates if m != record.move]
        record.move = None
    record.status, record.position_after, record.change_type = 'needs_review', None, 'review'
    record.board_check = None
    return record


def integrate_recheck(task):
    task = task.model_copy(deep=True)
    job = task.recheck
    if job is None or job.integrated or job.status not in ('ready', 'needs_review'):
        return task
    left, right = job.left_index, job.right_index
    confirmed = [r for r in job.records if r.move is not None and r.status in ('auto_verified', 'corrected')
                 and r.timing_status == 'observed' and r.action_start_s is not None and r.action_end_s is not None
                 and job.start_s <= r.action_start_s <= r.action_end_s < job.end_s]
    if job.endpoint_matches and not job.unresolved:
        job.added, job.changed, job.removed, job.manual_conflicts = comparison(task, left, right, confirmed)
        task = replace_range(task, left, right, confirmed, preserve_unknown=True)
        job = task.recheck
        job.right_index = left + len(confirmed) + 1
        job.status = 'applied'
        job.message = f'重检已完成，已自动更新对局步骤：补回 {job.added} 手、纠正 {job.changed} 手'
    elif confirmed:
        original = task.moves[left:right - 1]
        retained = [(unconfirmed_record(r) if r.status in ('needs_review', 'tentative') else r.model_copy(deep=True), r.index, None, True) for r in original
                    if r.move is not None or r.manual_override or r.position_edits is not None]
        added = changed = 0
        matched = set()
        for candidate in confirmed:
            event = next((i for i, (r, _, _, _) in enumerate(retained) if i not in matched and r.move is not None
                and r.move.piece_id == candidate.move.piece_id
                and (abs((r.action_start_s if r.action_start_s is not None else r.video_end_s) - candidate.action_start_s) <= .25
                     or abs((r.action_end_s if r.action_end_s is not None else r.video_end_s) - candidate.action_end_s) <= .25)), None)
            if event is not None:
                matched.add(event)
                old, old_index, _, _ = retained[event]
                if move_key(old) == move_key(candidate) and old.position_after is not None and candidate.position_after is not None and board_signature(old.position_after) == board_signature(candidate.position_after):
                    retained[event] = (old, old_index, candidate.index, True)
                else:
                    if old.manual_override:
                        if old_index not in job.manual_conflicts:
                            job.manual_conflicts.append(old_index)
                        retained[event] = (old, old_index, candidate.index, True)
                    else:
                        retained[event] = (candidate.model_copy(deep=True), old_index, candidate.index, False)
                        changed += 1
            else:
                retained.append((candidate.model_copy(deep=True), None, candidate.index, False))
                added += 1
        for review in job.records:
            if review.status in ('auto_verified', 'corrected'):
                continue
            retained.append((unconfirmed_record(review), None, review.index, False))
        retained.sort(key=lambda entry: entry[0].action_start_s if entry[0].move is not None and entry[0].action_start_s is not None
                      else entry[0].video_end_s if entry[0].move is not None else entry[0].video_start_s)
        delta = len(retained) - (right - left - 1)
        old_mapping = {r.index: r.index if r.index <= left else r.index + delta
                       for r in task.moves if r.index <= left or r.index >= right}
        old_mapping.update({old_index: left + i for i, (_, old_index, _, _) in enumerate(retained, 1) if old_index is not None})
        local_mapping = {local_index: left + i for i, (_, _, local_index, _) in enumerate(retained, 1) if local_index is not None}
        for record in task.moves[:left] + task.moves[right - 1:]:
            record.related_steps = [old_mapping[i] for i in record.related_steps if i in old_mapping]
        for record, old_index, _, from_old in retained:
            mapping = old_mapping if from_old else local_mapping
            record.related_steps = [mapping[i] for i in record.related_steps if i in mapping]
        task.moves[left:right - 1] = [r for r, _, _, _ in retained]
        renumber_records(task, {i: i for i in range(1, len(task.moves) + 1)})
        task = replay_corrections(task, left + 1, preserve_unknown=True)
        job = task.recheck
        job.right_index = right + delta
        job.added, job.changed, job.removed = added, changed, 0
        job.message = f'已自动补入 {added} 手、纠正 {changed} 手到对局步骤；剩余 {job.unresolved} 处作为待核对区间保留'
        task.status = 'needs_review'
        task.stage = '区域重检已写入确认走法，剩余视频疑点保留'
    else:
        job.added = job.changed = job.removed = 0
        job.message = '未找到可确认的新增走法，已有步骤保留；疑点与证据继续保存'
    job.integrated, job.integrated_count = True, len(confirmed)
    task.recovery = None
    return task


def archive_recheck(task, store):
    folder = store.task_dir(task.id) / 'recheck-history'
    folder.mkdir(exist_ok=True)
    target = folder / f'{task.recheck.id}-before.json'
    if not target.exists():
        target.write_text(task.model_dump_json(), encoding='utf8')


def run_recheck(task_id, data_root, job_id):
    store = TaskStore(Path(data_root))
    task = store.get(task_id)
    job = task.recheck
    if job is None or job.id != job_id or job.status != 'running':
        return
    last_saved = 0

    def persist(t, count, records, final=False):
        nonlocal last_saved
        if not final and monotonic() - last_saved < .5:
            return
        def update(latest):
            latest.recheck.processed_s, latest.recheck.scanned_frames = t, count
            latest.recheck.records = [r.model_copy(deep=True) for r in records]
            latest.recheck.unresolved = sum(r.status not in ('auto_verified', 'corrected') for r in records)
        store.update_recheck(task_id, job_id, update)
        last_saved = monotonic()

    try:
        records, matches = trace_range(store.task_dir(task.id) / 'source.mp4', task.calibration,
            task.moves[job.left_index - 1].position_after, job.start_s, job.end_s,
            store.task_dir(task.id) / 'evidence', persist)
        def complete(latest):
            state = latest.recheck
            state.records, state.endpoint_matches = records, matches
            state.processed_s = job.end_s
            state.unresolved = sum(r.status not in ('auto_verified', 'corrected') for r in records)
            state.added, state.changed, state.removed, state.manual_conflicts = comparison(task, job.left_index, job.right_index, records)
            state.status = 'ready' if matches and not state.unresolved else 'needs_review'
            archive_recheck(latest, store)
            fixed = integrate_recheck(latest)
            for field in ('moves', 'confirmed_prefix_length', 'status', 'stage', 'recheck', 'recovery'):
                setattr(latest, field, getattr(fixed, field))
        store.update_recheck(task_id, job_id, complete)
    except RevisionConflict:
        return
    except Exception as exc:
        def fail(latest):
            latest.recheck.status, latest.recheck.message = 'failed', f'区域重检未完成：{exc}'
        try:
            store.update_recheck(task_id, job_id, fail)
        except RevisionConflict:
            pass
