import type { MoveRecord, Position, Task } from './types';
export function recheckRange(task: Task, first: number, second: number): { left: number; right: number; start: number; end: number } {
  const [left, right] = [first, second].sort((a, b) => a - b);
  if (left === right) throw new Error('请选择两个不同的步骤');
  const earlier = task.moves[left - 1], later = task.moves[right - 1];
  if (!earlier || !later || earlier.action_end_s == null || later.action_start_s == null
      || earlier.timing_status === 'uncertain' || later.timing_status === 'uncertain') {
    throw new Error('边界动作时间尚未确定，请先校正这两步的动作开始和落定时间');
  }
  if (earlier.action_end_s >= later.action_start_s) throw new Error('两步动作时间重叠或顺序异常，请先核对边界时间');
  return { left, right, start: earlier.action_end_s, end: later.action_start_s };
}
export function stepSeekTime(record: MoveRecord): number {
  return Math.max(0, (record.action_start_s ?? (record.timing_status === 'uncertain' ? record.video_start_s : record.video_end_s)) - 1);
}
export function isIgnored(record: MoveRecord): boolean {
  return record.status === 'corrected' && record.change_type === 'review' && !record.move;
}
export function visibleRecords(task: Task): MoveRecord[] {
  return task.moves.filter(record => {
    if (isIgnored(record)) return false;
    const action = record.move ?? (record.candidates?.length === 1 && ['verified', 'partial'].includes(record.board_check?.status ?? '') ? record.candidates[0] : null);
    if (action) {
      const before = record.position_before ?? task.initial_position;
      if (['auto_verified', 'corrected'].includes(record.status) && before && record.position_after) {
        const original = before.pieces.find(p => p.id === action.piece_id);
        const landed = record.position_after.pieces.find(p => p.id === action.piece_id);
        if (!original || !landed || (original.square.file === landed.square.file && original.square.rank === landed.square.rank)) return false;
      }
      return action.from.file !== action.to.file || action.from.rank !== action.to.rank;
    }
    const before = record.position_before ?? task.initial_position;
    return ['auto_verified', 'corrected'].includes(record.status) && !!before && !!record.position_after?.pieces.some(p => {
      const previous = before.pieces.find(q => q.id === p.id);
      return previous && (previous.square.file !== p.square.file || previous.square.rank !== p.square.rank);
    });
  });
}
export function reviewIntervals(task: Task): { records: MoveRecord[]; start: number; end: number }[] {
  const visible = new Set(visibleRecords(task).map(r => r.index));
  const intervals: { records: MoveRecord[]; start: number; end: number }[] = [];
  let current: (typeof intervals)[number] | null = null;
  for (const record of task.moves) {
    if (visible.has(record.index)) { current = null; continue; }
    if (isIgnored(record) || !['needs_review', 'tentative'].includes(record.status)) continue;
    if (!current) {
      current = { records: [], start: record.video_start_s, end: record.video_end_s };
      intervals.push(current);
    }
    current.records.push(record);
    current.start = Math.min(current.start, record.video_start_s);
    current.end = Math.max(current.end, record.video_end_s);
  }
  return intervals;
}
export function displayStep(task: Task, index: number): number {
  return visibleRecords(task).filter(record => record.index <= index).length + (index > task.moves.length ? 1 : 0);
}
export function positionAt(task: Task, step: number): Position | null {
  if (step === 0) return task.initial_position;
  if (task.replay_mode === 'actual' || task.moves.some(r => r.manual_override)) {
    const record = task.moves[step - 1];
    if (record?.tracking_anchor) return record.tracking_anchor;
    return record && ['auto_verified', 'corrected'].includes(record.status) ? record.position_after : null;
  }
  if (step < 0 || step > task.confirmed_prefix_length) return null;
  return task.moves[step - 1]?.position_after ?? null;
}
export function nextStep(task: Task, step: number): number | null {
  const next = visibleRecords(task).find(record => record.index > step);
  if (next && task.moves.some(record => record.index > step && record.index < next.index
      && !isIgnored(record) && ['needs_review', 'tentative'].includes(record.status))) return null;
  return next && positionAt(task, next.index) ? next.index : null;
}
export function previousStep(task: Task, step: number): number {
  return visibleRecords(task).filter(record => record.index < step).at(-1)?.index ?? 0;
}
export function stepAtVideoTime(task: Task, timeS: number): number {
  let step = 0;
  for (const record of task.replay_mode === 'actual' || task.moves.some(r => r.manual_override) ? task.moves : task.moves.slice(0, task.confirmed_prefix_length)) {
    if (isIgnored(record) && !record.tracking_anchor) continue;
    if ((record.tracking_anchor_s ?? record.action_end_s ?? record.video_end_s) > timeS) break;
    if (positionAt(task, record.index)) step = record.index;
  }
  return step;
}
export function lastKnownStep(task: Task): number {
  const visible = new Set(visibleRecords(task).map(record => record.index));
  return task.moves.reduce((step, record) => (visible.has(record.index) || record.tracking_anchor)
    && positionAt(task, record.index) ? record.index : step, 0);
}
export function positionBefore(task: Task, index: number): Position | null {
  if (index === 1) return task.initial_position;
  const saved = task.moves[index - 2];
  if (saved?.tracking_anchor) return saved.tracking_anchor;
  if (saved && ['auto_verified', 'corrected'].includes(saved.status) && saved.position_after) return saved.position_after;
  if (task.rules_enabled) {
    const record = task.moves[index - 1];
    if (record?.position_before) return record.position_before;
    const previous = task.moves[index - 2];
    if (previous?.observed_positions?.length === 1) return previous.observed_positions[0];
  }
  for (let step = index - 1; step >= 0; step--) {
    const position = positionAt(task, step);
    if (position) return position;
  }
  return null;
}
