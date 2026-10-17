import type { MoveRecord, Position, Task } from './types';
export function isIgnored(record: MoveRecord): boolean {
  return record.status === 'corrected' && record.change_type === 'review' && !record.move;
}
export function visibleRecords(task: Task): MoveRecord[] {
  return task.moves.filter(record => !isIgnored(record));
}
export function displayStep(task: Task, index: number): number {
  return visibleRecords(task).filter(record => record.index <= index).length + (index > task.moves.length ? 1 : 0);
}
export function positionAt(task: Task, step: number): Position | null {
  if (step === 0) return task.initial_position;
  if (task.replay_mode === 'actual' || task.moves.some(r => r.manual_override)) {
    const record = task.moves[step - 1];
    return record && ['auto_verified', 'corrected'].includes(record.status) ? record.position_after : null;
  }
  if (step < 0 || step > task.confirmed_prefix_length) return null;
  return task.moves[step - 1]?.position_after ?? null;
}
export function nextStep(task: Task, step: number): number | null {
  const next = visibleRecords(task).find(record => record.index > step);
  return next && positionAt(task, next.index) ? next.index : null;
}
export function previousStep(task: Task, step: number): number {
  return visibleRecords(task).filter(record => record.index < step).at(-1)?.index ?? 0;
}
export function stepAtVideoTime(task: Task, timeS: number): number {
  let step = 0;
  for (const record of task.replay_mode === 'actual' || task.moves.some(r => r.manual_override) ? task.moves : task.moves.slice(0, task.confirmed_prefix_length)) {
    if (isIgnored(record)) continue;
    if (record.video_end_s > timeS) break;
    if (positionAt(task, record.index)) step = record.index;
  }
  return step;
}
export function lastKnownStep(task: Task): number {
  return visibleRecords(task).reduce((step, record) => positionAt(task, record.index) ? record.index : step, 0);
}
export function positionBefore(task: Task, index: number): Position | null {
  if (index === 1) return task.initial_position;
  const saved = task.moves[index - 2];
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
