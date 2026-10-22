import { expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MoveList } from './components/MoveList';
import { nextStep, positionAt, reviewIntervals, stepSeekTime } from './replay';
import { isAutomaticRecord, moveTimeText } from './types';
import type { MoveRecord, Position, Task } from './types';

const before: Position = { pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }], side_to_move: 'red' };
const predicted = { ...before, pieces: [{ ...before.pieces[0], square: { file: 0, rank: 7 } }] };
const record = { index: 1, status: 'tentative', move: { index: 1, piece_id: 'r', color: 'red',
  from: { file: 0, rank: 9 }, to: { file: 0, rank: 7 } }, position_before: before,
  position_after: null, predicted_position: predicted, video_start_s: 7, video_end_s: 38,
  action_start_s: null, action_end_s: 15.8, timing_status: 'uncertain', evidence_ids: [], candidates: [], reason: '' } as unknown as MoveRecord;
const task = { initial_position: before, moves: [record], replay_mode: 'actual', confirmed_prefix_length: 0 } as unknown as Task;
const markup = (value: Task) => renderToStaticMarkup(createElement(MoveList, { task: value, step: 1, onSelect: () => {} }));

it('最大概率走法继续上盘播放，但不会冒充自动确认或精确动作时间', () => {
  const html = markup(task);
  expect(html).toContain('最大概率推断');
  expect(html).not.toContain('>自动确认<');
  expect(html).toContain('约 00:14.800');
  expect(isAutomaticRecord(record)).toBe(false);
  expect(positionAt(task, 1)).toEqual(predicted);
  expect(nextStep(task, 0)).toBe(1);
  expect(stepSeekTime(record)).toBe(14.8);
  expect(record.status).toBe('tentative');
  expect(record.position_after).toBeNull();
});

it('未知动作时间仍保留区间定位入口，不伪造落子时间', () => {
  const legacy = { ...record, action_end_s: null };
  expect(moveTimeText(legacy)).toBe('时间待核对');
  expect(stepSeekTime(legacy)).toBe(6);
});

it('真正的人工审核走法不显示自动确认，也不自动播放', () => {
  const manual = { ...record, status: 'needs_review', predicted_position: null, tracking_anchor: predicted } as MoveRecord;
  const manualTask = { ...task, moves: [manual] };
  expect(markup(manualTask)).toContain('人工审核');
  expect(markup(manualTask)).not.toContain('status-pill auto_verified');
  expect(positionAt(manualTask, 1)).toEqual(predicted);
  expect(nextStep(manualTask, 0)).toBeNull();
});

it('已自动恢复局面有时间入口，并允许播放后续走法', () => {
  const snapshot = { ...record, move: null, tracking_anchor: predicted };
  const recovered = { ...task, moves: [snapshot, { ...record, index: 2 }] };
  expect(reviewIntervals(recovered)).toHaveLength(0);
  expect(nextStep(recovered, 0)).toBe(1);
  expect(nextStep(recovered, 1)).toBe(2);
  expect(markup(recovered)).toContain('推断恢复局面');
  expect(markup(recovered)).not.toContain('>自动确认<');
  const unresolved = { ...recovered, moves: [{ ...snapshot, predicted_position: null }, recovered.moves[1]] };
  expect(reviewIntervals(unresolved)).toHaveLength(1);
  expect(nextStep(unresolved, 0)).toBeNull();
});

it('推断后的清晰步骤仍单独显示自动确认，不要求前序全部确定', () => {
  const confirmed = { ...record, index: 2, status: 'auto_verified', position_after: predicted,
    position_before: before, action_start_s: 20, action_end_s: 21, timing_status: 'observed' } as MoveRecord;
  const mixed = { ...task, moves: [record, confirmed] };
  expect(isAutomaticRecord(confirmed)).toBe(true);
  expect(markup(mixed)).toContain('1 手已确认');
  expect(markup(mixed)).toContain('>自动确认<');
  expect(nextStep(mixed, 1)).toBe(2);
});

it('最后一条自动恢复局面也能播放上盘，不会停在此前的走法', () => {
  const finalSnapshot = { ...record, index: 2, move: null, tracking_anchor: predicted };
  expect(nextStep({ ...task, moves: [record, finalSnapshot] }, 1)).toBe(2);
});

it('有单个候选和局面的人工审核仍等待人工处理', () => {
  const candidate = { ...record, status: 'needs_review', move: null, candidates: [record.move!],
    predicted_position: null, tracking_anchor: predicted, board_check: { status: 'partial' } } as MoveRecord;
  expect(nextStep({ ...task, moves: [candidate] }, 0)).toBeNull();
});
