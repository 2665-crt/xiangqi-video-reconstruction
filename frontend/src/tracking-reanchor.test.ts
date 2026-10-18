import { expect, it } from 'vitest';
import { lastKnownStep, positionAt, positionBefore, reviewIntervals, stepAtVideoTime, visibleRecords } from './replay';
import type { MoveRecord, Position, Task } from './types';

const before = { pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }], side_to_move: 'red' } as Position;
const anchor = { ...before, pieces: [{ ...before.pieces[0], square: { file: 0, rank: 7 } }] };
const after = { ...anchor, pieces: [{ ...anchor.pieces[0], square: { file: 0, rank: 6 } }] };
const gap = { index: 1, status: 'needs_review', change_type: 'review', move: null, candidates: [],
  position_after: null, position_before: before, observed_positions: [], video_start_s: 0, video_end_s: 5,
  evidence_ids: ['first.jpg', 'second.jpg'], reason: '历史待审核，后续已经重新同步',
  tracking_anchor: anchor, tracking_anchor_s: 5 } as MoveRecord;
const move = { index: 2, status: 'auto_verified', change_type: 'move', candidates: [],
  move: { index: 2, piece_id: 'r', color: 'red', from: { file: 0, rank: 7 }, to: { file: 0, rank: 6 }, captured_piece_id: null },
  position_before: anchor, position_after: after, video_start_s: 5, video_end_s: 7, action_end_s: 7,
  evidence_ids: [], reason: '' } as MoveRecord;
const task = { initial_position: before, moves: [gap, move], replay_mode: 'actual', rules_enabled: true, confirmed_prefix_length: 0 } as Task;

it('重新同步保留审核区间，锚点不算一次走棋', () => {
  expect(visibleRecords(task)).toEqual([move]);
  expect(reviewIntervals(task)).toHaveLength(1);
  expect(task.moves[0].status).toBe('needs_review');
  expect(task.moves[0].position_after).toBeNull();
});

it('视频到达同步时刻后采用画面基准，后续自动确认不受历史前缀限制', () => {
  expect(stepAtVideoTime(task, 4.9)).toBe(0);
  expect(stepAtVideoTime(task, 5)).toBe(1);
  expect(positionAt(task, 1)).toEqual(anchor);
  expect(positionBefore(task, 2)).toEqual(anchor);
  expect(stepAtVideoTime(task, 7)).toBe(2);
  expect(positionAt(task, 2)).toEqual(after);
});

it('同步后尚无新落子时也能显示最新画面基准，不增加走棋数', () => {
  const recovering = { ...task, moves: [gap] };
  expect(lastKnownStep(recovering)).toBe(1);
  expect(visibleRecords(recovering)).toEqual([]);
});

it('人工忽略历史缺口后，视频同步仍采用后续独立画面基准', () => {
  const corrected = { ...task, moves: [{ ...gap, status: 'corrected', manual_override: true }, move] } as Task;
  expect(visibleRecords(corrected)).toEqual([move]);
  expect(stepAtVideoTime(corrected, 5)).toBe(1);
  expect(positionAt(corrected, 1)).toEqual(anchor);
});
