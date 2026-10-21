import { expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { lastKnownStep, nextStep, positionAt, reviewIntervals, stepAtVideoTime, visibleRecords } from './replay';
import { Board } from './components/Board';
import { MoveList } from './components/MoveList';
import type { Task } from './types';

const before = { pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }], side_to_move: 'red' };
const predicted = { ...before, pieces: [{ ...before.pieces[0], square: { file: 0, rank: 7 } }] };
const task = { initial_position: before, moves: [{ index: 1, status: 'tentative', move: { index: 1,
  piece_id: 'r', color: 'red', from: { file: 0, rank: 9 }, to: { file: 0, rank: 7 } },
  position_before: before, position_after: null, predicted_position: predicted, video_start_s: 1,
  video_end_s: 3, action_end_s: 3, evidence_ids: [], candidates: [], reason: '推断，待核对' }],
  replay_mode: 'actual', rules_enabled: true, confirmed_prefix_length: 0 } as unknown as Task;

it('推断走法可显示及跟随视频，仍保留未确认状态', () => {
  expect(positionAt(task, 1)).toEqual(predicted);
  expect(stepAtVideoTime(task, 3)).toBe(1);
  expect(visibleRecords(task)).toHaveLength(1);
  expect(task.moves[0].position_after).toBeNull();
  expect(task.moves[0].status).toBe('tentative');
});

it('棋盘与棋谱同时明确标注推断，不能只靠颜色', () => {
  const board = renderToStaticMarkup(createElement(Board, { position: predicted, step: 1,
    inferredMove: task.moves[0].move } as never));
  expect(board).toContain('推断落点，待核对');
  expect(board).toContain('stroke-dasharray');
  const list = renderToStaticMarkup(createElement(MoveList, { task, step: 1, onSelect: () => {} }));
  expect(list).toContain('推断待核对');
});

it('推断恢复局面可跟随视频并核对，自动播放在此暂停', () => {
  const snapshotTask = { ...task, moves: [{ ...task.moves[0], move: null, change_type: 'review',
    tracking_anchor: predicted, tracking_anchor_s: 3 }] } as unknown as Task;
  expect(lastKnownStep(snapshotTask)).toBe(1);
  expect(positionAt(snapshotTask, 1)).toEqual(predicted);
  expect(stepAtVideoTime(snapshotTask, 3)).toBe(1);
  expect(nextStep(snapshotTask, 0)).toBeNull();
  expect(reviewIntervals(snapshotTask)[0].records[0].index).toBe(1);
  snapshotTask.moves.push({ ...task.moves[0], index: 2, action_end_s: 4 });
  expect(nextStep(snapshotTask, 0)).toBeNull();
});
