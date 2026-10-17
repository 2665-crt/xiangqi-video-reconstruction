import { describe, expect, it } from 'vitest';
import type { Task } from './types';
import { positionAt, nextStep, stepAtVideoTime } from './replay';

const initial = { side_to_move: 'red', pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }, { id: 'p', color: 'black', kind: 'pawn', square: { file: 0, rank: 5 } }] };
const after = { side_to_move: 'black', pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 5 } }] };
const task = { initial_position: initial, confirmed_prefix_length: 1, moves: [
  { index: 1, status: 'auto_verified', position_after: after, video_start_s: 1, video_end_s: 2 },
  { index: 2, status: 'needs_review', position_after: null, video_start_s: 3, video_end_s: 4 },
] } as unknown as Task;

describe('棋谱回放', () => {
  it('第0步保留吃子前棋局，第一步移除被吃棋子', () => {
    expect(positionAt(task, 0)?.pieces.length).toBe(2);
    expect(positionAt(task, 1)?.pieces).toEqual([{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 5 } }]);
  });
  it('在待确认步骤之前停止', () => {
    expect(nextStep(task, 1)).toBeNull();
    expect(positionAt(task, 2)).toBeNull();
    expect(nextStep(task, 0)).toBe(1);
  });
  it('使用落稳视频时间联动，并阻止跳过疑点', () => {
    expect(stepAtVideoTime(task, 1.5)).toBe(0);
    expect(stepAtVideoTime(task, 2.5)).toBe(1);
    expect(stepAtVideoTime(task, 6)).toBe(1);
  });
  it('非零选区仍按原视频时间联动，不减去区间起点', () => {
    const selected = { ...task, start_s: 75, end_s: 200, moves: [
      { ...task.moves[0], video_start_s: 75, video_end_s: 79.553 },
    ] } as Task;
    expect(stepAtVideoTime(selected, 4.553)).toBe(0);
    expect(stepAtVideoTime(selected, 79.552)).toBe(0);
    expect(stepAtVideoTime(selected, 79.553)).toBe(1);
  });
  it('实际局面的单处疑点不把后面的清晰快照变成待审核', () => {
    const actual = { ...task, replay_mode: 'actual', moves: [...task.moves,
      { index: 3, status: 'auto_verified', position_after: initial, video_start_s: 4, video_end_s: 5 }],
    } as Task;
    expect(positionAt(actual, 2)).toBeNull();
    expect(positionAt(actual, 3)).toEqual(initial);
    expect(stepAtVideoTime(actual, 6)).toBe(3);
    expect(nextStep(actual, 1)).toBeNull();
  });
  it('按象棋规则复盘时，疑点只影响自身而不锁住后面的独立确认步骤', () => {
    const rules = { ...task, replay_mode: 'actual', rules_enabled: true, moves: [...task.moves,
      { index: 3, status: 'auto_verified', position_after: initial, video_start_s: 4, video_end_s: 5 }],
    } as unknown as Task;
    expect(positionAt(rules, 2)).toBeNull();
    expect(positionAt(rules, 3)).toEqual(initial);
    expect(stepAtVideoTime(rules, 6)).toBe(3);
  });
  it('悔棋涉及的原走棋仍可回放，审核提示不会自动退回棋盘', () => {
    const rules = { ...task, replay_mode: 'actual', rules_enabled: true, moves: [
      { ...task.moves[0], violation: 'takeback', violation_note: '疑似悔棋' },
      { ...task.moves[1], observed_positions: [initial], violation: 'takeback' },
    ] } as unknown as Task;
    expect(positionAt(rules, 1)).toEqual(after);
    expect(positionAt(rules, 2)).toBeNull();
    expect(stepAtVideoTime(rules, 6)).toBe(1);
  });
});
