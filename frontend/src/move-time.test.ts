import { expect, it } from 'vitest';
import { stepAtVideoTime, stepSeekTime } from './replay';
import { moveTimeText } from './types';
import type { MoveRecord, Task } from './types';

const record = { index: 1, move: null, evidence_ids: [], candidates: [], reason: '', status: 'auto_verified', position_after: { pieces: [], side_to_move: 'black' },
  video_start_s: 7, video_end_s: 38, action_start_s: 15.125, action_end_s: 15.8, timing_status: 'observed' } as MoveRecord;

it('定位到动作开始前一秒并保留毫秒', () => {
  expect(stepSeekTime(record)).toBe(14.125);
  expect(moveTimeText(record)).toBe('00:14.125');
});
it('首秒动作定位到视频零秒', () => {
  expect(stepSeekTime({ ...record, action_start_s: .4 })).toBe(0);
});
it('推断步骤显示具体推断时刻并定位到该时刻前一秒', () => {
  const inferred = { ...record, timing_status: 'uncertain', action_start_s: null } as MoveRecord;
  expect(moveTimeText(inferred)).toBe('约 00:14.800');
  expect(stepSeekTime(inferred)).toBe(14.8);
});

it('跨步回查得到的不同落定时间逐步同步，不直接跳到最后一手', () => {
  const task = { replay_mode: 'actual', moves: [record,
    { ...record, index: 2, action_start_s: 16.2, action_end_s: 17 },
    { ...record, index: 3, action_start_s: 18, action_end_s: 19 }] } as Task;
  expect(stepAtVideoTime(task, 15.9)).toBe(1);
  expect(stepAtVideoTime(task, 17.2)).toBe(2);
  expect(stepAtVideoTime(task, 19)).toBe(3);
});
it('棋盘同步按落定时刻而不是回合提交时刻或提前定位时间', () => {
  const task = { replay_mode: 'actual', moves: [record] } as Task;
  expect(stepAtVideoTime(task, 14.125)).toBe(0);
  expect(stepAtVideoTime(task, 15.79)).toBe(0);
  expect(stepAtVideoTime(task, 15.8)).toBe(1);
});
