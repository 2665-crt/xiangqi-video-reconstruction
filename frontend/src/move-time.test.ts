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
it('无法确定各步时间时不显示伪造的精确时间', () => {
  expect(moveTimeText({ ...record, timing_status: 'uncertain', action_start_s: null })).toBe('时间待核对');
});
it('棋盘同步按落定时刻而不是回合提交时刻或提前定位时间', () => {
  const task = { replay_mode: 'actual', moves: [record] } as Task;
  expect(stepAtVideoTime(task, 14.125)).toBe(0);
  expect(stepAtVideoTime(task, 15.79)).toBe(0);
  expect(stepAtVideoTime(task, 15.8)).toBe(1);
});
