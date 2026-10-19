import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it, vi, afterEach } from 'vitest';
import * as replay from './replay';
import { api } from './api';
import { ReviewPanel } from './components/ReviewPanel';
import { RecheckPanel } from './components/RecheckPanel';
import type { Task } from './types';

const position = { side_to_move: 'red', pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }] };
const task = { id: 'task', revision: 7, initial_position: position, start_s: 0, end_s: 8,
  metadata: { duration_s: 8 }, moves: [
    { index: 1, action_start_s: .5, action_end_s: 1, timing_status: 'observed', video_end_s: 2 },
    { index: 2, action_start_s: 6, action_end_s: 6.5, timing_status: 'observed', video_end_s: 7,
      move: { index: 2, piece_id: 'r', color: 'red', from: { file: 0, rank: 9 }, to: { file: 0, rank: 8 } },
      candidates: [], evidence_ids: [], position_before: position, status: 'auto_verified' },
  ] } as unknown as Task;

afterEach(() => vi.unstubAllGlobals());

it('倒序点选使用前步落定与后步起动，不使用识别时间或减一秒跳转时间', () => {
  const getRange = (replay as unknown as { recheckRange: (t: Task, a: number, b: number) => unknown }).recheckRange;
  expect(getRange).toBeTypeOf('function');
  expect(getRange(task, 2, 1)).toEqual({ left: 1, right: 2, start: 1, end: 6 });
});

it('插入面板不会预填旧步骤走法或显示不计入已有步骤按钮', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, {
    task, index: 2, position: task.initial_position!, picked: null, busy: false, videoTime: 3,
    insert: true, onSave: () => {}, onSavePosition: () => {}, onClose: () => {}, onPreview: () => {},
  } as unknown as Parameters<typeof ReviewPanel>[0]));
  expect(html).toContain('插入遗漏走棋');
  expect(html).not.toContain('已选红车');
  expect(html).not.toContain('此处不计入走棋');
});

it('插入请求发送独立端点，保留落定与动作开始时间', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ...task, revision: 8 }) });
  vi.stubGlobal('fetch', fetch);
  const insert = (api as unknown as { insert: (...args: unknown[]) => Promise<Task> }).insert;
  expect(insert).toBeTypeOf('function');
  const saved = await insert(task, 2, task.moves[1].move, 3, 2.5);
  expect(saved.revision).toBe(8);
  const [url, options] = fetch.mock.calls[0];
  expect(url).toBe('/api/tasks/task/moves/2/insert');
  expect(JSON.parse(options.body)).toMatchObject({ revision: 7, video_time_s: 3, action_start_s: 2.5 });
});

it('部分恢复自动写入后不再要求用户手动补已确认走法', () => {
  const updated = { ...task, recheck: { id: 'job', status: 'needs_review', integrated: true,
    integrated_count: 2, left_index: 1, right_index: 2, start_s: 1, end_s: 6, processed_s: 6,
    scanned_frames: 389, records: [], unresolved: 3, added: 2, changed: 0, removed: 0,
    manual_conflicts: [], message: '已自动补入2手', endpoint_matches: false } } as Task;
  const html = renderToStaticMarkup(createElement(RecheckPanel, { task: updated, busy: false,
    onApply: () => {}, onCancel: () => {}, onSelect: () => {} }));
  expect(html).toContain('已自动写入对局步骤 2 手');
  expect(html).not.toContain('写入已识别走棋');
  expect(html).not.toContain('使用中间插入');
});
